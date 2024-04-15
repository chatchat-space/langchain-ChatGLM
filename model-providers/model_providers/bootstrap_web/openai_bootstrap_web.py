import asyncio
import json
import logging
import multiprocessing as mp
import os
import pprint
import threading
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
from uvicorn import Config, Server

from model_providers.bootstrap_web.entities.model_provider_entities import (
    ProviderListResponse,
    ProviderModelTypeResponse,
)
from model_providers.bootstrap_web.message_convert import (
    convert_to_message,
    openai_chat_completion,
    openai_embedding_text,
    stream_openai_chat_completion,
)
from model_providers.core.bootstrap import OpenAIBootstrapBaseWeb
from model_providers.core.bootstrap.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ModelCard,
    ModelList,
)
from model_providers.core.bootstrap.providers_wapper import ProvidersWrapper
from model_providers.core.model_runtime.entities.message_entities import (
    PromptMessageTool,
)
from model_providers.core.model_runtime.entities.model_entities import (
    AIModelEntity,
    ModelType,
)
from model_providers.core.model_runtime.errors.invoke import InvokeError
from model_providers.core.utils.generic import dictify

logger = logging.getLogger(__name__)


class RESTFulOpenAIBootstrapBaseWeb(OpenAIBootstrapBaseWeb):
    """
    Bootstrap Server Lifecycle
    """

    def __init__(self, host: str, port: int):
        super().__init__()
        self._host = host
        self._port = port
        self._router = APIRouter()
        self._app = FastAPI()
        self._server_thread = None

    @classmethod
    def from_config(cls, cfg=None):
        host = cfg.get("host", "127.0.0.1")
        port = cfg.get("port", 20000)

        logger.info(
            f"Starting openai Bootstrap Server Lifecycle at endpoint: http://{host}:{port}"
        )
        return cls(host=host, port=port)

    def serve(self, logging_conf: Optional[dict] = None):
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._router.add_api_route(
            "/workspaces/current/model-providers",
            self.workspaces_model_providers,
            response_model=ProviderListResponse,
            methods=["GET"],
        )
        self._router.add_api_route(
            "/workspaces/current/models/model-types/{model_type}",
            self.workspaces_model_types,
            response_model=ProviderModelTypeResponse,
            methods=["GET"],
        )
        self._router.add_api_route(
            "/{provider}/v1/models",
            self.list_models,
            response_model=ModelList,
            methods=["GET"],
        )

        self._router.add_api_route(
            "/{provider}/v1/embeddings",
            self.create_embeddings,
            response_model=EmbeddingsResponse,
            status_code=status.HTTP_200_OK,
            methods=["POST"],
        )
        self._router.add_api_route(
            "/{provider}/v1/chat/completions",
            self.create_chat_completion,
            response_model=ChatCompletionResponse,
            status_code=status.HTTP_200_OK,
            methods=["POST"],
        )

        self._app.include_router(self._router)

        config = Config(
            app=self._app, host=self._host, port=self._port, log_config=logging_conf
        )
        server = Server(config)

        def run_server():
            server.run()

        self._server_thread = threading.Thread(target=run_server)
        self._server_thread.start()

    async def join(self):
        await self._server_thread.join()

    def set_app_event(self, started_event: mp.Event = None):
        @self._app.on_event("startup")
        async def on_startup():
            if started_event is not None:
                started_event.set()

    async def workspaces_model_providers(self, request: Request):
        provider_list = ProvidersWrapper(
            provider_manager=self._provider_manager.provider_manager
        ).get_provider_list(model_type=request.get("model_type"))
        return ProviderListResponse(data=provider_list)

    async def workspaces_model_types(self, model_type: str, request: Request):
        models_by_model_type = ProvidersWrapper(
            provider_manager=self._provider_manager.provider_manager
        ).get_models_by_model_type(model_type=model_type)
        return ProviderModelTypeResponse(data=models_by_model_type)

    async def list_models(self, provider: str, request: Request):
        logger.info(f"Received list_models request for provider: {provider}")
        # 返回ModelType所有的枚举
        llm_models: list[AIModelEntity] = []
        for model_type in ModelType.__members__.values():
            try:
                provider_model_bundle = (
                    self._provider_manager.provider_manager.get_provider_model_bundle(
                        provider=provider, model_type=model_type
                    )
                )
                llm_models.extend(
                    provider_model_bundle.model_type_instance.predefined_models()
                )
            except Exception as e:
                logger.error(
                    f"Error while fetching models for provider: {provider}, model_type: {model_type}"
                )
                logger.error(e)

        # models list[AIModelEntity]转换称List[ModelCard]

        models_list = [
            ModelCard(id=model.model, object=model.model_type.to_origin_model_type())
            for model in llm_models
        ]

        return ModelList(data=models_list)

    async def create_embeddings(
        self, provider: str, request: Request, embeddings_request: EmbeddingsRequest
    ):
        logger.info(
            f"Received create_embeddings request: {pprint.pformat(embeddings_request.dict())}"
        )
        model_instance = self._provider_manager.get_model_instance(
            provider=provider,
            model_type=ModelType.TEXT_EMBEDDING,
            model=embeddings_request.model,
        )
        texts = embeddings_request.input
        if isinstance(texts, str):
            texts = [texts]
        response = model_instance.invoke_text_embedding(texts=texts, user="abc-123")
        return await openai_embedding_text(response)

    async def create_chat_completion(
        self, provider: str, request: Request, chat_request: ChatCompletionRequest
    ):
        logger.info(
            f"Received chat completion request: {pprint.pformat(chat_request.dict())}"
        )

        model_instance = self._provider_manager.get_model_instance(
            provider=provider, model_type=ModelType.LLM, model=chat_request.model
        )
        prompt_messages = [
            convert_to_message(message) for message in chat_request.messages
        ]

        tools = []
        if chat_request.tools:
            tools = [
                PromptMessageTool(
                    name=f.function.name,
                    description=f.function.description,
                    parameters=f.function.parameters,
                )
                for f in chat_request.tools
            ]
        if chat_request.functions:
            tools.extend(
                [
                    PromptMessageTool(
                        name=f.name, description=f.description, parameters=f.parameters
                    )
                    for f in chat_request.functions
                ]
            )

        try:
            response = model_instance.invoke_llm(
                prompt_messages=prompt_messages,
                model_parameters={**chat_request.to_model_parameters_dict()},
                tools=tools,
                stop=chat_request.stop,
                stream=chat_request.stream,
                user="abc-123",
            )

            if chat_request.stream:
                return EventSourceResponse(
                    stream_openai_chat_completion(response),
                    media_type="text/event-stream",
                )
            else:
                return await openai_chat_completion(response)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        except InvokeError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )


def run(
    cfg: Dict,
    logging_conf: Optional[dict] = None,
    started_event: mp.Event = None,
):
    logging.config.dictConfig(logging_conf)  # type: ignore
    try:
        api = RESTFulOpenAIBootstrapBaseWeb.from_config(
            cfg=cfg.get("run_openai_api", {})
        )
        api.set_app_event(started_event=started_event)
        api.serve(logging_conf=logging_conf)

        async def pool_join_thread():
            await api.join()

        asyncio.run(pool_join_thread())
    except SystemExit:
        logger.info("SystemExit raised, exiting")
        raise