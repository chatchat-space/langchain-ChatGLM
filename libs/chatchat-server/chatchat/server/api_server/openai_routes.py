from __future__ import annotations

import asyncio
import base64
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, Iterable, Tuple

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
from openai import AsyncClient
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from chatchat.settings import Settings
from chatchat.server.utils import get_config_platforms, get_model_info, get_OpenAIClient
from chatchat.utils import build_logger

from .api_schemas import *

logger = build_logger()


DEFAULT_API_CONCURRENCIES = 5  # 默认单个模型最大并发数

model_semaphores: Dict[
    Tuple[str, str], asyncio.Semaphore
] = {}  # key: (model_name, platform)
openai_router = APIRouter(prefix="/v1", tags=["OpenAI 兼容平台整合接口"])


@asynccontextmanager
async def get_model_client(model_name: str) -> AsyncGenerator[AsyncClient]:
    """
    对重名模型进行调度，依次选择：空闲的模型 -> 当前访问数最少的模型
    """
    max_semaphore = 0
    selected_platform = ""
    model_infos = get_model_info(model_name=model_name, multiple=True)
    assert model_infos, f"specified model '{model_name}' cannot be found in MODEL_PLATFORMS."

    for m, c in model_infos.items():
        key = (m, c["platform_name"])
        api_concurrencies = c.get("api_concurrencies", DEFAULT_API_CONCURRENCIES)
        if key not in model_semaphores:
            model_semaphores[key] = asyncio.Semaphore(api_concurrencies)
        semaphore = model_semaphores[key]
        if semaphore._value >= api_concurrencies:
            selected_platform = c["platform_name"]
            break
        elif semaphore._value > max_semaphore:
            selected_platform = c["platform_name"]

    key = (m, selected_platform)
    semaphore = model_semaphores[key]
    try:
        await semaphore.acquire()
        yield get_OpenAIClient(platform_name=selected_platform, is_async=True)
    except Exception:
        logger.exception(f"failed when request to {key}")
    finally:
        semaphore.release()


async def openai_request(
    method, body, extra_json: Dict = {}, header: Iterable = [], tail: Iterable = []
):
    """
    helper function to make openai request with extra fields
    """

    async def generator():
        try:
            for x in header:
                if isinstance(x, str):
                    x = OpenAIChatOutput(content=x, object="chat.completion.chunk")
                elif isinstance(x, dict):
                    x = OpenAIChatOutput.model_validate(x)
                else:
                    raise RuntimeError(f"unsupported value: {header}")
                for k, v in extra_json.items():
                    setattr(x, k, v)
                yield x.model_dump_json()

            async for chunk in await method(**params):
                for k, v in extra_json.items():
                    setattr(chunk, k, v)
                yield chunk.model_dump_json()

            for x in tail:
                if isinstance(x, str):
                    x = OpenAIChatOutput(content=x, object="chat.completion.chunk")
                elif isinstance(x, dict):
                    x = OpenAIChatOutput.model_validate(x)
                else:
                    raise RuntimeError(f"unsupported value: {tail}")
                for k, v in extra_json.items():
                    setattr(x, k, v)
                yield x.model_dump_json()
        except asyncio.exceptions.CancelledError:
            logger.warning("streaming progress has been interrupted by user.")
            return
        except Exception as e:
            logger.error(f"openai request error: {e}")
            yield {"data": json.dumps({"error": str(e)})}

    params = body.model_dump(exclude_unset=True)
    if params.get("max_tokens") == 0:
        params["max_tokens"] = Settings.model_settings.MAX_TOKENS

    if hasattr(body, "stream") and body.stream:
        return EventSourceResponse(generator())
    else:
        result = await method(**params)
        for k, v in extra_json.items():
            setattr(result, k, v)
        return result.model_dump()


@openai_router.get("/models")
async def list_models() -> Dict:
    """
    整合所有平台的模型列表。
    """

    async def task(name: str, config: Dict):
        try:
            client = get_OpenAIClient(name, is_async=True)
            models = await client.models.list()
            return [{**x.model_dump(), "platform_name": name} for x in models.data]
        except Exception:
            logger.exception(f"failed request to platform: {name}")
            return []

    result = []
    tasks = [
        asyncio.create_task(task(name, config))
        for name, config in get_config_platforms().items()
    ]
    for t in asyncio.as_completed(tasks):
        result += await t

    return {"object": "list", "data": result}


@openai_router.post("/chat/completions")
async def create_chat_completions(
    body: OpenAIChatInput,
):
    async with get_model_client(body.model) as client:
        result = await openai_request(client.chat.completions.create, body)
        return result


@openai_router.post("/completions")
async def create_completions(
    request: Request,
    body: OpenAIChatInput,
):
    async with get_model_client(body.model) as client:
        return await openai_request(client.completions.create, body)


@openai_router.post("/embeddings")
async def create_embeddings(
    request: Request,
    body: OpenAIEmbeddingsInput,
):
    params = body.model_dump(exclude_unset=True)
    client = get_OpenAIClient(model_name=body.model)
    return (await client.embeddings.create(**params)).model_dump()


@openai_router.post("/images/generations")
async def create_image_generations(
    request: Request,
    body: OpenAIImageGenerationsInput,
):
    async with get_model_client(body.model) as client:
        return await openai_request(client.images.generate, body)


@openai_router.post("/images/variations")
async def create_image_variations(
    request: Request,
    body: OpenAIImageVariationsInput,
):
    async with get_model_client(body.model) as client:
        return await openai_request(client.images.create_variation, body)


@openai_router.post("/images/edit")
async def create_image_edit(
    request: Request,
    body: OpenAIImageEditsInput,
):
    async with get_model_client(body.model) as client:
        return await openai_request(client.images.edit, body)


@openai_router.post("/audio/translations", deprecated="暂不支持")
async def create_audio_translations(
    request: Request,
    body: OpenAIAudioTranslationsInput,
):
    async with get_model_client(body.model) as client:
        return await openai_request(client.audio.translations.create, body)


@openai_router.post("/audio/transcriptions", deprecated="暂不支持")
async def create_audio_transcriptions(
    request: Request,
    body: OpenAIAudioTranscriptionsInput,
):
    async with get_model_client(body.model) as client:
        return await openai_request(client.audio.transcriptions.create, body)


@openai_router.post("/audio/speech", deprecated="暂不支持")
async def create_audio_speech(
    request: Request,
    body: OpenAIAudioSpeechInput,
):
    async with get_model_client(body.model) as client:
        return await openai_request(client.audio.speech.create, body)


def _get_file_id(
    purpose: str,
    created_at: int,
    filename: str,
) -> str:
    today = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d")
    return base64.urlsafe_b64encode(f"{purpose}/{today}/{filename}".encode()).decode()


def _get_file_info(file_id: str) -> Dict:
    splits = base64.urlsafe_b64decode(file_id).decode().split("/")
    created_at = -1
    size = -1
    file_path = _get_file_path(file_id)
    if os.path.isfile(file_path):
        created_at = int(os.path.getmtime(file_path))
        size = os.path.getsize(file_path)

    return {
        "purpose": splits[0],
        "created_at": created_at,
        "filename": splits[2],
        "bytes": size,
    }


def _get_file_path(file_id: str) -> str:
    file_id = base64.urlsafe_b64decode(file_id).decode()
    return os.path.join(Settings.basic_settings.BASE_TEMP_DIR, "openai_files", file_id)


@openai_router.post("/files")
async def files(
    request: Request,
    file: UploadFile,
    purpose: str = "assistants",
) -> Dict:
    created_at = int(datetime.now().timestamp())
    file_id = _get_file_id(
        purpose=purpose, created_at=created_at, filename=file.filename
    )
    file_path = _get_file_path(file_id)
    file_dir = os.path.dirname(file_path)
    os.makedirs(file_dir, exist_ok=True)
    with open(file_path, "wb") as fp:
        shutil.copyfileobj(file.file, fp)
    file.file.close()

    return dict(
        id=file_id,
        filename=file.filename,
        bytes=file.size,
        created_at=created_at,
        object="file",
        purpose=purpose,
    )


@openai_router.get("/files")
def list_files(purpose: str) -> Dict[str, List[Dict]]:
    file_ids = []
    root_path = Path(Settings.basic_settings.BASE_TEMP_DIR) / "openai_files" / purpose
    for dir, sub_dirs, files in os.walk(root_path):
        dir = Path(dir).relative_to(root_path).as_posix()
        for file in files:
            file_id = base64.urlsafe_b64encode(
                f"{purpose}/{dir}/{file}".encode()
            ).decode()
            file_ids.append(file_id)
    return {
        "data": [{**_get_file_info(x), "id": x, "object": "file"} for x in file_ids]
    }


@openai_router.get("/files/{file_id}")
def retrieve_file(file_id: str) -> Dict:
    file_info = _get_file_info(file_id)
    return {**file_info, "id": file_id, "object": "file"}


@openai_router.get("/files/{file_id}/content")
def retrieve_file_content(file_id: str) -> Dict:
    file_path = _get_file_path(file_id)
    return FileResponse(file_path)


@openai_router.delete("/files/{file_id}")
def delete_file(file_id: str) -> Dict:
    file_path = _get_file_path(file_id)
    deleted = False

    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            deleted = True
    except:
        ...

    return {"id": file_id, "deleted": deleted, "object": "file"}
