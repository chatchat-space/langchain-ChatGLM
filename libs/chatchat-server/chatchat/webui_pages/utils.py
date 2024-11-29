# 该文件封装了对api.py的请求，可以被不同的webui使用
# 通过ApiRequest和AsyncApiRequest支持同步/异步调用

import base64
import contextlib
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import *

import httpx

from chatchat.settings import Settings
from chatchat.server.utils import api_address, get_httpx_client, set_httpx_config, get_default_embedding
from chatchat.utils import build_logger


logger = build_logger()

set_httpx_config()


class ApiRequest:
    """
    api.py调用的封装（同步模式）,简化api调用方式
    """

    def __init__(
        self,
        base_url: str = api_address(),
        timeout: float = Settings.basic_settings.HTTPX_DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self._use_async = False
        self._client = None

    @property
    def client(self):
        if self._client is None or self._client.is_closed:
            self._client = get_httpx_client(
                base_url=self.base_url, use_async=self._use_async, timeout=self.timeout
            )
        return self._client

    def get(
        self,
        url: str,
        params: Union[Dict, List[Tuple], bytes] = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                if stream:
                    return self.client.stream("GET", url, params=params, **kwargs)
                else:
                    return self.client.get(url, params=params, **kwargs)
            except Exception as e:
                msg = f"error when get {url}: {e}"
                logger.error(f"{e.__class__.__name__}: {msg}")
                retry -= 1

    def post(
        self,
        url: str,
        data: Dict = None,
        json: Dict = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                # print(kwargs)
                if stream:
                    return self.client.stream(
                        "POST", url, data=data, json=json, **kwargs
                    )
                else:
                    return self.client.post(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when post {url}: {e}"
                logger.error(f"{e.__class__.__name__}: {msg}")
                retry -= 1

    def delete(
        self,
        url: str,
        data: Dict = None,
        json: Dict = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                if stream:
                    return self.client.stream(
                        "DELETE", url, data=data, json=json, **kwargs
                    )
                else:
                    return self.client.delete(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when delete {url}: {e}"
                logger.error(f"{e.__class__.__name__}: {msg}")
                retry -= 1

    def _httpx_stream2generator(
        self,
        response: contextlib._GeneratorContextManager,
        as_json: bool = False,
    ):
        """
        将httpx.stream返回的GeneratorContextManager转化为普通生成器
        """

        async def ret_async(response, as_json):
            try:
                async with response as r:
                    chunk_cache = ""
                    async for chunk in r.aiter_text(None):
                        if not chunk:  # fastchat api yield empty bytes on start and end
                            continue
                        if as_json:
                            try:
                                if chunk.startswith("data: "):
                                    data = json.loads(chunk_cache + chunk[6:-2])
                                elif chunk.startswith(":"):  # skip sse comment line
                                    continue
                                else:
                                    data = json.loads(chunk_cache + chunk)

                                chunk_cache = ""
                                yield data
                            except Exception as e:
                                msg = f"接口返回json错误： ‘{chunk}’。错误信息是：{e}。"
                                logger.error(f"{e.__class__.__name__}: {msg}")

                                if chunk.startswith("data: "):
                                    chunk_cache += chunk[6:-2]
                                elif chunk.startswith(":"):  # skip sse comment line
                                    continue
                                else:
                                    chunk_cache += chunk
                                continue
                        else:
                            # print(chunk, end="", flush=True)
                            yield chunk
            except httpx.ConnectError as e:
                msg = f"无法连接API服务器，请确认 ‘api.py’ 已正常启动。({e})"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except httpx.ReadTimeout as e:
                msg = f"API通信超时，请确认已启动FastChat与API服务（详见Wiki '5. 启动 API 服务或 Web UI'）。（{e}）"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except Exception as e:
                msg = f"API通信遇到错误：{e}"
                logger.error(f"{e.__class__.__name__}: {msg}")
                yield {"code": 500, "msg": msg}

        def ret_sync(response, as_json):
            try:
                with response as r:
                    chunk_cache = ""
                    for chunk in r.iter_text(None):
                        if not chunk:  # fastchat api yield empty bytes on start and end
                            continue
                        if as_json:
                            try:
                                if chunk.startswith("data: "):
                                    data = json.loads(chunk_cache + chunk[6:-2])
                                elif chunk.startswith(":"):  # skip sse comment line
                                    continue
                                else:
                                    data = json.loads(chunk_cache + chunk)

                                chunk_cache = ""
                                yield data
                            except Exception as e:
                                msg = f"接口返回json错误： ‘{chunk}’。错误信息是：{e}。"
                                logger.error(f"{e.__class__.__name__}: {msg}")

                                if chunk.startswith("data: "):
                                    chunk_cache += chunk[6:-2]
                                elif chunk.startswith(":"):  # skip sse comment line
                                    continue
                                else:
                                    chunk_cache += chunk
                                continue
                        else:
                            # print(chunk, end="", flush=True)
                            yield chunk
            except httpx.ConnectError as e:
                msg = f"无法连接API服务器，请确认 ‘api.py’ 已正常启动。({e})"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except httpx.ReadTimeout as e:
                msg = f"API通信超时，请确认已启动FastChat与API服务（详见Wiki '5. 启动 API 服务或 Web UI'）。（{e}）"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except Exception as e:
                msg = f"API通信遇到错误：{e}"
                logger.error(f"{e.__class__.__name__}: {msg}")
                yield {"code": 500, "msg": msg}

        if self._use_async:
            return ret_async(response, as_json)
        else:
            return ret_sync(response, as_json)

    def _get_response_value(
        self,
        response: httpx.Response,
        as_json: bool = False,
        value_func: Callable = None,
    ):
        """
        转换同步或异步请求返回的响应
        `as_json`: 返回json
        `value_func`: 用户可以自定义返回值，该函数接受response或json
        """

        def to_json(r):
            try:
                return r.json()
            except Exception as e:
                msg = "API未能返回正确的JSON。" + str(e)
                logger.error(f"{e.__class__.__name__}: {msg}")
                return {"code": 500, "msg": msg, "data": None}

        if value_func is None:
            value_func = lambda r: r

        async def ret_async(response):
            if as_json:
                return value_func(to_json(await response))
            else:
                return value_func(await response)

        if self._use_async:
            return ret_async(response)
        else:
            if as_json:
                return value_func(to_json(response))
            else:
                return value_func(response)

    # 服务器信息
    def get_server_configs(self, **kwargs) -> Dict:
        response = self.post("/server/configs", **kwargs)
        return self._get_response_value(response, as_json=True)

    def get_prompt_template(
        self,
        type: str = "llm_chat",
        name: str = "default",
        **kwargs,
    ) -> str:
        data = {
            "type": type,
            "name": name,
        }
        response = self.post("/server/get_prompt_template", json=data, **kwargs)
        return self._get_response_value(response, value_func=lambda r: r.text)

    # 对话相关操作
    def chat_chat(
        self,
        query: str,
        metadata: dict,
        conversation_id: str = None,
        history_len: int = -1,
        history: List[Dict] = [],
        stream: bool = True,
        chat_model_config: Dict = None,
        tool_config: Dict = None,
        **kwargs,
    ):
        """
        对应api.py/chat/chat接口
        """
        data = {
            "query": query,
            "metadata": metadata,
            "conversation_id": conversation_id,
            "history_len": history_len,
            "history": history,
            "stream": stream,
            "chat_model_config": chat_model_config,
            "tool_config": tool_config,
        }

        # print(f"received input message:")
        # pprint(data)

        response = self.post("/chat/chat", json=data, stream=True, **kwargs)
        return self._httpx_stream2generator(response, as_json=True)

    def upload_temp_docs(
        self,
        files: List[Union[str, Path, bytes]],
        knowledge_id: str = None,
        chunk_size=Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap=Settings.kb_settings.OVERLAP_SIZE,
        zh_title_enhance=Settings.kb_settings.ZH_TITLE_ENHANCE,
    ):
        """
        对应api.py/knowledge_base/upload_temp_docs接口
        """

        def convert_file(file, filename=None):
            if isinstance(file, bytes):  # raw bytes
                file = BytesIO(file)
            elif hasattr(file, "read"):  # a file io like object
                filename = filename or file.name
            else:  # a local path
                file = Path(file).absolute().open("rb")
                filename = filename or os.path.split(file.name)[-1]
            return filename, file

        files = [convert_file(file) for file in files]
        data = {
            "knowledge_id": knowledge_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
        }

        response = self.post(
            "/knowledge_base/upload_temp_docs",
            data=data,
            files=[("files", (filename, file)) for filename, file in files],
        )
        return self._get_response_value(response, as_json=True)

    def file_chat(
        self,
        query: str,
        knowledge_id: str,
        top_k: int = Settings.kb_settings.VECTOR_SEARCH_TOP_K,
        score_threshold: float = Settings.kb_settings.SCORE_THRESHOLD,
        history: List[Dict] = [],
        stream: bool = True,
        model: str = None,
        temperature: float = 0.9,
        max_tokens: int = None,
        prompt_name: str = "default",
    ):
        """
        对应api.py/chat/file_chat接口
        """
        data = {
            "query": query,
            "knowledge_id": knowledge_id,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        response = self.post(
            "/chat/file_chat",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)

    # 知识库相关操作

    def list_knowledge_bases(
        self,
    ):
        """
        对应api.py/knowledge_base/list_knowledge_bases接口
        """
        response = self.get("/knowledge_base/list_knowledge_bases")
        return self._get_response_value(
            response, as_json=True, value_func=lambda r: r.get("data", [])
        )

    def create_knowledge_base(
        self,
        knowledge_base_name: str,
        vector_store_type: str = Settings.kb_settings.DEFAULT_VS_TYPE,
        embed_model: str = get_default_embedding(),
    ):
        """
        对应api.py/knowledge_base/create_knowledge_base接口
        """
        data = {
            "knowledge_base_name": knowledge_base_name,
            "vector_store_type": vector_store_type,
            "embed_model": embed_model,
        }

        response = self.post(
            "/knowledge_base/create_knowledge_base",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def delete_knowledge_base(
        self,
        knowledge_base_name: str,
    ):
        """
        对应api.py/knowledge_base/delete_knowledge_base接口
        """
        response = self.post(
            "/knowledge_base/delete_knowledge_base",
            json=f"{knowledge_base_name}",
        )
        return self._get_response_value(response, as_json=True)

    def list_kb_docs(
        self,
        knowledge_base_name: str,
    ):
        """
        对应api.py/knowledge_base/list_files接口
        """
        response = self.get(
            "/knowledge_base/list_files",
            params={"knowledge_base_name": knowledge_base_name},
        )
        return self._get_response_value(
            response, as_json=True, value_func=lambda r: r.get("data", [])
        )

    def search_kb_docs(
        self,
        knowledge_base_name: str,
        query: str = "",
        top_k: int = Settings.kb_settings.VECTOR_SEARCH_TOP_K,
        score_threshold: int = Settings.kb_settings.SCORE_THRESHOLD,
        file_name: str = "",
        metadata: dict = {},
    ) -> List:
        """
        对应api.py/knowledge_base/search_docs接口
        """
        data = {
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "file_name": file_name,
            "metadata": metadata,
        }

        response = self.post(
            "/knowledge_base/search_docs",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def upload_kb_docs(
        self,
        files: List[Union[str, Path, bytes]],
        knowledge_base_name: str,
        override: bool = False,
        to_vector_store: bool = True,
        chunk_size=Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap=Settings.kb_settings.OVERLAP_SIZE,
        zh_title_enhance=Settings.kb_settings.ZH_TITLE_ENHANCE,
        docs: Dict = {},
        not_refresh_vs_cache: bool = False,
    ):
        """
        对应api.py/knowledge_base/upload_docs接口
        """

        def convert_file(file, filename=None):
            if isinstance(file, bytes):  # raw bytes
                file = BytesIO(file)
            elif hasattr(file, "read"):  # a file io like object
                filename = filename or file.name
            else:  # a local path
                file = Path(file).absolute().open("rb")
                filename = filename or os.path.split(file.name)[-1]
            return filename, file

        files = [convert_file(file) for file in files]
        data = {
            "knowledge_base_name": knowledge_base_name,
            "override": override,
            "to_vector_store": to_vector_store,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
            "docs": docs,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        if isinstance(data["docs"], dict):
            data["docs"] = json.dumps(data["docs"], ensure_ascii=False)
        response = self.post(
            "/knowledge_base/upload_docs",
            data=data,
            files=[("files", (filename, file)) for filename, file in files],
        )
        return self._get_response_value(response, as_json=True)

    def delete_kb_docs(
        self,
        knowledge_base_name: str,
        file_names: List[str],
        delete_content: bool = False,
        not_refresh_vs_cache: bool = False,
    ):
        """
        对应api.py/knowledge_base/delete_docs接口
        """
        data = {
            "knowledge_base_name": knowledge_base_name,
            "file_names": file_names,
            "delete_content": delete_content,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        response = self.post(
            "/knowledge_base/delete_docs",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def update_kb_info(self, knowledge_base_name, kb_info):
        """
        对应api.py/knowledge_base/update_info接口
        """
        data = {
            "knowledge_base_name": knowledge_base_name,
            "kb_info": kb_info,
        }

        response = self.post(
            "/knowledge_base/update_info",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def update_kb_docs(
        self,
        knowledge_base_name: str,
        file_names: List[str],
        override_custom_docs: bool = False,
        chunk_size=Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap=Settings.kb_settings.OVERLAP_SIZE,
        zh_title_enhance=Settings.kb_settings.ZH_TITLE_ENHANCE,
        docs: Dict = {},
        not_refresh_vs_cache: bool = False,
    ):
        """
        对应api.py/knowledge_base/update_docs接口
        """
        data = {
            "knowledge_base_name": knowledge_base_name,
            "file_names": file_names,
            "override_custom_docs": override_custom_docs,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
            "docs": docs,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        if isinstance(data["docs"], dict):
            data["docs"] = json.dumps(data["docs"], ensure_ascii=False)

        response = self.post(
            "/knowledge_base/update_docs",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def recreate_vector_store(
        self,
        knowledge_base_name: str,
        allow_empty_kb: bool = True,
        vs_type: str = Settings.kb_settings.DEFAULT_VS_TYPE,
        embed_model: str = get_default_embedding(),
        chunk_size=Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap=Settings.kb_settings.OVERLAP_SIZE,
        zh_title_enhance=Settings.kb_settings.ZH_TITLE_ENHANCE,
    ):
        """
        对应api.py/knowledge_base/recreate_vector_store接口
        """
        data = {
            "knowledge_base_name": knowledge_base_name,
            "allow_empty_kb": allow_empty_kb,
            "vs_type": vs_type,
            "embed_model": embed_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
        }

        response = self.post(
            "/knowledge_base/recreate_vector_store",
            json=data,
            stream=True,
            timeout=None,
        )
        return self._httpx_stream2generator(response, as_json=True)

    def embed_texts(
        self,
        texts: List[str],
        embed_model: str = get_default_embedding(),
        to_query: bool = False,
    ) -> List[List[float]]:
        """
        对文本进行向量化，可选模型包括本地 embed_models 和支持 embeddings 的在线模型
        """
        data = {
            "texts": texts,
            "embed_model": embed_model,
            "to_query": to_query,
        }
        resp = self.post(
            "/other/embed_texts",
            json=data,
        )
        return self._get_response_value(
            resp, as_json=True, value_func=lambda r: r.get("data")
        )

    def chat_feedback(
        self,
        message_id: str,
        score: int,
        reason: str = "",
    ) -> int:
        """
        反馈对话评价
        """
        data = {
            "message_id": message_id,
            "score": score,
            "reason": reason,
        }
        resp = self.post("/chat/feedback", json=data)
        return self._get_response_value(resp)

    def list_tools(self) -> Dict:
        """
        列出所有工具
        """
        resp = self.get("/tools")
        return self._get_response_value(
            resp, as_json=True, value_func=lambda r: r.get("data", {})
        )

    def list_graphs(self) -> List[str]:
        """
        列出所有 graph
        """
        resp = self.get("/graphs")
        return self._get_response_value(
            resp, as_json=True, value_func=lambda r: r.get("data", {})
        )

    def call_tool(
        self,
        name: str,
        tool_input: Dict = {},
    ):
        """
        调用工具
        """
        data = {
            "name": name,
            "tool_input": tool_input,
        }
        resp = self.post("/tools/call", json=data)
        return self._get_response_value(
            resp, as_json=True, value_func=lambda r: r.get("data")
        )


class AsyncApiRequest(ApiRequest):
    def __init__(
        self, base_url: str = api_address(), timeout: float = Settings.basic_settings.HTTPX_DEFAULT_TIMEOUT
    ):
        super().__init__(base_url, timeout)
        self._use_async = True


def check_error_msg(data: Union[str, dict, list], key: str = "errorMsg") -> str:
    """
    return error message if error occured when requests API
    """
    if isinstance(data, dict):
        if key in data:
            return data[key]
        if "code" in data and data["code"] != 200:
            return data["msg"]
    return ""


def check_success_msg(data: Union[str, dict, list], key: str = "msg") -> str:
    """
    return error message if error occured when requests API
    """
    if (
        isinstance(data, dict)
        and key in data
        and "code" in data
        and data["code"] == 200
    ):
        return data[key]
    return ""


def get_img_base64(file_name: str) -> str:
    """
    get_img_base64 used in streamlit.
    absolute local path not working on windows.
    """
    image = f"{Settings.basic_settings.IMG_DIR}/{file_name}"
    # 读取图片
    with open(image, "rb") as f:
        buffer = BytesIO(f.read())
        base_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{base_str}"


if __name__ == "__main__":
    api = ApiRequest()
    aapi = AsyncApiRequest()

    # with api.chat_chat("你好") as r:
    #     for t in r.iter_text(None):
    #         print(t)

    # r = api.chat_chat("你好", no_remote_api=True)
    # for t in r:
    #     print(t)

    # r = api.duckduckgo_search_chat("室温超导最新研究进展", no_remote_api=True)
    # for t in r:
    #     print(t)

    # print(api.list_knowledge_bases())
