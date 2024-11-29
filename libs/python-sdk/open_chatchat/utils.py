import base64
import inspect
import os
from io import BytesIO
from pathlib import Path
from typing import Union, List, Dict

import httpx
from pydantic import BaseModel
from typing_extensions import TypeGuard

from open_chatchat._constants import HTTPX_TIMEOUT


def get_httpx_client(
        use_async: bool = False,
        proxies: Union[str, Dict] = None,
        timeout: float = HTTPX_TIMEOUT,
        unused_proxies: List[str] = [],
        **kwargs,
) -> Union[httpx.Client, httpx.AsyncClient]:
    """
    helper to get httpx client with default proxies that bypass local addesses.
    """
    default_proxies = {
        # do not use proxy for locahost
        "all://127.0.0.1": None,
        "all://localhost": None,
    }
    # do not use proxy for user deployed fastchat servers
    for x in unused_proxies:
        host = ":".join(x.split(":")[:2])
        default_proxies.update({host: None})

    # get proxies from system envionrent
    # proxy not str empty string, None, False, 0, [] or {}
    default_proxies.update(
        {
            "http://": (
                os.environ.get("http_proxy")
                if os.environ.get("http_proxy")
                   and len(os.environ.get("http_proxy").strip())
                else None
            ),
            "https://": (
                os.environ.get("https_proxy")
                if os.environ.get("https_proxy")
                   and len(os.environ.get("https_proxy").strip())
                else None
            ),
            "all://": (
                os.environ.get("all_proxy")
                if os.environ.get("all_proxy")
                   and len(os.environ.get("all_proxy").strip())
                else None
            ),
        }
    )
    for host in os.environ.get("no_proxy", "").split(","):
        if host := host.strip():
            # default_proxies.update({host: None}) # Origin code
            default_proxies.update(
                {"all://" + host: None}
            )  # PR 1838 fix, if not add 'all://', httpx will raise error

    # merge default proxies with user provided proxies
    if isinstance(proxies, str):
        proxies = {"all://": proxies}

    if isinstance(proxies, dict):
        default_proxies.update(proxies)

    # construct Client
    kwargs.update(timeout=timeout, proxies=default_proxies)

    if use_async:
        return httpx.AsyncClient(**kwargs)
    else:
        return httpx.Client(**kwargs)


def set_httpx_config(
        timeout: float = HTTPX_TIMEOUT,
        proxy: Union[str, Dict] = None,
        unused_proxies: List[str] = [],
):
    """
    设置httpx默认timeout。httpx默认timeout是5秒，在请求LLM回答时不够用。
    将本项目相关服务加入无代理列表，避免fastchat的服务器请求错误。(windows下无效)
    对于chatgpt等在线API，如要使用代理需要手动配置。搜索引擎的代理如何处置还需考虑。
    """

    import os

    import httpx

    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout

    # 在进程范围内设置系统级代理
    proxies = {}
    if isinstance(proxy, str):
        for n in ["http", "https", "all"]:
            proxies[n + "_proxy"] = proxy
    elif isinstance(proxy, dict):
        for n in ["http", "https", "all"]:
            if p := proxy.get(n):
                proxies[n + "_proxy"] = p
            elif p := proxy.get(n + "_proxy"):
                proxies[n + "_proxy"] = p

    for k, v in proxies.items():
        os.environ[k] = v

    # set host to bypass proxy
    no_proxy = [
        x.strip() for x in os.environ.get("no_proxy", "").split(",") if x.strip()
    ]
    no_proxy += [
        # do not use proxy for locahost
        "http://127.0.0.1",
        "http://localhost",
    ]
    # do not use proxy for user deployed fastchat servers
    for x in unused_proxies:
        host = ":".join(x.split(":")[:2])
        if host not in no_proxy:
            no_proxy.append(host)
    os.environ["NO_PROXY"] = ",".join(no_proxy)

    def _get_proxies():
        return proxies

    import urllib.request

    urllib.request.getproxies = _get_proxies


def get_img_base64(file_path: str) -> str:
    """
    get_img_base64 used in streamlit.
    """
    image = file_path
    # 读取图片
    with open(image, "rb") as f:
        buffer = BytesIO(f.read())
        base_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{base_str}"


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


def get_variable(*args):
    for var in args:
        if var:
            return var
    return None


def is_dict(obj: object) -> TypeGuard[dict[object, object]]:
    return isinstance(obj, dict)


def model_to_dict(model: BaseModel) -> dict[str, object]:
    return model.dict()


def get_function_default_params(func) -> dict:
    """
    获取函数的参数及其默认值。

    参数:
        func (function): 要分析的函数。

    返回:
        dict: 一个包含参数名称及其默认值的字典。
    """
    signature = inspect.signature(func)
    params = signature.parameters
    params_dict = {}

    for param_name, param in params.items():
        if param.default is inspect.Parameter.empty:
            params_dict[param_name] = None
        else:
            params_dict[param_name] = param.default

    return params_dict


def merge_dicts(dict1, dict2) -> dict:
    """
    合并两个字典，优先使用第一个字典中的非空值。

    参数:
        dict1 (dict): 第一个字典。
        dict2 (dict): 第二个字典。

    返回:
        dict: 合并后的字典。
    """
    merged_dict = {}

    # 遍历两个字典的键集合
    all_keys = set(dict1.keys()).union(set(dict2.keys()))

    for key in all_keys:
        value1 = dict1.get(key)
        value2 = dict2.get(key)

        # 如果第一个字典中的值不为空，使用第一个字典的值
        if value1:
            merged_dict[key] = value1
        else:
            # 否则使用第二个字典中的值
            merged_dict[key] = value2

    return merged_dict


def convert_file(file, filename=None):
    if isinstance(file, bytes):  # raw bytes
        file = BytesIO(file)
    elif hasattr(file, "read"):  # a file io like object
        filename = filename or file.name
    else:  # a local path
        file = Path(file).absolute().open("rb")
        filename = filename or os.path.split(file.name)[-1]
    return filename, file