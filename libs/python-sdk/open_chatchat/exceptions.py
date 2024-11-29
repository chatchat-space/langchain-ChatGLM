from __future__ import annotations

from open_chatchat.utils import is_dict
from typing import Any, Optional, cast
from typing_extensions import Literal

import httpx

__all__ = [
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
]


class ChatChatError(Exception):
    pass


class APIError(ChatChatError):
    message: str
    request: httpx.Request

    body: object | None
    """
        API响应体。
        如果API响应了一个有效的JSON结构，那么这个属性将是
        解码结果。
        如果它不是一个有效的JSON结构，那么这将是原始响应。
        如果没有与此错误相关的响应，那么它将是' None '。
    """

    code: Optional[str] = None
    param: Optional[str] = None
    type: Optional[str]

    def __init__(self, message: str, request: httpx.Request, *, body: object | None) -> None:
        super().__init__(message)
        self.request = request
        self.message = message
        self.body = body

        if is_dict(body):
            self.code = cast(str, body.get("code"))
            self.param = cast(str, body.get("param"))
            self.type = cast(str, body.get("type"))
        else:
            self.code = None
            self.param = None
            self.type = None


class APIResponseValidationError(APIError):
    response: httpx.Response
    status_code: int

    def __init__(self, response: httpx.Response, body: object | None, *, message: str | None = None) -> None:
        super().__init__(message or "API返回的数据对预期的模式无效。", response.request, body=body)
        self.response = response
        self.status_code = response.status_code


class APIStatusError(APIError):
    """当API响应的状态码为4xx或5xx时引发。"""

    response: httpx.Response
    status_code: int
    request_id: str | None

    def __init__(self, message: str, *, response: httpx.Response, body: object | None) -> None:
        super().__init__(message, response.request, body=body)
        self.response = response
        self.status_code = response.status_code
        self.request_id = response.headers.get("x-request-id")


class APIConnectionError(APIError):
    def __init__(self, *, message: str = "连接错误", request: httpx.Request) -> None:
        super().__init__(message, request, body=None)


class APITimeoutError(APIConnectionError):
    def __init__(self, request: httpx.Request) -> None:
        super().__init__(message="请求超时", request=request)


class BadRequestError(APIStatusError):
    status_code: Literal[400] = 400


class AuthenticationError(APIStatusError):
    status_code: Literal[401] = 401


class PermissionDeniedError(APIStatusError):
    status_code: Literal[403] = 403


class NotFoundError(APIStatusError):
    status_code: Literal[404] = 404


class ConflictError(APIStatusError):
    status_code: Literal[409] = 409


class UnprocessableEntityError(APIStatusError):
    status_code: Literal[422] = 422


class RateLimitError(APIStatusError):
    status_code: Literal[429] = 429


class InternalServerError(APIStatusError):
    pass
