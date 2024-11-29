from typing import Optional, Literal

from open_chatchat.types.standard_openai.base import OpenAIBaseInput


class OpenAIImageBaseInput(OpenAIBaseInput):
    model: str
    n: int = 1
    response_format: Optional[Literal["url", "b64_json"]] = None
    size: Optional[
        Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
    ] = "256x256"