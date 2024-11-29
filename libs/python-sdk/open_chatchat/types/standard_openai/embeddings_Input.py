from typing import Union, List, Optional, Literal

from open_chatchat.types.standard_openai.base import OpenAIBaseInput


class OpenAIEmbeddingsInput(OpenAIBaseInput):
    input: Union[str, List[str]]
    model: str
    dimensions: Optional[int] = None
    encoding_format: Optional[Literal["float", "base64"]] = None
