from typing import Union, Optional, Any

from pydantic import AnyUrl

from open_chatchat._constants import TEMPERATURE
from open_chatchat.types.standard_openai.base import OpenAIBaseInput


class OpenAIAudioTranslationsInput(OpenAIBaseInput):
    file: Union[Any, AnyUrl]
    model: str
    prompt: Optional[str] = None
    response_format: Optional[str] = None
    temperature: float = TEMPERATURE
