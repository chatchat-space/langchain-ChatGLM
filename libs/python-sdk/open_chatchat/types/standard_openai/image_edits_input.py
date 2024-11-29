from typing import Any, Union

from pydantic import AnyUrl

from open_chatchat.types.standard_openai.image_variations_input import OpenAIImageVariationsInput


class OpenAIImageEditsInput(OpenAIImageVariationsInput):
    prompt: str
    mask: Union[Any, AnyUrl]