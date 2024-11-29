from typing import Union, Any

from pydantic import AnyUrl

from open_chatchat.types.standard_openai.image_base_input import OpenAIImageBaseInput


class OpenAIImageVariationsInput(OpenAIImageBaseInput):
    image: Union[Any, AnyUrl]
