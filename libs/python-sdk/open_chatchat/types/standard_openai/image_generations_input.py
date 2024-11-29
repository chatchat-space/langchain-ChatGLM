from typing import Literal, Optional

from open_chatchat.types.standard_openai.image_base_input import OpenAIImageBaseInput


class OpenAIImageGenerationsInput(OpenAIImageBaseInput):
    prompt: str
    quality: Literal["standard", "hd"] = None
    style: Optional[Literal["vivid", "natural"]] = None