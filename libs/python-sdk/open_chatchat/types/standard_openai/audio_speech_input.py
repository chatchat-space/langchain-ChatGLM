from typing import Optional, Literal

from open_chatchat.types.standard_openai.base import OpenAIBaseInput


class OpenAIAudioSpeechInput(OpenAIBaseInput):
    input: str
    model: str
    voice: str
    response_format: Optional[
        Literal["mp3", "opus", "aac", "flac", "pcm", "wav"]
    ] = None
    speed: Optional[float] = None
