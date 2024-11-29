from typing import Optional, List, Literal

from open_chatchat.types.standard_openai.audio_translations_input import OpenAIAudioTranslationsInput


class OpenAIAudioTranscriptionsInput(OpenAIAudioTranslationsInput):
    language: Optional[str] = None
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = None
