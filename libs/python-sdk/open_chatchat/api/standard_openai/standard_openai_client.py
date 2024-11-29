from typing import Dict, List

from open_chatchat.api_client import ApiClient, get
from open_chatchat.types.standard_openai.audio_speech_input import OpenAIAudioSpeechInput
from open_chatchat.types.standard_openai.audio_transcriptions_input import OpenAIAudioTranscriptionsInput
from open_chatchat.types.standard_openai.audio_translations_input import OpenAIAudioTranslationsInput
from open_chatchat.types.standard_openai.chat_input import OpenAIChatInput
from open_chatchat.types.standard_openai.embeddings_Input import OpenAIEmbeddingsInput
from open_chatchat.types.standard_openai.image_edits_input import OpenAIImageEditsInput
from open_chatchat.types.standard_openai.image_generations_input import OpenAIImageGenerationsInput
from open_chatchat.types.standard_openai.image_variations_input import OpenAIImageVariationsInput

API_UTI_STANDARD_OPENAI_LIST_MODELS = "/v1/models"
API_UTI_STANDARD_OPENAI_CHAT_COMPLETIONS = "/v1/chat/completions"
API_UTI_STANDARD_OPENAI_COMPLETIONS = "/v1/chat/completions"
API_UTI_STANDARD_OPENAI_EMBEDDINGS = "/v1/embeddings"

API_UTI_STANDARD_OPENAI_IMAGE_GENERATIONS = "/v1//images/generations"
API_UTI_STANDARD_OPENAI_IMAGE_VARIATIONS = "/v1//images/variations"
API_UTI_STANDARD_OPENAI_IMAGE_EDIT = "/v1//images/edit"

API_UTI_STANDARD_OPENAI_AUDIO_TRANSLATIONS = "/v1//audio/translations"
API_UTI_STANDARD_OPENAI_AUDIO_TRANSCRIPTIONS = "/v1//audio/transcriptions"
API_UTI_STANDARD_OPENAI_AUDIO_SPEECH = "/v1/audio/speech"

API_UTI_STANDARD_OPENAI_FILES = "/v1/files"
API_UTI_STANDARD_OPENAI_LIST_FILES = "/v1/list_files"
API_UTI_STANDARD_OPENAI_RETRIEVE_FILE = "/v1//files/{file_id}"
API_UTI_STANDARD_OPENAI_RETRIEVE_FILE_CONTENT = "/v1//files/{file_id}/content"
API_UTI_STANDARD_OPENAI_DELETE_FILE = "/v1//files/{file_id}"


class StandardOpenaiClient(ApiClient):

    def list_models(self) -> dict:
        response = self._get(API_UTI_STANDARD_OPENAI_LIST_MODELS)
        return self._get_response_value(response, as_json=True)

    def chat_completions(self, chat_input: OpenAIChatInput) -> dict:
        response = self._post(API_UTI_STANDARD_OPENAI_CHAT_COMPLETIONS, json=chat_input.dict(), stream=True)
        return self._httpx_stream2generator(response, as_json=True)

    def completions(self, chat_input: OpenAIChatInput) -> dict:
        response = self._post(API_UTI_STANDARD_OPENAI_COMPLETIONS, json=chat_input.dict(), stream=True)
        return self._httpx_stream2generator(response, as_json=True)

    def embeddings(self, embeddings_input: OpenAIEmbeddingsInput):
        response = self._post(API_UTI_STANDARD_OPENAI_EMBEDDINGS, json=embeddings_input.dict())
        return self._get_response_value(response, as_json=True)

    def image_generations(
            self,
            data: OpenAIImageGenerationsInput,
    ):
        response = self._post(API_UTI_STANDARD_OPENAI_IMAGE_GENERATIONS, json=data.dict())
        return self._get_response_value(response, as_json=True)

    def image_variations(
            self,
            data: OpenAIImageVariationsInput,
    ):
        response = self._post(API_UTI_STANDARD_OPENAI_IMAGE_VARIATIONS, json=data.dict())
        return self._get_response_value(response, as_json=True)

    def image_edit(
            self,
            data: OpenAIImageEditsInput,
    ):
        response = self._post(API_UTI_STANDARD_OPENAI_IMAGE_EDIT, json=data.dict())
        return self._get_response_value(response, as_json=True)

    def audio_translations(
            self,
            data: OpenAIAudioTranslationsInput,
    ):
        response = self._post(API_UTI_STANDARD_OPENAI_AUDIO_TRANSLATIONS, json=data.dict())
        return self._get_response_value(response, as_json=True)

    def audio_transcriptions(
            self,
            data: OpenAIAudioTranscriptionsInput,
    ):
        response = self._post(API_UTI_STANDARD_OPENAI_AUDIO_TRANSCRIPTIONS, json=data.dict())
        return self._get_response_value(response, as_json=True)

    def audio_speech(
            self,
            data: OpenAIAudioSpeechInput,
    ):
        response = self._post(API_UTI_STANDARD_OPENAI_AUDIO_SPEECH, json=data.dict())
        return self._get_response_value(response, as_json=True)

    # todo 待完成
    async def files(
            self,
            file: str,
            purpose: str = "assistants",
    ) -> Dict:
        response = self._post(API_UTI_STANDARD_OPENAI_FILES)
        return self._get_response_value(response, as_json=True)

    def list_files(self, purpose: str) -> Dict[str, List[Dict]]:
        response = self._get(API_UTI_STANDARD_OPENAI_LIST_FILES)
        return self._get_response_value(response, as_json=True)

    def retrieve_file(self, file_id: str) -> Dict:
        response = self._get(API_UTI_STANDARD_OPENAI_RETRIEVE_FILE.format(file_id=file_id))
        return self._get_response_value(response, as_json=True)

    def retrieve_file_content(self, file_id: str) -> Dict:
        response = self._get(API_UTI_STANDARD_OPENAI_RETRIEVE_FILE_CONTENT.format(file_id=file_id))
        return self._get_response_value(response, as_json=True)

    def delete_file(self, file_id: str) -> Dict:
        response = self._delete(API_UTI_STANDARD_OPENAI_DELETE_FILE.format(file_id=file_id))
        return self._get_response_value(response, as_json=True)
