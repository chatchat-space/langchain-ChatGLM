from typing import Optional, List, Literal, Union

from pydantic import Field

from open_chatchat._constants import MAX_TOKENS, LLM_MODEL, TEMPERATURE, SCORE_THRESHOLD, VECTOR_SEARCH_TOP_K
from open_chatchat.api_client import ApiClient
from open_chatchat.types.chat.chat_feedback_param import ChatFeedbackParam
from open_chatchat.types.chat.chat_message import ChatMessage
from open_chatchat.types.chat.file_chat_param import FileChatParam
from open_chatchat.types.chat.kb_chat_param import KbChatParam

API_URI_CHAT_FEEDBACK = "/chat/feedback"
API_URI_FILE_CHAT = "/chat/file_chat"
API_URI_KB_CHAT = "/chat/kb_chat"


class ChatClient(ApiClient):

    def chat_feedback(self,
                      message_id: str,
                      score: int = 100,
                      reason: str = ""):
        data = ChatFeedbackParam(
            message_id=message_id,
            score=score,
            reason=reason,
        ).dict()
        resp = self._post(API_URI_CHAT_FEEDBACK, json=data)
        return self._get_response_value(resp, as_json=True)

    def kb_chat(self,
                query: str,
                mode: Literal["local_kb", "temp_kb", "search_engine"] = "local_kb",
                kb_name: str = "",
                top_k: int = VECTOR_SEARCH_TOP_K,
                score_threshold: float = SCORE_THRESHOLD,
                history: List[Union[ChatMessage, dict]] = [],
                stream: bool = True,
                model: str = LLM_MODEL,
                temperature: float = TEMPERATURE,
                max_tokens: Optional[int] = MAX_TOKENS,
                prompt_name: str = "default",
                return_direct: bool = False,
                ):
        kb_chat_param = KbChatParam(
            query=query,
            mode=mode,
            kb_name=kb_name,
            top_k=top_k,
            score_threshold=score_threshold,
            history=history,
            stream=stream,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_name=prompt_name,
            return_direct=return_direct,
        ).dict()
        response = self._post(API_URI_KB_CHAT, json=kb_chat_param, stream=True)
        return self._httpx_stream2generator(response, as_json=True)

    def file_chat(self,
                  query: str,
                  knowledge_id: str,
                  top_k: int = VECTOR_SEARCH_TOP_K,
                  score_threshold: float = SCORE_THRESHOLD,
                  history: List[Union[dict, ChatMessage]] = [],
                  stream: bool = True,
                  model_name: str = LLM_MODEL,
                  temperature: float = 0.01,
                  max_tokens: Optional[int] = MAX_TOKENS,
                  prompt_name: str = "default",
                  ):
        file_chat_param = FileChatParam(
            query=query,
            knowledge_id=knowledge_id,
            top_k=top_k,
            score_threshold=score_threshold,
            history=history,
            stream=stream,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_name=prompt_name,
        ).dict()
        response = self._post(API_URI_FILE_CHAT, json=file_chat_param, stream=True)
        return self._httpx_stream2generator(response, as_json=True)
