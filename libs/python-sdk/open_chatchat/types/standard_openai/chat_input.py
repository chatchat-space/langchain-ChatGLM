from typing import List, Optional, Dict, Union

from open_chatchat._constants import LLM_MODEL, TEMPERATURE
from open_chatchat.types.standard_openai.base import OpenAIBaseInput
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    completion_create_params,
)


class OpenAIChatInput(OpenAIBaseInput):
    messages: List[Union[Dict, ChatCompletionMessageParam]] = []
    model: str = LLM_MODEL
    frequency_penalty: Optional[float] = None
    function_call: Optional[completion_create_params.FunctionCall] = []
    functions: List[completion_create_params.Function] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: completion_create_params.ResponseFormat = None
    seed: Optional[int] = None
    stop: Union[Optional[str], List[str]] = None
    stream: Optional[bool] = True
    temperature: Optional[float] = TEMPERATURE
    tool_choice: Optional[Union[ChatCompletionToolChoiceOptionParam, str]] = None
    tools: List[Union[ChatCompletionToolParam, str]] = None
    top_logprobs: Optional[int] = None
    top_p: Optional[float] = None
