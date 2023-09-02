from fastapi import Body
from fastapi.responses import StreamingResponse
from configs.model_config import llm_model_dict, LLM_MODEL, llm_model_args
from server.chat.utils import wrap_done
from langchain import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI, ChatAnthropic

from typing import AsyncIterable
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from typing import List
from server.chat.utils import History


def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
         history: List[History] = Body([],
                                       description="历史对话",
                                       examples=[[
                                           {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                           {"role": "assistant", "content": "虎头虎脑"}]]
                                       ),
         stream: bool = Body(False, description="流式输出"),
         ):
    history = [History.from_data(h) for h in history]

    async def chat_iterator(query: str,
                            history: List[History] = [],
                            ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()

        if LLM_MODEL == "Azure-OpenAI":
            model = AzureChatOpenAI(
                temperature=llm_model_args["temperature"],
                streaming=llm_model_args["streaming"],
                verbose=llm_model_args["verbose"],
                max_tokens=llm_model_args["max_tokens"],
                callbacks=[callback],
                openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
                openai_api_version=llm_model_dict[LLM_MODEL]["api_version"],
                deployment_name=llm_model_dict[LLM_MODEL]["deployment_name"],
                openai_api_key=llm_model_dict[LLM_MODEL]["api_key"],
                openai_api_type="azure",
            )
        elif LLM_MODEL == "Anthropic":
            model = ChatAnthropic(
                temperature=llm_model_args["temperature"],
                streaming=llm_model_args["streaming"],
                verbose=llm_model_args["verbose"],
                max_tokens_to_sample=llm_model_args["max_tokens"],
                callbacks=[callback],
                anthropic_api_key=llm_model_dict[LLM_MODEL]["api_key"],
                anthropic_api_url=llm_model_dict[LLM_MODEL]["api_base_url"],
                model=llm_model_dict[LLM_MODEL]["model_name"]
            )
        else:
            model = ChatOpenAI(
                temperature=llm_model_args["temperature"],
                streaming=llm_model_args["streaming"],
                verbose=llm_model_args["verbose"],
                max_tokens=llm_model_args["max_tokens"],
                callbacks=[callback],
                openai_api_key=llm_model_dict[LLM_MODEL]["api_key"],
                openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
                model_name=llm_model_dict[LLM_MODEL]["model_name"]
            )

        input_msg = History(role="user", content="{{ input }}").to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield token
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield answer

        await task

    return StreamingResponse(chat_iterator(query, history),
                             media_type="text/event-stream")
