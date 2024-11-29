from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult


def dumps(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


class AgentStatus:
    llm_start: int = 1
    llm_new_token: int = 2
    llm_end: int = 3
    agent_action: int = 4
    agent_finish: int = 5
    tool_start: int = 6
    tool_end: int = 7
    error: int = 8


class AgentExecutorAsyncIteratorCallbackHandler(AsyncIteratorCallbackHandler):
    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()
        self.out = True

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        data = {
            "status": AgentStatus.llm_start,
            "text": "",
        }
        self.done.clear()
        self.queue.put_nowait(dumps(data))

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        special_tokens = ["\nAction:", "\nObservation:", "<|observation|>"]
        for stoken in special_tokens:
            if stoken in token:
                before_action = token.split(stoken)[0]
                data = {
                    "status": AgentStatus.llm_new_token,
                    "text": before_action + "\n",
                }
                self.queue.put_nowait(dumps(data))
                self.out = False
                break

        if token is not None and token != "" and self.out:
            data = {
                "status": AgentStatus.llm_new_token,
                "text": token,
            }
            self.queue.put_nowait(dumps(data))

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        data = {
            "status": AgentStatus.llm_start,
            "text": "",
        }
        self.done.clear()
        self.queue.put_nowait(dumps(data))

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        data = {
            "status": AgentStatus.llm_end,
            "text": response.generations[0][0].message.content,
        }
        self.queue.put_nowait(dumps(data))

    async def on_llm_error(
        self, error: Exception | KeyboardInterrupt, **kwargs: Any
    ) -> None:
        data = {
            "status": AgentStatus.error,
            "text": str(error),
        }
        self.queue.put_nowait(dumps(data))

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        data = {
            "run_id": str(run_id),
            "status": AgentStatus.tool_start,
            "tool": serialized["name"],
            "tool_input": input_str,
        }
        self.queue.put_nowait(dumps(data))

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        # if not isinstance(output, str):
        #     output = str(output)
        data = {
            "run_id": str(run_id),
            "status": AgentStatus.tool_end,
            "tool_output": output.data,
        }

        self.queue.put_nowait(dumps(data))

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""
        data = {
            "run_id": str(run_id),
            "status": AgentStatus.tool_end,
            "tool_output": str(error),
            "is_error": True,
        }
        # self.done.clear()
        self.queue.put_nowait(dumps(data))

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        data = {
            "status": AgentStatus.agent_action,
            "tool_name": action.tool,
            "tool_input": action.tool_input,
            "text": action.log,
        }
        self.queue.put_nowait(dumps(data))

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if "Thought:" in finish.return_values["output"]:
            finish.return_values["output"] = finish.return_values["output"].replace(
                "Thought:", ""
            )

        data = {
            "status": AgentStatus.agent_finish,
            "text": finish.return_values["output"],
        }
        self.queue.put_nowait(dumps(data))

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.done.set()
        self.out = True
