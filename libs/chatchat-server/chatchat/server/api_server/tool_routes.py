from __future__ import annotations

from typing import List

from fastapi import APIRouter, Body, Request

from chatchat.server.utils import BaseResponse, get_tool, get_tool_config
from chatchat.utils import build_logger


logger = build_logger()

tool_router = APIRouter(prefix="/tools", tags=["Toolkits"])


@tool_router.get("", response_model=BaseResponse)
async def list_tools():
    tools = get_tool()
    data = {
        t.name: {
            "name": t.name,
            "title": t.title,
            "description": t.description,
            "args": t.args,
            "config": get_tool_config(t.name),
        }
        for t in tools.values()
    }
    return {"data": data}


@tool_router.post("/call", response_model=BaseResponse)
async def call_tool(
    name: str = Body(examples=["calculate"]),
    tool_input: dict = Body({}, examples=[{"text": "3+5/2"}]),
):
    if tool := get_tool(name):
        try:
            result = await tool.ainvoke(tool_input)
            return {"data": result}
        except Exception:
            msg = f"failed to call tool '{name}'"
            logger.exception(msg)
            return {"code": 500, "msg": msg}
    else:
        return {"code": 500, "msg": f"no tool named '{name}'"}
