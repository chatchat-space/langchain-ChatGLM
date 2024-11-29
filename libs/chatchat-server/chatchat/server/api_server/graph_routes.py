from __future__ import annotations

from typing import List, Literal

from fastapi import APIRouter, Request

from chatchat.settings import Settings
from chatchat.server.utils import is_graph_enabled, get_default_graph, BaseResponse


graph_router = APIRouter(prefix="/graphs", tags=["LangGraph"])

@graph_router.get("/graph_enabled")
def get_graph_enabled():
    return is_graph_enabled()


@graph_router.get("", response_model=BaseResponse)
async def list_graph():
    return {"data": Settings.tool_settings.SUPPORT_GRAPHS}
