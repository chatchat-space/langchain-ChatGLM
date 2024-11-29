# LangChain 的 ArxivQueryRun 工具
from chatchat.server.pydantic_v1 import Field

from .tools_registry import BaseToolOutput, regist_tool


@regist_tool(title="ARXIV论文")
def arxiv(query: str = Field(description="The search query title")):
    """A wrapper around Arxiv.org for searching and retrieving scientific articles in various fields."""
    from langchain_community.utilities import ArxivAPIWrapper
    tool = ArxivAPIWrapper()
    return BaseToolOutput(tool.run(query))
