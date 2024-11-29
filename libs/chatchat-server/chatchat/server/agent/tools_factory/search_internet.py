from typing import Dict, List

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities.bing_search import BingSearchAPIWrapper
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.searx_search import SearxSearchWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from markdownify import markdownify
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

from chatchat.settings import Settings
from chatchat.server.pydantic_v1 import Field
from chatchat.server.utils import get_tool_config

from .tools_registry import BaseToolOutput, regist_tool, format_context


def searx_search(text ,config, top_k: int):
    search = SearxSearchWrapper(
        searx_host=config["host"],
        engines=config["engines"],
        categories=config["categories"],
    )
    search.params["language"] = config.get("language", "zh-CN")
    return search.results(text, top_k)


def bing_search(text, config, top_k:int):
    search = BingSearchAPIWrapper(
        bing_subscription_key=config["bing_key"],
        bing_search_url=config["bing_search_url"],
    )
    return search.results(text, top_k)


def tavily_search(text, config, top_k: int):
    search = TavilySearchAPIWrapper(
        tavily_api_key=config["tavily_key"],
    )
    return search.results(text, top_k)


def duckduckgo_search(text, config, top_k: int):
    search = DuckDuckGoSearchAPIWrapper()
    return search.results(text, top_k)


def metaphor_search(
    text: str,
    config: dict,
    top_k:int
) -> List[Dict]:
    from metaphor_python import Metaphor

    client = Metaphor(config["metaphor_api_key"])
    search = client.search(text, num_results=top_k, use_autoprompt=True)
    contents = search.get_contents().contents
    for x in contents:
        x.extract = markdownify(x.extract)
    if config["split_result"]:
        docs = [
            Document(page_content=x.extract, metadata={"link": x.url, "title": x.title})
            for x in contents
        ]
        text_splitter = RecursiveCharacterTextSplitter(
            ["\n\n", "\n", ".", " "],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )
        splitted_docs = text_splitter.split_documents(docs)
        if len(splitted_docs) > top_k:
            normal = NormalizedLevenshtein()
            for x in splitted_docs:
                x.metadata["score"] = normal.similarity(text, x.page_content)
            splitted_docs.sort(key=lambda x: x.metadata["score"], reverse=True)
            splitted_docs = splitted_docs[: top_k]

        docs = [
            {
                "snippet": x.page_content,
                "link": x.metadata["link"],
                "title": x.metadata["title"],
            }
            for x in splitted_docs
        ]
    else:
        docs = [
            {"snippet": x.extract, "link": x.url, "title": x.title} for x in contents
        ]

    return docs


SEARCH_ENGINES = {
    "bing": bing_search,
    "tavily": tavily_search,
    "duckduckgo": duckduckgo_search,
    "metaphor": metaphor_search,
    "searx": searx_search,
}


def search_result2docs(search_results) -> List[Document]:
    docs = []
    for result in search_results:
        doc = Document(
            page_content=result["snippet"] if "snippet" in result.keys() else "",
            metadata={
                "source": result["link"] if "link" in result.keys() else "",
                "filename": result["title"] if "title" in result.keys() else "",
            },
        )
        docs.append(doc)
    return docs


def search_engine(query: str, top_k:int=0, engine_name: str="", config: dict={}):
    config = config or get_tool_config("search_internet")
    if top_k <= 0:
        top_k = config.get("top_k", Settings.kb_settings.SEARCH_ENGINE_TOP_K)
    engine_name = engine_name or config.get("search_engine_name")
    search_engine_use = SEARCH_ENGINES[engine_name]
    results = search_engine_use(
        text=query, config=config["search_engine_config"][engine_name], top_k=top_k
    )
    docs = [x for x in search_result2docs(results) if x.page_content and x.page_content.strip()]
    return {"docs": docs, "search_engine": engine_name}


@regist_tool(title="互联网搜索")
def search_internet(query: str = Field(description="query for Internet search")):
    """Use this tool to use bing search engine to search the internet and get information."""
    return BaseToolOutput(search_engine(query=query), format=format_context)
