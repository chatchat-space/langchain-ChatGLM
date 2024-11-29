"""
通过jina-ai/reader项目，将url内容处理为llm易于理解的文本形式
"""
import requests
import re

from chatchat.server.pydantic_v1 import Field
from chatchat.server.agent.tools_factory.tools_registry import format_context
from chatchat.server.utils import get_tool_config, build_logger

from .tools_registry import BaseToolOutput, regist_tool

logger = build_logger()


@regist_tool(title="URL内容阅读")
def url_reader(
        url: str = Field(
            description="Based on the provided URL, call this function to retrieve the entire content of the web page. "
                        "Focus on extracting the article body, including text and URLs of images, videos, and other media. "
                        "Structure the article body as much as possible and return it in JSON format. "
                        "Do not perform any summary, generalization, or additional processing of the content. "
                        "Simply extract and structure the raw content exactly as it appears on the web page. "
                        "If the URL is inaccessible or the content cannot be parsed, return an appropriate error message."),
):
    """Use this tool to get the clear content of a URL."""

    tool_config = get_tool_config("url_reader")
    timeout = tool_config.get("timeout")

    # 提取url文本中的网页链接部分。url文本可能是一句话
    url_pattern = r'http[s]?://[a-zA-Z0-9./?&=_%#-]+'
    match = re.search(url_pattern, url)
    url = match.group(0) if match else None

    if url is None:
        return BaseToolOutput({"error": "No URL"})

    reader_url = "https://r.jina.ai/{url}".format(url=url)

    try:
        response = requests.get(reader_url, timeout=timeout)
        if response.status_code == 200:
            return BaseToolOutput(
                {"result": response.text,
                 "docs": [{"page_content": response.text, "metadata": {'source': url, 'id': ''}}]},
                format=format_context)
        else:
            return BaseToolOutput({"error": "Failed to fetch URL with status code: {response.status_code}"})
    except requests.exceptions.Timeout:
        # 打印超时错误
        logger.error("Request timed out with URL内容阅读")
        return BaseToolOutput({"error": "Timeout"})
    except requests.exceptions.RequestException as e:
        # 打印请求异常
        logger.error(f"Request failed with URL内容阅读: {e}")
        return BaseToolOutput({"error": "Request failed"})
