import requests
from requests.auth import HTTPBasicAuth
from urllib.parse import parse_qs
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from chatchat.server.pydantic_v1 import Field
from chatchat.server.utils import (
    get_tool_config,
    get_default_llm,
    get_ChatOpenAI,
    build_logger,
)

from .tools_registry import BaseToolOutput, regist_tool

logger = build_logger()

# Prompt for the prom_chain
PROMETHEUS_PROMPT_TEMPLATE = """ You are an expert in Prometheus, a powerful time-series monitoring service. 
Your main task is to translate users' specific requirements into PromQL queries. 
This includes understanding their monitoring needs, the specific metrics they are interested in, 
the time period for which they want the data, and any specific conditions or thresholds they want to apply. 
Your goal is to provide the most accurate and efficient PromQL query based on the given information.

Please return the PromQL in a format that can be used with the HTTP API, such as:

'query?query=up&time=2015-07-01T20:10:51.781Z' for instant data queries.
'query_range?query=up&start=2015-07-01T20:10:30.781Z&end=2015-07-01T20:11:00.781Z&step=15s' for range data queries.

I will automatically fill in the Prometheus IP and port. Please provide the query according to the example, 
and no other content is needed.

Question: {query} 
"""


def execute_promql_request(url: str, params: dict, auth: Optional[HTTPBasicAuth], promql: str) -> str:
    try:
        response = requests.get(url, params=params, auth=auth)
    except requests.exceptions.RequestException as e:
        logger.error(f"PromQL: {promql} 的错误: {e}")
        return f"PromQL: {promql} 的错误: {e}"

    if response.status_code == 200:
        return f"PromQL: {promql} 的查询结果: {response.json()}"
    else:
        logger.error(f"PromQL: {promql} 请求错误: {response.text}")
        return f"PromQL: {promql} 请求错误: {response.text}"


def query_prometheus(query: str, config: dict) -> str:
    prometheus_endpoint = config["prometheus_endpoint"]
    username = config["username"]
    password = config["password"]

    llm = get_ChatOpenAI(
        model_name=get_default_llm(),
        temperature=0.01,
        streaming=True,
        local_wrap=False,
    )

    prometheus_prompt = ChatPromptTemplate.from_template(PROMETHEUS_PROMPT_TEMPLATE)
    output_parser = StrOutputParser()

    prometheus_chain = (
        {"query": RunnablePassthrough()}
        | prometheus_prompt
        | llm
        | output_parser
    )

    # 根据用户名和密码是否存在来设置 auth 参数
    auth = HTTPBasicAuth(username, password) if username and password else None

    promql = prometheus_chain.invoke(query)
    logger.info(f"PromQL: {promql}")

    # debug
    # promql = 'query?query=kube_pod_status_phase{namespace="default"}'

    # 从 promql 字符串中提取 query_type 和 参数
    try:
        query_type, query_params = promql.split('?')
    except ValueError as e:
        logger.error(f"Promql split error: {e}")
        content = f"PromQL: {promql} 的错误: {e}\n"
        return content

    prometheus_url = f"{prometheus_endpoint}/api/v1/{query_type}"

    # params 的格式如下:
    """
    {
    "query": "kube_pod_status_phase{namespace=\"default\"}"
    }
    """
    params = {k: v[0] for k, v in parse_qs(query_params).items()}

    # 向 Prometheus 发起 HTTP 请求
    content = execute_promql_request(prometheus_url, params, auth, promql)

    logger.info(content)
    return content


@regist_tool(title="Prometheus对话")
def text2promql(
        query: str = Field(
            description="Tool for querying a Prometheus server, No need for PromQL statements, "
                        "just input the natural language that you want to chat with prometheus"
        )
):
    """Use this tool to chat with prometheus, Input natural language,
    then it will convert it into PromQL and execute it in the prometheus, then return the execution result."""
    tool_config = get_tool_config("text2promql")
    return BaseToolOutput(query_prometheus(query=query, config=tool_config))
