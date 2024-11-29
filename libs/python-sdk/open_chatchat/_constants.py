# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.
from typing import Literal

import httpx

RAW_RESPONSE_HEADER = "X-Stainless-Raw-Response"
OVERRIDE_CAST_TO_HEADER = "____stainless_override_cast_to"

# default timeout is 10 minutes
DEFAULT_TIMEOUT = httpx.Timeout(timeout=600.0, connect=5.0)
DEFAULT_MAX_RETRIES = 2
DEFAULT_CONNECTION_LIMITS = httpx.Limits(max_connections=1000, max_keepalive_connections=100)

INITIAL_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 8.0

EMBEDDING_MODEL: str = "bge-large-zh-v1.5"
HTTPX_TIMEOUT: float = 10.0
API_BASE_URI: str = 'http://127.0.0.1:7861/'

# 知识库相关
"""知识库中单段文本长度(不适用MarkdownHeaderTextSplitter)"""
CHUNK_SIZE: int = 250
"""知识库中相邻文本重合长度(不适用MarkdownHeaderTextSplitter)"""
OVERLAP_SIZE: int = 50
"""是否开启中文标题加强，以及标题增强的相关配置"""
ZH_TITLE_ENHANCE: bool = False
"""知识库匹配向量数量"""
VECTOR_SEARCH_TOP_K: int = 3  # TODO: 与 tool 配置项重复
"""知识库匹配相关度阈值，取值范围在0-2之间，SCORE越小，相关度越高，取到2相当于不筛选，建议设置在0.5左右"""
SCORE_THRESHOLD: float = 0.4
"""默认向量库/全文检索引擎类型"""
VS_TYPE: Literal["faiss", "milvus", "zilliz", "pg", "es", "relyt", "chromadb"] = "faiss"
# llm
TEMPERATURE: float = 0.7
LLM_MODEL = "chatglm-6b"
MAX_TOKENS = 2048

