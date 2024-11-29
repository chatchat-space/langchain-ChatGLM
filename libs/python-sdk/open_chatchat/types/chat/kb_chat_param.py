from typing import Optional, List, Literal

from pydantic import BaseModel, Field

from open_chatchat._constants import MAX_TOKENS, TEMPERATURE, SCORE_THRESHOLD, VECTOR_SEARCH_TOP_K, LLM_MODEL
from open_chatchat.types.chat.chat_message import ChatMessage


class KbChatParam(BaseModel):
    query: str = Field(..., description="用户输入", examples=["你好"]),
    mode: Literal["local_kb", "temp_kb", "search_engine"] = Field("local_kb", description="知识来源"),
    kb_name: str = Field("",
                         description="mode=local_kb时为知识库名称；temp_kb时为临时知识库ID，search_engine时为搜索引擎名称",
                         examples=["samples"]),
    top_k: int = Field(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
    score_threshold: float = Field(
        SCORE_THRESHOLD,
        description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
        ge=0,
        le=2,
    ),
    history: List[ChatMessage] = Field(
        [],
        description="历史对话",
        examples=[[
            {"role": "user",
             "content": "我们来玩成语接龙，我先来，生龙活虎"},
            {"role": "assistant",
             "content": "虎头虎脑"}]]
    ),
    stream: bool = Field(True, description="流式输出"),
    model: str = Field(LLM_MODEL, description="LLM 模型名称。"),
    temperature: float = Field(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
    max_tokens: Optional[int] = Field(
        MAX_TOKENS,
        description="限制LLM生成Token数量，默认None代表模型最大值"
    ),
    prompt_name: str = Field(
        "default",
        description="使用的prompt模板名称(在prompt_settings.yaml中配置)"
    ),
    return_direct: bool = Field(False, description="直接返回检索结果，不送入 LLM"),
