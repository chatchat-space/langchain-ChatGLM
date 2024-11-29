from typing import Optional, List

from pydantic import BaseModel, Field

from open_chatchat._constants import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD
from open_chatchat.types.chat.chat_message import ChatMessage


class FileChatParam(BaseModel):
    """文件对话类"""

    query: str = Field(..., description="用户输入", examples=["你好"]),
    knowledge_id: str = Field(..., description="临时知识库ID"),
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
        examples=[
            [
                {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                {"role": "assistant", "content": "虎头虎脑"},
            ]
        ],
    ),
    stream: bool = Field(False, description="流式输出"),
    model_name: str = Field(None, description="LLM 模型名称。"),
    temperature: float = Field(0.01, description="LLM 采样温度", ge=0.0, le=1.0),
    max_tokens: Optional[int] = Field(
        None, description="限制LLM生成Token数量，默认None代表模型最大值"
    ),
    prompt_name: str = Field(
        "default",
        description="使用的prompt模板名称(在 prompt_settings.yaml 中配置)",
    ),
