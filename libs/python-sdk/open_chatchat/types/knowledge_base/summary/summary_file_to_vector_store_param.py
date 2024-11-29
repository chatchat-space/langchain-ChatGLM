from typing import Optional

from pydantic import Field, BaseModel

from open_chatchat._constants import VS_TYPE, EMBEDDING_MODEL


class SummaryFileToVectorStoreParam(BaseModel):
    knowledge_base_name: str = Field(..., examples=["samples"]),
    file_name: str = Field(..., examples=["test.pdf"]),
    allow_empty_kb: bool = Field(True),
    vs_type: str = Field(VS_TYPE),
    embed_model: str = Field(EMBEDDING_MODEL),
    file_description: str = Field(""),
    model_name: str = Field(None, description="LLM 模型名称。"),
    temperature: float = Field(0.01, description="LLM 采样温度", ge=0.0, le=1.0),
    max_tokens: Optional[int] = Field(
        None, description="限制LLM生成Token数量，默认None代表模型最大值"
    ),
