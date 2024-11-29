from typing import Optional

from pydantic import Field, BaseModel


class CreateKnowledgeBaseParam(BaseModel):
    knowledge_base_name: str = Field(default=None, description="知识库名称")
    vector_store_type: str = Field(default=None, description="向量存储类型")
    kb_info: Optional[str] = Field(default=None, description="知识库信息")
    vs_type: Optional[str] = Field(default=None, description="向量库类型")
    embed_model: Optional[str] = Field(default=None, description="向量化模型")
