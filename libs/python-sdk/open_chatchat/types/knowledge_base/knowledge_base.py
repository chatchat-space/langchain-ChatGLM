from typing import Optional

from pydantic import BaseModel, Field
from datetime import datetime


class KnowledgeBaseInfo(BaseModel):
    id: int = Field(default=None,  description="知识库id")
    kb_name: str = Field(default=None,  description="知识库名称")
    kb_info: Optional[str] = Field(default=None,  description="知识库信息")
    vs_type: Optional[str] = Field(default=None,  description="向量库类型")
    embed_model: Optional[str] = Field(default=None,  description="向量化模型")
    file_count: Optional[int] = Field(default=None,  description="文件数量")
    create_time: Optional[datetime] = Field(default=None,  description="创建时间")