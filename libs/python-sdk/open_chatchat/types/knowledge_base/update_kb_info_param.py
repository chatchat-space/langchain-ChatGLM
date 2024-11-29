from pydantic import BaseModel, Field


class UpdateKbInfoParam(BaseModel):
    knowledge_base_name: str = Field(
        ..., description="知识库名称", examples=["samples"]
    ),
    kb_info: str = Field(..., description="知识库介绍", examples=["这是一个知识库"]),
