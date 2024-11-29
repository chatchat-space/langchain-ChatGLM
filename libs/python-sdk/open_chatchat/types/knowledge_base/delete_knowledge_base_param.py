from pydantic import Field, BaseModel


class DeleteKnowledgeBaseParam(BaseModel):
    knowledge_base_name: str = Field(..., description="知识库名称")
