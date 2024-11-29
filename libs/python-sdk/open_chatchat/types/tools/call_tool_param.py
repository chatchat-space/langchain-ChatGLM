from pydantic import Field, BaseModel


class CallToolParam(BaseModel):
    name: str = Field(..., description="工具名称")
    tool_input: dict = Field({}, description="知识库信息"),
