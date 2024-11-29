from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(...)
    content: str = Field(...)
