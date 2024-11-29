from pydantic import Field, BaseModel


class ChatFeedbackParam(BaseModel):
    message_id: str = Field("", max_length=32, description="聊天记录id"),
    score: int = Field(0, max=100, description="用户评分，满分100，越大表示评价越高"),
    reason: str = Field("", description="用户评分理由，比如不符合事实等"),
