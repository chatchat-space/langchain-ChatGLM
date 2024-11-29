from pydantic import BaseModel, Field

from open_chatchat._constants import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD


class SearchTempDocsParam(BaseModel):
    knowledge_id: str
    query: str
    top_k: int = Field(default=VECTOR_SEARCH_TOP_K, description="匹配向量数")
    score_threshold: float = Field(default=SCORE_THRESHOLD,
                                   ge=0.0,
                                   le=1.0,
                                   description="知识库匹配相关度阈值，取值范围在0-1之间，"
                                               "SCORE越小，相关度越高，"
                                               "取到1相当于不筛选，建议设置在0.5左右")
