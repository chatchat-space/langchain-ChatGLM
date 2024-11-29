from pydantic import BaseModel, Field

from open_chatchat._constants import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD


class SearchKbDocsParam(BaseModel):
    query: str = Field(description="检索内容")
    knowledge_base_name: str = Field(description="知识库名称")
    top_k: int = Field(default=VECTOR_SEARCH_TOP_K, description="匹配向量数")
    score_threshold: float = Field(default=SCORE_THRESHOLD,
                                   ge=0.0,
                                   le=1.0,
                                   description="知识库匹配相关度阈值，取值范围在0-1之间，"
                                               "SCORE越小，相关度越高，"
                                               "取到1相当于不筛选，建议设置在0.5左右")
    file_name: str = Field("", description="文件名称，支持 sql 通配符"),
    metadata: dict = Field({}, description="根据 metadata 进行过滤，仅支持一级键"),
