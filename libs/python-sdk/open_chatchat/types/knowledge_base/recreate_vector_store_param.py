
from pydantic import BaseModel, Field

from open_chatchat._constants import VS_TYPE, EMBEDDING_MODEL, CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE


class RecreateVectorStoreParam(BaseModel):
    knowledge_base_name: str = Field(..., examples=["samples"], description='知识库名称'),
    allow_empty_kb: bool = Field(True),
    vs_type: str = Field(VS_TYPE, description='向量库类型'),
    embed_model: str = Field(EMBEDDING_MODEL, description="向量模型"),
    chunk_size: int = Field(CHUNK_SIZE, description="知识库中单段文本最大长度"),
    chunk_overlap: int = Field(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
    zh_title_enhance: bool = Field(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
    not_refresh_vs_cache: bool = Field(False, description="暂不保存向量库（用于FAISS）")
