from pydantic import BaseModel, Field
from open_chatchat._constants import CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE


class UploadKbDocsParam(BaseModel):
    knowledge_base_name: str = Field(
        ..., description="知识库名称", examples=["samples"]
    ),
    override: bool = Field(False, description="覆盖已有文件"),
    to_vector_store: bool = Field(True, description="上传文件后是否进行向量化"),
    chunk_size: int = Field(CHUNK_SIZE, description="知识库中单段文本最大长度"),
    chunk_overlap: int = Field(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
    zh_title_enhance: bool = Field(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
    docs: str = Field("", description="自定义的docs，需要转为json字符串"),
    not_refresh_vs_cache: bool = Field(False, description="暂不保存向量库（用于FAISS）"),
