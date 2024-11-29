from typing import Union, List

from pydantic import BaseModel, Field
from open_chatchat._constants import CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE


class UploadTempDocsParam(BaseModel):
    prev_id: str = Field(None, description="前知识库ID"),
    chunk_size: int = Field(CHUNK_SIZE, description="知识库中单段文本最大长度"),
    chunk_overlap: int = Field(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
    zh_title_enhance: bool = Field(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),