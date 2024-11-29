from typing import List

from pydantic import BaseModel, Field


class DownloadKbDocParam(BaseModel):
    knowledge_base_name: str = Field(
        ..., description="知识库名称", examples=["samples"]
    ),
    file_name: str = Field(..., description="文件名称", examples=["test.txt"]),
    preview: bool = Field(False, description="是：浏览器内预览；否：下载"),
