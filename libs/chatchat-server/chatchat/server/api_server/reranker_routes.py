"""用于加载reranker模型，封装成api服务
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time

from fastapi import APIRouter, HTTPException, Response
from typing import List,  Optional, Union

from pydantic import BaseModel, Field

from sse_starlette.sse import EventSourceResponse
from FlagEmbedding import FlagReranker
# Set up limit request time
EventSourceResponse.DEFAULT_PING_INTERVAL = 1000
from chatchat.settings import Settings
from chatchat.utils import build_logger

logger = build_logger()
reranker_router = APIRouter(prefix="/reranker", tags=["reranker模型接口"])

reranker_model = FlagReranker(Settings.model_settings.RERANKER_CONFIG['local_path'], 
                              use_fp16=True,
                              device=Settings.model_settings.RERANKER_CONFIG['device']
                              )


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

class RerankerRequest(BaseModel):
    input: Union[List[str], List[List[str]]]
    model: str = Settings.model_settings.RERANKER_CONFIG['model']

class RerankerResponse(BaseModel):
    data: list
    model: str
    object: str
    # usage: dict


@reranker_router.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@reranker_router.post("/rerank_passage", response_model=RerankerResponse)
def rerank_answers(request: RerankerRequest):
    scores = reranker_model.compute_score(request.input,batch_size=32)
    if isinstance(scores, float):
        scores = [scores]
    response = {
        "data": [
            {
                "object": "reranker",
                "score": score,
                "index": index
            }
            for index, score in enumerate(scores)
        ],
        "model": request.model,
        "object": "list",
    }
    return response


@reranker_router.get("/models", response_model=ModelList)
async def list_models():
    model_card_rerank = ModelCard(
        id=Settings.model_settings.RERANKER_CONFIG['model']
    )
    return ModelList(
        data=[
            model_card_rerank
            ]
    )
