from __future__ import annotations

from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever

from chatchat.server.file_rag.retrievers.base import BaseRetrieverService
from chatchat.settings import Settings

class EnsembleRetrieverService(BaseRetrieverService):
    def do_init(
        self,
        retriever: BaseRetriever = None,
        top_k: int = 5,
    ):
        self.vs = None
        self.top_k = top_k
        self.retriever = retriever

    @staticmethod
    def from_vectorstore(
        vectorstore: VectorStore,
        top_k: int,
        score_threshold: int | float,
    ):
        faiss_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": top_k},
        )
        # TODO: 换个不用torch的实现方式
        # from cutword.cutword import Cutter
        import jieba

        # cutter = Cutter()
        docs = list(vectorstore.docstore._dict.values())
        bm25_retriever = BM25Retriever.from_documents(
            docs,
            preprocess_func=jieba.lcut_for_search,
        )
        bm25_retriever.k = top_k
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
        return EnsembleRetrieverService(retriever=ensemble_retriever, top_k=top_k)

    def get_relevant_documents(self, query: str):

        return self.retriever.get_relevant_documents(query)[: self.top_k]
