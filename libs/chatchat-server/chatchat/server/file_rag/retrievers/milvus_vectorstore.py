import warnings

from langchain.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

from chatchat.server.file_rag.retrievers.base import BaseRetrieverService

from langchain.docstore.document import Document
from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun
)

from typing import List

class MilvusRetriever(VectorStoreRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = self.vectorstore.similarity_search_with_relevance_scores(query, **self.search_kwargs)
            score_threshold = self.search_kwargs.get("score_threshold", None)
            
            if score_threshold is not None:  # can be 0, but not None
                docs_and_similarities = [
                doc
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                warnings.warn(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )
            return docs_and_similarities
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = await self.vectorstore.asimilarity_search(
                query, **self.search_kwargs
            )
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(query, **self.search_kwargs)
            )
            score_threshold = self.search_kwargs.get("score_threshold", None)

            if score_threshold is not None:  # can be 0, but not None
                docs_and_similarities = [
                doc
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                warnings.warn(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )
            return docs_and_similarities
        elif self.search_type == "mmr":
            docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs 

class MilvusVectorstoreRetrieverService(BaseRetrieverService):
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
        score_threshold: int or float,
    ):
        retriever = MilvusRetriever(vectorstore=vectorstore, 
                                    search_type="similarity_score_threshold",
                                    search_kwargs={"score_threshold": score_threshold, "k": top_k}
                                    )
        
        return MilvusVectorstoreRetrieverService(retriever=retriever, top_k=top_k)

    def get_relevant_documents(self, query: str):
        return self.retriever.get_relevant_documents(query)[: self.top_k]
