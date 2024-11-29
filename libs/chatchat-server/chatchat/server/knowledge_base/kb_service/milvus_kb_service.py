import os
from typing import Callable, Dict, List, Optional

from langchain.schema import Document
from langchain.vectorstores.milvus import Milvus

from chatchat.settings import Settings
from chatchat.server.db.repository import list_file_num_docs_id_by_kb_name_and_file_name
from chatchat.server.utils import get_Embeddings
from chatchat.server.file_rag.utils import get_Retriever
from chatchat.server.knowledge_base.kb_service.base import (
    KBService,
    SupportedVSType,
    score_threshold_process,
)
from chatchat.server.knowledge_base.utils import KnowledgeFile


class MilvusKBService(KBService):
    milvus: Milvus

    @staticmethod
    def get_collection(milvus_name):
        from pymilvus import Collection

        return Collection(milvus_name)

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        result = []
        if self.milvus.col:
            # ids = [int(id) for id in ids]  # for milvus if needed #pr 2725
            data_list = self.milvus.col.query(
                expr=f"pk in {[int(_id) for _id in ids]}", output_fields=["*"]
            )
            for data in data_list:
                text = data.pop("text")
                result.append(Document(page_content=text, metadata=data))
        return result

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        self.milvus.col.delete(expr=f"pk in {ids}")

    @staticmethod
    def search(milvus_name, content, limit=3):
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        c = MilvusKBService.get_collection(milvus_name)
        return c.search(
            content, "embeddings", search_params, limit=limit, output_fields=["content"]
        )

    def do_create_kb(self):
        pass

    def vs_type(self) -> str:
        return SupportedVSType.MILVUS

    def _load_milvus(self):
        self.milvus = Milvus(
            embedding_function=get_Embeddings(self.embed_model),
            collection_name=self.kb_name,
            connection_args=Settings.kb_settings.kbs_config.get("milvus"),
            index_params=Settings.kb_settings.kbs_config.get("milvus_kwargs")["index_params"],
            search_params=Settings.kb_settings.kbs_config.get("milvus_kwargs")["search_params"],
            auto_id=True,
            )

    def do_init(self):
        self._load_milvus()

    def do_drop_kb(self):
        if self.milvus.col:
            self.milvus.col.release()
            self.milvus.col.drop()

    def do_search(self, query: str, top_k: int, score_threshold: float):
        self._load_milvus()
        # embed_func = get_Embeddings(self.embed_model)
        # embeddings = embed_func.embed_query(query)
        self.milvus._select_relevance_score_fn = self._select_relevance_score_fn
        retriever = get_Retriever("milvusvectorstore").from_vectorstore(
            self.milvus,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        docs = retriever.get_relevant_documents(query)
        return docs

    def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
        for doc in docs:
            for k, v in doc.metadata.items():
                doc.metadata[k] = str(v)
            for field in self.milvus.fields:
                doc.metadata.setdefault(field, "")
            doc.metadata.pop(self.milvus._text_field, None)
            doc.metadata.pop(self.milvus._vector_field, None)

        ids = self.milvus.add_documents(docs)
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        return doc_infos

    def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
        id_list = list_file_num_docs_id_by_kb_name_and_file_name(
            kb_file.kb_name, kb_file.filename
        )
        if self.milvus.col:
            self.milvus.col.delete(expr=f"pk in {id_list}")

        # Issue 2846, for windows
        # if self.milvus.col:
        #     file_path = kb_file.filepath.replace("\\", "\\\\")
        #     file_name = os.path.basename(file_path)
        #     id_list = [item.get("pk") for item in
        #                self.milvus.col.query(expr=f'source == "{file_name}"', output_fields=["pk"])]
        #     self.milvus.col.delete(expr=f'pk in {id_list}')

    def do_clear_vs(self):
        if self.milvus.col:
            self.do_drop_kb()
            self.do_init()

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        def _map_l2_to_similarity(l2_distance: float) -> float:
            """Return a similarity score on a scale [0, 1].
            It is recommended that the original vector is normalized,
            Milvus only calculates the value before applying square root.
            l2_distance range: (0 is most similar, 4 most dissimilar)
            See
            https://milvus.io/docs/metric.md?tab=floating#Euclidean-distance-L2
            """
            return 1 - l2_distance / 4.0

        def _map_ip_to_similarity(ip_score: float) -> float:
            """Return a similarity score on a scale [0, 1].
            It is recommended that the original vector is normalized,
            ip_score range: (1 is most similar, -1 most dissimilar)
            See
            https://milvus.io/docs/metric.md?tab=floating#Inner-product-IP
            https://milvus.io/docs/metric.md?tab=floating#Cosine-Similarity
            """
            return (ip_score + 1) / 2.0
        
        metric_type = self.milvus.search_params.get("metric_type")
        if metric_type == "L2":
            return _map_l2_to_similarity
        elif metric_type in ["IP", "COSINE"]:
            return _map_ip_to_similarity
        else:
            raise ValueError(
                "No supported normalization function"
                f" for metric type: {metric_type}."
            )


if __name__ == "__main__":
    # 测试建表使用
    from chatchat.server.db.base import Base, engine

    Base.metadata.create_all(bind=engine)
    milvusService = MilvusKBService("test")
    # milvusService.add_doc(KnowledgeFile("README.md", "test"))

    print(milvusService.get_doc_by_ids(["444022434274215486"]))
    # milvusService.delete_doc(KnowledgeFile("README.md", "test"))
    # milvusService.do_drop_kb()
    # print(milvusService.search_docs("如何启动api服务"))
