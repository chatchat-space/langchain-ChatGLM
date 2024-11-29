import os
import shutil
from typing import Dict, List, Optional

from langchain.schema import Document
from sqlalchemy_vectorstores import SqliteDatabase, SqliteVectorStore
from sqlalchemy_vectorstores.tokenizers.jieba_tokenize import JiebaTokenize

from chatchat.settings import Settings
from chatchat.server.knowledge_base.kb_service.base import (
    KBService,
    SupportedVSType,
    score_threshold_process,
)
from chatchat.server.knowledge_base.utils import KnowledgeFile
from chatchat.server.utils import get_Embeddings


def _create_database():
    db_uri = Settings.kb_settings.kbs_config.get("sqlite", {}).get("uri", "knowledge_base.db")
    if not db_uri.startswith("sqlite") and not os.path.isabs(db_uri):
        db_uri = os.path.join(Settings.basic_settings.KB_ROOT_PATH, db_uri)
    if not db_uri.startswith("sqlite") and os.path.isabs(db_uri):
        db_uri = f"sqlite:///{db_uri}"
    return SqliteDatabase(db_uri, fts_tokenizers={"jieba": JiebaTokenize()}, echo=False)


_sqlite_db = _create_database()


class SqliteKBService(KBService):
    def do_init(self):
        def embed_func(text: str) -> List[float]:
            embeddings = get_Embeddings(self.embed_model)
            return embeddings.embed_documents([text])[0]
        self.sqlite_vs = SqliteVectorStore(
            _sqlite_db,
            table_prefix=self.kb_name,
            embedding_func=embed_func,
            fts_tokenize="jieba",
        )

    def do_create_kb(self):
        ...

    def _dict2document(self, data: Dict | List[dict]) -> Document | List[Document]:
        if isinstance(data, list):
            return [Document(page_content=x["content"], id=x["id"], metadata=x["metadata"]) for x in data]
        else:
            return Document(page_content=data["content"], id=data["id"], metadata=data["metadata"])

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        docs = self.sqlite_vs.get_document_by_ids(ids)
        return self._dict2document(docs)

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        self.sqlite_vs.delete_documents(ids)
        return super().del_doc_by_ids(ids)

    def vs_type(self) -> str:
        return SupportedVSType.SQLITE

    def do_drop_kb(self):
        self.sqlite_vs.drop_all_tables()
        shutil.rmtree(self.kb_path)

    def do_search(self, query: str, top_k: int=3, score_threshold: float=None) -> List[Document]:
        docs1 = self.sqlite_vs.search_by_vector(query=query,
                                               top_k=top_k,
                                               score_threshold=score_threshold)
        import jieba
        query = [x.lower() for x in jieba.lcut_for_search(query)]
        query = " OR ".join(query)
        docs2 = self.sqlite_vs.search_by_bm25(query=query,
                                               top_k=top_k,
                                               score_threshold=score_threshold)
        docs = []
        # deduplicate
        for d in docs1+docs2:
            if d["id"] not in [x["id"] for x in docs]:
                docs.append(d)
        return self._dict2document(docs)

    def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
        ids = []
        for d in docs:
            ids.append(self.sqlite_vs.add_document(src_id=kwargs.get("src_id"),
                                                   content=d.page_content,
                                                   metadata=d.metadata))
        doc_infos = [{"id": id, "metadata": doc.metadata}
                     for id, doc in zip(ids, docs)]
        return doc_infos

    def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
        path = self.get_relative_source_path(kb_file.filepath)
        docs = self.sqlite_vs.get_documents_by_meta({"source": path})
        self.sqlite_vs.delete_documents([x["id"] for x in docs])

    def do_clear_vs(self):
        self.sqlite_vs.init_database(clear_existed=True)
