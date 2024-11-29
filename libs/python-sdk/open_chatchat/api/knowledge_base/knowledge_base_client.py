from pydantic import Field

from open_chatchat._constants import EMBEDDING_MODEL, VS_TYPE, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, CHUNK_SIZE, \
    OVERLAP_SIZE, ZH_TITLE_ENHANCE, LLM_MODEL
from open_chatchat.api_client import ApiClient, post
from open_chatchat.types.knowledge_base.create_knowledge_base_param import CreateKnowledgeBaseParam
import json
import os
from io import BytesIO
from pathlib import Path
from typing import *

from open_chatchat.types.knowledge_base.delete_knowledge_base_param import DeleteKnowledgeBaseParam
from open_chatchat.types.knowledge_base.doc.delete_kb_docs_param import DeleteKbDocsParam
from open_chatchat.types.knowledge_base.doc.download_kb_doc_param import DownloadKbDocParam
from open_chatchat.types.knowledge_base.doc.search_kb_docs_param import SearchKbDocsParam
from open_chatchat.types.knowledge_base.doc.search_temp_docs_param import SearchTempDocsParam
from open_chatchat.types.knowledge_base.doc.upload_kb_docs_param import UploadKbDocsParam
from open_chatchat.types.knowledge_base.doc.upload_temp_docs_param import UploadTempDocsParam
from open_chatchat.types.knowledge_base.recreate_vector_store_param import RecreateVectorStoreParam
from open_chatchat.types.knowledge_base.summary.recreate_summary_vector_store_param import \
    RecreateSummaryVectorStoreParam
from open_chatchat.types.knowledge_base.summary.summary_doc_ids_to_vector_store_param import \
    SummaryDocIdsToVectorStoreParam
from open_chatchat.types.knowledge_base.summary.summary_file_to_vector_store_param import SummaryFileToVectorStoreParam
from open_chatchat.types.knowledge_base.update_kb_info_param import UpdateKbInfoParam
from open_chatchat.types.response.base import BaseResponse
from open_chatchat.utils import convert_file

API_URI_CREATE_KB = "/knowledge_base/create_knowledge_base"
API_URI_DELETE_KB = "/knowledge_base/delete_knowledge_base"
API_URI_KB_UPDATE_INFO = "/knowledge_base/update_info"
API_URI_LIST_KB = "/knowledge_base/list_knowledge_bases"

API_URI_URI_LIST_KB_FILE = "/knowledge_base/list_files"
API_URI_SEARCH_KB_DOCS = "/knowledge_base/search_docs"

API_URI_KB_UPLOAD_DOCS = "/knowledge_base/upload_docs"
API_URI_KB_DOWNLOAD_DOC = "/knowledge_base/download_doc"
API_URI_DELETE_KB_DOCS = "/knowledge_base/delete_docs"
API_URI_KB_RECREATE_VECTOR_STORE = "/knowledge_base/recreate_vector_store"
API_URI_KB_SEARCH_TEMP_DOCS = "/knowledge_base/search_temp_docs"
API_URI_KB_UPLOAD_TEMP_DOCS = "/knowledge_base/upload_temp_docs"

API_URI_KB_SUMMARY_FILE_TO_VECTOR_STORE = "/knowledge_base/kb_summary_api/summary_file_to_vector_store"
API_URI_KB_SUMMARY_DOC_IDS_TO_VECTOR_STORE = "/knowledge_base/kb_summary_api/summary_doc_ids_to_vector_store"
API_URI_KB_SUMMARY_RECREATE_VECTOR_STORE = "/knowledge_base/kb_summary_api/recreate_summary_vector_store"


class KbClient(ApiClient):

    @post(url=API_URI_CREATE_KB
        , body_model=CreateKnowledgeBaseParam)
    def create_kb(
            self,
            knowledge_base_name: str,
            kb_info: str = "",
            vector_store_type: str = VS_TYPE,
            embed_model: str = EMBEDDING_MODEL,
    ) -> BaseResponse:
        ...

    # def create_knowledge_base(
    #         self,
    #         knowledge_base_name: str,
    #         kb_info: str = "",
    #         vector_store_type: str = VS_TYPE,
    #         embed_model: str = EMBEDDING_MODEL,
    # ):
    #     data = CreateKnowledgeBaseParam(
    #         knowledge_base_name=knowledge_base_name,
    #         kb_info=kb_info,
    #         vector_store_type=vector_store_type,
    #         embed_model=embed_model,
    #     ).dict()
    #     response = self.post(API_URI_CREATE_KB, json=data)
    #     return self._get_response_value(response, as_json=True)

    def delete_kb(
            self,
            knowledge_base_name: str,
    ):
        response = self._post(API_URI_DELETE_KB, json=knowledge_base_name)
        return self._get_response_value(response, as_json=True)

    def list_kb(self):
        response = self._get(API_URI_LIST_KB)
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", []))

    def list_kb_docs_file(
            self,
            knowledge_base_name: str,
    ):
        params = DeleteKnowledgeBaseParam(knowledge_base_name=knowledge_base_name).dict()
        response = self._get(API_URI_URI_LIST_KB_FILE, params=params)
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", []))

    def search_kb_docs(
            self,
            knowledge_base_name: str,
            query: str = "",
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: float = SCORE_THRESHOLD,
            file_name: str = "",
            metadata: dict = {},
    ) -> List:
        data = SearchKbDocsParam(
            query=query,
            knowledge_base_name=knowledge_base_name,
            top_k=top_k,
            score_threshold=score_threshold,
            file_name=file_name,
            metadata=metadata,
        ).dict()
        response = self._post(API_URI_SEARCH_KB_DOCS, json=data)
        return self._get_response_value(response, as_json=True)

    def upload_kb_docs(
            self,
            files: List[Union[str, Path, bytes]],
            knowledge_base_name: str,
            override: bool = False,
            to_vector_store: bool = True,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
            docs: Dict = {},
            not_refresh_vs_cache: bool = False,
    ):
        files = [convert_file(file) for file in files]
        data = UploadKbDocsParam(
            knowledge_base_name=knowledge_base_name,
            override=override,
            to_vector_store=to_vector_store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            docs=json.dumps(docs, ensure_ascii=False),
            not_refresh_vs_cache=not_refresh_vs_cache,
        ).dict()
        response = self._post(API_URI_KB_UPLOAD_DOCS, data=data,
                              files=[("files", (filename, file)) for filename, file in files])
        return self._get_response_value(response, as_json=True)

    def delete_kb_docs(
            self,
            knowledge_base_name: str,
            file_names: List[str],
            delete_content: bool = False,
            not_refresh_vs_cache: bool = False,
    ):
        data = DeleteKbDocsParam(
            knowledge_base_name=knowledge_base_name,
            file_names=file_names,
            delete_content=delete_content,
            not_refresh_vs_cache=not_refresh_vs_cache,
        ).dict()
        response = self._post(API_URI_DELETE_KB_DOCS, json=data)
        return self._get_response_value(response, as_json=True)

    def update_kb_info(self, knowledge_base_name, kb_info):
        data = UpdateKbInfoParam(
            knowledge_base_name=knowledge_base_name,
            kb_info=kb_info,
        ).dict()
        response = self._post(API_URI_KB_UPDATE_INFO, json=data)
        return self._get_response_value(response, as_json=True)

    def recreate_vector_store(
            self,
            knowledge_base_name: str,
            allow_empty_kb: bool = True,
            vs_type: str = VS_TYPE,
            embed_model: str = EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
    ):
        data = RecreateVectorStoreParam(
            knowledge_base_name=knowledge_base_name,
            allow_empty_kb=allow_empty_kb,
            vs_type=vs_type,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
        ).dict()
        response = self._post(API_URI_KB_RECREATE_VECTOR_STORE, json=data, stream=True, timeout=None)
        return self._httpx_stream2generator(response, as_json=True)

    def recreate_summary_vector_store(self,
                                      knowledge_base_name: str,
                                      allow_empty_kb: bool = True,
                                      vs_type: str = VS_TYPE,
                                      embed_model: str = EMBEDDING_MODEL,
                                      file_description: str = "",
                                      model_name: str = None,
                                      temperature: float = 0.01,
                                      max_tokens: Optional[int] = None):
        data = RecreateSummaryVectorStoreParam(
            knowledge_base_name=knowledge_base_name,
            allow_empty_kb=allow_empty_kb,
            vs_type=vs_type,
            embed_model=embed_model,
            file_description=file_description,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens).dict()
        response = self._post(API_URI_KB_SUMMARY_RECREATE_VECTOR_STORE, json=data)
        return self._get_response_value(response, as_json=True)

    def summary_doc_ids_to_vector_store(self,
                                        knowledge_base_name: str,
                                        doc_ids: List = [],
                                        vs_type: str = VS_TYPE,
                                        embed_model: str = EMBEDDING_MODEL,
                                        file_description: str = "",
                                        model_name: str = None,
                                        temperature: float = 0.01,
                                        max_tokens: Optional[int] = None,
                                        ):
        data = SummaryDocIdsToVectorStoreParam(
            knowledge_base_name=knowledge_base_name,
            doc_ids=doc_ids,
            vs_type=vs_type,
            embed_model=embed_model,
            file_description=file_description,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        ).dict()
        response = self._post(API_URI_KB_SUMMARY_DOC_IDS_TO_VECTOR_STORE, json=data)
        return self._get_response_value(response, as_json=True)

    def summary_file_to_vector_store(self, knowledge_base_name: str,
                                     file_name: str,
                                     allow_empty_kb: bool = True,
                                     vs_type: str = VS_TYPE,
                                     embed_model: str = EMBEDDING_MODEL,
                                     file_description: str = "",
                                     model_name: str = LLM_MODEL,
                                     temperature: float = 0.01,
                                     max_tokens: Optional[int] = 1000):
        data = SummaryFileToVectorStoreParam(
            knowledge_base_name=knowledge_base_name,
            file_name=file_name,
            allow_empty_kb=allow_empty_kb,
            vs_type=vs_type,
            embed_model=embed_model,
            file_description=file_description,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        ).dict()
        response = self._post(API_URI_KB_SUMMARY_FILE_TO_VECTOR_STORE, json=data,stream=True)
        return self._httpx_stream2generator(response, as_json=True)

    def upload_temp_docs(self,
                         files: List[Union[str, Path, bytes]],
                         knowledge_id: str = None,
                         chunk_size: int = CHUNK_SIZE,
                         chunk_overlap: int = OVERLAP_SIZE,
                         zh_title_enhance: bool = ZH_TITLE_ENHANCE,
                         ):
        data = UploadTempDocsParam(
            prev_id=knowledge_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance
        ).dict()
        _files = [convert_file(file) for file in files]
        response = self._post(
            "/knowledge_base/upload_temp_docs",
            data=data,
            files=[("files", (filename, file)) for filename, file in _files],
        )
        return self._get_response_value(response, as_json=True)
        # _files = [convert_file(file) for file in files]
        # response = self._post(API_URI_KB_UPLOAD_TEMP_DOCS, data=data,
        #                       files=[("files", (filename, file)) for filename, file in _files])
        # return self._get_response_value(response, as_json=True)

    def search_temp_kb_docs(
            self,
            knowledge_id: str,
            query: str,
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: float = SCORE_THRESHOLD,
    ) -> List:
        data = SearchTempDocsParam(
            knowledge_id=knowledge_id,
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
        ).dict()
        response = self._post(API_URI_KB_SEARCH_TEMP_DOCS, json=data)
        return self._get_response_value(response, as_json=True)

    def download_kb_doc_file(self, knowledge_base_name: str, file_name: str, file_path: Optional[str] = None):
        params = DownloadKbDocParam(
            knowledge_base_name=knowledge_base_name,
            file_name=file_name,
            preview=False
        ).dict()
        response = self._get(API_URI_KB_DOWNLOAD_DOC, params=params)
        file_content = self._get_response_value(response, as_json=False, value_func=lambda r: r.content)
        if file_path is None:
            file_path = file_name
        with open(file_path, 'wb') as file:
            file.write(file_content)
        return file_path

    def kb_doc_file_content(self, knowledge_base_name: str, file_name: str):
        params = DownloadKbDocParam(
            knowledge_base_name=knowledge_base_name,
            file_name=file_name,
            preview=True
        ).dict()
        response = self._get(API_URI_KB_DOWNLOAD_DOC, params=params)
        file_content = self._get_response_value(response, as_json=False, value_func=lambda r: r.content)
        return file_content.decode('utf-8')
