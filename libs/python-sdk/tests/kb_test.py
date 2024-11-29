import logging

from open_chatchat.chatchat_api import ChatChat

chatchat = ChatChat()


def test_create_kb():
    print('create_kb', chatchat.knowledge.create_kb(knowledge_base_name="example_kb"))


def test_update_kb_info():
    print('update_kb_info', chatchat.knowledge.update_kb_info(knowledge_base_name="example_kb", kb_info='aaaaaaa'))


def test_list_kb():
    print('list_kb', chatchat.knowledge.list_kb())


def test_list_kb_docs_file():
    print('list_kb_docs_file', chatchat.knowledge.list_kb_docs_file(knowledge_base_name="samples"))


def test_delete_kb():
    print('delete_kb', chatchat.knowledge.delete_kb(knowledge_base_name="example_kb"))


def test_search_kb_docs():
    print('search_kb_docs', chatchat.knowledge.search_kb_docs(knowledge_base_name="example_kb", query="hello"))


def test_upload_kb_docs():
    print('upload_kb_docs', chatchat.knowledge.upload_kb_docs(
        files=["data/upload_file1.txt", "data/upload_file2.txt"],
        knowledge_base_name="example_kb",
    ))


def test_search_kb_docs_after_upload():
    print('search_kb_docs', chatchat.knowledge.search_kb_docs(knowledge_base_name="example_kb", query="hello"))


def test_recreate_vector_store():
    print('recreate_vector_store', chatchat.knowledge.recreate_vector_store(
        knowledge_base_name="samples",
    ))


def test_recreate_summary_vector_store():
    print('recreate_summary_vector_store', chatchat.knowledge.recreate_summary_vector_store(
        knowledge_base_name="example_kb",
        embed_model="embedding-2",
        model_name="glm-4",
    ))


def test_summary_file_to_vector_store():
    for data in chatchat.knowledge.summary_file_to_vector_store(
            knowledge_base_name="samples",
            file_name="data/upload_file1.txt",
            embed_model="embedding-2",
            max_tokens=10000):
        print(data)


def test_summary_doc_ids_to_vector_store():
    print('summary_file_to_vector_store', chatchat.knowledge.summary_doc_ids_to_vector_store(
        knowledge_base_name="samples",
        doc_ids=['22'],
    ))


def test_delete_kb_docs():
    print('delete_kb_docs', chatchat.knowledge.delete_kb_docs(
        knowledge_base_name="samples",
        file_names=["upload_file1.txt"],
    ))


def test_download_kb_doc_file():
    print(chatchat.knowledge.download_kb_doc_file(
        knowledge_base_name='example_kb',
        file_name='README.md'
    ))


def test_kb_doc_file_content():
    print(chatchat.knowledge.kb_doc_file_content(
        knowledge_base_name='example_kb',
        file_name='README.md'
    ))


def test_upload_temp_docs():
    print(chatchat.knowledge.upload_temp_docs(
        files=["README.md"],
        knowledge_id="4",
    ))


def test_search_temp_kb_docs():
    print(chatchat.knowledge.search_temp_kb_docs(
        knowledge_id="cf414f74bca24fbdaece1ae8bb4d3970",
        query="hello"
    ))
