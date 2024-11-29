

def test_search_docs():
    from chatchat.server.knowledge_base.kb_doc_api import search_docs
    from pytest_check import check
    docs = search_docs(
        query="你好",
        knowledge_base_name="samples",
        top_k=3,
        score_threshold=0.5
    )
    print(docs)
    with check:
        assert len(docs) == 3

