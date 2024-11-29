from open_chatchat.chatchat_api import ChatChat

chatchat = ChatChat()
kb_name = "example_kb"
default_model = "glm-4"


def test_kb_chat():
    for data in chatchat.chat.kb_chat(query='你好', kb_name=kb_name, model=default_model):
        print(data)


def test_file_chat():
    for data in chatchat.chat.file_chat(query='你好', knowledge_id='a9bb673176cd4e34a827c63fd72945f2', model_name=default_model):
        print(data)


def test_chat_feedback():
    print(chatchat.chat.chat_feedback(message_id='a9bb673176cd4e34a827c63fd72945f2'))
