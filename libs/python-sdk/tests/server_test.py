from open_chatchat.chatchat_api import ChatChat

chatchat = ChatChat()


def test_get_server_configs():
    print(chatchat.server.get_server_configs())


def get_prompt_template():
    print(chatchat.server.get_prompt_template())
