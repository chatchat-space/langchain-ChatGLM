from open_chatchat.chatchat_api import ChatChat

chatchat = ChatChat()


def test_tool():
    print(chatchat.tool.list())


def test_call():
    print(chatchat.tool.call('calculate', {"text": "3+5/2"}))
