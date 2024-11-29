import logging

from open_chatchat._constants import API_BASE_URI
from open_chatchat.api.chat.chat_client import ChatClient
from open_chatchat.api.knowledge_base.knowledge_base_client import KbClient
from open_chatchat.api.server.server_client import ServerClient
from open_chatchat.api.standard_openai.standard_openai_client import StandardOpenaiClient
from open_chatchat.api.tools.tool_client import ToolClient


class ChatChat:
    knowledge: KbClient = None
    tool: ToolClient = None
    server: ServerClient = None
    chat: ChatClient = None
    openai_adapter: StandardOpenaiClient = None

    def __init__(self,
                 base_url: str = API_BASE_URI,
                 timeout: float = 60,
                 use_async: bool = False,
                 use_proxy: bool = False,
                 proxies=None,
                 log_level: int = logging.INFO,
                 retry: int = 3,
                 retry_interval: int = 1, ):
        param = {
            'log_level': log_level,
            'retry': retry,
            'retry_interval': retry_interval,
            'base_url': base_url,
            'timeout': timeout,
            'use_async': use_async,
            'use_proxy': use_proxy,
            'proxies': proxies
        }

        self.knowledge = KbClient(**param)
        self.tool = ToolClient(**param)
        self.server = ServerClient(**param)
        self.chat = ChatClient(**param)
        self.openai_adapter = StandardOpenaiClient(**param)

