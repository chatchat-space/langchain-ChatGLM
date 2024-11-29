from open_chatchat.api_client import ApiClient
from open_chatchat.types.tools.call_tool_param import CallToolParam

API_URI_TOOL_CALL = "/tools/call"
API_URI_TOOL_LIST = "/tools"


class ToolClient(ApiClient):
    def list(self) -> dict:
        """
        列出所有工具
        """
        resp = self._get(API_URI_TOOL_LIST)
        return self._get_response_value(resp, as_json=True, value_func=lambda r: r.get("data", {}))

    def call(
            self,
            name: str,
            tool_input: dict = {},
    ):
        """
        调用工具
        """
        data = CallToolParam(name=name, tool_input=tool_input).dict()
        resp = self._post(API_URI_TOOL_CALL, json=data)
        return self._get_response_value(resp, as_json=True, value_func=lambda r: r.get("data"))
