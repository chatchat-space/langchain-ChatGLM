from open_chatchat.api_client import ApiClient

API_URI_GET_SERVER_CONFIGS = "/server/configs"
API_URI_GET_PROMPT_TEMPLATE = "/server/get_prompt_template"


class ServerClient(ApiClient):
    # 服务器信息
    def get_server_configs(self) -> dict:
        response = self._post(API_URI_GET_SERVER_CONFIGS)
        return self._get_response_value(response, as_json=True)

    def get_prompt_template(
            self,
            _type: str = "knowledge_base_chat",
            name: str = "default",
    ) -> str:
        data = {
            "type": _type,  # 模板类型
            "name": name  # 模板名称
        }
        response = self._post(API_URI_GET_PROMPT_TEMPLATE, json=data)
        return self._get_response_value(response, value_func=lambda r: r.text)
