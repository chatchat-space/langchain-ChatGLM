from functools import wraps
from typing import Type, get_type_hints

import httpx
import requests
from pydantic import BaseModel

from open_chatchat.api_client import ApiClient
from open_chatchat.types.knowledge_base.delete_knowledge_base_param import DeleteKnowledgeBaseParam
from open_chatchat.types.response.base import ListResponse

base_url = "https://api.example.com"
headers = {"Authorization": "Bearer token"}


def http_request(method):
    def decorator(url, base_url='', headers=None, body_model: Type[BaseModel] = None, **options):
        headers = headers or {}

        def wrapper(func):
            @wraps(func)
            def inner(*args, **kwargs):
                try:

                    print("args", args)
                    print("kwargs", kwargs)
                    # Prepare the request URL
                    full_url = base_url + url

                    # Prepare the request data
                    data = kwargs
                    return_type = get_type_hints(func).get('return')
                    print(f"Return type: {return_type}")
                    print(body_model)
                    print(f"body_model: {body_model}")
                    # Send the HTTP request
                    response = method(full_url, headers=headers, json=data)
                    response.raise_for_status()

                    # Return the response JSON
                    return response.json()
                except requests.exceptions.HTTPError as http_err:
                    print(f"HTTP error occurred: {http_err}")
                except Exception as err:
                    print(f"An error occurred: {err}")

            return inner

        return wrapper

    return decorator


# Usage example
post = http_request(httpx.post)


class MyAPIClient(ApiClient):

    @post(url='/api/kb/recreate_summary_vector_store', base_url=base_url, headers=headers,
          body_model=DeleteKnowledgeBaseParam)
    def recreate_summary_vector_store(
            self,
            a: int,
            b: int
    ) -> ListResponse:
        pass


@post(url='/api/kb/recreate_summary_vector_store', base_url=base_url, headers=headers,
      body_model=DeleteKnowledgeBaseParam)
def recreate_summary_vector_store(
        a: int,
        b: int
) -> ListResponse:
    pass


# Example usage
if __name__ == "__main__":
    # Call the decorated function
    # response = recreate_summary_vector_store(a=1, b=1)
    # print(response)
    api_client = MyAPIClient()
    response = api_client.recreate_summary_vector_store(a=1, b=2)
    print("response", response)
