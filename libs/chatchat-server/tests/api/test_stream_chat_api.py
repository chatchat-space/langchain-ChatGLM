import json
import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).parent.parent.parent))
from pprint import pprint

from chatchat.server.utils import api_address, get_default_llm


api_base_url = api_address()
api="/chat/chat/completions"
url = f"{api_base_url}{api}"
llm_model = get_default_llm()


def dump_input(d, title):
    print("\n")
    print("=" * 30 + title + "  input " + "=" * 30)
    pprint(d)


def dump_output(r, title):
    print("\n")
    print("=" * 30 + title + "  output" + "=" * 30)
    for line in r.iter_content(None, decode_unicode=True):
        print(line, end="", flush=True)


headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}


def test_llm_chat():
    data = {
        "model": llm_model,
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我是人工智能大模型"},
            {"role": "user", "content": "请用100字左右的文字介绍自己"},
        ],
        "stream": True,
        "temperature": 0.7,
    }

    dump_input(data, "LLM Chat")
    response = requests.post(url, headers=headers, json=data, stream=True)
    dump_output(response, "LLM Chat")
    assert response.status_code == 200


def test_agent_chat():
    tools = list(requests.get(f"{api_base_url}/tools").json()["data"])
    data = {
        "model": llm_model,
        "messages": [
            {"role": "user", "content": "37+48=？"},
        ],
        "stream": True,
        "temperature": 0.7,
        "tools": tools,
    }

    dump_input(data, "Agent Chat")
    response = requests.post(url, headers=headers, json=data, stream=True)
    dump_output(response, "Agent Chat")
    assert response.status_code == 200


def test_kb_chat_auto():
    data = {
        "messages": [
            {"role": "user", "content": "如何提问以获得高质量答案"},
        ],
        "model": llm_model,
        "tool_choice": "search_local_knowledgebase",
        "stream": True,
    }
    dump_input(data, "KB Chat (auto parameters)")
    response = requests.post(url, headers=headers, json=data, stream=True)
    print("\n")
    print("=" * 30 + "KB Chat (auto parameters)" + "  output" + "=" * 30)
    for line in response.iter_content(None, decode_unicode=True):
        if line.startswith("data: "):
            data = json.loads(line[6:])
            pprint(data)
    assert response.status_code == 200


def test_kb_chat_mannual():
    data = {
        "messages": [
            {"role": "user", "content": "如何提问以获得高质量答案"},
        ],
        "model": llm_model,
        "tool_choice": "search_local_knowledgebase",
        "extra_body": {"tool_input": {"database": "samples", "query": "如何提问以获得高质量答案"}},
        "stream": True,
    }
    dump_input(data, "KB Chat (mannual parameters)")
    response = requests.post(url, headers=headers, json=data, stream=True)
    print("\n")
    print("=" * 30 + "KB Chat (mannual parameters)" + "  output" + "=" * 30)
    for line in response.iter_content(None, decode_unicode=True):
        if line.startswith("data: "):
            data = json.loads(line[6:])
            pprint(data)
    assert response.status_code == 200


def test_search_engine_chat():
    url = f"{api_base_url}/knowledge_base/search_engine/searx/chat/completions"
    data = {
        "messages": [
            {"role": "user", "content": "介绍一下chatchat"},
        ],
        "model": llm_model,
        "stream": True,
    }
    dump_input(data, "Search Engine Chat")
    response = requests.post(url, headers=headers, json=data, stream=True)
    print("\n")
    print("=" * 30 + "Search Engine Chat" + "  output" + "=" * 30)
    for line in response.iter_content(None, decode_unicode=True):
        if line.startswith("data: "):
            data = json.loads(line[6:])
            pprint(data)
    assert response.status_code == 200
