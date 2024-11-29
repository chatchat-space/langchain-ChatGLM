import json
import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).parent.parent.parent))
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint

from chatchat.server.utils import api_address, get_default_llm

api_base_url = api_address()
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

def llm_chat():
    url = f"{api_base_url}/chat/chat/completions"
    data = {
        "model": llm_model,
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我是人工智能大模型"},
            {"role": "user", "content": "请用100字左右的文字介绍自己"},
        ],
        "stream": False,
        "temperature": 0.7,
    }

    dump_input(data, "LLM Chat")
    response = requests.post(url, headers=headers, json=data, stream=True)
    dump_output(response, "LLM Chat")
    assert response.status_code == 200


def kb_chat_mannual():
    url = f"{api_base_url}/chat/chat/completions"
    data = {
        "messages": [
            {"role": "user", "content": "如何提问以获得高质量答案"},
        ],
        "model": llm_model,
        "tool_choice": "search_local_knowledgebase",
        "extra_body": {"tool_input": {"database": "samples", "query": "如何提问以获得高质量答案"}},
        "stream": False,
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


def test_thread():
    for func in [llm_chat, kb_chat_mannual]:
        threads = []
        times = []
        pool = ThreadPoolExecutor()
        start = time.time()
        for i in range(10):
            t = pool.submit(func)
            threads.append(t)

        for r in as_completed(threads):
            end = time.time()
            times.append(end - start)
            print("\nResult:\n")
            pprint(r.result())

        print("\nTime used:\n")
        for x in times:
            print(f"{x}")
