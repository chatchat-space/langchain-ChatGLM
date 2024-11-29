"""
用于对kb_chat.py进行批量测试
测试用例从文件读取，每个测试用例包含一个query，文件名koca问题集.xlsx
方式是对每个query进行检索，检索结果是一个list，长度为Settings.kb_settings.VECTOR_SEARCH_TOP_K

"""
import pandas as pd
import requests
from pprint import pprint

def test_kb_chat(
        url: str="http://localhost:7861/",
        api: str="chat/kb_chat",
        file_name: str="/home/huangzg/codes/chat_instance/koca问题集.xlsx",
        top_k: int=10,
        score_threshold: float=2.0,
        model: str="K-GPT",
        temperature: float=0.1,
        max_tokens: int=20000,
        prompt_name: str="default",
        return_direct: bool=False,
        kb_name: str="koca",
        history_num: int=3,
    ):
    uri = url + api
    print("\n检索知识库：")
    print(file_name)
    df = pd.read_excel(file_name)[:20]
    json_template = {
    "query": "",
    "mode": "local_kb",
    "kb_name": kb_name,
    "top_k": top_k,
    "score_threshold": score_threshold,
    "history": [  ],
    "stream": False,
    "model": model,
    "temperature": temperature,
    "max_tokens": max_tokens,
    "prompt_name": prompt_name,
    "return_direct": return_direct,
    }
    answer = []
    sources = []
    sources_ori = []
    num_sources = []
    for i in range(len(df)):
        json_template["query"] = df["query"][i]
        print(json_template["query"])
        r = requests.post(uri, json=json_template)
        if r.status_code == 200:
            # 将其转换为字典
            if isinstance(r.json(), str):
                data = eval(r.json().replace("null",'None').replace("false","False"))
            elif isinstance(r.json(), dict):
                data = r.json()
            pprint(data)
            answer.append(data["choices"][0]["message"]["content"])
            sources.append(data["docs"])
            sources_ori.append(data["docs_original"])
            num_sources.append(len(data["docs"]))
    df["answer"] = answer
    df["sources"] = sources
    df["sources_ori"] = sources_ori
    df["num_sources"] = num_sources
    df.to_excel(file_name.replace("问题集","结果集"), index=False)
    print("测试完成！")

        

if __name__ == "__main__":
    test_kb_chat()

