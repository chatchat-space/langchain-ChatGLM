## 常用 API 接口调用方式

### 说明

所有接口可以到 `{api_address}/docs` 中查看参数和测试。

### /tools

该接口列出所有工具及其参数信息。  
输入参数：无  
输出示例：
```
{
  "code": 200,
  "msg": "success",
  "data": {
    "search_local_knowledgebase": {
      "name": "search_local_knowledgebase",
      "title": "本地知识库",
      "description": "Use local knowledgebase from one or more of these: test: 关于test的知识库 samples: 关于本项目issue的解答 to get information，Only local data on this knowledge use this tool. The 'database' should be one of the above [test samples].",
      "args": {
        "database": {
          "title": "Database",
          "description": "Database for Knowledge Search",
          "choices": [
            "test",
            "samples"
          ],
          "type": "string"
        },
        "query": {
          "title": "Query",
          "description": "Query for Knowledge Search",
          "type": "string"
        }
      },
      "config": {
        "use": false,
        "top_k": 3,
        "score_threshold": 1,
        "conclude_prompt": {
          "with_result": "<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 \"根据已知信息无法回答该问题\"，不允许在答案中添加编造成分，答案请使用中文。 </指令>\n<已知信息>{{ context }}</已知信息>\n<问题>{{ question }}</问题>\n",
          "without_result": "请你根据我的提问回答我的问题:\n{{ question }}\n请注意，你必须在回答结束后强调，你的回答是根据你的经验回答而不是参考资料回答的。\n"
        }
      }
    },
    ...
  }
}
```

### 通用对话接口（/chat/chat/completions）

最主要的对话接口，兼容 openai sdk 格式。它支持以下3种对话模式：  
- 纯 LLM 对话。传入 `model`, `messages` 参数即可，可选参数： `temperature`, `max_tokens`, `stream` 等。
- Agent 对话。在 LLM 对话的基础上，传入 `tools` 参数，可以让 LLM 选择合适的工具和参数，作为对话的参考。
- 半 Agent 对话。在 LLM 对话的基础上，传入 `tool_choice` 参数，可以让 LLM 解析参数，直接调用指定的工具，作为对话的参考。如果使用的 LLM 解析参数的效果不理想，也可以手动指定工具参数。

输入参数：与 openai sdk 参数一致。针对 chatchat 做了以下优化：  
- `tools` 参数可以使用 chatchat 中编写的工具名称，所有支持的工具可以通过 `/tools` 接口获取。
- 在指定 `tool_choice` 的情况下，可以在 `extra_body` 中传入 `tool_input={...}` 来手动指定工具参数。
- 使用 Agent 功能时，`stream` 参数必须指定为 `True`。因为 Agent 是分步执行的，必须通过 SSE 把每个步骤逐一输出。注意：此时 SSE 的单元是执行步骤，LLM 的输出是非流式的。

调用示例：
- LLM 对话：
    ```python3
    base_url = "http://127.0.0.1:7861/chat"
    data = {
        "model": "qwen1.5-chat",
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我是人工智能大模型"},
            {"role": "user", "content": "请用100字左右的文字介绍自己"},
        ],
        "stream": True,
        "temperature": 0.7,
    }

    # 方式一：使用 requests
    import requests
    response = requests.post(f"{base_url}/chat/completions", json=data, stream=True)
    for line in response.iter_content(None, decode_unicode=True):
        print(line, end="", flush=True)

    # 方式二：使用 openai sdk
    import openai
    client = openai.Client(base_url=base_url, api_key="EMPTY")
    resp = client.chat.completions.create(**data)
    for r in resp:
        print(r)
    ```

    ```shell
    # 方式一输出，SSE 格式
    data: {"id":"chat6aa251c3-3425-11ef-be81-603a7c6af450","choices":[{"delta":{"content":"","function_call":null,"role":"assistant","tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719452077,"model":"qwen1.5-chat","object":"chat.completion.chunk","system_fingerprint":null,"usage":null,"message_id":null,"status":null}
    data: {"id":"chat6aa251c3-3425-11ef-be81-603a7c6af450","choices":[{"delta":{"content":"我是","function_call":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719452077,"model":"qwen1.5-chat","object":"chat.completion.chunk","system_fingerprint":null,"usage":null,"message_id":null,"status":null}
    data: {"id":"chat6abf605c-3425-11ef-9f15-603a7c6af450","choices":[{"delta":{"content":"阿里云","function_call":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719452078,"model":"qwen1.5-chat","object":"chat.completion.chunk","system_fingerprint":null,"usage":null,"message_id":null,"status":null}
    data: {"id":"chat6ad00242-3425-11ef-af45-603a7c6af450","choices":[{"delta":{"content":"自主研发的","function_call":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1719452078,"model":"qwen1.5-chat","object":"chat.completion.chunk","system_fingerprint":null,"usage":null,"message_id":null,"status":null}
    ...
    ```
    ```shell
    # 方式二输出：
    ChatCompletionChunk(id='chat682070c8-3426-11ef-947d-603a7c6af450', choices=[Choice(delta=ChoiceDelta(content='', function_call=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1719452503, model='qwen1.5-chat', object='chat.completion.chunk', system_fingerprint=None, usage=None, message_id=None, status=None)
    ChatCompletionChunk(id='chat682070c8-3426-11ef-947d-603a7c6af450', choices=[Choice(delta=ChoiceDelta(content='我是', function_call=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1719452503, model='qwen1.5-chat', object='chat.completion.chunk', system_fingerprint=None, usage=None, message_id=None, status=None)
    ChatCompletionChunk(id='chat683fdd72-3426-11ef-be33-603a7c6af450', choices=[Choice(delta=ChoiceDelta(content='由阿里', function_call=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1719452503, model='qwen1.5-chat', object='chat.completion.chunk', system_fingerprint=None, usage=None, message_id=None, status=None)
    ChatCompletionChunk(id='chat68511ba1-3426-11ef-b2be-603a7c6af450', choices=[Choice(delta=ChoiceDelta(content='云开发', function_call=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1719452503, model='qwen1.5-chat', object='chat.completion.chunk', system_fingerprint=None, usage=None, message_id=None, status=None)
    ...
    ```
- Agent 对话  
    以下示例仅展示使用 `requests` 的情况，可以自行尝试使用 `openai sdk` 进行请求，参数和输出内容是一样的。
    - langchain agent 对话示例:
    ```python3
    base_url = "http://127.0.0.1:7861/chat"
    tools = list(requests.get(f"http://127.0.0.1:7861/tools").json()["data"])
    data = {
        "model": "qwen1.5-chat",
        "messages": [
            {"role": "user", "content": "37+48=？"},
        ],
        "stream": True,
        "temperature": 0.7,
        "tools": tools,
    }

    import requests
    response = requests.post(f"{base_url}/chat/completions", json=data, stream=True)
    for line in response.iter_content(None, decode_unicode=True):
        print(line)
    ```
    ```shell
    # 输出：
    data: {"id": "chat39830df6-d016-4b91-b502-e113bb71542c", "object": "chat.completion.chunk", "model": "qwen1.5-chat", "created": 1719453364, "status": 1, "message_type": 1, "message_id": null, "is_ref": false, "choices": [{"delta": {"content": "", "tool_calls": []}, "role": "assistant"}]}
    data: {"id": "chatb05f9cb2-1e93-4657-806b-29ec135483b9", "object": "chat.completion.chunk", "model": "qwen1.5-chat", "created": 1719453367, "status": 3, "message_type": 1, "message_id": null, "is_ref": false, "choices": [{"delta": {"content": "Thought: The problem involves adding two numbers: 37 and 48. To perform this calculation, I will use the calculator API.\nAction: calculate\nAction Input: {\"text\": \"37 + 48\"}", "tool_calls": []}, "role": "assistant"}]}
    data: {"id": "chat73adade0-b62f-412a-a448-9002a59cbc30", "object": "chat.completion.chunk", "model": "qwen1.5-chat", "created": 1719453367, "status": 4, "message_type": 1, "message_id": null, "is_ref": false, "choices": [{"delta": {"content": "Thought: The problem involves adding two numbers: 37 and 48. To perform this calculation, I will use the calculator API.\nAction: calculate\nAction Input: {\"text\": \"37 + 48\"}", "tool_calls": []}, "role": "assistant"}]}
    data: {"id": "chat7752232b-7360-4010-bc55-e50fa8ac9f44", "object": "chat.completion.chunk", "model": "qwen1.5-chat", "created": 1719453367, "status": 6, "message_type": 1, "message_id": null, "is_ref": false, "choices": [{"delta": {"content": "", "tool_calls": [{"index": 0, "id": "f2b20744-3958-4e3b-9e51-c5738d87a020", "type": "function", "function": {"name": "calculate", "arguments": "{'text': '37 + 48'}"}, "tool_output": null, "is_error": false}]}, "role": "assistant"}]}
    data: {"id": "chatef5f948e-4772-477d-823d-ce74b38ba586", "object": "chat.completion.chunk", "model": "qwen1.5-chat", "created": 1719453367, "status": 7, "message_type": 1, "message_id": null, "is_ref": false, "choices": [{"delta": {"content": "", "tool_calls": [{"index": 0, "id": "f2b20744-3958-4e3b-9e51-c5738d87a020", "type": "function", "function": {"name": "calculate", "arguments": "{'text': '37 + 48'}"}, "tool_output": "85", "is_error": false}]}, "role": "assistant"}]}
    data: {"id": "chatdee106c6-42e6-41cf-b2df-692431829e4d", "object": "chat.completion.chunk", "model": "qwen1.5-chat", "created": 1719453367, "status": 1, "message_type": 1, "message_id": null, "is_ref": false, "choices": [{"delta": {"content": "", "tool_calls": []}, "role": "assistant"}]}
    data: {"id": "chat819ef11b-576f-4489-b6bb-47565eb69ee8", "object": "chat.completion.chunk", "model": "qwen1.5-chat", "created": 1719453370, "status": 3, "message_type": 1, "message_id": null, "is_ref": false, "choices": [{"delta": {"content": " The calculation 37 + 48 has been successfully performed using the calculate API, resulting in the result of 85. Therefore, the final answer to the given question is 85. \n\nJSON Object:\n{\n  \"answer\": 85\n}", "tool_calls": []}, "role": "assistant"}]}
    data: {"id": "chatb6b1071b-5346-4713-922c-b2887728491f", "object": "chat.completion.chunk", "model": "qwen1.5-chat", "created": 1719453370, "status": 5, "message_type": 1, "message_id": null, "is_ref": false, "choices": [{"delta": {"content": " The calculation 37 + 48 has been successfully performed using the calculate API, resulting in the result of 85. Therefore, the final answer to the given question is 85. \n\nJSON Object:\n{\n  \"answer\": 85\n}", "tool_calls": []}, "role": "assistant"}]}
    ```
    输出中包含一个 `status` 字段，代表 Agent 当前执行阶段。在 `status` 为 6 和 7 的输出中，可以看到 tool_call 的相关信息。具体值为：
    ```python3
    class AgentStatus:
        llm_start: int = 1
        llm_new_token: int = 2
        llm_end: int = 3
        agent_action: int = 4
        agent_finish: int = 5
        tool_start: int = 6
        tool_end: int = 7
        error: int = 8
    ```

    输出中包含一个 `message_type` 字段，代表输出内容的类型，主要用于前端渲染不同的消息，当前除了`text2image` 工具是 `IMAGE`，其它都是 `TEXT`。具体值为：
    ```python3
    class MsgType:
        TEXT = 1
        IMAGE = 2
        AUDIO = 3
        VIDEO = 4
    ```
    - langgraph agent 对话示例:
    ```python3
    base_url = "http://127.0.0.1:7861/chat"
    tools = list(requests.get(f"http://127.0.0.1:7861/tools").json()["data"])
    data = {
        "messages": [
            {'role': 'user', 'content': 'The youtube video of Xiao Yixian in Fights Break Sphere?'},
        ],
        "model": "gpt-4o",      # 必填, 部分模型推理能力有限, 可能无法实现较复杂的推理过程, 如有条件请尽量选在线 api.
        "tools": tools,         # 必填, agent 可使用 tools.
        "stream": False,        # 可选, 默认为 False,
                                # 原因: 部分模型推理平台在"不使用 vllm 启动 llm 时"的场景下不支持 "stream = Ture".
                                # 为考虑兼容性, 默认为 False, 使用在线 API 或满足条件的用户可设置为 True.
        "temperature": 0.01,    # 可选, 但为保障大部分用户使用体验, 代码中优先指定为 `LLM_MODEL_CONFIG.action_model.temperature`, 
                                # 即 0.0.1, 硬性需求接口传入 temperature 的开发者请查看 graph_chat._create_agent_models.
        "graph": "base_graph",  # 必填, 用来启动 graph agent, 且指定 agent 使用的 graph 流程, 目前支持: base_graph 和 plan_and_execute.
        "conversation_id": 1,   # 必填, checkpoint 以 conversation_id 为纬度记录历史上下文.
    }

    import requests
    response = requests.post(f"{base_url}/chat/completions", json=data, stream=True)
    for line in response.iter_content(None, decode_unicode=True):
        print(line)
    ```
    ```shell
    stream = True
    # 输出:
    data: {"id": "chataae129d2-2832-4895-941f-9debfbb60c39", "object": "chat.completion.chunk", "model": "gpt-4o-mini", "created": 1723472598, "status": 5, "message_type": 1, "message_id": "6505a824f2864c79b3d93b70ef588d8d", "is_ref": false, "choices": [{"delta": {"content": "node: human  \ncontent: The youtube video of Xiao Yixian in Fights Break Sphere?  \n", "tool_calls": []}, "role": "assistant"}]}
    data: {"id": "chat1c7b1c90-778c-49e2-aa3f-60b38464a6bb", "object": "chat.completion.chunk", "model": "gpt-4o-mini", "created": 1723472598, "status": 5, "message_type": 1, "message_id": "6505a824f2864c79b3d93b70ef588d8d", "is_ref": false, "choices": [{"delta": {"content": "node: human  \ncontent: The youtube video of Xiao Yixian in Fights Break Sphere?  \n", "tool_calls": []}, "role": "assistant"}]}
    data: {"id": "chat6e7ca234-a6f3-441a-bbf0-d8decd9fa33e", "object": "chat.completion.chunk", "model": "gpt-4o-mini", "created": 1723472600, "status": 5, "message_type": 1, "message_id": "6505a824f2864c79b3d93b70ef588d8d", "is_ref": false, "choices": [{"delta": {"content": "node: ai  \ncontent: tool_calls:  \n  - type: tool_call  \n    name: search_youtube  \n    args: {'query': 'Xiao Yixian Fights Break Sphere'}  \n  \n", "tool_calls": []}, "role": "assistant"}]}
    data: {"id": "chat596297dc-ce38-4c57-9ac3-b7727069b4ce", "object": "chat.completion.chunk", "model": "gpt-4o-mini", "created": 1723472602, "status": 5, "message_type": 1, "message_id": "6505a824f2864c79b3d93b70ef588d8d", "is_ref": false, "choices": [{"delta": {"content": "node: tool  \nname: search_youtube  \ncontent: ['https://www.youtube.com/watch?v=em-VpMhQcZs&pp=ygUfWGlhbyBZaXhpYW4gRmlnaHRzIEJyZWFrIFNwaGVyZQ%3D%3D', 'https://www.youtube.com/watch?v=bqV-5qKtdTQ&pp=ygUfWGlhbyBZaXhpYW4gRmlnaHRzIEJyZWFrIFNwaGVyZQ%3D%3D']  \n", "tool_calls": []}, "role": "assistant"}]}
    data: {"id": "chat27c1317b-14de-48c6-88ed-e9c15f27206e", "object": "chat.completion.chunk", "model": "gpt-4o-mini", "created": 1723472605, "status": 5, "message_type": 1, "message_id": "6505a824f2864c79b3d93b70ef588d8d", "is_ref": false, "choices": [{"delta": {"content": "node: ai  \ncontent: Here are some YouTube videos featuring Xiao Yixian in \"Fights Break Sphere\":\n\n1. [Xiao Yixian in Fights Break Sphere - Video 1](https://www.youtube.com/watch?v=em-VpMhQcZs&pp=ygUfWGlhbyBZaXhpYW4gRmlnaHRzIEJyZWFrIFNwaGVyZQ%3D%3D)\n2. [Xiao Yixian in Fights Break Sphere - Video 2](https://www.youtube.com/watch?v=bqV-5qKtdTQ&pp=ygUfWGlhbyBZaXhpYW4gRmlnaHRzIEJyZWFrIFNwaGVyZQ%3D%3D)\n\nFeel free to check them out!  \n", "tool_calls": []}, "role": "assistant"}]}
    ```
    ```shell
    stream = False
    # 输出:
    {"id":"chat3609a5e1-fadd-4bc4-8280-4b3d8fb2c86e","object":"chat.completion","model":null,"created":null,"status":5,"message_type":1,"message_id":"3be4e0db8eed4747af6b5c647fcd7891","is_ref":false,"choices":[{"message":{"role":"assistant","content":"node: human  \ncontent: The youtube video of Xiao Yixian in Fights Break Sphere?  \nnode: human  \ncontent: The youtube video of Xiao Yixian in Fights Break Sphere?  \nnode: ai  \ncontent: tool_calls:  \n  - type: tool_call  \n    name: search_youtube  \n    args: {'query': 'Xiao Yixian Fights Break Sphere'}  \n  \nnode: tool  \nname: search_youtube  \ncontent: ['https://www.youtube.com/watch?v=em-VpMhQcZs&pp=ygUfWGlhbyBZaXhpYW4gRmlnaHRzIEJyZWFrIFNwaGVyZQ%3D%3D', 'https://www.youtube.com/watch?v=AAE3tXMQ2Vo&pp=ygUfWGlhbyBZaXhpYW4gRmlnaHRzIEJyZWFrIFNwaGVyZQ%3D%3D']  \n","finish_reason":"stop","tool_calls":[]}}]}
    ```
- 知识库对话（LLM 自动解析参数）  
    直接指定 `tool_choice` 为 `"search_local_knowledgebase"`工具即可使用知识库对话功能。其它工具对话类似。
    ```python3
    base_url = "http://127.0.0.1:7861/chat"
    data = {
        "messages": [
            {"role": "user", "content": "如何提问以获得高质量答案"},
        ],
        "model": "qwen1.5-chat",
        "tool_choice": "search_local_knowledgebase",
        "stream": True,
    }

    import requests
    response = requests.post(f"{base_url}/chat/completions", json=data, stream=True)
    for line in response.iter_content(None, decode_unicode=True):
        print(line)
    ```
    在 `status` 为 6 和 7 的返回值中，可以获取工具的调用和输出信息。  
    由于输出信息太多，这里不做展示，请自行测试。
- 知识库对话（手动传入参数）  
    直接指定 `tool_choice` 为 `"search_local_knowledgebase"`，再通过 `tool_input` 设定工具参数，即可手动调用工具，实现指定知识库对话。
    ```python3
    base_url = "http://127.0.0.1:7861/chat"
    data = {
        "messages": [
            {"role": "user", "content": "如何提问以获得高质量答案"},
        ],
        "model": "qwen1.5-chat",
        "tool_choice": "search_local_knowledgebase",
        "tool_input": {"database": "samples", "query": "如何提问以获得高质量答案"},
        "stream": True,
    }

    import requests
    response = requests.post(f"{base_url}/chat/completions", json=data, stream=True)
    for line in response.iter_content(None, decode_unicode=True):
        print(line)
    ```
    在输出中可以看出 `status` 为 6 和 7 的工具解析过程已经不存在了，说明工具调用不是通过 LLM 完成的。  
    由于输出信息太多，这里不做展示，请自行测试。


### RAG 接口 （/knowledge_base/chat/compleitons）
相比于 /chat/chat/completions 接口，本接口主要用于 RAG，支持更多的参数，其返回值也是 openai sdk 兼容的。除了 openai.chat.completions 规定的参数，还支持以下参数：  
- mode：检索模式：
  - "local_kb" 检索本地知识库，需提供 "kb_name" 指定知识库名称
  - "temp_kb" 文件对话，需提供 "knowledge_id" 指定文件对话的临时知识库 ID
  - "search_engine" 使用搜索引擎，需提供 "search_engine_name" 指定使用的搜索引擎
- top_k: 检索结果数量
- score_threshold: 匹配分值阈值
- prompt_name: 使用的 prompt 模板名称
- return_direct: 如果为 True，则仅返回检索结果，不经由 LLM
- messages: messages[-1]["content"] 将作为检索对象，messages[:-1] 将作为 history

返回值：

- stream 为 True 时，第一个 ChatCompletionChunk.docs 包含检索结果
- stream 为 False 时，ChatCompletion.docs 包含检索结果


调用示例（这里使用 openai sdk 演示本地知识库问答的情况，requests 参数相同，只是把 extra_body 中的内容放到 data 里即可）：

- 本地知识库问答
    ```python3
    base_url = "http://127.0.0.1:7861/knowledge_base/local_kb/samples"
    data = {
        "model": "qwen2-instruct",
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我是人工智能大模型"},
            {"role": "user", "content": "如何高质量提问？"},
        ],
        "stream": True,
        "temperature": 0.7,
        "extra_body": {
          "top_k": 3,
          "score_threshold": 2.0,
          "return_direct": True,
        },
    }

    import openai
    client = openai.Client(base_url=base_url, api_key="EMPTY")
    resp = client.chat.completions.create(**data)
    for r in resp:
        print(r)
    ```

    输出示例：
    ```shell
    ChatCompletionChunk(id='chat9973e445-8581-45ca-bde5-148fc724b30b', choices=[Choice(delta=None, finish_reason=None, index=None, logprobs=None, message={'role': 'assistant', 'content': '', 'finish_reason': 'stop', 'tool_calls': []})], created=1720592802, model=None, object='chat.completion', service_tier=None, system_fingerprint=None, usage=None, status=None, message_type=1, message_id=None, is_ref=False, docs=['出处 [1] [test_files/test.txt](http://127.0.0.1:7861//knowledge_base/download_doc?knowledge_base_name=samples&file_name=test_files%2Ftest.txt) \n\n[这就是那幅名画]: http://yesaiwen.com/art\nof\nasking\nchatgpt\nfor\nhigh\nquality\nansw\nengineering\ntechniques/#i\n3\t"《如何向ChatGPT提问并获得高质量的答案》"\n\n', '出处 [2] [test_files/test.txt](http://127.0.0.1:7861//knowledge_base/download_doc?knowledge_base_name=samples&file_name=test_files%2Ftest.txt) \n\nChatGPT是OpenAI开发的一个大型语言模型，可以提供各种主题的信息，\n# 如何向 ChatGPT 提问以获得高质量答案：提示技巧工程完全指南\n## 介绍\n我很高兴欢迎您阅读我的最新书籍《The Art of Asking ChatGPT for High-Quality Answers: A complete Guide to Prompt Engineering Techniques》。本书是一本全面指南，介绍了各种提示技术，用于从ChatGPT中生成高质量的答案。\n我们将探讨如何使用不同的提示工 程技术来实现不同的目标。ChatGPT是一款最先进的语言模型，能够生成类似人类的文本。然而，理解如何正确地向ChatGPT提问以获得我们所需的高质量输出非常重要。而这正是 本书的目的。\n无论您是普通人、研究人员、开发人员，还是只是想在自己的领域中将ChatGPT作为个人助手的人，本书都是为您编写的。我使用简单易懂的语言，提供实用的解释，并在每个提示技术中提供了示例和提示公式。通过本书，您将学习如何使用提示工程技术来控制ChatGPT的输出，并生成符合您特定需求的文本。\n在整本书中，我们还提供了如何结合不同的提示技术以实现更具体结果的示例。我希望您能像我写作时一样，享受阅读本书并从中获得知识。\n<div style="page\nbreak\nafter:always;"></div>\n## 第一章：Prompt 工程技术简介\n什么是 Prompt 工程？\nPrompt 工程是创建提示或指导像 ChatGPT 这样的语言模型输出的过程。它允许用户控制模型的输出并生成符合其特定需求的文本。\n\n', '出处 [3] [test_files/test.txt](http://127.0.0.1:7861//knowledge_base/download_doc?knowledge_base_name=samples&file_name=test_files%2Ftest.txt) \n\nPrompt 公式是提示的特定格式，通常由三个主要元素组成：**\n任务：对提示要求模型生成的内容进行清晰而简洁的陈述。\n指令：在生成文本时模型应遵循的指令。\n角色：模型在生成文本时应扮演的角色。\n在本书中，我们将探讨可用于 ChatGPT 的各种 Prompt 工程技术。我们将讨论不同类型的提示，以及如何使用它们实现您想要的特定目标。\n<div style="page\nbreak\nafter:always;"></div>\n## 第二章：指令提示技术\n现在，让我们开始探索“指令提示技术”，以及如何使用它从ChatGPT中生成高质量的文本。\n 指令提示技术是通过为模型提供具体指令来引导ChatGPT的输出的一种方法。这种技术对于确保输出相关和高质量非常有用。\n要使用指令提示技术，您需要为模型提供清晰简洁的任务，以及具体的指令以供模型遵循。\n例如，如果您正在生成客户服务响应，您将提供任务，例如“生成响应客户查询”的指令，例如“响应应该专业且提供准确的信息”。\n 提示公式：“按照以下指示生成[任务]：[指令]”\n示例：\n生成客户服务响应：**\n任务：生成响应客户查询\n指令：响应应该专业且提供准确的信息\n提示公式：“按照以下 指示生成专业且准确的客户查询响应：响应应该专业且提供准确的信息。”\n生成法律文件：**\n任务：生成法律文件\n指令：文件应符合相关法律法规\n提示公式：“按照以下 指示生成符合相关法律法规的法律文件：文件应符合相关法律法规。”\n使用指令提示技术时，重要的是要记住指令应该清晰具体。这将有助于确保输出相关和高质量。可以将指 令提示技术与下一章节中解释的“角色提示”和“种子词提示”相结合，以增强ChatGPT的输出。\n\n'])
    ```

- 文件对话
    ```python3
    # knowledge_id 为 /knowledge_base/upload_temp_docs 的返回值
    base_url = "http://127.0.0.1:7861/knowledge_base/temp_kb/{knowledge_id}"
    data = {
        "model": "qwen2-instruct",
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我是人工智能大模型"},
            {"role": "user", "content": "如何高质量提问？"},
        ],
        "stream": True,
        "temperature": 0.7,
        "extra_body": {
          "top_k": 3,
          "score_threshold": 2.0,
          "return_direct": True,
        },
    }

    import openai
    client = openai.Client(base_url=base_url, api_key="EMPTY")
    resp = client.chat.completions.create(**data)
    for r in resp:
        print(r)
    ```

- 搜索引擎问答
    ```python3
    engine_name = "bing" # 可选值：bing, duckduckgo, metaphor, searx
    base_url = f"http://127.0.0.1:7861/knowledge_base/search_engine/{engine_name}"
    data = {
        "model": "qwen2-instruct",
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我是人工智能大模型"},
            {"role": "user", "content": "如何高质量提问？"},
        ],
        "stream": True,
        "temperature": 0.7,
        "extra_body": {
          "top_k": 3,
          "score_threshold": 2.0,
          "return_direct": True,
        },
    }

    import openai
    client = openai.Client(base_url=base_url, api_key="EMPTY")
    resp = client.chat.completions.create(**data)
    for r in resp:
        print(r)
    ```
