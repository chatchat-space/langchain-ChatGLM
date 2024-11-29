import base64
import json
import os
from io import BytesIO

import rich
from langchain_core.messages import BaseMessage


def encode_file_to_base64(file):
    # 将文件内容转换为 Base64 编码
    buffer = BytesIO()
    buffer.write(file.read())
    return base64.b64encode(buffer.getvalue()).decode()


def process_files(files):
    result = {"videos": [], "images": [], "audios": []}
    for file in files:
        file_extension = os.path.splitext(file.name)[1].lower()

        # 检测文件类型并进行相应的处理
        if file_extension in [".mp4", ".avi"]:
            # 视频文件处理
            video_base64 = encode_file_to_base64(file)
            result["videos"].append(video_base64)
        elif file_extension in [".jpg", ".png", ".jpeg"]:
            # 图像文件处理
            image_base64 = encode_file_to_base64(file)
            result["images"].append(image_base64)
        elif file_extension in [".mp3", ".wav", ".ogg", ".flac"]:
            # 音频文件处理
            audio_base64 = encode_file_to_base64(file)
            result["audios"].append(audio_base64)

    return result


def get_title(response):
    # 定义一个字典来映射 graph node 到 title
    node_titles = {
        "function_call": "Function Call",
        "function_call_loop": "Function Call",
        "chatbot": "ChatBot",
        "planner": "Planner",
        "replan": "Replan",
        "agent": "Agent",
        "revise": "Revise",
        "draft": "Draft"
    }

    # 获取 node 值
    node = response["node"]

    # 特殊处理 tools 节点
    if node == "tools":
        title = "Function Call: " + response["content"]["name"]
    else:
        # 使用字典获取 title，如果 node 不在字典中，直接使用 node 值
        title = node_titles.get(node, node)

    return title


def serialize_to_json(obj):
    """ 序列化对象为JSON兼容的格式。 """
    if isinstance(obj, BaseMessage):
        return {
            "content": obj.content,
            "additional_kwargs": serialize_to_json(obj.additional_kwargs),
            "response_metadata": serialize_to_json(obj.response_metadata),
            "type": obj.type,
            "name": obj.name,
            "id": obj.id,
        }
    elif isinstance(obj, dict):
        return {key: serialize_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_to_json(item) for item in obj]
    else:
        return obj


def process_content_by_graph(selected_graph, response):
    """ 根据不同的 selected_graph 来定制化 UI 展示内容。 """
    content = response.get("content", {})
    if selected_graph == "base_graph":
        content_dict = serialize_to_json(content)
        final_text = content.get("content", "")
    elif selected_graph == "plan_and_execute":
        if isinstance(content, dict):
            content_dict = content
        elif isinstance(content, list):
            content_dict = {item.get("key", idx): item for idx, item in enumerate(content) if isinstance(item, dict)}
        else:
            content_dict = {}
        final_text = content.get("response", "") if response.get("node") == "replan" else ""
    elif selected_graph == "reflexion":
        if isinstance(content, list):
            content_dict = [item for item in content if isinstance(item, dict)]
        else:
            content_dict = content
        final_text = content.get("answer", "") if response.get("node") == "revise" else ""
    else:
        content_dict = serialize_to_json(content)
        final_text = content.get("content", "")

    json_text = json.dumps(content_dict, indent=2)
    return json_text, final_text
