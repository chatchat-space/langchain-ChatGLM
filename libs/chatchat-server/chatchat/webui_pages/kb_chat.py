from datetime import datetime
import uuid
from typing import List, Dict

import openai
import streamlit as st
import streamlit_antd_components as sac
from streamlit_chatbox import *
from streamlit_extras.bottom_container import bottom

from chatchat.settings import Settings
from chatchat.server.knowledge_base.utils import LOADER_DICT
from chatchat.server.utils import get_config_models, get_config_platforms, get_default_llm, api_address
from chatchat.webui_pages.dialogue.dialogue import (save_session, restore_session, rerun,
                                                    get_messages_history, upload_temp_docs,
                                                    add_conv, del_conv, clear_conv)
from chatchat.webui_pages.utils import *


chat_box = ChatBox(assistant_avatar=get_img_base64("chatchat_icon_blue_square_v2.png"))


def init_widgets():
    st.session_state.setdefault("history_len", Settings.model_settings.HISTORY_LEN)
    st.session_state.setdefault("selected_kb", Settings.kb_settings.DEFAULT_KNOWLEDGE_BASE)
    st.session_state.setdefault("kb_top_k", Settings.kb_settings.VECTOR_SEARCH_TOP_K)
    st.session_state.setdefault("se_top_k", Settings.kb_settings.SEARCH_ENGINE_TOP_K)
    st.session_state.setdefault("score_threshold", Settings.kb_settings.SCORE_THRESHOLD)
    st.session_state.setdefault("search_engine", Settings.kb_settings.DEFAULT_SEARCH_ENGINE)
    st.session_state.setdefault("return_direct", False)
    st.session_state.setdefault("cur_conv_name", chat_box.cur_chat_name)
    st.session_state.setdefault("last_conv_name", chat_box.cur_chat_name)
    st.session_state.setdefault("file_chat_id", None)


def kb_chat(api: ApiRequest):
    ctx = chat_box.context
    ctx.setdefault("uid", uuid.uuid4().hex)
    ctx.setdefault("file_chat_id", None)
    ctx.setdefault("llm_model", get_default_llm())
    ctx.setdefault("temperature", Settings.model_settings.TEMPERATURE)
    init_widgets()

    # sac on_change callbacks not working since st>=1.34
    if st.session_state.cur_conv_name != st.session_state.last_conv_name:
        save_session(st.session_state.last_conv_name)
        restore_session(st.session_state.cur_conv_name)
        st.session_state.last_conv_name = st.session_state.cur_conv_name

    # st.write(chat_box.cur_chat_name)
    # st.write(st.session_state)

    @st.dialog("模型配置", width="large")
    def llm_model_setting():
        # 模型
        cols = st.columns(3)
        platforms = ["所有"] + list(get_config_platforms())
        platform = cols[0].selectbox("选择模型平台", platforms, key="platform")
        llm_models = list(
            get_config_models(
                model_type="llm", platform_name=None if platform == "所有" else platform
            )
        )
        llm_models += list(
            get_config_models(
                model_type="image2text", platform_name=None if platform == "所有" else platform
            )
        )
        llm_model = cols[1].selectbox("选择LLM模型", llm_models, key="llm_model")
        temperature = cols[2].slider("Temperature", 0.0, 1.0, key="temperature")
        system_message = st.text_area("System Message:", key="system_message")
        if st.button("OK"):
            rerun()

    @st.dialog("重命名会话")
    def rename_conversation():
        name = st.text_input("会话名称")
        if st.button("OK"):
            chat_box.change_chat_name(name)
            restore_session()
            st.session_state["cur_conv_name"] = name
            rerun()

    # 配置参数
    with st.sidebar:
        tabs = st.tabs(["RAG 配置", "会话设置"])
        with tabs[0]:
            dialogue_modes = ["知识库问答",
                              "文件对话",
                              "搜索引擎问答",
                              ]
            dialogue_mode = st.selectbox("请选择对话模式：",
                                         dialogue_modes,
                                         key="dialogue_mode",
                                         )
            placeholder = st.empty()
            st.divider()
            # prompt_templates_kb_list = list(Settings.prompt_settings.rag)
            # prompt_name = st.selectbox(
            #     "请选择Prompt模板：",
            #     prompt_templates_kb_list,
            #     key="prompt_name",
            # )
            prompt_name="default"
            history_len = st.number_input("历史对话轮数：", 0, 20, key="history_len")
            kb_top_k = st.number_input("匹配知识条数：", 1, 20, key="kb_top_k")
            ## Bge 模型会超过1
            score_threshold = st.slider("知识匹配分数阈值：", 0.0, 2.0, step=0.01, key="score_threshold")
            return_direct = st.checkbox("仅返回检索结果", key="return_direct")



            def on_kb_change():
                st.toast(f"已加载知识库： {st.session_state.selected_kb}")

            with placeholder.container():
                if dialogue_mode == "知识库问答":
                    kb_list = [x["kb_name"] for x in api.list_knowledge_bases()]
                    selected_kb = st.selectbox(
                        "请选择知识库：",
                        kb_list,
                        on_change=on_kb_change,
                        key="selected_kb",
                    )
                elif dialogue_mode == "文件对话":
                    files = st.file_uploader("上传知识文件：",
                                            [i for ls in LOADER_DICT.values() for i in ls],
                                            accept_multiple_files=True,
                                            )
                    if st.button("开始上传", disabled=len(files) == 0):
                        st.session_state["file_chat_id"] = upload_temp_docs(files, api)
                elif dialogue_mode == "搜索引擎问答":
                    search_engine_list = list(Settings.tool_settings.search_internet["search_engine_config"])
                    search_engine = st.selectbox(
                        label="请选择搜索引擎",
                        options=search_engine_list,
                        key="search_engine",
                    )

        with tabs[1]:
            # 会话
            cols = st.columns(3)
            conv_names = chat_box.get_chat_names()

            def on_conv_change():
                print(conversation_name, st.session_state.cur_conv_name)
                save_session(conversation_name)
                restore_session(st.session_state.cur_conv_name)

            conversation_name = sac.buttons(
                conv_names,
                label="当前会话：",
                key="cur_conv_name",
                on_change=on_conv_change,
            )
            chat_box.use_chat_name(conversation_name)
            conversation_id = chat_box.context["uid"]
            if cols[0].button("新建", on_click=add_conv):
                ...
            if cols[1].button("重命名"):
                rename_conversation()
            if cols[2].button("删除", on_click=del_conv):
                ...

    # Display chat messages from history on app rerun
    chat_box.output_messages()
    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter。"

    llm_model = ctx.get("llm_model")

    # chat input
    with bottom():
        cols = st.columns([1, 0.2, 15,  1])
        if cols[0].button(":gear:", help="模型配置"):
            widget_keys = ["platform", "llm_model", "temperature", "system_message"]
            chat_box.context_to_session(include=widget_keys)
            llm_model_setting()
        if cols[-1].button(":wastebasket:", help="清空对话"):
            chat_box.reset_history()
            rerun()
        # with cols[1]:
        #     mic_audio = audio_recorder("", icon_size="2x", key="mic_audio")
        prompt = cols[2].chat_input(chat_input_placeholder, key="prompt")
    if prompt:
        history = get_messages_history(ctx.get("history_len", 0))
        messages = history + [{"role": "user", "content": prompt}]
        chat_box.user_say(prompt)

        extra_body = dict(
            top_k=kb_top_k,
            score_threshold=score_threshold,
            temperature=ctx.get("temperature"),
            prompt_name=prompt_name,
            return_direct=return_direct,
        )
    
        api_url = api_address(is_public=True)
        if dialogue_mode == "知识库问答":
            client = openai.Client(base_url=f"{api_url}/knowledge_base/local_kb/{selected_kb}", api_key="NONE")
            chat_box.ai_say([
                Markdown("...", in_expander=True, title="知识库匹配结果", state="running", expanded=return_direct),
                f"正在查询知识库 `{selected_kb}` ...",
            ])
        elif dialogue_mode == "文件对话":
            if st.session_state.get("file_chat_id") is None:
                st.error("请先上传文件再进行对话")
                st.stop()
            knowledge_id=st.session_state.get("file_chat_id")
            client = openai.Client(base_url=f"{api_url}/knowledge_base/temp_kb/{knowledge_id}", api_key="NONE")
            chat_box.ai_say([
                Markdown("...", in_expander=True, title="知识库匹配结果", state="running", expanded=return_direct),
                f"正在查询文件 `{st.session_state.get('file_chat_id')}` ...",
            ])
        else:
            client = openai.Client(base_url=f"{api_url}/knowledge_base/search_engine/{search_engine}", api_key="NONE")
            chat_box.ai_say([
                Markdown("...", in_expander=True, title="知识库匹配结果", state="running", expanded=return_direct),
                f"正在执行 `{search_engine}` 搜索...",
            ])

        text = ""
        first = True

        try:
            for d in client.chat.completions.create(messages=messages, model=llm_model, stream=True, extra_body=extra_body):
                # if first:
                #     chat_box.update_msg("\n\n".join(d.docs), element_index=0, streaming=False, state="complete")
                #     chat_box.update_msg("", streaming=False)
                #     first = False
                #     continue
                chat_box.update_msg("\n\n".join(d.docs), element_index=0, streaming=False, state="complete")
                chat_box.update_msg("", streaming=False)
                text += d.choices[0].delta.content or ""
                chat_box.update_msg(text.replace("\n", "\n\n"), streaming=True)
            chat_box.update_msg(text, streaming=False)
            # TODO: 搜索未配置API KEY时产生报错
        except Exception as e:
            st.error(e)

    now = datetime.now()
    with tabs[1]:
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
            "清空对话",
            use_container_width=True,
        ):
            chat_box.reset_history()
            rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # st.write(chat_box.history)
