from __future__ import annotations
import re
import copy
import asyncio, json
import uuid
from typing import AsyncIterable, List, Optional, Literal
import time
from fastapi import Body, Request
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.prompts.chat import ChatPromptTemplate

from langchain_openai.chat_models import ChatOpenAI
from chatchat.settings import Settings
from chatchat.server.agent.tools_factory.search_internet import search_engine
from chatchat.server.api_server.api_schemas import OpenAIChatOutput
from chatchat.server.chat.utils import History
from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory
from chatchat.server.knowledge_base.kb_doc_api import search_docs, search_temp_docs
from chatchat.server.knowledge_base.utils import format_reference
from chatchat.server.utils import (wrap_done, get_ChatOpenAI, get_default_llm,
                                   BaseResponse, get_prompt_template, build_logger,
                                   check_embed_model, api_address
                                )
from pprint import pprint

logger = build_logger()


async def adaptive_docs(
        docs:List[str],
        model:str,
        temperature:float,
        max_tokens:int,
        query:str,
        lang="zh",
                  ):
    """
    select related documents to send to LLM by self-criticize
    source_documents: list of documents
    llm: LLM model
    query: user query
    lang: language, "zh" or "en"

    """

    prompt_en = (
                "You’ll be provided with an instruction, along with evidence and possibly some preceding"
                "sentences. When there are preceding sentences, your focus should be on the sentence that"
                "comes after them. Your job is to determine if the evidence is relevant to the initial instruction"
                "and the preceding context, and provides useful information to complete the task described in"
                "the instruction. If the evidence meets this requirement, respond with [Relevant]; otherwise,"
                "generate [Irrelevant]. Here's two examples to help you understand the task better:\n\n"
                "Example 1:\n\n"
                "Instruction: Given four answer options, A, B, C, and D, choose the best answer."
                "Input Earth’s rotating causes"
                "A: the cycling of AM and PM"
                "B: the creation of volcanic eruptions"
                "C: the cycling of the tides"
                "D: the creation of gravity"
                "Evidence: Rotation causes the day-night cycle which also creates a corresponding cycle of"
                "temperature and humidity creates a corresponding cycle of temperature and humidity. Sea"
                "level rises and falls twice a day as the earth rotates."
                "**Rating** [Relevant]"
                "Explanation: The evidence explicitly mentions that the rotation causes a day-night cycle, as"
                "described in the answer option A."
                "Example 2:\n\n"
                "Instruction: age to run for US House of Representatives"
                "Evidence: The Constitution sets three qualifications for service in the U.S. Senate: age (at"
                "least thirty years of age); U.S. citizenship (at least nine years); and residency in the state a"
                "senator represents at the time of election."
                "**Rating** [Irrelevant]"
                "Explanation: The evidence only discusses the ages to run for the US Senate, not for the"
                "House of Representatives."
                "Examples completed.\n\n"
                "Please provide your response to the evidence in the following format: [Relevant] or [Irrelevant]."
                "Instruction: {}"
                "Evidence: {}"
                "Rating: "
                )
    prompt_zh = (
                "你将会收到一个指令，以及证据和可能的一些前述句子。当有前述句子时，你的重点应该放在它们之后的句子上。"
                "你的任务是判断证据是否与最初的指令和前述的上下文相关，并提供有用的信息来完成指令中描述的任务。"
                "如果证据符合这个要求，请回复[相关]；否则，请生成[不相关]。以下是两个示例，帮助你更好地理解任务："
                "示例1："
                "指令：在四个答案选项A、B、C和D中，选择最佳答案。"
                "输入：地球的旋转导致"
                "A: AM和PM的循环"
                "B: 火山喷发的形成"
                "C: 潮汐的循环"
                "D: 重力的形成"
                "证据：旋转导致昼夜循环，这也会产生相应的温度和湿度循环。随着地球的旋转，海平面每天上升和下降两次。"
                "**评级** [相关]"
                "解释：证据明确提到旋转导致昼夜循环，如答案选项A所述。"
                "示例2："
                "指令：竞选美国众议院的年龄"
                "证据：宪法为在美国参议院服务设定了三个资格：年龄（至少三十岁）；美国公民身份（至少九年）；以及在选举时参议员代表的州的居住权。"
                "**评级** [不相关]"
                "解释：证据只讨论了竞选美国参议院的年龄，而不是众议院。"
                "示例完成。"
                "请按照以下格式提供你对证据的回应：[相关]或[不相关]。直接输出评级，不要添加任何内容！"
                "指令：{}"
                "证据：{}"
                "评级："
                )
    prompt = prompt_zh if lang == "zh" else prompt_en
    prompts = [prompt.format(query, doc['page_content']) for doc in docs]
    # parallel generation with async
    messages = [[("human",prompt)] for prompt in prompts]
    llm = get_ChatOpenAI(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,

            )
    results = await llm.abatch(messages)
    results_key = [result.content for result in results]
    logger.info(f"adaptive_docs results_key: {results_key}")
    filter_word = "不相关" if lang == "zh" else "Irrelevant"
    relevance = [0 if filter_word in result else 1 for result in results_key]
    relevance_idx = [i for i, r in enumerate(relevance) if r == 1]
    docs_adaptive = [docs[i] for i in relevance_idx]
    return docs_adaptive
    

async def self_verify_evidence(
    docs:List[str],
    answer:str,
    model:str,
    temperature:float,
    max_tokens:int,
    query:str,
    lang="zh",
    ):
    """
    select related documents accroding to answer by self-verify whether the evidence support the answer
    """
    prompt_en = (
    "You will receive an instruction, evidence, and output, and optional preceding sentences. If the"
    "preceding sentence is given, the output should be the sentence that follows those preceding"
    "sentences. Your task is to evaluate if the output is fully supported by the information provided"
    "in the evidence."
    "Use the following entailment scale to generate a score:"
    "- [Fully supported] - All information in output is supported by the evidence, or extractions"
    "from the evidence. This is only applicable when the output and part of the evidence are"
    "almost identical."
    "- [Partially supported] - The output is supported by the evidence to some extent, but there"
    "is major information in the output that is not discussed in the evidence. For example, if an"
    "instruction asks about two concepts and the evidence only discusses either of them, it should"
    "be considered a [Partially supported]."
    "- [No support / Contradictory] - The output completely ignores evidence, is unrelated to the"
    "evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the"
    "instruction."
    "Make sure to not use any external information/knowledge to judge whether the output is true or not. "
    "Only check whether the output is supported by the evidence, and not"
    "whether the output follows the instructions or not."
    "Here's one example to help you understand the task better:\n\n"
    "Example:\n\n"
    "Instruction: Explain the use of word embeddings in Natural Language Processing."
    "Preceding sentences Word embeddings are one of the most powerful tools available for"
    "Natural Language Processing (NLP). They are mathematical representations of words or"
    "phrases in a vector space, allowing similarities between words and the context in which they"
    "are used to be measured."
    "Output Word embeddings are useful for tasks such as sentiment analysis, text classification,"
    "predicting the next word in a sequence, and understanding synonyms and analogies."
    "Evidence: Word embedding"
    "Word embedding is the collective name for a set of language modeling and feature learning"
    "techniques in natural language processing (NLP) where words or phrases from the vocabulary"
    "are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding"
    "from a space with one dimension per word to a continuous vector space with a much lower"
    "dimension. Methods to generate this mapping include neural networks, dimensionality"
    "reduction on the word co-occurrence matrix, probabilistic models, explainable knowledge"
    "base method, and explicit representation in terms of the context in which words appear. Word"
    "and phrase embeddings, when used as the underlying input representation, have been shown"
    "to boost the performance in NLP tasks such as syntactic parsing, sentiment analysis, next"
    "token predictions as well and analogy detection."
    "**Score**: [Fully supported]"
    "Explanation: The output sentence discusses the application of word embeddings, and the"
    "evidence mentions all of the applications syntactic parsing, sentiment analysis, next token"
    "predictions as well as analogy detection as the applications. Therefore, the score should be"
    "[Fully supported]."
    "Examples completed.\n\n"
    "Please provide your response to the evidence in the following format: [Fully supported], [Partially supported], or [No support / Contradictory]."
    "Instruction: {}"
    "Output: {}"
    "Evidence: {}"
    "Score: "
    )
    prompt_zh = (
    "你将收到一个指令、证据、输出和可选的前述句子。如果给出了前述句子，输出应该是跟在这些前述句子后面的句子。"
    "你的任务是评估输出是否完全由证据中提供的信息支持。"
    "使用以下蕴涵量表生成分数："
    "- [完全支持] - 输出中的所有信息都由证据或证据的提取支持。仅当输出和部分证据几乎相同时才适用。"
    "- [部分支持] - 输出在某种程度上由证据支持，但输出中有主要信息在证据中没有讨论。例如，如果指令询问两个概念，"
    "而证据只讨论其中一个，那么应该被视为[部分支持]。"
    "- [不支持/矛盾] - 输出完全忽略证据，与证据无关，或与证据矛盾。"
    "请确保不使用任何外部信息/知识来判断输出是否正确。只检查输出是否由证据支持，而不是检查输出是否遵循指令。"
    "以下是一个示例，帮助你更好地理解任务："
    "示例：\n\n"
    "指令：解释自然语言处理中词嵌入的使用。"
    "前述句子：词嵌入是自然语言处理（NLP）中最强大的工具之一。它们是词或短语在向量空间中的数学表示，"
    "允许测量单词之间的相似性以及它们的上下文。"
    "输出：词嵌入对情感分析、文本分类、预测序列中的下一个词以及理解同义词和类比等任务非常有用。"
    "证据：词嵌入是自然语言处理（NLP）中一组语言建模和特征学习技术的统称，其中词汇表中的词或短语被映射到实数向量。"
    "从概念上讲，它涉及从每个词一个维度的空间到一个具有更低维度的连续向量空间的数学嵌入。生成此映射的方法包括"
    "神经网络、词共现矩阵的降维、概率模型、可解释的知识库方法以及根据单词出现的上下文的显式表示。"
    "当用作底层输入表示时，词和短语嵌入已被证明可以提高NLP任务的性能，例如句法分析、情感分析、下一个令牌预测以及类比检测。"
    "**评分**: [完全支持]"
    "解释：输出句子讨论了词嵌入的应用，证据提到了所有的应用，如句法分析、情感分析、下一个令牌预测以及类比检测。"
    "因此，分数应该是[完全支持]。"
    "示例完成。\n\n"
    "请按照以下格式提供你对证据的评分：[完全支持]、[部分支持]或[不支持/矛盾]。直接输出评级，不要添加任何内容！"
    "指令：{}"
    "输出：{}"
    "证据：{}"
    "评分："
    )


    prompt = prompt_zh if lang == "zh" else prompt_en
    prompts = [prompt.format(query, answer, doc['page_content']) for doc in docs]
    # parallel generation with async
    messages = [[("human",prompt)] for prompt in prompts]
    llm = get_ChatOpenAI(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,

            )
    results = await llm.abatch(messages)
    results_key = [result.content for result in results]
    logger.info(f"self_verify results_key: {results_key}")
    ## -------keep fully support only---
    # filter_word = "完全支持" if lang == "zh" else "Fully supported"
    # support = [1 if filter_word in result else 0 for result in results_key]
    ## ---- keep fully support and partial support ----
    filter_word_ex = "不支持/矛盾" if lang == "zh" else "No support / Contradictory"
    support = [0 if filter_word_ex in result else 1 for result in results_key]

    support_idx = [i for i, r in enumerate(support) if r == 1]

    docs_self_verify = [docs[i] for i in support_idx]
    return docs_self_verify

async def self_verify_evidence_one_call(
        docs:List[str],
        answer:str,
        model:str,
        temperature:float,
        max_tokens:int,
        query:str,
        lang="zh",
        ):
    """
    select related documents accroding to answer by 
    self-verify whether the evidence support the answer by only one call
    docs: list of documents
    answer: user answer
    model: LLM model
    temperature: LLM temperature
    max_tokens: LLM max_tokens
    query: user query
    lang: language, "zh" or "en"
    """
    prompt_en = (
    "You will receive an instruction, several evidences, and output, and optional preceding sentences. If the"
    "preceding sentence is given, the output should be the sentence that follows those preceding"
    "sentences. Your task is to evaluate if the output is fully supported or partial supported by the information provided"
    "in the evidences, generate the scores and indides of evidences."
    "Use the following entailment scale to generate a score:"
    "- [Fully supported] - All information in output is supported by the evidences, or extractions"
    "from the evidences. This is only applicable when the output and part of the evidences are"
    "almost identical."
    "- [Partially supported] - The output is supported by the evidences to some extent, but there"
    "is major information in the output that is not discussed in the evidences. For example, if an"
    "instruction asks about two concepts and the evidences only discusses either of them, it should"
    "be considered a [Partially supported]."
    "- [No support / Contradictory] - The output completely ignores evidences, is unrelated to the"
    "evidences, or contradicts the evidences. This can also happen if the evidences are irrelevant to the"
    "instruction."
    "Make sure to not use any external information/knowledge to judge whether the output is true or not. "
    "Only check whether the output is supported by the evidences, and not"
    "whether the output follows the instructions or not."
    "Here's one example to help you understand the task better:\n\n"
    "Example:\n\n"
    "Instruction: Explain the use of word embeddings in Natural Language Processing."
    "Preceding sentences Word embeddings are one of the most powerful tools available for"
    "Natural Language Processing (NLP). They are mathematical representations of words or"
    "phrases in a vector space, allowing similarities between words and the context in which they"
    "are used to be measured."
    "Output Word embeddings are useful for tasks such as sentiment analysis, text classification,"
    "predicting the next word in a sequence, and understanding synonyms and analogies."
    "Evidence 1: Word embedding"
    "Word embedding is the collective name for a set of language modeling and feature learning"
    "techniques in natural language processing (NLP) where words or phrases from the vocabulary"
    "are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding"
    "from a space with one dimension per word to a continuous vector space with a much lower"
    "dimension. Methods to generate this mapping include neural networks, dimensionality"
    "reduction on the word co-occurrence matrix, probabilistic models, explainable knowledge"
    "base method, and explicit representation in terms of the context in which words appear. Word"
    "and phrase embeddings, when used as the underlying input representation, have been shown"
    "to boost the performance in NLP tasks such as syntactic parsing, sentiment analysis, next"
    "token predictions as well and analogy detection.\n"
    "Evidence 2: BERT"
    "BERT language model is an open source machine learning framework for natural language processing (NLP). "
    "BERT is designed to help computers understand the meaning of ambiguous language "
    "in text by using surrounding text to establish context.\n"
    "Evidence 3: GPT"
    "GPT is a large language model that can generate human-like text. It is trained on a large corpus of text data "
    "and can generate text that is similar to the text it was trained on.\n"
    "Evidence 1: [Fully supported]"
    "Evidence 2: [Partially supported]"
    "Evidence 3: [No support / Contradictory]"
    "Examples completed.\n\n"
    "Please provide your response to the evidences in the following format: "
    "Evidence index: [Fully supported], [Partially supported], or [No support / Contradictory]."

    "Instruction: {}"
    "Output: {}"
    "Evidences:\n {}"
    "Your response: "
    )

    prompt_zh = (
    "你将收到一个指令、多个证据、输出和可选的前述句子。如果给出了前述句子，输出应该是跟在这些前述句子后面的句子。"
    "你的任务是评估输出是否完全由证据支持或部分由证据支持，生成分数和证据的索引。"
    "使用以下蕴涵量表生成分数："
    "- [完全支持] - 输出中的所有信息都由证据或证据的提取支持。仅当输出和部分证据几乎相同时才适用。"
    "- [部分支持] - 输出在某种程度上由证据支持，但输出中有主要信息在证据中没有讨论。例如，如果指令询问两个概念，"
    "而证据只讨论其中一个，那么应该被视为[部分支持]。"
    "- [不支持/矛盾] - 输出完全忽略证据，与证据无关，或与证据矛盾。"
    "请确保不使用任何外部信息/知识来判断输出是否正确。只检查输出是否由证据支持，而不是检查输出是否遵循指令。"
    "以下是一个示例，帮助你更好地理解任务："
    "示例：\n\n"
    "指令：解释自然语言处理中词嵌入的使用。"
    "前述句子：词嵌入是自然语言处理（NLP）中最强大的工具之一。它们是词或短语在向量空间中的数学表示，"
    "允许测量单词之间的相似性以及它们的上下文。"
    "输出：词嵌入对情感分析、文本分类、预测序列中的下一个词以及理解同义词和类比等任务非常有用。"
    "证据 1：词嵌入"
    "词嵌入是自然语言处理（NLP）中一组语言建模和特征学习技术的统称，其中词汇表中的词或短语被映射到实数向量。"
    "从概念上讲，它涉及从每个词一个维度的空间到一个具有更低维度的连续向量空间的数学嵌入。生成此映射的方法包括"
    "神经网络、词共现矩阵的降维、概率模型、可解释的知识库方法以及根据单词出现的上下文的显式表示。"
    "当用作底层输入表示时，词和短语嵌入已被证明可以提高NLP任务的性能，例如句法分析、情感分析、下一个令牌预测以及类比检测。\n"
    "证据 2：BERT"
    "BERT语言模型是一种用于自然语言处理（NLP）的开源机器学习框架。"
    "BERT旨在通过使用周围文本来建立上下文，帮助计算机理解文本中的模糊语言的含义。\n"
    "证据 3：GPT"
    "GPT是一个可以生成类似人类文本的大型语言模型。它是在大量文本数据上训练的，可以生成与其训练文本相似的文本。\n"
    "证据 1：[完全支持]"
    "证据 2：[部分支持]"
    "证据 3：[不支持/矛盾]"
    "示例完成。\n\n"
    "请按照以下格式提供你对证据的评分："
    "证据索引：[完全支持]、[部分支持]或[不支持/矛盾]。不要添加任何额外内容！"
    "指令：{}"
    "输出：{}"
    "证据：\n {}"
    "你的回答："
    )
    prompt = prompt_zh if lang == "zh" else prompt_en
    keyword = "证据" if lang == "zh" else "Evidence"
    evidence = ""
    for i, doc in enumerate(docs):
        evidence += f"{keyword} {i+1}: {doc['page_content']}\n"
    prompt = prompt.format(query, answer, evidence)
    messages = [("human",prompt)]
    llm = get_ChatOpenAI(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,

            )
    result = llm.invoke(messages)
    pprint(result)
    result_key = result.content
    pattern = re.compile(rf'{keyword} (\d+)：\[(.*?)\]')
    result = pattern.findall(result_key)
    pprint(result_key)
    pprint(result)
    filter_key_ex = "不支持/矛盾" if lang == "zh" else "No support / Contradictory"
    supported_idex = [int(r[0])-1 for r in result if r[-1] != filter_key_ex]
    docs_self_verify = [docs[i] for i in supported_idex]
    return docs_self_verify


async def kb_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                mode: Literal["local_kb", "temp_kb", "search_engine"] = Body("local_kb", description="知识来源"),
                kb_name: str = Body("", description="mode=local_kb时为知识库名称；temp_kb时为临时知识库ID，search_engine时为搜索引擎名称", examples=["samples"]),
                top_k: int = Body(Settings.kb_settings.VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                score_threshold: float = Body(
                    Settings.kb_settings.SCORE_THRESHOLD,
                    description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                    ge=0,
                    le=2,
                ),
                history: List[History] = Body(
                    [],
                    description="历史对话",
                    examples=[[
                        {"role": "user",
                        "content": "我们来玩成语接龙，我先来，生龙活虎"},
                        {"role": "assistant",
                        "content": "虎头虎脑"}]]
                ),
                stream: bool = Body(True, description="流式输出"),
                model: str = Body(get_default_llm(), description="LLM 模型名称。"),
                temperature: float = Body(Settings.model_settings.TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
                max_tokens: Optional[int] = Body(
                    Settings.model_settings.MAX_TOKENS,
                    description="限制LLM生成Token数量，默认None代表模型最大值"
                ),
                prompt_name: str = Body(
                    "default",
                    description="使用的prompt模板名称(在prompt_settings.yaml中配置)"
                ),
                return_direct: bool = Body(False, description="直接返回检索结果，不送入 LLM"),
                request: Request = None,
                ):
    start = time.time()
    if mode == "local_kb":
        kb = KBServiceFactory.get_service_by_name(kb_name)
        if kb is None:
            return BaseResponse(code=404, msg=f"未找到知识库 {kb_name}")
    
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        start = time.time()
        try:
            nonlocal history, prompt_name, max_tokens

            history = [History.from_data(h) for h in history]

            if mode == "local_kb":
                kb = KBServiceFactory.get_service_by_name(kb_name)
                ok, msg = kb.check_embed_model()
                if not ok:
                    raise ValueError(msg)
                docs = await run_in_threadpool(search_docs,
                                                query=query,
                                                knowledge_base_name=kb_name,
                                                top_k=top_k,
                                                score_threshold=score_threshold,
                                                file_name="",
                                                metadata={})
                # source_documents = format_reference(kb_name, docs, api_address(is_public=True))
                doc_source = "kb"
            elif mode == "temp_kb":
                ok, msg = check_embed_model()
                if not ok:
                    raise ValueError(msg)
                docs = await run_in_threadpool(search_temp_docs,
                                                kb_name,
                                                query=query,
                                                top_k=top_k,
                                                score_threshold=score_threshold)
                doc_source = "kb"
                # source_documents = format_reference(kb_name, docs, api_address(is_public=True))
            elif mode == "search_engine":
                result = await run_in_threadpool(search_engine, query, top_k, kb_name)
                docs = [x.dict() for x in result.get("docs", [])]
                # source_documents = [
                #         f"""出处 [{i + 1}] [{d['metadata']['filename']}]({d['metadata']['source']}) \n\n{d['page_content']}\n\n""" 
                #                         for i,d in enumerate(docs)
                #                         ]
                doc_source = "se"
            else:
                logger.warning(f"mode {mode} not supported")
                docs = []
                source_documents = []
            # import rich
            # rich.print(dict(
            #     mode=mode,
            #     query=query,
            #     knowledge_base_name=kb_name,
            #     top_k=top_k,
            #     score_threshold=score_threshold,
            # ))
            # rich.print(docs)
            # # 加入reranker
            docs_original = copy.deepcopy(docs)
            # * -----------------add reranker---------------------------- 
            if Settings.model_settings.USE_RERANKER:
                from chatchat.server.reranker.reranker import reranker_docs
                docs = await reranker_docs(query, docs, top_k)
            if Settings.kb_settings.ADAPTIVE_DOCUMENTS:
                start = time.time()

                docs = await adaptive_docs(
                                        docs, 
                                        model=model,
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                        query=query,
                                        lang="zh"
                                        )
                end = time.time()
                logger.info(f"adaptive_docs time: {end-start}s")
            source_documents = format_reference(kb_name, 
                                                docs, 
                                                api_address(is_public=True), 
                                                doc_source=doc_source)
            # return filtered documents
            docs_filtered = [doc for doc in docs_original if doc not in docs]
            source_documents_filtered = format_reference(kb_name,
                                                docs_filtered,
                                                api_address(is_public=True),
                                                doc_source=doc_source)
            if return_direct:
                yield OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    model=None,
                    object="chat.completion",
                    content="",
                    role="assistant",
                    finish_reason="stop",
                    docs=source_documents,
                ) .model_dump_json()
                return

            callback = AsyncIteratorCallbackHandler()
            callbacks = [callback]

            # Enable langchain-chatchat to support langfuse
            import os
            langfuse_secret_key = os.environ.get('LANGFUSE_SECRET_KEY')
            langfuse_public_key = os.environ.get('LANGFUSE_PUBLIC_KEY')
            langfuse_host = os.environ.get('LANGFUSE_HOST')
            if langfuse_secret_key and langfuse_public_key and langfuse_host :
                from langfuse import Langfuse
                from langfuse.callback import CallbackHandler
                langfuse_handler = CallbackHandler()
                callbacks.append(langfuse_handler)

            if max_tokens in [None, 0]:
                max_tokens = Settings.model_settings.MAX_TOKENS


            # TODO： 视情况使用 API

            context = "\n\n".join([doc["page_content"] for doc in docs])

            if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
                prompt_name = "empty"
            prompt_template = get_prompt_template("rag", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])
            llm = get_ChatOpenAI(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbacks,
            )
            chain = chat_prompt | llm

            # Begin a task that runs in the background.
            task = asyncio.create_task(wrap_done(
                chain.ainvoke({"context": context, "question": query}),
                callback.done),
            )

            if len(source_documents) == 0:  # 没有找到相关文档
                source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

            if stream:
                answer = ""
                async for token in callback.aiter():
                    answer += token
                    ret = OpenAIChatOutput(
                        id=f"chat{uuid.uuid4()}",
                        object="chat.completion.chunk",
                        content=token,
                        role="assistant",
                        model=model,
                        docs=source_documents,
                        filtered_docs=source_documents_filtered,
                    )

                    yield ret.model_dump_json()
                if Settings.kb_settings.SELF_VERIFY_EVIDENCE:
                    start = time.time()
                        
                    docs = await self_verify_evidence(
                                            docs, 
                                            answer=answer,
                                            model=model,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            query=query,
                                            lang="zh"
                                            )
                    end = time.time()
                    logger.info(f"self_verify_evidence time: {end-start}s")
                    source_documents = format_reference(kb_name, 
                                                    docs, 
                                                    api_address(is_public=True), 
                                                    doc_source=doc_source)
                    # return filtered documents
                    docs_filtered = [doc for doc in docs_original if doc not in docs]
                    source_documents_filtered = format_reference(kb_name,
                                                        docs_filtered,
                                                        api_address(is_public=True),
                                                        doc_source=doc_source)
                    ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion.chunk",
                    content="",
                    role="assistant",
                    model=model,
                    docs=source_documents,
                    filtered_docs=source_documents_filtered,
                    )
                    yield ret.model_dump_json()
            else:
                answer = ""
                async for token in callback.aiter():
                    answer += token
                            # filter evidence with self-verify evidence
                if Settings.kb_settings.SELF_VERIFY_EVIDENCE:
                    start = time.time()

                    docs = await self_verify_evidence(
                                            docs, 
                                            answer=answer,
                                            model=model,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            query=query,
                                            lang="zh"
                                            )
                    end = time.time()
                    logger.info(f"self_verify_evidence time: {end-start}s")
                    
                    source_documents = format_reference(kb_name, 
                                                        docs, 
                                                        api_address(is_public=True), 
                                                        doc_source=doc_source)
                    # return filtered documents
                    docs_filtered = [doc for doc in docs_original if doc not in docs]
                    source_documents_filtered = format_reference(kb_name,
                                                        docs_filtered,
                                                        api_address(is_public=True),
                                                        doc_source=doc_source)
                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion",
                    content=answer,
                    role="assistant",
                    model=model,
                    docs=source_documents,
                    filtered_docs=source_documents_filtered,
                )
                end = time.time()
                logger.info(f"chat time: {end-start}s")
                yield ret.model_dump_json()
            await task

        except asyncio.exceptions.CancelledError:
            logger.warning("streaming progress has been interrupted by user.")
            return
        except Exception as e:
            logger.error(f"error in knowledge chat: {e}")
            yield {"data": json.dumps({"error": str(e)})}
            return
    end = time.time()
    logger.info(f"chat time: {end-start}s")
    if stream:
        return EventSourceResponse(knowledge_base_chat_iterator())
    else:
        return await knowledge_base_chat_iterator().__anext__()
