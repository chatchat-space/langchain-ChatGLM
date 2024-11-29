from __future__ import annotations

import asyncio
from chatchat.settings import Settings
from chatchat.server.agent.tools_factory.tools_registry import (
    BaseToolOutput,
    regist_tool,
    format_context,
)
from chatchat.server.agent.tools_factory.search_internet import search_engine
from chatchat.server.knowledge_base.kb_api import list_kbs
from chatchat.server.knowledge_base.kb_doc_api import search_docs
from chatchat.server.pydantic_v1 import Field
from chatchat.server.utils import get_tool_config

from typing import Dict, List
from typing_extensions import TypedDict
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from chatchat.server.utils import (
    get_prompt_template,
)
import rich

from chatchat.server.utils import get_ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from langchain import globals


# globals.set_debug(True)


template = (
    "Use local knowledgebase from one or more of these:\n{KB_info}\n to get information，Only local data on "
    "this knowledge use this tool. The 'database' should be one of the above [{key}]."
)
KB_info_str = "\n".join(
    [f"{key}: {value}" for key, value in Settings.kb_settings.KB_INFO.items()])
template_knowledge = template.format(KB_info=KB_info_str, key="samples")

# Data model


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
# Data model


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
# Data model


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    knowledge_base: str
    top_k: int
    score_threshold: float
    generation: str
    docs: List[Dict]
    retrieve_retry: int


llm = None


async def search_knowledgebase(query: str, database: str="samples", config: dict={}):
    config = config or get_tool_config("search_local_knowledgebase")
    # docs = search_docs(
    #     query=query,
    #     knowledge_base_name=database,
    #     top_k=config["top_k"],
    #     score_threshold=config["score_threshold"],
    #     file_name="",
    #     metadata={},
    # )
    model_name = "qwen2-instruct"
    global llm
    llm = get_ChatOpenAI(
        model_name=model_name,
        temperature=0.1,
        max_tokens=4096,
        streaming=True,
        local_wrap=False,
        # verbose=True,
    )

    from langgraph.graph import END, StateGraph, START

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query
    workflow.add_node("search_internet", search_internet)

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("search_internet", "grade_documents")
    workflow.add_conditional_edges(
        "search_internet",
        retry_search_internet,
        {
            "search_internet": "search_internet",
            "grade_documents": "grade_documents",
        }
        )

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "search_internet": "search_internet",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    # Compile
    app = workflow.compile()
    g = app.get_graph()
    with open("graph.png", "wb") as fp:
        fp.write(g.draw_mermaid_png())
    from langchain_core.messages import HumanMessage
    inputs = {"question": query, "knowledge_base": database, "retrieve_retry": 0,
              "top_k": config["top_k"], "score_threshold": config["score_threshold"],}
    async for e in app.astream_events(inputs, version="v2"):
        rich.print('\n\n')
        rich.print(e)

    # Final generation
    # print(value["generation"])
    # response = app.invoke(inputs)
    # return response["generation"]
    # return {"knowledge_base": database, "docs": response}


def retry_search_internet(state):
    if not state["docs"]:
        return "search_internet"
    return "grade_documents"


def search_internet(state):
    """
    search documents from internet
    """
    print("---Search Internet---")
    question = state["question"]
    top_k = state["top_k"]
    docs = search_engine(question, top_k)

    return {"docs": [x.dict() for x in docs["docs"]], "retrieve_retry": 2}


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    database = state["knowledge_base"]
    top_k = state["top_k"]
    score_threshold = state["score_threshold"]

    # Retrieval
    docs = search_docs(
        query=question,
        knowledge_base_name=database,
        top_k=top_k,
        score_threshold=score_threshold,
        file_name="",
        metadata={},
    )
    return {"knowledge_base": database, "docs": docs, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    docs = state["docs"]

    prompt_template = PromptTemplate.from_template(
        get_prompt_template("rag", "default"), template_format="jinja2"
    )
    rag_chain = prompt_template | llm | StrOutputParser()
    # RAG generation
    context = "\n\n".join([f"[{i+1}] {x['page_content']}" for i,x in enumerate(docs)])
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"docs": docs, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    docs = state["docs"]

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        Response in json format. here are some examples:
        - ```json\n{{binary_score: "yes"}}```
        - ```json\n{{binary_score: "no"}}```
        """
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    structured_llm_grader = llm.with_structured_output(GradeDocuments, method="json_mode")

    retrieval_grader = grade_prompt | structured_llm_grader

    # Score each doc
    filtered_docs = []
    for d in docs:
        score = retrieval_grader.invoke(
            {"question": question, "document": d["page_content"]}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    if state["retrieve_retry"] <= 1:
        filtered_docs = []
    return {"docs": filtered_docs, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    docs = state["docs"]

    # Prompt
    system = """You are a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
        Respond in Chinese."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    better_question = "介绍一下chatchat"
    return {"docs": docs, "question": better_question, "retrieve_retry": 1}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["docs"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        if state["retrieve_retry"] == 0:
            return "transform_query"
        else:
            return "search_internet"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    docs = state["docs"]
    generation = state["generation"]

    structured_llm_grader = llm.with_structured_output(GradeHallucinations, method="json_mode")

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
        Response in json format. here are some examples:
        - ```json\n{{binary_score: "yes"}}```
        - ```json\n{{binary_score: "no"}}```
        """
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Set of facts: \n\n {docs} \n\n LLM generation: {generation}"),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader
    score = hallucination_grader.invoke(
        {"docs": docs, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        # Prompt
        system = """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
            Response in json format. here are some examples:
            - ```json\n{{binary_score: "yes"}}```
            - ```json\n{{binary_score: "no"}}```
            """
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human",
                 "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        structured_llm_grader = llm.with_structured_output(GradeAnswer, method="json_mode")
        answer_grader = answer_prompt | structured_llm_grader
        score = answer_grader.invoke(
            {"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


@regist_tool(description=template_knowledge, title="本地知识库LangGraph")
def graph_search_local_knowledgebase(
    database: str = Field(
        description="Database for Knowledge Search",
        choices=[kb.kb_name for kb in list_kbs().data],
    ),
    query: str = Field(description="Query for Knowledge Search"),
):
    """temp docstr to avoid langchain error"""
    tool_config = get_tool_config("search_local_knowledgebase")
    ret = search_knowledgebase(
        query=query, database=database, config=tool_config)
    return BaseToolOutput(ret)


asyncio.run(search_knowledgebase("介绍一下chatchat", "samples", {}))
