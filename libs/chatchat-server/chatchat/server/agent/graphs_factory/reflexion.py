import datetime
from typing import List, Optional, Literal, Any

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langchain_openai.chat_models import ChatOpenAI

from chatchat.server.utils import get_graph_memory, build_logger
from chatchat.server.agent.tools_factory import search_internet
from .graphs_registry import regist_graph, InputHandler, EventHandler, State, async_history_manager

logger = build_logger()

MAX_ITERATIONS = 2
num_iterations = 0


class Reflection(BaseModel):
    missing: Optional[str] = Field(
        None,
        description="Critique of what is missing."
    )
    superfluous: Optional[str] = Field(
        None,
        description="Critique of what is superfluous"
    )


class AnswerQuestion(BaseModel):
    """Answer the question.
    Provide an answer, reflection, and then follow up with search queries to improve the answer."""

    answer: Optional[str] = Field(
        None,
        description="~250 word detailed answer to the question."
    )
    reflection: Optional[Reflection] = Field(
        None,
        description="Your reflection on the initial answer."
    )
    search_queries: Optional[List[str]] = Field(
        None,
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


# Extend the initial answer schema to include references.
# Forcing citation in the model encourages grounded responses
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question. Provide an answer, reflection,

    cite your reflection with references, and finally
    add search queries to improve the answer."""

    references: Optional[List[str]] = Field(
        None,
        description="Citations motivating your updated answer."
    )


class ReflexionState(State):
    """
    1. question: 用户的问题.
    2. answer: 答案.
    3. reflection: Your reflection on the initial answer.
    4. search_queries: 1-3 search queries for researching improvements to address the critique of your current answer.
    5. references: Citations motivating your updated answer.
    """
    question: Optional[BaseMessage]
    answer: Optional[str]
    reflection: Optional[Reflection]
    search_queries: Optional[List[str]]
    references: Optional[List[str]]


def extract_messages(messages):
    result = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            result.append(message)
        elif isinstance(message, AIMessage):
            result.append(message)
            break
    return result[::-1]


class ReflexionEventHandler(EventHandler):
    def __init__(self):
        pass

    def handle_event(self, node: str, events: ReflexionState) -> Any:
        """
        event example:
        """
        # import rich
        if node == "revise":
            revise_answer = ReviseAnswer(
                question=events["question"],
                answer=events["answer"],
                reflection=events["reflection"],
                search_queries=events["search_queries"],
                references=events["references"]
            )
            return revise_answer
        elif node == "draft":
            answer_question = AnswerQuestion(
                question=events["question"],
                answer=events["answer"],
                reflection=events["reflection"],
                search_queries=events["search_queries"],
            )
            return answer_question
        elif node == "function_call" or node == "function_call_loop":
            function_call = extract_messages(events["messages"])
            return function_call
        return None


@regist_graph(name="reflexion",
              input_handler=InputHandler,
              event_handler=ReflexionEventHandler)
def reflexion(llm: ChatOpenAI, tools: list[BaseTool], history_len: int) -> CompiledGraph:
    """
    description: https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/
    """
    # import rich

    if not isinstance(llm, ChatOpenAI):
        raise TypeError("llm must be an instance of ChatOpenAI")
    if not all(isinstance(tool, BaseTool) for tool in tools):
        raise TypeError("All items in tools must be instances of BaseTool")

    memory = get_graph_memory()

    async def history_manager(state: ReflexionState) -> ReflexionState:
        state = await async_history_manager(state=state, history_len=history_len)
        state["question"] = state["messages"][-1]
        return state

    # tool_node
    async def function_call(state: ReflexionState) -> ReflexionState:
        if state["search_queries"] is None or len(state["search_queries"]) == 0:
            state["search_queries"] = []
            for history_message in state["history"]:
                state["search_queries"].append(history_message)

        tool_node_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an excellent robot, good at invoking multiple functions (or the same function multiple times) to help users solve problems.
        Current time: {time}

        The following is a list of problems. Please call the appropriate tools in order and return the results:
        {search_queries}
        """
                ),
            ]
        ).partial(time=lambda: datetime.datetime.now().isoformat())
        # 担心用户没有选择任何 tools, 为保证效果, 强行追加一个 duckduckgo 搜索, 如开发者不需要可注释此行代码
        tools.append(search_internet)
        llm_with_tools = tool_node_template | llm.bind_tools(tools)
        func_call = llm_with_tools.invoke(state)
        state["messages"] = [func_call]
        state["history"].append(func_call)
        return state

    async def process_func_call_history(state: ReflexionState) -> ReflexionState:
        # ToolNode 默认只将结果追加到 messages 队列中, 所以需要手动在 history 中追加 ToolMessage 结果

        # 找到从末尾开始的第一个非ToolMessage的索引
        index = len(state["messages"]) - 1
        while index >= 0 and isinstance(state["messages"][index], ToolMessage):
            index -= 1

        # index现在指向最后一个非ToolMessage或者-1（如果所有消息都是ToolMessage）
        # 将index后面的所有ToolMessage追加到history中
        # 注意这里使用index + 1来开始切片，因为index指向的是最后一个非ToolMessage
        tool_messages_to_add = state["messages"][index + 1:]
        if tool_messages_to_add:
            state["history"].extend(tool_messages_to_add)
        return state

    tool_node = ToolNode(tools=tools)

    function_call_graph_builder = StateGraph(ReflexionState)
    function_call_graph_builder.add_node("function_call", function_call)
    function_call_graph_builder.add_node("tools", tool_node)
    function_call_graph_builder.add_node("process_func_call_history", process_func_call_history)
    function_call_graph_builder.set_entry_point("function_call")
    function_call_graph_builder.add_edge("function_call", "tools")
    function_call_graph_builder.add_edge("tools", "process_func_call_history")
    function_call_graph_builder.add_edge("process_func_call_history", END)
    function_call_graph = function_call_graph_builder.compile(checkpointer=memory)

    # Initial responder
    actor_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are an expert researcher.
    Current time: {time}
    User's question: {question}

    Actions taken so far:
    {history}

    Initial Answer:
    {answer}

    Reflection on the initial answer:
    {reflection}

    Steps:
    1. {first_instruction}
    2. Reflect and critique your answer based on the user's question and the actions taken so far. Be severe to maximize improvement.
    3. List information that needs further collection. Provide solutions only, no execution needed.
    4. Respond using the {function_name} struct to provide an updated response.

    Supporting References:
    {references}
    """,
            ),
        ]
    ).partial(
        time=lambda: datetime.datetime.now().isoformat(),
    )

    async def process_reflexion_result(state: ReflexionState, result) -> ReflexionState:
        if hasattr(result, 'reflection') and result.reflection is not None:
            if not isinstance(result.reflection, dict):
                result.reflection = result.reflection.dict()

        if hasattr(result, 'search_queries') and result.search_queries is not None:
            if not isinstance(result.search_queries, list):
                result.search_queries = list(result.search_queries)

        state["messages"].append(AIMessage(content=str(result)))
        state["history"].append(AIMessage(content=str(result)))
        state["answer"] = result.answer
        state["reflection"] = result.reflection
        state["search_queries"] = result.search_queries
        if hasattr(result, 'references'):
            state["references"] = result.references
        return state

    async def initial(state: ReflexionState) -> ReflexionState:
        initial_answer_chain = actor_prompt_template.partial(
            first_instruction="Provide correct and concise answers.",
            function_name=AnswerQuestion.__name__,
        ) | llm.with_structured_output(AnswerQuestion)

        initial_result = initial_answer_chain.invoke(state)
        return await process_reflexion_result(state, initial_result)

    # Revision
    async def revision(state: ReflexionState) -> ReflexionState:
        global num_iterations
        num_iterations += 1

        revise_instructions = """Revise your previous answer using the new information.
        - You should use the previous critique to add important information to your answer.
            - You MUST include numerical citations in your revised answer to ensure it can be verified.
            - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
                - https://example.com
                - https://example.com
        - You should use the previous critique to remove superfluous information from your answer.
        """
        revision_chain = actor_prompt_template.partial(
            first_instruction=revise_instructions,
            function_name=ReviseAnswer.__name__,
        ) | llm.with_structured_output(ReviseAnswer)

        revised_result = revision_chain.invoke(state)
        return await process_reflexion_result(state, revised_result)

    async def event_loop(state: ReflexionState) -> Literal["function_call_loop", "__end__"]:
        # in our case, we'll just stop after N plans
        global MAX_ITERATIONS, num_iterations
        if num_iterations >= MAX_ITERATIONS:
            return END
        return "function_call_loop"

    # Construct Graph
    builder = StateGraph(ReflexionState)

    builder.add_node("history_manager", history_manager)
    builder.add_node("draft", initial)
    builder.add_node("revise", revision)
    builder.add_node("function_call", function_call_graph)
    builder.add_node("function_call_loop", function_call_graph)

    builder.set_entry_point("history_manager")
    builder.add_edge("history_manager", "function_call")
    builder.add_edge("function_call", "draft")
    # draft -> execute_tools
    builder.add_edge("draft", "function_call_loop")
    # execute_tools -> revise
    builder.add_edge("function_call_loop", "revise")
    # revise -> execute_tools OR end
    builder.add_conditional_edges("revise", event_loop)

    graph = builder.compile(checkpointer=memory)

    return graph
