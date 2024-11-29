from typing import List, Any, Union, Optional, Literal

import rich
from langchain import hub
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent

from chatchat.server.utils import get_graph_memory, build_logger
from .graphs_registry import regist_graph, InputHandler, EventHandler, State, async_history_manager

logger = build_logger()


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
                    "If you need to further use tools to get the answer, use Plan."
    )


class PlanStepExecuteResult(BaseModel):
    step: str
    result: str


class PlanExecute(State):
    """
    plan_and_execute 的核心 state, 其中:
    1. plan
    2. past_steps
    3. response
    """
    plan: Optional[Plan]
    past_steps: Optional[List[PlanStepExecuteResult]]
    response: Optional[Response]


class PlanAndExecuteEventHandler(EventHandler):
    def __init__(self):
        pass

    def handle_event(self, node: str, events: PlanExecute) -> Any:
        '''
        event example:
        {'planner': {'messages': [HumanMessage(content='what is the hometown of the 2024 Australia open winner?', id='09da28f6-56af-4362-bd8f-f31cd98a103d')], 'history': [HumanMessage(content='what is the hometown of the 2024 Australia open winner?', id='09da28f6-56af-4362-bd8f-f31cd98a103d')], 'plan':
        Plan(steps=['Identify the winner of the 2024 Australian Open.', 'Determine the hometown of the identified winner.'])}}

        {'agent': {'messages': [HumanMessage(content='what is the hometown of the 2024 Australia open winner?', id='09da28f6-56af-4362-bd8f-f31cd98a103d')], 'history': [HumanMessage(content='what is the hometown of the 2024 Australia open winner?', id='09da28f6-56af-4362-bd8f-f31cd98a103d')], 'plan':
        Plan(steps=['Identify the winner of the 2024 Australian Open.', 'Determine the hometown of the identified winner.']), 'past_steps': [PlanStepExecuteResult(step='Identify the winner of the 2024 Australian Open.', result="The winner of the 2024 Australian Open men's singles is Jannik Sinner.")]}}

        {'replan': {'messages': [HumanMessage(content='what is the hometown of the 2024 Australia open winner?', id='09da28f6-56af-4362-bd8f-f31cd98a103d')], 'history': [HumanMessage(content='what is the hometown of the 2024 Australia open winner?', id='09da28f6-56af-4362-bd8f-f31cd98a103d')], 'plan':
        Plan(steps=["Determine the hometown of Jannik Sinner, the winner of the 2024 Australian Open men's singles."]), 'past_steps': [PlanStepExecuteResult(step='Identify the winner of the 2024 Australian Open.', result="The winner of the 2024 Australian Open men's singles is Jannik Sinner.")]}}

        {'agent': {'messages': [HumanMessage(content='what is the hometown of the 2024 Australia open winner?', id='09da28f6-56af-4362-bd8f-f31cd98a103d')], 'history': [HumanMessage(content='what is the hometown of the 2024 Australia open winner?', id='09da28f6-56af-4362-bd8f-f31cd98a103d')], 'plan':
        Plan(steps=["Determine the hometown of Jannik Sinner, the winner of the 2024 Australian Open men's singles."]), 'past_steps': [PlanStepExecuteResult(step='Identify the winner of the 2024 Australian Open.', result="The winner of the 2024 Australian Open men's singles is Jannik Sinner."),
        PlanStepExecuteResult(step="Determine the hometown of Jannik Sinner, the winner of the 2024 Australian Open men's singles.", result="Jannik Sinner, the winner of the 2024 Australian Open men's singles, is from San Candido, Italy.")]}}

        {'replan': {'messages': "The hometown of Jannik Sinner, the winner of the 2024 Australian Open men's singles, is San Candido, Italy.", 'history': [HumanMessage(content='what is the hometown of the 2024 Australia open winner?', id='09da28f6-56af-4362-bd8f-f31cd98a103d'), AIMessage(content="The
        hometown of Jannik Sinner, the winner of the 2024 Australian Open men's singles, is San Candido, Italy.")], 'plan': Plan(steps=["Determine the hometown of Jannik Sinner, the winner of the 2024 Australian Open men's singles."]), 'past_steps': [PlanStepExecuteResult(step='Identify the winner of the 2024
        Australian Open.', result="The winner of the 2024 Australian Open men's singles is Jannik Sinner."), PlanStepExecuteResult(step="Determine the hometown of Jannik Sinner, the winner of the 2024 Australian Open men's singles.", result="Jannik Sinner, the winner of the 2024 Australian Open men's singles, is
        from San Candido, Italy.")], 'response': "The hometown of Jannik Sinner, the winner of the 2024 Australian Open men's singles, is San Candido, Italy."}}
        '''
        handler_map = {
            "planner": self.handle_planner,
            "agent": self.handle_agent,
            "replan": self.handle_replan
        }

        handler = handler_map.get(node)
        if handler:
            return handler(events)
        else:
            raise ValueError(f"Unsupported plan_and_execute node type: {node}")

    @staticmethod
    def handle_planner(event_data: PlanExecute) -> Plan:
        return event_data["plan"]

    @staticmethod
    def handle_agent(event_data: PlanExecute) -> List[PlanStepExecuteResult]:
        return event_data["past_steps"]

    @staticmethod
    def handle_replan(event_data: PlanExecute) -> Union[Plan, Response]:
        if event_data["response"] is None:
            return event_data["plan"]
        else:
            return event_data["response"]


@regist_graph(name="plan_and_execute",
              input_handler=InputHandler,
              event_handler=PlanAndExecuteEventHandler)
def plan_and_execute(llm: ChatOpenAI, tools: list[BaseTool], history_len: int) -> CompiledGraph:
    """
    description: https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/
    """
    if not isinstance(llm, ChatOpenAI):
        raise TypeError("llm must be an instance of ChatOpenAI")
    if not all(isinstance(tool, BaseTool) for tool in tools):
        raise TypeError("All items in tools must be instances of BaseTool")

    memory = get_graph_memory()

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("wfh/react-agent-executor")
    # prompt.pretty_print()

    # Choose the LLM that will drive the agent
    agent_executor = create_react_agent(llm, tools, messages_modifier=prompt)

    async def history_manager(state: PlanExecute) -> State:
        state = await async_history_manager(state, history_len)
        return state

    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    planner = planner_prompt | llm.with_structured_output(Plan)

    async def plan_step(state: PlanExecute) -> PlanExecute:
        plan_steps = await planner.ainvoke({"messages": state["history"]})
        plan = Plan(steps=plan_steps.steps)
        state["plan"] = plan
        return state

    async def execute_step(state: PlanExecute) -> PlanExecute:
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan.steps))
        task = plan.steps[0]
        task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
        agent_response = await agent_executor.ainvoke(
            {"messages": [("user", task_formatted)]}
        )
        plan_step_execute_result = PlanStepExecuteResult(step=task, result=agent_response["messages"][-1].content)
        if state["past_steps"] is None:
            state["past_steps"] = []
        state["past_steps"].append(plan_step_execute_result)
        return state

    replanner_prompt = ChatPromptTemplate.from_template(
        """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    Your objective was this:
    {history}

    Your original plan was this:
    {plan}

    You have currently done the follow steps:
    {past_steps}

    Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
    )
    replanner = replanner_prompt | llm.with_structured_output(Act)

    async def replan_step(state: PlanExecute) -> PlanExecute:
        output = await replanner.ainvoke(state)
        # 检查 output.action 是否是 Response 类型
        if isinstance(output.action, Response):
            state["response"] = Response(response=output.action.response)
            state["messages"] = [AIMessage(content=output.action.response)]
        # 检查 output.action 是否是 Plan 类型
        elif isinstance(output.action, Plan):
            state["plan"] = Plan(steps=output.action.steps)
        else:
            raise ValueError("Unexpected action type in replan_step output")
        return state

    def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
        if state["response"]:
            return "__end__"
        else:
            return "agent"

    graph_builder = StateGraph(PlanExecute)

    graph_builder.add_node("history_manager", history_manager)

    # Add the plan node
    graph_builder.add_node("planner", plan_step)

    # Add the execution step
    graph_builder.add_node("agent", execute_step)

    # Add a replan node
    graph_builder.add_node("replan", replan_step)

    graph_builder.add_edge(START, "history_manager")

    graph_builder.add_edge("history_manager", "planner")

    # From plan we go to agent
    graph_builder.add_edge("planner", "agent")

    # From agent, we replan
    graph_builder.add_edge("agent", "replan")

    graph_builder.add_conditional_edges(
        "replan",
        # Next, we pass in the function that will determine which node is called next.
        should_end,
    )

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    graph = graph_builder.compile(checkpointer=memory)

    return graph
