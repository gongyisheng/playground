import random
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import _convert_message_to_dict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode


@tool
def pick_random(choices: list[str]) -> str:
    """Randomly pick one item from a list of choices."""
    return random.choice(choices)


tools = [pick_random]
llm = ChatOpenAI(model="gpt-5-mini").bind_tools(tools)


def agent(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

def init_agent():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", END])
    graph.add_edge("tools", "agent")
    app = graph.compile()
    return app

def to_openai_messages(lc_messages):
    return [_convert_message_to_dict(m) for m in lc_messages]

def test_single_turn():
    app = init_agent()
    result = app.invoke({
        "messages": ["Pick a random item from: pizza, sushi, tacos, ramen, burger."]
    })
    for msg in to_openai_messages(result["messages"]):
        print(msg)

def test_multi_turn():
    app = init_agent()
    # simulate prior multi-turn conversation with tool calls
    history = [
        HumanMessage(content="pick a drink"),
        AIMessage(content="", tool_calls=[{"name": "pick_random", "args": {"choices": ["water", "cola", "juice"]}, "id": "call_1"}]),
        ToolMessage(content="cola", tool_call_id="call_1"),
        AIMessage(content="I picked cola for you!"),
        HumanMessage(content="now pick a dessert"),
        AIMessage(content="", tool_calls=[{"name": "pick_random", "args": {"choices": ["cake", "ice cream", "pie"]}, "id": "call_2"}]),
        ToolMessage(content="pie", tool_call_id="call_2"),
        AIMessage(content="You got pie!"),
        # new turn - agent will continue from here
        HumanMessage(content="pick a main dish"),
    ]
    result = app.invoke({"messages": history})
    for msg in to_openai_messages(result["messages"]):
        print(msg)

if __name__ == "__main__":
    test_multi_turn()
