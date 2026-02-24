import random
import pymysql
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import _convert_message_to_dict
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver


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

conn = pymysql.connect(host="localhost", port=3306, user="mysql", password="mysql", database="langgraph", autocommit=True)
saver = PyMySQLSaver(conn)
saver.setup()

def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=saver)

def to_openai_messages(lc_messages):
    return [_convert_message_to_dict(m) for m in lc_messages]

def test_single_turn():
    app = build_graph()
    config = {"configurable": {"thread_id": "single-turn-1"}}
    result = app.invoke(
        {"messages": ["Pick a random item from: pizza, sushi, tacos, ramen, burger."]},
        config=config,
    )
    for msg in to_openai_messages(result["messages"]):
        print(msg)

def test_multi_turn():
    app = build_graph()
    config = {"configurable": {"thread_id": "multi-turn-1"}}
    # turn 1
    app.invoke({"messages": [HumanMessage(content="pick a drink")]}, config=config)
    # turn 2 - checkpointer auto-loads prior messages
    app.invoke({"messages": [HumanMessage(content="now pick a dessert")]}, config=config)
    # turn 3
    result = app.invoke({"messages": [HumanMessage(content="pick a main dish")]}, config=config)
    for msg in to_openai_messages(result["messages"]):
        print(msg)

if __name__ == "__main__":
    test_multi_turn()
