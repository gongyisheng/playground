import random
from langchain_openai import ChatOpenAI
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


graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, ["tools", END])
graph.add_edge("tools", "agent")
app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "messages": ["Pick a random item from: pizza, sushi, tacos, ramen, burger."]
    })
    for msg in result["messages"]:
        print(msg)
