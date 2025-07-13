#!/usr/bin/env python3
"""
Example of LangChain tool usage with OpenAI models for math operations.
"""

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract second number from first number."""
    return a - b

def create_math_agent():
    """Create a LangChain agent with math tools."""
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Define tools
    tools = [add_numbers, subtract_numbers]
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can perform math operations using the provided tools."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    return agent_executor

def run_example1():
    """Run example queries with the math agent."""
    agent_executor = create_math_agent()
    
    examples = [
        "What is 15 + 27?",
        "Calculate 100 - 43",
        "I need to add 25.5 and 14.3, then subtract 10 from the result",
        "What's the result of 50 + 30 - 20?"
    ]
    
    for example in examples:
        print(f"\n{'='*50}")
        print(f"Query: {example}")
        print('='*50)
        
        result = agent_executor.invoke({"input": example})
        print(f"Final Answer: {result['output']}")

def run_example2():
    """Interactive chat with tools."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    tools = [add_numbers, subtract_numbers]
    
    # Bind tools to the model
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [
        ("system", "You are a helpful assistant that can perform math operations."),
        ("user", "What is 25 + 17?")
    ]
    
    print("\n" + "="*50)
    print("Chat with Tools Example")
    print("="*50)
    
    # Get response with tool calls
    response = llm_with_tools.invoke(messages)
    print(f"Assistant response: {response}")
    
    # Check if there are tool calls
    if response.tool_calls:
        print(f"Tool calls made: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            print(f"Tool: {tool_call['name']}, Args: {tool_call['args']}")

if __name__ == "__main__":
    print("LangChain Tool Usage Examples")
    
    # Example 1: Agent-based approach
    run_example1()
    
    # Example 2: Chat with tools
    run_example2()