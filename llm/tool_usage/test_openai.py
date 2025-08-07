#!/usr/bin/env python3
"""
Example of OpenAI function calling with math tools.
"""

import json
from openai import OpenAI

def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

def subtract_numbers(a: float, b: float) -> float:
    """Subtract second number from first number."""
    return a - b

# Tool definitions for OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "subtract_numbers",
            "description": "Subtract second number from first number",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

def execute_function(function_name: str, arguments: dict):
    """Execute the requested function with given arguments."""
    if function_name == "add_numbers":
        return add_numbers(arguments["a"], arguments["b"])
    elif function_name == "subtract_numbers":
        return subtract_numbers(arguments["a"], arguments["b"])
    else:
        raise ValueError(f"Unknown function: {function_name}")

def chat_with_tools(client: OpenAI, user_message: str):
    """Chat with OpenAI using function calling."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can perform math operations."},
        {"role": "user", "content": user_message}
    ]
    
    # First API call
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    messages.append(response_message)
    
    # Check if the model wants to call functions
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"Calling function: {function_name} with args: {function_args}")
            
            # Execute the function
            function_result = execute_function(function_name, function_args)
            
            # Add function result to messages
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(function_result)
            })
        
        # Second API call to get final response
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        return final_response.choices[0].message.content
    else:
        return response_message.content

def main():
    # Initialize OpenAI client
    client = OpenAI()  # Uses OPENAI_API_KEY environment variable
    
    # Example usage
    examples = [
        "What is 15 + 27?",
        "Calculate 100 - 43",
        "I need to add 25.5 and 14.3, then subtract 10 from the result"
    ]
    
    for example in examples:
        print(f"\nUser: {example}")
        response = chat_with_tools(client, example)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()