#!/usr/bin/env python3
"""
Example of OpenAI function calling with math tools.
"""

import json
from openai import OpenAI

def calculate_magic_score(a: float, b: float) -> float:
    """Calculate the magic score of two numbers (defined as their sum)."""
    return (a + b) / 2

# Tool definitions for OpenAI
tools = [
    {
        "type": "function",
        "name": "magic_score",
        "description": "Calculate the magic score of two numbers (defined as their sum)",
        "parameters": {
            "type": "object",
            "properties": {
                "number1": {
                    "type": "number",
                    "description": "The first number for the magic score calculation"
                },
                "number2": {
                    "type": "number", 
                    "description": "The second number for the magic score calculation"
                }
            },
            "required": ["number1", "number2"]
        }
    }
]

def execute_function(function_name: str, arguments: str):
    """Execute the requested function with given arguments."""
    if function_name == "magic_score":
        # Parse the arguments from JSON string
        args = json.loads(arguments)
        return calculate_magic_score(args["number1"], args["number2"])
    else:
        raise ValueError(f"Unknown function: {function_name}")

def chat_with_tools(client: OpenAI, user_message: str):
    """Chat with OpenAI using function calling."""
    # Create a running input list
    input_list = [
        {"role": "user", "content": user_message}
    ]
    
    # First API call
    response = client.responses.create(
        model="gpt-5-mini",
        input=input_list,
        reasoning={
            "effort": "low", # minimal, low, medium, and high
            "summary": "detailed" # auto, concise, or detailed
        },
        tools=tools,
        tool_choice="auto"
    )

    print(response.output)
    
    # Save function call outputs for subsequent requests
    input_list += response.output
    
    # Process any function calls
    for item in response.output:
        if item.type == "function_call" and item.name == "magic_score":
            print(f"Calling function: {item.name} with args: {item.arguments}, call_id: {item.call_id}")
            
            # Execute the function
            result = execute_function(item.name, item.arguments)
            
            # Add function result to input list
            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({
                    "magic_score": result
                })
            })

            print(input_list)

            response = client.responses.create(
                model="gpt-5-mini",
                input=input_list,
                reasoning={
                    "effort": "low", # minimal, low, medium, and high
                    "summary": "detailed" # auto, concise, or detailed
                },
                tools=tools
            )

            print(response.output)

    return response.output_text

def main():
    # Initialize OpenAI client
    client = OpenAI()  # Uses OPENAI_API_KEY environment variable
    
    # Example usage
    examples = [
        "You can only call one tool at a time. question: assume the magic score of 15 and 27 is A, please calculate magic score of A and 30. "
    ]
    
    for example in examples:
        print(f"\nUser: {example}")
        response = chat_with_tools(client, example)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()