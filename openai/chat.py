import os
import openai
from prompt import prompt

openai.api_key = os.getenv("OPENAI_API_KEY")

def _chat_round_trip(input_text):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": input_text}]
        )
    return str(completion.choices[0].message['content'])

def chat(limit=50):
    for i in range(limit):
        prompt = input("You:\n")
        output = _chat_round_trip(prompt)
        print(f"ChatGPT:\n{output}")

def chat_with_prompt(limit=50):
    supported_prompts = prompt.keys()
    for i in range(limit):
        print("Supported prompts: " + ", ".join(supported_prompts))
        prompt_type = input("Which prompt do you want to use? \n")
        if prompt_type not in supported_prompts:
            print("Invalid prompt.")
            continue
        customized_request = input("Please input your request:\n")
        prompt_text = prompt[prompt_type] % customized_request
        output = _chat_round_trip(prompt_text)
        print(f"ChatGPT:\n{output}")

if __name__ == "__main__":
    #chat()
    chat_with_prompt()