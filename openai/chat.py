import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def chat():
    for i in range(50):
        prompt = input("You: ")
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
        )
        print(f"ChatGPT: {completion.choices[0].message}")

if __name__ == "__main__":
    chat()