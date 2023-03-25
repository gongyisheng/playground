import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def chat():
    for i in range(50):
        prompt = input("You:\n")
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
        )
        print(f"ChatGPT:\n{str(completion.choices[0].message['content'])}")

if __name__ == "__main__":
    chat()