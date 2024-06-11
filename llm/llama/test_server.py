from openai import OpenAI

host = "https://ai.freedl.cc/api"

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the purpose of life?"},
]

OPENAI_CLIENT = OpenAI(base_url=host, api_key="sk-123")
stream = OPENAI_CLIENT.chat.completions.create(
    model="llama3",
    messages=conversation,
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    print(content)
