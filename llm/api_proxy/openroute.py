import os
from openai import OpenAI

class Global:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTE_API_KEY"),
    )

def send_to_openroute(model, message):
    completion = Global.client.chat.completions.create(
        model=model,
        messages=message,
    )
    dict_resp = completion.to_dict()
    if "error" in dict_resp:
        raise Exception(f"OpenRoute API error: {dict_resp['error']}")
    else:
        return dict_resp["choices"][0]["message"]["content"]

if __name__ == "__main__":
    message = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What is the capital of France"
        },
    ]
    print(send_to_openroute("google/gemini-2.0-flash-001", message))
