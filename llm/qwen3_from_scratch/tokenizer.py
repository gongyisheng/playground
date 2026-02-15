import json
from pathlib import Path

from jinja2 import Template
from tokenizers import Tokenizer


class Qwen3Tokenizer:

    def __init__(self, tokenizer: Tokenizer, chat_template: Template):
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    @classmethod
    def from_model_dir(cls, path: str | Path) -> "Qwen3Tokenizer":
        path = Path(path)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        with open(path / "tokenizer_config.json") as f:
            template_str = json.load(f)["chat_template"]
        chat_template = Template(template_str)
        return cls(tokenizer, chat_template)

    def apply_chat_template(self, messages: list[dict], enable_thinking: bool = False) -> str:
        return self.chat_template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


if __name__ == "__main__":
    tokenizer = Qwen3Tokenizer.from_model_dir("checkpoint/Qwen3-0.6B")

    # single turn
    messages = [{"role": "user", "content": "What is 2+2?"}]
    formatted = tokenizer.apply_chat_template(messages)
    print("=== Single Turn ===")
    print(formatted)
    print("Token IDs:", tokenizer.encode(formatted)[:20], "...")

    # multi turn
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    formatted = tokenizer.apply_chat_template(messages)
    print("\n=== Multi Turn ===")
    print(formatted)
