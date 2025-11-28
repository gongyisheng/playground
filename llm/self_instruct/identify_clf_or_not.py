"""
Identify whether instructions are classification tasks or not using GPT-3.

This module processes a batch of machine-generated instructions and determines
whether each instruction represents a classification task with finite output labels.
"""
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List
import tqdm

from templates import CLF_TASK_TEMPLATE
from utils import init_openai_client, evoke_batch_requests, DataWriter


@dataclass
class Config:
    """Configuration for classification identification task."""

    input_path: str = "outputs/machine_generated_instructions.jsonl"
    output_path: str = "outputs/is_clf_or_not_results.jsonl"

    model_name: str = "Qwen/Qwen3-14B"
    batch_size: int = 8
    base_url: str = "https://vllm.yellowday.day/v1"

    max_tokens: int = 16
    temperature: float = 0.3
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate input file exists
        if not Path(self.input_path).exists():
            raise ValueError(f"Input file not found: {self.input_path}")

        # Create output directory if it doesn't exist
        output_dir = Path(self.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)


class ClassificationIdentifier:
    """Identifies whether instructions are classification tasks."""

    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.input_path = Path(config.input_path)
        self.output_path = Path(config.output_path)
        self.client = init_openai_client(base_url=config.base_url)

    def load_instructions(self) -> List[str]:
        """Load instructions from input file."""
        with open(self.input_path) as f:
            lines = f.readlines()

        return lines

    def create_prompt(self, instruction: str) -> str:
        """Create a prompt for the given instruction."""
        return CLF_TASK_TEMPLATE.format(task=instruction.strip())

    async def process_batch(
        self,
        batch: List[dict]
    ) -> List[dict]:
        """Process a batch of instructions."""
        # Make API requests for instructions
        prompts = [self.create_prompt(d["instruction"]) for d in batch]

        # Convert prompts to messages format
        messages_list = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        results = await evoke_batch_requests(
            self.client,
            messages_list,
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}} # for qwen3 models, disable think
        )

        # Process results
        output_data = []
        for i, data in enumerate(batch):
            is_classification = ""
            if results[i] is not None and results[i].choices:
                is_classification = results[i].choices[0].message.content.strip('.')
                if is_classification not in ["Yes", "No"]:
                    is_classification = ""
            
            output_data.append({
                "instruction": data["instruction"],
                "is_classification": is_classification
            })

        return output_data

    async def run(self):
        """Run the classification identification process."""
        print(f"Loading instructions from {self.input_path}")
        lines = self.load_instructions()
        print(f"Processing {len(lines)} instructions")

        progress_bar = tqdm.tqdm(total=len(lines), desc="Processing instructions")

        with DataWriter(str(self.output_path), mode="w") as writer:
            for batch_idx in range(0, len(lines), self.config.batch_size):
                batch_lines = lines[batch_idx:batch_idx + self.config.batch_size]
                batch = [json.loads(line) for line in batch_lines]

                output_data = await self.process_batch(batch)

                for data in output_data:
                    writer.write(data)

                progress_bar.update(len(batch))

        progress_bar.close()
        print(f"\nResults saved to {self.output_path}")


async def main():
    config = Config()
    identifier = ClassificationIdentifier(config)
    await identifier.run()

if __name__ == "__main__":
    asyncio.run(main())
