"""
Generate instances for instructions using GPT-3 API.
This script processes machine-generated instructions and creates example instances for each.
"""
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import tqdm

from utils import init_openai_client, evoke_batch_requests, DataWriter
from templates import INSTANCE_GENERATE_TEMPLATE_CLF, INSTANCE_GENERATE_TEMPLATE_GEN


@dataclass
class Config:
    """Configuration for instance generation task."""

    input_path: str = "outputs/machine_generated_instructions.jsonl"
    output_path: str = "outputs/machine_generated_instances.jsonl"
    clf_types_path: str = "outputs/is_clf_or_not_results.jsonl"

    model_name: str = "Qwen/Qwen3-14B"
    batch_size: int = 8
    base_url: str = "https://vllm.yellowday.day/v1"

    generation_tasks_only: bool = False
    classification_tasks_only: bool = False

    max_token: int = 512
    temperature: float = 0.0
    top_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 1.5

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.generation_tasks_only and self.classification_tasks_only:
            raise ValueError(
                "Cannot specify both generation_tasks_only and classification_tasks_only"
            )

        if not Path(self.input_path).exists():
            raise ValueError(f"Input file not found: {self.input_path}")

        if not Path(self.clf_types_path).exists():
            raise ValueError(f"Classification types file not found: {self.clf_types_path}")

        # Create output directory if it doesn't exist
        output_dir = Path(self.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)


class InstanceGenerator:
    """Generates instances for instructions."""

    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.input_path = Path(config.input_path)
        self.output_path = Path(config.output_path)
        self.clf_types_path = Path(config.clf_types_path)

        self.client = init_openai_client(base_url=config.base_url)

        self.tasks: List[Dict] = []
        self.task_clf_types: Dict[str, bool] = {}

    def load_tasks(self) -> None:
        """Load tasks from input file."""
        with open(self.input_path) as f:
            lines = f.readlines()

            for line in lines:
                data = json.loads(line)
                self.tasks.append(data)

        print(f"Loaded {len(self.tasks)} tasks")

    def load_classification_types(self) -> None:
        """Load classification types for tasks."""
        with open(self.clf_types_path) as f:
            for line in f:
                data = json.loads(line)
                instruction = data["instruction"]
                is_clf = data["is_classification"].strip().lower() == "yes"
                self.task_clf_types[instruction] = is_clf

        print(f"Loaded classification types for {len(self.task_clf_types)} tasks")

    def filter_tasks(self) -> None:
        """Filter tasks based on configuration."""
        if self.config.classification_tasks_only:
            self.tasks = [
                task for task in self.tasks
                if self.task_clf_types.get(task["instruction"], False)
            ]
            print(f"Filtered to {len(self.tasks)} classification tasks")

        elif self.config.generation_tasks_only:
            self.tasks = [
                task for task in self.tasks
                if not self.task_clf_types.get(task["instruction"], False)
            ]
            print(f"Filtered to {len(self.tasks)} generation tasks")

    def create_prompt(self, task: Dict) -> str:
        """Create prompt for a task based on its type."""
        instruction = task["instruction"].strip()
        is_clf = self.task_clf_types.get(task["instruction"], False)

        if is_clf:
            return INSTANCE_GENERATE_TEMPLATE_CLF.format(task=instruction)
        else:
            return INSTANCE_GENERATE_TEMPLATE_GEN.format(task=instruction)

    async def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of tasks to generate instances."""
        # Create prompts for batch
        prompts = [self.create_prompt(task) for task in batch]

        # Convert prompts to messages format for chat API
        messages_list = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        # Make API requests using the async API
        results = await evoke_batch_requests(
            client=self.client,
            messages_list=messages_list,
            model=self.config.model_name,
            max_tokens=self.config.max_token,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
        )

        # Process results
        outputs = []
        for task, result in zip(batch, results):
            if result is not None and result.choices:
                task["raw_instances"] = result.choices[0].message.content
            else:
                task["raw_instances"] = ""

            outputs.append(task)

        return outputs

    async def run(self):
        """Run the instance generation process."""
        print(f"Loading tasks from {self.input_path}")
        self.load_tasks()

        print(f"Loading classification types from {self.clf_types_path}")
        self.load_classification_types()

        self.filter_tasks()

        print(f"Processing {len(self.tasks)} tasks")

        with DataWriter(str(self.output_path), mode="w") as writer:
            progress_bar = tqdm.tqdm(total=len(self.tasks), desc="Generating instances")

            for batch_idx in range(0, len(self.tasks), self.config.batch_size):
                batch = self.tasks[batch_idx:batch_idx + self.config.batch_size]
                outputs = await self.process_batch(batch)

                writer.write_batch(outputs)
                progress_bar.update(len(batch))

            progress_bar.close()

        print(f"\nGeneration complete. Output saved to: {self.output_path}")


async def main():
    config = Config()
    generator = InstanceGenerator(config)
    await generator.run()

if __name__ == "__main__":
    asyncio.run(main())
