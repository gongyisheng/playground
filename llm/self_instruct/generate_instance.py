"""
Generate instances for instructions using GPT-3 API.
This script processes machine-generated instructions and creates example instances for each.
"""
import os
import json
import random
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, List, Dict
import tqdm

from utils import init_openai_client, evoke_batch_requests
from templates import output_first_template_for_clf, input_first_template_for_gen


random.seed(42)


@dataclass
class GenerationConfig:
    """Configuration for instance generation."""

    # Required parameters
    batch_dir: str

    # File paths
    input_file: str = "machine_generated_instructions.jsonl"
    output_file: str = "machine_generated_instances.jsonl"
    clf_types_file: str = "is_clf_or_not_davinci_template_1.jsonl"

    # Generation parameters
    num_instructions: Optional[int] = None
    max_instances_to_generate: int = 5
    generation_tasks_only: bool = False
    classification_tasks_only: bool = False

    # API parameters
    engine: str = "davinci"
    request_batch_size: int = 5
    api_key: Optional[str] = None
    organization: Optional[str] = None

    # Model parameters
    max_tokens_clf: int = 300
    max_tokens_gen: int = 350
    temperature: float = 0.0
    top_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 1.5

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.generation_tasks_only and self.classification_tasks_only:
            raise ValueError(
                "Cannot specify both generation_tasks_only and classification_tasks_only"
            )

        if not os.path.exists(self.batch_dir):
            raise ValueError(f"Batch directory does not exist: {self.batch_dir}")


class InstanceGenerator:
    """Handles generation of instances for instructions."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.tasks: List[Dict] = []
        self.task_clf_types: Dict[str, bool] = {}
        self.existing_requests: Dict[str, Dict] = {}
        self.client = None

    def load_tasks(self) -> None:
        """Load tasks from input file."""
        input_path = os.path.join(self.config.batch_dir, self.config.input_file)

        with open(input_path) as fin:
            lines = fin.readlines()

            if self.config.num_instructions is not None:
                lines = lines[:self.config.num_instructions]

            for line in lines:
                data = json.loads(line)
                self.tasks.append(data)

        print(f"Loaded {len(self.tasks)} tasks")

    def load_classification_types(self) -> None:
        """Load classification types for tasks."""
        clf_path = os.path.join(self.config.batch_dir, self.config.clf_types_file)

        with open(clf_path) as fin:
            for line in fin:
                data = json.loads(line)
                instruction = data["instruction"]
                is_clf = data["is_classification"].strip() in ["Yes", "yes", "YES"]
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

    def load_existing_requests(self) -> None:
        """Load existing requests from output file if it exists."""
        output_path = os.path.join(self.config.batch_dir, self.config.output_file)

        if not os.path.exists(output_path):
            return

        with open(output_path) as fin:
            for line in tqdm.tqdm(fin, desc="Loading existing requests"):
                try:
                    data = json.loads(line)
                    self.existing_requests[data["instruction"]] = data
                except json.JSONDecodeError:
                    continue

        print(f"Loaded {len(self.existing_requests)} existing requests")

    def create_prompt(self, task: Dict) -> str:
        """Create prompt for a task based on its type."""
        instruction = task["instruction"].strip()
        is_clf = self.task_clf_types.get(task["instruction"], False)

        if is_clf:
            return output_first_template_for_clf + " " + instruction + "\n"
        else:
            return input_first_template_for_gen + " " + instruction + "\n"

    def format_output(self, data: Dict) -> Dict:
        """Format output data in consistent order."""
        return OrderedDict(
            (k, data[k]) for k in [
                "instruction",
                "raw_instances",
            ] if k in data
        )

    async def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of tasks to generate instances."""
        # Check if all tasks in batch already exist
        if all(task["instruction"] in self.existing_requests for task in batch):
            return [
                self.format_output(self.existing_requests[task["instruction"]])
                for task in batch
            ]

        # Initialize client if not already done
        if self.client is None:
            self.client = init_openai_client(
                api_key=self.config.api_key,
                organization=self.config.organization
            )

        # Create prompts for batch
        prompts = [self.create_prompt(task) for task in batch]

        # Determine max tokens based on task types
        has_clf_task = any(
            self.task_clf_types.get(task["instruction"], False)
            for task in batch
        )
        max_tokens = (
            self.config.max_tokens_clf if has_clf_task
            else self.config.max_tokens_gen
        )

        # Convert prompts to messages format for chat API
        messages_list = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        # Make API requests using the new async API
        results = await evoke_batch_requests(
            client=self.client,
            messages_list=messages_list,
            model=self.config.engine,
            max_tokens=max_tokens,
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

            outputs.append(self.format_output(task))

        return outputs

    async def generate(self) -> None:
        """Main generation loop."""
        output_path = os.path.join(self.config.batch_dir, self.config.output_file)

        with open(output_path, "w") as fout:
            progress_bar = tqdm.tqdm(total=len(self.tasks), desc="Generating instances")

            for batch_idx in range(0, len(self.tasks), self.config.request_batch_size):
                batch = self.tasks[batch_idx:batch_idx + self.config.request_batch_size]
                outputs = await self.process_batch(batch)

                for output in outputs:
                    fout.write(json.dumps(output, ensure_ascii=False) + "\n")

                progress_bar.update(len(batch))

            progress_bar.close()

        print(f"Generation complete. Output saved to: {output_path}")


async def main(config: GenerationConfig) -> None:
    """Main entry point."""
    # Validate configuration
    config.validate()

    # Initialize generator
    generator = InstanceGenerator(config)

    # Load data
    generator.load_tasks()
    generator.load_classification_types()
    generator.filter_tasks()
    generator.load_existing_requests()

    # Generate instances
    await generator.generate()


if __name__ == "__main__":
    # Example usage
    config = GenerationConfig(
        batch_dir="./data/batch_01",
        engine="gpt-4",
    )
    asyncio.run(main(config))
