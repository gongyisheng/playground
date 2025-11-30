import asyncio
import json
import random
import re
from tqdm import tqdm
from typing import List

from dataclasses import dataclass
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

from utils import init_openai_client, evoke_batch_requests, DataWriter
from prompt import TASK_GENERATION_PROMPT

# Set random seed for reproducibility
random.seed(42)

@dataclass
class Config:
    """Configuration for task generation."""
    # Model configuration
    base_url: str = "https://vllm.yellowday.day/v1"
    output_path: str = "outputs/random_tasks.jsonl"
    seed_tasks_path: str = "data/seed_tasks.jsonl"
    n_sample: int = 100
    n_cot_seed: int = 4
    n_cot_machine: int = 4
    batch_size: int = 8
    rouge_similarity_threshold: float = 0.7
    max_task_len = 32
    max_options_num = 8

    model_name: str = "Qwen/Qwen3-14B"
    max_token: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 2.0


class RandomTaskGenerator:
    def __init__(self, config: Config):
        """Initialize the task generator with config."""
        self.config = config
        self.client = init_openai_client(base_url=config.base_url)
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.seed_tasks = self.load_seed_tasks()

    def load_seed_tasks(self) -> List[dict]:
        """Load seed tasks from the jsonl file.

        Returns:
            List of seed task dictionaries
        """
        seed_tasks = []
        try:
            with open(self.config.seed_tasks_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        seed_tasks.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: Seed tasks file not found at {self.config.seed_tasks_path}")
        except Exception as e:
            print(f"Error loading seed tasks: {e}")
        return seed_tasks

    def build_messages(self, all_tasks: List[str]) -> List[dict]:
        """Build messages with randomly selected examples from seed tasks and machine-generated tasks.

        Args:
            all_tasks: List of machine-generated tasks to sample from

        Returns:
            List of message dictionaries for API request
        """
        selected_examples = []

        # Randomly select from seed tasks
        if self.seed_tasks:
            n_seed_to_select = min(self.config.n_cot_seed, len(self.seed_tasks))
            selected_examples.extend(random.sample(self.seed_tasks, n_seed_to_select))

        # Randomly select from machine-generated tasks
        if all_tasks:
            n_machine_to_select = min(self.config.n_cot_machine, len(all_tasks))
            selected_examples.extend(random.sample(all_tasks, n_machine_to_select))

        # build examples section
        examples_text = "\n".join([f'eg, task={example['task']}, options={example['options']}' for example in selected_examples])
        prompt = TASK_GENERATION_PROMPT.format(examples=examples_text)
        return [{"role": "user", "content": prompt}]

    async def generate_tasks(self, all_tasks: List[str] = []) -> List[str]:
        """Generate multiple batches of random selection tasks.

        Args:
            all_tasks: List of existing tasks to use as examples

        Returns:
            List of generated tasks
        """
        # Create multiple message lists for batch requests
        messages_list = []
        for _ in range(self.config.batch_size):
            messages = self.build_messages(all_tasks)
            messages_list.append(messages)

        # Use evoke_batch_requests to send all API calls concurrently
        responses = await evoke_batch_requests(
            self.client,
            messages_list,
            model=self.config.model_name,
            max_tokens=self.config.max_token,
            temperature=self.config.temperature,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}} # for qwen3 models, disable think
        )

        # Parse all responses using regex to extract tasks from <task>...</task>
        generated_tasks = []
        task_pattern = r'<task>(.*?)</task>'
        options_pattern = r'<options>(.*?)</options>'
        option_pattern = r'<option>(.*?)</option>'

        for response in responses:
            if response is None:
                continue

            content = response.choices[0].message.content
            stop_reason = response.choices[0].finish_reason

            if stop_reason == "length":
                continue

            # Extract all tasks from XML tags
            try:
                task_matches = re.findall(task_pattern, content, re.DOTALL)
                assert len(task_matches) == 1
                task = task_matches[0].strip()
                options_matches = re.findall(options_pattern, content, re.DOTALL)
                assert len(options_matches) == 1
                options_text = options_matches[0].strip()
                option_matches = re.findall(option_pattern, options_text, re.DOTALL)
                assert len(option_matches) > 0
                options = []
                for option in option_matches:
                    options.append(option.strip())
                generated_tasks.append({
                    "task": task,
                    "options": options
                })
            except Exception as e:
                print(f"Parse task error: {e}")

        return generated_tasks

    def is_similar(self, task: str, existing_tasks: List[str]) -> bool:
        """Check if task is too similar to any existing task using ROUGE-L score."""
        for existing_task in existing_tasks:
            scores = self.scorer.score(existing_task, task)
            rouge_l_f1 = scores['rougeL'].fmeasure
            if rouge_l_f1 > self.config.rouge_similarity_threshold:
                return True
        return False

    async def run(self):
        """Generate and save the complete dataset of random selection tasks."""
        all_tasks = []
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14b")
        n_sample = self.config.n_sample

        pbar = tqdm(total=n_sample, desc="Generating tasks", unit="task")
        with DataWriter(
            output_path=self.config.output_path,
            mode="w",
            auto_flush=True,
            create_dirs=True
        ) as writer:
            
            while len(all_tasks) < n_sample:
                batch_results = await self.generate_tasks(all_tasks)
                for result in batch_results:
                    if len(tokenizer.encode(result["task"])) > self.config.max_task_len:
                        continue
                    if len(result["options"]) > self.config.max_options_num:
                        continue
                    if not self.is_similar(result["task"], [i["task"] for i in all_tasks]):
                        all_tasks.append(result)
                        writer.write(result)
                        pbar.update(1)


async def main():

    config = Config()
    generator = RandomTaskGenerator(config)

    await generator.run()


if __name__ == "__main__":
    asyncio.run(main())
