import asyncio
import random
import re
from tqdm import tqdm
from typing import List

from dataclasses import dataclass
from rouge_score import rouge_scorer

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
    n_sample: int = 100
    n_cot: int = 8
    batch_size: int = 8
    rouge_similarity_threshold: float = 0.7

    model_name: str = "Qwen/Qwen3-14B"
    max_token: int = 128
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

    def build_messages(self, all_tasks: List[str]) -> List[dict]:
        """Build messages with randomly selected examples from all_tasks.

        Args:
            all_tasks: List of existing tasks to sample from

        Returns:
            List of message dictionaries for API request
        """
        # randomly select some as examples
        n_to_select = min(self.config.n_cot, len(all_tasks))
        selected_examples = random.sample(all_tasks, n_to_select)

        # build examples section
        examples_text = "\n".join([f'eg, "{example}"' for example in selected_examples])
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
        pattern = r'<task>(.*?)</task>'

        for response in responses:
            if response is None:
                continue

            content = response.choices[0].message.content
            stop_reason = response.choices[0].finish_reason

            if stop_reason == "length":
                continue

            # Extract all tasks from XML tags
            try:
                matches = re.findall(pattern, content, re.DOTALL)
                for match in matches:
                    task = match.strip()
                    if task:
                        generated_tasks.append(task)
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
        all_tasks = ["pick a random number from 1 to 6"]

        n_sample = self.config.n_sample

        pbar = tqdm(total=n_sample, desc="Generating tasks", unit="task")
        with DataWriter(
            output_path=self.config.output_path,
            mode="w",
            auto_flush=True,
            create_dirs=True
        ) as writer:
            
            while len(all_tasks) < n_sample:
                batch_tasks = await self.generate_tasks(all_tasks)
                for task in batch_tasks:
                    print(task)
                    if not self.is_similar(task, all_tasks):
                        task_dict = {"task": task}
                        all_tasks.append(task)
                        writer.write(task_dict)
                        pbar.update(1)


async def main():

    config = Config()
    generator = RandomTaskGenerator(config)

    await generator.run()


if __name__ == "__main__":
    asyncio.run(main())
