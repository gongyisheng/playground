from distilabel.pipeline import Pipeline
from distilabel.steps import GeneratorStep, TaskStep
from distilabel.tasks.openai import OpenAITask

# 1. Define a generator step to provide seed rows
class SeedGenerator(GeneratorStep):
    def load(self):
        # Distilabel expects list[dict]
        return [
            {"topic": "climate change"},
            {"topic": "quantum computing"},
            {"topic": "healthy cooking"},
        ]

# 2. Define the generation task (OpenAI)
gen_task = OpenAITask(
    id="generate_qna",
    model="gpt-4o-mini",  # or gpt-3.5-turbo, gpt-4o
    template="Write a question about {topic}, then answer it clearly.",
    output_key="qna"  # where the generated text will be stored
)

# 3. Define a judging task (OpenAI)
judge_task = OpenAITask(
    id="judge_quality",
    model="gpt-4o-mini",
    template=(
        "Evaluate the following Q&A:\n\n"
        "{qna}\n\n"
        "Answer only with 'good' if it's coherent and useful, or 'bad' if not."
    ),
    input_keys=["qna"],   # take the generated text as input
    output_key="judgment" # store output here
)

# 4. Wrap tasks into steps
gen_step = TaskStep(id="generator", task=gen_task)
judge_step = TaskStep(id="judge", task=judge_task)

# 5. Build pipeline
pipeline = Pipeline(steps=[SeedGenerator(id="seed"), gen_step, judge_step])

# 6. Run pipeline
distiset = pipeline.run()

# 7. Inspect results
for row in distiset["train"]:
    print("Topic:", row["topic"])
    print("QnA:", row["qna"])
    print("Judgment:", row["judgment"])
    print("---")
