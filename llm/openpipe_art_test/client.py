import asyncio

import art
from art.local import LocalBackend


async def main():
    model = art.TrainableModel(
        # the name of your model as it will appear in W&B
        # and other observability platforms
        name="test",
        # keep your project name constant between all the models you train
        # for a given task to consistently group metrics
        project="test-agentic-task",
        # the model that you want to train from
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
    )

    backend = LocalBackend(
        in_process=True,
        path="/media/hdddisk/yisheng/.art"
    )
    await model.register(backend)


if __name__ == "__main__":
    asyncio.run(main())