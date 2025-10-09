import argparse
import asyncio
import os
import random

from dotenv import load_dotenv
from rollout import TicTacToeScenario, rollout

import art
from art.local.backend import LocalBackend

load_dotenv()

random.seed(42)

STEP = 50
DEPLOY_MODEL = False
GENERATE_BENCHMARKS = False
DESTROY_AFTER_RUN = False

async def main():

    backend = LocalBackend()

    model = art.TrainableModel(
        name="llama-8b-007",
        project="tic-tac-toe",
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )

    print("registering")
    await model.register(backend)

    print("training")
    for i in range(await model.get_step(), STEP):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, TicTacToeScenario(step=i)) for _ in range(96)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
        )
        await model.delete_checkpoints()
        await model.train(train_groups, config=art.TrainConfig(learning_rate=5e-5))
        await backend._experimental_push_to_s3(model)

    if DEPLOY_MODEL:
        print("deploying")
        deployment_result = await backend._experimental_deploy(
            deploy_to="together",
            model=model,
            step=STEP,
            verbose=True,
            pull_s3=False,
            wait_for_completion=True,
        )
        if deployment_result.status == "Failed":
            raise Exception(f"Deployment failed: {deployment_result.failure_reason}")

        deployed_model_name = deployment_result.model_name

        lora_model = art.Model(
            name=deployed_model_name,
            project="tic-tac-toe",
            inference_api_key=os.environ["TOGETHER_API_KEY"],
            inference_base_url="https://api.together.xyz/v1",
            inference_model_name=deployed_model_name,
        )

        print("Starting a rollout using the deployed model!")
        traj = await rollout(lora_model, TicTacToeScenario(step=0))

        print(traj)

    if DESTROY_AFTER_RUN:
        await backend.down()

    if GENERATE_BENCHMARKS:
        gpt_4o_mini = art.Model(
            name="gpt-4o-mini",
            project="tic-tac-toe",
            inference_model_name="gpt-4o-mini",
            inference_api_key=os.getenv("OPENAI_API_KEY"),
            inference_base_url="https://api.openai.com/v1",
        )
        await gpt_4o_mini.register(backend)

        gpt_4o = art.Model(
            name="gpt-4o",
            project="tic-tac-toe",
            inference_model_name="gpt-4o",
            inference_api_key=os.getenv("OPENAI_API_KEY"),
            inference_base_url="https://api.openai.com/v1",
        )
        await gpt_4o.register(backend)

        async def benchmark_comparison_model(comparison_model: art.Model):
            trajectories = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        rollout(comparison_model, TicTacToeScenario(step=0))
                        for _ in range(12)
                    )
                    for _ in range(1)
                ),
                pbar_desc=f"gather {comparison_model.name}",
                max_exceptions=1,
            )
            await comparison_model.log(
                trajectories,
                split="val",
            )

        await benchmark_comparison_model(gpt_4o_mini)
        await benchmark_comparison_model(gpt_4o)


if __name__ == "__main__":
    asyncio.run(main())