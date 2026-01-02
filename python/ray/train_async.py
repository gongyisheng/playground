"""
Async training

This example demonstrates:
- Overlapping data generation with training using Ray's async pattern
- Using ray.get() strategically to minimize idle time
- The prefetching pattern from miles/train_async.py

Key Insight:
- Instead of: generate -> train -> generate -> train (sequential)
- We do:      generate[0] -> train[0] + generate[1] -> train[1] + generate[2] -> ...
- This hides data generation latency behind training time
"""

import time

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(num_cpus=0.5)
class TrainActor:
    """Simulates training (takes some time)."""

    def __init__(self, rank: int):
        self.rank = rank

    def train(self, rollout_id: int, data):
        """Simulate training step (takes ~1 second)."""
        start = time.time()
        time.sleep(0.1)  # Simulate training time
        elapsed = time.time() - start
        print(f"[Train rank={self.rank}] Rollout {rollout_id}: trained on {len(data)} samples ({elapsed:.2f}s)")
        return {"rank": self.rank, "rollout_id": rollout_id}


@ray.remote(num_cpus=0.5)
class RolloutEngine:
    """Simulates inference/rollout generation (takes some time)."""

    def __init__(self, engine_id: int):
        self.engine_id = engine_id

    def generate(self, rollout_id: int, num_samples: int = 10):
        """Simulate data generation (takes ~0.8 seconds)."""
        start = time.time()
        time.sleep(0.5)  # Simulate generation time
        elapsed = time.time() - start
        data = [f"sample_{i}_from_rollout_{rollout_id}" for i in range(num_samples)]
        print(f"[Rollout engine={self.engine_id}] Rollout {rollout_id}: generated {len(data)} samples ({elapsed:.2f}s)")
        return data


def train_sync(train_actors, rollout_engines, num_rollouts=3):
    """
    Synchronous training loop - waits for each step to complete.
    This is similar to miles/train.py

    Timeline:
        |--generate--|--train--|--generate--|--train--|--generate--|--train--|

    Note: Even with 2 engines, sync doesn't benefit since we wait for each step.
    """
    print("SYNCHRONOUS Training")

    total_start = time.time()

    for rollout_id in range(num_rollouts):
        # Step 1: Generate data (BLOCKING - wait for completion)
        # Alternate between engines, though it doesn't help in sync mode
        engine = rollout_engines[rollout_id % len(rollout_engines)]
        rollout_data = ray.get(engine.generate.remote(rollout_id))

        # Step 2: Train (BLOCKING - wait for completion)
        train_futures = [actor.train.remote(rollout_id, rollout_data) for actor in train_actors]
        ray.get(train_futures)

    total_time = time.time() - total_start
    print(f"Sync total time: {total_time:.2f}s")
    return total_time


def train_async(train_actors, rollout_engines, num_rollouts=3):
    """
    Asynchronous training loop with prefetching.
    This is similar to miles/train_async.py

    Timeline (with 2 engines):
        |--generate[0] on E0--|--train[0]--|--train[1]--|--train[2]--|
                              |--generate[1] on E1--|
                                           |--generate[2] on E0--|

    The key is: start generating next rollout BEFORE waiting for training to finish.
    With 2 engines, we can alternate to maximize parallelism.
    """
    print("ASYNCHRONOUS Training")

    total_start = time.time()

    # Start first rollout generation on engine 0
    rollout_data_next_future = rollout_engines[0].generate.remote(0)

    for rollout_id in range(num_rollouts):
        # Wait for current rollout data (this was started in previous iteration or before loop)
        rollout_data = ray.get(rollout_data_next_future)

        # Immediately start NEXT rollout generation (non-blocking) if there's more to do
        # Alternate between engines for better parallelism
        if rollout_id < num_rollouts - 1:
            next_engine = rollout_engines[(rollout_id + 1) % len(rollout_engines)]
            rollout_data_next_future = next_engine.generate.remote(rollout_id + 1)
            print(f"[Prefetch] Started generating rollout {rollout_id + 1} on engine {(rollout_id + 1) % len(rollout_engines)}")

        # Train on current data (while next generation runs in background)
        train_futures = [actor.train.remote(rollout_id, rollout_data) for actor in train_actors]
        ray.get(train_futures)

    total_time = time.time() - total_start
    print(f"Async total time: {total_time:.2f}s")
    return total_time


def train_async_with_object_refs(train_actors, rollout_engines, num_rollouts=3):
    """
    Even more async - pass object refs directly without ray.get().

    This pattern passes the ObjectRef directly to training, allowing Ray
    to schedule optimally. Training will start as soon as data is ready.

    It's not really used in practice because we usually use 
    a single rollout engine to manage multiple gpus.

    Timeline (with 2 engines):
        Engine 0: |--gen[0](0.5)--|--gen[2](0.5)--|
        Engine 1: |--gen[1](0.5)--|
        Training:          |--train[0]--|--train[1]--|     |--train[2]--|

        gen[0] ready at t=0.5, gen[1] ready at t=0.5, gen[2] ready at t=1.0
        Total: ~1.1s (vs 1.6s with 1 engine)
    """
    print("ASYNC with ObjectRef Passing")

    total_start = time.time()

    # Start all generations upfront, distributed across engines
    # This allows parallel generation on different engines
    rollout_futures = [
        rollout_engines[i % len(rollout_engines)].generate.remote(i)
        for i in range(num_rollouts)
    ]

    for rollout_id in range(num_rollouts):
        # Get data ref (might already be ready)
        rollout_data_ref = rollout_futures[rollout_id]

        # Pass the ref directly - Ray handles the dependency
        # Training will start as soon as rollout_data_ref is ready
        train_futures = [actor.train.remote(rollout_id, rollout_data_ref) for actor in train_actors]
        ray.get(train_futures)

    total_time = time.time() - total_start
    print(f"ObjectRef passing total time: {total_time:.2f}s")
    return total_time


def main():
    ray.init(num_cpus=4, num_gpus=4)

    # Create placement group: 2 train actors + 2 rollout engines
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(4)]  # 2 train + 1 rollout
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())

    # Allocate train actors
    train_actors = []
    for i in range(2):
        actor = TrainActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            )
        ).remote(rank=i)
        train_actors.append(actor)

    # Allocate 1 rollout engines for parallel generation
    rollout_engines = []
    for i in range(1):
        engine = RolloutEngine.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=2 + i,
            )
        ).remote(engine_id=i)
        rollout_engines.append(engine)

    num_rollouts = 3

    # Run all versions and compare
    sync_time = train_sync(train_actors, rollout_engines, num_rollouts)
    async_time = train_async(train_actors, rollout_engines, num_rollouts)
    objref_time = train_async_with_object_refs(train_actors, rollout_engines, num_rollouts)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (with 2 rollout engines)")
    print("=" * 60)
    print(f"Synchronous:           {sync_time:.2f}s")
    print(f"Async (prefetch):      {async_time:.2f}s  (saved {sync_time - async_time:.2f}s)")
    print(f"Async (objectref):     {objref_time:.2f}s (saved {sync_time - objref_time:.2f}s)")

    # Expected with 2 engines:
    # Synchronous: 1.8s, (0.5+0.1)*3
    # Async (prefetch): ~1.6s, 0.5+0.1+0.5+0.5 
    # Async (objectref): ~1.6s, will be 0.1 if use 2 rollout engines


if __name__ == "__main__":
    main()
