import verl
from verl.algorithms import PPO
from verl.envs import GymEnv

def main():
    # Create the CartPole environment
    env = GymEnv("CartPole-v1")

    # Create the PPO agent
    ppo_agent = PPO(
        env=env,
        num_steps=2048,     # rollout steps per update
        num_epochs=10,      # epochs per update
        batch_size=64,      # minibatch size
        learning_rate=3e-4,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Train the agent
    ppo_agent.train(total_timesteps=100000)  # total timesteps to train

if __name__ == "__main__":
    main()
