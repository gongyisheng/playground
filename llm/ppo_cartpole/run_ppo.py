import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define env
env = gym.make('CartPole-v1')

# Define network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy_net = PolicyNetwork(state_dim, action_dim)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

value_net = ValueNetwork(state_dim)

# training
optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=0.001)

def compute_advantages(rewards, values, gamma=0.99, lambd=0.95):
    advantages = []
    gae = 0
    for delta, value in zip(reversed(rewards - values), reversed(values)):
        gae = delta + gamma * lambd * gae
        advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32)

def train_ppo(env, policy_net, value_net, optimizer, epochs=10, epsilon=0.2):
    for epoch in range(epochs):
        states, actions, rewards, dones, values = [], [], [], [], []
        state, _ = env.reset()
        done = False
        while not done:
            state_np = np.array(state)
            state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
            action_probs = policy_net(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
            next_state, reward, done = list(env.step(action))[:3]
            value = value_net(state_tensor).item()
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            state = next_state

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        advantages = compute_advantages(rewards, values)
        returns = advantages + values

        for _ in range(10):
            action_probs = policy_net(states)
            action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1))
            ratios = torch.exp(action_log_probs - torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1).detach()))
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.nn.functional.mse_loss(returns, values)
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

epoch = 100
train_ppo(env, policy_net, value_net, optimizer, epochs=epoch)