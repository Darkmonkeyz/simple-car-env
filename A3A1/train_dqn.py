import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import simple_driving
# import pybullet_envs
import pybullet as p
import math
from collections import defaultdict
import pickle
np.bool8 = np.bool_


# Use your environment
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Define Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = QNetwork(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Replay memory
memory = deque(maxlen=10000)
batch_size = 64
gamma = 0.95

# Epsilon-greedy
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995 

# Training loop
num_episodes = 30000
for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0

    for t in range(400):  # max steps per episode
        if random.random() < epsilon:
            action = env.action_space.sample()
            biased_probs = [0.04, 0.04, 0.04, 0.01, 0.01, 0.01, 0.25, 0.35, 0.25]
            action = np.random.choice(len(biased_probs), p=biased_probs)
            
        else:
            with torch.no_grad():
                q_values = model(torch.tensor(state, dtype=torch.float32))
                action = torch.argmax(q_values).item()

        next_state, reward, done, _, info = env.step(action)
        total_reward += reward

        memory.append((state, action, reward, next_state, done))
        state = next_state

        # Training step
        if len(memory) >= batch_size:
            minibatch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)

            states_tensor = torch.tensor(states, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
            dones_tensor = torch.tensor(dones, dtype=torch.bool)

            q_values = model(states_tensor).gather(1, actions_tensor).squeeze()
            with torch.no_grad():
                next_q_values = model(next_states_tensor).max(1)[0]
                target_q_values = rewards_tensor + gamma * next_q_values * (~dones_tensor)

            loss = loss_fn(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # Decay epsilon
    decay_rate = 5e-4
    epsilon = epsilon_min + (1.0 - epsilon_min) * math.exp(-decay_rate * episode)

    print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

# Save model
torch.save(model.state_dict(), "simple_driving_qlearning.pkl")

env.close()