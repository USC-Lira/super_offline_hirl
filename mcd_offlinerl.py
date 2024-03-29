import numpy as np
import random
from collections import namedtuple, deque
import os

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb
import argparse
from tqdm import tqdm

import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="set the name of the experiment")
parser.add_argument("--numits", type=int, help="sets number of iterations")
parser.add_argument("--penalty", type=int, help="sets the penalty for evaluation")
parser.add_argument("--data", type=str, help="set the data file for the experiment")
parser.add_argument("--wandb", help="sets wandb to be true", action="store_true")
args = parser.parse_args()

exp_name = args.name
num_iterations = args.numits
eval_penalty = -float(args.penalty)
data_path = args.data
learning_rate = 0.0005
iswandb = args.wandb

if iswandb:
    wandb_hyperparameters = {'env_name': 'MountainCar-v0', 'num_iterations': num_iterations, 'learning_rate': learning_rate}
    wandb.init(project="modified_hirl", name=exp_name, config=wandb_hyperparameters)

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
def is_catastrophe(state):
    tuple_state = tuple(state)
    position = tuple_state[0]
    return position < -1.15

env = gym.make("MountainCar-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

with open('./dataset/' + str(data_path), 'rb') as f:
    dataset = pickle.load(f)

np.random.shuffle(dataset)

print("Dataset loaded. Number of samples:", len(dataset))

# Assuming your dataset is in the form of an array of tuples (state, action, reward, next_state, done)
states = np.vstack(dataset[:, 0])
actions = np.vstack(dataset[:, 1]).astype(np.float32)
rewards = np.vstack(dataset[:, 2]).astype(np.float32)
next_states = np.vstack(dataset[:, 3])
dones = np.vstack(dataset[:, 4]).astype(np.float32)

# Convert to PyTorch tensors
states = torch.tensor(states, dtype=torch.float).to(device)
actions = torch.tensor(actions, dtype=torch.long).to(device)
rewards = torch.tensor(rewards, dtype=torch.float).to(device)
next_states = torch.tensor(next_states, dtype=torch.float).to(device)
dones = torch.tensor(dones, dtype=torch.float).to(device)

# Model, optimizer, and loss function
model = QNetwork(state_size, action_size, seed=0).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.999)

def evaluate_model(env, model, num_episodes=10):
    total_rewards = 0.0
    total_catastrophes = 0
    catastrophe_flag = 0
    for _ in range(num_episodes):
        state, _ = env.reset()  # Ensure state is reset correctly at the start of each episode
        done = False
        while not done:
            state_tensor = torch.tensor([state], dtype=torch.float).to(device)  # Convert state to tensor
            with torch.no_grad():
                action = model(state_tensor).max(1)[1].view(1, 1).item()
            if is_catastrophe(state): 
                total_catastrophes += 1
                catastrophe_flag = 1
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if catastrophe_flag == 1:
                reward = eval_penalty
                catastrophe_flag = 0
            total_rewards += reward
            state = next_state  # Update state to next_state for the next iteration
    average_reward = total_rewards / num_episodes
    average_catastrophes = total_catastrophes / num_episodes
    return average_reward, average_catastrophes

# Training loop adjusted 
for epoch in range(num_iterations):
    optimizer.zero_grad()
    q_values = model(states)
    max_next_q_values = model(next_states).detach().max(1)[0].unsqueeze(1)
    expected_q_values = rewards + (0.99 * max_next_q_values * (1 - dones))
    loss = criterion(q_values.gather(1, actions), expected_q_values)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        evaluation_reward, evaluation_catastrophes = evaluate_model(env, model, num_episodes=10)
        print(f"Epoch {epoch}, Loss: {loss.item()}, Evaluation Reward: {evaluation_reward}, Evaluation Catastrophes: {evaluation_catastrophes}")
        if iswandb: wandb.log({"epoch": epoch, "loss": loss.item(), "evaluation_reward": evaluation_reward, 'evaluation_catastrophes': evaluation_catastrophes})