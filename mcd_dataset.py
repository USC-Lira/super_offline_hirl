import pickle
from tqdm import tqdm
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import os
import argparse
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def detect_catastrophe(state, action, force=0.001, gravity=0.0025):
    tuple_state = tuple(state)
    position = tuple_state[0]
    velocity = tuple_state[1]

    new_velocity = np.clip(velocity + (action - 1) * force - np.cos(3 * position) * gravity, -0.07, 0.07)
    new_position = np.clip(position + new_velocity, -1.2, 0.6)
    
    return new_position < -1.15

def is_catastrophe(state):
    tuple_state = tuple(state)
    position = tuple_state[0]
    return position < -1.15

def discretize_state(state):
    discretized_state = (state - env.observation_space.low) * np.array([20, 20])
    return tuple(discretized_state.astype(int))

def frame_write(frame):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((100, 100), "catastrophe", fill=(255, 0, 0)) #, font=font)
    frame_with_text = np.array(img)
    return frame_with_text

def generate_dataset_noha(policy, env, num_episodes=1000):
    dataset = []
    for _ in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        flag = 0
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = np.argmax(policy(state_tensor).cpu().data.numpy())
                robot_action = action

            if detect_catastrophe(state, action):
                flag = 1
                action = 2

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if flag == 1:
                dataset.append((state, robot_action, penalty, next_state, done))
            else:
                dataset.append((state, robot_action, reward, next_state, done))
            state = next_state
    return dataset

def generate_dataset_ha(policy, env, num_episodes=1000):
    dataset = []
    for _ in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        flag = 0
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = np.argmax(policy(state_tensor).cpu().data.numpy())
                robot_action = action

            if detect_catastrophe(state, action):
                flag = 1
                action = 2

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if flag == 1:
                dataset.append((state, robot_action, penalty, next_state, done))
                dataset.append((state, action, reward, next_state, done))
            else:
                dataset.append((state, robot_action, reward, next_state, done))
            state = next_state
    return dataset

def generate_dataset_clean(policy, env, num_episodes=1000):
    dataset = []
    for _ in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = np.argmax(policy(state_tensor).cpu().data.numpy())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            dataset.append((state, action, reward, next_state, done))
            state = next_state
    return dataset

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="set the name of the experiment")
parser.add_argument("--numeps", type=int, help="sets number of episodes")
parser.add_argument("--penalty", type=int, help="sets the penalty for evaluation")
parser.add_argument("--policy", type=str, help="optimal policy weights")
parser.add_argument("--type", type=str, help="clean, ha, or noha")
args = parser.parse_args()

exp_name = args.name
num_episodes = args.numeps
penalty = -float(args.penalty)
policy_path = '/home/jaiv/super_offlinerl/checkpoints/' + args.policy
exp_type = args.type

env = gym.make("MountainCar-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

optimal_policy = QNetwork(state_size, action_size).to(device)
optimal_policy.load_state_dict(torch.load(policy_path, map_location=device))
optimal_policy.eval()

if exp_type == "clean": dataset = generate_dataset_clean(optimal_policy, env, num_episodes)
elif exp_type == "ha": dataset = generate_dataset_ha(optimal_policy, env, num_episodes)
elif exp_type == "noha": dataset = generate_dataset_noha(optimal_policy, env, num_episodes)
else: print("check exp_type")

dataset = np.array(dataset, dtype=object)

dataset_directory = './dataset'
if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)

# Now, save the dataset to file safely
dataset_path = os.path.join(dataset_directory, exp_name + '.pkl')
with open(dataset_path, 'wb') as f:
    pickle.dump(dataset, f)

print(f"Dataset saved to '{dataset_path}'")