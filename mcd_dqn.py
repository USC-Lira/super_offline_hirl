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
from PIL import Image, ImageDraw

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-3               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        wandb.log({"loss": loss}) if iswandb else None

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

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

def evaluate_model(agent, num_episodes=10):

    def frame_write(frame):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.text((100, 100), "catastrophe", fill=(255, 0, 0))
        frame_with_text = np.array(img)
        return frame_with_text
    
    frames = []
    total_rewards = 0.0
    num_catastrophes = 0
    catastrophe_flag = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state, 0)

            if is_catastrophe(state):
                num_catastrophes += 1
                catastrophe_flag = 1
                frame = frame_write(env.render())
            else:
                frame = env.render()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if catastrophe_flag == 1:
                reward = penalty
                catastrophe_flag = 0
            
            frames.append(frame)
            total_rewards += reward
            state = next_state
            if done: break

    video_frames = np.array(frames)
    video_frames = np.transpose(video_frames, (0, 3, 1, 2))
    wandb.log({"video": wandb.Video(video_frames, fps=15)})

    average_test_catastrophes = num_catastrophes / num_episodes
    average_test_reward = total_rewards / num_episodes
    wandb.log({"test_catastophe": average_test_catastrophes, "test_score": average_test_reward})

def train(agent, n_episodes=2000, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.999):
    """Deep Q-Learning.

    Params
    ======
        agent: agent to train
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        catastrophe_flag = 0
        num_interventions = 0
        num_catastrophes = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            robot_action = action

            if detect_catastrophe(state, action):
                catastrophe_flag = 1
                action = 2
                num_interventions += 1

            if is_catastrophe(state):
                num_catastrophes += 1

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if catastrophe_flag == 1:
                reward = penalty
                catastrophe_flag = 0
            
            agent.step(state, robot_action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if iswandb: wandb.log({"num_episodes": i_episode, "score": score, "average_score": np.mean(scores_window), "epsilon": eps, "train_catastrophe": num_catastrophes, "train_blocker": num_interventions})
        if i_episode % 100 == 0:
            savename = "./checkpoints/" + str(experiment_name)+ "/dqn_" + str(i_episode) + ".pth"
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), savename)
            if iswandb and (i_episode % 200 == 0): evaluate_model(agent)
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="set the name of the experiment")
    parser.add_argument("--numeps", type=int, help="sets number of episodes")
    parser.add_argument("--penalty", type=int, help="sets penalty")
    parser.add_argument("--seed", type=int, help="sets seed for the QNetwork")
    parser.add_argument("--wandb", help="sets wandb to be true", action="store_true")
    args = parser.parse_args()

    experiment_name = args.name
    experiment_version = args.ver
    n_episodes = args.numeps
    penalty = -float(args.penalty)
    seed = args.seed 
    iswandb = args.wandb
    
    if iswandb: wandb.init(project="modified_hirl", name=experiment_name, config={'env_name': 'MountainCar-v0', 'n_episodes': n_episodes, 'penalty': penalty})

    os.makedirs("./checkpoints/" + str(experiment_name), exist_ok=True)
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    state_size = 2
    action_size = 3
    agent = Agent(state_size=state_size, action_size=action_size, seed=seed)
    scores = train(agent, n_episodes)