from mariokart_env import MarioKartEnv
from rainbow_dqn_agent import RainbowDQN
import torch
import numpy as np

env = MarioKartEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = RainbowDQN(state_size, action_size)

num_episodes = 1000
epsilon = 0.1
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = model.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")