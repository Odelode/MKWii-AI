# file: loop.py
from mariokart_env import MarioKartEnv
from rainbow_dqn_agent import RainbowDQN
from replay_buffer import ReplayBuffer
from helper import compute_rainbow_loss
import torch
from torch.optim import Adam

env = MarioKartEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = RainbowDQN(state_size, action_size)
target_model = RainbowDQN(state_size, action_size)
target_model.load_state_dict(model.state_dict())

buffer = ReplayBuffer(capacity=10000)
optimizer = Adam(model.parameters(), lr=1e-4)

batch_size = 32
gamma = 0.99
target_update = 10
num_episodes = 1000
epsilon = 0.1

for ep in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = model.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            states_v = torch.from_numpy(states)
            actions_v = torch.from_numpy(actions).long()
            rewards_v = torch.from_numpy(rewards)
            next_states_v = torch.from_numpy(next_states)
            dones_v = torch.from_numpy(dones)

            loss = compute_rainbow_loss(
                model, target_model,
                states_v, actions_v,
                rewards_v, next_states_v,
                dones_v, model.support, gamma
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if (ep + 1) % target_update == 0:
        target_model.load_state_dict(model.state_dict())

    print(f"Episode {ep+1}: Total Reward: {total_reward}")