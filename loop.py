from mariokart_env import MarioKartEnv
from rainbow_dqn_agent import RainbowDQN
from replay_buffer import ReplayBuffer
import torch
import torch.optim as optim
import numpy as np

env = MarioKartEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = RainbowDQN(state_size, action_size)
target = RainbowDQN(state_size, action_size)
target.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=1e-3)
buffer = ReplayBuffer(capacity=10000)

num_episodes = 100
batch_size = 32
gamma = 0.99
epilson = 0.1
update_target_every = 10

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = model.act(state, epilson)
        next_state, reward, done, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            states = torch.FloatTensor(states)
            nex_states = torch.FloatTensor(next_states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)

            dist = model(states)
            dist = dist.gather(1, actions.unsqueeze(-1).expand(-1, -1, model.n_atoms)).squeeze(1)

            next_dist = target(next_states)
            next_q = torch.sum(next_dist * model.support, dim=2)
            next_actions = next_q.argmax(1)
            next_dist = next_dist[range(batch_size), next_actions]

            Tz = rewards + (1 - dones) * gamma * model.support
            Tz = torch.clamp(Tz, model.v_min, model.v_max)
            b = (Tz - model.v_min) / model.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            m = torch.zeros(batch_size, model.n_atoms).to(states.device)
            for i in range(batch_size):
                for j in range(model.n_atoms):
                    lj, uj = l[i, j], u[i, j]
                    pj = next_dist[i, j]
                    m[i, lj] += pj * (uj - b[i, j])
                    m[i, uj] += pj * (b[i, j] - lj)

            loss = -(m * dist.log()).sum(1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % update_target_every == 0:
        target.load_state_dict(model.state_dict())

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
