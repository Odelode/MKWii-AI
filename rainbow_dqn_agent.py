import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Hyperparameters for distributional RL
V_MIN = -10
V_MAX = 110
N_ATOMS = 51

class RainbowDQN(nn.Module):
    def __init__(self, state_size, action_size, n_atoms=N_ATOMS, v_min=V_MIN, v_max=V_MAX):
        super(RainbowDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.support = torch.linspace(v_min, v_max, n_atoms)

        # Define network layers
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, action_size * n_atoms)

    def forward(self, x):
        # x should be of shape [batch_size, state_size]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        # Reshape the output to [batch, action_size, n_atoms]
        x = x.view(-1, self.action_size, self.n_atoms)
        # Softmax over the atoms dimension to obtain a probability distribution for each action
        x = torch.softmax(x, dim=2)
        return x

    def act(self, state, epsilon):
        # Epsilon-greedy policy: with probability epsilon, choose a random action
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            # Evaluate Q-values as the expected value of the distribution
            state = torch.FloatTensor(state).unsqueeze(0)  # [1, state_size]
            dist = self.forward(state)  # [1, action_size, n_atoms]
            # Compute expected values: sum(probabilities * support)
            q_values = torch.sum(dist * self.support.to(dist.device), dim=2)  # [1, action_size]
            return torch.argmax(q_values, dim=1).item()

# Example usage:
if __name__ == "__main__":
    # Assuming state size 1 (lap progression) and 3 possible actions
    model = RainbowDQN(state_size=1, action_size=3)
    sample_state = [1.0]
    action = model.act(sample_state, epsilon=0.05)
    print(f"Selected action: {action}")