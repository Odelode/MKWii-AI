import torch

def project_distribution(rewards, dones, next_dist, support, gamma):
    """
    rewards:    [B]
    dones:      [B] (1 if terminal)
    next_dist:  [B, N] probability from target network
    support:    [N] atom values
    return:     [B, N] projected target distribution
    """
    v_min, v_max = support[0], support[-1]
    delta_z = support[1] - support[0]
    batch_size, n_atoms = next_dist.size()

    # 1) Compute Tz_j = r + γ * z_j for non-terminals
    Tz = rewards.unsqueeze(1) + gamma * support.unsqueeze(0) * (1 - dones.unsqueeze(1))
    Tz = Tz.clamp(v_min, v_max)

    # 2) b = (Tz - v_min) / delta_z
    b = (Tz - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    # 3) Distribute mass
    m = torch.zeros_like(next_dist)
    offset = (
        torch.linspace(0, (batch_size - 1) * n_atoms, batch_size)
        .unsqueeze(1)
        .expand(batch_size, n_atoms)
        .to(l.device)
        .long()
    )

    # lower
    m.view(-1).index_add_(
        0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
    )
    # upper
    m.view(-1).index_add_(
        0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
    )

    return m

def compute_rainbow_loss(model, target_model, states, actions, rewards, next_states, dones, support, gamma):
    """
    states, next_states: [B, state_size]
    actions:             [B]
    rewards, dones:      [B]
    support:             [N]
    """
    batch_size = states.size(0)
    dist = model(states)  # [B, A, N]
    dist = dist[range(batch_size), actions]  # [B, N]

    # next-state distribution
    next_dist_all = target_model(next_states)  # [B, A, N]
    # choose greedy action by expected value
    q_vals = torch.sum(next_dist_all * support, dim=2)  # [B, A]
    next_actions = q_vals.argmax(1)  # [B]
    next_dist = next_dist_all[range(batch_size), next_actions]  # [B, N]

    # project onto support
    target_dist = project_distribution(rewards, dones, next_dist, support, gamma)

    # cross-entropy loss
    log_p = torch.log(dist + 1e-8)
    loss = - (target_dist * log_p).sum(1).mean()
    return loss