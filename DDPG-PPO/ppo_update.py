import torch
import torch.nn.functional as F

# Constants, ensure these are defined or imported if they are used elsewhere
GAMMA = 0.95
EPS_CLIP = 0.2

def ppo_update(batch, old_log_probs, actor, target_actor, critic, target_critic, optimizer_actor, optimizer_critic, device):
    states, actions, rewards, new_states, dones, old_log_probs = zip(*batch)

    states = torch.tensor(states, device=device, dtype=torch.float)
    actions = torch.tensor(actions, device=device, dtype=torch.float)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)
    new_states = torch.tensor(new_states, device=device, dtype=torch.float)
    old_log_probs = torch.tensor(old_log_probs, device=device, dtype=torch.float)

    # Calculating advantages
    with torch.no_grad():
        target_q_values = target_critic(new_states, target_actor(new_states))
        q_values = critic(states, actions)
        td_target = rewards + GAMMA * target_q_values * (1 - torch.tensor(dones, dtype=torch.float))
        advantages = td_target - q_values

    # Actor update
    new_log_probs = actor.get_log_probs(states, actions)
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()

    # Critic update (as usual in DDPG)
    critic_loss = F.mse_loss(td_target, q_values)
    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()
