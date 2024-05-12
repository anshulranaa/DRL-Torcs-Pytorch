import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from ActorNetwork import *
from CriticNetwork import *

class PPO(nn.Module):
    def __init__(self, S_DIM, A_DIM, A_LR, C_LR, A_UPDATE_STEPS, C_UPDATE_STEPS, METHOD):
        super(PPO, self).__init__()
        self.S_DIM = S_DIM
        self.A_DIM = A_DIM
        self.A_UPDATE_STEPS = A_UPDATE_STEPS
        self.C_UPDATE_STEPS = C_UPDATE_STEPS
        self.METHOD = METHOD

        # Instantiate actor and critic networks
        self.actor = Actor('actor')
        self.critic = Critic()

        # Define optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=A_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=C_LR)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        distribution = self.actor(state)
        action = distribution.sample().numpy()[0]
        return action

    def update(self, s, a, r):
        s = torch.tensor(s, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.float)
        r = torch.tensor(r, dtype=torch.float)

        # Update critic
        v = self.critic(s)
        advantage = r - v.detach()
        critic_loss = (advantage ** 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        distribution = self.actor(s)
        old_distribution = distribution.detach()

        for _ in range(self.A_UPDATE_STEPS):
            log_probs = distribution.log_prob(a)
            old_log_probs = old_distribution.log_prob(a)
            ratio = torch.exp(log_probs - old_log_probs)

            if self.METHOD['name'] == 'kl_pen':
                kl = torch.distributions.kl_divergence(old_distribution, distribution).mean()
                actor_loss = -(ratio * advantage).mean() - self.METHOD['lam'] * kl
            else:
                clipped_ratio = torch.clamp(ratio, 1 - self.METHOD['epsilon'], 1 + self.METHOD['epsilon'])
                actor_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # Optional lambda adjustment for kl_pen
        if self.METHOD['name'] == 'kl_pen':
            if kl < self.METHOD['kl_target'] / 1.5:
                self.METHOD['lam'] /= 2
            elif kl > self.METHOD['kl_target'] * 1.5:
                self.METHOD['lam'] *= 2
            self.METHOD['lam'] = np.clip(self.METHOD['lam'], 1e-4, 10)

    def get_v(self, s):
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        return self.critic(s).item()
