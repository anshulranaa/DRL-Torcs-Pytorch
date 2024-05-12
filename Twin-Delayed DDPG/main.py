import torch
import numpy as np
import random
from gym_torcs import TorcsEnv
import collections
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import os

state_size = 29
action_size = 3
LRA = 0.0001
LRC = 0.001
BUFFER_SIZE = 100000  # to change
BATCH_SIZE = 32
GAMMA = 0.95
EXPLORE = 100000.
epsilon = 1
train_indicator = True  # train or not
TAU = 0.001

VISION = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ou_noise = OU()

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)

actor = ActorNetwork(state_size).to(device)
actor.apply(init_weights)
critic = CriticNetwork(state_size, action_size).to(device)

# load model
print("loading model")
try:
    actor.load_state_dict(torch.load('actormodel.pth'))
    actor.eval()
    critic.load_state_dict(torch.load('criticmodel.pth'))
    critic.eval()
    print("model loaded successfully")
except:
    print("cannot find the model")

target_actor = ActorNetwork(state_size).to(device)
target_critic = CriticNetwork(state_size, action_size).to(device)
target_actor.load_state_dict(actor.state_dict())
target_actor.eval()
target_critic.load_state_dict(critic.state_dict())
target_critic.eval()

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LRA)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LRC)

buff = ReplayBuffer(BUFFER_SIZE)

# env environment
env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)

torch.set_default_dtype(torch.float32)

steps = 0
for i in range(2000):

    if np.mod(i, 3) == 0:
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()

    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

    for j in range(100000):
        epsilon -= 1.0 / EXPLORE
        a_t = np.zeros([1, action_size])
        noise_t = np.zeros([1, action_size])

        a_t_original = actor(torch.tensor(s_t.reshape(1, -1), dtype=torch.float32, device=device))
        a_t_original = a_t_original.detach().cpu().numpy()

        noise_t[0][0] = train_indicator * max(epsilon, 0) * ou_noise.noise(a_t_original[0][0].item(), 0.0, 0.60, 0.30)
        noise_t[0][1] = train_indicator * max(epsilon, 0) * ou_noise.noise(a_t_original[0][1].item(), 0.5, 1.00, 0.10)
        noise_t[0][2] = train_indicator * max(epsilon, 0) * ou_noise.noise(a_t_original[0][2].item(), -0.1, 1.00, 0.05)

        if random.random() <= 0.1:
            print("apply the brake")
            noise_t[0][2] = train_indicator * max(epsilon, 0) * ou_noise.noise(a_t_original[0][2].item(), 0.2, 1.00, 0.10)
        a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
        a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
        a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

        ob, r_t, done, info = env.step(a_t[0])

        s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

        buff.add(s_t, a_t[0], r_t, s_t1, done)

        batch = buff.getBatch(BATCH_SIZE)

        states = torch.tensor(np.asarray([e[0] for e in batch]), device=device, dtype=torch.float32)
        actions = torch.tensor(np.asarray([e[1] for e in batch]), device=device, dtype=torch.float32)
        rewards = torch.tensor(np.asarray([e[2] for e in batch]), device=device, dtype=torch.float32)
        new_states = torch.tensor(np.asarray([e[3] for e in batch]), device=device, dtype=torch.float32)

        dones = np.asarray([e[4] for e in batch])

        target_q_values1, target_q_values2 = target_critic(new_states, target_actor(new_states))

        target_q_values = torch.min(target_q_values1, target_q_values2)

        dones_tensor = torch.tensor(dones, device=device, dtype=torch.float32)

        target_q_values = target_q_values.unsqueeze(-1)

        y_t = rewards + GAMMA * target_q_values.detach() * (1 - dones_tensor)

        if train_indicator:
            q_values1, q_values2 = critic(states, actions)
            q_values1 = q_values1.squeeze(-1)
            q_values2 = q_values2.squeeze(-1)
            y_t = y_t.unsqueeze(-1)

            loss = torch.nn.functional.mse_loss(q_values1, y_t) + torch.nn.functional.mse_loss(q_values2, y_t)
            optimizer_critic.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_critic.step()

            optimizer_actor.zero_grad()
            actor_actions = actor(states)
            actor_loss = -critic.Q1(states, actor_actions).mean()
            actor_loss.backward()
            optimizer_actor.step()

            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        s_t = s_t1
        steps += 1
        print("---Episode ", i, "  Action:", a_t, "  Reward:", r_t, "  Loss:", int(loss))
        print(str(i) + " " + str(steps) + " " + str(ob.angle) + " " + str(ob.trackPos) + " " + str(ob.speedX) + " " + str(ob.speedY) + " " + str(ob.speedZ) + "\n")
        if done:
            print("I'm dead")
            break
    if np.mod(i, 3) == 0:
        if train_indicator:
            print("saving model")
            torch.save(actor.state_dict(), 'actor.pth')
            torch.save(critic.state_dict(), 'critic.pth')

env.end()
print("Finish.")
