import torch
from torch.autograd import Variable
import numpy as np
import random
from gym_torcs import TorcsEnv
import collections

from ReplayBuffer import ReplayBuffer
from QNetwork import QNetwork  # Assume you have defined this network correctly

# Hyperparameters
state_size = 29
action_size = 3  # Adjust as needed depending on the discretization of the action space in Torcs
LRA = 0.0001
BUFFER_SIZE = 100000
BATCH_SIZE = 32
GAMMA = 0.95
EXPLORE = 100000.
epsilon = 1
TAU = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)

def map_action(action_index):
    steer_map = {
        0: 0.2,  # Steer left
        1: 0.0,   # No steering
        2: 0.2    # Steer right
    }
    steer = steer_map.get(action_index, 0.0)
    throttle = 0.5  
    brake = 0.0   
    return np.array([steer, throttle, brake])


# Q-Network
q_network = QNetwork(state_size, action_size).to(device)
q_network.apply(init_weights)

target_q_network = QNetwork(state_size, action_size).to(device)
target_q_network.load_state_dict(q_network.state_dict())
target_q_network.eval()

optimizer = torch.optim.Adam(q_network.parameters(), lr=LRA)
criterion = torch.nn.MSELoss()

buff = ReplayBuffer(BUFFER_SIZE)
env = TorcsEnv(vision=False, throttle=True, gear_change=False)

file_distances = open("distances.txt","w") 
file_reward = open("rewards.txt","w") 
file_distances.close()
file_reward.close()

for episode in range(1000):
    if np.mod(episode, 3) == 0:
        ob,distFromStart = env.reset(relaunch = True)
    else:
        ob,distFromStart = env.reset()
    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
    loss = 0
    for t in range(100000):
        epsilon -= 1.0 / EXPLORE
        if random.random() < epsilon:
            action_index = random.randint(0, action_size - 1)
        else:
            with torch.no_grad():
                state = torch.tensor([s_t], device=device).float()
                action_index = q_network(state).argmax().item()

        continuous_action = map_action(action_index)
        ob, distFromStart, reward, done, _ = env.step(continuous_action)
        s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        buff.add(s_t, continuous_action, reward, s_t1, done)
        s_t = s_t1
        
        if buff.size() > BATCH_SIZE:
            states, actions, rewards, next_states, dones = buff.getBatch(BATCH_SIZE)
            states = torch.tensor(np.array(states), device=device).float()
            actions = torch.tensor(np.array([a[0] for a in actions]), device=device).long()  # Get the first element of each action list
            rewards = torch.tensor(np.array(rewards), device=device).float()
            next_states = torch.tensor(np.array(next_states), device=device).float()
            dones = torch.tensor(np.array(dones), device=device).float()
            q_values = q_network(states)
            q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            next_q_values = target_q_network(next_states).max(1)[0].detach()
            expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)
            loss = criterion(q_values, expected_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("---Episode ", episode, "  Action:", continuous_action, "  Reward:", reward)
        with open("rewards.txt","a") as f:
            f.write(str(episode) + " "+ str(reward) + "\n")

        if done:
            break
    with open("distances.txt","a") as f:
        f.write(str(episode) + " "+ str(distFromStart) +"\n")

    # Update target network
    if episode % 10 == 0:
        target_q_network.load_state_dict(q_network.state_dict())

    if episode % 100 == 0:
        print(f"Episode {episode}: Loss = {loss}, Epsilon = {epsilon}")

env.end()
print("Finish.")

