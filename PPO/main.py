import torch
import numpy as np
import gym
from ppo import PPO
from gym_torcs import TorcsEnv

# Configuration Constants
EP_MAX = 2000
EP_LEN = 1000
GAMMA = 0.95
A_LR = 1e-4
C_LR = 1e-4
BATCH = 128
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 29, 3
METHOD = {'name': 'clip', 'epsilon': 0.1}  # Choose the method for optimization
train_test = 1
irestart = 1

# Initialize the environment
env = TorcsEnv(vision=False, throttle=True, gear_change=False)
ppo = PPO(S_DIM, A_DIM, A_LR, C_LR, A_UPDATE_STEPS, C_UPDATE_STEPS, METHOD)

if irestart == 1:
    try:
        ppo.load_state_dict(torch.load("model.pth"))
        print("model load successfully")
    except (FileNotFoundError, EOFError):
        iter_num = 0
        print("No valid model found, starting training from scratch.")

# Start Training
for ep in range(iter_num, EP_MAX):
    print("\n" + "-" * 50)
    print(f"Episode: {ep}")

    if np.mod(ep, 100) == 0:
        ob = env.reset(relaunch=True)  # Relaunch TORCS every 100 episode due to memory leak error
        print("Relaunching TORCS environment.")
    else:
        ob = env.reset()

    s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):
        a = ppo.choose_action(s)
        a = np.clip(a, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        ob, r, done, _ = env.step(a)
        s_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

        if train_test == 0:
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)

        s = s_
        ep_r += r

        if train_test == 0:
            if (t + 1) % BATCH == 0 or t == EP_LEN - 1 or done:
                v_s_ = ppo.get_v(s_)
                discounted_r = []
                for r in reversed(buffer_r):
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                ppo.update(torch.tensor(np.vstack(buffer_s), dtype=torch.float),
                           torch.tensor(np.vstack(buffer_a), dtype=torch.float),
                           torch.tensor(np.vstack(discounted_r), dtype=torch.float))

                buffer_s, buffer_a, buffer_r = [], [], []

        print("---Episode ", ep , "  Action:", a, "  Reward:", r,)  # Added the print statement as per your request.

        if done:
            break

    print(f'Episode: {ep} | Episode Reward: {ep_r:.4f}', end='')
    if METHOD['name'] == 'clip':
        print(f' | Lambda: {METHOD["epsilon"]:.4f}')
    else:
        print('')

    if train_test == 0 and ep % 25 == 0:
        print("saving model")
        torch.save(ppo.state_dict(), "model.pth")

print("Finish.")
