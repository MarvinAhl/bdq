import sys
sys.path.insert(1, '../..')

from bdq import BDQ

import torch

import gym

import numpy as np
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')

env = gym.make('LunarLanderContinuous-v2')

nS = env.observation_space.shape[0]
nA = (3, 3)  # Engine Off, 50%, On; Nozzle Left, Off, Right

agent = BDQ(nS, nA, epsilon_decay_steps=10000, new_actions_prob=0.1, buffer_size_max=30000,
            buffer_size_min=500, beta_increase_steps=30000, device=device)

episode = 0
while True:
    try:
        episode += 1
        print(f'Episode {episode} started')
        obsv, done = env.reset(), False

        while not done:
            env.render()

            discrete_actions = agent.act(obsv)
            actions = discrete_actions.astype(np.float32) - 1.0 + 0.0001

            new_obsv, reward, done, info = env.step(actions)

            time_out = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
            terminal = done and not time_out

            agent.experience(obsv, discrete_actions, reward, new_obsv, terminal)
            agent.train()

            obsv = new_obsv
    except KeyboardInterrupt:
        break

agent.save_net('lunar_lander.net')