import numpy as np
import os
import math
import sys
import random
import time
import json
import copy
from tqdm import tqdm
import pickle
from lock_continuous import Lock

from collections import deque

import torch

from utils import set_seed_everywhere
from replay_memory import ReplayBuffer


os.environ["OMP_NUM_THREADS"] = "1"


def evaluate_policy(env, epsilon, args):
    returns = np.zeros((args.num_envs, 1))

    obs = env.reset()
    for h in range(args.horizon):
        action = eps_greedy_actions(env, args, epsilon)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        # print(reward)
        returns += reward

    return np.mean(returns)


def eps_greedy_actions(env, epsilon=-1):
    """
    return a list of epsilon greedy actions
    """
    h = env.h
    if epsilon == -1:
        epsilon = 1/env.horizon
    action = np.zeros(env.action_dim)
    if env.get_state() == 0:
        action[env.opt_a[h]] = 1
    else:
        action[env.opt_b[h]] = 1
    if np.random.rand() < epsilon:
        action = np.random.rand(env.action_dim)

    return action


def collect_offline_buffer(batch_size, env, num_episodes, option="epsilon", verbose=False, buffers=None, device=torch.device('cpu')):
    """
    collect offline replay buffer with an epsilon greedy policy
    Args:
        - :param: `args` (parsed argument): the main arguments
        - :param: 'num_episodes': the number of episodes to collect
        - :param: 'option': this method supports two type of datasets, one is 
        option 'epsilon' and the other one is 'mixed', more details in the paper
        - :param: 'verbose': if set to True, print out fraction of episodes that
            reach the end
        - :param: 'buffers': a list of H buffers to start with
    Return:
        - :param: 'buffers': a list of ReplayBuffer of the number of horizon
    """

    horizon = env.horizon

    num_runs = int(num_episodes)

    # num_reaches keep track of the number of episodes that make to the end
    num_reaches = 0
    buffers = [ReplayBuffer(env.observation_space.shape,
                            env.action_space.shape[0],
                            int(num_episodes)*2,
                            batch_size,
                            device) for _ in range(horizon)]

    for n in tqdm(range(num_runs)):
        obs = env.reset()
        for h in range(horizon):
            action = eps_greedy_actions(
                env, 1/horizon)
            next_obs, reward, done, _ = env.step(action)
            buffers[h].add(obs, action, reward, next_obs, done, h)
            obs = next_obs

            if h == horizon - 1:
                count = env.get_counts()

        num_reaches += count[:2].sum()

    print(f'num_reaches: {num_reaches}')
    print("rewards: {}".format(buffers[h].rewards.sum()))

    return buffers


def test_q(env, horizon, q_net, policy_net):
    print(f'OPT A: {env.opt_a}')
    print(f'OPT B: {env.opt_b}')
    actions = torch.eye(5)
    with torch.no_grad():
        for h in range(env.horizon):
            print(f'=======> H = {h}')
            env.horizon = h
            toprint = 'STATE 0: '
            for a in range(5):
                toprint += f'[{a}] {q_net(torch.Tensor(np.concatenate([env.make_obs(0), actions[a]], axis = -1)).unsqueeze(0)).item()},'
            print('Q values:' + toprint)
            toprint = 'STATE 0: '
            for a in range(1):
                toprint += f' {q_net.get_value(torch.Tensor(env.make_obs(0)).unsqueeze(0), policy_net).item()},'
            print('V values:' + toprint)
            toprint = 'STATE 1: '
            for a in range(5):
                toprint += f'[{a}] {q_net(torch.Tensor(np.concatenate([env.make_obs(1), actions[a]], axis = -1)).unsqueeze(0)).item()},'
            print('Q values:' + toprint)
            toprint = 'STATE 1: '
            for a in range(1):
                toprint += f' {q_net.get_value(torch.Tensor(env.make_obs(1)).unsqueeze(0), policy_net).item()},'
            print('V values:' + toprint)
            toprint = 'STATE 2: '
            for a in range(5):
                toprint += f'[{a}] {q_net(torch.Tensor(np.concatenate([env.make_obs(2), actions[a]], axis = -1)).unsqueeze(0)).item()},'
            print('Q values:' + toprint)
            toprint = 'STATE 2: '
            for a in range(1):
                toprint += f' {q_net.get_value(torch.Tensor(env.make_obs(2)).unsqueeze(0), policy_net).item()},'
            print('V values:' + toprint)

# set_seed_everywhere(123)
# env = Lock()
# env.init(horizon=25,
#              action_dim=10)

# collect_offline_buffer(128,env,1000)
