import os
import random
from collections import namedtuple
import numpy as np
import torch
torch.set_default_tensor_type('torch.DoubleTensor')

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward', 'horizon'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, num_actions, capacity, batch_size, device, recent_size=0):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.num_actions = num_actions

        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, num_actions), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.horizons = np.empty((capacity, 1), dtype=np.int32)

        self.recent_size = recent_size

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done, horizons):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.horizons[self.idx], horizons)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    # def add_batch(self, obs, action, reward, next_obs, size):
    #     np.copyto(self.obses[self.idx:self.idx+size], obs)
    #     aoh = np.zeros((size, self.num_actions), dtype=np.int)
    #     aoh[np.arange(size), action] = 1
    #     np.copyto(self.actions[self.idx:self.idx+size], aoh)
    #     np.copyto(self.rewards[self.idx:self.idx+size], reward)
    #     np.copyto(self.next_obses[self.idx:self.idx+size], next_obs)

    #     self.idx = (self.idx + size) % self.capacity
    #     self.full = self.full or self.idx == 0

    # def add_from_buffer(self, buf, batch_size=1):
    #     obs, action, reward, next_obs = buf.sample(batch_size=batch_size)
    #     # print(self.obses[self.idx: self.idx + batch_size].shape)
    #     # print(self.obses.shape)
    #     # print(self.idx)
    #     # print(batch_size)
    #     np.copyto(self.obses[self.idx: self.idx + batch_size], obs)
    #     np.copyto(self.actions[self.idx: self.idx + batch_size], action)
    #     np.copyto(self.rewards[self.idx: self.idx + batch_size], reward)
    #     np.copyto(self.next_obses[self.idx: self.idx + batch_size], next_obs)

    #     self.idx = (self.idx + batch_size) % self.capacity
    #     self.full = self.full or self.idx == 0

    # def get_full(self, recent_size=0, device=None):

    #     if device is None:
    #         device = self.device

    #     if self.idx <= recent_size or recent_size == 0:
    #         start_index = 0
    #     else:
    #         start_index = self.idx - recent_size

    #     if self.full:
    #         obses = torch.as_tensor(self.obses[start_index:], device=device)
    #         actions = torch.as_tensor(
    #             self.actions[start_index:], device=device)
    #         rewards = torch.as_tensor(
    #             self.rewards[start_index:], device=device)
    #         next_obses = torch.as_tensor(
    #             self.next_obses[start_index:], device=device)

    #         return obses, actions, rewards, next_obses

    #     else:
    #         obses = torch.as_tensor(
    #             self.obses[start_index:self.idx], device=device)
    #         actions = torch.as_tensor(
    #             self.actions[start_index:self.idx], device=device)
    #         rewards = torch.as_tensor(
    #             self.rewards[start_index:self.idx], device=device)
    #         next_obses = torch.as_tensor(
    #             self.next_obses[start_index:self.idx], device=device)

    #         return obses, actions, rewards, next_obses

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(
            self.obses[idxs], device=self.device, dtype=torch.double)
        actions = torch.as_tensor(
            self.actions[idxs], device=self.device, dtype=torch.double)
        rewards = torch.as_tensor(
            self.rewards[idxs], device=self.device, dtype=torch.double)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device, dtype=torch.double)
        not_dones = torch.as_tensor(
            self.not_dones[idxs], device=self.device, dtype=torch.double)
        horizons = self.horizons[idxs]

        return obses, actions, rewards, next_obses, not_dones, horizons

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.idx = end
