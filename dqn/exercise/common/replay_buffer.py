import torch
import numpy as np
import random
from collections import namedtuple
from .circularbuffer import CircularBuffer
from .segment_tree import SumSegmentTree
from .segment_tree import MinSegmentTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    ''' Fixed-size buffer to store experience tuples '''

    def __init__(self, action_size: int, buffer_size: int):
        '''Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        '''
        self.action_size = action_size
        self.memory = CircularBuffer(buff_capacity=buffer_size)
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        ''' Return the current size of internal memory '''
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        ''' Add a new experience to memory '''
        e = self.experience(state, action, reward, next_state, done)

        return self.memory.append(e)

    def _build_batch(self, experiences: any) -> tuple:
        ''' Construct the batch of experiences that will be returned
            to the agent
        '''
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None
                                            ])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None
                                             ])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None
                                             ])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None
                      ]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def sample(self, batch_size: int = 1):
        ''' Randomly sample a batch of experiences from memory '''
        if batch_size >= len(self.memory):
            raise Exception('Batch size must not be greater than the buffer capacity')
        sampled_experiences = random.sample(self.memory, k=batch_size)
        return self._build_batch(experiences=sampled_experiences)


class PrioritizedReplayBuffer(ReplayBuffer):
    ''' Prioritized Replay Buffer '''

    def __init__(self, action_size: int, buffer_size: int, alpha: float):
        super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size)
        if alpha < 0:
            raise Exception('Alpha must be bigger than zero')
        self._alpha = alpha
        capacity = 1
        while capacity < buffer_size:
            capacity *= 2
        self._it_sum = SumSegmentTree(capacity=capacity)
        self._it_min = MinSegmentTree(capacity=capacity)
        self._max_priority = 1.

    def add(self, *args, **kwargs) -> None:
        idx = super(PrioritizedReplayBuffer, self).add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority**self._alpha
        self._it_min[idx] = self._max_priority**self._alpha

    def _sample_proportional(self, batch_size: int) -> list:
        idxs = []
        p_total = self._it_sum.sum(0, self.__len__() - 1)
        every_range_len = p_total / batch_size
        for batch_n in range(batch_size):
            mass = random.random() * every_range_len + batch_n * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            idxs.append(idx)
        return idxs

    def sample(self, batch_size: int, beta: float) -> tuple:
        '''
            Sample a batch of experiences
            campared to ReplayBuffer.sample it also returns importance
            weights and indices of sampled experiences
        '''
        if beta <= 0:
            raise Exception('beta must be bigger than zero.')
        idxs = self._sample_proportional(batch_size=batch_size)
        importance_sampling_weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.__len__())**-beta
        for idx in idxs:
            sample_prob = self._it_sum[idx] / self._it_sum.sum()
            sample_weight = (sample_prob * self.__len__())**-beta
            importance_sampling_weights.append(sample_weight / max_weight)
        sampled_experiences = self.memory.as_numpy()[idxs]
        return self._build_batch(sampled_experiences) + (torch.from_numpy(
            np.vstack(importance_sampling_weights)).float().to(device),) + (torch.from_numpy(
                np.vstack(idxs)).int().to(device),)

    def update_priorities(self, idxs: list, priorities: list) -> None:
        ''' Update priorities of sampled transitions '''
        if len(idxs) != len(priorities):
            raise Exception('Number of indices must equal number of priorities')
        for idx, priority in zip(idxs, priorities):
            if priority <= 0:
                raise Exception('Priorities cannot be zero values')
            if not (0 <= idx < self.__len__()):
                raise Exception('Index is out of Buffer capacity\'s range')
            self._it_sum[idx] = priority**self._alpha
            self._it_min[idx] = priority**self._alpha
            self._max_priority = max(self._max_priority, priority)

    @property
    def alpha(self) -> float:
        ''' Returns the value of alpha '''
        return self._alpha

    @alpha.setter
    def alpha(self, new_a: float) -> None:
        ''' Sets a new value of alpha '''
        self._alpha = new_a
