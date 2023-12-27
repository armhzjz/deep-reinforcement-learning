import torch
import numpy as np
import random
from collections import namedtuple
from .circularbuffer import CircularBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size: int, buffer_size: int):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = CircularBuffer(buff_capacity=buffer_size)
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)

        self.memory.append(e)

    def sample(self, batch_size: int = 1):
        """Randomly sample a batch of experiences from memory."""
        if batch_size >= len(self.memory):
            raise Exception('Batch size must not be greater than the buffer capacity')
        experiences = random.sample(self.memory, k=batch_size)
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
