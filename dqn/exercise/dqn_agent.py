import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from common.replay_buffer import ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.988  # discount factor
TAU = 3e-1  # for soft update of target parameters
LR = 5e-4  # learning rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, C=2):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, 'local', seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, 'target', seed).to(device)
        # initialize weights
        self.qnetwork_local.apply(self._init_weights)
        self.qnetwork_target.apply(self._init_weights)
        self.q_local = self.qnetwork_local
        self.q_target = self.qnetwork_target
        #self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=LR)
        #self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1., end_factor=0.5, total_iters=50000)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.parameter_update_step = 0
        self.last_action = random.choice(np.arange(self.action_size))
        # softupdates frecuency w.r.t. a learn step
        self.C = C
        self.target_update_step = 0

    def __del__(self):
        """ Delete an agent intance.
            Explicitly remove qnetworks created whithin this agent
        """
        del self.qnetwork_local
        del self.qnetwork_target

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            #nn.init.zeros_(m.weight)
            m.bias.data.fill_(0.00)

    def step(self, state, action, reward, next_state, done, t_step):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps (every four steps - controlled on the notebook.
        if t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(np.array(state, dtype=float)).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            self.last_action = np.argmax(action_values.cpu().data.numpy())
        else:
            self.last_action = random.choice(np.arange(self.action_size))
        return self.last_action

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        target_action_vals = self.q_target(next_states).detach().max(1)[0].unsqueeze(1)
        targets = rewards + (gamma * target_action_vals * (1 - dones))
        expected = self.q_local(states).gather(1, actions)
        loss = F.mse_loss(targets, expected)  #.clip(-1., 1.)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #self.scheduler.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.q_local, self.q_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        self.target_update_step = (self.target_update_step + 1) % self.C
        if self.target_update_step == 0:
            for target_param, local_param in zip(target_model.parameters(),
                                                 local_model.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        #  returns either the local or the target network,
        #  each with 50% probability
        self.q_target = random.choices([self.qnetwork_local, self.qnetwork_target], [0.5, 0.5])[0]
        self.q_local = self.qnetwork_local if self.q_target.nettype == 'target' else self.qnetwork_target
