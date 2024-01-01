import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

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

    def __init__(self):
        raise NotImplementedError


class DoubleDQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, C=2):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE)
        self._init(state_size=state_size, action_size=action_size, seed=seed, C=C)

    def __del__(self):
        """ Delete an agent intance.
            Explicitly remove qnetworks created whithin this agent
        """
        del self._qnetwork
        del self.qnetwork_target

    def _init(self, state_size: int, action_size: int or tuple, seed: float, C: int) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self._qnetwork = QNetwork(state_size, action_size, 'local', seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, 'target', seed).to(device)
        # initialize weights
        self._qnetwork.apply(self._init_weights)
        self.qnetwork_target.apply(self._init_weights)
        self._training_model = self._qnetwork
        self._target_model = self.qnetwork_target
        self.optimizer = optim.RMSprop(self._qnetwork.parameters(), lr=LR)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.parameter_update_step = 0
        self.last_action = random.choice(np.arange(self.action_size))
        # softupdates frecuency w.r.t. a learn step
        self.C = C
        self.target_update_step = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            #nn.init.zeros_(m.weight)
            m.bias.data.fill_(0.00)

    def _underline_step(self, state, action, reward, next_state, done, t_step) -> bool:
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps (every four steps - controlled on the notebook.
        if t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                return True
        return False

    def step(self, state, action, reward, next_state, done, t_step):
        if self._underline_step(state, action, reward, next_state, done, t_step):
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
        self._qnetwork.eval()
        with torch.no_grad():
            action_values = self._qnetwork(state)
        self._qnetwork.train()

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
        Q_next = self._target_model(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target = rewards + (gamma * Q_next * (1 - dones))
        Q = self._training_model(states).gather(1, actions)
        loss = F.mse_loss(Q_target, Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self._training_model, self._target_model, TAU)

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
        self._target_model = random.choices([self._qnetwork, self.qnetwork_target], [0.5, 0.5])[0]
        self._training_model = self._qnetwork if self._target_model.nettype == 'target' else self.qnetwork_target

    @property
    def qnetwork_model(self):
        return self._qnetwork


class AgentPrioritizedReplayBuf(Agent):

    def __init__(self,
                 state_size: int,
                 action_size: int or tuple,
                 seed: float,
                 C: int = 2,
                 alpha: float = 0.):
        self.memory = PrioritizedReplayBuffer(action_size=action_size,
                                              buffer_size=BUFFER_SIZE,
                                              alpha=alpha)
        self._init(state_size=state_size, action_size=action_size, seed=seed, C=C)

    def step(self, state: any, action: any, reward: float, next_state: any, done: bool, t_step,
             new_alpha: float, new_beta: float) -> None:
        self.memory.alpha = new_alpha
        if self._underline_step(state, action, reward, next_state, done, t_step):
            *experiences, weights, indices = self.memory.sample(batch_size=BATCH_SIZE,
                                                                beta=new_beta)
            self.learn(experiences=experiences, gamma=GAMMA, weights=weights, indices=indices)

    def learn(self, experiences: any, gamma: float, weights: any, indices: any) -> None:
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        Q_next = self._target_model(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target = rewards + (gamma * Q_next * (1 - dones))
        Q = self._training_model(states).gather(1, actions)
        td_error = torch.abs(Q - Q_target).detach()
        loss = torch.mean((Q - Q_target)**2 * weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self._training_model, self._target_model, TAU)
        # update priorities
        self.memory.update_priorities(idxs=indices, priorities=td_error.numpy() + 5e-5)
