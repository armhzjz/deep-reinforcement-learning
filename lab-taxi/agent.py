import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA:int=6, initial_epsilon_value:float=1., gamma:float=1., alpha:float=.75) -> None:
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.initial_epsilon_value = initial_epsilon_value
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_val = initial_epsilon_value

    @property
    def initial_epsilon(self) -> float:
        return self.initial_epsilon_value

    @property
    def epsilon(self) -> float:
        return self.epsilon_val

    @epsilon.setter
    def epsilon(self, new_epsilon) -> None:
        self.epsilon_val = new_epsilon
    def get_expected_state_value(self, state:int) -> float:
        policy_s = np.ones(self.nA) * self.epsilon_val/self.nA
        policy_s[np.argmax(self.Q[state])] = 1 - self.epsilon_val + self.epsilon_val/self.nA
        return np.dot(policy_s, self.Q[state])

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy_s = np.ones(self.nA) * self.epsilon/self.nA
        policy_s[np.argmax(self.Q[state])] = 1 - self.epsilon + (self.epsilon/self.nA)
        return np.random.choice(np.arange(self.nA), p=policy_s)

    def step(self, state:int, action:int, reward:float, next_state:int, done:bool) -> None:
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += (self.alpha * (reward + (self.gamma*self.get_expected_state_value(next_state)) - self.Q[state][action]))
