import numpy as np
from collections import defaultdict
from enum import Enum, auto

class Agent:
    class Algorithm(Enum):
        def _generate_next_value_(name, start, count, last_value):
            return name.lower()
        QLEARNING = auto()
        EXPECTED_SARSA = auto()

    def __init__(
        self,
        nA:int=6,
        algorithm=Algorithm.EXPECTED_SARSA,
        initial_epsilon=0.9999999,
        epsilon_decay=0.9999999,
        get_beta=None,
        gamma:float=1.,
        alpha:float=.01,
        initial_beta=.8,
        c1:float=0.02,
        c2:float=3.0) -> None:
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.ones(self.nA))
        self.recent = defaultdict(lambda: np.ones(self.nA))  # this is the path memory table
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.beta = initial_beta
        self.get_beta = (lambda i: self.beta) if get_beta is None else get_beta
        self.i_episode = 0
        self.c1 = c1  # weight of value function in stochastic action distribution
        self.c2 = c2  # inverse weight of path memory in stochastic action distribution
        self.algo = algorithm  # algorithm to use - either Q-Learning or Expected SARSA

    def softmax(self, a:np.array):     # Cacluate softmax values from an array of real numbers
        e = np.exp(a)
        # print(f'Returned Softmax: {e/e.sum()}')
        return e/e.sum()

    def get_expected_state_value(self, state:int) -> float:
        ''' Return the expected state value according to the actual Q table '''
        q = np.asarray(self.Q[state])       # state's action values
        p = self.softmax(q)  # calculating probabilities based on the current policy only!!
        return np.dot(p, self.Q[state])

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if state not in self.Q:
            # Case where there are no action values entered yet for this state
            # Just choose randomly
            return np.random.choice(self.nA)
        q = np.asarray(self.Q[state])       #  state's action values
        r = np.asarray(self.recent[state])  #  state's path memory
        p = self.softmax(q*self.c1 - r*self.c2)
        greedy_action = np.asarray(self.Q[state]).argmax()          #  greedy action / non-stochastic action
        random_action = np.random.choice(self.nA, p=p)              #  stochastic action
        return np.random.choice([random_action, greedy_action], p=[self.epsilon, 1-self.epsilon])

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
        # update path memory table by incrementing count for most recent choice
        self.recent[state][action] += 1
        if not done:
            if self.algo == self.Algorithm.EXPECTED_SARSA:
                # this is the step for expected sarsa
                self.Q[state][action] += (self.alpha * (reward + (self.gamma * self.get_expected_state_value(next_state)) - self.Q[state][action]))
            elif self.algo == self.Algorithm.QLEARNING:
                self.Q[state][action] += (self.alpha * (reward + (self.gamma * np.max(self.Q[next_state])) - self.Q[state][action]))
            else:
                raise f'Algorithm used by the agent must be either {self.Algorithm.EXPECTED_SARSA} or {self.Algorithm.QLEARNING}'
        else:
            self.Q[state][action] += (self.alpha * (reward - self.Q[state][action]))
            # Decay path memory from current episode
            for state in self.recent:
                self.recent[state] = np.array([count*self.beta for count in self.recent[state]])
            self.epsilon *= self.epsilon_decay
            self.beta = self.get_beta(self.i_episode)
            self.i_episode += 1
