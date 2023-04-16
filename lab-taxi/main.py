from agent import Agent
from monitor import interact
import gym
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

env = gym.make('Taxi-v3')
initial_beta = 0.1087677305041776
betas = np.arange(0.0, initial_beta, initial_beta/30000)    # 20000 number of episodes hardcoded because the                                                                   # goal is to solve within this number of episodes
betas = np.flip(betas)
beta_getter = lambda i: max(0.01, betas[i])
print("Training Expected Sarsa...")
agent_expected_sarsa = Agent(
    alpha=0.17533766816679272,
    gamma=0.977141935762803,
    initial_epsilon=0.9250269486667245,
    epsilon_decay=0.6862400109005145,
    initial_beta=initial_beta,
    c1=0.20162495004999603,
    c2=2.520736687728135,
    algorithm=Agent.Algorithm.EXPECTED_SARSA,
    get_beta=beta_getter)
avg_rewards, best_avg_reward = interact(env, agent_expected_sarsa, num_episodes=30000, print_logs=True)
print()
print("Training Q-Learning")
agent_qlearning = Agent(
    alpha=0.630755690170461,
    gamma=0.37488609966832215,
    initial_epsilon=0.378849657245437,
    epsilon_decay=0.8085500017288388,
    initial_beta=0.5825758712357205,
    c1=1.4595435739219753,
    c2=6.781447920582464,
    algorithm=Agent.Algorithm.QLEARNING,
    get_beta=None)
avg_rewards, best_avg_reward = interact(env, agent_qlearning, num_episodes=30000, print_logs=True)
