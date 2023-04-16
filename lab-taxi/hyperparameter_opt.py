### Some references ###
# https://github.com/crazyleg/gym-taxi-v2-v3-solution/blob/e21e115d5c32e37d918abc4754bbf8a9b2c46ab9/hyper_opt.py
# https://www.gymlibrary.dev/environments/toy_text/taxi/
# https://github.com/hyperopt/hyperopt
from agent import Agent
from monitor import interact
from hyperopt import hp, fmin, tpe, space_eval

import gym
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def objective_f(args) -> float:
    env = gym.make('Taxi-v3')
    best_scores = []
    initial_beta = args[4]
    if args[7]:
        betas = np.arange(0.0, initial_beta, initial_beta/20000)    # 20000 number of episodes hardcoded because the
                                                                    # goal is to solve within this number of episodes
        betas = np.flip(betas)
        beta_getter = lambda i: max(0.01, betas[i])
    else:
        beta_getter = None
    for i in range(5):
        agent = Agent(
            alpha=args[0],
            gamma=args[1],
            initial_epsilon=args[2],
            epsilon_decay=args[3],
            initial_beta=initial_beta,
            c1=args[5],
            c2=args[6],
            algorithm=Agent.Algorithm.QLEARNING,
            get_beta=beta_getter)
        _, best_avg_reward = interact(env, agent, print_logs=False)
        best_scores.append(best_avg_reward)
    return -np.mean(np.array(best_scores))

# search space
space = [
    hp.uniform('alpha', 0., 1.),
    hp.uniform('gamma', 0., 1.),
    hp.uniform('initial_epsilon', 0., 1.),
    hp.uniform('epsilon_decay', 0., 1.),
    hp.uniform('initial_beta', 0., 1.),
    hp.uniform('c1', 0., 10.),
    hp.uniform('c2', 0., 10.),
    hp.choice('beta_getter', [True, False])
]

# find the optimal parameters ....
best = fmin(
    fn=objective_f,
    space=space,
    algo=tpe.suggest,
    max_evals=100)

print(f'Best parameters found:\n\t{best}')
print(f'Eval space:\n\t{space_eval(space, best)}')
