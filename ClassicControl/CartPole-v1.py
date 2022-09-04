import matplotlib.pyplot as plt
import numpy as np
import gym
from utils import *

env = gym.make("CartPole-v1")
goal_pos = None
Max = env.observation_space.high
Max[1] = 4.6
Max[3] = 4.6
Min = env.observation_space.low
Min[1] = -4.6
Min[3] = -4.6
dimension = len(Max)
num_actions = env.action_space.n


Env = {
    "env": env,
    "goal_pos": goal_pos,
    "dim": dimension,
    "num_actions": num_actions,
    "max": Max,
    "min": Min,
    "name": "CartPole-v1"
}

cpAgent = Agent(Env)

cpAgent.SolveWith_Q_Learning(episodes=30000, display_every=3000, epsilon=0.05, alpha=0.12, state_discretization=[20, 20, 20, 20])

cpAgent.SolveWith_SARSA(episodes=1500, display_every=150, alpha=0.3, num_tilings=4, num_tiles=16, epsilon=0.05)

cpAgent.SolveWith_Diff_SARSA(episodes=1500, display_every=150, alpha=0.09, beta=0.03, num_tilings=4, num_tiles=16, epsilon=0.05)

cpAgent.SolveWith_Actor_Critic(num_tilings=4, num_tiles=10, episodes=1500, display_every=150, step_sizes=[0.2, 0.3, 0.12])


method = "q_learning" # "sarsa", "d_sarsa", "actor_critic"

cpAgent.Test(method)