import matplotlib.pyplot as plt
import numpy as np
import gym
from utils import *

env = gym.make("MountainCar-v0")
goal_pos = env.goal_position
Max = env.observation_space.high
Min = env.observation_space.low
dimension = len(Max)
num_actions = env.action_space.n

Env = {
    "env": env,
    "goal_pos": goal_pos,
    "dim": dimension,
    "num_actions": num_actions,
    "max": Max,
    "min": Min,
    "name": "MountainCar-v0"
}

mcAgent = Agent(Env)

mcAgent.SolveWith_Actor_Critic(num_tilings=10, num_tiles=10, episodes=10000, display_every=1000, step_sizes=[0.4, 0.3, 0.15])
mcAgent.SolveWith_Diff_SARSA(episodes=10000, display_every=1000, alpha=0.03, beta=0.02, num_tilings=8, num_tiles=15, epsilon=0.05)
mcAgent.SolveWith_Q_Learning(episodes=10000, display_every=1000, epsilon=0.05, alpha=0.015)
mcAgent.SolveWith_SARSA(episodes=10000, display_every=1000, alpha=0.1, num_tilings=10, num_tiles=10, epsilon=0.05)


method = "q_learning" # "sarsa", "d_sarsa", "actor_critic"

mcAgent.Test(method)