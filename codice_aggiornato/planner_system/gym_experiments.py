import gym
from planner_system.DDQN import DDQNAgent
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from planner_system.experience_replay_buf import experienceReplayBuffer

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)



er_buf = experienceReplayBuffer(burn_in=1000)
env = gym.make('CartPole-v0')
ddqn = DDQNAgent(env=env, buffer=er_buf, batch_size=64 )
ddqn.train()

