import gym
from old.DDQN import DDQNAgent
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from old.experience_replay_buf import experienceReplayBuffer
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)



er_buf = experienceReplayBuffer()
env = gym.make('CartPole-v0')
ddqn = DDQNAgent(env=env, buffer=er_buf, batch_size=64 )
ddqn.train()

