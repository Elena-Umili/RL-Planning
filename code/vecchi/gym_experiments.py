import gym
from DDQN import DDQNAgent
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque, OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from QNetwork import QNetwork
from experience_replay_buf import experienceReplayBuffer
import warnings
from myWrapper import StateDiscretizerWrapper

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)



er_buf = experienceReplayBuffer()
env = StateDiscretizerWrapper(gym.make('LunarLander-v2'))
theModel = QNetwork(env=env, n_hidden_nodes=64)
ddqn = DDQNAgent(env=env, network=theModel, buffer=er_buf, batch_size=64 )
ddqn.train()

