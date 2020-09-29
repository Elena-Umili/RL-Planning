import gym
from ddqn_ae_esterno.DDQN import DDQNAgent
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque, OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from ddqn_ae_esterno.QNetwork import QNetwork
from ddqn_ae_esterno.experience_replay_buf import experienceReplayBuffer
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)



er_buf = experienceReplayBuffer()
env = gym.make('LunarLander-v2')
theModel = QNetwork(env=env, n_hidden_nodes=256, activation_function='relu', n_hidden_layers=4)
ddqn = DDQNAgent(env=env, network=theModel, buffer=er_buf, batch_size=64)
ddqn.train()

