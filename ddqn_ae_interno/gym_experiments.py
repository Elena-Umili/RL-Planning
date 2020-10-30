import gym
from ddqn_ae_interno.DDQN import DDQNAgent
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque, OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from ddqn_ae_interno.QNetwork import QNetwork
from ddqn_ae_interno.experience_replay_buf import experienceReplayBuffer

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)



er_buf = experienceReplayBuffer(burn_in=1000)
env = gym.make('LunarLander-v2')
ddqn = DDQNAgent(env=env, buffer=er_buf, batch_size=64 )
ddqn.train()

