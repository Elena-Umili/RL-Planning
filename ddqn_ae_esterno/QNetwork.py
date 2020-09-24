from copy import deepcopy
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from experience_replay_buf import experienceReplayBuffer
from collections import namedtuple, deque, OrderedDict
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

class QNetwork(nn.Module):

    def __init__(self, env, learning_rate=1e-3, n_hidden_layers=4,
                 n_hidden_nodes=256, bias=True, activation_function='relu',
                 tau=1, device='cpu', *args, **kwargs):
        super(QNetwork, self).__init__()
        self.device = device
        self.actions = np.arange(env.action_space.n)
        self.tau = tau
        #n_inputs = env.observation_space.shape[0] * tau
        n_inputs = 25
        self.n_inputs = n_inputs
        n_outputs = env.action_space.n

        activation_function = activation_function.lower()
        if activation_function == 'relu':
            act_func = nn.ReLU()
        elif activation_function == 'tanh':
            act_func = nn.Tanh()
        elif activation_function == 'elu':
            act_func = nn.ELU()
        elif activation_function == 'sigmoid':
            act_func = nn.Sigmoid()
        elif activation_function == 'selu':
            act_func = nn.SELU()

        # Build a network dependent on the hidden layer and node parameters
        layers = OrderedDict()
        n_layers = 2 * (n_hidden_layers - 1)
        for i in range(n_layers + 1):
            if n_hidden_layers == 0:
                layers[str(i)] = nn.Linear(
                    n_inputs,
                    n_outputs,
                    bias=bias)
            elif i == n_layers:
                layers[str(i)] = nn.Linear(
                    n_hidden_nodes,
                    n_outputs,
                    bias=bias)
            elif i % 2 == 0 and i == 0:
                layers[str(i)] = nn.Linear(
                    n_inputs,
                    n_hidden_nodes,
                    bias=bias)
            elif i % 2 == 0 and i < n_layers - 1:
                layers[str(i)] = nn.Linear(
                    n_hidden_nodes,
                    n_hidden_nodes,
                    bias=bias)
            else:
                layers[str(i)] = act_func

        self.network = nn.Sequential(layers)

        # Set device for GPU's
        if self.device == 'cuda':
            self.network.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate)

    def get_action(self, state, epsilon=0.2):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.greedy_action(state)
        return action

    def greedy_action(self, state):
        qvals = self.get_qvals(state)
        return torch.max(qvals, dim=-1)[1].item()

    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.network(state_t)