from copy import deepcopy
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from ddqn_ae_interno.experience_replay_buf import experienceReplayBuffer
from ddqn_ae_interno.AutoEncoder import AutoEncoder, Encoder, Decoder

from collections import namedtuple, deque, OrderedDict
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

class Net(nn.Module):
    def __init__(self, n_inputs, n_hidden_nodes, n_outputs, bias, encoder):
        super().__init__()
        self.encoder = encoder
        #self.encoder.load_state_dict(torch.load('lunar_models/code100_enc.pt'))

        self.layers0 = nn.Linear(
            200,
            n_hidden_nodes,
            bias=bias).to('cuda')
        self.layers1 = nn.ReLU().to('cuda')
        self.layers2 = nn.Linear(
                    n_hidden_nodes,
                    n_hidden_nodes,
                    bias=bias).to('cuda')
        self.layers3 = nn.ReLU()
        self.layers4 = nn.Linear(
                    n_hidden_nodes,
                    n_hidden_nodes,
                    bias=bias).to('cuda')
        self.layers5 = nn.ReLU().to('cuda')
        self.layers6 = nn.Linear(
                    n_hidden_nodes,
                    n_outputs,
                    bias=bias).to('cuda')
        #self.layers7 = nn.ReLU()

    def forward(self, data, temperature):

        out, _ = self.encoder(data.to('cuda'),temperature,False)
        out = self.layers0(out.to('cuda'))
        out = self.layers1(out).to('cuda')
        out = self.layers2(out).to('cuda')
        out = self.layers3(out).to('cuda')
        out = self.layers4(out).to('cuda')
        out = self.layers5(out).to('cuda')
        out = self.layers6(out).to('cuda')
        #out = self.layers7(out)
        return out

    def enc_forw(self, enc_data):
        out = self.layers0(enc_data.to('cuda'))
        out = self.layers1(out).to('cuda')
        out = self.layers2(out).to('cuda')
        out = self.layers3(out).to('cuda')
        out = self.layers4(out).to('cuda')
        out = self.layers5(out).to('cuda')
        out = self.layers6(out).to('cuda')
        # out = self.layers7(out)
        return torch.max(out)

class QNetwork(nn.Module):

    def __init__(self, env, encoder, learning_rate=1e-3, n_hidden_layers=4,
                 n_hidden_nodes=256, bias=True, activation_function='relu',
                 tau=1, device='cpu', norm_out = False, epsilon = 0.05, *args, **kwargs):
        super(QNetwork, self).__init__()
        self.norm_out = norm_out
        self.device = device
        self.actions = np.arange(env.action_space.n)
        self.tau = tau
        #n_inputs = env.observation_space.shape[0] * tau
        n_inputs = 8
        self.n_inputs = n_inputs
        self.epsilon = epsilon
        n_outputs = env.action_space.n

        self.network = Net(n_inputs, n_hidden_nodes, n_outputs, bias, encoder)
        # Set device for GPU's
        if self.device == 'cuda':
            self.network.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate)

    def get_action(self, state, temperature):

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.greedy_action(state,temperature)
        return action

    def greedy_action(self, state, temperature):
        qvals = self.get_qvals(state, temperature)
        return torch.max(qvals, dim=-1)[1].item()

    def get_qvals(self, state, temperature):
        if type(state) is tuple:
            #print("TUPLE!!!!!!!!!!!!!!!!!!!!")
            state = np.array([np.ravel(s) for s in state])
            state_t = torch.FloatTensor(state).to('cuda')
            #print(state_t)
        else:
            #print("NO TUPLE!")
            state_t = torch.from_numpy(np.asarray([state])).to('cuda')
            #print(state_t)
        out = self.network(state_t,temperature)
        #print("OUT = ", out)
        return out

    def get_enc_value(self,enc_state):
        return self.network.enc_forw(enc_state)