import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ddqn_ae_interno.AutoEncoder import Encoder,Decoder,AutoEncoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class TransitionDelta(nn.Module):

    def __init__(self, code_size, action_size):
        super().__init__()

        self.code_size = code_size
        self.action_size = action_size
        input_size = code_size + action_size

        self.layer1 = nn.Linear(input_size, input_size * 2).to(device)
        self.layer2 = nn.Linear(input_size * 2, code_size).to(device)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.001)

    def forward(self, z, action):
        #print("z.shape ", z.shape)
        #print("action.shape ",action.shape)
        action = action.type(torch.float32).to('cuda')
        cat = torch.cat((z, action), -1)
        # print(cat.shape)
        delta_z = torch.sigmoid(self.layer1(cat))
        delta_z = torch.tanh(self.layer2(delta_z))
        z_prime = z + delta_z
        y = torch.ones(self.code_size).to(device)
        x = torch.zeros(self.code_size).to(device)
        z_prime = z_prime.where(z_prime < 1.0, y)
        z_prime = z_prime.where(z_prime >= 0.0, x)
        return z_prime


class Transition(nn.Module):

    def __init__(self, encoder, decoder, transition_delta):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transition_delta = transition_delta
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.001)

    def forward(self, x, action, x_prime, epoch, n_epochs):
        z = self.encoder(x, epoch, n_epochs)
        z_prime = self.encoder(x_prime, epoch, n_epochs)

        delta_z = self.transition_delta(z, action)

        z_prime_hat = z + delta_z

        x_prime_hat = self.decoder(z_prime_hat)

        return z_prime_hat - z_prime, x_prime_hat, z_prime_hat

class Predictor(nn.Module):
    def __init__(self, trans_delta):
        super().__init__()
        self.delta = trans_delta
    def forward(self, x, a):
        out = self.delta(x,a)
        return out




########################## Sizes
state_size = 8
action_size = 4
code_size = 100

'''
loss_function = nn.MSELoss()
# loss_function = nn.L1Loss()
optimizerTR = optim.SGD(tr.parameters(), lr=0.01)
optimizerAE = optim.Adam(ae.parameters(), lr=0.0001)
'''
