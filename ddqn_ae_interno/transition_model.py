import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransitionDelta(nn.Module):

    def __init__(self, code_size, action_size):
        super().__init__()

        self.code_size = code_size
        self.action_size = action_size
        input_size = code_size + action_size

        self.layer1 = nn.Linear(input_size, input_size * 2).to(device)
        self.layer2 = nn.Linear(input_size * 2, code_size).to(device)

    def forward(self, z, action):
        cat = torch.cat((z, action), 1)
        delta_z = torch.sigmoid(self.layer1(cat))
        delta_z = torch.tanh(self.layer2(delta_z))

        return delta_z


class Transition(nn.Module):

    def __init__(self, encoder, decoder, transition_delta):
        super().__init__()
        self.categorical_size = 10
        self.encoder = encoder
        self.decoder = decoder
        self.transition_delta = transition_delta
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x, action, x_prime, temp, hard):
        z, qy = self.encoder(x, temp, hard)
        z_prime, qy_prime = self.encoder(x_prime, temp, hard)

        delta_z = self.transition_delta(z, action)

        z_prime_hat = z + delta_z

        x_prime_hat = self.decoder(z_prime_hat)

        error_z = z_prime_hat - z_prime

        return error_z, x_prime_hat, qy, qy_prime

    def loss_function_transition(self, error_z, x_prime, recon_x_prime, qy):
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, state_size), size_average=False) / x.shape[0]
        l = nn.MSELoss()
        RE = l(recon_x_prime, x_prime)
        E = torch.norm(error_z)
        log_ratio = torch.log(qy * self.categorical_size + 1e-20)
        KLD = torch.sum(qy * log_ratio, dim=-1).mean()

        return RE + KLD + E