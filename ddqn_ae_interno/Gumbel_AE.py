######################## Transition Model 3 ##############################
# Come il 2 ma l'encoder è un Gumbel-sogtmax activation variational encoder
# => il codice è una matrice latent_size(N) x categorical_size(M) flattata, ossia abbiamo N vettori one-hot a M componenti

# stato discreto: SI

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################## Sizes
state_size = 8
action_size = 4
latent_size = 20
categorical_size = 10
code_size = latent_size * categorical_size

temp_min = 0.5
ANNEAL_RATE = 0.00003


########################## Gumbel
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits.to('cuda') + sample_gumbel(logits.size()).to('cuda')
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, latent_size * categorical_size)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_size * categorical_size)


class EncoderGumbel(nn.Module):

    def __init__(self, input_size, latent_size, categorical_size):
        super().__init__()

        self.input_size = input_size
        code_size = latent_size * categorical_size
        self.code_size = code_size
        self.latent_size = latent_size
        self.categorical_size = categorical_size

        self.layer1 = nn.Linear(input_size, input_size * 2).to(device)
        self.layer2 = nn.Linear(input_size * 2, input_size * 4).to(device)
        self.layer3 = nn.Linear(input_size * 4, input_size * 16).to(device)
        self.layer4 = nn.Linear(input_size * 16, code_size).to(device)

    def encode(self, x):
        x = x.to(device)
        z = torch.relu(self.layer1(x))
        z = torch.relu(self.layer2(z))
        z = torch.relu(self.layer3(z))
        z = torch.relu(self.layer4(z))

        return z

    def forward(self, x, temp, hard):
        q = self.encode(x)
        q_y = q.view(q.size(0), latent_size, categorical_size)
        z = gumbel_softmax(q_y, temp, hard)

        return z, F.softmax(q_y, dim=-1).reshape(*q.size())


class DecoderGumbel(nn.Module):
    def __init__(self, input_size, code_size):
        super().__init__()

        self.input_size = input_size
        self.code_size = code_size

        self.layer1 = nn.Linear(code_size, input_size * 16).to(device)
        self.layer2 = nn.Linear(input_size * 16, input_size * 4).to(device)
        self.layer3 = nn.Linear(input_size * 4, input_size * 2).to(device)
        self.layer4 = nn.Linear(input_size * 2, input_size).to(device)

    def forward(self, z):
        # print("x.shape ", x.shape)

        x = torch.relu(self.layer1(z))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.sigmoid(self.layer4(x))

        return x