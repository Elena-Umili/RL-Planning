import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

one_hot = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size
        self.enc_linear_1 = nn.Linear(INPUT_SIZE, INPUT_SIZE * 4).to(device)
        self.enc_linear_2 = nn.Linear(INPUT_SIZE * 4, INPUT_SIZE * 16).to(device)
        self.enc_linear_3 = nn.Linear(INPUT_SIZE * 16, INPUT_SIZE * 16).to(device)
        self.enc_linear_4 = nn.Linear(INPUT_SIZE * 16, code_size).to(device)


    def forward(self,data, epoch, n_epochs):
        code = (self.enc_linear_1(data.to('cuda')))
        code = F.selu(self.enc_linear_2(code))
        code = F.selu(self.enc_linear_3(code))
        code = F.sigmoid(self.enc_linear_4(code))
        y = torch.ones(self.code_size).to(device)
        x = torch.zeros(self.code_size).to(device)
        if (epoch >= n_epochs / 2):
            code = code.where(code < 0.5, y)
            code = code.where(code >= 0.5, x)

        return code

class Decoder(nn.Module):
    def __init__(self, code_size):
        super().__init__()
        self.dec_linear_1 = nn.Linear(code_size, INPUT_SIZE * 16).to(device)
        self.dec_linear_2 = nn.Linear(INPUT_SIZE * 16, INPUT_SIZE * 8).to(device)
        self.dec_linear_3 = nn.Linear(INPUT_SIZE * 8, INPUT_SIZE * 4).to(device)
        self.dec_linear_4 = nn.Linear(INPUT_SIZE * 4, INPUT_SIZE).to(device)


    def forward(self, code):
        out = (self.dec_linear_1(code))
        out = F.selu(self.dec_linear_2(out))
        out = F.selu(self.dec_linear_3(out))
        out = (self.dec_linear_4(out))
        return out

class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder, code_size):
        super().__init__()
        self.code_size = code_size
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.001)

    def forward(self, data, epoch, n_epochs):
        code = self.encoder(data, epoch, n_epochs)
        out = self.decoder(code)
        return out, code


INPUT_SIZE = 4
