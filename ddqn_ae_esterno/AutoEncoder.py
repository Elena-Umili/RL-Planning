import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
import glob
from torchsummary import summary
from torchvision import datasets, transforms
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

one_hot = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AutoEncoder(nn.Module):

    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size

        # Encoder specification
        self.enc_linear_1 = nn.Linear(INPUT_SIZE, INPUT_SIZE*4).to(device)
        self.enc_linear_2 = nn.Linear(INPUT_SIZE*4, INPUT_SIZE*16).to(device)
        self.enc_linear_3 = nn.Linear(INPUT_SIZE*16, INPUT_SIZE*16).to(device)
        self.enc_linear_4 = nn.Linear(INPUT_SIZE*16, self.code_size).to(device)

        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, INPUT_SIZE*16).to(device)
        self.dec_linear_2 = nn.Linear(INPUT_SIZE*16, INPUT_SIZE*8).to(device)
        self.dec_linear_3 = nn.Linear(INPUT_SIZE*8, INPUT_SIZE*4).to(device)
        self.dec_linear_4 = nn.Linear(INPUT_SIZE*4, INPUT_SIZE).to(device)

    def forward(self, images, epoch, n_epochs, clusters):
        code = self.encode(images)
        y = torch.ones(self.code_size).to(device)
        x = torch.zeros(self.code_size).to(device)

        if(one_hot):
            if(epoch >= n_epochs/2):
                code = code.where(code < 0.5, y)
                code = code.where(code >= 0.5, x)
        else:
            if (epoch >= n_epochs / 2):
                code = code * clusters
                code = code.round()
        out = self.decode(code)
        return out, code

    def encode(self, images):
        code = (self.enc_linear_1(images))
        code = F.selu(self.enc_linear_2(code))
        code = F.selu(self.enc_linear_3(code))
        code = F.sigmoid(self.enc_linear_4(code))
        return code

    def decode(self, code):
        out = (self.dec_linear_1(code))
        out = F.selu(self.dec_linear_2(out))
        out = F.selu(self.dec_linear_3(out))
        out = (self.dec_linear_4(out))
        return out


INPUT_SIZE = 8
