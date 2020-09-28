import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self, input_size, code_size):
        super().__init__()

        self.input_size = input_size        
        self.code_size = code_size


        self.layer1 = nn.Linear(input_size , input_size*2).to(device)
        self.layer2 = nn.Linear(input_size*2, input_size*4).to(device)
        self.layer3 = nn.Linear(input_size*4, input_size*16).to(device)
        self.layer4 = nn.Linear(input_size*16, code_size).to(device)

    def forward(self, x):
        #print("x.shape ", x.shape)
        z = F.sigmoid(  self.layer1(x)  )
        z = F.sigmoid(  self.layer2(z)  )
        z = F.sigmoid(  self.layer3(z)  )
        z = F.sigmoid(  self.layer4(z)  )
        #print("z.shape ", z.shape)
        return z


class TransitionDelta(nn.Module):

    def __init__(self, code_size, action_size):
        super().__init__()
        
        self.code_size = code_size
        self.action_size = action_size
        input_size = code_size + action_size 

        self.layer1 = nn.Linear(input_size , input_size*2).to(device)
        self.layer2 = nn.Linear(input_size*2, code_size).to(device)

    def forward(self, z, action):
       
        print("z.shape ", z.shape)
        print("action.shape ",action.shape)
        cat =  torch.cat((z, action), 1)
        print(cat.shape)
        delta_z = F.sigmoid(  self.layer1(  cat )  )
        delta_z = F.tanh(  self.layer2(delta_z))

        return delta_z
     
class Transition(nn.Module):

    def __init__(self, encoder, transition_delta):
        super().__init__()

        self.encoder = encoder
        self.transition_delta = transition_delta

    def forward(self, x, action, x_prime):

        z = self.encoder(x)
        z_prime = self.encoder(x_prime)
        #print("z.shape ", z.shape)
        #print("z_prime.shape ", z_prime.shape)
        delta_z = self.transition_delta(z, action)
    
        z_prime_hat = z + delta_z

        return z_prime_hat - z_prime

state_size = 8
action_size = 4
code_size = 100

enc = Encoder(state_size, code_size)

td = TransitionDelta(state_size, action_size)

tr = Transition(enc, td)

print(tr)
params = list(tr.parameters())
print(len(params))
print(params[0].size())

x = torch.randn(1,state_size)
a = torch.randn(1,action_size)
x_prime = torch.randn(1,state_size)

out = tr(x, a, x_prime)
print(out)



