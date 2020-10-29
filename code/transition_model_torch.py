##### Primo transition model provato #########
# Results: soffre di soluzione degenere (se viene trainato abbastanza impara ad abbassare la loss mettendo tutti i codici allo stesso numero e tutti i delta a 0)

# stato discreto: NO


import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loadDataset import loadDatasetLLTransitions



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

    def forward(self, x, epoch, n_epochs):
        #print("x.shape ", x.shape)

        z =   self.layer1(x) 
        z =   torch.selu( self.layer2(z) )
        z =   torch.selu( self.layer3(z) )
        z = torch.sigmoid(  self.layer4(z)  )
        #print("z.shape ", z.shape)
        '''
        if(epoch >= n_epochs/2):
            z = z.where(z < 0.5, torch.ones(self.code_size).to(device))
            z = z.where(z >= 0.5, torch.zeros(self.code_size).to(device))
        '''
        return z

class Decoder(nn.Module):
    def __init__(self, input_size, code_size):
        super().__init__()

        self.input_size = input_size        
        self.code_size = code_size


        self.layer1 = nn.Linear(code_size , input_size*16).to(device)
        self.layer2 = nn.Linear(input_size*16, input_size*4).to(device)
        self.layer3 = nn.Linear(input_size*4, input_size*2).to(device)
        self.layer4 = nn.Linear(input_size*2, input_size).to(device)

    def forward(self, z):
        #print("x.shape ", x.shape)

        x =   self.layer1(z) 
        x =   torch.selu(self.layer2(x)  )
        x =   torch.selu(self.layer3(x)  )
        x =   torch.sigmoid(self.layer4(x) )

        return x

class AE(nn.Module):
     def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
     def forward(self, x, epoch, n_epochs):
        code = self.encoder(x, epoch, n_epochs)
        out = self.decoder(code)
        return out

class TransitionDelta(nn.Module):

    def __init__(self, code_size, action_size):
        super().__init__()
        
        self.code_size = code_size
        self.action_size = action_size
        input_size = code_size + action_size 

        self.layer1 = nn.Linear(input_size , input_size*2).to(device)
        self.layer2 = nn.Linear(input_size*2, code_size).to(device)

    def forward(self, z, action):
       
        #print("z.shape ", z.shape)
        #print("action.shape ",action.shape)
        cat =  torch.cat((z, action), 1)
        #print(cat.shape)
        delta_z = torch.sigmoid(  self.layer1(  cat )  )
        delta_z = torch.tanh(  self.layer2(delta_z))

        return delta_z

class Transition(nn.Module):

    def __init__(self, encoder, transition_delta):
        super().__init__()

        self.encoder = encoder
        self.transition_delta = transition_delta

    def forward(self, x, action, x_prime, epoch, n_epochs):

        z = self.encoder(x, epoch, n_epochs)
        z_prime = self.encoder(x_prime, epoch, n_epochs)
        #print("z.shape ", z.shape)
        #print("z_prime.shape ", z_prime.shape)
        delta_z = self.transition_delta(z, action)
        #print("delta transition")
        #print(delta_z)
        z_prime_hat = z + delta_z

        return z_prime_hat - z_prime

########################## Sizes
state_size = 8
action_size = 4
code_size = 100
########################## Network
enc = Encoder(state_size, code_size)
dec = Decoder(state_size, code_size)
ae = AE(enc, dec)

td = TransitionDelta(code_size, action_size)
tr = Transition(enc, td)

loss_function = nn.MSELoss()
#loss_function = nn.L1Loss()
optimizerTR = optim.SGD(tr.parameters(), lr=0.01)
optimizerAE = optim.Adam(ae.parameters(), lr=0.0001)

########################### Test
print(tr)
params = list(tr.parameters())
print(len(params))
print(params[0].size())

########################### Load data
X, actions, X_prime = loadDatasetLLTransitions()

X = torch.from_numpy(X).float()
actions = torch.from_numpy(actions).float()
X_prime = torch.from_numpy(X_prime).float()

print(X.shape)
print(actions.shape)
print(X_prime.shape)

print("first 50 codes BEFORE AE training")
print(enc(X[0], 0, 5))
print(enc(X[1], 0, 5))
########################### Train AE
batch_size = 100
n_epochs = 200
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(int(X.shape[0]/batch_size)):
        # get the inputs; data is a list of [inputs, labels]
        x = X[i* batch_size:(i+1)*batch_size]

        target = x

        # zero the parameter gradients
        optimizerAE.zero_grad()

        # forward + backward + optimize
        outputs = ae(x, epoch, n_epochs)
        loss = loss_function(outputs, target)
        loss.backward()
        optimizerAE.step()

        # print statistics
        #print(loss.item())
        running_loss += loss.item()
        
    running_loss = running_loss/batch_size 
    print('[%d] loss: %.3f' % (epoch + 1, running_loss ))


print('Finished Training')


print("first 50 codes AFTER ae training")
print(enc(torch.zeros(1, state_size), 5, 5))
print(enc(torch.ones(1, state_size), 5, 5))

########################## Train transition
batch_size = 100
n_epochs = 100
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(int(X.shape[0]/batch_size)):
        # get the inputs; data is a list of [inputs, labels]
        x = X[i* batch_size:(i+1)*batch_size]
        a = actions[i* batch_size:(i+1)*batch_size]
        x_prime = X_prime[i* batch_size:(i+1)*batch_size]

        target = torch.zeros(batch_size, code_size)

        # zero the parameter gradients
        optimizerTR.zero_grad()

        # forward + backward + optimize
        outputs = tr(x, a, x_prime, epoch, n_epochs)
        loss = loss_function(outputs, target)
        loss.backward()
        optimizerTR.step()

        # print statistics
        #print(loss.item())
        running_loss += loss.item()
        
       
    print('[%d] loss: %.3f' % (epoch + 1, running_loss ))


print('Finished Training')


print("first 50 codes AFTER transition training")
print(enc(X[0], 0, 5))
print(enc(X[1], 0, 5))


