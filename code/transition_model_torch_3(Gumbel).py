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
from loadDataset import loadDatasetLLTransitions



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################## Sizes
state_size = 8
action_size = 4
latent_size = 10
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
    y = logits + sample_gumbel(logits.size())
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


        self.layer1 = nn.Linear(input_size , input_size*2).to(device)
        self.layer2 = nn.Linear(input_size*2, input_size*4).to(device)
        self.layer3 = nn.Linear(input_size*4, input_size*16).to(device)
        self.layer4 = nn.Linear(input_size*16, code_size).to(device)

    def encode(self, x):

        z =   torch.relu(self.layer1(x) )
        z =   torch.relu( self.layer2(z) )
        z =   torch.relu( self.layer3(z) )
        z = torch.relu(  self.layer4(z)  )

        return z

    def forward(self, x, temp, hard):

        q =   self.encode(x) 
        q_y = q.view(q.size(0), latent_size, categorical_size)
        z = gumbel_softmax(q_y, temp, hard)

        return z, F.softmax(q_y, dim=-1).reshape(*q.size())


class DecoderGumbel(nn.Module):
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

        x =   torch.relu( self.layer1(z) )
        x =   torch.relu(self.layer2(x)  )
        x =   torch.relu(self.layer3(x)  )
        x =   torch.sigmoid(self.layer4(x) )

        return x

class GumbelVae(nn.Module):
     def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

     def forward(self, x, temp, hard):
        code, qy = self.encoder(x, temp, hard)
        out = self.decoder(code)
        return out, qy

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

    def __init__(self, encoder, decoder, transition_delta):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.transition_delta = transition_delta

    def forward(self, x, action, x_prime, temp, hard):

        z, qy = self.encoder(x, temp, hard)
        z_prime, qy_prime = self.encoder(x_prime, temp, hard)

        delta_z = self.transition_delta(z, action)

        z_prime_hat = z + delta_z

        x_prime_hat = self.decoder(z_prime_hat)

        error_z = z_prime_hat - z_prime

        return  error_z, x_prime_hat, qy, qy_prime


########################## Network
enc = EncoderGumbel(state_size, latent_size, categorical_size)
dec = DecoderGumbel(state_size, code_size)
gvae = GumbelVae(enc, dec)

td = TransitionDelta(code_size, action_size)
tr = Transition(enc, dec, td)

#loss_function = nn.MSELoss()

################## Loss
def loss_function_gumbel(recon_x, x, qy):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, state_size), size_average=False) / x.shape[0]
    l = nn.MSELoss()
    RE = l(recon_x, x)
    log_ratio = torch.log(qy * categorical_size + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return RE + KLD

optimizerTR = optim.Adam(tr.parameters(), lr=0.0001)
optimizerAE = optim.Adam(gvae.parameters(), lr=0.0001)

def loss_function_transition(error_z, x_prime, recon_x_prime, qy, qy_prime):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, state_size), size_average=False) / x.shape[0]
    l = nn.MSELoss()
    RE = l(recon_x_prime, x_prime)
    E = torch.norm(error_z)
    log_ratio = torch.log(qy * categorical_size + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return RE + KLD + E

optimizerTR = optim.Adam(tr.parameters(), lr=0.0001)
optimizerAE = optim.Adam(gvae.parameters(), lr=0.0001)

########################### Test
'''
print(tr)
params = list(tr.parameters())
print(len(params))
print(params[0].size())
'''
########################### Load data
X, actions, X_prime = loadDatasetLLTransitions()

X = torch.from_numpy(X).float()
actions = torch.from_numpy(actions).float()
X_prime = torch.from_numpy(X_prime).float()

print(X.shape)
print(actions.shape)
print(X_prime.shape)

print("first 50 codes BEFORE AE training")
zero, q = enc(torch.zeros(1, state_size), temp_min, False)
print(['%.0f' % zero[0,i] for i in range(zero.shape[1]) ])
print(' ')
one, q = enc(torch.ones(1, state_size), temp_min, False)
print(['%.0f' % one[0,i] for i in range(one.shape[1]) ])

########################### Train AE
batch_size = 100
n_epochs = 50
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    temp = 1.0
    for i in range(int(X.shape[0]/batch_size)):
        # get the inputs; data is a list of [inputs, labels]
        x = X[i* batch_size:(i+1)*batch_size]

        target = x

        # zero the parameter gradients
        optimizerAE.zero_grad()

        # forward + backward + optimize
        recon_batch, qy = gvae(x, temp, False)
        loss = loss_function_gumbel(recon_batch, x, qy)

        loss.backward()
        optimizerAE.step()

        # print statistics
        #print(loss.item())
        running_loss += loss.item()
        
        if i % 5 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * i), temp_min)

    running_loss = running_loss/batch_size 
    print('[%d] loss: %.3f' % (epoch + 1, running_loss ))


print('Finished Training')


print("opposite codes AFTER ae training")
zero, q = enc(torch.zeros(1, state_size), temp_min, True)
print(['%.0f' % zero[0,i] for i in range(zero.shape[1]) ])
print(' ')
one, q = enc(torch.ones(1, state_size), temp_min, True)
print(['%.0f' % one[0,i] for i in range(one.shape[1]) ])


########################## Train transition
batch_size = 100
n_epochs = 500
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    temp = 1.0
    for i in range(int(X.shape[0]/batch_size)):
        # get the inputs; data is a list of [inputs, labels]
        x = X[i* batch_size:(i+1)*batch_size]
        a = actions[i* batch_size:(i+1)*batch_size]
        x_prime = X_prime[i* batch_size:(i+1)*batch_size]

        target_z = torch.zeros(batch_size, code_size) 
        target_x = x_prime

        # zero the parameter gradients
        optimizerTR.zero_grad()

        # forward + backward + optimize
        error_z, x_prime_hat, qy, qy_prime = tr(x, a, x_prime, temp, False)
        #err_z , gen_x = tr(x, a, x_prime, epoch, n_epochs)
        loss = loss_function_transition(error_z, x_prime, x_prime_hat, qy, qy_prime)

    
        loss.backward()
        optimizerTR.step()

        # print statistics
        #print(loss.item())
        running_loss += loss.item()
        
        if i % 5 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * i), temp_min)  
    running_loss = running_loss/batch_size      
    print('[%d] loss: %.3f' % (epoch + 1, running_loss ))


print('Finished Training')


print("opposite codes AFTER transition training")
zero, q = enc(torch.zeros(1, state_size), temp_min, True)
print(['%.0f' % zero[0,i] for i in range(zero.shape[1]) ])
print(' ')
one, q = enc(torch.ones(1, state_size), temp_min, True)
print(['%.0f' % one[0,i] for i in range(one.shape[1]) ])


##### scrivere action schema ####
class ActionSchema:

   def __init__(self):
     self.actions = []

   def add_action(a):
     self.actions.append(a)

   def __init__(self, transition_delta_model)

class ActionStrips:

   def __init__(self, name, pre, eff):
     self.name = name
     self.preconditions = pre
     self.effects = eff

   def __init__(self, prec_codes, delta_code):
     #prec_codes: ?xcode_size numpy of all the codes that results in the same delta_code 

   
class Precondition:
    def __init__(self, positive = [], negative = []):
      self.positive = positive
      self.negative = negative

class Postcondition:
   def __init__(self, add_sym = [], del_sym = []):
     self.add_sym = add_sym
     self.del_sym = del_sym






