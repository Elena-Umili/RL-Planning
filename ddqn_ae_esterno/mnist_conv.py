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
from AutoEncoder import AutoEncoder

# Hyperparameters
one_hot = True
INPUT_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(one_hot):
    code_size = 25
    num_epochs = 200
    batch_size = 128
    lr = 1e-3
    optimizer_cls = optim.Adam
    loss_fn = nn.L1Loss(reduction='mean')

else:
    code_size = 1
    num_epochs = 100
    batch_size = 64
    lr = 1e-3
    optimizer_cls = optim.RMSprop
    loss_fn = nn.L1Loss()
# Load data

f = open('feature_dataset/normalized_lunar.txt')
line = f.readline()
dataset = []

while(line != ''):
    record = line.split(" ")
    record.remove("\n")
    for r in record:
        r = float(r)
    dataset.append(np.asarray(record))
    line = f.readline()




x_train = np.asarray(dataset)


#x_train = x_train.reshape((len(x_train), x_train[0].shape[0],x_train[0].shape[1],1))
x_train = x_train.reshape((len(x_train), INPUT_SIZE))
x_train = x_train.astype('float32')
print("SHAPE = " + str(x_train.shape))

train_loader = torch.utils.data.DataLoader(x_train, shuffle=True, batch_size=batch_size, num_workers=4,drop_last=True)

# Instantiate model
autoencoder = AutoEncoder(code_size)
optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)
# Training loop
for epoch in range(num_epochs):
    print("Epoch %d" % epoch)
    print(train_loader)
    for _, images in enumerate(train_loader):  # Ignore image labels
        #print(images)
        images = images.to(device)
        out, code = autoencoder(Variable(images), epoch, num_epochs, 50)

        optimizer.zero_grad()
        loss = loss_fn(out, images)
        loss.backward()
        optimizer.step()

    print("Loss = " + str(loss.data))

# Try reconstructing on test data

if(one_hot):
    name = "lunar_models/code"+str(code_size)+".pt"
else:
    name = "clusters_mnist/minst_clusterer_conv.pt"

torch.save(autoencoder.state_dict(), name)
f = open("lunar_images/outs.txt", "w")
f2 = open("lunar_images/ins.txt", "w")
for _, images in enumerate(train_loader):  # Ignore image labels
    # print(images)
    images = images.to(device)
    out, code = autoencoder(Variable(images), num_epochs, num_epochs, 50)
    for el in out:
        f.write(str(el) + "\n")
    for el in images:
        f2.write(str(el) + "\n")


