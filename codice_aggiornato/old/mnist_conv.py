import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from old.AutoEncoder import AutoEncoder,Encoder,Decoder

# Hyperparameters
one_hot = True
INPUT_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

code_size = 100
num_epochs = 50
batch_size = 128
lr = 1e-3
optimizer_cls = optim.Adam
loss_fn = nn.L1Loss(reduction='mean')


f = open('feature_dataset/lunar.txt')
line = f.readline()
dataset = []

while(line != ''):
    line = line.replace("\n", "")
    record = line.split(" ")
    if "" in record:
        record.remove("")
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
encoder = Encoder(code_size)
decoder = Decoder(code_size)
autoencoder = AutoEncoder(encoder,decoder, code_size)
optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)
# Training loop
for epoch in range(num_epochs):
    print("Epoch %d" % epoch)
    print(train_loader)
    for _, images in enumerate(train_loader):  # Ignore image labels
        #print(images)
        images = images.to(device)
        out, code = autoencoder(Variable(images), epoch, num_epochs)

        optimizer.zero_grad()
        loss = loss_fn(out, images)
        loss.backward()
        optimizer.step()

    print("Loss = " + str(loss.data))

# Try reconstructing on test data

name = "lunar_models/code"+str(code_size)+".pt"
enc_name = "lunar_models/code"+str(code_size)+"_enc.pt"
dec_name = "lunar_models/code"+str(code_size)+"_dec.pt"

torch.save(autoencoder.state_dict(), name)
torch.save(autoencoder.encoder.state_dict(), enc_name)
torch.save(autoencoder.decoder.state_dict(),dec_name)
f = open("res/outs.txt", "w")
f2 = open("res/ins.txt", "w")
for _, images in enumerate(train_loader):  # Ignore image labels
    # print(images)
    images = images.to(device)
    out, code = autoencoder(Variable(images), num_epochs, num_epochs)
    for el in out:
        f.write(str(el) + "\n")
    for el in images:
        f2.write(str(el) + "\n")
    break

ex_vec = np.zeros(8)


print(autoencoder.encoder(torch.from_numpy(ex_vec).type(torch.FloatTensor).to(device),50,50))

