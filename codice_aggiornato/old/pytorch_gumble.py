# Code to implement VAE-gumple_softmax in pytorch
# author: Devinder Kumar (devinder.kumar@uwaterloo.ca), modified by Yongfei Yan
# The code has been modified from pytorch example vae code and inspired by the origianl \
# tensorflow implementation of gumble-softmax by Eric Jang.

import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from old.Gumbel_AE import VAE_gumbel, VAE_gumbel_enc, VAE_gumbel_dec, latent_dim, categorical_dim

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--temp', type=float, default=1.0, metavar='S',
                    help='tau(temperature) (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hard', action='store_true', default=True,
                    help='hard Gumbel softmax')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
INPUT_SIZE = 8

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

f = open('feature_dataset/lunar.txt')
line = f.readline()
dataset = []
i = 0
while(line != ''):
    line = line.replace("\n", "")
    record = line.split(" ")
    print(i)
    print(record)
    if "" in record:
        record.remove("")
    for r in record:
        r = float(r)

    dataset.append(np.asarray(record))
    line = f.readline()
    i += 1

x_train = np.asarray(dataset)

#x_train = x_train.reshape((len(x_train), x_train[0].shape[0],x_train[0].shape[1],1))
x_train = x_train.reshape((len(x_train), INPUT_SIZE))
x_train = x_train.astype('float32')
print("SHAPE = " + str(x_train.shape))
train_loader = torch.utils.data.DataLoader(x_train, shuffle=True, batch_size=args.batch_size, num_workers=4,drop_last=True)

temp_min = 0.5
ANNEAL_RATE = 0.00003
model_enc = VAE_gumbel_enc()
model_dec = VAE_gumbel_dec()
model = VAE_gumbel(args.temp, model_enc, model_dec)
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy):
    '''
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, INPUT_SIZE), size_average=False) / x.shape[0]

    log_ratio = torch.log(qy * categorical_dim + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return (BCE + KLD)*100
    '''

loss_fn = nn.L1Loss(reduction='mean')
def train(epoch):
    model.train()
    train_loss = 0
    temp = args.temp
    for batch_idx,data in enumerate(train_loader):

        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, qy, code = model(data, temp, args.hard, epoch, args.epochs)
        #loss = loss_function(recon_batch, data, qy)
        loss = loss_fn(recon_batch,data)
        loss.backward()
        train_loss += loss.item() * len(data)
        optimizer.step()

        if batch_idx % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(x_train)))


def test():
    f = open("res/outs.txt", "w")
    f2 = open("res/ins.txt", "w")
    for _, data in enumerate(train_loader):  # Ignore image labels
        # print(images)
        data = data.to('cuda')
        temp = args.temp
        out, qy, code = model(data, temp, args.hard, 10, args.epochs)

        for el in out:
            f.write(str(el) + "\n")
        for el in data:
            f2.write(str(el) + "\n")
        break


def run():
    for epoch in range(1, args.epochs + 1):
        train(epoch)

    #test()
    name_model = "lunar_models/gumbel_no_norm_"+str(latent_dim)+"*"+str(categorical_dim)+".pt"
    name_enc = "lunar_models/gumbel_no_norm_" + str(latent_dim) + "*" + str(categorical_dim) + "_enc.pt"
    name_dec = "lunar_models/gumbel_no_norm_" + str(latent_dim) + "*" + str(categorical_dim) + "_dec.pt"
    torch.save(model.state_dict(), name_model)
    torch.save(model.encoder.state_dict(), name_enc)
    torch.save(model.decoder.state_dict(), name_dec)


if __name__ == '__main__':
    run()