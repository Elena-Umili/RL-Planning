import torch
import torch.nn as nn
import numpy as np
import torchvision

INPUT_DIM = 2
EPOCHS = 100
LATENT_SPACE = 1
HIDDEN_DIM = 24
f = open("salvapesi.txt", 'w+')

class AE_encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=HIDDEN_DIM
        )
        self.hidden_layer1 = nn.Linear(
            in_features=HIDDEN_DIM, out_features=HIDDEN_DIM
        )
        self.hidden_layer2 = nn.Linear(
            in_features=HIDDEN_DIM, out_features=HIDDEN_DIM
        )
        self.hidden_layer3 = nn.Linear(
            in_features=HIDDEN_DIM, out_features=HIDDEN_DIM
        )

        self.output_layer = nn.Linear(
            in_features=HIDDEN_DIM, out_features=LATENT_SPACE
        )


    def forward(self, features):
        step1 = self.input_layer(features)
        step1 = torch.sigmoid(step1)
        step2 = self.hidden_layer1(step1)
        step2 = torch.sigmoid(step2)
        step3 = self.hidden_layer2(step2)
        step3 = torch.sigmoid(step3)

        step4 = self.hidden_layer3(step3)
        step4 = torch.sigmoid(step4)
        coded = self.output_layer(step4)
        coded = torch.sigmoid(coded)
        return coded


class AE_decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_layer = nn.Linear(
            in_features=LATENT_SPACE, out_features=HIDDEN_DIM
        )
        self.hidden_layer1 = nn.Linear(
            in_features=HIDDEN_DIM, out_features=HIDDEN_DIM
        )
        self.hidden_layer2 = nn.Linear(
            in_features=HIDDEN_DIM, out_features=HIDDEN_DIM
        )
        self.hidden_layer3 = nn.Linear(
            in_features=HIDDEN_DIM, out_features=HIDDEN_DIM
        )

        self.output_layer = nn.Linear(
            in_features=HIDDEN_DIM, out_features=kwargs["input_shape"]
        )


    def forward(self, code):
        step1 = self.input_layer(code)
        step1 = torch.sigmoid(step1)
        step2 = self.hidden_layer1(step1)
        step2 = torch.sigmoid(step2)
        step3 = self.hidden_layer2(step2)
        step3 = torch.sigmoid(step3)
        step4 = self.hidden_layer3(step3)
        step4 = torch.sigmoid(step4)
        reconstructed = self.output_layer(step4)
        reconstructed = torch.sigmoid(reconstructed)
        return reconstructed

class AutoEncoder(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()

        self.enc = enc
        self.dec = dec
    def forward(self,x):
        encoded = self.enc(x)
        encoded = (encoded*100).round()

        #print(encoded)
        decoded = self.dec(encoded)
        print(encoded)
        return decoded


def getData():
    f = open("mountain_car_captions2.txt")
    dataList = []
    while(True):
        line = f.readline()
        if(line != ''):
            line = line.split(",")
            dataList.append(np.asarray([(float(line[0])), (float(line[1]))]))
        else:
            break
    posMin = 100
    posMax = -100
    velMin = 100
    velMax = -100
    for i in range(len(dataList)):
        if(dataList[i][0] > posMax):
            posMax = dataList[i][0]
        if (dataList[i][0] < posMin):
            posMin = dataList[i][0]
        if (dataList[i][1] > velMax):
            velMax = dataList[i][1]
        if (dataList[i][1] < velMin):
            velMin = dataList[i][1]

    for i in range(len(dataList)):
        dataList[i][0] = (dataList[i][0] - posMin)/(posMax - posMin)
        dataList[i][1] = (dataList[i][1] - velMin)/(velMax - velMin)
    l_len = len(dataList)
    trainList = dataList[:int(l_len/2)]
    testList = dataList[int(l_len/2):]
    return dataList,posMax,posMin,velMax,velMin

#  use gpu if available
device = torch.device("cpu")


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
def normalizer(data,posMax,posMin,velMax, velMin):
    data[0] = (data[0] - posMin) / (posMax - posMin)
    data[1] = (data[1] - velMin) / (velMax - velMin)
    return data


def executeNN(data):
    train_dataset,posMax,posMin,velMax,velMin = getData()
    data = normalizer(data,posMax,posMin,velMax,velMin)
    data = torch.FloatTensor(data).to(device)
    print(data,velMax,velMin)
    encoder = AE_encoder(input_shape=INPUT_DIM).to(device)
    decoder = AE_decoder(input_shape=INPUT_DIM).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    model = AutoEncoder(encoder,decoder)

    model.load_state_dict(torch.load('myAEModel_200_buono3.pt'))
    epochs = EPOCHS
    coded = model(data)

    print("coded = ",coded)
    return coded

executeNN([0.4,-0.04])