import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 100
EPOCHS = 50
CUDA = False
TEMP = 1.0
SEED = 1
LOG_INTERVAL = 10
HARD = True
INPUT_SIZE = 8
latent_dim = 10
categorical_dim = 10  # one-of-K vector
class VAE_gumbel_enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_linear_1 = nn.Linear(INPUT_SIZE, INPUT_SIZE * 4).to('cpu')
        self.enc_linear_2 = nn.Linear(INPUT_SIZE * 4, INPUT_SIZE * 16).to('cpu')
        self.enc_linear_3 = nn.Linear(INPUT_SIZE * 16, INPUT_SIZE * 16).to('cpu')
        self.enc_linear_4 = nn.Linear(INPUT_SIZE * 16, latent_dim * categorical_dim).to('cpu')

    def forward(self, x, temp, hard):
        code = (self.enc_linear_1(x))
        code = F.selu(self.enc_linear_2(code))
        code = F.selu(self.enc_linear_3(code))
        code = F.selu(self.enc_linear_4(code))
        #print(code.size())
        q_y = code.view(code.size(), latent_dim, categorical_dim).to('cpu')
        z = gumbel_softmax(code, temp, hard).to('cpu')

        return code, q_y, z

class VAE_gumbel_dec(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_linear_1 = nn.Linear(latent_dim * categorical_dim, INPUT_SIZE * 16).to('cpu')
        self.dec_linear_2 = nn.Linear(INPUT_SIZE * 16, INPUT_SIZE * 8).to('cpu')
        self.dec_linear_3 = nn.Linear(INPUT_SIZE * 8, INPUT_SIZE * 4).to('cpu')
        self.dec_linear_4 = nn.Linear(INPUT_SIZE * 4, INPUT_SIZE).to('cpu')
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = (self.dec_linear_1(z))
        out = F.selu(self.dec_linear_2(out))
        out = F.selu(self.dec_linear_3(out))
        out = (self.dec_linear_4(out))

        return (out)


class VAE_gumbel(nn.Module):
    def __init__(self, temp, encoder, decoder):
        super(VAE_gumbel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, temp, hard, epoch, n_epochs):
        q,q_y,z = self.encoder(x.view(-1, INPUT_SIZE), temp, hard)


        return self.decoder(z), F.softmax(q_y, dim=-1).reshape(*q.size()),z



def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if CUDA:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits.to('cpu') + sample_gumbel(logits.size()).to('cpu')
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)

