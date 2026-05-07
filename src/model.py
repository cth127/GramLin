from torch import nn
import torch
import math


class AbstractModel(nn.Module):
    def __init__(self):
        super(AbstractModel, self).__init__()
        self.layers = None
    
    def __call__(self, x, return_hidden=False):
        return self.forward(x, return_hidden)

    def forward(self, x, return_hidden=False):
        raise NotImplementedError


class DNN(AbstractModel):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_hidden,
                 bias):
        super(DNN, self).__init__()
        self.num_hidden = num_hidden
        self.layers = nn.Sequential()
        for h in range(num_hidden):
            if h == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
        self.nonlin = nn.GELU()

    def forward(self, x, return_hidden):
        hiddens_ret = tuple()
        hidden = x.view(x.size(0), -1)
        for n, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if n + 1 < len(self.layers):
                hidden = self.nonlin(hidden)
                hiddens_ret += (hidden, )
            else:
                output = hidden
        if return_hidden:
            return output, hiddens_ret
        else:
            return output
        

class CNN(AbstractModel):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 bias,
                 num_maxpool=None, 
                 maxpool_layer=None,
                 in_channels=3, 
                 out_channels=16, 
                 num_conv=1, 
                 kernel_size=3,):
        super(CNN, self).__init__()
        self.layers = nn.Sequential()
        for i in range(num_conv):
            if i == 0:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=bias)
            else:
                conv = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=bias)
            self.layers.append(conv)
        self.nonlin = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.num_maxpool = num_maxpool if num_maxpool is not None else min(num_conv, math.log2(input_dim[0]) - 3)
        if maxpool_layer == None:
            self.maxpool_layer = list(range(self.num_maxpool))
        else:
            assert self.num_maxpool == len(maxpool_layer)
            assert len(self.layers) >= len(maxpool_layer)
            self.maxpool_layer = maxpool_layer
        output_shape = int(input_dim[0] / (2 ** self.num_maxpool))
        fc = nn.Linear(out_channels * (output_shape ** 2), output_dim, bias=bias)
        self.layers.append(fc)
        self.unfold = nn.Unfold(kernel_size)

    def forward(self, x, return_hidden):
        hiddens = ()
        hidden = x
        for n, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if n in self.maxpool_layer:   
                hidden = self.maxpool(hidden)
            if n+1 != len(self.layers): 
                hidden = self.nonlin(hidden)
                hiddens += (hidden, )
            if n+2 == len(self.layers):      
                hidden = hidden.view(x.size(0), -1)
        if return_hidden:
            return hidden, hiddens
        else:
            return hidden


class VAE(nn.Module):
    def __init__(self,
                 in_channels,
                 input_size,
                 hidden_dim,
                 latent_dim,
                 num_hidden):
        """
        MLP-based VAE.
        Args:
            in_channels: number of image channels (1 for MNIST, 3 for CIFAR)
            input_size: spatial size H=W (28 for MNIST, 32 for CIFAR)
            hidden_dim: hidden layer width
            latent_dim: latent space dimension
            num_hidden: number of hidden layers in encoder/decoder
        """
        super(VAE, self).__init__()
        self.input_dim = in_channels * input_size * input_size
        self.latent_dim = latent_dim
        self.nonlin = nn.GELU()

        # Encoder
        enc_layers = []
        for i in range(num_hidden):
            enc_layers.append(nn.Linear(self.input_dim if i == 0 else hidden_dim, hidden_dim))
            enc_layers.append(nn.GELU())
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        dec_layers = []
        for i in range(num_hidden):
            dec_layers.append(nn.Linear(latent_dim if i == 0 else hidden_dim, hidden_dim))
            dec_layers.append(nn.GELU())
        dec_layers.append(nn.Linear(hidden_dim, self.input_dim))
        dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x.view(x.size(0), -1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, return_hidden=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        if return_hidden:
            return recon, mu, logvar, z
        return recon, mu, logvar
