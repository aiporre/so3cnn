import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import util
from . import spherical

class SphConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True, n_filter_params=0):
        super(SphConv, self).__init__()
        self.use_bias = use_bias
        self.n_filter_params = n_filter_params
        self.in_channels = in_channels
        self.out_channels = out_channels

        n = 2  # Assuming n is 2 for simplicity, adjust as needed
        std = 2. / (2 * np.pi * np.sqrt((n // 2) * in_channels))

        if n_filter_params == 0:
            self.weights = nn.Parameter(torch.randn(in_channels, n // 2, out_channels) * std)
        else:
            nw_in = min(n_filter_params, n // 2)
            self.weights = nn.Parameter(torch.randn(in_channels, nw_in, out_channels) * std)
            xw_in = np.linspace(0, 1, nw_in)
            xw_out = np.linspace(0, 1, n // 2)
            id_out = np.searchsorted(xw_in, xw_out)
            subws = []
            for i, x in zip(id_out, xw_out):
                subws.append(self.weights[:, i-1, :] + (self.weights[:, i, :] - self.weights[:, i-1, :]) * (x - xw_in[i-1]) / (xw_in[i] - xw_in[i-1]))
            self.weights = torch.stack(subws, dim=1).unsqueeze(1).unsqueeze(3)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1, 1, 1, out_channels))
        else:
            self.bias = torch.zeros(1, 1, 1, out_channels)

    def forward(self, inputs):
        conv = spherical.sph_conv_batch(inputs, self.weights)
        if self.use_bias:
            conv = conv + self.bias
        return conv

class Block(nn.Module):
    def __init__(self, params, fun, is_training=None):
        super(Block, self).__init__()
        self.params = params
        self.fun = fun
        self.is_training = is_training
        self.use_bias = not params.batch_norm
        self.batch_norm = nn.BatchNorm2d(params.out_channels) if params.batch_norm else None

    def forward(self, *args, **kwargs):
        curr = self.fun(*args, **kwargs, use_bias=self.use_bias)
        if self.params.batch_norm:
            curr = self.batch_norm(curr)
        return nonlin(self.params)(curr)

def nonlin(params):
    return getattr(F, params.nonlin, globals().get(params.nonlin))

def identity(inputs):
    return inputs

class PReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.1):
        super(PReLU, self).__init__()
        self.alphas = nn.Parameter(torch.tensor(init * np.ones(num_parameters), dtype=torch.float32))

    def forward(self, inputs):
        pos = F.relu(inputs)
        neg = self.alphas * (inputs - abs(inputs)) * 0.5
        return pos + neg

def area_weights(x, invert=False):
    n = x.shape[1]
    phi, theta = util.sph_sample(n)
    phi += np.diff(phi)[0] / 2
    if invert:
        x /= torch.sin(torch.tensor(phi)).unsqueeze(0).unsqueeze(0).unsqueeze(3)
    else:
        x *= torch.sin(torch.tensor(phi)).unsqueeze(0).unsqueeze(0).unsqueeze(3)
    return x