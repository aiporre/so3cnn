import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from so3conv import util
from so3conv import spherical

class SphConv(nn.Module):
    def __init__(self, in_channels, out_channels, n, use_bias=True, n_filter_params=0):
        super(SphConv, self).__init__()
        self.use_bias = use_bias
        self.n_filter_params = n_filter_params
        self.in_channels = in_channels
        self.out_channels = out_channels

        # n = 2  # Assuming n is 2 for simplicity, adjust as needed
        # as explained in the paper (Esteves et al 2018,) this is the standard initialization
        # they dein the variac to stay constant for every layer
        # in_channels is because we sum over the input channels
        # 2pi is the integral of the spherical harmonics over the so(3)
        # the 2 takes into account the non-zero mean
        # as in paper (He et al. 2015)
        std = 2. / (2 * np.pi * np.sqrt((n // 2) * in_channels))

        if n_filter_params == 0:
            # filters for all the n spherical frequencies
            self.weights = nn.Parameter(torch.randn(in_channels, n // 2, out_channels) * std)
        else:
            # induces the zonal filters
            nw_in = min(n_filter_params, n // 2)
            self.weights = nn.Parameter(torch.randn(in_channels, nw_in, out_channels) * std)
            self.xw_in = np.linspace(0, 1, nw_in)
            self.xw_out = np.linspace(0, 1, n // 2)
            self.id_out = np.searchsorted(self.xw_in, self.xw_out)
            # subws = []
            # # does the interpolation
            # for i, x in zip(id_out, xw_out):
            #     subws.append(self.weights[:, i-1, :] + (self.weights[:, i, :] - self.weights[:, i-1, :]) * (x - xw_in[i-1]) / (xw_in[i] - xw_in[i-1]))
            # self.weights = torch.stack(subws, dim=1).unsqueeze(1).unsqueeze(3)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1, 1, 1, out_channels))
        else:
            self.bias = torch.zeros(1, 1, 1, out_channels)

    def forward(self, inputs, *args,**kwargs):
        if self.n_filter_params == 0:
            h = self.weights.unsqueeze(1).unsqueeze(3)
        else:
            # interpolate
            subws = []
            # does the interpolation
            for i, x in zip(self.id_out, self.xw_out):
                w_minus_1 = self.weights[:, i-1, :]
                w_zero = self.weights[:, i, :]
                alpha = (x - self.xw_in[i-1]) / (self.xw_in[i] - self.xw_in[i-1])
                subws.append(w_minus_1 + (w_zero - w_minus_1) * alpha)
                # subws.append(self.weights[:, i-1, :] + (self.weights[:, i, :] - self.weights[:, i-1, :]) * (x - xw_in[i-1]) / (xw_in[i] - xw_in[i-1]))
            h = torch.stack(subws, dim=1).unsqueeze(1).unsqueeze(3)
        conv = spherical.sph_conv_batch(inputs, h, *args, **kwargs)
        if self.use_bias:
            conv = conv + self.bias
        return conv

class WeightedGlobalAvgPool(nn.Module):
    """ Global average pooling with weights
    the weights are the sin of the latitude according to Esteves et al. 2018
    this is used instead of the standard global average pooling

    params: dictionary of parameters from text
    pool_wga: avg/sum/max
    """
    def __init__(self, params):

        self.pool_func = getattr(F, params.pool_func)
        super(WeightedGlobalAvgPool, self).__init__()

    def forward(self, inputs):
        weights = self._loggitud_weights(inputs)
        return self.pool_func(inputs * weights, dim=(2, 3), keepdim=True)

    @staticmethod
    def _loggitud_weights(x):
        n = x.shape[1]
        phi, theta = util.sph_sample(n)
        phi += np.diff(phi)[0] / 2
        return torch.sin(torch.tensor(phi)).unsqueeze(0).unsqueeze(0).unsqueeze(3)

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

# test internal main
def main():
    n = 2
    in_channels = 1
    out_channels = 1
    use_bias = True
    n_filter_params = 0
    sphconv = SphConv(in_channels, out_channels, use_bias, n_filter_params)
    print(sphconv)

    params = util.AttrDict()
    params.out_channels = 1
    params.batch_norm = False
    params.nonlin = 'relu'
    params.use_bias = True
    params.filter = 'sphconv'
    params.filter_params = 0
    params.filter_size = 5
    params.filter_stride = 1
    params.filter_pool = 2
    params.filter_nonlin = 'relu'
    params.filter_batch_norm = False
    params.filter_use_bias = True

if __name__ == '__main__':
    main()