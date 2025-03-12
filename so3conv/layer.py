import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from so3conv import util
from so3conv import spherical
from so3conv.spherical import SphericalHarmonics


class SphConv(nn.Module):
    def __init__(self, in_channels, out_channels, n, use_bias=True, n_filter_params=0, spectral_pool=0, real=True):
        super(SphConv, self).__init__()
        self.use_bias = use_bias
        self.n_filter_params = n_filter_params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.real = real
        self.spectral_pool = spectral_pool

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

        self.spherical_harmonics = SphericalHarmonics(n, as_torch=True, real=self.real)

        if self.spherical_pooling > 0:
            self.spherical_harmonics_low = SphericalHarmonics(n // 2, as_torch=True, real=self.real)
        else:
            self.spherical_harmonics_low = None

    def forward(self, inputs, *args,**kwargs): # **kwaargs will cotainnthe harmonsi to ube use din lie sphe conv batch
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
        # TODO: here goes the the spectral pooling flag and connect the sherical harmoinxlow
        conv = self.spherical_harmonics.conv_batch(inputs, h, spectral_pool=self.spectral_pool, harmonics_low=self.spherical_harmonics_low) # this is the convolution with harmonics tensor in in kwars
        if self.use_bias:
            conv = conv + self.bias
        return conv

class WeightedGlobalAvgPool(nn.Module):
    """ Global average pooling with weights
    the weights are the sin of the latitude according to Esteves et al. 2018
    this is used instead of the standard global average pooling

    params: dictionary of parameters from text
    pool_wga: avg/max
    """
    def __init__(self, params):
        super(WeightedGlobalAvgPool, self).__init__()
        self.pool_name = params.pool_func
        if params.pool_func == 'avg':
            self.pool_func =nn.AdaptiveAvgPool2d(1)
        elif params.pool_func == 'max':
            self.pool_func = nn.AdaptiveMaxPool2d(1)
        elif params.pool_func == 'spectral':
            self.pool_func = self._spectral_pool

        else:
            raise ValueError('Unknown pool function: {}'.format(params.pool_func))


    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        weights = self._loggitud_weights(x)
        x = x * weights
        # permute to batch, channel, height, width
        if self.pool_name in ['avg', 'max']:
            # the max and avg are the same thing
            x = x.permute(0, 3, 1, 2)
            x = self.pool_func(x)
            x = x.squeeze(3).squeeze(2)
            return x
        else:
            # this for spectral we don't need permutations and so on
            return self.pool_func(x, *args, **kwargs)

    @staticmethod
    def _spectral_pool(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x_sft = spherical.sph_harm_transform_batch(x, m0_only=True, *args, **kwargs)
        x_sft = torch.sum(x_sft ** 2, dim=(1,3))
        x_sft = torch.real(x_sft)
        return x_sft.flatten(start_dim=1)


    @staticmethod
    def _loggitud_weights(x: torch.Tensor) -> torch.Tensor:
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