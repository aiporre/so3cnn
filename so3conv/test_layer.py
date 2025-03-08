from  unittest import TestCase

import torch

from so3conv.layer import SphConv
from so3conv.params import AttrDict


class TestLayer(TestCase):
    def test_sphconv_non_localized(self):
        n = 32
        batch_size = 10
        in_channels = 16
        out_channels = 32
        use_bias = True
        n_filter_params = 0 # no-localized filters
        sphconv = SphConv(in_channels, out_channels, n, use_bias, n_filter_params)
        print(sphconv)

        params = AttrDict()
        params.out_channels = 1
        params.batch_norm = False
        params.nonlin = 'relu'
        params.use_bias = True

        for key, value in params.items():
            print(key, value)
        # test forward
        # x is batch channel height width
        x = torch.rand(batch_size,n,n,in_channels)
        y = sphconv(x)
        print(y.shape)
        print('forward works')
        # test backward
        loss = torch.abs(y.sum())
        print('loss', loss)
        loss.backward()
        print('backward works')
    def test_sphconv_localized(self):

        n = 32
        batch_size = 10
        in_channels = 16
        out_channels = 32
        use_bias = True
        n_filter_params = 10 # no-localized filters
        sphconv = SphConv(in_channels, out_channels, n, use_bias, n_filter_params)
        print(sphconv)

        params = AttrDict()
        params.out_channels = 1
        params.batch_norm = False
        params.nonlin = 'relu'
        params.use_bias = True

        for key, value in params.items():
            print(key, value)
        # test forward
        # x is batch channel height width
        x = torch.rand(batch_size,n,n,in_channels)
        y = sphconv(x)
        print(y.shape)
        print('forward works')
        # test backward
        loss = torch.abs(y.sum())
        print('loss', loss)
        loss.backward()
        print('backward works')

