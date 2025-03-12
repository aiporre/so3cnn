from torch import nn
from so3conv.layer import SphConv
from so3conv.params import AttrDict, ArchParams


class SphericalCNN(nn.Module):

    def __init__(self, config_file):
        super(SphericalCNN, self).__init__()
        self.params = ArchParams(config_file)
        layers = nn.ModuleList()
        for i, params in enumerate(self.params.arch_params):
            layer_type = params.layer_type
            if layer_type == 'conv':
                layer = SphConv(params.in_channels, params.out_channels, params.kernel_size, params.stride, params.padding)
            elif layer_type == 'pool':
                layer = nn.MaxPool2d(params.kernel_size, params.stride)
            else:
                raise ValueError('Unknown layer type: {}'.format(layer_type))
            setattr(self, 'block{}_{}'.format(i, layer_type), layer)
            layers.append(layer)
        self.layers = layers

def forward(self, x):
    for layer in self.layers:
        x = layer(x)
    return x



def main():
    n = 2
    in_channels = 1
    out_channels = 1
    use_bias = True
    n_filter_params = 0
    sphconv = SphConv(in_channels, out_channels, use_bias, n_filter_params)
    print(sphconv)

    params = AttrDict()
    params.out_channels = 1
    params.batch_norm = False
    params.nonlin = 'relu'
    params.use_bias = True

    for key, value in params.items():
        print(key, value)

if __name__ == "__main__":
    main()

