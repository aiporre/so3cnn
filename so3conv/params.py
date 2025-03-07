class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class ArchParams:
    """
    This class is used to store the architecture parameters.
    text example
    conv1: 1 32 3 1 1
    conv2: 32 64 3 1 1
    pool1: 2 2
    """
    def __init__(self, arch_file):
        self.arch_file = arch_file
        self.arch_params = self.parse_arch_file()
    def parse_arch_file(self):
        arch_params = []
        with open(self.arch_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split(':')
                if parts[0].startswith('conv'):
                    parts = parts[1].split()
                    params = AttrDict()
                    params.in_channels = int(parts[0])
                    params.out_channels = int(parts[1])
                    params.kernel_size = int(parts[2])
                    params.stride = int(parts[3])
                    params.padding = int(parts[4])
                    arch_params.append(params)
                elif parts[0].startswith('pool'):
                    parts = parts[1].split()
                    params = AttrDict()
                    params.kernel_size = int(parts[0])
                    params.stride = int(parts[1])
                    arch_params.append(params)
        return arch_params
