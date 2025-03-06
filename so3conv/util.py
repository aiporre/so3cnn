import sys

import torch
import numpy as np

try:
    from pyshtools import SHGrid
except ImportError:
    pass


def sph_sample(n, mode='DH'):
    """ Sample grid on a sphere.

    Args:
        n (int): dimension is n x n
        mode (str): sampling mode; DH or GLQ

    Returns:
        theta, phi (1D arrays): polar and azimuthal angles
    """
    assert n % 2 == 0
    j = np.arange(0, n)
    if mode == 'DH':
        return j * np.pi / n, j * 2 * np.pi / n
    elif mode == 'ours':
        return (2 * j + 1) * np.pi / 2 / n, j * 2 * np.pi / n
    elif mode == 'GLQ':
        from pyshtools.shtools import GLQGridCoord
        phi, theta = GLQGridCoord(n - 1)
        # convert latitude to [0, np.pi/2]
        return np.radians(phi + 90), np.radians(theta)
    elif mode == 'naive':
        # repeat first and last points; useful for plotting
        return np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, n)


def sphrot_shtools(f, x, lmax=None, latcols=True):
    """ Rotate function on sphere f by Euler angles x (Z-Y-Z?)  """
    if 'pyshtools' not in sys.modules:
        raise ImportError('pyshtools not available!')

    if latcols:
        f = f.T
    c = SHGrid.from_array(f).expand()
    c_r = c.rotate(*x, degrees=False)
    f_r = c_r.expand(lmax=lmax).to_array()
    if latcols:
        f_r = f_r.T

    return f_r


def tfrecord2np(fname, shape, dtype='float32', get_meta=False, max_records=np.inf):
    """ Load tfrecord containing serialized tensors x and y as numpy arrays. """
    import tensorflow as tf
    example = tf.train.Example()
    X = []
    Y = []
    # dataset may contain angles
    A = []
    meta = []
    for i, record in enumerate(tf.python_io.tf_record_iterator(fname)):
        example.ParseFromString(record)
        f = example.features.feature
        X.append(np.frombuffer(f['x'].bytes_list.value[0], dtype=dtype))
        Y.append(f['y'].int64_list.value[0])
        if f.get('a', None) is not None:
            A.append(np.frombuffer(f['a'].bytes_list.value[0], dtype=dtype))
        if get_meta:
            meta.append({'fname': f['fname'].bytes_list.value[0],
                         'idrot': f['idrot'].int64_list.value[0]})

        if i >= max_records - 1:
            break

    if len(A) > 0:
        # dataset contain angle channel;
        # load 'a' and 'x', fix shapes and concatenate into channels
        shapea = (*shape[:-1], 1)
        A = np.stack(A).reshape(shapea)
        shape = (*shape[:-1], shape[-1] - 1)
        X = np.stack(X).reshape(shape)
        X = np.concatenate([X, A], axis=-1)
    else:
        X = np.stack(X).reshape(shape)

    Y = np.stack(Y)

    assert len(X) == len(Y)

    if get_meta:
        return X, Y, meta
    else:
        return X, Y


def tf_config():
    """ Default tensorflow config. """
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    return config


def safe_cast(x, y):
    """ Cast x to type of y or y to type of x, without loss of precision.

    Works with complex and floats of any precision
    """
    t = 'complex' if (x.dtype.is_complex or y.dtype.is_complex) else 'float'
    s = max(x.dtype.itemsize, y.dtype.itemsize)
    dtype = '{}{}'.format(t, s * 8)
    if dtype.startswith('float'):
        if '32' in dtype:
            dtype = torch.float
        elif '64' in dtype:
            dtype = torch.double
    elif dtype.startswith('complex'):
        if '64' in dtype:
            dtype = torch.complex64
        elif '128' in dtype:
            dtype = torch.complex128
    assert dtype in [torch.float, torch.double, torch.complex64, torch.complex128], "failed to cast {} to {}".format(x.dtype, y.dtype)
    return x.type_as(torch.empty(1, dtype=dtype)), y.type_as(torch.empty(1, dtype=dtype))
    # return x.astype(dtype), y.astype(dtype)



class AttrDict(dict):
    """ Dict that allows access like attributes (d.key instead of d['key']) .

    From: http://stackoverflow.com/a/14620633/6079076
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def shuffle_all(*args):
    """ Do the same random permutation to all inputs. """
    idx = np.random.permutation(len(args[0]))
    return [a[idx] for a in args]