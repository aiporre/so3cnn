import torch
import numpy as np
from so3conv.util import safe_cast

def istorch(x):
    """ Check if argument is a PyTorch structure (or list of). """
    if isinstance(x, list):
        x = x[0]
    return isinstance(x, torch.Tensor)

def shape(x):
    """ Return shape of either PyTorch Tensor or numpy array. """
    return tuple(x.shape) if istorch(x) else x.shape

def sum(*args, **kwargs):
    """ Return np.sum or torch.sum according to input. """
    return fun(['sum', 'sum'], *args, **kwargs)

def fun(fun_name, *args, **kwargs):
    """ Return np or torch version of function according to input.

    Args:
        fun_name (list or str): if str, return torch.fun_name or np.fun_name
                                if list, return torch.fun_name[0] or np.fun_name[1]
    """
    if isinstance(fun_name, list):
        f1, f2 = fun_name
    else:
        f1, f2 = fun_name, fun_name
    return (getattr(torch, f1)(*args, **kwargs) if any([istorch(a) for a in args])
            else getattr(np, f2)(*args, **kwargs))

def dot(x, y, *args, **kwargs):
    """ Return np.tensordot or torch.tensordot according to input. """
    if istorch(x) and not istorch(y):
        y = torch.tensor(y)
    if istorch(y) and not istorch(x):
        x = torch.tensor(x)
    if istorch(x) and istorch(y):
        x, y = safe_cast(x, y)
        # change dims to axes
        if 'axes' in kwargs:
            kwargs['dims'] = kwargs.pop('axes')
        return torch.tensordot(x, y, *args, **kwargs)
    else:
        if 'dims' in kwargs:
            kwargs['axes'] = kwargs.pop('dims')
        return np.tensordot(x, y, *args, **kwargs)

def concat(*args, axis=0):
    """ Return np.concatenate or torch.cat according to input. """
    return (torch.cat([*args], dim=axis) if istorch(args[0])
            else np.concatenate(args, axis=axis))

def fft(x, *args, **kwargs):
    """ Return np.fft.fft or torch.fft.fft according to input. """
    return (torch.fft.fft(x, *args, **kwargs) if istorch(x)
            else np.fft.fft(x, *args, **kwargs))

def conj(*args, **kwargs):
    return fun('conj', *args, **kwargs)

def transpose(*args, **kwargs):
    return fun(['permute', 'transpose'], *args, **kwargs)


# need to call

def reshape(*args, **kwargs):
    return fun('reshape', *args, **kwargs)

def real_or_imag(x, part):
    """ Return real or imaginary part of either PyTorch Tensor or numpy array. """
    if istorch(x):
        fun = getattr(torch, part)
        if x.is_complex():
            return fun(x)
        else:
            nbits = x.element_size() * 8
            return fun(x.to(torch.complex64 if nbits == 4 else torch.complex128))
    else:
        fun = getattr(np, part)
        return fun(x)

def real(x):
    return real_or_imag(x, 'real')

def imag(x):
    return real_or_imag(x, 'imag')

