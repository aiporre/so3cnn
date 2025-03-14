import functools

import torch
import numpy as np
from scipy.special import sph_harm

from so3conv.util import safe_cast
from so3conv import util

from so3conv.tensor_np_compatible import istorch
from so3conv import tensor_np_compatible as tnp

class SphericalHarmonics:
    """ Spherical harmonics class. """
    def __init__(self, n, as_torchvar=False, real=False):
        self.n = n
        self.as_torchvar = as_torchvar
        self.real = real
        self.harmonics = sph_harm_all(n, as_torchvar=as_torchvar, real=real)

    def transform(self, f):
        return sph_harm_transform(f, harmonics=self.harmonics)

    def inverse(self, c):
        return sph_harm_inverse(c, harmonics=self.harmonics)

    def transform_batch(self, f, m0_only=False):
        return sph_harm_transform_batch(f, harmonics=self.harmonics, m0_only=m0_only)

    def inverse_batch(self, c):
        return sph_harm_inverse_batch(c, harmonics=self.harmonics)

    def conv(self, f, g):
        return sph_conv(f, g, harmonics=self.harmonics)

    def conv_batch(self, f, g, spectral_pool=0, harmonics_low=None):
        return sph_conv_batch(f, g, harmonics_or_legendre=self.harmonics, spectral_pool=spectral_pool, harmonics_or_legendre_low=harmonics_low)

    def __repr__(self):
        return "SphericalHarmonics(n={}, as_torchvar={}, real={})".format(self.n, self.as_torchvar, self.real)

    def __str__(self):
        return self.__repr__()

# cache outputs; 2050 > 32*64
@functools.lru_cache(maxsize=2050, typed=False)
def sph_harm_lm(l, m, n):
    """ Wrapper around scipy.special.sph_harm. Return spherical harmonic of degree l and order m. """
    phi, theta = util.sph_sample(n)
    phi, theta = np.meshgrid(phi, theta)
    f = sph_harm(m, l, theta, phi)

    return f

def sph_harm_all(n, as_torchvar=False, real=False):
    """ Compute spherical harmonics for an n x n input (degree up to n // 2)

    Args:
        n (int): input dimensions; order will be n // 2
        as_torchvar (bool): if True, return as list of torch Variables.
        real (bool): if True, return real harmonics
    """
    harmonics = []

    for l in range(n // 2):
        if real:
            minl = 0
        else:
            minl = -l
        row = []
        for m in range(minl, l+1):
            row.append(sph_harm_lm(l, m, n))
        harmonics.append(row)

    if as_torchvar:
        return torch.tensor(sph_harm_to_shtools(harmonics), dtype=torch.complex64)
    else:
        return harmonics

def DHaj(n, mode='DH'):
    """ Sampling weights. """
    if mode == 'DH':
        gridfun = lambda j: np.pi * j / n
    elif mode == 'ours':
        gridfun = lambda j: np.pi * (2 * j + 1) / 2 / n
    else:
        raise NotImplementedError()

    l = np.arange(0, n / 2)
    a = [(2 * np.sqrt(2) / n *
          np.sin(gridfun(j)) *
          (1 / (2 * l + 1) * np.sin((2 * l + 1) * gridfun(j))).sum())
         for j in range(n)]

    return a

def sph_harm_transform(f , mode = 'DH' , harmonics =None):
    """ Project spherical function into the spherical harmonics basis. """
    assert f.shape[0] == f.shape[1]
    is_tensor = isinstance(f, torch.Tensor)
    if is_tensor:
        sumfun = torch.sum
        conjfun = lambda x: torch.conj(x)
        n = f.shape[0]
    else:
        sumfun = np.sum
        conjfun = np.conj
        n = f.shape[0]
    assert np.log2(n).is_integer()

    if harmonics is None:
        harmonics = sph_harm_all(n)
        if is_tensor:
            harmonics = [torch.tensor(h, dtype=torch.complex128) for h in harmonics]

    a = DHaj(n, mode)

    # f = f * np.array(a)[np.newaxis, :]
    if is_tensor:
        f = f * torch.tensor(a, dtype=torch.float).unsqueeze(0)
    else:
        f = f * np.array(a)[np.newaxis, :]

    real = is_real_sft(harmonics)

    coeffs = []
    for l in range(n // 2):
        row = []
        minl = 0 if real else -l
        for m in range(minl, l + 1):
            factor = 2 * np.sqrt(np.pi)
            row.append(sumfun(factor * np.sqrt(2 * np.pi) / n * f * conjfun(harmonics[l][m - minl])))
        coeffs.append(row)

    return coeffs

def sph_harm_inverse(c, harmonics=None):
    """ Inverse spherical harmonics transform. """
    n = 2 * len(c)

    real = is_real_sft(c)
    dtype = torch.float32 if real else c[1][1].dtype
    if harmonics is None:
        harmonics = sph_harm_all(n, real=real)
        if istorch(c):
            harmonics = [torch.tensor(h, dtype=torch.complex64) for h in harmonics]

    if isinstance(c[0][0], torch.Tensor):
        f = torch.zeros((n, n), dtype=dtype)
    else:
        f = np.zeros((n, n), dtype=dtype)

    for l in range(n // 2):
        lenm = l + 1 if real else 2 * l + 1
        for m in range(lenm):
            if real:
                factor = 1 if m == 0 else 2
                f += factor * (tnp.real(c[l][m]) * tnp.real(harmonics[l][m]) -
                               tnp.imag(c[l][m]) * tnp.imag(harmonics[l][m]))
            else:
                f += c[l][m] * harmonics[l][m]

    return f

def sph_harm_transform_batch(f, method=None, *args, **kwargs):
    return sph_harm_transform_batch_naive(f, *args, **kwargs)

def sph_harm_inverse_batch(c, method=None, *args, **kwargs):
    return sph_harm_inverse_batch_naive(c, *args, **kwargs)




def sph_harm_transform_batch_naive(f, harmonics=None, m0_only=False):
    """ Spherical harmonics batch-transform.

    Args:
        f (n, l, l, c)-array : functions are on l x l grid
        harmonics (2, l/2, l/2, l, l)-array:
        m0_only (bool): return only coefficients with order 0;
                        only them are needed when computing convolutions

    Returns:
        coeffs ((n, 2, l/2, l/2, c)-array):
    """
    shapef = f.shape
    n, l = shapef[:2]
    assert shapef[2] == l
    if harmonics is None:
        harmonics = sph_harm_to_shtools(sph_harm_all(l))
    shapeh = harmonics.shape
    assert shapeh[1:] == (l // 2, l // 2, l, l)
    assert shapeh[0] in [1, 2]

    aj = np.array(DHaj(l))

    if m0_only:
        harmonics = harmonics[slice(0, 1), :, slice(0, 1), ...]

    na = np.newaxis
    coeffs = tnp.transpose(2 * np.sqrt(2) * np.pi / l *
                             tnp.dot(f * aj[na, na, :, na],
                                             tnp.conj(harmonics),
                                             dims=([1, 2], [3, 4])),
                             (0, 2, 3, 4, 1))
    return coeffs

def sph_harm_inverse_batch_naive(c, harmonics=None):
    """ Spherical harmonics batch inverse transform.

    Args:
        c ((n, 2, l/2, l/2, c)-array): sph harm coefficients; max degree is l/2
        harmonics (2, l/2, l/2, l, l)-array:

    Returns:
        recons ((n, l, l, c)-array):
    """
    shapec = c.shape
    l = 2 * shapec[2]
    assert shapec[3] == l // 2
    if harmonics is None:
        harmonics = sph_harm_to_shtools(sph_harm_all(l))
    shapeh = harmonics.shape
    assert shapeh[1:] == (l // 2, l // 2, l, l)
    assert shapeh[0] in [1, 2]

    real = True if shapeh[0] == 1 else False

    na = np.newaxis

    if real:
        factor = np.ones(c.shape[1:])[np.newaxis, ...]
        factor[..., 0, :] = factor[..., 0, :] / 2
        c = c * factor
        recons = tnp.transpose(2 * (tnp.dot(tnp.real(c), tnp.real(harmonics),
                                                      dims=([1, 2, 3], [0, 1, 2])) -
                                      tnp.dot(tnp.imag(c), tnp.imag(harmonics),
                                                      dims=([1, 2, 3], [0, 1, 2]))),
                                (0, 2, 3, 1))
    else:
        recons = tnp.transpose(tnp.dot(c, harmonics,
                                                 dims=([1, 2, 3], [0, 1, 2])),
                                 (0, 2, 3, 1))
    return recons

def sph_conv(f, g, harmonics=None):
    """ Spherical convolution f * g. """
    stackfun = torch.stack if isinstance(f, torch.Tensor) else np.array
    cf, cg = [sph_harm_transform(x, harmonics=harmonics) for x in [f, g]]
    cfg = [2 * np.pi * np.sqrt(4 * np.pi / (2 * l + 1)) * stackfun(c1) * c2[l]
           for l, (c1, c2) in enumerate(zip(cf, cg))]

    return sph_harm_inverse(cfg)

def sph_conv_batch(f, g,
                   harmonics_or_legendre=None,
                   method=None,
                   spectral_pool=0,
                   harmonics_or_legendre_low=None):
    """ CNN-like batch spherical convolution.

    Args:
        f (n, l, l, c)-array: input feature map. n entries, c channels
        g (c, l, l, d)-array: convolution kernels
        harmonics_or_legendre (): spherical harmonics or legendre polynomials to expand f and g
        method (str): see sph_harm_transform_batch
        spectral_pool (int): if > 0 run spectral pooling before ISHT
    (bandwidth is reduced by a factor of 2**spectral_pool)
        harmonics_or_legendre_low (): low frequency harmonics of legendre to be used when spectral_pool==True

    Returns:
        fg (n, l, l, d)-array
    """
    shapef, shapeg = [x.shape for x in [f, g]]
    spectral_filter = True if len(shapeg) == 5 else False
    spectral_input = True if len(shapef) == 5 else False
    n = shapef[2]
    if spectral_input:
        n *= 2

    if not spectral_input:
        cf = sph_harm_transform_batch(f, method, harmonics_or_legendre, m0_only=False)
    else:
        cf = f
    if not spectral_filter:
        cg = sph_harm_transform_batch(g, method, harmonics_or_legendre, m0_only=True)
    else:
        cg = g

    shapecf, shapecg = [x.shape for x in [cf, cg]]
    assert shapecf[4] == shapecg[0]
    assert shapecf[2] == shapecg[2]

    na = np.newaxis
    factor = (2 * np.pi * np.sqrt(4 * np.pi / (2 * np.arange(n / 2) + 1)))[na, na, :, na, na, na]
    cg = tnp.transpose(cg, (1, 2, 3, 0, 4))[na, ...]
    cf = cf[..., na]
    if isinstance(cg, torch.Tensor) and isinstance(cf, torch.Tensor):
        cg, cf = safe_cast(cg, cf)
        factor = torch.tensor(factor, dtype=torch.float32)
        cfg = factor * cf * cg
    else:
        cfg = factor * cf * cg

    if spectral_pool > 0:
        cfg = cfg[:, :, :n // (2 ** (spectral_pool + 1)), :n // (2 ** (spectral_pool + 1)), ...]
        hol = harmonics_or_legendre_low
    else:
        hol = harmonics_or_legendre

    cfg = tnp.sum(cfg, axis=-2)

    return sph_harm_inverse_batch(cfg, method, hol)

def is_real_sft(h_or_c):
    """ Detect if list of lists of harmonics or coefficients assumes real inputs (m>0) """
    if isinstance(h_or_c[1], torch.Tensor):
        d = h_or_c[1].shape[0]
    else:
        d = len(h_or_c[1])

    isreal = True if d == 2 else False

    return isreal

def sph_harm_to_shtools(c):
    """ Convert our list format for the sph harm coefficients/harmonics to pyshtools (2, n, n) format. """
    n = len(c)
    real = is_real_sft(c)
    dim1 = 1 if real else 2
    out = np.zeros((dim1, n, n, *c[0][0].shape)) + 0j
    for l, cc in enumerate(c):
        cc = np.array(cc)
        if not real:
            m_minus = cc[:l][::-1]
            m_plus = cc[l:]
        else:
            m_minus = np.array([])
            m_plus = cc

        if m_minus.size > 0:
            out[1, l, 1:l + 1, ...] = m_minus
        out[0, l, :l + 1, ...] = m_plus

    return out