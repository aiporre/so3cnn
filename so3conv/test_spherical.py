# test spherical harmonics
import warnings
from unittest import TestCase
import sys
import numpy as np
import torch

from so3conv.util import sph_sample, sphrot_shtools

from so3conv.spherical import sph_harm_lm, sph_harm_inverse, sph_harm_transform, sph_harm_to_shtools, sph_harm_all, \
    sph_conv, sph_conv_batch
from so3conv.spherical import sph_harm_transform_batch, sph_harm_inverse_batch

class TestSpherical(TestCase):
    def test_sh_harm_lm(self):
        n = 10
        phi, theta = np.meshgrid(*sph_sample(n))

        # eval the second harmonics
        l, m = 2, -2
        Y = sph_harm_lm(l, m, n)
        Y_true =  1/4*np.sqrt(15/2/np.pi)*(np.sin(phi))**2 * np.exp(-2j * theta)
        assert np.allclose(Y, Y_true)

        l, m = 2, 1
        Y = sph_harm_lm(l, m, n)
        Y_ref = -1 / 2 * np.sqrt(15 / 2 / np.pi) * np.sin(phi) * np.cos(phi) * np.exp(1j * theta)
        assert np.allclose(Y, Y_ref)

    def test_spherical_harmonics_inverse_fft(self):
        """ Tests the SFT by doing ISFT -> SFT -> ISFT. """
        for n in [8, 16]:
            # sft for real signals: c_{-m} = (-1)^m Re(c_m) + (-1)^{m+1} Im(c_m)
            coeffs = [np.zeros(2 * l + 1, dtype='complex') for l in range(n // 2)]

            coeffs[0][0], coeffs[1][1], coeffs[2][2] = [np.random.rand() for _ in range(3)]

            c = np.random.rand() + 1j * np.random.rand()
            coeffs[1][0] = c
            coeffs[1][2] = -np.real(c) + 1j * np.imag(c)

            c = np.random.rand() + 1j * np.random.rand()
            coeffs[2][0] = c
            coeffs[2][4] = np.real(c) - 1j * np.imag(c)

            c = np.random.rand() + 1j * np.random.rand()
            coeffs[2][1] = c
            coeffs[2][3] = -np.real(c) + 1j * np.imag(c)

            f = sph_harm_inverse(coeffs)

            coeffs_ = sph_harm_transform(f)
            f_ = sph_harm_inverse(coeffs_)

            for c1, c2 in zip(coeffs, coeffs_):
                assert np.allclose(c1, c2)
            assert np.allclose(f, f_)
    def test_spherical_harmonic_sft(self):
        """ Test the SFT by comparing to the FFT. """
        # Tests the SFT by doing SFT -> ISFT -> SFT -> ISFT
        f = np.random.rand(16, 16)
        coeffs = sph_harm_transform(f)
        # we are constraining the bandwidth to n/2 here, so f1 != f
        f1 = sph_harm_inverse(coeffs)
        coeffs1 = sph_harm_transform(f1)
        f2 = sph_harm_inverse(coeffs1)
        # now both f1 and f2 have constrained bandwidths, so they must be equal
        assert np.allclose(f1, f2)

        # coeffs = sph_harm_transform(torch.tensor(f, dtype=torch.float32))
        # # we are constraining the bandwidth to n/2 here, so f1 != f
        # f1 = sph_harm_inverse(coeffs)
        # coeffs1 = sph_harm_transform(f1)
        # f2 = sph_harm_inverse(coeffs1)
        # # now both f1 and f2 have constrained bandwidths, so they must be equal
        # assert np.allclose(f1.numpy(), f2.numpy())

    def test_spherical_harmonics_batch_inverse(self):
        # validate the batch inverse
        f = np.random.rand(10, 32, 32, 3) # 10 samples of 32x32x3
        c1 = sph_harm_transform_batch(f)
        f1 = sph_harm_inverse_batch(c1)
        c2 = sph_harm_transform_batch(f1)
        f2 = sph_harm_inverse_batch(c2)
        assert np.allclose(f1, f2)

    def test_sph_harm_batch_real_complex(self):
        """ Test batch form of spherical harmonics transform and inverse """
        f = np.random.rand(10, 32, 32, 3)
        h_complex = sph_harm_to_shtools(sph_harm_all(32))
        h_real = sph_harm_to_shtools(sph_harm_all(32, real=True))

        c1 = sph_harm_transform_batch(f, harmonics=h_complex)
        c2 = sph_harm_transform_batch(f.astype('complex'), harmonics=h_complex)
        c3 = sph_harm_transform_batch(f, harmonics=h_real)
        c4 = sph_harm_transform_batch(f.astype('complex'), harmonics=h_real)

        assert np.allclose(c1, c2)
        assert np.allclose(c2[:, [0]], c3)
        assert np.allclose(c3, c4)

    def test_sph_harm_batch(self):
        """ Test batch form of spherical harmonics transform and inverse """
        f = np.random.rand(1, 32, 32, 1)
        # computing harmonics on the fly
        c1 = sph_harm_transform_batch(f)[0, ..., 0]
        c2 = sph_harm_to_shtools(sph_harm_transform(f[0, ..., 0]))
        assert np.allclose(c1, c2)

        # caching harmonics
        for real in [False, True]:
            h = sph_harm_all(32, real=real)
            c1 = sph_harm_transform_batch(f, harmonics=sph_harm_to_shtools(h))[0, ..., 0]
            c2 = sph_harm_to_shtools(sph_harm_transform(f[0, ..., 0], harmonics=h))

            assert np.allclose(c1, c2)

    def test_sph_harm_batch_harmonics_input(self):
        """ Test batch form of spherical harmonics transform and inverse """
        for real in [False, True]:
            harmonics = sph_harm_to_shtools(sph_harm_all(32, real=real))

            f = np.random.rand(10, 32, 32, 3)
            c1 = sph_harm_transform_batch(f, harmonics=harmonics)
            r1 = sph_harm_inverse_batch(c1, harmonics=harmonics)
            c2 = sph_harm_transform_batch(r1, harmonics=harmonics)
            r2 = sph_harm_inverse_batch(c2, harmonics=harmonics)

            assert np.allclose(r1, r2)

    def test_sph_harm_tf(self):
        """ Test spherical harmonics expansion/inversion with PyTorch Tensors. """
        n = 32
        f = np.random.rand(n, n)
        ref = sph_harm_inverse(sph_harm_transform(f))

        inp = torch.tensor(f, dtype=torch.complex128)
        coeffs = sph_harm_transform(inp)
        recons = sph_harm_inverse(coeffs)

        c, r = coeffs, recons

        assert np.allclose(r, ref)
        for x1, x2 in zip(c, sph_harm_transform(f)):
            assert np.allclose(x1, x2)

    def test_sph_harm_tf_harmonics_input(self):
        """ Test spherical harmonics inputs as PyTorch Variables. """
        n = 32
        f = np.random.rand(n, n)

        for real in [False, True]:
            dtype = torch.complex64
            harmonics = [[torch.tensor(hh.astype('complex64')) for hh in h]
                         for h in sph_harm_all(n, real=real)]

            inp = torch.tensor(f, dtype=dtype)
            c1 = sph_harm_transform(inp, harmonics=harmonics)
            r1 = sph_harm_inverse(c1, harmonics=harmonics)
            c2 = sph_harm_transform(inp, harmonics=harmonics)
            r2 = sph_harm_inverse(c2, harmonics=harmonics)

            c1v, c2v, r1v, r2v = c1, c2, r1, r2

            for x1, x2 in zip(c1v, c2v):
                assert np.allclose(x1, x2), "x {} != {}".format(x1, x2)
            for x1, x2 in zip(r1v, r2v):
                assert np.allclose(x1, x2), "x {} != {}".format(x1, x2)

    def test_sph_harm_shtools(self):
        """ Compare our sph harmonics expansion with pyshtools. """
        if 'pyshtools' not in sys.modules:
            warnings.warn('pyshtools not available; skipping test_sph_harm_shtools')
            return
        import pyshtools

        n = 32
        f = np.random.rand(n, n)
        # lowpass
        f = sph_harm_inverse(sph_harm_transform(f))

        c_mine = sph_harm_transform(f)
        c_pysh = pyshtools.SHGrid.from_array(f.T).expand(csphase=-1, normalization='ortho')

        c1 = sph_harm_to_shtools(c_mine)
        c2 = c_pysh.coeffs

        # there seems to be a bug on the coefficient of highest degree l, order -l in pyshtools
        # we don't test that value
        c1[1][(n // 2) - 1][-1] = c2[1][(n // 2) -1][-1] = 0
        assert np.allclose(c1, c2)

    def test_sph_conv(self):
        """ Test spherical convolution and rotation commutativity.

        sph_conv and sphrot_shtools are exercised here.
        """
        if 'pyshtools' not in sys.modules:
            warnings.warn('pyshtools not available; skipping test_sph_conv')
            return

        n = 32
        f = np.random.rand(n, n)
        # lowpass
        f = sph_harm_inverse(sph_harm_transform(f))

        g = np.zeros_like(f)
        g[:, :5] = np.random.rand(5)
        g /= g.sum()

        ang = np.random.rand(3)*2*np.pi

        # check if (pi f * g) == pi(f * g)
        rot_conv = sphrot_shtools(sph_conv(f, g), ang)
        conv_rot = sph_conv(sphrot_shtools(f, ang), g)

        assert not np.allclose(rot_conv, f)
        assert not np.allclose(rot_conv, g)
        assert np.allclose(rot_conv, conv_rot)

    def test_sph_conv_tf(self):
        """ Test spherical convolution with PyTorch Tensors. """
        n = 32
        kernel_size = 5
        f = np.random.rand(32, 32)

        inp = torch.tensor(f, dtype=torch.complex128)
        weights = torch.nn.Parameter(torch.randn(kernel_size, dtype=torch.complex128))
        ker = torch.cat([weights.unsqueeze(0).repeat(32, 1),
                         torch.zeros((n, 27), dtype=torch.complex128)], dim=1)
        conv = sph_conv(inp, ker)

        conv_v, ker_v = conv.detach().numpy(), ker.detach().numpy()

        assert np.allclose(conv_v, sph_conv(f, ker_v))

    def test_sph_conv_batch(self):
        """ Test batch spherical convolution """
        f = np.random.rand(2, 32, 32, 3).astype('float32') + 0j
        g = np.random.rand(3, 32, 32, 6).astype('float32') + 0j
        res = sph_conv_batch(f, g)

        # compare with regular convolution
        for i in range(f.shape[0]):
            for j in range(g.shape[-1]):
                conv = np.zeros_like(f[0, ..., 0]).astype('complex')
                for k in range(f.shape[-1]):
                    conv += sph_conv(f[i, ..., k], g[k, ..., j])
                assert np.allclose(conv, res[i, ..., j])

        # test using PyTorch Tensors
        tfconv = sph_conv_batch(torch.tensor(f), torch.tensor(g))
        restf = tfconv.detach().numpy()

        assert np.allclose(res, restf)

    def test_sph_conv_real_complex(self):
        """ Compare spherical convolution assuming real inputs and not. """
        f = np.random.rand(2, 32, 32, 3).astype('float32')
        g = np.random.rand(3, 32, 32, 6).astype('float32')

        h_complex = sph_harm_to_shtools(sph_harm_all(32))
        h_real = sph_harm_to_shtools(sph_harm_all(32, real=True))

        res_complex = sph_conv_batch(f, g, harmonics_or_legendre=h_complex).real
        res_real = sph_conv_batch(f, g, harmonics_or_legendre=h_real)
        assert np.allclose(res_complex, res_real)

        # test using PyTorch Tensors
        tfconv = sph_conv_batch(torch.tensor(f), torch.tensor(g),
                                harmonics_or_legendre=torch.tensor(h_real))
        restf = tfconv.detach().numpy()

        assert np.allclose(res_real, restf)

    def test_sph_conv_batch_spectral_filters(self):
        f = np.random.rand(2, 32, 32, 3).astype('float32') + 0j
        g = np.random.rand(3, 32, 32, 6).astype('float32') + 0j
        ref = sph_conv_batch(f, g)

        cg = sph_harm_transform_batch(g, m0_only=True)
        test = sph_conv_batch(f, cg)

        assert np.allclose(ref, test)

    def test_sph_conv_batch_spectral_input(self):
        f = np.random.rand(2, 32, 32, 3).astype('float32') + 0j
        g = np.random.rand(3, 32, 32, 6).astype('float32') + 0j
        ref = sph_conv_batch(f, g)

        cf = sph_harm_transform_batch(f, m0_only=False)
        test = sph_conv_batch(cf, g)
        assert np.allclose(ref, test)

        cg = sph_harm_transform_batch(g, m0_only=True)
        test = sph_conv_batch(cf, cg)
        assert np.allclose(ref, test)