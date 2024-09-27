import numpy as np
import torch
from folding3d import window_3d, ovadd_3d, divisor_3d


class TestFolding3D:

    a = torch.randn(2, 4, 96, 96, 96) + 200
    
    def test_divisor(self):
        ones = divisor_3d(96, 32, 8)
        print(ones.detach().numpy())
        assert ones.shape[-1] == 96
        assert ones.shape[-2] == 96
        assert ones.shape[-3] == 96

    def test_windowing(self):
        a = self.a
        b = window_3d(a, 32, 16)
        for j in range(1, 4):
            assert b.shape[-j] == 32
        assert b.shape[0] == a.shape[0]
        assert b.shape[1] == a.shape[1]

    def test_ovadd(self):
        a = self.a
        print(a.shape)
        b = window_3d(a, 16, 16)
        print(b.shape)
        b = ovadd_3d(b, 96, 16, 16, True).detach().numpy()
        print(b.shape)
        a = a.detach().numpy()
        assert sum(np.array(a.shape) != np.array(b.shape)) == 0
        assert np.allclose(a, b)

    def test_divisor_2(self):
        dim = 96
        kernel_sizes = np.arange(16, 32, 8)
        strafes = (np.ones(len(kernel_sizes)) * 8).astype(int)
        for j, (kz, sf) in enumerate(zip(kernel_sizes, strafes)):
            print(kz, sf)
            ova_ones = divisor_3d(dim, kz, sf).detach().numpy().squeeze()
            mir_ones = np.flipud(ova_ones).astype(int)
            assert np.allclose(
                ova_ones,
                mir_ones
            )
