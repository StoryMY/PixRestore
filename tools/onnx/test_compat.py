"""Run from repository root: python -m unittest discover -s tools/onnx -p test_*.py"""
import unittest
import torch
from torch import nn
from pixrestore.onnx_compat import PatchConvMatMul


class PatchConvTests(unittest.TestCase):
    def test_equivalence_and_patch_order(self):
        torch.manual_seed(42)
        for kernel, shape, bias in [((8, 8), (1, 6, 32, 48), False),
                                    ((2, 3), (2, 3, 8, 15), True)]:
            with self.subTest(kernel=kernel, bias=bias):
                conv = nn.Conv2d(shape[1], 13, kernel, stride=kernel, bias=bias).double()
                x = torch.randn(shape, dtype=torch.float64)
                torch.testing.assert_close(PatchConvMatMul(conv)(x), conv(x), rtol=1e-12, atol=1e-12)

    def test_rejects_unsupported_convolution(self):
        with self.assertRaises(ValueError):
            PatchConvMatMul(nn.Conv2d(6, 8, 3, stride=1, padding=1))

    def test_rejects_partial_patches(self):
        layer = PatchConvMatMul(nn.Conv2d(6, 8, 8, stride=8))
        with self.assertRaises(ValueError):
            layer(torch.zeros(1, 6, 17, 16))
