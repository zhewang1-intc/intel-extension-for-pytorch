import torch
import unittest
from common_utils import TestCase


class TestJblas(TestCase):
    def test_jblas(self):
        torch.manual_seed(0)
        m = 256
        n = 1024
        k = 512
        activation = torch.rand(m, k, dtype=torch.float)
        raw_wei = torch.rand(k, n, dtype=torch.float)
        pack_wei = torch.ops.torch_ipex.jblas_prepack_perchannel_int4_weight(raw_wei)
        ref = torch.matmul(activation, raw_wei)
        tar = torch.zeros(m, n, dtype=torch.float)
        tar = torch.ops.jblas_woq_int4_perchannel_linear(activation, pack_wei, tar)
        torch.allclose(tar, ref, rtol=0.03)


if __name__ == "__main__":
    test = unittest.main()
