"""CPU emulation of every CUDA kernel, compared with its pyccel kernel (runs without a GPU).

Uses the same ``<name>_test_args.py`` modules as the GPU parity test. Each CUDA kernel is compiled as C++ by
``cunumpy.kernel_testing.emulate_cuda_kernel`` and run thread by thread (in blocks, with shared memory and
barriers), with the launch size of the declared kernel.
"""
import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler

from feectools.linalg.tests.test_cuda_parity import CUDA_CASES

requires_compiler = pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")


def arrays(args):
    """Host copies of the arrays among the arguments."""
    return [a.copy() for a in args if isinstance(a, np.ndarray)]


@requires_compiler
@pytest.mark.parametrize("kernel", CUDA_CASES)
def test_emulated_parity(kernel):
    module = kernel.test_args
    rtol, atol = getattr(module, "RTOL", 1e-12), getattr(module, "ATOL", 0.0)
    with xp.use_backend("numpy"):
        for seed in range(len(module.CASES)):
            host_args = module.make_args("numpy", seed)
            emulated_args = module.make_args("numpy", seed)
            before = arrays(host_args)
            kernel(*host_args)
            n_threads = getattr(module, "N_THREADS", None)  # as in cunumpy's check_parity
            if callable(n_threads):
                n_threads = n_threads(emulated_args)
            emulate_cuda_kernel(kernel.cuda_kernel, *emulated_args, n_threads=n_threads)
            host, emulated = arrays(host_args), arrays(emulated_args)
            assert any(not np.array_equal(a, b) for a, b in zip(before, host)), "the kernel changed nothing"
            for h, e in zip(host, emulated):
                np.testing.assert_allclose(e, h, rtol=rtol, atol=atol)
