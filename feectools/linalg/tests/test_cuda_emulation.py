"""CPU emulation of every CUDA kernel, compared with its pyccel kernel (runs without a GPU).

Runs the cases of ``cuda_parity_cases.PARITY_CASES``, like the GPU parity test. Each CUDA kernel is compiled as C++ by
``cunumpy.kernel_testing.emulate_cuda_kernel`` and run thread by thread (in blocks, with shared memory and
barriers), with the launch size of the declared kernel.
"""
import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler

from feectools.linalg.tests.cuda_parity_cases import PARITY_CASES
from feectools.linalg.tests.test_cuda_parity import CUDA_KERNELS

requires_compiler = pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")


def arrays(args):
    """Host copies of the arrays among the arguments."""
    return [a.copy() for a in args if isinstance(a, np.ndarray)]


# kernels that serial emulation cannot run (warp shuffles); the GPU parity test (test_cuda_parity) covers them
GPU_ONLY = {"stencil_dot_3d": "warp shuffles"}


@requires_compiler
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=pytest.mark.skip(reason=f"uses {GPU_ONLY[name]}, which emulation cannot run"))
        if name in GPU_ONLY
        else name
        for name in PARITY_CASES
    ],
)
def test_emulated_parity(name):
    kernel, spec = CUDA_KERNELS[name], PARITY_CASES[name]
    with xp.use_backend("numpy"):
        for case in spec.cases:
            host_args = spec.build(case)
            emulated_args = spec.build(case)
            before = arrays(host_args)
            kernel(*host_args)
            n_threads = spec.n_threads(emulated_args) if spec.n_threads else None
            emulate_cuda_kernel(kernel.cuda_kernel, *emulated_args, n_threads=n_threads)
            host, emulated = arrays(host_args), arrays(emulated_args)
            assert any(not np.array_equal(a, b) for a, b in zip(before, host)), f"{name}{case}: the kernel changed nothing"
            for h, e in zip(host, emulated):
                np.testing.assert_allclose(e, h, rtol=spec.rtol, atol=spec.atol, err_msg=f"{name}{case}")


@requires_compiler
def test_emulated_stencil_dot_3d_rows():
    """The one-thread-per-entry stencil_dot_3d (used on the fake CuPy instead of the warp-shuffle kernel) against
    the pyccel kernel, on the parity cases of stencil_dot_3d."""
    from feectools.linalg.kernels.stencil_dot_3d import stencil_dot_3d_rows

    kernel, spec = CUDA_KERNELS["stencil_dot_3d"], PARITY_CASES["stencil_dot_3d"]
    rows = stencil_dot_3d_rows()
    with xp.use_backend("numpy"):
        for case in spec.cases:
            host_args = spec.build(case)
            emulated_args = spec.build(case)
            kernel(*host_args)
            emulate_cuda_kernel(rows, *emulated_args, n_threads=rows.n_threads_from(emulated_args))
            for h, e in zip(arrays(host_args), arrays(emulated_args)):
                np.testing.assert_allclose(e, h, rtol=spec.rtol, atol=spec.atol, err_msg=f"stencil_dot_3d_rows{case}")
