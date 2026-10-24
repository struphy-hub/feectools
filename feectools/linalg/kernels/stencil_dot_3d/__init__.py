"""The kernel of this folder: pyccel in ``stencil_dot_3d_kernels.py``, CUDA in ``stencil_dot_3d_cuda.cu``."""

from functools import cache
from pathlib import Path

from cunumpy.kernels import Kernel

# STENCIL_DOT_LANES = 8 threads per entry of `out` (the third argument), see stencil_dot_3d_cuda.cu
stencil_dot_3d = Kernel.from_folder(__name__, n_threads_from=lambda args: 8 * args[2].size)


@cache
def stencil_dot_3d_rows():
    """The CUDA kernel with one thread per entry of `out` and no warp shuffles (``stencil_dot_3d_rows.cu``).

    For launches on cunumpy's fake CuPy (tests without a GPU), whose CPU emulation runs one thread at a time and
    rejects warp shuffles; on a GPU :data:`stencil_dot_3d` is used.
    """
    from cunumpy.kernels import CudaKernel

    kernel = CudaKernel.from_file(Path(__file__).with_name("stencil_dot_3d_rows.cu"), "stencil_dot_3d_rows")
    kernel.n_threads_from = lambda args: args[2].size
    return kernel


def stencil_dot_3d_fake_cupy(*args):
    """:data:`stencil_dot_3d` on cunumpy's fake CuPy: :func:`stencil_dot_3d_rows` for device arguments (emulated
    launches), the usual dispatch (the pyccel kernel) for host arguments."""
    import cunumpy as xp

    if any(xp.is_gpu(a) for a in args[:3]):
        return stencil_dot_3d_rows()(*args)
    return stencil_dot_3d(*args)
