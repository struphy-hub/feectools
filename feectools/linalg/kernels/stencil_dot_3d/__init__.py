"""The kernel of this folder: pyccel in ``stencil_dot_3d_kernels.py``, CUDA in ``stencil_dot_3d_cuda.cu``."""

from cunumpy.kernels import Kernel

# STENCIL_DOT_LANES = 8 threads per entry of `out` (the third argument), see stencil_dot_3d_cuda.cu
stencil_dot_3d = Kernel.from_folder(__name__, n_threads_from=lambda args: 8 * args[2].size)
