"""Kernel launches that reuse their preparation when a kernel is launched with the same buffers again.

A launch of a :class:`cunumpy.kernels.CudaKernel` checks and converts every argument (dtype, contiguity, device,
array views packed into pointer, shape and strides), computes the launch shape and validates it against the device
limits. In a Krylov solver the same few kernels (axpy, inner products, banded passes) are launched with the same
buffers in every iteration, so this preparation is repeated thousands of times for the same result.
:class:`CachedLaunch` prepares a launch once per set of buffers and launches the compiled kernel directly afterwards.
"""

from collections import OrderedDict

import numpy as np

__all__ = ("CachedLaunch",)


def _real_device_array(value) -> bool:
    """Whether ``value`` is a CuPy array on a GPU (not cunumpy's fake CuPy)."""
    module = type(value).__module__
    if not module.startswith("cupy"):
        return False
    from cunumpy.kernel_testing import fake_cupy_active

    return not fake_cupy_active()


def _current_device() -> int:
    """The current CUDA device (0 without CuPy)."""
    try:
        import cupy
    except ImportError:
        return 0
    return int(cupy.cuda.runtime.getDevice())


class CachedLaunch:
    """A kernel whose CUDA launches with the same buffers reuse their prepared arguments and launch shape.

    Wraps a :class:`cunumpy.kernels.Kernel` (host or CUDA kernel by its dispatch) or a
    :class:`cunumpy.kernels.CudaKernel` and is called like it. A CUDA launch is keyed by the buffers (pointer, shape,
    strides, dtype) and the types of the scalar arguments, and by the launch arguments: the first launch with a key
    runs the kernel's own checks and conversions and stores them, later ones only convert the scalars (e.g. the
    coefficient of an axpy, which changes every iteration) with the kernel's checkers and launch the compiled kernel.

    Everything else goes to the wrapped kernel unchanged: host calls, cunumpy's fake CuPy, struct or other argument
    objects, an explicit stream, and kernels in debug mode or with ``check_finite``.

    Parameters
    ----------
    kernel : cunumpy.kernels.Kernel | cunumpy.kernels.CudaKernel
        The kernel to launch.
    max_entries : int
        Number of prepared launches kept (the oldest is dropped first).
    """

    def __init__(self, kernel, max_entries: int = 64):
        self._kernel = kernel
        self._max_entries = int(max_entries)
        self._launches = OrderedDict()
        self.n_prepared = 0
        """Number of launches prepared (cache misses on the GPU)."""
        self.n_reused = 0
        """Number of launches that reused a prepared one."""

    @property
    def kernel(self):
        """The wrapped kernel."""
        return self._kernel

    def __getattr__(self, name):
        return getattr(self._kernel, name)

    def _cuda_kernel(self, args):
        """The CUDA kernel this call launches on a GPU, or None (the call goes to the wrapped kernel)."""
        if not any(_real_device_array(a) for a in args):
            return None
        from cunumpy.kernels import CudaKernel

        if isinstance(self._kernel, CudaKernel):
            return self._kernel
        return getattr(self._kernel, "cuda_kernel", None)

    @staticmethod
    def _key(args, launch):
        """The key of a launch, and the positions of its scalar arguments; None if it is not cached."""
        key, scalars = [], []
        for i, a in enumerate(args):
            if _real_device_array(a):
                key.append(("a", int(a.data.ptr), a.shape, a.strides, a.dtype.str))
            elif isinstance(a, (bool, int, float, np.generic)) and not isinstance(a, np.void):
                key.append(("s", type(a)))
                scalars.append(i)
            else:
                return None, None
        for name in ("n_threads", "grid", "block"):
            value = launch.get(name)
            key.append(tuple(value) if isinstance(value, (list, tuple)) else value)
        key.append(int(launch.get("shared_mem", 0)))
        key.append(_current_device())
        return tuple(key), scalars

    def __call__(self, *args, **launch):
        cuda = self._cuda_kernel(args)
        if cuda is None or launch.get("stream") is not None or cuda.debug_active() or cuda.check_finite:
            return self._kernel(*args, **launch)
        key, scalars = self._key(args, launch)
        if key is None:
            return self._kernel(*args, **launch)

        entry = self._launches.get(key)
        if entry is None:
            # the first launch with these buffers: the kernel's own checks, conversions and validation
            result = self._kernel(*args, **launch)
            grid, block = cuda.launch_shape(
                launch.get("n_threads"), grid=launch.get("grid"), block=launch.get("block"), args=args
            )
            values = [np.asarray(v) if isinstance(v, np.void) else v for v in cuda.prepare_args(*args)]
            checkers = getattr(cuda, "_checkers", None) if getattr(cuda, "_signature", None) is not None else None
            if checkers is not None:
                checkers = {i: checkers[i] for i in scalars}
            entry = (cuda.compile(), grid, block, values, checkers, int(launch.get("shared_mem", 0)))
            self._launches[key] = entry
            if len(self._launches) > self._max_entries:
                self._launches.popitem(last=False)
            self.n_prepared += 1
            return result

        raw, grid, block, values, checkers, shared_mem = entry
        self._launches.move_to_end(key)
        if scalars:
            values = list(values)
            for i in scalars:
                values[i] = checkers[i](args[i]) if checkers is not None else args[i]
        if 0 not in grid:
            raw(grid, block, tuple(values), shared_mem=shared_mem)
        self.n_reused += 1
        return None
