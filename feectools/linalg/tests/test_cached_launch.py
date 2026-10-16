"""CachedLaunch (feectools.utilities.cached_launch): prepared launches are reused only for the same buffers.

Without a GPU the CUDA path is checked with a stub kernel and stub device arrays; host calls go to the wrapped kernel.
"""

import numpy as np
import pytest

import feectools.utilities.cached_launch as cached_launch
from feectools.utilities.cached_launch import CachedLaunch


class _Buffer:
    def __init__(self, ptr):
        self.ptr = ptr


class StubDeviceArray:
    """Stands in for a cupy.ndarray: pointer, shape, strides, dtype."""

    def __init__(self, ptr, shape=(4,)):
        self.data = _Buffer(ptr)
        self.shape = shape
        self.strides = (8,)
        self.dtype = np.dtype(float)


class StubCudaKernel:
    """Records the full calls and the raw launches; scalars are converted to float64 by the 'checkers'."""

    def __init__(self, n_args):
        self.calls = []
        self.launches = []
        self.debug = False
        self.check_finite = False
        self._signature = object()
        self._checkers = [np.float64] * n_args

    def debug_active(self):
        return self.debug

    def __call__(self, *args, **launch):
        self.calls.append((args, launch))

    def launch_shape(self, n_threads, *, grid=None, block=None, args=None):
        return (max(1, (n_threads or 1) // 128),), (128,)

    def prepare_args(self, *args):
        return tuple(a.data.ptr if isinstance(a, StubDeviceArray) else np.float64(a) for a in args)

    def compile(self):
        def raw(grid, block, values, shared_mem=0):
            self.launches.append((grid, block, values, shared_mem))

        return raw


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(cached_launch, "_real_device_array", lambda a: isinstance(a, StubDeviceArray))
    kernel = StubCudaKernel(3)
    monkeypatch.setattr(CachedLaunch, "_cuda_kernel", lambda self, args: kernel if any(
        isinstance(a, StubDeviceArray) for a in args) else None)
    return kernel


def test_reuses_prepared_launch(stub):
    launch = CachedLaunch(stub)
    x, y = StubDeviceArray(100), StubDeviceArray(200)

    launch(2.0, x, y, n_threads=256)
    assert len(stub.calls) == 1 and launch.n_prepared == 1 and not stub.launches

    # same buffers, another coefficient: no full call, the scalar converted and passed in its slot
    launch(3.5, x, y, n_threads=256)
    assert len(stub.calls) == 1 and launch.n_reused == 1
    grid, block, values, _ = stub.launches[-1]
    assert values == (np.float64(3.5), 100, 200)
    assert isinstance(values[0], np.float64)
    assert grid == (2,) and block == (128,)


def test_new_buffers_or_launch_shape_prepare_again(stub):
    launch = CachedLaunch(stub)
    x, y = StubDeviceArray(100), StubDeviceArray(200)
    launch(1.0, x, y, n_threads=256)
    launch(1.0, x, StubDeviceArray(300), n_threads=256)
    launch(1.0, x, StubDeviceArray(200, shape=(5,)), n_threads=256)
    launch(1.0, x, y, n_threads=512)
    assert launch.n_prepared == 4 and len(stub.calls) == 4 and not stub.launches


def test_pass_through(stub):
    launch = CachedLaunch(stub)
    x, y = StubDeviceArray(100), StubDeviceArray(200)
    # an argument object (e.g. a struct), an explicit stream, debug mode, check_finite
    launch(object(), x, y)
    launch(1.0, x, y, stream=object())
    stub.debug = True
    launch(1.0, x, y)
    launch(1.0, x, y)
    stub.debug, stub.check_finite = False, True
    launch(1.0, x, y)
    assert len(stub.calls) == 5 and launch.n_prepared == 0 and not stub.launches


def test_bounded(stub):
    launch = CachedLaunch(stub, max_entries=2)
    a, b, c = (StubDeviceArray(p) for p in (1, 2, 3))
    for buf in (a, b, c):
        launch(1.0, buf, buf)
    assert len(launch._launches) == 2
    launch(1.0, a, a)  # dropped first: prepared again
    assert launch.n_prepared == 4


def test_host_calls_go_to_the_kernel():
    """NumPy arguments: the wrapped kernel is called as it is (here the stencil axpy of feectools)."""
    from feectools.linalg.stencil import stencil_kernels

    axpy = stencil_kernels["axpy"][1]
    launch = CachedLaunch(axpy)
    x = np.arange(6.0)
    y = np.ones(6)
    expected = y + 2.0 * x
    launch(2.0, x, y)
    np.testing.assert_array_equal(y, expected)
    assert launch.n_prepared == 0 and launch.n_reused == 0
    assert launch.kernel is axpy
