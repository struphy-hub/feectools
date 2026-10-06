"""Parity of every CUDA kernel with its pyccel kernel, from the ``<name>_test_args.py`` modules of the folders.

The kernels are the ones the folders declare (``<name> = Kernel.from_folder(...)`` in ``__init__.py``), with
their launch options, i.e. the objects the code calls.
"""
import importlib

import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import check_parity, parity_cases, requires_cupy
from cunumpy.kernels import Kernel, KernelCatalog

# all kernel packages with one folder per kernel, for tests that go through every kernel
PACKAGES = ("feectools.linalg.kernels",)
CATALOGS = {package: KernelCatalog.from_package(package) for package in PACKAGES}
DECLARED = KernelCatalog(
    {
        name: getattr(importlib.import_module(f"{package}.{name}"), name)
        for package, catalog in CATALOGS.items()
        for name in catalog
    }
)
CUDA_CASES = parity_cases(DECLARED)


def test_folders_declare_their_kernels():
    """Each kernel folder's __init__.py declares its kernel under the folder name (the import used in the code)."""
    assert len(DECLARED) > 0
    for catalog in CATALOGS.values():
        for name in catalog:
            kernel = DECLARED[name]
            assert isinstance(kernel, Kernel) and kernel.name == name
            assert kernel.has_cuda == catalog[name].has_cuda


def test_signatures():
    """The pyccel and CUDA versions of every kernel take the same arguments in the same order."""
    DECLARED.check_signatures()


def test_cuda_kernels_have_test_args():
    """Every kernel with a CUDA version has a <name>_test_args.py with make_args and CASES."""
    for name, kernel in DECLARED.parity_cases():
        assert kernel.test_args_module is not None, f"add {name}_test_args.py to the folder of {name}"
        assert callable(kernel.test_args.make_args) and len(kernel.test_args.CASES) > 0


def test_stencil_kernels_have_cuda():
    """The stencil operations of solver loops (dot, transpose, inner, axpy in 1-3D) all have CUDA versions."""
    for operation in ("dot", "transpose", "inner", "axpy"):
        for ndim in (1, 2, 3):
            assert DECLARED[f"stencil_{operation}_{ndim}d"].has_cuda


@requires_cupy
@pytest.mark.parametrize("kernel", CUDA_CASES)
def test_parity(kernel):
    for seed in range(len(kernel.test_args.CASES)):
        check_parity(kernel, seed=seed)


@requires_cupy
def test_solver_loop_has_no_host_transfers():
    """dot and axpy on the CuPy backend run on the device: no host copies once the kernels are compiled."""
    from cunumpy.profiling import assert_no_transfers

    from feectools.linalg.tests.kernel_test_args import stencil_matrix, stencil_vector

    with xp.use_backend("cupy"):
        rng = np.random.default_rng(0)
        A, V, W = stencil_matrix((12, 10, 8), (12, 10, 8), (2, 2, 3), rng)
        x = stencil_vector(V, rng)
        y = A.dot(x)  # warm up: compiles the CUDA kernels
        V.axpy(0.5, y, x)
        with assert_no_transfers():
            for _ in range(5):
                A.dot(x, out=y)
                V.axpy(0.5, y, x)
