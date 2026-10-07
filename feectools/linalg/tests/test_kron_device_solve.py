"""Kronecker solves of device data stay on the device.

`KroneckerLinearSolver` (and `BandedSolver`/`SparseSolver` called with device arrays) solve
device data with the dense inverses of the 1D matrices (`DenseInverse`), one matrix product on
the device, instead of copying the data to the host for LAPACK/SuperLU. These tests check

* on the CPU: `DenseInverse` gives the host solvers' results (clamped and periodic matrices,
  transposed, real and complex), and the Kronecker solve matches a global reference, serial and
  distributed (``mpirun -n 2 python -m pytest -m mpi --with-mpi``);
* with cunumpy's fake CuPy (host memory, CuPy semantics, in a subprocess): device solves give
  the host results and make no host/device transfers;
* on a GPU (skipped otherwise): the same, serial and with MPI.
"""
import importlib.util
import os
import subprocess
import sys
from functools import reduce

import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import requires_cupy
from cunumpy.profiling import assert_no_transfers
from maybempi import MPI
from scipy.sparse import csr_matrix

from feectools.ddm.cart import CartDecomposition, DomainDecomposition
from feectools.linalg.direct_solvers import BandedSolver, DenseInverse, SparseSolver
from feectools.linalg.kron import KroneckerLinearSolver
from feectools.linalg.stencil import StencilVector, StencilVectorSpace

# Dense inverse times rhs vs. LU solve: both are backward stable, the matrices are well conditioned.
RTOL = 1e-12

# (npts, pads, periods): clamped, periodic and mixed directions, 1D to 3D
CASES = [
    ((12,), (2,), (False,)),
    ((10,), (3,), (True,)),
    ((12, 9), (2, 1), (False, True)),
    ((8, 10, 7), (3, 2, 1), (True, False, False)),
    ((8, 6, 9), (2, 3, 2), (True, True, True)),
]
SOLVER_KINDS = ("banded", "sparse")


# ===============================================================================
def matrix_1d(n, p, periodic, dtype=float):
    """A non-symmetric, diagonally dominant matrix with the band of a degree-p mass matrix.

    Periodic matrices have corners (cyclic band), as the 1D mass matrices of periodic directions.
    """
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        for d in range(-p, p + 1):
            j = i + d
            if periodic:
                j %= n
            elif not 0 <= j < n:
                continue
            A[i, j] += 1.0 / (1 + abs(d)) * (1 + 0.1 * np.sin(i + 2 * j))
            if np.dtype(dtype).kind == "c":
                A[i, j] += 0.05j * np.cos(3 * i - j)
        A[i, i] += 2 * p + 2
    return A


def banded_solver(A):
    """`BandedSolver` of a dense matrix (band storage as in `to_bnd`)."""
    n = A.shape[0]
    rows, cols = np.nonzero(A)
    la = int(max(0, (rows - cols).max()))
    ua = int(max(0, (cols - rows).max()))
    ab = np.zeros((1 + ua + 2 * la, n), dtype=A.dtype)
    for i, j in zip(rows, cols):
        ab[la + ua + i - j, j] = A[i, j]
    return BandedSolver(ua, la, ab)


def solver_1d(A, kind):
    return banded_solver(A) if kind == "banded" else SparseSolver(csr_matrix(A))


def make_space(npts, pads, periods, dtype=float, comm=None):
    ndim = len(npts)
    D = DomainDecomposition(list(npts), periods=list(periods), comm=comm)
    gs, ge = [], []
    for axis in range(ndim):
        ee = np.array(D.global_element_ends[axis]).copy()
        ee[-1] = npts[axis] - 1
        ge.append(ee)
        gs.append(np.array([0] + (ee[:-1] + 1).tolist()))
    C = CartDecomposition(D, list(npts), gs, ge, pads=list(pads), shifts=[1] * ndim)
    return StencilVectorSpace(C, dtype=dtype)


def owned_slices(V):
    return tuple(slice(int(s), int(e) + 1) for s, e in zip(V.starts, V.ends))


def scatter(V, glob):
    """This rank's part of a global array, in a new vector on the active backend."""
    v = StencilVector(V)
    v[owned_slices(V)] = xp.asarray(glob[owned_slices(V)])
    v.update_ghost_regions()
    return v


def global_rhs(npts, dtype=float):
    """A deterministic global right-hand side, independent of the decomposition."""
    idx = np.indices(npts).reshape(len(npts), -1)
    b = np.cos(1.0 + idx.T @ np.arange(1, len(npts) + 1)).reshape(npts)
    if np.dtype(dtype).kind == "c":
        b = b + 1j * np.sin(idx.T @ np.arange(2, len(npts) + 2)).reshape(npts)
    return b.astype(dtype)


def reference_solution(mats, b, transposed=False):
    K = reduce(np.kron, [A.T if transposed else A for A in mats])
    return np.linalg.solve(K, b.ravel()).reshape(b.shape)


def kron_solve(npts, pads, periods, kind, *, transposed=False, dtype=float, comm=None, check_transfers=False):
    """Solves the case on the active backend; returns this rank's part (host), and the global rhs and matrices."""
    mats = [matrix_1d(n, p, per, dtype) for n, p, per in zip(npts, pads, periods)]
    V = make_space(npts, pads, periods, dtype, comm)
    S = KroneckerLinearSolver(V, V, [solver_1d(A, kind) for A in mats])
    if transposed:
        S = S.transpose()
    b = global_rhs(npts, dtype)
    rhs = scatter(V, b)
    x = StencilVector(V)
    if check_transfers:
        with assert_no_transfers():
            S.solve(rhs, out=x)
    else:
        S.solve(rhs, out=x)
    return xp.to_numpy(x[owned_slices(V)]), b, mats, owned_slices(V)


def assert_close(actual, desired, what=""):
    scale = np.abs(desired).max()
    np.testing.assert_allclose(actual, desired, rtol=0, atol=RTOL * scale, err_msg=what)


def compare_device_with_host(npts, pads, periods, kind, transposed=False, dtype=float, comm=None):
    """Device solve (CuPy backend, no transfers) vs host solve (NumPy backend) vs global reference."""
    with xp.use_backend("numpy"):
        host, b, mats, owned = kron_solve(npts, pads, periods, kind, transposed=transposed, dtype=dtype, comm=comm)
    with xp.use_backend("cupy"):
        device, _, _, _ = kron_solve(npts, pads, periods, kind, transposed=transposed, dtype=dtype, comm=comm,
                                     check_transfers=True)
    what = f"{npts=} {periods=} {kind=} {transposed=} {dtype=}"
    assert_close(device, host, what)
    assert_close(device, reference_solution(mats, b, transposed)[owned], what)


def compare_1d_device_with_host(kind, periodic, transposed=False, dtype=float):
    """`BandedSolver`/`SparseSolver` on device arrays: host results, no transfers after the first solve."""
    A = matrix_1d(11, 2, periodic, dtype)
    solver = solver_1d(A, kind)
    if transposed:
        solver = solver.transpose()
    B = np.stack([global_rhs((11,), dtype) * (k + 1) for k in range(4)])
    host = solver.solve(B)
    with xp.use_backend("cupy"):
        B_dev = xp.to_cupy(B)
        solver.solve(B_dev)  # first device solve copies the inverse to the device
        with assert_no_transfers():
            out_dev = solver.solve(B_dev)
            inplace_dev = B_dev.copy()
            solver.solve(inplace_dev, out=inplace_dev)
        for result in (out_dev, inplace_dev):
            assert xp.is_gpu(result)
            assert_close(xp.to_numpy(result), host, f"{kind=} {periodic=} {transposed=} {dtype=}")


def run_device_checks():
    """All device checks; run in a process whose CuPy backend is the real or fake CuPy."""
    for npts, pads, periods in CASES:
        for kind in SOLVER_KINDS:
            for transposed in (False, True):
                compare_device_with_host(npts, pads, periods, kind, transposed)
    compare_device_with_host((9, 8), (2, 2), (True, False), "banded", dtype=complex)
    for kind in SOLVER_KINDS:
        for periodic in (False, True):
            for transposed in (False, True):
                compare_1d_device_with_host(kind, periodic, transposed)
    compare_1d_device_with_host("sparse", True, dtype=complex)
    print("device checks passed")


# ===============================================================================
# CPU tests
# ===============================================================================
@pytest.mark.parametrize("kind", SOLVER_KINDS)
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [float, complex])
def test_dense_inverse_matches_host_solver(kind, periodic, transposed, dtype):
    A = matrix_1d(13, 3, periodic, dtype)
    solver = solver_1d(A, kind)
    if transposed:
        solver = solver.transpose()
    op = A.T if transposed else A
    B = np.stack([global_rhs((13,), dtype) * (k + 1) + k for k in range(5)])

    inverse = DenseInverse(solver, 13, dtype)
    assert inverse.shape == (13, 13)
    host = solver.solve(B.copy())
    assert_close(host, np.linalg.solve(op, B.T).T)
    assert_close(inverse.solve(B), host)
    assert_close(inverse.solve(B[2]), host[2])  # a single right-hand side
    inplace = B.copy()
    assert inverse.solve(inplace, out=inplace) is inplace
    assert_close(inplace, host)


@pytest.mark.parametrize("npts, pads, periods", CASES)
@pytest.mark.parametrize("kind", SOLVER_KINDS)
@pytest.mark.parametrize("transposed", [False, True])
def test_kronecker_solve_matches_reference(npts, pads, periods, kind, transposed):
    with xp.use_backend("numpy"):
        x, b, mats, owned = kron_solve(npts, pads, periods, kind, transposed=transposed)
    assert_close(x, reference_solution(mats, b, transposed)[owned])


@pytest.mark.mpi
@pytest.mark.parametrize("npts, pads, periods", [c for c in CASES if len(c[0]) > 1])
@pytest.mark.parametrize("kind", SOLVER_KINDS)
def test_kronecker_solve_matches_reference_mpi(npts, pads, periods, kind):
    with xp.use_backend("numpy"):
        x, b, mats, owned = kron_solve(npts, pads, periods, kind, comm=MPI.COMM_WORLD)
    assert_close(x, reference_solution(mats, b)[owned])


_MPI_LAUNCHER_PREFIXES = ("OMPI_", "PMIX_", "PMI_", "HYDRA_", "I_MPI_", "SLURM_")


def _real_cupy_installed():
    module = sys.modules.get("cupy")
    if module is not None:
        return not getattr(module, "__cunumpy_fake__", False)
    return importlib.util.find_spec("cupy") is not None


@pytest.mark.skipif(_real_cupy_installed(),
                    reason="CuPy is installed: the fake CuPy cannot be used, the GPU tests cover this")
def test_device_solve_on_fake_cupy():
    """The device path with cunumpy's fake CuPy, in a serial child process with a clean environment."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_MPI_LAUNCHER_PREFIXES)}
    env.pop("CUNUMPY_BACKEND", None)
    env.update(MAYBEMPI="0", CUNUMPY_FAKE_CUPY="1")
    code = "from feectools.linalg.tests.test_kron_device_solve import run_device_checks; run_device_checks()"
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "device checks passed" in proc.stdout


# ===============================================================================
# GPU tests
# ===============================================================================
@requires_cupy
@pytest.mark.parametrize("npts, pads, periods", CASES)
@pytest.mark.parametrize("kind", SOLVER_KINDS)
@pytest.mark.parametrize("transposed", [False, True])
def test_device_kronecker_solve_gpu(npts, pads, periods, kind, transposed):
    compare_device_with_host(npts, pads, periods, kind, transposed)


@requires_cupy
@pytest.mark.parametrize("kind", SOLVER_KINDS)
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
def test_device_1d_solve_gpu(kind, periodic, transposed):
    compare_1d_device_with_host(kind, periodic, transposed)


@requires_cupy
@pytest.mark.mpi
@pytest.mark.parametrize("npts, pads, periods", [c for c in CASES if len(c[0]) > 1])
@pytest.mark.parametrize("kind", SOLVER_KINDS)
def test_device_kronecker_solve_gpu_mpi(npts, pads, periods, kind):
    compare_device_with_host(npts, pads, periods, kind, comm=MPI.COMM_WORLD)
