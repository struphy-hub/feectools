"""Inner products stay on the device, and the Krylov solvers keep their scalars there.

On the CuPy backend `StencilVectorSpace.inner` (and `BlockVectorSpace.inner`) return a 0-d device array, `axpy`
accepts it without copying it to the host, and CG, PCG, BiCG, BiCGStab and PBiCGStab copy one scalar per
iteration to the host (the residual norm of the convergence test). On the NumPy backend nothing changes.

Without a GPU the CuPy path runs under cunumpy's fake CuPy, in a child process
(`inner_on_device_child.py`); the GPU tests are skipped.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import requires_cupy

from feectools.ddm.cart import CartDecomposition, DomainDecomposition
from feectools.linalg.block import BlockVector, BlockVectorSpace
from feectools.linalg.solvers import inverse
from feectools.linalg.stencil import StencilMatrix, StencilVector, StencilVectorSpace

CHILD = Path(__file__).with_name("inner_on_device_child.py")
DEVICE_SOLVERS = ("cg", "pcg", "bicg", "bicgstab", "pbicgstab")
HOST_SOLVERS = ("minres", "lsmr", "gmres")


def problem(n=40, p=2):
    D = DomainDecomposition([n - p], [False])
    C = CartDecomposition(D, [n], [np.array([0])], [np.array([n - 1])], [p], [1])
    V = StencilVectorSpace(C, dtype=float)
    A = StencilMatrix(V, V)
    A[:, -p:0] = -1
    A[:, 0:1] = 2 * p
    A[:, 1:p + 1] = -1
    A.remove_spurious_entries()
    xe = StencilVector(V)
    xe[0:n] = xp.asarray(np.random.default_rng(0).random(n))
    return V, A, xe, A.dot(xe)


# ===============================================================================
# NumPy backend: unchanged
# ===============================================================================
def test_inner_numpy_returns_numpy_scalar():
    with xp.use_backend("numpy"):
        V, A, xe, b = problem()
        s1 = xe.inner(b)
        s2 = xe.inner(xe)  # reuses the reduction buffer of xe
        assert isinstance(s1, np.floating) and isinstance(s2, np.floating)
        assert s1 == pytest.approx(float(np.dot(xe.toarray(), b.toarray())), rel=1e-14)
        assert s2 == pytest.approx(float(np.dot(xe.toarray(), xe.toarray())), rel=1e-14)

        W = BlockVectorSpace(V, V)
        u = BlockVector(W, blocks=[xe, b])
        assert isinstance(u.inner(u), np.floating)


@pytest.mark.parametrize("solver", DEVICE_SOLVERS + HOST_SOLVERS)
def test_solver_info_numpy(solver):
    """The convergence information is made of host scalars, as before."""
    with xp.use_backend("numpy"):
        V, A, xe, b = problem()
        kw = {"pc": A.diagonal(inverse=True)} if solver in ("pcg", "pbicgstab") else {}
        S = inverse(A, solver, tol=1e-12, maxiter=1000, **kw)
        x = S.solve(b)
        info = S.get_info()
        assert info["success"]
        assert isinstance(info["res_norm"], float)
        assert np.allclose(x.toarray(), xe.toarray(), atol=1e-10)


# ===============================================================================
# CuPy backend without a GPU: fake CuPy in a child process
# ===============================================================================
def fake_cupy_env():
    """A clean environment for a serial child process with the fake CuPy (no MPI launcher variables)."""
    prefixes = ("OMPI_", "PMIX_", "PMI_", "HYDRA_", "I_MPI_", "SLURM_", "CUNUMPY_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    env.update(CUNUMPY_FAKE_CUPY="1", CUNUMPY_BACKEND="cupy", MAYBEMPI="0")
    root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join([root, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


@pytest.fixture(scope="module")
def fake_cupy_result():
    try:
        import cupy  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("CuPy is installed: the fake CuPy cannot be used")
    proc = subprocess.run(
        [sys.executable, str(CHILD), *DEVICE_SOLVERS, *HOST_SOLVERS],
        env=fake_cupy_env(), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_inner_returns_device_scalar(fake_cupy_result):
    r = fake_cupy_result["inner"]
    assert r["stencil_type"] == "cupy.ndarray" and r["stencil_ndim"] == 0
    assert r["block_type"] == "cupy.ndarray"
    # the result is a copy, not a view of the reduction buffer that the next inner product overwrites
    assert r["independent"]


def test_axpy_with_device_scalar_does_not_copy(fake_cupy_result):
    assert fake_cupy_result["inner"]["axpy_transfers"] == 0


@pytest.mark.parametrize("solver", DEVICE_SOLVERS)
def test_one_host_copy_per_iteration(fake_cupy_result, solver):
    """Before: 2 (CG), 3 (PCG), 4 (BiCG), 6 (BiCGStab), 5 (PBiCGStab) copies per iteration."""
    r = fake_cupy_result["solvers"][solver]
    assert r["to_host_per_iter"] == 1
    assert r["copies_per_iter"] == 1
    assert r["to_device"] == 0
    assert r["info_types"] in (["builtins"], ["builtins", "numpy"])


@pytest.mark.parametrize("solver", DEVICE_SOLVERS + HOST_SOLVERS)
def test_same_result_as_numpy(fake_cupy_result, solver):
    r = fake_cupy_result["solvers"][solver]
    assert r["niter"] == r["numpy_niter"]
    np.testing.assert_allclose(r["x"], r["numpy_x"], rtol=1e-12, atol=1e-14)


# ===============================================================================
# GPU
# ===============================================================================
@requires_cupy
def test_inner_on_gpu():
    with xp.use_backend("cupy"):
        V, A, xe, b = problem()
        s = xe.inner(b)
        assert xp.is_gpu(s) and s.ndim == 0
        assert float(s) == pytest.approx(float(xp.to_numpy(xe._data) @ xp.to_numpy(b._data)), rel=1e-12)


@requires_cupy
@pytest.mark.parametrize("solver", DEVICE_SOLVERS)
def test_one_host_copy_per_iteration_on_gpu(solver):
    with xp.use_backend("cupy"):
        V, A, xe, b = problem()
        kw = {"pc": A.diagonal(inverse=True)} if solver in ("pcg", "pbicgstab") else {}
        inverse(A, solver, tol=1e-12, maxiter=1000, **kw).solve(b)
        counts = []
        for maxiter in (5, 15):
            S = inverse(A, solver, tol=1e-30, maxiter=maxiter, **kw)
            with xp.profiling.count_transfers() as c:
                S.solve(b)
            counts.append(c.to_host)
        assert (counts[1] - counts[0]) == 10
