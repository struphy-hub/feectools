"""Child process of `test_inner_on_device.py`: inner products and Krylov solvers under cunumpy's fake CuPy.

Run with ``CUNUMPY_FAKE_CUPY=1 CUNUMPY_BACKEND=cupy MAYBEMPI=0`` in a clean environment (the fake must be
installed before cunumpy picks its backend, which is why this runs in its own process). Prints one JSON object.

The fake CuPy cannot launch CUDA kernels, so `Kernel` calls with device arrays run the kernel's host
implementation on the host buffers of the fake device arrays (no copy, like a launch on the device data).
Two counts are reported: ``to_host`` of `cunumpy.profiling.count_transfers` (copies made through cunumpy,
e.g. the convergence test of a solver) and all device -> host copies, counted by wrapping the methods of the
fake ``cupy.ndarray`` that copy (``.get()``, which `to_numpy` uses, ``.item()``, ``float()``, ``bool()``, ...):
`count_transfers` does not see the implicit conversions.
"""
import json
import sys

import cunumpy as xp
import numpy as np
from cunumpy import _fake_cupy
from cunumpy._dispatch import Kernel

assert _fake_cupy.is_active() and xp.get_backend() == "cupy", "run with CUNUMPY_FAKE_CUPY=1 CUNUMPY_BACKEND=cupy"

import cupy  # noqa: E402  (the fake)


# --- device kernel launches: the host implementation on the device buffers --------------------------------
def _buffer(a):
    return a._a if isinstance(a, cupy.ndarray) else a


def _launch(self, *args, **launch):
    if any(isinstance(a, cupy.ndarray) for a in args):
        return self.host_kernel.kernel(*[_buffer(a) for a in args])
    return self.host_kernel.kernel(*args)


Kernel.__call__ = _launch

# --- every device -> host copy, implicit or not ---------------------------------------------------------------
COPIES = []


def _counting(name, method):
    def wrapper(self, *args, **kwargs):
        COPIES.append(name)
        return method(self, *args, **kwargs)
    return wrapper


for _name in ("__bool__", "__float__", "__complex__", "__int__", "__index__", "__format__", "get"):
    setattr(cupy.ndarray, _name, _counting(_name, getattr(cupy.ndarray, _name)))
cupy.ndarray.item = _counting("item", lambda self, *args: self._a.item(*args))

from feectools.ddm.cart import CartDecomposition, DomainDecomposition  # noqa: E402
from feectools.linalg.block import BlockVector, BlockVectorSpace  # noqa: E402
from feectools.linalg.solvers import inverse  # noqa: E402
from feectools.linalg.stencil import StencilMatrix, StencilVector, StencilVectorSpace  # noqa: E402

N, P = 40, 2
TOL = 1e-12


def problem():
    """A symmetric positive definite banded matrix, an exact solution and the right-hand side, on the active backend."""
    D = DomainDecomposition([N - P], [False])
    C = CartDecomposition(D, [N], [np.array([0])], [np.array([N - 1])], [P], [1])
    V = StencilVectorSpace(C, dtype=float)
    A = StencilMatrix(V, V)
    A[:, -P:0] = -1
    A[:, 0:1] = 2 * P
    A[:, 1:P + 1] = -1
    A.remove_spurious_entries()
    xe = StencilVector(V)
    xe[0:N] = xp.asarray(np.random.default_rng(0).random(N))
    return V, A, xe, A.dot(xe)


def solver_kwargs(name, A):
    return {"pc": A.diagonal(inverse=True)} if name in ("pcg", "pbicgstab") else {}


def host(a):
    return xp.to_numpy(a).tolist()


def check_inner():
    V, A, xe, b = problem()
    y = b.copy()
    out = {}
    s1 = xe.inner(y)
    s2 = xe.inner(xe)  # reuses the reduction buffer of xe
    out["stencil_type"] = type(s1).__module__ + "." + type(s1).__name__
    out["stencil_ndim"] = s1.ndim
    out["independent"] = host(s1) != host(s2)
    W = BlockVectorSpace(V, V)
    u = BlockVector(W, blocks=[xe, y])
    sb = u.inner(u)
    out["block_type"] = type(sb).__module__ + "." + type(sb).__name__
    out["block_value"] = host(sb)
    # axpy with a device scalar, no copy to the host
    z = y.copy()
    COPIES.clear()
    with xp.profiling.count_transfers() as c:
        z.mul_iadd(s1, xe)
    out["axpy_transfers"] = c.to_host + len(COPIES)
    out["axpy_value"] = host(z._data)
    out["inner_values"] = [host(s1), host(s2)]
    return out


def run_solvers(names):
    out = {}
    for name in names:
        V, A, xe, b = problem()
        kw = solver_kwargs(name, A)
        inverse(A, name, tol=TOL, maxiter=1000, **kw).solve(b)  # warm up
        counts = {}
        for maxiter in (5, 15):
            S = inverse(A, name, tol=1e-30, maxiter=maxiter, **kw)
            COPIES.clear()
            with xp.profiling.count_transfers() as c:
                S.solve(b)
            counts[maxiter] = (c.to_host, len(COPIES), c.to_device)
        S = inverse(A, name, tol=TOL, maxiter=1000, **kw)
        x = S.solve(b)
        info = S.get_info()
        out[name] = {
            # per iteration: the difference between 15 and 5 iterations, over 10 iterations
            "to_host_per_iter": (counts[15][0] - counts[5][0]) / 10,
            "copies_per_iter": (counts[15][1] - counts[5][1]) / 10,
            "to_device": counts[15][2],
            "x": host(x._data),
            "niter": int(info["niter"]),
            "res_norm": float(info["res_norm"]),
            "info_types": sorted({type(v).__module__.split(".")[0] for v in info.values()}),
        }
        # the NumPy reference, same process
        with xp.use_backend("numpy"):
            V, A, xe, b = problem()
            S = inverse(A, name, tol=TOL, maxiter=1000, **solver_kwargs(name, A))
            x = S.solve(b)
            out[name]["numpy_x"] = x._data.tolist()
            out[name]["numpy_niter"] = int(S.get_info()["niter"])
    return out


if __name__ == "__main__":
    result = {"inner": check_inner(), "solvers": run_solvers(sys.argv[1:])}
    print("RESULT " + json.dumps(result))
