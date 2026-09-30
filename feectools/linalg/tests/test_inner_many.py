#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
Tests for the fused multi-scalar reduction `VectorSpace.inner_many`, and for
the equivalence of the `inner_many`-based PCG with the textbook recurrence it
replaces.

These run on whichever array backend is active (NumPy or CuPy, selected with
the ARRAY_BACKEND environment variable), serially and under MPI.
"""
from math import sqrt

import pytest
import cunumpy as xp

from feectools.ddm.mpi import mpi as MPI
from feectools.ddm.cart import DomainDecomposition, CartDecomposition
from feectools.linalg.basic import IdentityOperator, MatrixFreeLinearOperator
from feectools.linalg.block import BlockVectorSpace, BlockVector
from feectools.linalg.solvers import inverse
from feectools.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix

# ===============================================================================
def compute_global_starts_ends(domain_decomposition, npts):
    ndims = len(npts)
    global_starts = [None] * ndims
    global_ends = [None] * ndims

    for axis in range(ndims):
        ee = domain_decomposition.global_element_ends[axis]
        global_ends[axis] = ee.copy()
        global_ends[axis][-1] = npts[axis] - 1
        global_starts[axis] = xp.array([0] + (global_ends[axis][:-1] + 1).tolist())

    return global_starts, global_ends


# ===============================================================================
def make_space(npts, pads, dtype, comm=None):
    """Build a StencilVectorSpace over `npts` points, distributed if `comm`."""
    ndim = len(npts)
    D = DomainDecomposition(list(npts), periods=[True] * ndim, comm=comm)
    global_starts, global_ends = compute_global_starts_ends(D, list(npts))
    C = CartDecomposition(D, list(npts), global_starts, global_ends,
                          pads=list(pads), shifts=[1] * ndim)
    return StencilVectorSpace(C, dtype=dtype)


# ===============================================================================
def fill(v, seed):
    """Fill the owned coefficients of `v` with reproducible values."""
    V = v.space
    ranges = [range(int(s), int(e) + 1) for s, e in zip(V.starts, V.ends)]

    def value(idx):
        r = sum((k + 1) * (i + seed) for k, i in enumerate(idx)) % 17 + 1
        return r + 1j * (r % 5 - 2) if V.dtype == complex else float(r)

    if len(ranges) == 1:
        for i1 in ranges[0]:
            v[i1] = value((i1,))
    elif len(ranges) == 2:
        for i1 in ranges[0]:
            for i2 in ranges[1]:
                v[i1, i2] = value((i1, i2))
    else:
        for i1 in ranges[0]:
            for i2 in ranges[1]:
                for i3 in ranges[2]:
                    v[i1, i2, i3] = value((i1, i2, i3))
    v.update_ghost_regions()
    return v


# ===============================================================================
def assert_same(got, expected):
    """Compare two scalars that must agree to the last bit: `inner_many` sums
    exactly the same terms in the same order as `inner`."""
    assert complex(got) == complex(expected)


# ===============================================================================
# SERIAL TESTS
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('npts, pads', [((15,), (2,)),
                                        ((8, 11), (2, 3)),
                                        ((7, 6, 9), (1, 2, 3))])
def test_inner_many_serial(dtype, npts, pads):
    """inner_many agrees with the same inner products taken one at a time."""
    V = make_space(npts, pads, dtype)
    x, y, z = (fill(StencilVector(V), s) for s in (1, 2, 3))

    expected = (V.inner(x, x), V.inner(x, y), V.inner(z, y), V.inner(y, z))
    got = V.inner_many((x, x), (x, y), (z, y), (y, z))

    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert_same(g, e)

    # The dtype of the space is carried by the results, as it is for `inner`
    for g in got:
        assert xp.dtype(type(g)) == xp.dtype(dtype)

    # Degenerate and single-pair cases
    assert V.inner_many() == ()
    assert_same(V.inner_many((x, y))[0], V.inner(x, y))


# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
def test_inner_many_is_conjugating(dtype):
    """The first argument of each pair is conjugated, as for `inner`: a plain
    (non-conjugating) dot product would pass the real case but fail here."""
    V = make_space((6, 5), (1, 2), dtype)
    x, y = (fill(StencilVector(V), s) for s in (1, 2))

    xy, yx = V.inner_many((x, y), (y, x))

    assert_same(xy, V.inner(x, y))
    assert_same(yx, complex(xy).conjugate())
    # inner(x, x) is real and positive for a non-zero vector
    assert complex(V.inner_many((x, x))[0]).imag == 0.0
    assert complex(V.inner_many((x, x))[0]).real > 0.0


# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
def test_inner_many_results_are_independent(dtype):
    """Results must survive later calls, which recycle the same scratch."""
    V = make_space((6, 7), (2, 1), dtype)
    x, y = (fill(StencilVector(V), s) for s in (1, 2))

    first = V.inner_many((x, x), (x, y))
    kept = tuple(complex(c) for c in first)

    for _ in range(3):
        V.inner_many((y, y), (y, x), (x, x))

    assert tuple(complex(c) for c in first) == kept


# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
def test_vector_inner_many(dtype):
    """The Vector-level shorthand matches the space-level call."""
    V = make_space((5, 6), (1, 1), dtype)
    x, y, z = (fill(StencilVector(V), s) for s in (1, 2, 3))

    got = x.inner_many(x, y, z)
    for g, e in zip(got, (x.inner(x), x.inner(y), x.inner(z))):
        assert_same(g, e)


# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
def test_block_inner_many_serial(dtype):
    """A BlockVectorSpace sums its blocks into one fused reduction."""
    V1 = make_space((6, 5), (1, 2), dtype)
    V2 = make_space((4, 7), (2, 1), dtype)
    W = BlockVectorSpace(V1, V2)

    def block(seeds):
        return BlockVector(W, [fill(StencilVector(V1), seeds[0]),
                               fill(StencilVector(V2), seeds[1])])

    x, y = block((1, 2)), block((3, 4))

    expected = (W.inner(x, x), W.inner(x, y), W.inner(y, x))
    got = W.inner_many((x, x), (x, y), (y, x))
    for g, e in zip(got, expected):
        assert_same(g, e)

    # Nested product spaces reduce through the same single collective
    WW = BlockVectorSpace(W, W)
    xx = BlockVector(WW, [x, y])
    yy = BlockVector(WW, [y, x])
    assert_same(WW.inner_many((xx, yy))[0], WW.inner(xx, yy))


# ===============================================================================
def _pcg_reference(A, b, pc, x0, tol, maxiter):
    """The PCG recurrence as it was written before `inner_many`, used as the
    reference the optimized solver must reproduce."""
    x = x0.copy()
    v = b.space.zeros()
    r = b.space.zeros()

    A.dot(x, out=v)
    b.copy(out=r)
    r -= v
    nrmr_sqr = r.inner(r).real
    s = pc.dot(r)
    am = s.inner(r)
    p = s.copy()

    tol_sqr = tol ** 2
    for k in range(2, maxiter + 1):
        if nrmr_sqr < tol_sqr:
            k -= 1
            break
        v = A.dot(p, out=v)
        l = am / v.inner(p)
        x.mul_iadd(l, p)
        r.mul_iadd(-l, v)
        nrmr_sqr = r.inner(r).real
        s = pc.dot(r, out=s)
        am1 = s.inner(r)
        s.mul_iadd((am1 / am), p)
        s, p = p, s
        am = am1

    return x, {'niter': k, 'success': nrmr_sqr < tol_sqr,
               'res_norm': sqrt(nrmr_sqr)}


# ===============================================================================
def _laplacian(V):
    """Symmetric positive-definite stencil matrix on the space V."""
    ndim = len(V.npts)
    A = StencilMatrix(V, V)
    center = [slice(None)] * ndim + [0] * ndim
    A[tuple(center)] = 2.0 * ndim + 0.5
    for axis in range(ndim):
        for shift in (-1, 1):
            key = [slice(None)] * ndim + [0] * ndim
            key[ndim + axis] = shift
            A[tuple(key)] = -1.0
    A.remove_spurious_entries()
    return A


# ===============================================================================
def _shifted_identity(V, shift):
    """A Hermitian positive-definite operator built from vector operations
    only. Used to exercise the complex case, which the compiled stencil matvec
    kernel does not support (it is typed on float64)."""
    def dot(v, out=None):
        w = v.copy(out=out)
        w *= V.dtype(shift)
        return w

    return MatrixFreeLinearOperator(domain=V, codomain=V, dot=dot,
                                    dot_transpose=dot)


# ===============================================================================
@pytest.mark.parametrize('npts, pads', [((12,), (1,)), ((7, 9), (1, 2))])
def test_pcg_matches_reference(npts, pads):
    """The solver reproduces the reference recurrence: same solution, same
    iteration count, same reported residual."""
    V = make_space(npts, pads, float)
    A = _laplacian(V)
    b = fill(StencilVector(V), 5)
    x0 = StencilVector(V)
    pc = IdentityOperator(V)

    tol, maxiter = 1e-12, 200
    x_ref, info_ref = _pcg_reference(A, b, pc, x0, tol, maxiter)

    solver = inverse(A, 'pcg', pc=pc, x0=x0, tol=tol, maxiter=maxiter)
    x_new = solver.solve(b)
    info_new = solver.get_info()

    assert info_new['niter'] == info_ref['niter']
    assert info_new['success'] == info_ref['success']
    assert info_new['success']
    assert abs(info_new['res_norm'] - info_ref['res_norm']) <= 1e-10 * max(
        1.0, info_ref['res_norm'])

    diff = x_new - x_ref
    assert sqrt(abs(complex(diff.inner(diff)))) <= 1e-10 * sqrt(
        abs(complex(x_ref.inner(x_ref))))

    # And it really solved the system
    res = b - A.dot(x_new)
    assert sqrt(abs(complex(res.inner(res)))) <= 1e-6


# ===============================================================================
def test_pcg_complex_matches_reference():
    """PCG on a complex space: the Hermitian inner products keep the recurrence
    real where it has to be, and the fused reductions change nothing."""
    V = make_space((6, 7), (1, 2), complex)
    A = _shifted_identity(V, 3.0)
    b = fill(StencilVector(V), 5)
    x0 = StencilVector(V)
    pc = IdentityOperator(V)

    tol, maxiter = 1e-13, 100
    x_ref, info_ref = _pcg_reference(A, b, pc, x0, tol, maxiter)

    solver = inverse(A, 'pcg', pc=pc, x0=x0, tol=tol, maxiter=maxiter)
    x_new = solver.solve(b)

    assert solver.get_info()['niter'] == info_ref['niter']
    assert solver.get_info()['success']

    # A = 3*I, so the solution is b/3 -- known in closed form
    expected = b.copy()
    expected *= complex(1.0 / 3.0)
    diff = x_new - expected
    assert sqrt(abs(complex(diff.inner(diff)))) <= 1e-10


# ===============================================================================
def test_pcg_recycle_and_out():
    """`recycle` and `out=` keep working with the fused reductions."""
    V = make_space((8, 8), (1, 1), float)
    A = _laplacian(V)
    b = fill(StencilVector(V), 5)
    x0 = StencilVector(V)

    solver = inverse(A, 'pcg', x0=x0, tol=1e-12, maxiter=200, recycle=True)

    out = StencilVector(V)
    returned = solver.solve(b, out=out)
    assert returned is out
    niter_first = solver.get_info()['niter']

    # With `recycle` the solution was stored as the next initial guess, so
    # solving the same system again converges immediately.
    solver.solve(b)
    assert solver.get_info()['niter'] <= niter_first
    assert solver.get_info()['success']


# ===============================================================================
# PARALLEL TESTS
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('npts, pads', [((16, 12), (2, 3)),
                                        ((10, 9, 8), (1, 2, 1))])
@pytest.mark.mpi
def test_inner_many_parallel(dtype, npts, pads):
    """Under MPI, the fused reduction agrees with the unfused one."""
    comm = MPI.COMM_WORLD
    V = make_space(npts, pads, dtype, comm=comm)
    x, y, z = (fill(StencilVector(V), s) for s in (1, 2, 3))

    expected = (V.inner(x, x), V.inner(x, y), V.inner(z, y))
    got = V.inner_many((x, x), (x, y), (z, y))
    for g, e in zip(got, expected):
        assert_same(g, e)

    # Every rank must come out of the collective with the same values
    per_rank = comm.allgather(tuple(complex(g) for g in got))
    assert all(vals == per_rank[0] for vals in per_rank)


# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.mpi
def test_block_inner_many_parallel(dtype):
    """Blocks distributed over the same communicator share one collective."""
    comm = MPI.COMM_WORLD
    V1 = make_space((12, 10), (1, 2), dtype, comm=comm)
    V2 = make_space((8, 14), (2, 1), dtype, comm=comm)
    W = BlockVectorSpace(V1, V2)

    x = BlockVector(W, [fill(StencilVector(V1), 1), fill(StencilVector(V2), 2)])
    y = BlockVector(W, [fill(StencilVector(V1), 3), fill(StencilVector(V2), 4)])

    expected = (W.inner(x, x), W.inner(x, y))
    got = W.inner_many((x, x), (x, y))
    for g, e in zip(got, expected):
        assert_same(g, e)


# ===============================================================================
@pytest.mark.mpi
def test_block_inner_many_mixed_serial_and_parallel():
    """A product of a distributed block and a replicated (serial) one cannot
    share one collective: summing the blocks locally first would count the
    serial block once per rank. The result must still be right."""
    comm = MPI.COMM_WORLD
    V_par = make_space((12, 10), (1, 2), float, comm=comm)
    V_ser = make_space((6, 6), (1, 1), float)
    W = BlockVectorSpace(V_par, V_ser)

    assert W._reduction_comms() is None  # fused path correctly declined

    x = BlockVector(W, [fill(StencilVector(V_par), 1),
                        fill(StencilVector(V_ser), 2)])
    y = BlockVector(W, [fill(StencilVector(V_par), 3),
                        fill(StencilVector(V_ser), 4)])

    expected = V_par.inner(x.blocks[0], y.blocks[0]) \
        + V_ser.inner(x.blocks[1], y.blocks[1])
    assert_same(W.inner_many((x, y))[0], expected)
    assert_same(W.inner(x, y), expected)


# ===============================================================================
@pytest.mark.mpi
def test_pcg_matches_reference_parallel():
    """The distributed solver reproduces the reference recurrence too."""
    comm = MPI.COMM_WORLD
    V = make_space((16, 12), (1, 2), float, comm=comm)
    A = _laplacian(V)
    b = fill(StencilVector(V), 5)
    x0 = StencilVector(V)
    pc = IdentityOperator(V)

    tol, maxiter = 1e-12, 300
    x_ref, info_ref = _pcg_reference(A, b, pc, x0, tol, maxiter)

    solver = inverse(A, 'pcg', pc=pc, x0=x0, tol=tol, maxiter=maxiter)
    x_new = solver.solve(b)
    info_new = solver.get_info()

    assert info_new['niter'] == info_ref['niter']
    assert info_new['success'] == info_ref['success']

    diff = x_new - x_ref
    assert sqrt(abs(complex(diff.inner(diff)))) <= 1e-10 * sqrt(
        abs(complex(x_ref.inner(x_ref))))


# ===============================================================================
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, '-v']))
