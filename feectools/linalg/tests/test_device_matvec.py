#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
Tests for `StencilMatrix.dot`, which calls the kernel `stencil_dot_<n>d` of the
active backend: the pyccel kernel on NumPy, its CUDA version on CuPy.

The reference is the same stencil sum expressed with shifted array views. It is
backend-independent, so on the NumPy backend these tests check the pyccel kernel
against the reference, and on the CuPy backend the CUDA kernel. The kernels are
also compared with each other directly, on a GPU (test_cuda_parity.py) and by
CPU emulation (test_cuda_emulation.py).
"""
import itertools

import numpy as np
import pytest
import cunumpy as xp

from feectools.ddm.cart import DomainDecomposition, CartDecomposition
from feectools.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix, stencil_kernels

# ===============================================================================
def make_space(npts, pads, dtype, periods=None, shifts=None):
    ndim = len(npts)
    D = DomainDecomposition(list(npts), periods=list(periods) if periods else [True] * ndim)
    global_starts, global_ends = [], []
    for axis in range(ndim):
        ee = D.global_element_ends[axis].copy()
        ee[-1] = npts[axis] - 1
        global_ends.append(ee)
        global_starts.append(xp.array([0] + (ee[:-1] + 1).tolist()))
    C = CartDecomposition(D, list(npts), global_starts, global_ends,
                          pads=list(pads), shifts=list(shifts) if shifts else [1] * ndim)
    return StencilVectorSpace(C, dtype=dtype)


# ===============================================================================
def dot_args(A, ndim):
    """The per-direction matvec parameters of `A`, as lists of ints."""
    def seq(key):
        val = A._args[key]
        if ndim == 1:
            return [int(val)]
        return [int(k) for k in xp.to_numpy(val)]

    return {k: seq(k) for k in
            ('s_in', 'p_in', 'add', 's_out', 'e_out', 'p_out')}


# ===============================================================================
def reference_matvec(A, v, out):
    """
    out = A @ v, as a sum of shifted elementwise products.

    The matrix has n_k diagonals along direction k (from its data, pads
    q_k = (n_k - 1) // 2 up to the pads p_in of the domain); interior rows use
    all of them, the last row along a direction uses n_k - 1 + add, so every
    combination of (interior, last) over the directions is accumulated
    separately. Diagonal d of row i is the column i - q + d.
    """
    ndim = v.space.ndim
    a = dot_args(A, ndim)
    n = [e - s + 1 for s, e in zip(a['s_out'], a['e_out'])]
    p_out, p_in, add = a['p_out'], a['p_in'], a['add']
    n_diags = [int(n) for n in A._data.shape[ndim:]]
    off = [so - si + pi - (nd - 1) // 2 for so, si, pi, nd in zip(a['s_out'], a['s_in'], p_in, n_diags)]

    out._data[...] = 0

    for last in itertools.product([False, True], repeat=ndim):
        rows = []
        for k in range(ndim):
            if last[k]:
                rows.append(slice(p_out[k] + n[k] - 1, p_out[k] + n[k]))
            else:
                rows.append(slice(p_out[k], p_out[k] + n[k] - 1))
        if any(r.stop <= r.start for r in rows):
            continue

        nrow = [r.stop - r.start for r in rows]
        base = [r.start - p_out[k] for k, r in enumerate(rows)]
        bounds = [n_diags[k] - 1 + add[k] if last[k] else n_diags[k] for k in range(ndim)]

        for d in itertools.product(*[range(b) for b in bounds]):
            src = tuple(slice(off[k] + base[k] + d[k],
                              off[k] + base[k] + d[k] + nrow[k])
                        for k in range(ndim))
            out._data[tuple(rows)] += A._data[tuple(rows) + tuple(d)] * v._data[src]

    return out


# ===============================================================================
def fill_like(arr, seed):
    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in arr.shape)
    values = rng.random(shape)
    if np.dtype(arr.dtype).kind == 'c':
        values = values + 1j * rng.random(shape)
    return xp.asarray(values.astype(arr.dtype))


# ===============================================================================
def build(npts_domain, npts_codomain, pads, dtype, matrix_pads=None, periods=None):
    V = make_space(npts_domain, pads, dtype, periods)
    W = V if npts_domain == npts_codomain else make_space(npts_codomain, pads, dtype, periods)

    A = StencilMatrix(V, W, pads=matrix_pads)
    A._data[...] = fill_like(A._data, 1)
    A.remove_spurious_entries()

    v = StencilVector(V)
    v._data[...] = fill_like(v._data, 2)
    v.update_ghost_regions()

    return V, W, A, v


# ===============================================================================
# Square matrices, and rectangular ones whose spaces differ by one point in a
# direction -- the case that makes `add` zero there, as for derivative operators.
# The cases with matrix pads have fewer diagonals than 2 * p + 1 (blocks between
# spaces of different degree, derivative-type stencils), on non-periodic spaces.
CASES = [
    ('1d-square', (24,), (24,), (2,)),
    ('1d-rect', (23,), (24,), (2,)),
    ('2d-square', (10, 12), (10, 12), (2, 3)),
    ('2d-rect', (11, 10), (12, 10), (2, 2)),
    ('2d-rect-both', (11, 9), (12, 10), (1, 2)),
    ('3d-square', (7, 8, 9), (7, 8, 9), (1, 2, 3)),
    ('3d-rect', (8, 8, 9), (9, 8, 10), (1, 2, 2)),
]

# (name, npts of the domain, npts of the codomain, pads of the spaces, pads of the matrix, periods)
CASES_FEWER_DIAGONALS = [
    ('1d-fewer', (24,), (24,), (3,), (1,), (False,)),
    ('1d-rect-fewer', (13,), (12,), (2,), (1,), (False,)),
    ('2d-fewer', (10, 9), (10, 9), (3, 2), (1, 2), (False, True)),
    ('2d-rect-fewer', (9, 8), (8, 8), (2, 3), (2, 1), (False, False)),
    ('3d-fewer', (6, 7, 8), (6, 7, 8), (2, 3, 2), (1, 1, 2), (True, False, True)),
    ('3d-rect-nonperiodic', (8, 7, 9), (7, 7, 8), (2, 2, 3), None, (False, True, False)),
    ('3d-rect-fewer', (7, 8, 6), (8, 7, 6), (3, 2, 2), (1, 1, 1), (False, False, True)),
    ('3d-rect-one-diagonal', (7, 6, 5), (8, 6, 6), (1, 2, 1), (0, 2, 0), (False, True, False)),
]


@pytest.mark.parametrize('name, npts_d, npts_c, pads', CASES,
                         ids=[c[0] for c in CASES])
def test_matvec_matches_reference(name, npts_d, npts_c, pads):
    """`StencilMatrix.dot` agrees with the shifted-view reference, whichever
    kernel the active backend selects."""
    V, W, A, v = build(npts_d, npts_c, pads, float)

    got = A.dot(v, out=StencilVector(W))
    ref = reference_matvec(A, v, StencilVector(W))

    assert xp.allclose(got._data, ref._data, rtol=0.0, atol=1e-12)


# ===============================================================================
def test_matvec_leaves_padding_zeroed():
    """The kernel writes only the owned rows; the padding of `out` must come
    out zeroed, as it does on the host path, even when `out` is reused."""
    V, W, A, v = build((7, 8), (7, 8), (2, 3), float)

    out = StencilVector(W)
    out._data[...] = fill_like(out._data, 7)   # dirty the buffer, padding too
    A.dot(v, out=out)

    a = dot_args(A, 2)
    n = [e - s + 1 for s, e in zip(a['s_out'], a['e_out'])]
    interior = tuple(slice(a['p_out'][k], a['p_out'][k] + n[k]) for k in range(2))

    mask = xp.ones(tuple(int(s) for s in out._data.shape), dtype=bool)
    mask[interior] = False
    assert not bool(xp.any(out._data[mask] != 0))


# ===============================================================================
def test_matvec_out_and_repeated_calls_agree():
    """Reusing an `out` vector gives the same answer as a fresh one."""
    V, W, A, v = build((7, 8, 9), (7, 8, 9), (1, 2, 2), float)

    fresh = A.dot(v)
    reused = StencilVector(W)
    for _ in range(3):
        A.dot(v, out=reused)

    assert xp.allclose(fresh._data, reused._data, rtol=0.0, atol=1e-14)
    assert not fresh.ghost_regions_in_sync


# ===============================================================================
@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_dot_calls_the_folder_kernel(ndim):
    """`StencilMatrix.dot` calls the kernel of the folder `stencil_dot_<n>d`,
    which has a CUDA version: on CuPy there is no host fallback."""
    name, npts_d, npts_c, pads = next(c for c in CASES if len(c[1]) == ndim)
    V, W, A, v = build(npts_d, npts_c, pads, float)
    assert A._func is stencil_kernels['dot'][ndim]
    assert A._func.name == f'stencil_dot_{ndim}d' and A._func.has_cuda


# ===============================================================================
@pytest.mark.parametrize('name, npts_d, npts_c, pads, matrix_pads, periods', CASES_FEWER_DIAGONALS,
                         ids=[c[0] for c in CASES_FEWER_DIAGONALS])
def test_fewer_diagonals_match_dense_reference(name, npts_d, npts_c, pads, matrix_pads, periods):
    """Matrices the kernels used to reject (pads of the matrix smaller than
    those of the spaces, i.e. fewer than 2 * p + 1 diagonals): dot, vdot and
    transpose agree with the dense matrix, and dot with the shifted-view
    reference."""
    V, W, A, v = build(npts_d, npts_c, pads, float, matrix_pads, periods)
    dense = xp.to_numpy(A.toarray())
    x = xp.to_numpy(v.toarray())

    got = A.dot(v)
    assert np.allclose(xp.to_numpy(got.toarray()), dense @ x, rtol=0.0, atol=1e-12)
    assert xp.allclose(got._data, reference_matvec(A, v, StencilVector(W))._data, rtol=0.0, atol=1e-12)
    assert np.allclose(xp.to_numpy(A.vdot(v).toarray()), dense @ x, rtol=0.0, atol=1e-12)

    AT = A.transpose()
    assert AT.pads == A.pads
    assert np.array_equal(xp.to_numpy(AT.toarray()), dense.T)
    # transpose into an existing matrix gives the same
    out = StencilMatrix(W, V, pads=matrix_pads)
    A.transpose(out=out)
    assert np.array_equal(xp.to_numpy(out.toarray()), dense.T)


# ===============================================================================
def test_kernels_reject_spaces_with_shifts():
    """The stencil kernels assume spaces without shifts (shifts == 1); a matrix
    between spaces with shifts raises in dot and transpose instead of giving
    wrong results."""
    V = make_space((8, 9), (2, 2), float, shifts=(2, 1))
    A = StencilMatrix(V, V)
    with pytest.raises(NotImplementedError, match='without shifts'):
        A.dot(StencilVector(V))
    with pytest.raises(NotImplementedError, match='without shifts'):
        A.transpose()


# ===============================================================================
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, '-v']))
