"""Arguments of the stencil kernels for the parity tests.

Each builder returns the arguments of its kernel on the active backend, as ``StencilMatrix`` and
``StencilVectorSpace`` pass them, from random data that is the same on both backends (seeded with the case index).
The parity cases built from them are in :mod:`feectools.linalg.tests.cuda_parity_cases`.
"""
import numpy as np
import cunumpy as xp

from feectools.ddm.cart import DomainDecomposition, CartDecomposition
from feectools.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix

# (npts of the domain, npts of the codomain, pads of the spaces, pads of the matrix, periods): square matrices, and
# rectangular ones whose spaces differ by one point in a direction, which makes `add` zero (or one) there, as for
# derivative operators. A matrix whose pads are None has those of the spaces (2 * p + 1 diagonals), periods None
# means periodic in every direction. The cases after the first three have fewer diagonals than 2 * p + 1 in some
# direction (pads of the matrix smaller than those of the spaces) and/or non-periodic directions: blocks between
# spaces of different degree per direction, and derivative-type (one-sided, offset) stencils.
MATRIX_CASES = {
    1: [((24,), (24,), (2,), None, None), ((23,), (24,), (2,), None, None), ((24,), (23,), (3,), None, None),
        ((24,), (24,), (3,), (1,), (False,)), ((13,), (12,), (2,), (1,), (False,)), ((12,), (13,), (3,), (0,), (False,))],
    2: [((10, 12), (10, 12), (2, 3), None, None), ((11, 10), (12, 10), (2, 2), None, None),
        ((12, 9), (11, 10), (1, 2), None, None),
        ((10, 9), (10, 9), (3, 2), (1, 2), (False, True)), ((9, 8), (8, 8), (2, 3), (2, 1), (False, False))],
    3: [((7, 8, 9), (7, 8, 9), (1, 2, 3), None, None), ((8, 8, 9), (9, 8, 10), (1, 2, 2), None, None),
        ((6, 5, 7), (5, 6, 7), (2, 1, 1), None, None),
        # square, fewer diagonals in two directions (e.g. a mass matrix of a space of lower degree)
        ((6, 7, 8), (6, 7, 8), (2, 3, 2), (1, 1, 2), (True, False, True)),
        # rectangular between non-periodic spaces of different size per direction, all diagonals
        ((8, 7, 9), (7, 7, 8), (2, 2, 3), None, (False, True, False)),
        # derivative-type: rectangular, non-periodic, fewer diagonals in every direction
        ((7, 8, 6), (8, 7, 6), (3, 2, 2), (1, 1, 1), (False, False, True)),
        # one diagonal in two directions (pads 0), the codomain larger in both
        ((7, 6, 5), (8, 6, 6), (1, 2, 1), (0, 2, 0), (False, True, False))],
}

# (npts, pads) of the vector spaces
VECTOR_CASES = {
    1: [((24,), (2,)), ((7,), (1,))],
    2: [((10, 12), (2, 3)), ((5, 9), (1, 1))],
    3: [((7, 8, 9), (1, 2, 3)), ((6, 5, 4), (2, 1, 1))],
}


def make_space(npts, pads, dtype=float, periods=None):
    """A serial StencilVectorSpace, periodic in every direction unless `periods` says otherwise."""
    ndim = len(npts)
    D = DomainDecomposition(list(npts), periods=list(periods) if periods is not None else [True] * ndim)
    global_starts, global_ends = [], []
    for axis in range(ndim):
        ee = D.global_element_ends[axis].copy()
        ee[-1] = npts[axis] - 1
        global_ends.append(ee)
        global_starts.append(xp.array([0] + (ee[:-1] + 1).tolist()))
    C = CartDecomposition(D, list(npts), global_starts, global_ends, pads=list(pads), shifts=[1] * ndim)
    return StencilVectorSpace(C, dtype=dtype)


def random_like(array, rng):
    """Random data of the shape and dtype of `array`, on the active backend."""
    shape = tuple(int(n) for n in array.shape)
    return xp.asarray(rng.random(shape).astype(array.dtype))


def stencil_matrix(npts_domain, npts_codomain, pads, matrix_pads, periods, rng):
    """A StencilMatrix with random entries (spurious entries removed), its domain and its codomain."""
    V = make_space(npts_domain, pads, periods=periods)
    W = V if npts_domain == npts_codomain else make_space(npts_codomain, pads, periods=periods)
    A = StencilMatrix(V, W, pads=matrix_pads)
    A._data[...] = random_like(A._data, rng)
    A.remove_spurious_entries()
    return A, V, W


def stencil_vector(V, rng):
    """A StencilVector of `V` with random entries and up-to-date ghost regions."""
    v = StencilVector(V)
    v._data[...] = random_like(v._data, rng)
    v.update_ghost_regions()
    return v


def dot_arguments(npts_domain, npts_codomain, pads, matrix_pads, periods, seed):
    """The arguments of ``stencil_dot_<n>d`` as ``StencilMatrix.dot`` passes them."""
    rng = np.random.default_rng(seed)
    A, V, W = stencil_matrix(npts_domain, npts_codomain, pads, matrix_pads, periods, rng)
    v = stencil_vector(V, rng)
    out = StencilVector(W)
    return (A._data, v._data, out._data, *A._args.values())


def transpose_arguments(npts_domain, npts_codomain, pads, matrix_pads, periods, seed):
    """The arguments of ``stencil_transpose_<n>d`` as ``StencilMatrix.transpose`` passes them."""
    rng = np.random.default_rng(seed)
    A, V, W = stencil_matrix(npts_domain, npts_codomain, pads, matrix_pads, periods, rng)
    A.update_ghost_regions()
    out = StencilMatrix(W, V, pads=matrix_pads)
    return (A._data, out._data, *A._transpose_args.values())


def inner_arguments(npts, pads, seed):
    """The arguments of ``stencil_inner_<n>d`` as ``StencilVectorSpace.inner`` passes them."""
    rng = np.random.default_rng(seed)
    V = make_space(npts, pads)
    x = stencil_vector(V, rng)
    y = stencil_vector(V, rng)
    res = xp.zeros(1)
    return (x._data, y._data, *V._inner_consts, res)


def axpy_arguments(npts, pads, seed):
    """The arguments of ``stencil_axpy_<n>d`` as ``StencilVectorSpace.axpy`` passes them."""
    rng = np.random.default_rng(seed)
    V = make_space(npts, pads)
    x = stencil_vector(V, rng)
    y = stencil_vector(V, rng)
    return (float(rng.random()) - 0.5, x._data, y._data)
