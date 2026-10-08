from functools import reduce

import pytest
import cunumpy as xp
from scipy.sparse import kron

from feectools.ddm.cart       import DomainDecomposition, CartDecomposition
from feectools.linalg.stencil import StencilVectorSpace
from feectools.linalg.stencil import StencilVector
from feectools.linalg.stencil import StencilMatrix
from feectools.linalg.kron    import KroneckerStencilMatrix
#===============================================================================
def compute_global_starts_ends(domain_decomposition, npts):
    ndims         = len(npts)
    global_starts = [None]*ndims
    global_ends   = [None]*ndims

    for axis in range(ndims):
        ee = domain_decomposition.global_element_ends  [axis]

        global_ends  [axis]     = ee.copy()
        global_ends  [axis][-1] = npts[axis]-1
        global_starts[axis]     = xp.array([0] + (global_ends[axis][:-1]+1).tolist())

    return tuple(global_starts), tuple(global_ends)

#==============================================================================
@pytest.mark.parametrize('dtype', [float])
@pytest.mark.parametrize('npts', [(5, 7, 8)])
@pytest.mark.parametrize('pads', [(2, 3, 5)])
@pytest.mark.parametrize('periodic', [(True, False, False)])

def test_KroneckerStencilMatrix(dtype, npts, pads, periodic):

    # Extract input parameters
    n1, n2, n3 = npts
    p1, p2, p3 = pads
    P1, P2, P3 = periodic

    # Define data type with a factor
    if dtype==complex:
        factor=1j
    else:
        factor=1

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1, n3-1], periods=[P1,P2,P3])

    # Partition the points
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2,p3], shifts=[1,1,1])

    # 3D vector space and element
    W = StencilVectorSpace( cart, dtype=dtype)
    w = StencilVector(W)

    # 1D vector space

    D1 = DomainDecomposition([n1-1], periods=[P1])
    D2 = DomainDecomposition([n2-1], periods=[P2])
    D3 = DomainDecomposition([n3-1], periods=[P3])

    # Partition the points
    global_starts1, global_ends1 = compute_global_starts_ends(D1, [n1])
    global_starts2, global_ends2 = compute_global_starts_ends(D2, [n2])
    global_starts3, global_ends3 = compute_global_starts_ends(D3, [n3])

    cart1 = CartDecomposition(D1, [n1], global_starts1, global_ends1, pads=[p1], shifts=[1])
    cart2 = CartDecomposition(D2, [n2], global_starts2, global_ends2, pads=[p2], shifts=[1])
    cart3 = CartDecomposition(D3, [n3], global_starts3, global_ends3, pads=[p3], shifts=[1])

    V1 = StencilVectorSpace( cart1, dtype=dtype )
    V2 = StencilVectorSpace( cart2, dtype=dtype )
    V3 = StencilVectorSpace( cart3, dtype=dtype )

    # 1D stencil matrices
    M1 = StencilMatrix(V1, V1)
    M2 = StencilMatrix(V2, V2)
    M3 = StencilMatrix(V3, V3)

    # ...
    # Fill in stencil matrix values
    for k1 in range(-p1, p1+1):
        M1[:, k1] = 10 + k1*factor

    for k2 in range(-p2, p2+1):
        M2[:, k2] = 20 + k2*factor

    for k3 in range(-p3, p3+1):
        M3[:, k3] = 40 + k3*factor

    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()
    # ...

    # Fill in vector values
    w[:, :, :] = factor

    # Create Kronecker matrix 
    M = KroneckerStencilMatrix(W, W, M1, M2, M3)

    # Scipy sparse matrices used for comparison
    M1_sp = M1.tosparse().tocsr()
    M2_sp = M2.tosparse().tocsr()
    M3_sp = M3.tosparse().tocsr()
    M_sp  = reduce(kron, (M1_sp, M2_sp, M3_sp)).tocsr()

    # Test transpose
    assert (M_sp.T - M.T.tosparse().tocsr()).count_nonzero() == 0

    # Test dot product
    expected = M_sp.dot(xp.to_numpy(w.toarray()))
    assert xp.array_equal(xp.asarray(expected), M.dot(w).toarray())

#===============================================================================
# Kronecker matrices with factors on groups of axes, products and solvers
#===============================================================================
import numpy as np
from scipy.sparse import csr_matrix

from maybempi import MPI
from feectools.linalg.basic          import ComposedLinearOperator
from feectools.linalg.direct_solvers import SparseSolver
from feectools.linalg.kron           import KroneckerLinearSolver, kronecker_solve
from feectools.linalg.kron           import ComposedKroneckerStencilMatrix


def make_space(comm, npts, pads, periods, mpi_dims_mask=None):
    """Distributed StencilVectorSpace (serial if comm is None)."""
    D = DomainDecomposition([n-1 for n in npts], periods=periods, comm=comm,
                            mpi_dims_mask=mpi_dims_mask)
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=pads, shifts=[1]*len(npts))
    return StencilVectorSpace(cart)


def make_factor(W, grp, mpads, seed):
    """
    Process-local factor on the axes grp of W (rows owned by this process),
    with band mpads and random entries; also returns the global dense matrix.
    """
    npts    = [W.npts[a] for a in grp]
    periods = [W.periods[a] for a in grp]
    spads   = list(mpads)  # factor space pads may be smaller than the ones of W
    starts  = [W.starts[a] for a in grp]
    ends    = [W.ends[a] for a in grp]

    D    = DomainDecomposition([n-1 for n in npts], periods=periods)
    cart = CartDecomposition(D, npts, [np.array([s]) for s in starts], [np.array([e]) for e in ends],
                             pads=spads, shifts=[1]*len(grp))
    V    = StencilVectorSpace(cart)
    M    = StencilMatrix(V, V, pads=tuple(mpads))

    # same random values on all processes; diagonally dominant
    rng    = np.random.default_rng(seed)
    ndiags = tuple(2*p+1 for p in mpads)
    vals   = rng.random(tuple(npts) + ndiags)
    vals[(Ellipsis, *mpads)] += 2 * np.prod(ndiags)

    N     = int(np.prod(npts))
    dense = np.zeros((N, N))
    for i in np.ndindex(*npts):
        for kk in np.ndindex(*ndiags):
            col = []
            for i_d, k_d, p, n, P in zip(i, kk, mpads, npts, periods):
                c = i_d + k_d - p
                if P:
                    c %= n
                elif not 0 <= c < n:
                    break
                col.append(c)
            else:
                v = vals[(*i, *kk)]
                dense[np.ravel_multi_index(i, npts), np.ravel_multi_index(col, npts)] += v
                if all(s <= i_d <= e for i_d, s, e in zip(i, starts, ends)):
                    M._data[(*(i_d - s + sp for i_d, s, sp in zip(i, starts, spads)), *kk)] = v
    return M, dense


def local_slices(W):
    return tuple(slice(s, e+1) for s, e in zip(W.starts, W.ends))


def local_rows(W):
    """Flattened (C order) global indices of the rows owned by this process."""
    grids = np.meshgrid(*[np.arange(s, e+1) for s, e in zip(W.starts, W.ends)], indexing='ij')
    return np.ravel_multi_index(tuple(g.ravel() for g in grids), tuple(W.npts))


def random_vector(W, seed):
    wglob = np.random.default_rng(seed).random(tuple(W.npts))
    w = StencilVector(W)
    w[local_slices(W)] = xp.asarray(wglob[local_slices(W)])
    return w, wglob.ravel()


def assert_local_equal(W, y, expected):
    y_loc = xp.to_numpy(y[local_slices(W)])
    assert np.allclose(y_loc, expected.reshape(tuple(W.npts))[local_slices(W)], rtol=1e-12, atol=1e-12)


def make_kron(W, factor_ndims, mpads, seed):
    axes, d = [], 0
    for n in factor_ndims:
        axes.append(tuple(range(d, d + n)))
        d += n
    mats, dense = [], []
    for k, grp in enumerate(axes):
        M, Md = make_factor(W, grp, [mpads[a] for a in grp], seed + k)
        mats.append(M)
        dense.append(Md)
    from functools import reduce as _reduce
    return KroneckerStencilMatrix(W, W, *mats), _reduce(np.kron, dense), mats, dense


NPTS    = [6, 7, 8]
PADS    = [2, 2, 3]
PERIODS = [True, False, True]
GROUPS  = [(1, 1, 1), (2, 1), (1, 2), (3,)]


def check_grouped_kron(comm, factor_ndims):
    W = make_space(comm, NPTS, PADS, PERIODS)
    M, Md, _, _ = make_kron(W, factor_ndims, [1, 2, 2], seed=10)
    w, wglob = random_vector(W, seed=1)

    assert M.ndim == 3
    assert len(M.axes) == len(factor_ndims)

    # dot
    assert_local_equal(W, M.dot(w), Md @ wglob)

    # tosparse (local rows)
    rows = local_rows(W)
    assert np.allclose(M.tosparse().tocsr()[rows].toarray(), Md[rows])

    # __getitem__: row i, diagonal offset k
    i = tuple(W.starts)
    k = (1, -1, 2)
    col = [(ii + kk) % n for ii, kk, n in zip(i, k, NPTS)]
    assert np.isclose(float(M[(*i, *k)]),
                      Md[np.ravel_multi_index(i, NPTS), np.ravel_multi_index(col, NPTS)])

    # scaling
    assert_local_equal(W, (M * 3.).dot(w), 3. * Md @ wglob)
    assert_local_equal(W, (-M).dot(w), -Md @ wglob)

    # tostencil, transpose (process-local factors are only complete in serial)
    if comm is None:
        assert np.allclose(M.tostencil().toarray(), Md)
        assert np.allclose(M.T.toarray(), Md.T)


def check_matmul(comm, factor_ndims):
    W = make_space(comm, NPTS, PADS, PERIODS)
    A, Ad, _, _ = make_kron(W, factor_ndims, [1, 2, 2], seed=20)
    B, Bd, _, _ = make_kron(W, factor_ndims, [2, 1, 3], seed=30)
    w, wglob = random_vector(W, seed=2)
    rows = local_rows(W)

    C = A @ B
    assert type(C) is ComposedKroneckerStencilMatrix
    assert isinstance(C, ComposedLinearOperator)
    assert C.domain is B.domain and C.codomain is A.codomain
    assert C.axes == A.axes and C.ndim == 3
    assert C.multiplicands == (A, B)
    for Ck, Ak, Bk in zip(C.mats, A.mats, B.mats):
        assert Ck.pads == tuple(min(pa + pb, n//2 if P else n-1) for pa, pb, n, P
                                in zip(Ak.pads, Bk.pads, Ak.domain.npts, Ak.domain.periods))

    # exact factors and dot through the operands
    assert np.allclose(C.tosparse().tocsr()[rows].toarray(), (Ad @ Bd)[rows])
    assert_local_equal(W, C.dot(w), Ad @ Bd @ wglob)
    out = StencilVector(W)
    assert C.dot(w, out=out) is out
    assert_local_equal(W, out, Ad @ Bd @ wglob)

    # chains are flattened, on both sides
    for D in (C @ A, A @ (B @ A), (A @ B) @ A):
        assert type(D) is ComposedKroneckerStencilMatrix
        assert D.multiplicands == (A, B, A)
        assert np.allclose(D.tosparse().tocsr()[rows].toarray(), (Ad @ Bd @ Ad)[rows])
        assert_local_equal(W, D.dot(w), Ad @ Bd @ Ad @ wglob)

    # scaling, copy (keep the type, leave C unchanged)
    for C2, f in ((C * 2., 2.), (2. * C, 2.), (-C, -1.), (C.copy(), 1.)):
        assert type(C2) is ComposedKroneckerStencilMatrix
        assert_local_equal(W, C2.dot(w), f * Ad @ Bd @ wglob)
        assert np.allclose(C2.tosparse().tocsr()[rows].toarray(), f * (Ad @ Bd)[rows])
    C2 = C.copy()
    C2 *= 0.5
    assert_local_equal(W, C2.dot(w), 0.5 * Ad @ Bd @ wglob)
    assert_local_equal(W, C.dot(w), Ad @ Bd @ wglob)
    assert np.allclose(C.tosparse().tocsr()[rows].toarray(), (Ad @ Bd)[rows])

    # the factors of the product do not fit into the ghost regions of the domain
    with pytest.raises(ValueError):
        KroneckerStencilMatrix(W, W, *C.mats)

    # transpose (process-local factors are only complete in serial)
    if comm is None:
        assert type(C.T) is ComposedKroneckerStencilMatrix
        assert np.allclose(C.T.toarray(), (Ad @ Bd).T)
        assert_local_equal(W, C.T.dot(w), (Ad @ Bd).T @ wglob)

    # other operands fall back to LinearOperator.__matmul__
    E, Ed, _, _ = make_kron(W, (1, 1, 1) if factor_ndims != (1, 1, 1) else (2, 1), [1, 1, 1], seed=40)
    for AE in (A @ E, C @ E):
        assert type(AE) is ComposedLinearOperator
    assert_local_equal(W, (A @ E).dot(w), Ad @ Ed @ wglob)
    assert_local_equal(W, A @ w, Ad @ wglob)


def check_solver(comm, factor_ndims, mpi_dims_mask=None):
    W = make_space(comm, NPTS, PADS, PERIODS, mpi_dims_mask=mpi_dims_mask)
    M, Md, _, dense = make_kron(W, factor_ndims, [1, 2, 2], seed=50)
    solvers = [SparseSolver(csr_matrix(D)) for D in dense]
    b, bglob = random_vector(W, seed=3)

    S = KroneckerLinearSolver(W, W, solvers, factor_ndims=factor_ndims)
    assert S.factor_ndims == factor_ndims
    assert_local_equal(W, S.solve(b), np.linalg.solve(Md, bglob))
    assert_local_equal(W, kronecker_solve(solvers, b, factor_ndims=factor_ndims), np.linalg.solve(Md, bglob))
    assert_local_equal(W, S.T.solve(b), np.linalg.solve(Md.T, bglob))

    # solver for a product of Kronecker matrices (the factors hold all rows only in serial)
    C = M @ M
    assert type(C) is ComposedKroneckerStencilMatrix
    if comm is None:
        S2 = KroneckerLinearSolver(W, W, [SparseSolver(Ck.tosparse().tocsr()) for Ck in C.mats],
                                   factor_ndims=factor_ndims)
        assert_local_equal(W, S2.solve(b), np.linalg.solve(Md @ Md, bglob))


#===============================================================================
@pytest.mark.parametrize('factor_ndims', GROUPS)
def test_grouped_kron_ser(factor_ndims):
    check_grouped_kron(None, factor_ndims)

@pytest.mark.parametrize('factor_ndims', GROUPS)
def test_kron_matmul_ser(factor_ndims):
    check_matmul(None, factor_ndims)

@pytest.mark.parametrize('factor_ndims', GROUPS)
def test_kron_solver_groups_ser(factor_ndims):
    check_solver(None, factor_ndims)

def test_kron_constructor_checks():
    W = make_space(None, NPTS, PADS, PERIODS)
    M, _, mats, _ = make_kron(W, (2, 1), [1, 1, 1], seed=0)
    with pytest.raises(AssertionError):
        KroneckerStencilMatrix(W, W, mats[0])                      # too few axes
    with pytest.raises(AssertionError):
        KroneckerStencilMatrix(W, W, mats[1], mats[0])             # npts do not match
    with pytest.raises(AssertionError):
        KroneckerStencilMatrix(W, W, *mats, mats[1])               # too many axes
    with pytest.raises(AssertionError):
        KroneckerLinearSolver(W, W, [SparseSolver(csr_matrix(np.eye(2)))] * 2, factor_ndims=(1, 1))

@pytest.mark.mpi
@pytest.mark.parametrize('factor_ndims', GROUPS)
def test_grouped_kron_par(factor_ndims):
    check_grouped_kron(MPI.COMM_WORLD, factor_ndims)

@pytest.mark.mpi
@pytest.mark.parametrize('factor_ndims', GROUPS)
def test_kron_matmul_par(factor_ndims):
    check_matmul(MPI.COMM_WORLD, factor_ndims)

@pytest.mark.mpi
@pytest.mark.parametrize('factor_ndims', [(1, 1, 1), (2, 1)])
def test_kron_solver_groups_par(factor_ndims):
    # grouped axes are not distributed: only the last axis is
    check_solver(MPI.COMM_WORLD, factor_ndims, mpi_dims_mask=[False, False, True])

@pytest.mark.mpi
def test_kron_solver_distributed_group_par():
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 1:
        pytest.skip('needs more than one process')
    W = make_space(comm, NPTS, PADS, PERIODS, mpi_dims_mask=[True, False, False])
    _, _, _, dense = make_kron(W, (2, 1), [1, 1, 1], seed=0)
    with pytest.raises(NotImplementedError):
        KroneckerLinearSolver(W, W, [SparseSolver(csr_matrix(D)) for D in dense], factor_ndims=(2, 1))


def test_multiplicants_deprecated():
    W = make_space(None, NPTS, PADS, PERIODS)
    A = make_kron(W, (1, 1, 1), [1, 1, 1], seed=0)[0]
    C = ComposedLinearOperator(W, W, A, A)
    with pytest.warns(DeprecationWarning):
        assert C.multiplicants == C.multiplicands == (A, A)
