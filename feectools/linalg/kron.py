#coding = utf-8
from __future__ import annotations

from collections.abc import Sequence
from functools import reduce

import numpy as np
import cunumpy as xp
from cunumpy.mpi import synchronize_for_mpi
from scipy.sparse import kron
from scipy.sparse import coo_matrix

from feectools.ddm.cart       import CartDecomposition
from feectools.linalg.basic   import ComposedLinearOperator, LinearOperator, LinearSolver
from feectools.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix

__all__ = ('KroneckerStencilMatrix',
           'ComposedKroneckerStencilMatrix',
           'KroneckerLinearSolver',
           'KroneckerDenseMatrix',
           'kronecker_solve')

#==============================================================================
class KroneckerStencilMatrix(LinearOperator):
    r"""
    Kronecker product $M = A_1 \otimes A_2 \otimes \dots \otimes A_m$ of stencil matrices.

    Each factor $A_k$ is a StencilMatrix acting on a group of consecutive
    axes of the domain and codomain; its number of axes is ``A_k.domain.ndim``.
    The axes of all factors must add up to ``V.ndim``. The usual case is one
    1d factor per axis, but other groupings are allowed, e.g. a 2d x 1d
    product on a 3d space::

        M = KroneckerStencilMatrix(V, W, A_xy, A_z)    # M.axes == ((0, 1), (2,))

    The factors are process-local: along its axes, the factor $A_k$ owns the
    same rows (``starts``/``ends``) as the codomain ``W`` on this process,
    but it lives on its own spaces, without a communicator. The pads of the
    factors must not exceed the pads of the domain ``V``, whose ghost regions
    are read by ``dot``.

    The product ``A @ B`` of two Kronecker matrices with the same axis groups
    is a :class:`ComposedKroneckerStencilMatrix`.

    Parameters
    ----------
    V : StencilVectorSpace
        The domain.

    W : StencilVectorSpace
        The codomain.

    *args : StencilMatrix
        Factors of the Kronecker product, ordered by axis. For each factor
        ``A_k``, ``A_k.domain.npts`` and ``A_k.codomain.npts`` must equal the
        ``npts`` of ``V`` and ``W`` along the axes of that factor. The pads
        of ``A_k`` must not exceed the pads of ``V``.
    """

    def __init__(self, V: StencilVectorSpace, W: StencilVectorSpace, *args: StencilMatrix):

        assert isinstance(V, StencilVectorSpace)
        assert isinstance(W, StencilVectorSpace)
        assert V.ndim == W.ndim
        assert len(args) > 0, 'A KroneckerStencilMatrix needs at least one factor.'

        # group the axes of V and W by factor
        axes = []
        d = 0
        for A in args:
            assert isinstance(A, StencilMatrix), \
                f'Factors must be of type StencilMatrix, got {type(A)}.'
            n   = A.domain.ndim
            grp = tuple(range(d, d + n))
            assert d + n <= V.ndim, \
                f'The factors have more axes than the domain ({V.ndim}).'
            assert tuple(A.domain.npts) == tuple(V.npts[d:d+n]), \
                f'Domain npts {A.domain.npts} of factor on axes {grp} do not match {V.npts}.'
            assert tuple(A.codomain.npts) == tuple(W.npts[d:d+n]), \
                f'Codomain npts {A.codomain.npts} of factor on axes {grp} do not match {W.npts}.'
            axes.append(grp)
            d += n
        assert d == V.ndim, \
            f'The factors cover {d} axes, but the domain has {V.ndim}.'

        # dot reads the ghost regions of x, so the band must fit into them
        for A, grp in zip(args, axes):
            for p, a in zip(A.pads, grp):
                if p > V.pads[a]:
                    raise ValueError(f'Pads {A.pads} of factor on axes {grp} exceed the domain pads {V.pads}. '
                                     'Products of Kronecker matrices are ComposedKroneckerStencilMatrix (A @ B).')

        self._domain   = V
        self._codomain = W
        self._mats     = tuple(args)
        self._axes     = tuple(axes)

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def domain(self) -> StencilVectorSpace:
        return self._domain

    # ...
    @property
    def codomain(self) -> StencilVectorSpace:
        return self._codomain

    # ...
    @property
    def dtype(self):
        return self.domain.dtype

    # ...
    @property
    def ndim(self) -> int:
        """Number of axes of the domain (not the number of factors, see ``axes``)."""
        return self._domain.ndim

    # ...
    @property
    def mats(self) -> tuple[StencilMatrix, ...]:
        """Factors of the Kronecker product, ordered by axis."""
        return self._mats

    # ...
    @property
    def axes(self) -> tuple[tuple[int, ...], ...]:
        """Axes of the domain/codomain on which each factor acts, e.g. ``((0, 1), (2,))``."""
        return self._axes

    # ...
    @property
    def nbytes(self) -> int:
        """Local (per-MPI-rank) memory footprint of the factor matrices, in bytes."""
        return int(sum(getattr(mat, 'nbytes', 0) for mat in self._mats))

    # ...
    def dot(self, x: StencilVector, out: StencilVector | None = None) -> StencilVector:
        """
        Matrix-vector product ``M @ x``.

        Parameters
        ----------
        x : StencilVector
            Vector in the domain.

        out : StencilVector, optional
            Vector in the codomain, in which the result is stored.

        Returns
        -------
        StencilVector
            The result, in the codomain (``out`` if given).
        """

        assert isinstance(x, StencilVector)
        assert x.space is self.domain

        if out is not None:
            assert isinstance(out, StencilVector)
            assert out.space is self.codomain
        else:
            out = StencilVector(self.codomain)

        # Necessary if vector space is periodic or distributed across processes
        if not x.ghost_regions_in_sync:
            x.update_ghost_regions()

        starts = self._codomain.starts
        ends   = self._codomain.ends
        pads   = self._codomain.pads
        shifts = self._codomain.shifts

        mats   = self.mats
        axes   = self.axes
        nrows  = tuple(e-s+1 for s,e in zip(starts, ends))

        # per axis: band of the factor, row offset in its data array and ghost offset of x
        mpads   = tuple(p for A in mats for p in A.pads)
        row_off = tuple(p*m for A in mats for p,m in zip(A.codomain.pads, A.codomain.shifts))
        x_off   = tuple(p*m for p,m in zip(self._domain.pads, self._domain.shifts))
        pnrows  = tuple(2*p+1 for p in mpads)

        for ii in xp.ndindex(*nrows):
            v = 0.
            xx = tuple(i+p*s for i,p,s in zip(ii, pads, shifts))
            rr = tuple(i+o for i,o in zip(ii, row_off))

            for jj in xp.ndindex(*pnrows):
                i_mats = [mat._data[(*(rr[a] for a in grp), *(jj[a] for a in grp))]
                          for mat,grp in zip(mats, axes)]
                ii_jj = tuple(i+j-p+o for i,j,p,o in zip(ii, jj, mpads, x_off))
                # ``array_api_compat.cupy`` does not accept a Python list in
                # ``prod``; multiplying the scalar factors also avoids a
                # temporary device array in this innermost loop.
                v += x._data[ii_jj] * reduce(lambda a, b: a * b, i_mats, 1)

            out._data[xx] = v

        # IMPORTANT: flag that ghost regions are not up-to-date
        out.ghost_regions_in_sync = False
        return out

    # ...
    def copy(self) -> KroneckerStencilMatrix:
        mats = [m.copy() for m in self.mats]
        return KroneckerStencilMatrix(self.domain, self.codomain, *mats)

    # ...
    def __neg__(self) -> KroneckerStencilMatrix:
        mats = [-self.mats[0], *(m.copy() for m in self.mats[1:])]
        return KroneckerStencilMatrix(self.domain, self.codomain, *mats)

    # ...
    def __mul__(self, a) -> KroneckerStencilMatrix:
        mats = [*(m.copy() for m in self.mats[:-1]), self.mats[-1] * a]
        return KroneckerStencilMatrix(self.domain, self.codomain, *mats)

    # ...
    def __imul__(self, a) -> KroneckerStencilMatrix:
        last  = self._mats[-1]
        last *= a
        return self

    # ...
    def __matmul__(self, B):
        """
        Product ``self @ B``.

        If ``B`` is a KroneckerStencilMatrix or a ComposedKroneckerStencilMatrix
        with the same axis groups, the result is a
        :class:`ComposedKroneckerStencilMatrix`. In all other cases (a different
        operator, other axis groups, or a vector) the call is passed to
        ``LinearOperator.__matmul__``.

        Parameters
        ----------
        B : LinearOperator | Vector
            Right operand. Its codomain must be the domain of ``self``.

        Returns
        -------
        ComposedKroneckerStencilMatrix | LinearOperator | Vector
            The product.
        """
        if _is_kronecker_with_axes(B, self.axes):
            return ComposedKroneckerStencilMatrix(B.domain, self.codomain, self, B)
        return super().__matmul__(B)

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------

    def __getitem__(self, key):
        """
        Entry ``M[i_1, ..., i_d, k_1, ..., k_d]`` for row indices ``i`` and
        diagonal offsets ``k``, i.e. the product of the corresponding entries
        of the factors.
        """
        rows = key[:self.ndim]
        cols = key[self.ndim:]
        elements = [A[(*(rows[a] for a in grp), *(cols[a] for a in grp))]
                    for A,grp in zip(self.mats, self.axes)]
        return reduce(lambda a, b: a * b, elements, 1)

    def tostencil(self) -> StencilMatrix:
        """Convert to a StencilMatrix on the domain and codomain."""

        mats  = self.mats
        ssc   = self.codomain.starts
        eec   = self.codomain.ends
        ssd   = self.domain.starts
        eed   = self.domain.ends
        pads  = [p for A in self.mats for p in A.pads]
        xpads = self.domain.pads

        # Number of rows in matrix (along each dimension)
        nrows       = [ed-s+1 for s,ed in zip(ssd, eed)]
        nrows_extra = [0 if ec<=ed else ec-ed for ec,ed in zip(eec,eed)]

        # create the stencil matrix
        M  = StencilMatrix(self.domain, self.codomain, pads=tuple(pads))

        # row offset of each axis in the data array of its factor
        row_off = [p*m for A in mats for p,m in zip(A.codomain.pads, A.codomain.shifts)]

        mats = [mat._data for mat in mats]

        self._tostencil(M._data, mats, self.axes, nrows, nrows_extra, pads, xpads, row_off)
        return M

    @staticmethod
    def _tostencil(M, mats, axes, nrows, nrows_extra, pads, xpads, row_off):

        ndiags = [2*p + 1 for p in pads]
        diff   = [xp-p for xp,p in zip(xpads, pads)]
        ndim   = len(nrows)

        for xx in xp.ndindex( *nrows ):

            ii = tuple(xp + x for xp, x in zip(xpads, xx) )
            rr = tuple(o + x for o, x in zip(row_off, xx) )

            for kk in xp.ndindex( *ndiags ):

                values        = [mat[(*(rr[a] for a in grp), *(kk[a] for a in grp))]
                                 for mat,grp in zip(mats, axes)]
                M[(*ii, *kk)] = reduce(lambda a, b: a * b, values, 1)

        # handle partly-multiplied rows
        new_nrows = nrows.copy()
        for d,er in enumerate(nrows_extra):

            rows = new_nrows.copy()
            del rows[d]

            for n in range(er):
                for xx in xp.ndindex(*rows):
                    xx = list(xx)
                    xx.insert(d, nrows[d]+n)

                    ii     = tuple(x+xp for x,xp in zip(xx, xpads))
                    rr     = tuple(x+o for x,o in zip(xx, row_off))
                    ee     = [max(x-l+1,0) for x,l in zip(xx, nrows)]
                    jj     = tuple( slice(x+d, x+d+2*p+1-e) for x,p,d,e in zip(xx, pads, diff, ee) )
                    ndiags = [2*p + 1-e for p,e in zip(pads,ee)]
                    kk     = [slice(None,diag) for diag in ndiags]
                    ii_kk  = tuple( list(ii) + kk )

                    for kk in xp.ndindex( *ndiags ):
                        values        = [mat[(*(rr[a] for a in grp), *(kk[a] for a in grp))]
                                         for mat,grp in zip(mats, axes)]
                        M[(*ii, *kk)] = reduce(lambda a, b: a * b, values, 1)
            new_nrows[d] += er

    def tosparse(self):
        """Convert the local rows to a scipy sparse matrix (Kronecker product of the factors' ``tosparse``)."""
        return reduce(kron, (m.tosparse() for m in self.mats))

    def toarray(self):
        """Convert the local rows to a dense array."""
        return self.tosparse().toarray()

    def transpose(self, conjugate: bool = False) -> KroneckerStencilMatrix:
        """Transpose of the matrix (Hermitian transpose if ``conjugate`` is True); the factors are transposed."""
        mats_tr = [Mi.transpose(conjugate=conjugate) for Mi in self.mats]
        return KroneckerStencilMatrix(self.codomain, self.domain, *mats_tr)

#==============================================================================
class ComposedKroneckerStencilMatrix(ComposedLinearOperator):
    r"""
    Product $M = F_1 F_2 \cdots F_n$ of Kronecker matrices with the same axis groups.

    Created by ``A @ B`` of two KroneckerStencilMatrix (or of products of them).
    As for any :class:`ComposedLinearOperator`, ``multiplicands`` are the
    operands $F_1, \dots, F_n$ (chains are flattened, so ``(A @ B) @ C`` has
    the multiplicands ``(A, B, C)``) and ``dot`` applies them from right to left.

    In addition, the product is again a Kronecker product,
    $M = C_1 \otimes \dots \otimes C_m$ with $C_k = F_{1,k} F_{2,k} \cdots F_{n,k}$,
    and ``mats`` holds its exact factors $C_k$, computed at construction. Their
    band is wider than the ghost regions of the domain, so they live on
    process-local spaces with larger pads. They are used by ``tosparse`` and can
    be used to build a :class:`KroneckerLinearSolver` for $M$. ``mats`` is a
    snapshot: changing an operand in place afterwards changes ``dot`` but not
    ``mats``.

    Constructing the product is collective if the operands are distributed:
    each process needs all rows of the factors of the right operand that its
    rows of the left factor couple to, so they are gathered.

    Parameters
    ----------
    domain : StencilVectorSpace
        Domain of the last operand.

    codomain : StencilVectorSpace
        Codomain of the first operand.

    *args : KroneckerStencilMatrix | ComposedKroneckerStencilMatrix
        The operands, with the same axis groups.
    """

    def __init__(self,
                 domain: StencilVectorSpace,
                 codomain: StencilVectorSpace,
                 *args: KroneckerStencilMatrix | ComposedKroneckerStencilMatrix):

        assert len(args) >= 2, 'A ComposedKroneckerStencilMatrix needs at least two operands.'
        axes = args[0].axes
        for a in args:
            assert _is_kronecker_with_axes(a, axes), \
                'All operands must be Kronecker matrices with the same axis groups.'

        super().__init__(domain, codomain, *args)

        # exact factors of the product, multiplied from the right
        mats = list(args[-1].mats)
        for a in reversed(args[:-1]):
            mats = [_multiply_factors(A_k, C_k, a.domain) for A_k, C_k in zip(a.mats, mats)]

        self._mats = tuple(mats)
        self._axes = axes

    @classmethod
    def _from_parts(cls, domain, codomain, multiplicands, mats) -> ComposedKroneckerStencilMatrix:
        """Build from known operands and factors of the product (no multiplication, not collective)."""
        obj = cls.__new__(cls)
        ComposedLinearOperator.__init__(obj, domain, codomain, *multiplicands)
        obj._mats = tuple(mats)
        obj._axes = multiplicands[0].axes
        return obj

    #--------------------------------------
    # Kronecker structure
    #--------------------------------------
    @property
    def mats(self) -> tuple[StencilMatrix, ...]:
        """Exact factors $C_k$ of the product, ordered by axis (process-local, wide band)."""
        return self._mats

    @property
    def axes(self) -> tuple[tuple[int, ...], ...]:
        """Axes of the domain/codomain on which each factor acts, e.g. ``((0, 1), (2,))``."""
        return self._axes

    @property
    def ndim(self) -> int:
        """Number of axes of the domain."""
        return self.domain.ndim

    @property
    def dtype(self):
        return self.domain.dtype

    @property
    def nbytes(self) -> int:
        """Local (per-MPI-rank) memory footprint of the factors of the product and of the operands, in bytes."""
        nbytes = sum(getattr(mat, 'nbytes', 0) for mat in self._mats)
        nbytes += sum(F.nbytes for F in self.multiplicands)
        return int(nbytes)

    #--------------------------------------
    # Operations that keep the type
    #--------------------------------------
    def copy(self) -> ComposedKroneckerStencilMatrix:
        return self._from_parts(self.domain, self.codomain,
                                [F.copy() for F in self.multiplicands],
                                [m.copy() for m in self.mats])

    def __neg__(self) -> ComposedKroneckerStencilMatrix:
        return self * -1

    def __mul__(self, a) -> ComposedKroneckerStencilMatrix:
        multiplicands = [*(F.copy() for F in self.multiplicands[:-1]), self.multiplicands[-1] * a]
        mats = [*(m.copy() for m in self.mats[:-1]), self.mats[-1] * a]
        return self._from_parts(self.domain, self.codomain, multiplicands, mats)

    def __rmul__(self, a) -> ComposedKroneckerStencilMatrix:
        return self * a

    def __matmul__(self, B):
        """
        Product ``self @ B``: a ComposedKroneckerStencilMatrix if ``B`` is a Kronecker
        matrix with the same axis groups, else ``LinearOperator.__matmul__``.
        """
        if _is_kronecker_with_axes(B, self.axes):
            return ComposedKroneckerStencilMatrix(B.domain, self.codomain, self, B)
        return super().__matmul__(B)

    def transpose(self, conjugate: bool = False) -> ComposedKroneckerStencilMatrix:
        """Transpose: the operands are transposed in reverse order, and so are the factors of the product."""
        multiplicands = [F.transpose(conjugate=conjugate) for F in reversed(self.multiplicands)]
        mats = [m.transpose(conjugate=conjugate) for m in self.mats]
        return self._from_parts(self.codomain, self.domain, multiplicands, mats)

    #--------------------------------------
    # Conversion
    #--------------------------------------
    def tosparse(self):
        """Convert the local rows to a scipy sparse matrix (Kronecker product of the factors of the product)."""
        return reduce(kron, (m.tosparse() for m in self.mats))

    def toarray(self):
        """Convert the local rows to a dense array."""
        return self.tosparse().toarray()

#==============================================================================
def _is_kronecker_with_axes(B, axes) -> bool:
    """Whether B is a (composed) Kronecker stencil matrix with the given axis groups."""
    return isinstance(B, (KroneckerStencilMatrix, ComposedKroneckerStencilMatrix)) and B.axes == axes


def _multiply_factors(A: StencilMatrix, B: StencilMatrix, U: StencilVectorSpace) -> StencilMatrix:
    """
    Compute the process-local factor ``C = A @ B`` of a Kronecker product.

    The product is computed in scipy sparse format. Its band is wider than
    the ones of ``A`` and ``B``, so ``C`` is stored on new spaces with
    the decomposition of ``B.domain`` / ``A.codomain`` and larger pads.

    Parameters
    ----------
    A, B : StencilMatrix
        Process-local factors acting on the same axes.

    U : StencilVectorSpace
        Codomain of the Kronecker matrix of ``B``. If it is distributed,
        the rows of ``B`` owned by other processes are gathered over its
        communicator.

    Returns
    -------
    StencilMatrix
        The product, with rows ``A.codomain.starts`` to ``A.codomain.ends``.
    """
    for S in (A.domain, A.codomain, B.domain, B.codomain):
        if any(m != 1 for m in S.shifts):
            raise NotImplementedError('Products of factors with shifts != 1 are not supported.')
    assert tuple(A.domain.npts) == tuple(B.codomain.npts)
    assert tuple(A.codomain.periods) == tuple(B.domain.periods)

    # all rows of B that rows of A on this process can couple to
    B_sp = B.tosparse().tocoo()
    if U.parallel:
        parts = U.cart.comm.allgather((B_sp.row, B_sp.col, B_sp.data))
        rows  = np.concatenate([p[0] for p in parts]).astype(np.int64)
        cols  = np.concatenate([p[1] for p in parts]).astype(np.int64)
        data  = np.concatenate([p[2] for p in parts])
        # processes with the same rows along these axes send them twice; keep one copy
        _, idx = np.unique(rows * B_sp.shape[1] + cols, return_index=True)
        B_sp = coo_matrix((data[idx], (rows[idx], cols[idx])), shape=B_sp.shape)

    C_sp = (A.tosparse().tocsr() @ B_sp.tocsr()).tocoo()

    # band of C: diagonal offset of each entry along each axis
    cod, dom = A.codomain, B.domain
    periods  = cod.periods
    rr = np.unravel_index(C_sp.row, tuple(cod.npts))
    cc = np.unravel_index(C_sp.col, tuple(dom.npts))
    kk = []
    pads = []
    for d, (pA, pB, n, P) in enumerate(zip(A.pads, B.pads, dom.npts, periods)):
        k = cc[d] - rr[d]
        if P:
            k = (k + n//2) % n - n//2
            pads.append(min(pA + pB, n//2))
        else:
            pads.append(min(pA + pB, n - 1))
        kk.append(k)
        assert k.size == 0 or np.abs(k).max() <= pads[-1]

    # new spaces with the same decomposition and wider pads
    def widen(S):
        cart = CartDecomposition(S.cart.domain_decomposition, S.npts,
                                 S.cart.global_starts, S.cart.global_ends,
                                 pads=pads, shifts=list(S.shifts))
        return StencilVectorSpace(cart, dtype=C_sp.dtype)

    C = StencilMatrix(widen(dom), widen(cod))

    index = (*(xp.asarray(r - s + p) for r,s,p in zip(rr, cod.starts, pads)),
             *(xp.asarray(k + p) for k,p in zip(kk, pads)))
    C._data[index] = xp.asarray(C_sp.data)

    return C

#==============================================================================
class KroneckerDenseMatrix(LinearOperator):
    """
    Kronecker product of 1D dense matrices.

    Parameters
    ----------
    V : StencilVectorSpace
        The domain.

    W : StencilVectorSpace
        The codomain.

    args : list of ndarray
        Factors of the Kronecker product (one for each dimension).

    """

    def __init__(self, V, W, *args , with_pads=False):

        assert isinstance(V, StencilVectorSpace)
        assert isinstance(W, StencilVectorSpace)
        assert V.pads == W.pads

        for i,A in enumerate(args):
            assert isinstance(A, xp.ndarray)
            if with_pads:
                assert A.shape[1] == V.npts[i] + 2*V.pads[i]
            else:
                assert A.shape[1] == V.npts[i]

        if not with_pads:
            args = [xp.pad(a,p) for a,p in zip(args, W.pads)]

        self._domain   = V
        self._codomain = W
        self._mats     = list(args)
        self._ndim     = len(args)

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def domain(self):
        return self._domain

    # ...
    @property
    def codomain(self):
        return self._codomain

    # ...
    @property
    def dtype(self):
        return self.domain.dtype

    # ...
    @property
    def ndim(self):
        return self._ndim

    # ...
    @property
    def mats(self):
        return self._mats

    # ...
    def dot(self, x, out=None):

        dot = xp.dot

        assert isinstance(x, StencilVector)
        assert x.space is self.domain

        # Necessary if vector space is periodic or distributed across processes
        if not x.ghost_regions_in_sync:
            x.update_ghost_regions()

        if out is not None:
            assert isinstance(out, StencilVector)
            assert out.space is self.codomain
        else:
            out = StencilVector(self.codomain)

        d_starts = self._domain.starts
        d_ends   = self._domain.ends
        c_starts = self._codomain.starts
        c_ends   = self._codomain.ends
        pads     = self._codomain.pads
        mats     = self.mats

        nrows  = tuple(e-s+1 for s,e in zip(c_starts, c_ends))
        ncols  = tuple(e-s+1+2*p for s,e,p in zip(d_starts, d_ends, pads))
        kk     = tuple(slice(s, s+nc) for nc,s in zip(ncols, d_starts))
        x_data   = x._data.ravel()
        out_data = out._data

        for xx in xp.ndindex(*nrows):
            ii     = tuple(x+p for x,p in zip(xx,pads))
            i_mats = [mat[i+s, k] for i,s,k,mat in zip(ii, c_starts, kk, mats)]
            out_data[ii] = xp.dot(x_data, xp.outer(*i_mats).ravel())

        # IMPORTANT: flag that ghost regions are not up-to-date
        out.ghost_regions_in_sync = False
        return out

    # ...
    def copy(self):
        mats = [m.copy() for m in self.mats]
        return KroneckerDenseMatrix(self.domain, self.codomain, *mats, with_pads=True)

    # ...
    def __neg__(self):
        mats = [-self.mats[0], *(m.copy() for m in self.mats[1:])]
        return KroneckerDenseMatrix(self.domain, self.codomain, *mats, with_pads=True)

    # ...
    def __mul__(self, a):
        mats = [*(m.copy() for m in self.mats[:-1]), self.mats[-1] * a]
        return KroneckerDenseMatrix(self.domain, self.codomain, *mats, with_pads=True)

    # ...
    def __rmul__(self, a):
        mats = [a * self.mats[0], *(m.copy() for m in self.mats[1:])]
        return KroneckerDenseMatrix(self.domain, self.codomain, *mats, with_pads=True)

    # ...
    def __imul__(self, a):
        self.mats[-1] *= a
        return self

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------

    def tosparse(self, **kwargs):
        return coo_matrix(reduce(kron, (m[p:-p,p:-p] for m,p in zip(self.mats, self.domain.pads))))

    def toarray(self):
        return reduce(kron, (m[p:-p,p:-p] for m,p in zip(self.mats, self.domain.pads)))

    def transpose(self, conjugate=False):
        mats = [Mi.conj() for Mi in self.mats] if conjugate else self.mats
        mats_tr = [Mi.T for Mi in mats]
        return KroneckerDenseMatrix(self.codomain, self.domain, *mats_tr, with_pads=True)

    def exchange_assembly_data( self ):
        pass

    def set_backend(self, backend, precompiled=False):
        pass
#==============================================================================
class KroneckerLinearSolver(LinearOperator):
    r"""
    Solver for $A x = b$, where $A = A_1 \otimes A_2 \otimes \dots \otimes A_m$
    is a Kronecker product given by one solver per factor.

    Each factor acts on a group of consecutive axes of the space; by default
    every factor is 1d (one solver per axis). For other groupings, pass
    ``factor_ndims``, e.g. ``factor_ndims=(2, 1)`` for a 2d x 1d product on a
    3d space. A solver of a factor with several axes receives the vectors
    flattened in C order over these axes (as in ``StencilMatrix.tosparse``).

    Factors are solved in parallel (with MPI_Alltoallv) along a distributed
    axis. This is only implemented for 1d factors: a factor with several
    axes must not be distributed across processes along any of its axes.

    Parameters
    ----------
    V : StencilVectorSpace
        The space b will live in; i.e. which gives us information about
        the distribution of the right-hand side b.

    W : StencilVectorSpace
        The space x will live in; i.e. which gives us information about
        the distribution of the unknown vector x.

    solvers : sequence of LinearSolver
        Solvers for the factors of A, ordered by axis.

    factor_ndims : sequence of int, optional
        Number of axes of each factor. Defaults to 1 for every factor.

    Attributes
    ----------
    domain : StencilVectorSpace
        The space of the rhs vector b.

    codomain : StencilVectorSpace
        The space of the unknown vector x.
    """
    def __init__(self,
                 V: StencilVectorSpace,
                 W: StencilVectorSpace,
                 solvers: Sequence[LinearSolver],
                 factor_ndims: Sequence[int] | None = None):
        assert isinstance(V, StencilVectorSpace)
        assert isinstance(W, StencilVectorSpace)
        assert hasattr( solvers, '__iter__' )
        for solver in solvers:
            assert isinstance(solver, LinearSolver)

        if factor_ndims is None:
            factor_ndims = (1,) * len(solvers)
        factor_ndims = tuple(int(n) for n in factor_ndims)
        assert len(factor_ndims) == len(solvers), \
            f'Got {len(solvers)} solvers but {len(factor_ndims)} factor_ndims.'
        assert all(n >= 1 for n in factor_ndims)

        assert V.ndim == sum(factor_ndims)
        assert W.ndim == sum(factor_ndims)
        assert V.npts == W.npts

        # axes of each factor
        axes = []
        d = 0
        for n in factor_ndims:
            axes.append(tuple(range(d, d + n)))
            d += n

        # general arguments
        self._domain = V
        self._codomain = W
        self._solvers = solvers
        self._factor_ndims = factor_ndims
        self._axes = tuple(axes)
        self._parallel = self._domain.parallel
        self._dtype = self._codomain._dtype
        if self._parallel:
            self._mpi_type = self._domain._mpi_type
        else:
            self._mpi_type = None
        # number of factors (= number of solve passes)
        self._ndim = len(solvers)

        # compute and setup solver arguments
        self._setup_solvers()

        # compute reordering permutations between the steps
        self._setup_permutations()

        # for now: allocate temporary arrays here (can be removed later)
        self._temp1, self._temp2 = self._allocate_temps()
    
    def _setup_solvers(self):
        """
        Computes the distribution of elements and sets up the solvers
        (which potentially utilize MPI).
        """
        # slice sizes -- domain-decomposition bookkeeping (sizes/starts/ends),
        # always host-resident regardless of the active array backend, matching
        # self._domain.starts/ends (which are already plain host ints, see
        # CartDecomposition in feectools.ddm.cart) and the MPI sizes/displacements
        # computed from them below in KroneckerSolverParallelPass.
        starts = np.array(self._domain.starts)
        ends = np.array(self._domain.ends) + 1
        self._slice = tuple([slice(s, e) for s,e in zip(starts, ends)])

        # local and global sizes, per factor (axes of a factor are flattened into one)
        npts = self._domain.npts
        nlocals_axis = ends - starts
        nglobals = [int(np.prod([npts[a] for a in grp])) for grp in self._axes]
        nlocals = np.array([np.prod(nlocals_axis[list(grp)]) for grp in self._axes], dtype=int)
        self._localsize = np.prod(nlocals)
        mglobals = self._localsize // nlocals
        self._nlocals = nlocals

        # solver passes (and mlocal size)
        solver_passes = [None] * self._ndim

        tempsize = self._localsize
        self._allserial = True
        for i, grp in enumerate(self._axes):
            # decide for each direction individually, if we should
            # use a serial or a parallel/distributed solver
            # useful e.g. if we have little data in some directions
            # (and thus no data distributed there)

            distributed = self._parallel and any(self._domain.cart.subcomm[a].size > 1 for a in grp)

            if not distributed:
                # serial solve
                solver_passes[i] = KroneckerLinearSolver.KroneckerSolverSerialPass(
                        self._solvers[i], nglobals[i], mglobals[i])
            elif len(grp) > 1:
                raise NotImplementedError(f'The factor on axes {grp} is distributed across processes; '
                                          'parallel solves are only implemented for 1d factors.')
            else:
                # for the parallel case, use Alltoallv
                solver_passes[i] = KroneckerLinearSolver.KroneckerSolverParallelPass(
                        self._solvers[i], self._domain._mpi_type, grp[0],
                        self._domain.cart, mglobals[i], nglobals[i], nlocals[i], self._localsize)

                # we have a parallel solve pass now, so we are not completely local any more
                self._allserial = False
            
            # update memory requirements
            tempsize = max(tempsize, solver_passes[i].required_memory())
        
        # we want to start with the last dimension
        self._solver_passes = list(reversed(solver_passes))
        self._tempsize = tempsize

    def _setup_permutations(self):
        """
        Creates the permutations and matrix shapes which occur during reordering
        the data for the Kronecker solve operations.
        """

        # we use a single permutation for all steps
        # it is: (n, 1, 2, ..., n-1)
        # host bookkeeping (ndim-sized), like self._shapes/self._nlocals it
        # indexes -- see the note on _setup_solvers above.
        self._perm = np.arange(self._ndim)
        self._perm[1:] = self._perm[:-1]
        self._perm[0] = self._ndim - 1

        # side note: we tried out other permutations:
        # swapping one dimension with the last one each time showed a bad performance...

        # re-order the shapes based on the permutations
        self._shapes = [None] * self._ndim
        self._shapes[0] = self._nlocals
        for i in range(1, self._ndim):
            self._shapes[i] = self._shapes[i-1][self._perm]
    
    def _allocate_temps(self):
        """
        Allocates all temporary data needed for the solve operation.
        """
        temp1 = xp.empty((int(self._tempsize),), dtype=self._dtype)
        if self._ndim <= 1 and self._allserial:
            # if ndim==1 and we have no parallelism,
            # we can avoid allocating a second temp array
            temp2 = None
        else:
            temp2 = xp.empty((int(self._tempsize),), dtype=self._dtype)
        return temp1, temp2

    @property
    def domain(self) -> StencilVectorSpace:
        return self._domain
    
    @property
    def codomain(self) -> StencilVectorSpace:
        return self._codomain
    
    @property
    def dtype(self):
        return None

    @property
    def factor_ndims(self) -> tuple[int, ...]:
        """Number of axes of each factor."""
        return self._factor_ndims

    def transpose(self, conjugate: bool = False) -> KroneckerLinearSolver:
        new_domain = self._codomain
        new_codomain = self._domain
        new_solvers = [solver.transpose() for solver in self._solvers]
        return KroneckerLinearSolver(new_domain, new_codomain, new_solvers, factor_ndims=self._factor_ndims)

    def dot(self, v: StencilVector, out: StencilVector | None = None) -> StencilVector:
        return self.solve(v, out=out)

    @property
    def solvers(self) -> tuple[LinearSolver, ...]:
        """
        Returns an immutable view onto references to the solvers of the factors.
        """
        return tuple(self._solvers)

    def solve(self, rhs: StencilVector, out: StencilVector | None = None) -> StencilVector:
        """
        Solves Ax=b where A is a Kronecker product matrix (and represented as such),
        and b is a suitable vector.

        Parameters
        ----------
        rhs : StencilVector
            The right-hand side b, in the domain.

        out : StencilVector, optional
            Vector in the codomain, in which the solution is stored.

        Returns
        -------
        StencilVector
            The solution x (``out`` if given).
        """

        # type checks
        assert rhs.space is self._domain

        if out is not None:
            assert isinstance( out, StencilVector )
            assert out.space is self._codomain
        else:
            out = StencilVector( rhs.space )
        
        inslice = rhs[self._slice]
        outslice = out[self._slice]

        # call the actual kernel
        self._solve_nd(inslice, outslice)
        
        out.update_ghost_regions()
        return out
 
    def _solve_nd(self, inslice, outslice):
        """
        The internal solve loop. Can handle arbitrary dimensions.
        """
        temp1 = self._temp1
        temp2 = self._temp2

        # copy input
        self._inslice_to_temp(inslice, temp1)

        # internal passes
        for i in range(self._ndim - 1):
            # solve direction
            self._solver_passes[i].solve_pass(temp1, temp2)

            # reorder and swap
            self._reorder_temp_to_temp(temp1, temp2, i)
            temp1, temp2 = temp2, temp1
        
        # last pass
        self._solver_passes[-1].solve_pass(temp1, temp2)

        # copy to output
        self._reorder_temp_to_outslice(temp1, outslice)

    def _inslice_to_temp(self, inslice, target):
        """
        Copies data to an internal, 1-dimensional temporary array.
        Does not allocate any new array.
        """
        targetview = target[:self._localsize]
        targetview.shape = inslice.shape

        targetview[:] = inslice
    
    def _reorder_temp_to_temp(self, source, target, i):
        """
        Reorders the dimensions of the temporary arrays, and copies data from one to another.
        Does not allocate any new array.
        """
        sourceview = source[:self._localsize]
        self._shapes[i] = tuple(int(x) for x in self._shapes[i])
        self._shapes[i+1] = tuple(int(x) for x in self._shapes[i+1])
        sourceview.shape = self._shapes[i]

        targetview = target[:self._localsize]
        targetview.shape = self._shapes[i+1]
        
        # targetview[:] = sourceview.transpose(self._perm)
        perm = tuple(int(p) for p in self._perm)

        targetview[:] = sourceview.transpose(perm)
    
    def _reorder_temp_to_outslice(self, source, outslice):
        """
        Reorders the dimensions of the temporary array for a final time, and copies it to the output.
        Does not allocate any new array.
        """
        sourceview = source[:self._localsize]
        sourceview.shape = self._shapes[-1]

        # outslice[:] = sourceview.transpose(self._perm)
        perm = tuple(int(p) for p in self._perm)
        if len(self._axes) == outslice.ndim:
            outslice[:] = sourceview.transpose(perm)
        else:
            # factors with several axes: back from one axis per factor to all axes
            outslice[:] = sourceview.transpose(perm).reshape(outslice.shape)

    class KroneckerSolverSerialPass:
        """
        Solves a linear equation for several right-hand sides at the same time,
        given that the data is already in memory.

        Parameters
        ----------
        solver : BandedSolver or SparseSolver
            The internally used solver class.
        
        nglobal : int
            The length of the dimension which we want to solve for.
        
        mglobal : int
            The number of right-hand sides we want to solve. Equals the product of the
            number of dimensions which we do NOT want to solve for
            (when squashing all these dimensions into a single one).
            I.e. mglobal*nglobal is the total data size.
        """
        def __init__(self, solver, nglobal, mglobal):
            self._numrhs = mglobal
            self._dimrhs = nglobal
            self._datasize = nglobal*mglobal
            self._solver = solver
            self._view = None
        
        def required_memory(self):
            """
            Returns the required memory for this operation. Minimum size for the workmem and tempmem parameters.
            """
            return self._datasize

        def solve_pass(self, workmem, tempmem):
            """
            Solves the data available in workmem, assuming that all data is available locally.

            Parameters
            ----------
            workmem : ndarray
                The data which is to be solved. It is a one-dimensional ndarray
                which contains all columns contiguously ordered in memory one after another.
                Its minimum size is also given by `self.required_mem()`.
            
            tempmem : ndarray
                Ignored, it exists for compatibility with the parallel solver.
            """
            # reshape necessary memory in column-major
            view = workmem[:self._datasize]
            view.shape = (int(self._numrhs), int(self._dimrhs))

            # call solver in in-place mode
            self._solver.solve(view, out=view)

    class KroneckerSolverParallelPass:
        """
        Solves a linear equation for several right-hand sides at the same time,
        using an Alltoallv operation to distribute the data.

        The parameters use the form of n and m; here n denotes the
        length of the dimension we want to solve for, and m is the
        length of all other dimensions, multiplied with each other.
        These n and m are then suffixed with local and global,
        denoting how much of them we have (or want to have) locally.
        So, nglobal is the dimension of the columns we want to solve,
        nlocal is the part we have on our local processor. mglobal is
        the number of right-hand sides to solve in the whole communicator,
        and mlocal is the number of right-hand sides we will solve on our
        local processor.

        Parameters
        ----------
        solver : BandedSolver or SparseSolver
            The internally used solver class.
        
        mpi_type : MPI type
            The MPI type of the space. Used for the Alltoallv.
        
        i : int
            The index of the dimension.
        
        cart : CartDecomposition
            The cartesian decomposition we use.

        mglobal : int
            The number of right-hand sides we want to solve. Equals the product of the
            number of dimensions which we do NOT want to solve for
            (when squashing all these dimensions into a single one).
            I.e. mglobal*nglobal is the total data size in our communicator (not on the whole grid though).
        
        nglobal : int
            The length of the dimension which we want to solve.
            (the total length, not the one we have on this process)
        
        nlocal : int
            The length of the part of the dimension to solve which is located on this process already.

        localsize : int
            The size of data on our local process.
            Equals mlocal * nlocal (given that we know the former).
        """

        # To understand the following, here is a short explaination. Consider two processes like this:
        #
        # Pr1 | Pr2
        # 0 1 | 2 3
        # 4 5 | 6 7
        # 8 9 | A B 
        # C D | E F
        #
        # i.e. Pr1 has 0 1 4 5 8 9 C D; Pr2 has 2 3 6 7 A B E F
        #
        # We now would like to get each line on at least one process. So, we do an AlltoAll like this:
        #
        # Pr1 | Pr2
        # 0 1 | 2 3 | to Pr1
        # 4 5 | 6 7 | to Pr1
        # ------------------
        # 8 9 | A B | to Pr2
        # C D | E F | to Pr2
        #
        # But the data is transported per process, i.e. we get in this order:
        # 0 1 4 5 2 3 6 7 on Pr1
        # 8 9 C D A B E F on Pr2
        #
        # so we still need to re-order (i.e. partially transpose) locally to finally get what we want.
        # 0 1 2 3 4 5 6 7 on Pr1
        # 8 9 A B C D E F on Pr2
        #

        # NOTE: ideas for future improvements, if this is too slow:
        #
        # * Use MPI composite Datatypes (i.e. MPI contiguous and vector).
        #       This may improve performance, depending on the implementation
        #       (therefore, a library-level switch or similar would be an option here).
        #       Mainly, we could push what happens in _blocked_to_contiguous and
        #       _contiguous_to_blocked methods into the MPI implementation.
        #
        # * Use Alltoall instead of Alltoallv, when applicable, since it might be faster as well.
        #       This if for example the case, if the cartesian communicator (cart argument)
        #       assigns the same number of data points to all processes.
        #       (i.e. global_ends[i] - global_starts[i] is constant)
        #       Then, we only need mlocal to be constant (except the last element) as well.
        #
        
        def __init__(self, solver, mpi_type, i, cart, mglobal, nglobal, nlocal, localsize):
            self._nglobal = nglobal

            # cartesian distribution
            comm = cart.subcomm[i]
            cartend = cart.global_ends[i] + 1
            cartstart = cart.global_starts[i]
            cartsize = cartend - cartstart

            # source MPI sizes and disps -- these are passed straight to
            # mpi4py's Alltoallv as counts/displacements, which (like
            # cart.global_starts/global_ends above) must be host int arrays
            # regardless of backend; keep this whole computation on numpy.
            mlocal_pre = mglobal // comm.size
            mlocal_add = mglobal % comm.size
            sourcesizes = np.full((comm.size,), mlocal_pre, dtype=int)
            sourcesizes[:mlocal_add] += 1
            mlocal = sourcesizes[comm.rank]
            sourcesizes *= nlocal

            # disps, created from the sizes
            sourcedisps = np.zeros((comm.size+1,), dtype=int)
            np.cumsum(sourcesizes, out=sourcedisps[1:])
            sourcedisps = sourcedisps[:-1]

            # target MPI sizes and disps
            # (mlocal is the same over all processes in the communicator)
            targetsizes = cartsize * mlocal
            targetdisps = cartstart * mlocal

            # setting all arguments to keep
            self._mlocal = mlocal
            self._localsize = localsize
            self._datasize = mlocal * nglobal
            self._source_transfer = (sourcesizes, sourcedisps)
            self._target_transfer = (targetsizes, targetdisps)
            self._mpi_type = mpi_type
            self._cartstart = cartstart
            self._cartend = cartend
            self._comm = comm
            self._serialsolver = KroneckerLinearSolver.KroneckerSolverSerialPass(solver, nglobal, mlocal)

        def required_memory(self):
            """
            Returns the required memory for this operation. Minimum size for the workmem and tempmem parameters.
            """
            return max(self._datasize, self._localsize)

        def _blocked_to_contiguous(self, blocked, contiguous):
            """
            Copies from a blocked view to a contiguous view.
            Equals roughly a partial transpose, if the block sizes in the cartesian grid are the same.
            """
            blocked_view = blocked[:self._datasize]
            blocked_view.shape = (self._mlocal,self._nglobal)
            for start, end in zip(self._cartstart, self._cartend):
                contiguouspart = contiguous[start*self._mlocal:end*self._mlocal]
                contiguouspart.shape = (self._mlocal,end-start)
                blocked_view[:,start:end] = contiguouspart
        
        def _contiguous_to_blocked(self, blocked, contiguous):
            """
            Copies from a contiguous view to a blocked view.
            Equals roughly a partial transpose, if the block sizes in the cartesian grid are the same.
            """
            blocked_view = blocked[:self._datasize]
            blocked_view.shape = (self._mlocal,self._nglobal)
            for start, end in zip(self._cartstart, self._cartend):
                contiguouspart = contiguous[start*self._mlocal:end*self._mlocal]
                contiguouspart.shape = (self._mlocal,end-start)
                contiguouspart[:] = blocked_view[:,start:end]

        def solve_pass(self, workmem, tempmem):
            """
            Solves the data available in workmem in a distributed manner, using MPI_Alltoallv.

            Parameters
            ----------
            workmem : ndarray
                The data which is used for solving.
                All columns to be solved are ordered contiguously.
                Its minimum size is given by `self.required_mem()`
            
            tempmem : ndarray
                Temporary array of the same minimum size as workmem.
            """
            # preparation
            sourceargs = [workmem[:self._localsize], self._source_transfer, self._mpi_type]
            targetargs = [tempmem[:self._datasize], self._target_transfer, self._mpi_type]

            # parts of stripes -> blocked stripes
            # (MPI reads/writes the work arrays directly; on a device backend the
            # kernels that produced them must have finished first.)
            synchronize_for_mpi(workmem, tempmem)
            self._comm.Alltoallv(sourceargs, targetargs)

            # blocked stripes -> ordered stripes
            self._blocked_to_contiguous(workmem, tempmem)

            # actual solve (source contains the data)
            self._serialsolver.solve_pass(workmem, tempmem)

            # ordered stripes -> blocked stripes
            self._contiguous_to_blocked(workmem, tempmem)

            # blocked stripes -> parts of stripes
            synchronize_for_mpi(workmem, tempmem)
            self._comm.Alltoallv(targetargs, sourceargs)

#==============================================================================
def kronecker_solve(solvers: Sequence[LinearSolver],
                    rhs: StencilVector,
                    out: StencilVector | None = None,
                    factor_ndims: Sequence[int] | None = None) -> StencilVector:
    """
    Solve the linear system Ax=b with A = kron(A_1, A_2, ..., A_m), given
    one linear solver L_k per factor ($L_k$ solves $A_k x_k = b_k$).

    Parameters
    ----------
    solvers : sequence of LinearSolver
        Solvers for the factors, ordered by axis: [L_1, L_2, ..., L_m].

    rhs : StencilVector
        Right hand side vector of linear system Ax=b.

    out : StencilVector, optional
        Vector in the space of rhs, in which the solution is stored.

    factor_ndims : sequence of int, optional
        Number of axes of each factor (see KroneckerLinearSolver). Defaults to 1 for every factor.

    Returns
    -------
    StencilVector
        The solution x (``out`` if given).
    """
    # all these feasability checks are again performed in the KroneckerLinearSolver class
    assert hasattr(solvers, '__iter__')
    for solver in solvers:
        assert isinstance(solver, LinearSolver)

    assert isinstance(rhs, StencilVector)
    if factor_ndims is None:
        assert rhs.space.ndim == len(solvers)
    else:
        assert rhs.space.ndim == sum(factor_ndims)

    if out is not None:
        assert isinstance(out, StencilVector)
        assert out.space is rhs.space
    else:
        out = StencilVector(rhs.space)

    kronsolver = KroneckerLinearSolver(rhs.space, rhs.space, solvers, factor_ndims=factor_ndims)
    return kronsolver.solve(rhs, out=out)
