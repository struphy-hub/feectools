# coding: utf-8

from math import sqrt

import cunumpy as xp

from feectools.linalg.basic import Vector
from feectools.linalg.block import BlockVector, BlockVectorSpace
from feectools.linalg.stencil import StencilVector, StencilVectorSpace
from feectools.linalg.topetsc import get_npts_per_block, petsc_local_to_psydac

__all__ = (
    'array_to_psydac',
    'petsc_to_psydac',
    '_sym_ortho',
)


#==============================================================================
def array_to_psydac(x, V):
    """
    Convert a NumPy array to a Vector of the space V. This function is designed to be
    the inverse of the method .toarray() of the class Vector.
    Note: This function works in parallel but it is very costly and should be avoided
    if performance is a priority.
    """

    assert x.ndim == 1, 'Array must be 1D.'
    if x.dtype == complex:
        assert V.dtype == complex, 'Complex array cannot be converted to a real StencilVector'
    assert x.size == V.dimension, 'Array must have the same global size as the space.'

    u = V.zeros()
    _array_to_psydac_recursive(x, u)
    u.update_ghost_regions()

    return u


def _array_to_psydac_recursive(x, u):
    """Recursive function filling in the coefficients of each block of u."""

    assert isinstance(u, Vector)
    V = u.space

    assert x.ndim == 1, 'Array must be 1D.'
    if x.dtype == complex:
        assert V.dtype == complex, 'Complex array cannot be converted to a real StencilVector'
    assert x.size == V.dimension, 'Array must have the same global size as the space.'

    if isinstance(V, BlockVectorSpace):
        for i, V_i in enumerate(V.spaces):
            x_i = x[:V_i.dimension]
            x = x[V_i.dimension:]
            u_i = u[i]
            _array_to_psydac_recursive(x_i, u_i)

    elif isinstance(V, StencilVectorSpace):
        index_global = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))
        u[index_global] = x.reshape(V.npts)[index_global]

    else:
        raise NotImplementedError(f'Can only handle StencilVector or BlockVector spaces, got {type(V)} instead')


#==============================================================================
def petsc_to_psydac(x, Xh, out=None):
    """
    Convert a PETSc.Vec object to a StencilVector or BlockVector.

    In the case of a BlockVector, the blocks must be StencilVector. The general case
    is not yet implemented.
    """

    if isinstance(Xh, BlockVectorSpace):
        if any(isinstance(Xh.spaces[b], BlockVectorSpace) for b in range(len(Xh.spaces))):
            raise NotImplementedError('Block of blocks not implemented.')

        if out is not None:
            assert isinstance(out, BlockVector)
            assert out.space is Xh
            u = out
        else:
            u = BlockVector(Xh)

        comm = x.comm
        dtype = Xh._dtype
        localsize, globalsize = x.getSizes()
        assert globalsize == u.shape[0], 'Sizes of global vectors do not match'

        npts_local_per_block_per_process = xp.array(get_npts_per_block(Xh))
        local_sizes_per_block_per_process = xp.prod(npts_local_per_block_per_process, axis=-1)
        index_shift = 0 + xp.sum(local_sizes_per_block_per_process[:, :comm.Get_rank()], dtype=int)

        for local_petsc_index in range(localsize):
            block_index, psydac_index = petsc_local_to_psydac(Xh, local_petsc_index)
            value = x.getValue(local_petsc_index + index_shift)
            if value != 0:
                u[block_index[0]]._data[psydac_index] = value if dtype is complex else value.real

    elif isinstance(Xh, StencilVectorSpace):
        if out is not None:
            assert isinstance(out, StencilVector)
            assert out.space is Xh
            u = out
        else:
            u = StencilVector(Xh)

        comm = x.comm
        dtype = Xh.dtype
        localsize, globalsize = x.getSizes()
        assert globalsize == u.shape[0], 'Sizes of global vectors do not match'

        npts_local_per_block_per_process = xp.array(get_npts_per_block(Xh))[0]
        local_sizes_per_block_per_process = xp.prod(npts_local_per_block_per_process, axis=-1)
        index_shift = 0 + xp.sum(local_sizes_per_block_per_process[:comm.Get_rank()], dtype=int)

        for local_petsc_index in range(localsize):
            block_index, psydac_index = petsc_local_to_psydac(Xh, local_petsc_index)
            value = x.getValue(local_petsc_index + index_shift)
            if value != 0:
                u._data[psydac_index] = value if dtype is complex else value.real

    else:
        raise ValueError('Xh must be a StencilVectorSpace or a BlockVectorSpace')

    u.update_ghost_regions()

    return u


#==============================================================================
def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.

    This function was taken from the scipy repository:
    https://github.com/scipy/scipy/blob/master/scipy/sparse/linalg/isolve/lsqr.py
    """

    if b == 0:
        return _scalar_sign(a), 0, abs(a)
    elif a == 0:
        return 0, _scalar_sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = _scalar_sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = _scalar_sign(a) / sqrt(1 + tau * tau)
        s = c * tau
        r = a / c
    return c, s, r


#==============================================================================
def _scalar_sign(x):
    """
    Sign of a real Python scalar. `xp.sign` (array_api_compat) requires its
    argument to expose a `.dtype` attribute, which plain Python floats don't have.
    """

    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    return 0.0
