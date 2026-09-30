"""
Device (CUDA) counterpart of the compiled stencil matrix-vector kernels in
``feectools.linalg.stencil_dot_kernels``.

Those kernels are host code. Handing them CuPy arrays makes
:class:`cunumpy.PyccelKernel` copy the matrix *and* the vector off the device,
run a serial loop on the CPU, and copy the result back -- which costs far more
than the product itself. This module computes the same thing on the device.

The operation is the stencil product

    out[i] = sum_d mat[i, d] * x[i + d - s_in]

over the owned rows ``i`` of the codomain, where ``d`` runs over the
``2 * p_in + 1`` diagonals in each direction. The last owned row along a
direction uses ``2 * p_in + add`` diagonals instead, which is how the compiled
kernels handle a rectangular matrix (for a square one ``add == 1`` and the
distinction disappears).

One thread computes one output point and loops over the diagonals internally,
so a product costs a single kernel launch regardless of the bandwidth.
"""

import numpy as np

__all__ = ('device_matvec', 'supports')

# Dtypes the generated kernels cover, mapped to their CUDA C type.
_CTYPES = {
    np.dtype(np.float64): 'double',
    np.dtype(np.complex128): 'complex<double>',
}

# Cache of compiled kernels, keyed by (ndim, dtype).
_KERNELS = {}

_THREADS_PER_BLOCK = 256


def supports(ndim, dtype):
    """Whether a device kernel exists for this dimensionality and dtype."""
    return ndim in (1, 2, 3) and np.dtype(dtype) in _CTYPES


def _source(ndim, ctype):
    """Generate the CUDA C source of the matvec kernel for `ndim` dimensions.

    Indices are named per direction so the generated code stays close to the
    compiled kernels it mirrors:

    * ``i{k}``   -- local row index along direction k, in [0, n{k})
    * ``d{k}``   -- diagonal index along direction k
    * ``po{k}``  -- padding of the codomain, the offset of row 0 in `mat`/`out`
    * ``of{k}``  -- s_out - s_in, the offset of row 0 in `x`
    """
    dims = range(ndim)

    params = ', '.join(
        [f'const int n{k}' for k in dims]
        + [f'const int nd{k}' for k in dims]     # diagonals, interior rows
        + [f'const int na{k}' for k in dims]     # diagonals, last row
        + [f'const long ms{k}' for k in dims]    # mat strides, row axes
        + [f'const long md{k}' for k in dims]    # mat strides, diagonal axes
        + [f'const long xs{k}' for k in dims]
        + [f'const long os{k}' for k in dims]
        + [f'const int po{k}' for k in dims]
        + [f'const int of{k}' for k in dims]
    )

    # Unflatten the thread id into one index per direction (last varies fastest)
    total = ' * '.join(f'(long)n{k}' for k in dims)
    unflatten = []
    for k in reversed(list(dims)):
        divisor = ' * '.join(f'(long)n{j}' for j in range(k + 1, ndim))
        if k == 0:
            unflatten.append(f'    int i0 = (int)(tid / ({divisor}));'
                             if divisor else '    int i0 = (int)tid;')
        elif divisor:
            unflatten.append(f'    int i{k} = (int)((tid / ({divisor})) % n{k});')
        else:
            unflatten.append(f'    int i{k} = (int)(tid % n{k});')
    unflatten = '\n'.join(reversed(unflatten))

    mat_base = ' + '.join(f'(long)(po{k} + i{k}) * ms{k}' for k in dims)
    x_base = ' + '.join(f'(long)(of{k} + i{k}) * xs{k}' for k in dims)
    out_index = ' + '.join(f'(long)(po{k} + i{k}) * os{k}' for k in dims)

    # The last row along a direction uses a different number of diagonals.
    bounds = '\n'.join(
        f'    const int b{k} = (i{k} == n{k} - 1) ? na{k} : nd{k};' for k in dims
    )

    loops = ''
    for k in dims:
        loops += '    ' * (k + 1) + f'for (int d{k} = 0; d{k} < b{k}; ++d{k})\n'
    body_indent = '    ' * (ndim + 1)
    mat_off = ' + '.join(f'(long)d{k} * md{k}' for k in dims)
    x_off = ' + '.join(f'(long)d{k} * xs{k}' for k in dims)
    loops += (f'{body_indent}val += mat[mbase + {mat_off}]\n'
              f'{body_indent}     * x[xbase + {x_off}];\n')

    return f'''
#include <cupy/complex.cuh>

extern "C" __global__
void stencil_matvec(const {ctype}* __restrict__ mat,
                    const {ctype}* __restrict__ x,
                    {ctype}* __restrict__ out,
                    {params})
{{
    long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (tid >= {total}) return;

{unflatten}

{bounds}

    const long mbase = {mat_base};
    const long xbase = {x_base};

    {ctype} val = {ctype}(0);
{loops}
    out[{out_index}] = val;
}}
'''


def _kernel(ndim, dtype):
    """Compile (once) and return the kernel for this dimensionality/dtype."""
    key = (ndim, np.dtype(dtype))
    if key not in _KERNELS:
        import cupy as cp
        _KERNELS[key] = cp.RawKernel(_source(ndim, _CTYPES[key[1]]),
                                     'stencil_matvec')
    return _KERNELS[key]


def _strides(arr, axes):
    """Strides of `arr` along `axes`, in elements rather than bytes."""
    return [arr.strides[a] // arr.itemsize for a in axes]


def device_matvec(mat, x, out, s_in, p_in, add, s_out, e_out, p_out):
    """
    Compute ``out = mat @ x`` on the device, in place.

    Only the owned rows of `out` are written, exactly as the compiled kernels
    do; the caller is responsible for the state of the padding.

    Parameters
    ----------
    mat : cupy.ndarray
        Matrix data, of shape (rows..., diagonals...) -- 2 * ndim axes.

    x, out : cupy.ndarray
        Domain and codomain vector data, each of ndim axes.

    s_in, p_in, add, s_out, e_out, p_out : sequence[int]
        Per-direction start of the domain, padding of the domain, rectangular
        correction, and start/end/padding of the codomain -- the same values
        the compiled kernels take.
    """
    ndim = x.ndim

    n = [int(e) - int(s) + 1 for s, e in zip(s_out, e_out)]
    nd = [2 * int(p) + 1 for p in p_in]
    na = [2 * int(p) + int(a) for p, a in zip(p_in, add)]
    off = [int(so) - int(si) for so, si in zip(s_out, s_in)]
    po = [int(p) for p in p_out]

    args = (mat, x, out,
            *n, *nd, *na,
            *_strides(mat, range(ndim)),
            *_strides(mat, range(ndim, 2 * ndim)),
            *_strides(x, range(ndim)),
            *_strides(out, range(ndim)),
            *po, *off)

    total = 1
    for k in n:
        total *= k
    blocks = (total + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK

    _kernel(ndim, x.dtype)((blocks,), (_THREADS_PER_BLOCK,), args)
    return out
