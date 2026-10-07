"""Stencil matrix-vector product ``out = mat @ x`` on the owned rows (1D).

The host version of ``stencil_dot_1d``.

Moved from ``feectools.linalg.stencil_dot_kernels.matvec_1d_kernel``. The CUDA version in ``stencil_dot_1d_cuda.cu``
takes the same arguments in the same order.

The number of diagonals is read from ``mat``: ``n = mat.shape[1]`` diagonals, the pads of the matrix are
``q = (n - 1) // 2`` (at most the pads ``p_in`` of the domain), and diagonal ``d`` of row ``i`` is the column
``i - q + d``. Interior rows use all ``n`` diagonals, the last owned row ``n - 1 + add``.
"""


def stencil_dot_1d(mat: 'float[:, :]',
                   x: 'float[:]',
                   out: 'float[:]',
                   s_in: int,
                   p_in: int,
                   add: int,
                   s_out: int,
                   e_out: int,
                   p_out: int):

    n_diags1 = mat.shape[1]
    # x index of diagonal 0 minus the global row index: column i1 - q + d1 is at x[i1 - q + d1 - s_in + p_in]
    off1 = p_in - (n_diags1 - 1) // 2 - s_in

    for i1 in range(s_out, e_out + 1):  # global row index
        i1_loc = i1 - s_out  # local row index
        nd1 = n_diags1
        if i1 == e_out:
            nd1 = n_diags1 - 1 + add

        val = 0.
        for d1 in range(nd1):
            val += mat[p_out + i1_loc, d1] * x[i1 + d1 + off1]

        out[p_out + i1_loc] = val
