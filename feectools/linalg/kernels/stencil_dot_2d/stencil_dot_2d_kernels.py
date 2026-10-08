"""Stencil matrix-vector product ``out = mat @ x`` on the owned rows (2D).

The host version of ``stencil_dot_2d``.

Moved from ``feectools.linalg.stencil_dot_kernels.matvec_2d_kernel``. The CUDA version in ``stencil_dot_2d_cuda.cu``
takes the same arguments in the same order.

The number of diagonals is read from ``mat``, per direction as in ``stencil_dot_1d``: ``n_k = mat.shape[2 + k]``,
the pads of the matrix are ``q_k = (n_k - 1) // 2`` and diagonal ``d_k`` of row ``i_k`` is the column
``i_k - q_k + d_k``. Interior rows use all ``n_k`` diagonals, the last owned row ``n_k - 1 + add[k]``.
"""


def stencil_dot_2d(mat: 'float[:, :, :, :]',
                   x: 'float[:, :]',
                   out: 'float[:, :]',
                   s_in: 'int[:]',
                   p_in: 'int[:]',
                   add: 'int[:]',
                   s_out: 'int[:]',
                   e_out: 'int[:]',
                   p_out: 'int[:]'):

    n_diags1 = mat.shape[2]
    n_diags2 = mat.shape[3]
    # x index of diagonal 0 minus the global row index, per direction
    off1 = p_in[0] - (n_diags1 - 1) // 2 - s_in[0]
    off2 = p_in[1] - (n_diags2 - 1) // 2 - s_in[1]

    for i1 in range(s_out[0], e_out[0] + 1):  # global row indices
        i1_loc = i1 - s_out[0]  # local row indices
        nd1 = n_diags1
        if i1 == e_out[0]:
            nd1 = n_diags1 - 1 + add[0]

        for i2 in range(s_out[1], e_out[1] + 1):
            i2_loc = i2 - s_out[1]
            nd2 = n_diags2
            if i2 == e_out[1]:
                nd2 = n_diags2 - 1 + add[1]

            val = 0.
            for d1 in range(nd1):
                for d2 in range(nd2):
                    val += mat[p_out[0] + i1_loc, p_out[1] + i2_loc, d1, d2] * x[i1 + d1 + off1, i2 + d2 + off2]

            out[p_out[0] + i1_loc, p_out[1] + i2_loc] = val
