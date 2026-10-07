"""Transpose of a stencil matrix, ``matT = mat.T`` on the owned rows of ``matT`` (1D).

The host version of ``stencil_transpose_1d``.

Moved from ``feectools.linalg.stencil_transpose_kernels.transpose_1d_kernel``. The CUDA version in
``stencil_transpose_1d_cuda.cu`` takes the same arguments in the same order.

The numbers of diagonals are read from the matrices: ``matT`` has ``n = matT.shape[1]`` diagonals, so its pads are
``qT = (n - 1) // 2`` and diagonal ``d`` of row ``i`` is the column ``j = i - qT + d``; that entry is row ``j`` of
``mat`` at the diagonal ``q + i - j``, with ``q = (mat.shape[1] - 1) // 2``. Interior rows of ``matT`` get all ``n``
diagonals, the last owned row ``n - 1 + add``.
"""


def stencil_transpose_1d(mat: 'float[:, :]',
                         matT: 'float[:, :]',
                         s_in: int,  # refers to matT
                         p_in: int,
                         add: int,
                         s_out: int,
                         e_out: int,
                         p_out: int):

    n_diags1 = matT.shape[1]
    qT1 = (n_diags1 - 1) // 2  # pads of matT
    q1 = (mat.shape[1] - 1) // 2  # pads of mat

    for i1 in range(s_out, e_out + 1):  # global row index of matT = global column index of mat
        i1_loc = i1 - s_out  # local row index of matT
        nd1 = n_diags1
        if i1 == e_out:
            nd1 = n_diags1 - 1 + add

        for d1 in range(nd1):
            j1 = i1 - qT1 + d1  # global column index of matT
            j1_loc = j1 - s_in  # local column index of matT = local row index of mat

            matT[p_out + i1_loc, d1] = mat[p_in + j1_loc, q1 + i1 - j1]
