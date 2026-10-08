"""Transpose of a stencil matrix, ``matT = mat.T`` on the owned rows of ``matT`` (3D).

The host version of ``stencil_transpose_3d``.

Moved from ``feectools.linalg.stencil_transpose_kernels.transpose_3d_kernel``. The CUDA version in
``stencil_transpose_3d_cuda.cu`` takes the same arguments in the same order.

The numbers of diagonals are read from the matrices, per direction as in ``stencil_transpose_1d``: diagonal ``d_k``
of row ``i_k`` of ``matT`` is the column ``j_k = i_k - qT_k + d_k``, which is row ``j_k`` of ``mat`` at the diagonal
``q_k + i_k - j_k`` (``qT_k``, ``q_k``: the pads of ``matT`` and ``mat``, from their shapes).
"""


def stencil_transpose_3d(mat: 'float[:, :, :, :, :, :]',
                         matT: 'float[:, :, :, :, :, :]',
                         s_in: 'int[:]',  # refers to matT
                         p_in: 'int[:]',
                         add: 'int[:]',
                         s_out: 'int[:]',
                         e_out: 'int[:]',
                         p_out: 'int[:]'):

    n_diags1 = matT.shape[3]
    n_diags2 = matT.shape[4]
    n_diags3 = matT.shape[5]
    qT1 = (n_diags1 - 1) // 2  # pads of matT
    qT2 = (n_diags2 - 1) // 2
    qT3 = (n_diags3 - 1) // 2
    q1 = (mat.shape[3] - 1) // 2  # pads of mat
    q2 = (mat.shape[4] - 1) // 2
    q3 = (mat.shape[5] - 1) // 2

    for i1 in range(s_out[0], e_out[0] + 1):  # global row indices of matT = global column indices of mat
        i1_loc = i1 - s_out[0]  # local row indices of matT
        nd1 = n_diags1
        if i1 == e_out[0]:
            nd1 = n_diags1 - 1 + add[0]

        for i2 in range(s_out[1], e_out[1] + 1):
            i2_loc = i2 - s_out[1]
            nd2 = n_diags2
            if i2 == e_out[1]:
                nd2 = n_diags2 - 1 + add[1]

            for i3 in range(s_out[2], e_out[2] + 1):
                i3_loc = i3 - s_out[2]
                nd3 = n_diags3
                if i3 == e_out[2]:
                    nd3 = n_diags3 - 1 + add[2]

                for d1 in range(nd1):
                    j1 = i1 - qT1 + d1  # global column indices of matT
                    j1_loc = j1 - s_in[0]  # local column indices of matT = local row indices of mat
                    for d2 in range(nd2):
                        j2 = i2 - qT2 + d2
                        j2_loc = j2 - s_in[1]
                        for d3 in range(nd3):
                            j3 = i3 - qT3 + d3
                            j3_loc = j3 - s_in[2]

                            matT[p_out[0] + i1_loc,
                                 p_out[1] + i2_loc,
                                 p_out[2] + i3_loc,
                                 d1, d2, d3] = mat[p_in[0] + j1_loc,
                                                   p_in[1] + j2_loc,
                                                   p_in[2] + j3_loc,
                                                   q1 + i1 - j1,
                                                   q2 + i2 - j2,
                                                   q3 + i3 - j3]
