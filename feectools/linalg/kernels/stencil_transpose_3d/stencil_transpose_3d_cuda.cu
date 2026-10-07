// CUDA version of stencil_transpose_3d (stencil_transpose_3d_kernels.py), with the same arguments in the same order.
#include "cunumpy/array_view.cuh"
#include "cunumpy/index.cuh"

/**
 * Transpose of a stencil matrix, matT = mat.T on the owned rows of matT, as the pyccel kernel
 * stencil_transpose_3d.
 *
 * One thread per entry (i1, i2, i3, d1, d2, d3) of `matT` (n_threads = matT.size, see __init__.py), the last
 * axis varying fastest. A thread outside the owned rows of matT, or at a diagonal the pyccel kernel does not
 * write (per direction k: d_k >= matT.shape[3 + k] - 1 + add[k] in the last owned row), returns without
 * writing; every other thread copies one entry, as in the 1D kernel. Both matrices are 6D views, so their
 * shapes (rows and diagonals) come with them.
 *
 * @param mat   matrix data, shape (rows of the codomain of mat..., diagonals...)
 * @param matT  data of the transposed matrix; the owned rows are written
 * @param s_in, p_in, add, s_out, e_out, p_out  per direction (length 3), as in the 1D kernel
 */
extern "C" __global__ void stencil_transpose_3d(Array6D<double> mat, Array6D<double> matT, const long long* s_in,
                                                const long long* p_in, const long long* add,
                                                const long long* s_out, const long long* e_out,
                                                const long long* p_out)
{
    CUNUMPY_THREAD_1D(thread, matT.size());

    long long rest = thread;
    const long long d3 = rest % matT.shape[5];
    rest /= matT.shape[5];
    const long long d2 = rest % matT.shape[4];
    rest /= matT.shape[4];
    const long long d1 = rest % matT.shape[3];
    rest /= matT.shape[3];
    const long long i3_loc = rest % matT.shape[2] - p_out[2];  // local row indices of matT
    rest /= matT.shape[2];
    const long long i2_loc = rest % matT.shape[1] - p_out[1];
    const long long i1_loc = rest / matT.shape[1] - p_out[0];
    if (i1_loc < 0 || i1_loc > e_out[0] - s_out[0]) return;
    if (i2_loc < 0 || i2_loc > e_out[1] - s_out[1]) return;
    if (i3_loc < 0 || i3_loc > e_out[2] - s_out[2]) return;
    const long long i1 = s_out[0] + i1_loc;  // global row indices of matT = global column indices of mat
    const long long i2 = s_out[1] + i2_loc;
    const long long i3 = s_out[2] + i3_loc;
    if (i1 == e_out[0] && d1 >= matT.shape[3] - 1 + add[0]) return;
    if (i2 == e_out[1] && d2 >= matT.shape[4] - 1 + add[1]) return;
    if (i3 == e_out[2] && d3 >= matT.shape[5] - 1 + add[2]) return;

    const long long j1 = i1 - (matT.shape[3] - 1) / 2 + d1;  // global column indices of matT
    const long long j2 = i2 - (matT.shape[4] - 1) / 2 + d2;
    const long long j3 = i3 - (matT.shape[5] - 1) / 2 + d3;
    const long long j1_loc = j1 - s_in[0];  // local column indices of matT = local row indices of mat
    const long long j2_loc = j2 - s_in[1];
    const long long j3_loc = j3 - s_in[2];

    matT(p_out[0] + i1_loc, p_out[1] + i2_loc, p_out[2] + i3_loc, d1, d2, d3) =
        mat(p_in[0] + j1_loc, p_in[1] + j2_loc, p_in[2] + j3_loc,
            (mat.shape[3] - 1) / 2 + i1 - j1, (mat.shape[4] - 1) / 2 + i2 - j2, (mat.shape[5] - 1) / 2 + i3 - j3);
}
