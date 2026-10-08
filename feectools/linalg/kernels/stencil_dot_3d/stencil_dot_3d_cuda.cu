// CUDA version of stencil_dot_3d (stencil_dot_3d_kernels.py), with the same arguments in the same order.
#include "cunumpy/array_view.cuh"
#include "cunumpy/index.cuh"

/**
 * Stencil matrix-vector product out = mat @ x on the owned rows, as the pyccel kernel stencil_dot_3d.
 *
 * One thread per entry of `out` (n_threads = out.size, see __init__.py), the last axis varying fastest. A
 * thread outside the owned rows returns without writing, so the padding of `out` is left as it is, as in
 * pyccel. Along each direction k the matrix has n_k = mat.shape[3 + k] diagonals; interior rows use all of
 * them, the last owned row (i_k == e_out[k]) uses n_k - 1 + add[k], as in the 1D kernel. The diagonals are
 * summed in the same order as in pyccel (d1, d2, d3 innermost).
 *
 * @param mat   matrix data (a 6D view), shape (rows of `out`..., diagonals...)
 * @param x     data of the domain vector, ghost regions included
 * @param out   data of the codomain vector; the owned rows are written
 * @param s_in, p_in, add, s_out, e_out, p_out  per direction (length 3), as in the 1D kernel
 */
extern "C" __global__ void stencil_dot_3d(Array6D<double> mat, Array3D<double> x, Array3D<double> out,
                                          const long long* s_in, const long long* p_in, const long long* add,
                                          const long long* s_out, const long long* e_out,
                                          const long long* p_out)
{
    CUNUMPY_THREAD_1D(thread, out.size());

    const long long i1_loc = thread / (out.shape[1] * out.shape[2]) - p_out[0];  // local row indices
    const long long i2_loc = (thread / out.shape[2]) % out.shape[1] - p_out[1];
    const long long i3_loc = thread % out.shape[2] - p_out[2];
    if (i1_loc < 0 || i1_loc > e_out[0] - s_out[0]) return;
    if (i2_loc < 0 || i2_loc > e_out[1] - s_out[1]) return;
    if (i3_loc < 0 || i3_loc > e_out[2] - s_out[2]) return;
    const long long i1 = s_out[0] + i1_loc;  // global row indices
    const long long i2 = s_out[1] + i2_loc;
    const long long i3 = s_out[2] + i3_loc;

    const long long nd1 = (i1 == e_out[0]) ? mat.shape[3] - 1 + add[0] : mat.shape[3];
    const long long nd2 = (i2 == e_out[1]) ? mat.shape[4] - 1 + add[1] : mat.shape[4];
    const long long nd3 = (i3 == e_out[2]) ? mat.shape[5] - 1 + add[2] : mat.shape[5];
    // x index of diagonal 0 minus the global row index, per direction
    const long long off1 = p_in[0] - (mat.shape[3] - 1) / 2 - s_in[0];
    const long long off2 = p_in[1] - (mat.shape[4] - 1) / 2 - s_in[1];
    const long long off3 = p_in[2] - (mat.shape[5] - 1) / 2 - s_in[2];

    double val = 0.;
    for (long long d1 = 0; d1 < nd1; ++d1)
        for (long long d2 = 0; d2 < nd2; ++d2)
            for (long long d3 = 0; d3 < nd3; ++d3)
                val += mat(p_out[0] + i1_loc, p_out[1] + i2_loc, p_out[2] + i3_loc, d1, d2, d3) *
                       x(i1 + d1 + off1, i2 + d2 + off2, i3 + d3 + off3);

    out(p_out[0] + i1_loc, p_out[1] + i2_loc, p_out[2] + i3_loc) = val;
}
