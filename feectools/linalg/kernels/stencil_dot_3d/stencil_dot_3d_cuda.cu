// CUDA version of stencil_dot_3d (stencil_dot_3d_kernels.py), with the same arguments in the same order.
#include "cunumpy/array_view.cuh"
#include "cunumpy/index.cuh"

// Lanes that share one row of `out` (a divisor of 32): n_threads = STENCIL_DOT_LANES * out.size, see __init__.py.
// 8 was the fastest of 8, 16 and 32 for 3 x 5 x 7 diagonals at 128^3 on an H100 (1.52 ms vs 1.69 ms with one thread
// per row).
#define STENCIL_DOT_LANES 8

/**
 * Stencil matrix-vector product out = mat @ x on the owned rows, as the pyccel kernel stencil_dot_3d.
 *
 * STENCIL_DOT_LANES consecutive threads per entry of `out` (the last axis of `out` varying fastest): the lanes split
 * the diagonals of the row among themselves (lane k takes the flat diagonal indices k, k + LANES, ...), so that
 * neighbouring threads read neighbouring diagonals of the row of `mat` (its last three, contiguous axes) instead of
 * addresses n_diagonals * 8 bytes apart; the partial sums are then added with warp shuffles. The summation order
 * differs from pyccel (round-off only).
 *
 * Rows outside the owned rows are not written, so the padding of `out` is left as it is, as in pyccel. Along each
 * direction k the matrix has n_k = mat.shape[3 + k] diagonals; interior rows use all of them, the last owned row
 * (i_k == e_out[k]) uses n_k - 1 + add[k], as in the 1D kernel.
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
    // CUDA-only: the row (entry of `out`) of this group of lanes and the lane within it. No thread returns before the
    // shuffles at the end (they need every lane of the warp): threads past the end and rows outside the owned ones
    // just have nothing to add.
    const long long thread = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long row = thread / STENCIL_DOT_LANES;
    const int lane = (int)(thread % STENCIL_DOT_LANES);

    const long long i1_loc = row / (out.shape[1] * out.shape[2]) - p_out[0];  // local row indices
    const long long i2_loc = (row / out.shape[2]) % out.shape[1] - p_out[1];
    const long long i3_loc = row % out.shape[2] - p_out[2];
    const bool owned = row < out.size() && i1_loc >= 0 && i1_loc <= e_out[0] - s_out[0] && i2_loc >= 0 &&
                       i2_loc <= e_out[1] - s_out[1] && i3_loc >= 0 && i3_loc <= e_out[2] - s_out[2];

    const long long r1 = p_out[0] + i1_loc;  // row of `mat` and `out`
    const long long r2 = p_out[1] + i2_loc;
    const long long r3 = p_out[2] + i3_loc;

    double val = 0.;
    if (owned) {
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

        // 32-bit index arithmetic in the loop (64-bit division is slow on the GPU); the diagonals of the row and the
        // matching entries of x through pointers and strides
        const int n3 = (int)nd3, n23 = (int)(nd2 * nd3), nd = (int)(nd1 * nd2 * nd3);
        const double* mat_row = &mat(r1, r2, r3, 0, 0, 0);
        const double* x_row = &x(i1 + off1, i2 + off2, i3 + off3);
        const long long m3 = mat.strides[3], m4 = mat.strides[4], m5 = mat.strides[5];
        const long long x1 = x.strides[0], x2 = x.strides[1], x3 = x.strides[2];

        for (int d = lane; d < nd; d += STENCIL_DOT_LANES) {
            const int d1 = d / n23;
            const int d23 = d - d1 * n23;
            const int d2 = d23 / n3;
            const int d3 = d23 - d2 * n3;
            val += mat_row[d1 * m3 + d2 * m4 + d3 * m5] * x_row[d1 * x1 + d2 * x2 + d3 * x3];
        }
    }

    // sum of the partial sums of the row's lanes (the first lane ends up with the total)
    for (int offset = STENCIL_DOT_LANES / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffffu, val, offset, STENCIL_DOT_LANES);

    if (owned && lane == 0) out(r1, r2, r3) = val;
}
