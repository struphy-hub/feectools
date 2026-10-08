// CUDA version of stencil_transpose_1d (stencil_transpose_1d_kernels.py), with the same arguments in the same order.
#include "cunumpy/array_view.cuh"
#include "cunumpy/index.cuh"

/**
 * Transpose of a stencil matrix, matT = mat.T on the owned rows of matT, as the pyccel kernel
 * stencil_transpose_1d.
 *
 * One thread per entry (row, diagonal) of `matT` (n_threads = matT.size, see __init__.py). matT has
 * n = matT.shape[1] diagonals (pads qT = (n - 1) / 2), mat has the pads q = (mat.shape[1] - 1) / 2. A thread
 * outside the owned rows of matT, or at a diagonal d1 the pyccel kernel does not write (d1 >= n - 1 + add in
 * the last owned row), returns without writing. Every other thread copies one entry: row i1 of matT, column
 * j1 = i1 - qT + d1, is row j1 of mat at diagonal q + i1 - j1.
 *
 * @param mat   matrix data, shape (rows of the codomain of mat, diagonals)
 * @param matT  data of the transposed matrix; the owned rows are written
 * @param s_in  global start of the rows of mat (= columns of matT) of this process
 * @param p_in  padding of the rows of mat
 * @param add   1 if the last row of matT uses all diagonals, else 0
 * @param s_out global start of the rows of matT of this process
 * @param e_out global end (inclusive) of the rows of matT of this process
 * @param p_out padding of the rows of matT
 */
extern "C" __global__ void stencil_transpose_1d(Array2D<double> mat, Array2D<double> matT, long long s_in,
                                                long long p_in, long long add, long long s_out,
                                                long long e_out, long long p_out)
{
    CUNUMPY_THREAD_1D(thread, matT.size());

    const long long d1 = thread % matT.shape[1];
    const long long i1_loc = thread / matT.shape[1] - p_out;  // local row index of matT
    if (i1_loc < 0 || i1_loc > e_out - s_out) return;
    const long long i1 = s_out + i1_loc;  // global row index of matT = global column index of mat
    if (i1 == e_out && d1 >= matT.shape[1] - 1 + add) return;

    const long long j1 = i1 - (matT.shape[1] - 1) / 2 + d1;  // global column index of matT
    const long long j1_loc = j1 - s_in;  // local column index of matT = local row index of mat

    matT(p_out + i1_loc, d1) = mat(p_in + j1_loc, (mat.shape[1] - 1) / 2 + i1 - j1);
}
