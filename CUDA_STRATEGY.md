# CUDA strategy for feectools

Plan for running feectools (the FEEC data and stencil operations used by struphy) on NVIDIA GPUs, next to the
existing pyccel (CPU) kernels. It follows the same rules as struphy's
[CUDA strategy](https://github.com/struphy-hub/struphy/blob/cuda-pr-17-feectools-cuda/CUDA_STRATEGY.md)
(section "feectools" there), and progress is tracked in struphy-hub/struphy#650.

## PR stack

The work is a stack of PRs, set up like struphy's: each PR's base branch is the branch of the PR below it, each
branch contains the one below it, and upstream changes (from `devel-tiny`) come in through merge commits, bottom
up. Each PR's diff on GitHub is therefore only its own step.

| PR | Branch | Base | What | Status |
| --- | --- | --- | --- | --- |
| [#90](https://github.com/struphy-hub/feectools/pull/90) | `cuda-development` | `devel-tiny` | **CUDA 1**: feectools runs on the CuPy backend (the content of [#85](https://github.com/struphy-hub/feectools/pull/85), merged into `cuda-development`) | open |
| [#86](https://github.com/struphy-hub/feectools/pull/86) | `cuda-2-mpi-sync` | `cuda-development` | **CUDA 2**: MPI with device buffers | open |
| [#87](https://github.com/struphy-hub/feectools/pull/87) | `cuda-3-device-binding` | `cuda-2-mpi-sync` | **CUDA 3**: one GPU per MPI rank | open |
| [#88](https://github.com/struphy-hub/feectools/pull/88) | `cuda-4-device-kernels` | `cuda-3-device-binding` | **CUDA 4**: stencil operations on the device, one folder per kernel | open |

Merge bottom up: #90 into `devel-tiny`, then retarget #86 to `devel-tiny` and merge it, and so on. Struphy's CUDA
PRs point their `feectools` submodule at the top of the stack (`cuda-4-device-kernels`) until the stack is merged
and released.

## Goal

feectools runs on either backend with the same code: on the CuPy backend the stencil data are CuPy arrays, MPI
passes device buffers, every rank has its own GPU, and the operations in a solver loop (`dot`, `transpose`,
`inner`, `axpy`) run on the device without host copies.

A developer who adds a GPU version of a feectools kernel only has to write `<name>_cuda.cu` in the folder of its
pyccel kernel and add its parity cases to `linalg/tests/cuda_parity_cases.py`; loading, dispatch and the parity tests
come from
[cunumpy](https://github.com/struphy-hub/cunumpy), as in struphy.

## Principles

- **The backend decides.** The cunumpy backend (`CUNUMPY_BACKEND=cupy` or `cunumpy.set_backend("cupy")`) selects
  CuPy arrays and the CUDA kernels; with NumPy everything runs as before.
- **Data lives where the backend says.** Stencil data are `xp` arrays. Host-only metadata (MPI and index
  bookkeeping, Kronecker solver sizes) stays on NumPy.
- **One folder per kernel** (from CUDA 4 on). A kernel that is called with stencil data in a solver loop lives in
  `feectools/<package>/kernels/<name>/` with `<name>_kernels.py` (pyccel), `<name>_cuda.cu` and an `__init__.py`
  that declares `<name> = Kernel.from_folder(__name__)`. Code imports and calls the kernel. Test inputs are not in
  the folder: the parity cases of every CUDA kernel are in `linalg/tests/cuda_parity_cases.py`.
- **1:1 correspondence.** The CUDA kernel has the same name and the same arguments, in the same order, as its
  pyccel kernel.
- **No silent CPU fallback** for folder kernels: on the CuPy backend a folder kernel without CUDA version raises.
  The other pyccel kernels (B-splines, field evaluation, DOF kernels, used at setup) are wrapped in
  `cunumpy.kernels.PyccelKernel` and copy their arrays to the host and back; see [Open questions](#open-questions).
- **Same cunumpy as struphy.** `cunumpy >= 0.6.1` (the 3D stencil kernels need its 6D array views); names are imported from the submodules
  (`cunumpy.kernels`, `cunumpy.arguments`, `cunumpy.cuda`, `cunumpy.mpi`, `cunumpy.kernel_testing`); cunumpy 0.6 removed
  the old top-level names, and `CudaKernel`/`CudaKernelVariants` are in `cunumpy.kernels` since 0.6.
- **Small steps.** Every PR keeps the NumPy path working and tested.

## CUDA 1 implementation notes (#90, from #85)

feectools works when cunumpy's backend is CuPy, without device kernels:

- Pyccel kernels (stencil, B-splines, field evaluation, DOF kernels) are wrapped in
  `cunumpy.kernels.PyccelKernel`, so they accept CuPy arrays (copied to the host and back).
- Host-only metadata stays on NumPy: MPI and index bookkeeping in `ddm` and `fem.partitioning`, Kronecker solver
  sizes, index arithmetic with Python ints.
- Host-only libraries (LAPACK/SuperLU, SciPy FFT, SciPy sparse) get host copies per array, not by global backend
  (the Kronecker solver no longer does, see [Kronecker solver on the device](#kronecker-solver-on-the-device-96)).
- The 1D collocation matrices of the global projectors are built vectorized (element-wise indexing was one device
  round trip per entry: 334 s of a 348 s Derham setup on the GPU).
- Bug fix on both backends: `StencilMatrix._update_ghost_regions_serial` uses a ghost region `pads * shifts` wide.
- `feectools.ddm.mpi` still disables MPI when `CUNUMPY_BACKEND=cupy` (lifted in CUDA 2).
- Depends on `cunumpy >= 0.5.0, < 0.6`; `PyccelKernel` is imported from `cunumpy.kernels`.

## CUDA 2 implementation notes (#86)

MPI is allowed on the CuPy backend and passes device buffers directly, which needs a **CUDA-aware MPI library** (a
library that is not CUDA-aware segfaults on device buffers; that is what the old check in `feectools.ddm.mpi`
guarded against).

- `cunumpy.mpi.synchronize_for_mpi` is called before every MPI call on device buffers: CuPy kernels run
  asynchronously and MPI does not know CUDA streams, so a buffer still being written would be sent wrong without an
  error. Called in the blocking, non-blocking and interface data exchangers, the `Allreduce` of
  `StencilVectorSpace.inner` and the `Alltoallv` calls of the parallel Kronecker solver. No-op on NumPy.
- `linalg/tests/test_mpi_device.py` compares distributed results with global references (a distributed run
  compared with another distributed run hides stale ghost regions, since every rank agrees on the wrong answer)
  and checks that the exchangers synchronize before calling MPI.

## CUDA 3 implementation notes (#87)

Each MPI rank uses its own GPU instead of always GPU 0.

- `feectools` calls `cunumpy.cuda.bind_local_device()` when it is imported (in `cart` until the move to maybempi): the process uses GPU
  `local_rank % device_count`, from the node-local rank the MPI launcher exports (`cunumpy.mpi.local_rank`).
  No-op on the NumPy backend.
- The CUDA context must exist before MPI is initialized (CUDA-aware MPI requires it). Since feectools uses
  maybempi, the device is bound in `feectools/__init__.py`, which runs before any feectools module imports
  `from maybempi import MPI` (which starts MPI), so the order holds whatever the user imports first.
- `ddm/tests/test_device_binding.py` checks the current device on a GPU (`requires_cupy`).

## CUDA 4 implementation notes (#88)

The stencil operations of solver loops run on the device: `StencilMatrix.dot` (and `vdot`), `StencilMatrix.transpose`,
`StencilVectorSpace.inner` and `StencilVectorSpace.axpy`. Each calls one kernel per dimension, a
`cunumpy.kernels.Kernel` declared in its own folder, which runs the pyccel kernel on NumPy and the CUDA kernel on
CuPy. There are no backend branches and no host staging at these call sites any more, and no generated CUDA source.

| Operation | Kernels (one folder each) | pyccel version moved from |
| --- | --- | --- |
| `StencilMatrix.dot`, `vdot` | `stencil_dot_1d`, `_2d`, `_3d` | `linalg/stencil_dot_kernels.py` (`matvec_<n>d_kernel`) |
| `StencilMatrix.transpose` | `stencil_transpose_1d`, `_2d`, `_3d` | `linalg/stencil_transpose_kernels.py` (`transpose_<n>d_kernel`) |
| `StencilVectorSpace.inner` | `stencil_inner_1d`, `_2d`, `_3d` | `linalg/kernels/inner_kernels.py` (`inner_<n>d`) |
| `StencilVectorSpace.axpy` | `stencil_axpy_1d`, `_2d`, `_3d` | `linalg/kernels/axpy_kernels.py` (`axpy_<n>d`) |

- **Variants: one folder per dimension, float64 on the device.** pyccel needs one function per array rank, and a
  cunumpy `Kernel` holds one CUDA kernel, so each dimension is its own kernel (`stencil_dot_3d`, ...), with the same
  argument list for all dimensions of an operation. The CUDA kernels are written for `double`: the precompiled
  pyccel `dot`/`transpose` kernels are float-only too, while complex `inner`/`axpy` work on NumPy and raise a
  `TypeError` (dtype check of `CudaKernel`) on CuPy. The previous `CudaKernelVariants` over (dimension, dtype)
  of generated source is gone.
- **Signature changes.** The inner kernels add their result to an argument `res` (`res[0] += ...`, the caller
  zeroes it; `inner` passes the MPI send buffer) instead of returning it, because a CUDA kernel cannot return a
  value; on the GPU each block reduces in shared memory and adds its sum with one `atomicAdd`. (The transpose
  kernels had an argument `e_in` for the shape of the 3D matrix data; it is gone, see the 6D views below.)
- **6D matrix data:** see [6D matrix views](#6d-matrix-views-stencil-6d-views). (In #88 the 3D kernels took the
  six-axis data as raw pointers and assumed `2 * p + 1` diagonals; `dot`, `vdot` and `transpose` raised for other
  matrices.)
- **Launch sizes** are declared in each folder's `__init__.py` (`n_threads_from`): one thread per entry of `out`,
  `matT`, `v1` or `x`, the last axis varying fastest; threads outside the owned rows or diagonals return without
  writing, as the pyccel loops do. The inner kernels use blocks of 256 threads.
- **NumPy path:** unchanged results and timings (32×32×16 cells, degree 3: `dot` 6.0 ms, `transpose` 8.2 ms,
  `inner` 0.02 ms, `axpy` 0.013 ms before and after).
- `psydac-accelerate` compiles every `*_kernels.py`, so the new folders are compiled like the old modules; `.cu`
  files are shipped as package data.

## Kronecker solver on the device (#96)

[#96](https://github.com/struphy-hub/feectools/pull/96): `KroneckerLinearSolver` solves device data on the device. Before, its 1D solvers (`BandedSolver`, LAPACK
`?gbtrs`; `SparseSolver`, SuperLU) copied the data of every direction to the host and back in every solve, i.e. in
every CG iteration preconditioned by struphy's `MassMatrixPreconditioner` (struphy-hub/struphy#650, #689).

- **Dense inverses of the 1D matrices.** `direct_solvers.DenseInverse` inverts a 1D matrix once on the host, with
  the solver itself (`M = solver.solve(I)`, so pivoting, transposition and the solver's own quirks are those of
  the host path), and applies it as `rhs @ M`: one GEMM (cuBLAS) over all right-hand sides of a pass. The 1D
  matrices are small (n ~ tens to hundreds), clamped (banded) or periodic (with corners, so banded storage has
  full bandwidth anyway), and an n x n GEMM over m right-hand sides is a better fit for a GPU than m banded
  triangular solves. `cupyx.scipy.linalg` offers dense `lu_factor`/`lu_solve` but, to our knowledge, no
  `solve_banded`; `lu_solve` would need the dense matrix too and run two triangular solves instead of one GEMM.
- **Any linear 1D solver.** The Kronecker passes build the inverse from `solver.solve` on host arrays, so
  struphy's `FFTSolver` (circulant mass matrices, `scipy.linalg.solve_circulant`) works without changes. A 1D
  solver that must not be replaced by its dense matrix sets `dense_on_device = False`: the FFT "solvers" of
  `linalg/fft.py` do (an FFT is O(n log n)); they still stage through the host.
- **Decided by the array**, as before: `KroneckerSolverSerialPass.solve_pass` (also used inside the distributed
  pass, between the two `Alltoallv`) and `BandedSolver`/`SparseSolver.solve` use the dense inverse for
  `xp.is_gpu(array)` and LAPACK/SuperLU otherwise. The host path is unchanged.
- **Copied to the device once, at setup.** A `KroneckerLinearSolver` whose temporaries are on the device (CuPy
  backend) builds the inverses and copies them to the device in its constructor (and in `transpose()`, which
  builds a new solver), so `solve` makes no transfer. A standalone `BandedSolver`/`SparseSolver` called with
  device arrays copies its inverse on the first device solve (one `to_device`), cached per dtype.
- **Accuracy.** Inverse-times-vector and LU solve are both backward stable; for the well-conditioned mass
  matrices the results agree to round-off (tests use 1e-12 relative to the solution).
- **Tests** (`linalg/tests/test_kron_device_solve.py`): `DenseInverse` against the host solvers (banded and
  sparse, clamped and periodic, transposed, real and complex) and the Kronecker solve against a global
  reference, serial and with MPI, on the CPU; the device path with cunumpy's fake CuPy in a serial subprocess
  (clean environment, `MAYBEMPI=0`), including `assert_no_transfers` around `solve`; on a GPU (`requires_cupy`)
  device against host solves under `assert_no_transfers`, serial and with MPI. Before this change a 3D solve on
  the fake CuPy made 3 `to_host` copies (one per direction) and 3 uncounted copies back.

## 6D matrix views (`stencil-6d-views`)

With cunumpy 0.6.1 (`Array5D`/`Array6D`, `CArray5D`/`CArray6D` in `cunumpy/array_view.cuh`) the 3D kernels take
the matrix data as views, like the 1D/2D kernels (`Array2D`/`Array4D`):

- `stencil_dot_3d(Array6D<double> mat, Array3D<double> x, Array3D<double> out, s_in, p_in, add, s_out, e_out,
  p_out)` and `stencil_transpose_3d(Array6D<double> mat, Array6D<double> matT, s_in, p_in, add, s_out, e_out,
  p_out)`; the pyccel versions take `float[:, :, :, :, :, :]` with the same arguments in the same order. Strided
  views (`Array6D`, not `CArray6D`), as in 1D/2D, so a non-contiguous matrix is not refused.
- **No `e_in`.** The transpose kernels (all dimensions, the argument list is the same for all) lost `e_in`: it was
  only needed for the row extents of the raw pointer in 3D.
- **The number of diagonals comes from the data.** All six kernels (1D, 2D, 3D, pyccel and CUDA) read the number
  of diagonals `n_k` of each direction from the shape of the matrix data instead of assuming `2 * p_in + 1`: the
  pads of the matrix are `q_k = (n_k - 1) // 2` (at most the pads of the spaces, `StencilMatrix(V, W, pads=...)`),
  diagonal `d` of row `i` is the column `i - q + d` (in `x` at `i - q + d - s_in + p_in`), and the last owned row
  uses `n_k - 1 + add[k]` diagonals. With `q = p` this is the old loop, in the same order and with bitwise the same
  results (checked against the kernels of #88 run as Python); the pyccel kernels are now one loop nest with the
  number of diagonals picked per row instead of the eight (3D) spelled-out combinations.
- **`StencilMatrix`.** `dot`, `vdot` and `transpose` no longer raise for matrices with fewer diagonals (blocks
  between spaces of different degree, derivative-type stencils); `transpose(out=...)` asserts that `out` has the
  pads of the matrix. The one restriction left is spaces with **shifts > 1**, which still raise
  `NotImplementedError` (`_check_kernel_shifts`): the stencil kernels have never handled them, and there is no
  reference to match, since the general psydac kernels (`matvec_<n>d`, `transpose_<n>d`) disagree with `toarray()`
  and with each other (the transpose is not the adjoint of the product) for shifts > 1, and psydac never tested
  them ("TODO: verify for s>1").
- **Tests.** `MATRIX_CASES` (parity and CPU emulation) has matrices with fewer diagonals and non-periodic
  rectangular blocks between spaces of different size per direction, including pads 0 (one diagonal);
  `test_device_matvec.py` compares `dot`, `vdot` and `transpose` of such matrices with dense `toarray()`
  references, and `test_mpi_device.py` checks a matrix with fewer diagonals and the adjoint identity of its
  transpose against the global field on any number of ranks.

## Kernel folders

```
feectools/linalg/kernels/
├── __init__.py                       # documentation only
├── stencil_dot_3d/
│   ├── __init__.py                   # stencil_dot_3d = Kernel.from_folder(__name__, n_threads_from=...)
│   ├── stencil_dot_3d_kernels.py     # pyccel (compiled by psydac-accelerate)
│   └── stencil_dot_3d_cuda.cu        # CUDA (compiled at runtime by CuPy/NVRTC)
├── stencil_dot_1d/, stencil_dot_2d/, stencil_transpose_<n>d/, stencil_inner_<n>d/, stencil_axpy_<n>d/
└── matvec_kernels.py, transpose_kernels.py, ...   # pyccel kernels without CUDA versions
```

Conventions, as in struphy:

- The folder name, the pyccel function, the `extern "C" __global__` function and the declared `Kernel` have the
  same name. The pyccel file ends in `_kernels.py` (found by `psydac-accelerate`); the CUDA file is `<name>_cuda.cu`.
- Every CUDA function has a `/** ... */` comment naming its pyccel counterpart; parameter and index names are the
  pyccel ones (`mat`, `x`, `out`, `s_in`, ..., `i1_loc`, `d1`).
- Code imports the kernel (`from feectools.linalg.kernels.stencil_dot_3d import stencil_dot_3d`) and calls it with
  positional arguments; `StencilMatrix` and `StencilVectorSpace` select them by dimension from `stencil_kernels`.
- A new folder is picked up by the tests below through `test_cuda_parity.PACKAGES`; a new CUDA kernel also needs
  its parity cases in `linalg/tests/cuda_parity_cases.py` (`test_cuda_kernels_have_parity_cases` checks it).

## Testing

- Every PR runs the serial tests (`pytest feectools -m "not mpi and not petsc"`) and the MPI tests
  (`mpirun -n 2 pytest feectools -m "mpi and not petsc" --with-mpi`) on the NumPy backend.
- `linalg/tests/cuda_parity_cases.py`: `PARITY_CASES[name]` holds the cases of every CUDA kernel, a plain
  `build(case)` that returns its arguments on the active backend, and the tolerances (the same structure as in
  struphy's `pic/tests/cuda_parity_cases.py`).
- `linalg/tests/test_cuda_parity.py`: every folder declares its kernel, the pyccel and CUDA signatures match,
  the CUDA kernels and the parity cases match, and on a GPU `cunumpy.kernel_testing.assert_kernels_agree` runs every
  case (one test per case); `test_solver_loop_has_no_host_transfers` runs `dot` and `axpy` under `assert_no_transfers`.
- `linalg/tests/test_cuda_emulation.py`: without a GPU, every CUDA kernel is compiled as C++ and run thread by
  thread (`cunumpy.kernel_testing.emulate_cuda_kernel`) on the same cases and compared with its pyccel kernel.
  Threads run one after another, so races are not tested; the inner kernels' barriers and shared memory are.
- GPU tests are skipped without CuPy and a GPU (`cunumpy.kernel_testing.requires_cupy`). Until a GPU runner exists,
  they are run by hand on an H100 before a PR that touches CUDA code is merged, and the PR description says so.

## Open questions

- **Setup kernels on the GPU.** B-spline, field evaluation and DOF kernels still run on the host through
  `PyccelKernel` (host copies). They run at setup, not in the time loop; they move into kernel folders with CUDA
  versions when a profile shows they matter.
- **GPU CI.** No GPU runner yet; GPU tests are run by hand.
- **Shifts > 1.** `StencilMatrix.dot`/`transpose` raise for spaces with shifts > 1 (see
  [6D matrix views](#6d-matrix-views-stencil-6d-views)); supporting them needs a verified definition of the data
  layout first.
- **Complex data on the device.** Not needed by struphy so far; would need a second CUDA kernel per folder (or
  dtype dispatch in `cunumpy.kernels.Kernel`).
- **`inner` reduction.** One `atomicAdd` per block of 256 threads, then a copy of the 8-byte result to the host in
  the serial case (in the parallel case it goes into the MPI reduction). To be measured on the H100.
- **Interface matrices** (`StencilInterfaceMatrix`) and the remaining stencil kernels (`stencil2coo`, ...) still use
  `PyccelKernel` with host copies.
- **FFT on the device.** `DistributedFFT` and friends still stage their data through the host (SciPy FFT); they
  could use `cupyx.scipy.fft` for device data.
- **Large 1D matrices.** The dense inverse costs n^2 memory and O(m n^2) per pass. Fine for the mass matrices of
  struphy's preconditioners; for 1D sizes in the thousands a batched banded solve on the device would be cheaper.

## MPI through maybempi

`feectools.ddm.mpi` is gone. Every module imports `from maybempi import MPI`: [maybempi](https://github.com/max-models/maybempi)
returns `mpi4py.MPI` when the process was started by an MPI launcher and a serial stand-in otherwise, without
importing mpi4py in serial runs. `maybempi.is_serial(MPI)` replaces the `isinstance(MPI, MockMPI)` checks. The
serial stand-in behaves like MPI on one rank (`bcast` returns the object, `Gather`/`Allreduce` copy), where the old
`MockMPI`/`MockComm` turned every call into a no-op returning `None`. The override variable is `MAYBEMPI=0`/`1`;
`FEECTOOLS_MPI`, `STRUPHY_MPI` and `feectools.use_mpi` are no longer read. cunumpy (since 0.6) uses the same package,
so feectools, cunumpy and struphy make the same serial/MPI decision.
