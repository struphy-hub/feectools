# Bind this rank to its own GPU (by its rank within the node) and create the CUDA context before MPI starts, as
# CUDA-aware MPI requires. Every feectools module imports this package first, and MPI starts only when a module
# imports `from maybempi import MPI` (maybempi uses mpi4py under an MPI launcher and a serial stand-in otherwise;
# MAYBEMPI=0/1 overrides the detection). A no-op on the NumPy backend.
from cunumpy.cuda import bind_local_device

bind_local_device()
