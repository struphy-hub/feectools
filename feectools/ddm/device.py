"""
Binding of MPI processes to GPUs.

Kept free of any MPI import on purpose: the CUDA context should exist before
``MPI_Init`` runs, and importing :mod:`feectools.ddm.mpi` initialises MPI as a
side effect. The rank of the process within its node is therefore taken from
the environment variables the launcher sets, which are available before
``MPI_Init``, rather than from a communicator.
"""

import os

import cunumpy as xp
from cunumpy.xp import array_backend

__all__ = ('local_rank', 'bind_local_device', 'synchronize_for_mpi')

# Node-local rank, as exported by the common launchers.
_LOCAL_RANK_VARS = (
    'OMPI_COMM_WORLD_LOCAL_RANK',   # Open MPI
    'MV2_COMM_WORLD_LOCAL_RANK',    # MVAPICH2
    'MPI_LOCALRANKID',              # Intel MPI
    'PMI_LOCAL_RANK',               # MPICH / PMI
    'SLURM_LOCALID',                # Slurm
)


def synchronize_for_mpi(*arrays):
    """
    Wait for pending device work before MPI touches `arrays`.

    CuPy launches kernels asynchronously on the current stream; MPI knows
    nothing about that stream. Handing it a device buffer that a kernel is
    still writing lets it send whatever happens to be in memory at that
    moment, which shows up as silently wrong ghost regions rather than as an
    error. Every MPI call that reads or writes device memory must therefore be
    preceded by this.

    The reverse direction needs no barrier: MPI completes its own transfers
    before the corresponding wait returns, so kernels launched afterwards see
    the received data.

    Parameters
    ----------
    *arrays : array | None
        The buffers about to be given to MPI. Synchronization happens only if
        at least one of them lives on a device, so host-only exchanges (and the
        whole NumPy backend) pay nothing.
    """
    if not any(xp.is_gpu(a) for a in arrays if a is not None):
        return

    import cupy as cp

    cp.cuda.get_current_stream().synchronize()


def local_rank():
    """
    The rank of this process within its node, or 0 if no launcher told us
    (which is the right answer for a serial run).
    """
    for var in _LOCAL_RANK_VARS:
        value = os.environ.get(var)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


def bind_local_device():
    """
    Bind this process to one GPU, chosen round-robin by its node-local rank, and
    create the CUDA context.

    Without this every rank on a node would share GPU 0: they would contend for
    one device while the others idled, and one device's memory would have to
    hold every rank's data. `CUDA_VISIBLE_DEVICES` still applies first, so a
    launcher that already hands each rank its own device keeps working (each
    process then sees a single device and picks index 0).

    Returns
    -------
    int | None
        The index of the device that was selected, or None if the CuPy backend
        is not active or no device is available.
    """
    if array_backend.backend != 'cupy':
        return None

    try:
        import cupy as cp

        count = cp.cuda.runtime.getDeviceCount()
        if count == 0:
            return None

        device = local_rank() % count
        cp.cuda.Device(device).use()
        cp.cuda.Stream.null.synchronize()
        return device
    except Exception:  # noqa: BLE001 - a driver/runtime failure must not be fatal
        return None
