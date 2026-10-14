#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from abc import ABC, abstractmethod
from functools import reduce

import cunumpy as xp
import numpy as np
from cunumpy.mpi import exchange, get_mpi_cuda_aware, synchronize_for_mpi
from maybempi import MPI


__all__ = ('CartDataExchanger',)

#===============================================================================
class _DeviceRegionExchange:
    """
    Copy regions of a device (CuPy) array between ranks without MPI derived datatypes.

    Open MPI's CUDA-aware path moves a subarray datatype on device memory one contiguous
    piece at a time (~15 us per piece on H100 nodes), so a strided ghost region of a few
    hundred bytes can take milliseconds. Here all regions are gathered on the device into
    one contiguous buffer with a single kernel (flat indices computed once), the regions
    going to the same rank travel as one message, a region that a rank sends to itself
    (periodic direction that is not decomposed) is copied without MPI, and one more kernel
    scatters everything received into the array.

    All regions are read (gathered) before any is written, as in a nonblocking MPI exchange.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Communicator of the ranks in `transfers`.

    transfers : list of (send_idx, dest, recv_idx, source)
        Send the region `send_idx` of the array to rank `dest` and receive the region
        `recv_idx` from rank `source` (MPI.PROC_NULL: no message). Every rank lists the
        transfers in the same order: the regions exchanged between two ranks are matched
        by that order.

    shape, dtype :
        Shape and dtype of the (C-contiguous) arrays the exchange is used for.
    """
    TAG = 4242

    def __init__(self, comm, transfers, shape, dtype):

        me = comm.Get_rank()
        self._comm = comm
        # C-order element strides: the flat indices of a region are built from per-axis ranges, so the work and
        # memory scale with the region, not with the whole array (a stencil matrix block has ~10^8 entries)
        strides = [1] * len(shape)
        for axis in range(len(shape) - 2, -1, -1):
            strides[axis] = strides[axis + 1] * int(shape[axis + 1])

        def flat(idx):
            """Flat (C-order) indices of the region `idx` of the array, in C order of the region."""
            axes = [np.arange(*sl.indices(int(n)), dtype=np.int64) * st for sl, n, st in zip(idx, shape, strides)]
            return reduce(np.add, np.ix_(*axes)).ravel()

        local_send, local_recv, sends, recvs = [], [], {}, {}
        for send_idx, dest, recv_idx, source in transfers:
            if dest == me and source == me:
                local_send.append(flat(send_idx))
                local_recv.append(flat(recv_idx))
                continue
            if dest != MPI.PROC_NULL:
                sends.setdefault(dest, []).append(flat(send_idx))
            if source != MPI.PROC_NULL:
                recvs.setdefault(source, []).append(flat(recv_idx))

        def layout(groups, local):
            """Flat indices of [message to/from each rank..., local regions], and each message's slice."""
            parts, slices, offset = [], [], 0
            for rank, regions in groups.items():
                n = sum(r.size for r in regions)
                parts += regions
                slices.append((rank, slice(offset, offset + n)))
                offset += n
            local_slice = slice(offset, offset + sum(r.size for r in local))
            parts += local
            index = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
            return xp.asarray(index), slices, local_slice

        self._send_index, send_slices, self._local_send = layout(sends, local_send)
        self._recv_index, recv_slices, self._local_recv = layout(recvs, local_recv)
        self._send_buf = xp.empty(self._send_index.size, dtype=dtype)
        self._recv_buf = xp.empty(self._recv_index.size, dtype=dtype)
        self._sends = [(self._send_buf[sl], rank) for rank, sl in send_slices]
        self._recvs = [(self._recv_buf[sl], rank) for rank, sl in recv_slices]

    def start(self, array):
        """Gather the regions to send and post the messages; returns the requests for :meth:`finish`.

        With CUDA-aware MPI the device buffers go to MPI directly after one stream synchronization
        (which also orders the receives after the previous scatter from the receive buffer), and
        the transfers proceed until :meth:`finish`: several exchanges (e.g. the components of a
        block vector) can be in flight at once. Otherwise the buffers are staged through the host
        and the exchange completes here.
        """
        assert array.flags.c_contiguous, "device ghost exchange needs a C-contiguous array"
        values = array.reshape(-1)

        # gather every region to send (or copy locally) with one kernel
        if self._send_index.size:
            xp.take(values, self._send_index, out=self._send_buf)

        if not (self._sends or self._recvs):
            return []

        # device buffers straight to MPI if it is CUDA-aware (assumed when unknown, as for the
        # datatype exchange this replaces), staged through the host otherwise
        cuda_aware = get_mpi_cuda_aware()
        if cuda_aware is False:
            exchange(
                self._comm,
                sends=[(buf, rank, self.TAG) for buf, rank in self._sends],
                receives=[(buf, rank, self.TAG) for buf, rank in self._recvs],
                cuda_aware=False,
            )
            return []

        synchronize_for_mpi(self._send_buf)
        requests = [self._comm.Irecv(buf, source=rank, tag=self.TAG) for buf, rank in self._recvs]
        requests += [self._comm.Isend(buf, dest=rank, tag=self.TAG) for buf, rank in self._sends]
        return requests

    def finish(self, array, requests):
        """Wait for the messages of :meth:`start` and scatter everything received (and the local copies)."""
        if requests:
            MPI.Request.Waitall(requests)

        values = array.reshape(-1)
        # scatter everything received (and the local copies) with one kernel
        if self._recv_index.size:
            if self._local_recv.stop > self._local_recv.start:
                self._recv_buf[self._local_recv] = self._send_buf[self._local_send]
            values[self._recv_index] = self._recv_buf

    def __call__(self, array):
        self.finish(array, self.start(array))


#===============================================================================
class CartDataExchanger(ABC):
    """
    Type that takes care of updating the ghost regions (padding) of a
    multi-dimensional array distributed according to the given Cartesian
    decomposition of a tensor-product grid of coefficients.

    Each coefficient in the decomposed grid may have multiple components,
    contiguous in memory.

    Parameters
    ----------
    cart : feectools.ddm.CartDecomposition
        Object that contains all information about the Cartesian decomposition
        of a tensor-product grid of coefficients.

    dtype : [type | str | numpy.dtype | mpi4py.MPI.Datatype]
        Datatype of single coefficient (if scalar) or of each of its
        components (if vector).

    coeff_shape : [tuple(int) | list(int)]
        Shape of a single coefficient, if this is multi-dimensional
        (optional: by default, we assume scalar coefficients).

    """

    #---------------------------------------------------------------------------
    # Public interface
    #---------------------------------------------------------------------------

    @abstractmethod
    def prepare_communications(self, u):
        pass

    @abstractmethod
    def start_update_ghost_regions( self, array, requests ):
        """
        Update ghost regions in a numpy array with dimensions compatible with
        CartDecomposition (and coeff_shape) provided at initialization.

        Parameters
        ----------
        array : numpy.ndarray
            Multidimensional array corresponding to local subdomain in
            decomposed tensor grid, including padding.

        requests : tuple|None
            The requests of the communications.

        """

    @abstractmethod
    def end_update_ghost_regions( self, array, requests ):
        pass

    @abstractmethod
    def start_exchange_assembly_data( self, array ):
        """
        Update ghost regions after the assembly algorithm in a numpy array
        with dimensions compatible with CartDecomposition (and coeff_shape)
        provided at initialization.

        Parameters
        ----------
        array : numpy.ndarray
            Multidimensional array corresponding to local subdomain in
            decomposed tensor grid, including padding.
        """

    @abstractmethod
    def end_exchange_assembly_data( self, array ):
        pass

    #---------------------------------------------------------------------------
    # Device (CuPy) arrays: contiguous buffers instead of MPI derived datatypes
    #---------------------------------------------------------------------------
    @staticmethod
    def _region(starts, buf_shape, array_shape):
        """Index of the region `buf_shape` at `starts` of the decomposed axes, whole along the coefficient axes."""
        idx = [slice(int(s), int(s) + int(b)) for s, b in zip(starts, buf_shape)]
        idx += [slice(0, int(n)) for n in array_shape[len(idx):]]
        return tuple(idx)

    def _device_plan(self, key, array, comm, transfers):
        """The `_DeviceRegionExchange` of `transfers` for `array` (built by the callable on first use)."""
        plans = self.__dict__.setdefault('_device_plans', {})
        plan_key = (key, tuple(array.shape), np.dtype(array.dtype))
        plan = plans.get(plan_key)
        if plan is None:
            plan = plans[plan_key] = _DeviceRegionExchange(comm, transfers(), array.shape, array.dtype)
        return plan

    def _device_exchange(self, key, array, comm, transfers):
        """Run the `_DeviceRegionExchange` of `transfers` (built by the callable on first use) on `array`."""
        self._device_plan(key, array, comm, transfers)(array)

    def _device_update_ghost_regions_along(self, array, direction):
        """Ghost regions of `direction` (both sides) from the neighbours, as the blocking exchange."""
        cart = self._cart

        def transfers():
            out = []
            for disp in [-1, 1]:
                info = cart.get_shift_info(direction, disp)
                out.append((self._region(info['send_starts'], info['buf_shape'], array.shape), info['rank_dest'],
                            self._region(info['recv_starts'], info['buf_shape'], array.shape), info['rank_source']))
            return out

        self._device_exchange(('ghost', direction), array, self._comm, transfers)

    def _device_exchange_assembly_data(self, array):
        """`start_exchange_assembly_data` for a device array: same transfers and sums, contiguous buffers."""
        cart  = self._cart
        gcomm = self._comm
        ndim  = cart.ndim
        axis  = self._axis

        for direction in range(ndim):
            if direction == axis:
                continue
            comm = cart.subcomm[direction] if axis is not None else gcomm
            disp = 1
            info = cart.get_shift_info(direction, disp)

            def transfers(info=info, comm=comm):
                rank_source, rank_dest = info['rank_source'], info['rank_dest']
                if axis is not None:
                    rank_source = gcomm.group.Translate_ranks([int(rank_source)], comm.group)[0]
                    rank_dest   = gcomm.group.Translate_ranks([int(rank_dest)], comm.group)[0]
                buf_shape   = list(info['buf_shape'])
                send_starts = list(info['send_assembly_starts'])
                recv_starts = list(info['recv_assembly_starts'])
                if axis is not None:
                    buf_shape[axis]   = self._shape[axis]
                    send_starts[axis] = 0
                    recv_starts[axis] = 0
                return [(self._region(send_starts, buf_shape, array.shape), rank_dest,
                         self._region(recv_starts, buf_shape, array.shape), rank_source)]

            self._device_exchange(('assembly', direction), array, comm, transfers)
            self._add_assembly_ghosts(array, direction)

    def _add_assembly_ghosts(self, array, direction):
        """Add the received assembly data of `direction` (disp = 1) to the owned entries."""
        cart = self._cart
        ndim = cart.ndim
        info = cart.get_shift_info(direction, 1)
        pads = [0]*ndim
        pads[direction] = cart._pads[direction]*cart._shifts[direction]
        idx_from = tuple(slice(s,s+b) for s,b in zip(info['recv_starts'],info['buf_shape']))
        idx_to   = tuple(slice(s+p,s+b+p) for s,b,p in zip(info['recv_starts'],info['buf_shape'],pads))
        array[idx_to] += array[idx_from]

