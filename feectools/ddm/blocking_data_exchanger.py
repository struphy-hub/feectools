# coding: utf-8

import cunumpy as xp
import numpy as np
from cunumpy.xp import array_backend
from feectools.ddm.mpi import mpi as MPI

from .cart import CartDecomposition, find_mpi_type
from .device import synchronize_for_mpi
from .basic import CartDataExchanger


__all__ = ('BlockingCartDataExchanger',)


def _boundary_slice( array_ndim, direction, starts, buf_shape ):
    """Index tuple selecting one ghost-exchange boundary slab of `array`.

    `starts`/`buf_shape` (from `CartDecomposition.get_shift_info`) already
    cover the full extent on every cart axis except `direction` -- see
    `CartDecomposition._compute_shift_info` -- so no special-casing of
    `direction` is needed here. Any array axes beyond the cart's own
    (component axes, from `coeff_shape`) are taken in full, matching how
    `_create_buffer_types` appends them to the MPI subarray datatype.
    """
    cart_ndim = len( starts )
    slc = [ slice( s, s+b ) for s, b in zip( starts, buf_shape ) ]
    slc += [ slice( None ) ] * ( array_ndim - cart_ndim )
    return tuple( slc )


class BlockingCartDataExchanger(CartDataExchanger):
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
    def __init__( self, cart, dtype, *, coeff_shape=(),  assembly=False, axis=None, shape=None ):

        self._send_types, self._recv_types = self._create_buffer_types(
                cart, dtype, coeff_shape=coeff_shape )

        self._cart = cart
        self._comm = cart.comm_cart
        self._axis = axis

        if assembly:
            self._assembly_send_types, self._assembly_recv_types = self._create_assembly_buffer_types(
                cart, dtype, coeff_shape=coeff_shape, axis=axis, shape=shape)

        # Contiguous device buffers for the GPU ghost-exchange path (see
        # start_update_ghost_regions), pre-allocated once and reused on every
        # call. Allocating fresh (xp.empty_like/xp.ascontiguousarray) on every
        # exchange gives MPI/UCX a never-before-seen device address each time,
        # so its memory-registration cache can never warm up and every call
        # pays full pinning/registration latency -- measured at ~5ms/call
        # against a raw ping-pong latency of ~30us for the same cluster.
        self._gpu_send_bufs = {}
        self._gpu_recv_bufs = {}
        if array_backend.backend == 'cupy':
            coeff_shape_t = tuple( coeff_shape )
            for direction in range( cart.ndim ):
                for disp in (-1, 1):
                    info = cart.get_shift_info( direction, disp )
                    buf_shape = tuple( info['buf_shape'] ) + coeff_shape_t
                    self._gpu_send_bufs[direction, disp] = xp.empty( buf_shape, dtype=dtype )
                    self._gpu_recv_bufs[direction, disp] = xp.empty( buf_shape, dtype=dtype )

        # Pooled buffers for start_exchange_assembly_data (used during
        # matrix/vector assembly). The recv derived-datatype in the original
        # code writes straight into `array`, and for a PROC_NULL direction
        # (non-periodic boundary) the accumulation step deliberately reads
        # back whatever an *earlier* direction in the same call already wrote
        # into that overlapping region -- how corner contributions are
        # chained across the 2D+ direction sweep (verified: an earlier
        # attempt that read straight from an isolated recv buffer instead of
        # writing it into `array` first gave a wrong, smaller result on a
        # non-periodic axis). The fix below still writes the received data
        # into `array[idx_from]` -- skipped on PROC_NULL, exactly like the
        # original derived-datatype recv being a no-op there -- so that
        # dependency is preserved; only the slow non-contiguous-datatype
        # transfer itself is replaced. Scoped to axis is None (the ordinary,
        # non-multipatch/interface case); that case keeps the original path.
        self._gpu_assembly_send_bufs   = {}
        self._gpu_assembly_recv_bufs   = {}
        self._gpu_assembly_send_slices = {}
        self._gpu_assembly_recv_slices = {}
        if assembly and array_backend.backend == 'cupy' and axis is None:
            coeff_shape_t = tuple( coeff_shape )
            array_ndim = cart.ndim + len( coeff_shape_t )
            for direction in range( cart.ndim ):
                info = cart.get_shift_info( direction, 1 )
                buf_shape = tuple( info['buf_shape'] ) + coeff_shape_t
                self._gpu_assembly_send_bufs[direction]   = xp.empty( buf_shape, dtype=dtype )
                self._gpu_assembly_recv_bufs[direction]   = xp.empty( buf_shape, dtype=dtype )
                self._gpu_assembly_send_slices[direction] = _boundary_slice(
                    array_ndim, None, info['send_assembly_starts'], info['buf_shape'] )
                self._gpu_assembly_recv_slices[direction] = _boundary_slice(
                    array_ndim, None, info['recv_starts'], info['buf_shape'] )

    #---------------------------------------------------------------------------
    # Public interface
    #---------------------------------------------------------------------------
    def get_send_type( self, *args ):
        direction = args[0]
        disp      = args[1]
        return self._send_types[direction, disp]

    # ...
    def get_recv_type( self, *args ):
        direction = args[0]
        disp      = args[1]
        return self._recv_types[direction, disp]

    # ...
    def get_assembly_send_type( self,*args ):
        direction = args[0]
        disp      = args[1]
        return self._assembly_send_types[direction, disp]

    # ...
    def get_assembly_recv_type( self, *args ):
        direction = args[0]
        disp      = args[1]
        return self._assembly_recv_types[direction, disp]

    # ...
    def prepare_communications(self, u):
        pass

    # ...
    def start_update_ghost_regions( self, array, requests ):

        assert isinstance( array, xp.ndarray )

        # MPI reads/writes `array` directly; on a device backend the
        # kernels that produced it must have finished first.
        synchronize_for_mpi( array )

        # Shortcuts
        cart = self._cart
        comm = self._comm

        # Choose non-negative invertible function tag(disp) >= 0
        # NOTES:
        #   . different values of disp must return different tags!
        #   . tag at receiver must match message tag at sender
        tag = lambda disp: 42+disp

        if xp.is_gpu( array ):
            # Not all CUDA-aware MPI/UCX builds have an efficient GPU-side pack
            # kernel for derived (non-contiguous) datatypes -- some fall back to
            # a slow, per-block loop whose cost scales with the boundary length
            # instead of one bulk transfer. Pack/unpack explicitly into
            # contiguous device buffers instead, so only a single contiguous
            # transfer per direction/disp crosses MPI. The host (NumPy) path
            # below is unaffected: MPI derived datatypes over host memory are
            # not subject to this.
            for direction in range( self._cart.ndim ):
                requests   = []
                recv_slots = []

                # Start receiving data (MPI_IRECV) into a pre-allocated,
                # reused contiguous buffer (see __init__). A PROC_NULL source
                # (non-periodic boundary) is a genuine no-op in MPI: nothing
                # is written, and the array's existing ghost values must be
                # left untouched -- so no slot is created for it, and it is
                # skipped again below when unpacking.
                for disp in [-1,1]:
                    info = cart.get_shift_info( direction, disp )
                    if info['rank_source'] == MPI.PROC_NULL:
                        continue
                    recv_slc  = _boundary_slice( array.ndim, direction, info['recv_starts'], info['buf_shape'] )
                    recv_buf  = self._gpu_recv_bufs[direction, disp]
                    recv_slots.append( (recv_slc, recv_buf) )
                    requests.append( comm.Irecv( recv_buf, info['rank_source'], tag(disp) ) )

                # Pack into a pre-allocated, reused contiguous buffer and
                # start sending it (MPI_ISEND).
                for disp in [-1,1]:
                    info = cart.get_shift_info( direction, disp )
                    if info['rank_dest'] == MPI.PROC_NULL:
                        continue
                    send_slc  = _boundary_slice( array.ndim, direction, info['send_starts'], info['buf_shape'] )
                    send_buf  = self._gpu_send_bufs[direction, disp]
                    send_buf[...] = array[ send_slc ]
                    requests.append( comm.Isend( send_buf, info['rank_dest'], tag(disp) ) )

                # Wait for end of data exchange (MPI_WAITALL), then unpack.
                MPI.Request.Waitall( requests )

                for recv_slc, recv_buf in recv_slots:
                    array[ recv_slc ] = recv_buf
            return

        for direction in range( self._cart.ndim ):
            # Requests' handles
            requests = []

            # Start receiving data (MPI_IRECV)
            for disp in [-1,1]:
                info     = cart.get_shift_info( direction, disp )
                recv_typ = self.get_recv_type ( direction, disp )
                recv_buf = (array, 1, recv_typ)
                recv_req = comm.Irecv( recv_buf, info['rank_source'], tag(disp) )
                requests.append( recv_req )

            # Start sending data (MPI_ISEND)
            for disp in [-1,1]:
                info     = cart.get_shift_info( direction, disp )
                send_typ = self.get_send_type ( direction, disp )
                send_buf = (array, 1, send_typ)
                send_req = comm.Isend( send_buf, info['rank_dest'], tag(disp) )
                requests.append( send_req )

            # Wait for end of data exchange (MPI_WAITALL)
            MPI.Request.Waitall( requests )

    def end_update_ghost_regions(self,  array, requests ):
        pass

    # ...
    def start_exchange_assembly_data( self, array ):

        assert isinstance( array, xp.ndarray )

        if self._cart.single_process:
            # Every neighbour is this process; see
            # CartDataExchanger._local_exchange_assembly_data.
            self._local_exchange_assembly_data( array )
            return

        synchronize_for_mpi( array )

        # Shortcuts
        cart  = self._cart
        comm  = self._comm
        gcomm = comm
        ndim  = cart.ndim

        # Choose non-negative invertible function tag(disp) >= 0
        # NOTES:
        #   . different values of disp must return different tags!
        #   . tag at receiver must match message tag at sender
        tag = lambda disp: 42+disp

        if xp.is_gpu( array ) and self._axis is None and self._gpu_assembly_send_bufs:
            for direction in range( ndim ):
                info = cart.get_shift_info( direction, 1 )

                send_req = None
                if info['rank_dest'] >= 0:
                    send_buf = self._gpu_assembly_send_bufs[direction]
                    send_buf[...] = array[ self._gpu_assembly_send_slices[direction] ]
                    send_req = comm.Isend( send_buf, info['rank_dest'], tag(1) )

                recv_req = None
                if info['rank_source'] >= 0:
                    recv_buf = self._gpu_assembly_recv_bufs[direction]
                    recv_req = comm.Irecv( recv_buf, info['rank_source'], tag(1) )

                reqs = [r for r in (recv_req, send_req) if r is not None]
                if reqs:
                    MPI.Request.Waitall( reqs )

                # Mimic the original derived-datatype recv's effect (a no-op,
                # leaving `array` untouched, when rank_source is PROC_NULL --
                # see the __init__ comment on why this matters).
                if info['rank_source'] >= 0:
                    array[ self._gpu_assembly_recv_slices[direction] ] = recv_buf

                pads = [0]*ndim
                pads[direction] = cart._pads[direction]*cart._shifts[direction]
                idx_from = tuple(slice(s,s+b) for s,b in zip(info['recv_starts'],info['buf_shape']))
                idx_to   = tuple(slice(s+p,s+b+p) for s,b,p in zip(info['recv_starts'],info['buf_shape'],pads))
                array[idx_to] += array[idx_from]
            return

        # Requests' handles

        for direction in range( ndim ):
            if direction == self._axis: continue
            if self._axis is not None: comm = cart.subcomm[direction]

            # Start receiving data (MPI_IRECV)
            disp        = 1
            info        = cart.get_shift_info( direction, disp )
            recv_typ    = self.get_assembly_recv_type ( direction, disp )
            rank_source = info['rank_source']

            if self._axis is not None:
                rank_source = gcomm.group.Translate_ranks(np.array([rank_source]), comm.group)[0]
            
            recv_buf = (array, 1, recv_typ)
            recv_req = comm.Irecv( recv_buf, rank_source, tag(disp) )

            # Start sending data (MPI_ISEND)
            send_typ = self.get_assembly_send_type ( direction, disp )
            rank_dest = info['rank_dest']

            if self._axis is not None:
                rank_dest = gcomm.group.Translate_ranks(xp.array([rank_dest]), comm.group)[0]

            send_buf = (array, 1, send_typ)
            send_req = comm.Isend( send_buf, rank_dest, tag(disp) )

            # Wait for end of data exchange (MPI_WAITALL)
            MPI.Request.Waitall( [recv_req, send_req] )

            if disp == 1:
                info = cart.get_shift_info( direction, disp )
                pads = [0]*ndim
                pads[direction] = cart._pads[direction]*cart._shifts[direction]
                idx_from = tuple(slice(s,s+b) for s,b in zip(info['recv_starts'],info['buf_shape']))
                idx_to   = tuple(slice(s+p,s+b+p) for s,b,p in zip(info['recv_starts'],info['buf_shape'],pads))
                array[idx_to] += array[idx_from]
            else:
                info = cart.get_shift_info( direction, disp )
                pads = [0]*ndim
                pads[direction] = cart._pads[direction]*cart._shifts[direction]
                idx_from = tuple(slice(s,s+b) for s,b in zip(info['recv_starts'],info['buf_shape']))
                idx_to   = tuple(slice(s-p,s+b-p) for s,b,p in zip(info['recv_starts'],info['buf_shape'],pads))
                array[idx_to] += array[idx_from]

    def end_exchange_assembly_data( self, array ):
        pass

    #---------------------------------------------------------------------------
    # Private methods
    #---------------------------------------------------------------------------
    @staticmethod
    def _create_buffer_types( cart, dtype, *, coeff_shape=() ):
        """
        Create MPI subarray datatypes for updating the ghost regions (padding)
        of a multi-dimensional array distributed according to the given Cartesian
        decomposition of a tensor-product grid of coefficients.

        MPI requires a subarray datatype for accessing non-contiguous slices of
        a multi-dimensional array; this is a typical situation when updating the
        ghost regions.

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
            Shape of a single coefficient, if this is multidimensional
            (optional: by default, we assume scalar coefficients).

        Returns
        -------
        send_types : dict
            Dictionary of MPI subarray datatypes for SEND BUFFERS, accessed
            through the integer pair (direction, displacement) as key;
            'direction' takes values from 0 to ndim, 'disp' is -1 or +1.

        recv_types : dict
            Dictionary of MPI subarray datatypes for RECEIVE BUFFERS, accessed
            through the integer pair (direction, displacement) as key;
            'direction' takes values from 0 to ndim, 'disp' is -1 or +1.

        """
        assert isinstance( cart, CartDecomposition )

        mpi_type = find_mpi_type( dtype )

        # Possibly, each coefficient could have multiple components
        coeff_shape = list( coeff_shape )
        coeff_start = [0] * len( coeff_shape )

        data_shape = list( cart.shape ) + coeff_shape
        send_types = {}
        recv_types = {}

        for direction in range( cart.ndim ):
            for disp in [-1, 1]:
                info = cart.get_shift_info( direction, disp )

                buf_shape   = list( info[ 'buf_shape' ] ) + coeff_shape
                send_starts = list( info['send_starts'] ) + coeff_start
                recv_starts = list( info['recv_starts'] ) + coeff_start

                send_types[direction,disp] = mpi_type.Create_subarray(
                    sizes    = [int(x) for x in data_shape] ,
                    subsizes =  [int(x) for x in buf_shape] ,
                    starts   = [int(x) for x in send_starts],
                ).Commit()

                recv_types[direction,disp] = mpi_type.Create_subarray(
                    sizes    = [int(x) for x in data_shape] ,
                    subsizes =  [int(x) for x in buf_shape] ,
                    starts   = [int(x) for x in recv_starts],
                ).Commit()

        return send_types, recv_types

    # ...
    @staticmethod
    def _create_assembly_buffer_types( cart, dtype, *, coeff_shape=(), axis=None, shape=None ):
        """
        Create MPI subarray datatypes for updating the ghost regions (padding)
        of a multi-dimensional array distributed according to the given Cartesian
        decomposition of a tensor-product grid of coefficients.
        MPI requires a subarray datatype for accessing non-contiguous slices of
        a multi-dimensional array; this is a typical situation when updating the
        ghost regions.
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
            Shape of a single coefficient, if this is multidimensional
            (optional: by default, we assume scalar coefficients).

        axis: int,optional
           The axis of which we don't update the ghost regions.

        shape:
            the shape of data when axis is not None
        
        Returns
        -------
        send_types : dict
            Dictionary of MPI subarray datatypes for SEND BUFFERS, accessed
            through the integer pair (direction, displacement) as key;
            'direction' takes values from 0 to ndim, 'disp' is -1 or +1.

        recv_types : dict
            Dictionary of MPI subarray datatypes for RECEIVE BUFFERS, accessed
            through the integer pair (direction, displacement) as key;
            'direction' takes values from 0 to ndim, 'disp' is -1 or +1.
        """
        assert isinstance( cart, CartDecomposition )

        mpi_type = find_mpi_type( dtype )

        # Possibly, each coefficient could have multiple components
        coeff_shape = list( coeff_shape )
        coeff_start = [0] * len( coeff_shape )

        data_shape = list( cart.shape ) + coeff_shape
        send_types = {}
        recv_types = {}

        if axis is not None:
            data_shape[axis] = shape[axis]

        for direction in range( cart.ndim ):
            for disp in [-1, 1]:
                info = cart.get_shift_info( direction, disp )

                buf_shape   = list( info[ 'buf_shape' ] ) + coeff_shape
                send_starts = list( info['send_assembly_starts'] ) + coeff_start
                recv_starts = list( info['recv_assembly_starts'] ) + coeff_start
                if direction == axis:continue
                if axis is not None:
                    buf_shape[axis]   = shape[axis]
                    send_starts[axis] = 0
                    recv_starts[axis] = 0

                send_types[direction,disp] = mpi_type.Create_subarray(
                    sizes    = [int(x) for x in data_shape] ,
                    subsizes =  [int(x) for x in buf_shape] ,
                    starts   = [int(x) for x in send_starts],
                ).Commit()

                recv_types[direction,disp] = mpi_type.Create_subarray(
                    sizes    = [int(x) for x in data_shape] ,
                    subsizes =  [int(x) for x in buf_shape] ,
                    starts   = [int(x) for x in recv_starts],
                ).Commit()

        return send_types, recv_types
