# coding: utf-8

import cunumpy as xp
import numpy as np
from cunumpy.xp import array_backend
from itertools import product

from feectools.ddm.mpi import mpi as MPI
from .cart import CartDecomposition, find_mpi_type
from .device import synchronize_for_mpi
from .basic import CartDataExchanger
from .blocking_data_exchanger import _boundary_slice

__all__ = ('NonBlockingCartDataExchanger',)

class NonBlockingCartDataExchanger(CartDataExchanger):
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
    # Tag for the coalesced per-peer ghost messages. A single tag is enough:
    # there is exactly one group per peer, and start/end_update_ghost_regions
    # are called in matched pairs so only one exchange per exchanger is ever
    # in flight.
    _GROUP_TAG = 7718

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
        # prepare_communications / start_update_ghost_regions /
        # end_update_ghost_regions). As in BlockingCartDataExchanger, MPI
        # derived (non-contiguous) datatypes bound directly to device memory
        # are a known slow path on this cluster's CUDA-aware MPI/UCX build;
        # pack/unpack into pre-allocated, reused contiguous buffers instead.
        self._gpu_send_bufs    = {}
        self._gpu_recv_bufs    = {}
        self._gpu_send_slices  = {}
        self._gpu_recv_slices  = {}
        self._gpu_full_shape   = None
        self._gpu_send_flat    = None
        self._gpu_recv_flat    = None
        self._gpu_send_idx     = None
        self._gpu_recv_idx     = None
        self._gpu_send_groups  = None
        self._gpu_recv_groups  = None
        if array_backend.backend == 'cupy':
            coeff_shape_t = tuple( coeff_shape )
            array_ndim = cart.ndim + len( coeff_shape_t )
            for shift in product( [-1,0,1], repeat=cart.ndim ):
                if all( s == 0 for s in shift ):
                    continue
                info = cart.get_shift_info_non_blocking( shift )
                buf_shape = tuple( info['buf_shape'] ) + coeff_shape_t
                if info['rank_dest'] >= 0:
                    self._gpu_send_bufs[shift]   = xp.empty( buf_shape, dtype=dtype )
                    self._gpu_send_slices[shift] = _boundary_slice( array_ndim, None, info['send_starts'], info['buf_shape'] )
                if info['rank_source'] >= 0:
                    self._gpu_recv_bufs[shift]   = xp.empty( buf_shape, dtype=dtype )
                    self._gpu_recv_slices[shift] = _boundary_slice( array_ndim, None, info['recv_starts'], info['buf_shape'] )

            # A 3D cart has 26 shift directions, so the per-shift loops above
            # issue 26 pack + 26 unpack device copies per ghost update. Each
            # slab is only a couple of kB, so that loop is entirely dominated
            # by per-call launch overhead rather than by data movement.
            #
            # Lay every per-shift buffer out as a view into one flat buffer and
            # precompute the flat source/destination indices, so the whole pack
            # is a single gather and the whole unpack a single scatter.
            # `_fuse_layout` returns None if the layout is unusable (e.g. the
            # ghost slabs are not disjoint), in which case the per-shift loops
            # remain in use.
            self._gpu_full_shape = tuple( cart.shape ) + coeff_shape_t
            fused = self._fuse_layout( dtype )
            if fused is not None:
                ( self._gpu_send_flat, self._gpu_send_idx,
                  self._gpu_recv_flat, self._gpu_recv_idx ) = fused

    def _fuse_layout( self, dtype ):
        """
        Repoint the per-shift GPU buffers at slices of a single flat buffer and
        build the flat gather/scatter index arrays that pack/unpack them in one
        device call each.

        Returns
        -------
        tuple or None
            ``(send_flat, send_idx, recv_flat, recv_idx)``, or None if the
            fused layout cannot be used and the per-shift path must be kept.
        """
        full_shape = self._gpu_full_shape

        def flat_indices( slices ):
            ranges = [ np.arange( *s.indices( n ) )
                       for s, n in zip( slices, full_shape ) ]
            return np.ravel_multi_index( np.ix_( *ranges ), full_shape ).ravel()

        cart = self._cart

        # Many shift directions share a peer: on a 2x2x1 process grid all 26
        # directions resolve to only 4 distinct ranks, so the per-shift loop
        # posts 52 requests to exchange a few kB. Group the shifts by peer and
        # lay each group out contiguously, so one message per peer suffices.
        #
        # Both sides of a pair enumerate the same shifts in the same canonical
        # order (my `rank_dest == R` set is R's `rank_source == me` set), so
        # the grouped messages match element for element; `_verify_groups`
        # checks that against the actual neighbours before it is relied on.
        def group_by_peer( bufs, key ):
            groups = {}
            for shift in bufs:
                peer = cart.get_shift_info_non_blocking( shift )[key]
                groups.setdefault( peer, [] ).append( shift )
            return { peer: groups[peer] for peer in sorted( groups ) }

        send_groups = group_by_peer( self._gpu_send_bufs, 'rank_dest'   )
        recv_groups = group_by_peer( self._gpu_recv_bufs, 'rank_source' )

        out = []
        for bufs, slices, groups in (
                ( self._gpu_send_bufs, self._gpu_send_slices, send_groups ),
                ( self._gpu_recv_bufs, self._gpu_recv_slices, recv_groups ) ):
            order     = [ shift for shifts in groups.values() for shift in shifts ]
            idx_parts = [ flat_indices( slices[shift] ) for shift in order ]
            idx = ( np.concatenate( idx_parts ) if idx_parts
                    else np.empty( 0, dtype=np.intp ) )
            out.append( ( bufs, order, idx_parts, idx, groups ) )

        # The unpack is a scatter, so it is only equivalent to the per-shift
        # loop if the receive slabs are pairwise disjoint.
        recv_idx = out[1][3]
        if np.unique( recv_idx ).size != recv_idx.size:
            return None

        if not self._verify_groups( send_groups, recv_groups ):
            return None

        flats = []
        for bufs, order, idx_parts, idx, groups in out:
            flat   = xp.empty( int( idx.size ), dtype=dtype )
            sizes  = { shift: part.size for shift, part in zip( order, idx_parts ) }
            offset = 0
            segments = []
            for peer, shifts in groups.items():
                start = offset
                for shift in shifts:
                    n = sizes[shift]
                    # Contiguous slice of a contiguous buffer, i.e. a view, so
                    # the per-peer message and the per-shift buffers address
                    # the same memory.
                    bufs[shift] = flat[offset:offset+n].reshape( bufs[shift].shape )
                    offset += n
                segments.append( ( peer, flat[start:offset] ) )
            flats.append( ( flat, xp.asarray( idx ), segments ) )

        self._gpu_send_groups = flats[0][2]
        self._gpu_recv_groups = flats[1][2]

        return ( flats[0][0], flats[0][1], flats[1][0], flats[1][1] )

    def _verify_groups( self, send_groups, recv_groups ):
        """
        Check that each peer packs the group it sends us in the order we expect
        to unpack it, by exchanging the (shift, element-count) signatures.

        Grouping several shift directions into one message per peer is only
        valid if both sides agree on the contents and ordering of that message;
        a mismatch would otherwise transfer the right number of bytes into the
        wrong ghost regions. Returns False to fall back to the per-shift path.
        """
        cart = self._cart
        comm = self._comm
        ndim = cart.ndim

        def signature( shifts, bufs ):
            sig = []
            for shift in shifts:
                sig.extend( shift )
                sig.append( bufs[shift].size )
            return np.array( sig, dtype=np.int64 )

        expected = { peer: signature( shifts, self._gpu_recv_bufs )
                     for peer, shifts in recv_groups.items() }

        tag      = 7717
        requests = []
        incoming = {}
        for peer, sig in expected.items():
            incoming[peer] = np.empty( sig.size, dtype=np.int64 )
            requests.append( comm.Irecv( incoming[peer], peer, tag ) )
        outgoing = { peer: signature( shifts, self._gpu_send_bufs )
                     for peer, shifts in send_groups.items() }
        for peer, sig in outgoing.items():
            requests.append( comm.Isend( sig, peer, tag ) )
        MPI.Request.Waitall( requests )

        for peer, sig in expected.items():
            got = incoming[peer]
            if got.size != sig.size:
                return False
            # The sender's shift s is received into our shift s region, and
            # the element counts of the two must agree.
            if not np.array_equal( got.reshape( -1, ndim+1 )[:, ndim],
                                   sig.reshape( -1, ndim+1 )[:, ndim] ):
                return False
            if not np.array_equal( got.reshape( -1, ndim+1 )[:, :ndim],
                                   sig.reshape( -1, ndim+1 )[:, :ndim] ):
                return False
        return True

    def _fused_usable( self, array ):
        """Whether the fused flat gather/scatter applies to `array`."""
        return ( self._gpu_send_idx is not None
                 and array.shape == self._gpu_full_shape
                 and array.flags.c_contiguous )

    #---------------------------------------------------------------------------
    # Public interface
    #---------------------------------------------------------------------------
    def get_send_type( self, *args ):
        shift = args[0]
        return self._send_types[shift]

    # ...
    def get_recv_type( self, *args ):
        shift = args[0]
        return self._recv_types[shift]

    # ...
    def get_assembly_send_type( self, *args ):
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

        cart = self._cart
        comm = cart._comm_cart

        if xp.is_gpu( u ):
            # Persistent requests bound to the pre-allocated contiguous
            # buffers (see __init__), not to `u` itself -- `u`'s own memory
            # is packed into/unpacked from them in start/end_update_ghost_regions.
            if self._gpu_send_groups is not None:
                # One message per peer rather than one per shift direction;
                # the group layout was checked against the peers in
                # _verify_groups.
                requests = []
                for peer, segment in self._gpu_recv_groups:
                    requests.append( comm.Recv_init( segment, peer, self._GROUP_TAG ) )
                for peer, segment in self._gpu_send_groups:
                    requests.append( comm.Send_init( segment, peer, self._GROUP_TAG ) )
                return tuple( requests )

            requests = []
            for shift in product( [-1,0,1], repeat=cart.ndim ):
                if all( s == 0 for s in shift ):
                    continue
                info = cart.get_shift_info_non_blocking( shift )
                if shift in self._gpu_recv_bufs:
                    requests.append( comm.Recv_init( self._gpu_recv_bufs[shift], info['rank_source'], info['tag'] ) )
                if shift in self._gpu_send_bufs:
                    requests.append( comm.Send_init( self._gpu_send_bufs[shift], info['rank_dest'], info['tag'] ) )
            return tuple( requests )

        # Requests' handles
        requests = []
        for shift in product( [-1,0,1], repeat=cart._ndims ):
            if all(s==0 for s in shift):
                continue

            info     = cart.get_shift_info_non_blocking( shift )

            recv_typ = self.get_recv_type( shift )
            if recv_typ != MPI.DATATYPE_NULL:
                recv_buf = (u, 1, recv_typ)
                recv_req = comm.Recv_init( recv_buf, info['rank_source'], info['tag'] )
                requests.append( recv_req )

            send_typ = self.get_send_type( shift )
            if send_typ != MPI.DATATYPE_NULL:
                send_buf = (u, 1, send_typ)
                send_req = comm.Send_init( send_buf, info['rank_dest'], info['tag'] )
                requests.append( send_req )

        return tuple(requests)

    def start_update_ghost_regions(self, array, requests ):
        # The persistent requests read/write `array` directly (host path) or
        # the packed buffers (device path, filled here); on a device backend
        # the kernels that produced `array` must have finished first either way.
        synchronize_for_mpi( array )
        if xp.is_gpu( array ):
            if self._fused_usable( array ):
                xp.take( array.reshape( -1 ), self._gpu_send_idx,
                         out=self._gpu_send_flat )
            else:
                for shift, send_buf in self._gpu_send_bufs.items():
                    send_buf[...] = array[ self._gpu_send_slices[shift] ]
        MPI.Prequest.Startall( requests )

    def end_update_ghost_regions(self, array, requests):
        MPI.Prequest.Waitall  ( requests )
        if xp.is_gpu( array ):
            if self._fused_usable( array ):
                array.reshape( -1 )[ self._gpu_recv_idx ] = self._gpu_recv_flat
            else:
                for shift, recv_buf in self._gpu_recv_bufs.items():
                    array[ self._gpu_recv_slices[shift] ] = recv_buf

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
        for shift in product( [-1,0,1], repeat=cart._ndims ):
            if all(s == 0 for s in shift):
                continue
            info = cart.get_shift_info_non_blocking( shift )

            buf_shape   = list( info[ 'buf_shape' ] ) + coeff_shape
            send_starts = list( info['send_starts'] ) + coeff_start
            recv_starts = list( info['recv_starts'] ) + coeff_start

            if info['rank_dest']>=0:
                # send_types[shift] = mpi_type.Create_subarray(
                #     sizes    = data_shape,
                #     subsizes = buf_shape,
                #     starts   = send_starts,
                # ).Commit()
                send_types[shift] = mpi_type.Create_subarray(
                    sizes    = [int(x) for x in data_shape],
                    subsizes = [int(x) for x in buf_shape],
                    starts   = [int(x) for x in send_starts],
                ).Commit()
            else:
                send_types[shift] = MPI.DATATYPE_NULL

            if info['rank_source']>=0:
                # recv_types[shift] = mpi_type.Create_subarray(
                #     sizes    = data_shape,
                #     subsizes = buf_shape,
                #     starts   = recv_starts,
                # ).Commit()
                recv_types[shift] = mpi_type.Create_subarray(
                    sizes    = [int(x) for x in data_shape],
                    subsizes = [int(x) for x in buf_shape],
                    starts   = [int(x) for x in recv_starts],
                ).Commit()
            else:
                recv_types[shift] = MPI.DATATYPE_NULL

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
                    # sizes    = data_shape ,
                    # subsizes =  buf_shape ,
                    # starts   = send_starts,
                    sizes    = [int(x) for x in data_shape],
                    subsizes = [int(x) for x in buf_shape],
                    starts   = [int(x) for x in send_starts],
                ).Commit()

                recv_types[direction,disp] = mpi_type.Create_subarray(
                    # sizes    = data_shape ,
                    # subsizes =  buf_shape ,
                    # starts   = recv_starts,
                    sizes    = [int(x) for x in data_shape],
                    subsizes = [int(x) for x in buf_shape],
                    starts   = [int(x) for x in recv_starts],
                ).Commit()

        return send_types, recv_types
