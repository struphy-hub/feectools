#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from abc import ABC, abstractmethod


__all__ = ('CartDataExchanger',)
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
    # Shared implementation
    #---------------------------------------------------------------------------

    def _local_exchange_assembly_data(self, array):
        """Do the assembly exchange without MPI, on a single-process cart.

        With one process per direction every neighbour in the Cartesian
        topology is this process itself, so `start_exchange_assembly_data`
        degenerates to a self-message followed by a local accumulation. MPI
        has to walk the strided subarray datatype element by element to
        deliver that message -- ruinously so for a device buffer -- while the
        same data movement is two slice operations here.

        This reproduces the MPI path exactly, including two details that the
        older `_exchange_assembly_data_serial` helpers in
        `feectools.linalg.stencil` do not:

        * the received block really is written into the opposite ghost region
          (the wrap), not just accumulated into the interior;
        * across a non-periodic boundary MPI transfers nothing (both ranks are
          `MPI.PROC_NULL`) but the accumulation step still runs, on whatever
          the receiving ghost region already held.
        """
        cart    = self._cart
        ndim    = cart.ndim
        axis    = self._axis
        periods = cart.periods
        disp    = 1

        for direction in range(ndim):
            if direction == axis:
                continue

            info      = cart.get_shift_info(direction, disp)
            buf_shape = info['buf_shape']

            if periods[direction]:
                # The self-message: send block -> receive block. Copy the
                # source first; the two blocks are disjoint for any grid
                # wider than its ghost regions, but not by construction.
                idx_send = tuple(slice(s, s + b) for s, b in zip(info['send_assembly_starts'], buf_shape))
                idx_recv = tuple(slice(s, s + b) for s, b in zip(info['recv_assembly_starts'], buf_shape))
                array[idx_recv] = array[idx_send].copy()

            # ... and the accumulation the MPI path performs afterwards,
            # which runs whether or not anything was received.
            pads            = [0] * ndim
            pads[direction] = cart._pads[direction] * cart._shifts[direction]

            idx_from = tuple(slice(s, s + b) for s, b in zip(info['recv_starts'], buf_shape))
            idx_to   = tuple(slice(s + p, s + b + p) for s, b, p in zip(info['recv_starts'], buf_shape, pads))
            array[idx_to] += array[idx_from]

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

