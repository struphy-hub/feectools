"""Parity-test arguments of ``stencil_inner_3d``: vector spaces with different pads (GPU: the summation order differs)."""

from feectools.linalg.tests.kernel_test_args import VECTOR_CASES, inner_arguments

CASES = VECTOR_CASES[3]
RTOL = 1e-12


def make_args(backend, seed):
    return inner_arguments(*CASES[seed], seed)
