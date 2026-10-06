"""Parity-test arguments of ``stencil_axpy_2d``: vector spaces with different pads."""

from feectools.linalg.tests.kernel_test_args import VECTOR_CASES, axpy_arguments

CASES = VECTOR_CASES[2]
RTOL = 1e-14
ATOL = 1e-15


def make_args(backend, seed):
    return axpy_arguments(*CASES[seed], seed)
