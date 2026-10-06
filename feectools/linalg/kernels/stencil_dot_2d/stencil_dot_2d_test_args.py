"""Parity-test arguments of ``stencil_dot_2d``: square and rectangular matrices."""

from feectools.linalg.tests.kernel_test_args import MATRIX_CASES, dot_arguments

CASES = MATRIX_CASES[2]
RTOL = 1e-13
ATOL = 1e-14


def make_args(backend, seed):
    return dot_arguments(*CASES[seed], seed)
