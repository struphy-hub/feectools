"""Parity-test arguments of ``stencil_transpose_1d``: square and rectangular matrices."""

from feectools.linalg.tests.kernel_test_args import MATRIX_CASES, transpose_arguments

CASES = MATRIX_CASES[1]
RTOL = 0.0


def make_args(backend, seed):
    return transpose_arguments(*CASES[seed], seed)
