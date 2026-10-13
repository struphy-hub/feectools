"""Compare the threaded compiled kernels with the serial Python kernels in fresh processes.

Run after ``psydac-accelerate --openmp``. Separate processes ensure that the OpenMP runtime reads each requested
thread count before the extensions are loaded.
"""
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# 3D DOF kernels with the number of quadrature weights per direction they take (0: no weights)
DOF_KERNELS = {
    "evaluate_dofs_3d_0form": 0,
    "evaluate_dofs_3d_1form": 3,
    "evaluate_dofs_3d_2form": 3,
    "evaluate_dofs_3d_3form": 3,
    "evaluate_dofs_3d_vec": 0,
}


def _compiled_with_openmp():
    from feectools.accelerate.accelerate import _omp_stamp_file

    try:
        with open(_omp_stamp_file) as f:
            return bool(f.read().strip())
    except FileNotFoundError:
        return False


def test_openmp_matches_serial_loops(tmp_path):
    if not _compiled_with_openmp():
        pytest.skip("compile kernels with `psydac-accelerate --openmp` to exercise threaded execution")

    import feectools

    # the processes import the same feectools as this one
    root = str(Path(feectools.__file__).resolve().parents[1])
    for mode in ("reference", "1", "2", "4"):
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS="1" if mode == "reference" else mode, OMP_DYNAMIC="FALSE")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), mode, str(tmp_path / f"{mode}.npz")],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )

    with np.load(tmp_path / "reference.npz") as expected, np.load(tmp_path / "1.npz") as serial:
        for threads in (1, 2, 4):
            with np.load(tmp_path / f"{threads}.npz") as actual:
                assert actual.files == expected.files
                for name in expected.files:
                    # the compiled code may round differently from Python (fused multiply-add, vectorization)
                    np.testing.assert_allclose(
                        actual[name],
                        expected[name],
                        rtol=1e-12,
                        atol=1e-12,
                        err_msg=f"{name}, OMP_NUM_THREADS={threads}",
                    )
                    # but no kernel may depend on the number of threads, not even in the last bit
                    np.testing.assert_array_equal(
                        actual[name], serial[name], err_msg=f"{name}, OMP_NUM_THREADS={threads}"
                    )


def _load(module_name, reference):
    """The serial Python module (reference) or the compiled extension of `module_name`."""
    if reference:
        short_name = module_name.rsplit(".", 1)[-1]
        source = Path(importlib.util.find_spec(module_name).origin).parent / f"{short_name}.py"
        spec = importlib.util.spec_from_file_location(f"_serial_{short_name}", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
        assert Path(module.__file__).suffix != ".py", f"{module_name} is not compiled"
    return module


def _run_cases(reference, output):
    from feectools.linalg.tests.cuda_parity_cases import PARITY_CASES
    from feectools.linalg.tests.kernel_test_args import axpy_arguments, dot_arguments, inner_arguments
    from feectools.linalg.tests.kernel_test_args import transpose_arguments

    results = {}

    def execute(module_name, name, args, case):
        getattr(_load(module_name, reference), name)(*args)
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                results[f"{name}-{case}-{i}"] = arg.copy()

    # grids larger than the parity cases, so that every thread gets several rows
    larger = {
        2: (((40, 36), (40, 36), (3, 2), None, None), ((40, 36), (39, 36), (2, 3), (1, 2), (False, True))),
        3: (((20, 18, 16), (20, 18, 16), (3, 2, 3), None, None),
            ((20, 18, 16), (19, 18, 17), (2, 3, 1), (1, 2, 1), (False, True, False))),
    }
    for ndim in (2, 3):
        for operation in ("dot", "transpose", "inner", "axpy"):
            name = f"stencil_{operation}_{ndim}d"
            module_name = f"feectools.linalg.kernels.{name}.{name}_kernels"
            spec = PARITY_CASES[name]
            cases = [spec.build(case) for case in spec.cases]
            for index, case in enumerate(larger[ndim]):
                seed = 100 + index
                if operation == "dot":
                    cases.append(dot_arguments(*case, seed))
                elif operation == "transpose":
                    cases.append(transpose_arguments(*case, seed))
                elif operation == "inner":
                    cases.append(inner_arguments(case[0], case[2], seed))
                else:
                    cases.append(axpy_arguments(case[0], case[2], seed))
            for case_id, args in enumerate(cases):
                execute(module_name, name, args, case_id)

    rng = np.random.default_rng(7)
    n, k = (11, 9, 7), (3, 2, 4)
    for name, n_weights in DOF_KERNELS.items():
        n_forms = 1 if name in ("evaluate_dofs_3d_0form", "evaluate_dofs_3d_3form") else 3
        weights = [rng.random((n[d], k[d])) for d in range(n_weights)]
        outputs = [np.zeros(n) for _ in range(n_forms)]
        if n_weights == 0:
            points = [rng.random(n) for _ in range(n_forms)]
        elif name == "evaluate_dofs_3d_1form":
            # component d is integrated along direction d
            points = [rng.random(tuple(n[e] * k[e] if e == d else n[e] for e in range(3))) for d in range(3)]
        elif name == "evaluate_dofs_3d_2form":
            # component d is integrated over the two other directions
            points = [rng.random(tuple(n[e] if e == d else n[e] * k[e] for e in range(3))) for d in range(3)]
        else:
            points = [rng.random(tuple(n[e] * k[e] for e in range(3)))]
        execute("feectools.feec.dof_kernels", name, (*weights, *outputs, *points), 0)

    np.savez_compressed(output, **results)


if __name__ == "__main__":
    _run_cases(sys.argv[1] == "reference", sys.argv[2])
