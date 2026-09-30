"""Each process is bound to its own GPU when feectools.ddm.cart is imported."""
import pytest
import cunumpy


@pytest.mark.skipif(not cunumpy.cupy_available(), reason="CuPy/GPU not available")
def test_rank_is_bound_to_its_local_device():
    import cupy as cp

    import feectools.ddm.cart  # noqa: F401 -- binds the device on import

    if cunumpy.get_backend() != "cupy":
        pytest.skip("device binding only happens on the CuPy backend")
    expected = cunumpy.local_rank() % cunumpy.device_count()
    assert cp.cuda.runtime.getDevice() == expected
