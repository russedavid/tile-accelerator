import numpy as np
import pytest
from test_execution import simulator, values
from tile_accelerator.compiler import compile_mlp


def test_native_vm_bridge_resets_each_request(simulator):
    pytest.importorskip("iree.runtime")
    pytest.importorskip("iree.compiler")
    from tile_accelerator.iree_bridge import IreeBridge

    shape = (3, 7, 13)
    x, *weights = values(shape)
    program, _ = compile_mlp(shape, weights)
    bridge = IreeBridge(simulator)
    for value in [x, -x, np.zeros_like(x)]:
        actual = bridge(program, value).reshape(x.shape)
        expected = np.maximum(
            np.maximum(value @ weights[0] + weights[1], 0) @ weights[2]
            + weights[3]
            + value,
            0,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
