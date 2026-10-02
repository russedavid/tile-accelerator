import numpy as np
import pytest
from tile_accelerator.torch_frontend import compile_exported_cnn
from tile_accelerator.runtime import execute


from test_execution import simulator


def test_pytorch_export_to_binary_to_cpp(tmp_path, simulator):
    torch = pytest.importorskip("torch")
    binary = simulator

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dw = torch.nn.Conv2d(7, 7, 3, padding=1, groups=7)
            self.pw = torch.nn.Conv2d(7, 7, 1)

        def forward(self, x):
            return (self.pw(self.dw(x).relu()) + x).relu()

    torch.manual_seed(1729)
    module = Block().eval()
    x = torch.randn(1, 7, 11, 13)
    p, trace = compile_exported_cnn(torch.export.export(module, (x,)))
    input_hwc = x[0].permute(1, 2, 0).contiguous().numpy()
    actual, _ = execute(p, input_hwc, binary)
    expected = module(x).detach()[0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(
        actual.reshape(expected.shape), expected, rtol=1e-4, atol=1e-5
    )
    assert trace["shape"] == [1, 7, 11, 13]
