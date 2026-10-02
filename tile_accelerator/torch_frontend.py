"""Bounded lowering of a verified PyTorch exported perception block."""

import hashlib
import numpy as np
from .compiler import Target, compile_cnn


def compile_exported_cnn(program, target=Target()):
    graph = program.graph_module.graph
    calls = [n for n in graph.nodes if n.op == "call_function"]
    names = [str(n.target) for n in calls]
    if names != [
        "aten.conv2d.default",
        "aten.relu.default",
        "aten.conv2d.default",
        "aten.add.Tensor",
        "aten.relu.default",
    ]:
        raise ValueError("requires depthwise/ReLU/pointwise/residual/ReLU graph")
    nodes = {n.name: n for n in graph.nodes if n.op == "placeholder"}
    users = [
        s for s in program.graph_signature.input_specs if s.kind.name == "USER_INPUT"
    ]
    if len(users) != 1:
        raise ValueError("one input required")
    x = nodes[users[0].arg.name]
    values = {
        nodes[s.arg.name]: program.state_dict[s.target]
        for s in program.graph_signature.input_specs
        if s.kind.name in ("PARAMETER", "BUFFER")
    }
    dw, relu, pw, add, last = calls
    if (
        dw.args[0] != x
        or relu.args[0] != dw
        or pw.args[0] != relu
        or add.args[:2] != (pw, x)
        or last.args[0] != add
    ):
        raise ValueError("unsupported convolution connectivity")
    shape = tuple(x.meta["val"].shape)
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError("batch-one NCHW input required")
    _, channels, height, width = shape

    def conv_options(node):
        defaults = ([1, 1], [0, 0], [1, 1], 1)
        return tuple(
            node.args[j] if len(node.args) > j else defaults[j - 3] for j in range(3, 7)
        )

    if conv_options(dw) != ([1, 1], [1, 1], [1, 1], channels) or conv_options(pw) != (
        [1, 1],
        [0, 0],
        [1, 1],
        1,
    ):
        raise ValueError("unsupported convolution stride/padding/dilation/groups")
    if add.kwargs.get("alpha", 1) != 1:
        raise ValueError("unsupported residual scaling")
    try:
        tensors = tuple(
            values[n] for n in (dw.args[1], dw.args[2], pw.args[1], pw.args[2])
        )
    except KeyError as e:
        raise ValueError("weights/biases must be frozen") from e
    import torch

    if x.meta["val"].dtype != torch.float32 or any(
        t.dtype != torch.float32 for t in tensors
    ):
        raise ValueError("float32 graph required")
    wd, bd, wp, bp = (t.detach().cpu().numpy() for t in tensors)
    if wd.shape != (channels, 1, 3, 3) or wp.shape != (channels, channels, 1, 1):
        raise ValueError("unsupported filter shapes")
    output = next(n for n in graph.nodes if n.op == "output")
    if output.args[0] != (last,):
        raise ValueError("single final output required")
    binary, mapping = compile_cnn(
        (height, width, channels), (wd[:, 0], bd, wp[:, :, 0, 0].T.copy(), bp), target
    )
    return binary, {
        "source_graph": str(graph),
        "source_sha256": hashlib.sha256(str(graph).encode()).hexdigest(),
        "input_layout": "NCHW -> HWC by runtime adapter",
        "output_layout": "HWC -> NCHW by runtime adapter",
        "shape": list(shape),
        "mapping": mapping,
    }
