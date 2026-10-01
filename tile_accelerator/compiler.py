"""Explicit tensor-to-instruction lowering with capacity and dependency checks."""

from dataclasses import dataclass
import numpy as np
from .isa import Instruction as I, Op, Program, MAX_ELEMENTS, MAX_INSTRUCTIONS


@dataclass(frozen=True)
class Target:
    scratch_bytes: int = 16384
    tile_m: int = 16
    tile_n: int = 32
    tile_k: int = 32
    double_buffer: bool = False
    native_depthwise: bool = True
    operand_dtype: str = "f32"

    def validate(self):
        if min(self.tile_m, self.tile_n, self.tile_k) < 1 or self.scratch_bytes < 4:
            raise ValueError("invalid target limits")
        if self.operand_dtype not in ("f32", "f16"):
            raise ValueError("operand dtype must be f32 or f16")
        if self.scratch_bytes // 4 > MAX_ELEMENTS:
            raise ValueError("scratch exceeds binary format limit")
        stages = 2 if self.double_buffer else 1
        required = (
            stages * (self.tile_m * self.tile_k + self.tile_k * self.tile_n)
            + 2 * self.tile_m * self.tile_n
            + self.tile_n
        ) * 4
        if required > self.scratch_bytes:
            raise ValueError(
                f"tile needs {required} bytes; target has {self.scratch_bytes}"
            )


class Assembler:
    def __init__(self, target):
        target.validate()
        self.target = target
        self.memory = []
        self.at = 0
        self.instructions = []
        self.source_map = []

    def allocate(self, value):
        flat = np.asarray(value, dtype=np.float32).reshape(-1)
        if not np.isfinite(flat).all():
            raise ValueError("finite constants required")
        offset = self.at
        self.memory.append(flat)
        self.at += flat.size
        return offset

    def emit(self, op, depends=None, **kw):
        event = len(self.instructions) + 1
        if event > MAX_INSTRUCTIONS:
            raise ValueError("instruction limit exceeded")
        dep = event - 1 if depends is None else depends
        self.instructions.append(I(op, event=event, depends=dep, **kw))
        return event

    def matmul(
        self,
        input_offset,
        weight_offset,
        bias_offset,
        output_offset,
        M,
        N,
        K,
        residual_offset=None,
        label="projection",
    ):
        t = self.target
        slots = 2 if t.double_buffer else 1
        a_size = t.tile_m * t.tile_k
        b_size = t.tile_k * t.tile_n
        b = slots * a_size
        c = b + slots * b_size
        bias = c + t.tile_m * t.tile_n
        residual = bias + t.tile_n
        for mi in range(0, M, t.tile_m):
            tm = min(t.tile_m, M - mi)
            for ni in range(0, N, t.tile_n):
                tn = min(t.tile_n, N - ni)
                begin = len(self.instructions)
                base = begin
                zero = self.emit(Op.ZERO, a=c, m=tm * tn)
                tiles = list(range(0, K, t.tile_k))
                ready = {}

                def load(index, reuse_dependency):
                    ki = tiles[index]
                    tk = min(t.tile_k, K - ki)
                    slot = index % slots
                    aa = slot * a_size
                    bb = b + slot * b_size
                    first = self.emit(
                        Op.LOAD2D,
                        depends=reuse_dependency,
                        a=input_offset + mi * K + ki,
                        b=aa,
                        m=tm,
                        n=tk,
                        stride=K,
                        flags=int(t.operand_dtype == "f16"),
                    )
                    ready[index] = self.emit(
                        Op.LOAD2D,
                        depends=first,
                        a=weight_offset + ki * N + ni,
                        b=bb,
                        m=tk,
                        n=tn,
                        stride=N,
                        flags=int(t.operand_dtype == "f16"),
                    )

                if t.double_buffer:
                    for j in range(min(2, len(tiles))):
                        load(j, base)
                previous_compute = zero
                for j, ki in enumerate(tiles):
                    if not t.double_buffer:
                        load(j, previous_compute)
                    slot = j % slots
                    tk = min(t.tile_k, K - ki)
                    current = self.emit(
                        Op.MATMUL,
                        depends=ready[j],
                        a=slot * a_size,
                        b=b + slot * b_size,
                        c=c,
                        m=tm,
                        n=tn,
                        k=tk,
                        flags=1,
                        aux=previous_compute,
                    )
                    previous_compute = current
                    if t.double_buffer and j + 2 < len(tiles):
                        load(j + 2, current)
                # All prefetched commands have been consumed before the epilogue.
                self.emit(
                    Op.LOAD2D,
                    depends=previous_compute,
                    a=bias_offset + ni,
                    b=bias,
                    m=1,
                    n=tn,
                    stride=N,
                )
                if residual_offset is not None:
                    self.emit(
                        Op.LOAD2D,
                        a=residual_offset + mi * N + ni,
                        b=residual,
                        m=tm,
                        n=tn,
                        stride=N,
                        flags=int(t.operand_dtype == "f16"),
                    )
                self.emit(
                    Op.EPILOGUE,
                    a=c,
                    b=bias,
                    c=residual,
                    m=tm,
                    n=tn,
                    flags=1 | (2 if residual_offset is not None else 0),
                )
                self.emit(
                    Op.STORE2D, a=c, b=output_offset + mi * N + ni, m=tm, n=tn, stride=N
                )
                self.source_map.append(
                    {
                        "instructions": [begin, len(self.instructions)],
                        "layer": label,
                        "output_tile": [mi, ni, tm, tn],
                        "reduction": K,
                    }
                )

    def finish(self, input_elements, output_offset, output_elements):
        self.emit(Op.HALT)
        p = Program(
            self.instructions,
            np.concatenate(self.memory),
            self.target.scratch_bytes // 4,
            input_elements,
            output_offset,
            output_elements,
        )
        p.encode()
        return p, self.source_map


def compile_mlp(shape, weights, target=Target()):
    rows, channels, hidden = map(int, shape)
    if min(rows, channels, hidden) < 1:
        raise ValueError("positive dimensions required")
    if (
        rows * (2 * channels + hidden) + 2 * channels * hidden + hidden + channels
        > MAX_ELEMENTS
    ):
        raise ValueError("graph exceeds binary memory limit")
    w1, b1, w2, b2 = [np.asarray(w, dtype=np.float32) for w in weights]
    if [w.shape for w in (w1, b1, w2, b2)] != [
        (channels, hidden),
        (hidden,),
        (hidden, channels),
        (channels,),
    ]:
        raise ValueError("incompatible weights")
    a = Assembler(target)
    x = a.allocate(np.zeros((rows, channels), np.float32))
    aw1, ab1, aw2, ab2 = [a.allocate(w) for w in (w1, b1, w2, b2)]
    intermediate = a.allocate(np.zeros((rows, hidden), np.float32))
    output = a.allocate(np.zeros((rows, channels), np.float32))
    a.matmul(
        x, aw1, ab1, intermediate, rows, hidden, channels, label="first_projection_relu"
    )
    a.matmul(
        intermediate,
        aw2,
        ab2,
        output,
        rows,
        channels,
        hidden,
        x,
        label="second_projection_residual_relu",
    )
    return a.finish(rows * channels, output, rows * channels)


def lower_depthwise_matrix(a, x, weight, bias, output, height, width, channels):
    """Dense engine fallback; gathers patches directly into bounded scratch."""
    t = a.target
    rows = height * width
    reduction = 9 * channels
    sa = 0
    sb = t.tile_m * t.tile_k
    sc = sb + t.tile_k * t.tile_n
    sbi = sc + t.tile_m * t.tile_n
    for mi in range(0, rows, t.tile_m):
        tm = min(t.tile_m, rows - mi)
        for ni in range(0, channels, t.tile_n):
            tn = min(t.tile_n, channels - ni)
            start = len(a.instructions)
            a.emit(Op.ZERO, a=sc, m=tm * tn)
            for ki in range(0, reduction, t.tile_k):
                tk = min(t.tile_k, reduction - ki)
                for rr in range(tm):
                    yy = (mi + rr) // width
                    xx = (mi + rr) % width
                    offset = 0
                    while offset < tk:
                        feature = ki + offset
                        channel = feature % channels
                        window = feature // channels
                        sy = yy + window // 3 - 1
                        sx = xx + window % 3 - 1
                        length = min(channels - channel, tk - offset)
                        if 0 <= sy < height and 0 <= sx < width:
                            a.emit(
                                Op.LOAD2D,
                                a=x + (sy * width + sx) * channels + channel,
                                b=sa + rr * tk + offset,
                                m=1,
                                n=length,
                                stride=length,
                                flags=int(t.operand_dtype == "f16"),
                            )
                        else:
                            a.emit(Op.ZERO, a=sa + rr * tk + offset, m=length)
                        offset += length
                a.emit(
                    Op.LOAD2D,
                    a=weight + ki * channels + ni,
                    b=sb,
                    m=tk,
                    n=tn,
                    stride=channels,
                    flags=int(t.operand_dtype == "f16"),
                )
                a.emit(Op.MATMUL, a=sa, b=sb, c=sc, m=tm, n=tn, k=tk, flags=1)
            a.emit(Op.LOAD2D, a=bias + ni, b=sbi, m=1, n=tn, stride=channels)
            a.emit(Op.EPILOGUE, a=sc, b=sbi, m=tm, n=tn, flags=1)
            a.emit(
                Op.STORE2D,
                a=sc,
                b=output + mi * channels + ni,
                m=tm,
                n=tn,
                stride=channels,
            )
            a.source_map.append(
                {
                    "instructions": [start, len(a.instructions)],
                    "layer": "matrix_depthwise_relu",
                    "output_tile": [mi, ni, tm, tn],
                }
            )


def compile_cnn(shape, weights, target=Target()):
    """HWC batch-one 3x3 depthwise/ReLU -> pointwise/residual/ReLU."""
    height, width, channels = map(int, shape)
    if min(height, width, channels) < 1:
        raise ValueError("positive dimensions required")
    weight_cost = (
        channels * channels + 11 * channels
        if target.native_depthwise
        else 10 * channels * channels + 2 * channels
    )
    if 3 * height * width * channels + weight_cost > MAX_ELEMENTS:
        raise ValueError("graph exceeds binary memory limit")
    dw, db, pw, pb = [np.asarray(w, np.float32) for w in weights]
    if [w.shape for w in (dw, db, pw, pb)] != [
        (channels, 3, 3),
        (channels,),
        (channels, channels),
        (channels,),
    ]:
        raise ValueError("incompatible CNN weights")
    a = Assembler(target)
    x = a.allocate(np.zeros((height, width, channels), np.float32))
    if target.native_depthwise:
        depthwise_weights = dw
    else:
        depthwise_weights = np.zeros((9 * channels, channels), np.float32)
        for ch in range(channels):
            for y in range(3):
                for xoffset in range(3):
                    depthwise_weights[(y * 3 + xoffset) * channels + ch, ch] = dw[
                        ch, y, xoffset
                    ]
    adw, adb, apw, apb = [a.allocate(w) for w in (depthwise_weights, db, pw, pb)]
    intermediate = a.allocate(np.zeros((height, width, channels), np.float32))
    output = a.allocate(np.zeros((height, width, channels), np.float32))
    if target.native_depthwise:
        begin = len(a.instructions)
        a.emit(
            Op.DEPTHWISE,
            a=x,
            b=adw,
            c=intermediate,
            m=height,
            n=width,
            k=channels,
            aux=adb,
            flags=1 | (2 if target.operand_dtype == "f16" else 0),
        )
        a.source_map.append(
            {
                "instructions": [begin, len(a.instructions)],
                "layer": "native_depthwise_relu",
                "shape": list(shape),
            }
        )
    else:
        lower_depthwise_matrix(a, x, adw, adb, intermediate, height, width, channels)
    a.matmul(
        intermediate,
        apw,
        apb,
        output,
        height * width,
        channels,
        channels,
        x,
        label="pointwise_residual_relu",
    )
    return a.finish(height * width * channels, output, height * width * channels)
