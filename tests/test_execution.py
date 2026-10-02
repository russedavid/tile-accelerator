from pathlib import Path
import os
import shutil
import struct
import subprocess
import numpy as np
import pytest
from tile_accelerator.compiler import Target, compile_mlp
from tile_accelerator.isa import HEADER, Instruction, Op, Program, decode


@pytest.fixture(scope="session")
def simulator(tmp_path_factory):
    override = os.environ.get("TILE_SIM")
    if override:
        return Path(override)
    compiler = shutil.which("c++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    binary = tmp_path_factory.mktemp("sim") / "tile-sim"
    source = Path(__file__).resolve().parents[1] / "cpp/simulator.cpp"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-contract=off",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary


def values(shape, seed=4):
    m, k, h = shape
    rng = np.random.default_rng(seed)
    return tuple(
        a.astype(np.float32)
        for a in (
            rng.standard_normal((m, k)),
            rng.standard_normal((k, h)) / k**0.5,
            rng.standard_normal(h) * 0.1,
            rng.standard_normal((h, k)) / h**0.5,
            rng.standard_normal(k) * 0.1,
        )
    )


def execute(simulator, tmp_path, program, x):
    (tmp_path / "program.bin").write_bytes(program.encode())
    np.asarray(x, dtype="<f4").tofile(tmp_path / "input.bin")
    return subprocess.run(
        [
            str(simulator),
            str(tmp_path / "program.bin"),
            str(tmp_path / "input.bin"),
            str(tmp_path / "output.bin"),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )


@pytest.mark.parametrize("shape", [(1, 3, 5), (17, 7, 13), (33, 32, 64), (65, 31, 47)])
def test_binary_matches_independent_numpy(simulator, tmp_path, shape):
    x, w1, b1, w2, b2 = values(shape)
    p, _ = compile_mlp(shape, (w1, b1, w2, b2), Target(tile_m=8, tile_n=16, tile_k=8))
    blob = p.encode()
    assert decode(blob).encode() == blob
    for changed in [x, -x, np.zeros_like(x)]:
        proc = execute(simulator, tmp_path, p, changed)
        assert proc.returncode == 0, proc.stderr
        actual = np.fromfile(tmp_path / "output.bin", dtype="<f4").reshape(x.shape)
        expected = np.maximum(np.maximum(changed @ w1 + b1, 0) @ w2 + b2 + changed, 0)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_invalid_dependency_and_uninitialized_reads_are_rejected(simulator, tmp_path):
    for instructions in [
        [
            Instruction(Op.STORE2D, a=0, b=0, m=1, n=1, stride=1, event=1),
            Instruction(Op.HALT, event=2, depends=1),
        ],
        [
            Instruction(Op.ZERO, m=1, event=1, depends=1),
            Instruction(Op.HALT, event=2, depends=1),
        ],
    ]:
        p = Program(instructions, np.zeros(1, np.float32), 1, 1, 0, 1)
        assert execute(simulator, tmp_path, p, [1]).returncode == 2


def test_corrupt_binary_rejected_by_python_and_cpp(simulator, tmp_path):
    x, *weights = values((1, 3, 5))
    p, _ = compile_mlp((1, 3, 5), weights)
    valid = p.encode()
    for blob in [
        valid[:12],
        valid[:-1],
        valid + b"\0",
        b"BADMAGIC" + valid[8:],
        valid[:8] + struct.pack("<I", 2) + valid[12:],
    ]:
        with pytest.raises(ValueError):
            decode(blob)
        (tmp_path / "bad.bin").write_bytes(blob)
        x.astype("<f4").tofile(tmp_path / "input.bin")
        proc = subprocess.run(
            [
                str(simulator),
                str(tmp_path / "bad.bin"),
                str(tmp_path / "input.bin"),
                str(tmp_path / "out.bin"),
            ],
            capture_output=True,
            timeout=3,
        )
        assert proc.returncode == 2


def test_capacity_and_shape_fail_before_encoding():
    with pytest.raises(ValueError):
        Target(scratch_bytes=1024).validate()
    x, *weights = values((2, 7, 13))
    with pytest.raises(ValueError):
        compile_mlp((2, 8, 13), weights)


def test_double_buffered_schedule_matches_serial(simulator, tmp_path):
    shape = (17, 65, 79)
    x, *weights = values(shape)
    serial, _ = compile_mlp(shape, weights, Target(scratch_bytes=32768))
    buffered, _ = compile_mlp(
        shape, weights, Target(scratch_bytes=32768, double_buffer=True)
    )
    assert execute(simulator, tmp_path, serial, x).returncode == 0
    expected = np.fromfile(tmp_path / "output.bin", dtype="<f4").copy()
    assert execute(simulator, tmp_path, buffered, x).returncode == 0
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "output.bin", dtype="<f4"), expected
    )
    from tile_accelerator.analysis import analyze, Architecture

    arch = Architecture(scratch_bytes=32768, macs_per_cycle=16)
    assert analyze(buffered, arch)["cycles"] < analyze(serial, arch)["cycles"]


def test_native_cnn_matches_direct_convolution(simulator, tmp_path):
    from tile_accelerator.compiler import compile_cnn

    h, w, c = 5, 7, 3
    rng = np.random.default_rng(2)
    x = rng.standard_normal((h, w, c)).astype(np.float32)
    dw = rng.standard_normal((c, 3, 3)).astype(np.float32)
    db = np.zeros(c, np.float32)
    pw = np.eye(c, dtype=np.float32)
    pb = np.zeros(c, np.float32)
    p, _ = compile_cnn((h, w, c), (dw, db, pw, pb))
    expected = np.zeros_like(x)
    for y in range(h):
        for xx in range(w):
            for ch in range(c):
                total = np.float32(0)
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if 0 <= y + dy < h and 0 <= xx + dx < w:
                            total += x[y + dy, xx + dx, ch] * dw[ch, dy + 1, dx + 1]
                expected[y, xx, ch] = max(0, total)
    expected = np.maximum(expected + x, 0)
    result = execute(simulator, tmp_path, p, x)
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(
        np.fromfile(tmp_path / "output.bin", dtype="<f4").reshape(x.shape),
        expected,
        rtol=1e-4,
        atol=1e-5,
    )


def test_seeded_shape_fuzz(simulator, tmp_path):
    rng = np.random.default_rng(44)
    for _ in range(20):
        shape = tuple(map(int, rng.integers(1, 40, size=3)))
        x, *weights = values(shape, int(rng.integers(10000)))
        p, _ = compile_mlp(shape, weights, Target(tile_m=5, tile_n=7, tile_k=9))
        proc = execute(simulator, tmp_path, p, x)
        assert proc.returncode == 0, proc.stderr
        actual = np.fromfile(tmp_path / "output.bin", dtype="<f4").reshape(x.shape)
        expected = np.maximum(
            np.maximum(x @ weights[0] + weights[1], 0) @ weights[2] + weights[3] + x, 0
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_out_of_bounds_and_nonfinite_input_fail(simulator, tmp_path):
    p = Program(
        [
            Instruction(Op.LOAD2D, a=0, b=8, m=1, n=1, stride=1, event=1),
            Instruction(Op.HALT, event=2, depends=1),
        ],
        np.zeros(1, np.float32),
        1,
        1,
        0,
        1,
    )
    assert execute(simulator, tmp_path, p, [1]).returncode == 2
    p = Program([Instruction(Op.HALT, event=1)], np.zeros(1, np.float32), 1, 1, 0, 1)
    assert execute(simulator, tmp_path, p, [float("nan")]).returncode == 2


def test_half_operands_and_f32_accumulation(simulator, tmp_path):
    shape = (17, 31, 47)
    x, w1, b1, w2, b2 = values(shape)
    p, _ = compile_mlp(shape, (w1, b1, w2, b2), Target(operand_dtype="f16"))
    proc = execute(simulator, tmp_path, p, x)
    assert proc.returncode == 0, proc.stderr
    half = lambda a: a.astype(np.float16).astype(np.float32)
    hidden = np.maximum(half(x) @ half(w1) + b1, 0)
    expected = np.maximum(half(hidden) @ half(w2) + b2 + half(x), 0)
    np.testing.assert_allclose(
        np.fromfile(tmp_path / "output.bin", dtype="<f4").reshape(x.shape),
        expected,
        rtol=1e-4,
        atol=1e-5,
    )


def test_half_ties_subnormals_and_overflow(simulator, tmp_path):
    values = np.array(
        [0, -0.0, 2**-24, 2**-25, 3 * 2**-25, 1 + 2**-11, 1 + 3 * 2**-11, 65504.0],
        np.float32,
    )
    count = values.size
    p = Program(
        [
            Instruction(
                Op.LOAD2D, a=0, b=0, m=1, n=count, stride=count, flags=1, event=1
            ),
            Instruction(
                Op.STORE2D, a=0, b=0, m=1, n=count, stride=count, event=2, depends=1
            ),
            Instruction(Op.HALT, event=3, depends=2),
        ],
        np.zeros(count, np.float32),
        count,
        count,
        0,
        count,
    )
    proc = execute(simulator, tmp_path, p, values)
    assert proc.returncode == 0, proc.stderr
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "output.bin", dtype="<f4"),
        values.astype(np.float16).astype(np.float32),
    )
    values[-1] = 1e10
    assert execute(simulator, tmp_path, p, values).returncode == 2


def test_overflow_is_not_hidden_by_activation(simulator, tmp_path):
    shape = (1, 1, 1)
    weights = (
        np.ones((1, 1), np.float32),
        np.zeros(1, np.float32),
        np.zeros((1, 1), np.float32),
        np.zeros(1, np.float32),
    )
    p, _ = compile_mlp(shape, weights, Target(operand_dtype="f16"))
    assert execute(simulator, tmp_path, p, [-1e10]).returncode == 2


def test_depthwise_feature_and_matrix_fallback_agree(simulator, tmp_path):
    from tile_accelerator.compiler import compile_cnn
    from tile_accelerator.analysis import analyze

    rng = np.random.default_rng(1729)
    shape = (5, 7, 7)
    x = rng.standard_normal(shape).astype(np.float32)
    weights = (
        rng.standard_normal((7, 3, 3)).astype(np.float32),
        np.zeros(7, np.float32),
        np.eye(7, dtype=np.float32),
        np.zeros(7, np.float32),
    )
    native, _ = compile_cnn(shape, weights, Target())
    matrix, _ = compile_cnn(shape, weights, Target(native_depthwise=False))
    assert execute(simulator, tmp_path, native, x).returncode == 0
    expected = np.fromfile(tmp_path / "output.bin", dtype="<f4").copy()
    assert execute(simulator, tmp_path, matrix, x).returncode == 0
    np.testing.assert_allclose(
        np.fromfile(tmp_path / "output.bin", dtype="<f4"),
        expected,
        rtol=1e-4,
        atol=1e-5,
    )
    assert analyze(matrix)["multiply_adds"] > analyze(native)["multiply_adds"]
