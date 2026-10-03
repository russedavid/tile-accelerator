import argparse
import json
from pathlib import Path
import numpy as np
from .compiler import Target, compile_mlp
from .isa import decode, disassemble
from .analysis import analyze, sweep
from .runtime import execute


def main():
    p = argparse.ArgumentParser(
        description="Compile, execute and inspect a small tensor accelerator program"
    )
    sub = p.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--simulator", type=Path, required=True)
    dump = sub.add_parser("disassemble")
    dump.add_argument("binary", type=Path)
    args = p.parse_args()
    if args.command == "disassemble":
        print(disassemble(decode(args.binary.read_bytes())), end="")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    shape = (33, 31, 47)
    m, k, h = shape
    rng = np.random.default_rng(1729)
    x = rng.standard_normal((m, k)).astype(np.float32)
    weights = [
        (rng.standard_normal((k, h)) / k**0.5).astype(np.float32),
        np.zeros(h, np.float32),
        (rng.standard_normal((h, k)) / h**0.5).astype(np.float32),
        np.zeros(k, np.float32),
    ]
    program, source_map = compile_mlp(shape, weights)
    (args.output / "model.bin").write_bytes(program.encode())
    (args.output / "instructions.txt").write_text(disassemble(program))
    (args.output / "source-map.json").write_text(
        json.dumps(source_map, indent=2) + "\n"
    )
    actual, counters = execute(program, x, args.simulator)
    expected = np.maximum(
        np.maximum(x @ weights[0] + weights[1], 0) @ weights[2] + weights[3] + x, 0
    )
    actual = actual.reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    np.save(args.output / "actual.npy", actual)
    np.save(args.output / "expected.npy", expected)
    estimate = analyze(program)
    (args.output / "estimate.json").write_text(json.dumps(estimate, indent=2) + "\n")
    (args.output / "sweep.json").write_text(
        json.dumps(sweep(shape, weights), indent=2) + "\n"
    )
    result = {
        "passed": True,
        "shape": list(shape),
        "max_abs_error": float(np.max(np.abs(actual - expected))),
        "execution_counters": counters,
        "analytical_latency_us": estimate["latency_us"],
        "limits": "numerical correctness and estimates; no real accelerator silicon",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
