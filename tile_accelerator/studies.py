"""Reproducible compiler scaling and architecture studies using original data."""

import argparse, json, resource, time, hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .compiler import Target, compile_mlp, compile_cnn
from .analysis import Architecture, analyze, sweep
from .runtime import execute


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--simulator", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(1729)
    k, h = 65, 79
    weights = [
        (rng.standard_normal((k, h)) / k**0.5).astype(np.float32),
        np.zeros(h, np.float32),
        (rng.standard_normal((h, k)) / h**0.5).astype(np.float32),
        np.zeros(k, np.float32),
    ]
    scaling = []
    for rows in [1, 17, 65, 257, 1024]:
        start = time.perf_counter()
        program, _ = compile_mlp((rows, k, h), weights)
        compile_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        blob = program.encode()
        encode_ms = (time.perf_counter() - start) * 1000
        scaling.append(
            {
                "rows": rows,
                "compile_ms": compile_ms,
                "encode_ms": encode_ms,
                "instructions": len(program.instructions),
                "binary_bytes": len(blob),
                "process_peak_rss_kib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
            }
        )
    shape = (17, k, h)
    x = rng.standard_normal((17, k)).astype(np.float32)
    expected = np.maximum(np.maximum(x @ weights[0], 0) @ weights[2] + x, 0)
    buffered = []
    for overlap in [False, True]:
        target = Target(scratch_bytes=32768, double_buffer=overlap)
        program, _ = compile_mlp(shape, weights, target)
        actual, counters = execute(program, x, args.simulator)
        np.testing.assert_allclose(
            actual.reshape(expected.shape), expected, rtol=1e-4, atol=1e-5
        )
        estimate = analyze(
            program, Architecture(scratch_bytes=32768, macs_per_cycle=16)
        )
        buffered.append(
            {
                "target": asdict(target),
                "correctness": True,
                "execution_counters": counters,
                "estimated_cycles": estimate["cycles"],
                "estimated_latency_us": estimate["latency_us"],
                "dma_bytes": estimate["dma_bytes"],
                "energy_uj": estimate["energy_uj"],
            }
        )
    # Bandwidth/energy assumptions are explicitly varied rather than fixed silently.
    sensitivity = []
    program, _ = compile_mlp(
        shape, weights, Target(scratch_bytes=32768, double_buffer=True)
    )
    for bandwidth in [8, 16, 32]:
        for energy in [10, 20, 40]:
            result = analyze(
                program,
                Architecture(
                    scratch_bytes=32768,
                    macs_per_cycle=16,
                    dma_bytes_per_cycle=bandwidth,
                    dram_pj_per_byte=energy,
                ),
            )
            sensitivity.append(
                {
                    "dma_bytes_per_cycle": bandwidth,
                    "dram_pj_per_byte": energy,
                    "latency_us": result["latency_us"],
                    "energy_uj": result["energy_uj"],
                }
            )
    cnn_shape = (5, 7, 7)
    image = rng.standard_normal(cnn_shape).astype(np.float32)
    cnn_weights = (
        rng.standard_normal((7, 3, 3)).astype(np.float32),
        np.zeros(7, np.float32),
        np.eye(7, dtype=np.float32),
        np.zeros(7, np.float32),
    )
    native_study = []
    reference = None
    for native in [True, False]:
        program, _ = compile_cnn(
            cnn_shape, cnn_weights, Target(native_depthwise=native)
        )
        result, counters = execute(program, image, args.simulator)
        if reference is None:
            reference = result.copy()
        else:
            np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-5)
        estimate = analyze(program)
        native_study.append(
            {
                "native_depthwise": native,
                "correctness": True,
                "instructions": len(program.instructions),
                "multiply_adds": estimate["multiply_adds"],
                "estimated_latency_us": estimate["latency_us"],
                "dma_bytes": estimate["dma_bytes"],
                "limits": "Native feature area/energy cost not calibrated; not a free silicon improvement.",
            }
        )
    report = {
        "kind": "functional checks plus analytical estimates; no physical accelerator measurements",
        "compiler_scaling": scaling,
        "double_buffering": buffered,
        "architecture_sweep": sweep(shape, weights),
        "sensitivity": sensitivity,
        "native_feature_comparison": native_study,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")
        },
        "limits": [
            "peak RSS is cumulative within one process",
            "one compiler timing per size; illustrative scaling",
            "area proxy is not calibrated ASIC area",
            "architecture component costs are user-supplied assumptions",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "scaling_cases": len(scaling),
                "buffered_correctness": True,
                "latency_us": [r["estimated_latency_us"] for r in buffered],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
