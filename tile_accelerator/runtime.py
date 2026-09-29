"""Launch the independent C++ interpreter and preserve binary/request evidence."""

import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np


def execute(program, x, simulator, timeout=30):
    x = np.asarray(x, dtype=np.float32)
    if x.size != program.input_elements or not np.isfinite(x).all():
        raise ValueError("input extent or finite-value contract failed")
    with tempfile.TemporaryDirectory(prefix="tile-request-") as tmp:
        root = Path(tmp)
        (root / "program.bin").write_bytes(program.encode())
        x.astype("<f4").tofile(root / "input.bin")
        proc = subprocess.run(
            [
                str(simulator),
                str(root / "program.bin"),
                str(root / "input.bin"),
                str(root / "output.bin"),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode:
            raise RuntimeError(proc.stderr.strip())
        result = np.fromfile(root / "output.bin", dtype="<f4")
        if result.size != program.output_elements or not np.isfinite(result).all():
            raise RuntimeError("invalid simulator output")
        return result, json.loads(proc.stdout)
