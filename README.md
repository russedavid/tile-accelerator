# Tile Accelerator

Compile a small neural-network graph into explicit tile transfers, matrix operations and dependencies, then execute the binary in an independent C++ simulator.

The project supports a two-layer residual MLP and a depthwise/pointwise CNN block exported from PyTorch. It includes a disassembler, source-to-instruction mapping, scratch-capacity checks, serial and double-buffered schedules, FP32/FP16 operand contracts, and a separate architecture cost model.

## Build and run

Python 3.10+, NumPy and a C++17 compiler are sufficient for the core tools. CMake is optional when compiling directly.

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
.venv/bin/python -m pytest -q
.venv/bin/python -m tile_accelerator.cli demo --output runs/demo-001 --simulator build/tile-sim
.venv/bin/python -m tile_accelerator.studies --output runs/study-001 --simulator build/tile-sim
```

Output directories must be new. The demo retains its binary, disassembly, source map, numerical comparison, command counters and analytical report. Studies compare compiler scaling, tiles, memory capacities, compute rates, DMA overlap, and sensitivity to bandwidth/energy assumptions.

For a direct build:

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off cpp/simulator.cpp -o tile-sim
```

For PyTorch export and IREE integration, the tested CPU setup is:

```sh
.venv/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[test,iree]'
```

## The execution path

```text
Supported PyTorch graph or NumPy weights
        → verified shape/operator contract
        → tiles and scratch allocation
        → DMA, matrix and epilogue instructions
        → versioned binary
        → C++ functional engine
        → output comparison
```

The CNN's 3×3 stage can use a native depthwise feature or a generic matrix mapping that gathers patches directly into scratch. Pointwise convolutions and MLP projections use the tiled matrix engine. Unsupported patterns fail compilation with a diagnostic. The optional IREE bridge executes target binaries across an IREE VM native-module boundary using the same C++ engine. It is a host integration; no proprietary hardware or custom HAL driver is implied.

## What is measured

Binary execution provides numerical outputs and command counts. A separate model predicts cycles from resource rates and event dependencies, and estimates energy from action counts. Its area quantity is an illustrative proxy. These are explicit architecture assumptions, not silicon measurements or a cycle-accurate simulator.

FP16 mode rounds operands to binary16 with nearest/ties-to-even behavior; accumulation, bias and output remain FP32. Serialization and interpreter containers remain FP32. This is a logical precision experiment and does not claim packed-memory savings.

[Binary and runtime contract](docs/isa.md) · [Evaluation and limits](docs/evaluation.md) · [Recorded results](docs/results.md)
