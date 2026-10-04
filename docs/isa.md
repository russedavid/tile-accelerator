# TILEAC01 binary and runtime contract

All fields are little-endian. The 36-byte header contains an 8-byte magic, a uint32 version (1), instruction count, global-memory element count, scratch element count, input element count, output offset and output element count. Each instruction is sixteen uint32 fields (64 bytes). Constants follow as IEEE float32 values. The input replaces the initial input extent; every request starts with new memory, scratch-initialization flags and event state.

The instruction fields are opcode, addresses a/b/c, dimensions m/n/k, flags, event, dependency, stride, auxiliary and four reserved zeros. Addresses and extents count float32 elements. Matrix tiles in scratch are compact row-major. Global DMA rows use the explicit source/destination stride. Event 0 is already complete; instruction `i` produces event `i+1`. Dependencies may name only preceding events.

| Operation | Meaning |
|---|---|
| LOAD2D | Copy m×n values from global a with row stride to compact scratch b |
| STORE2D | Copy compact scratch a to global b with row stride |
| ZERO | Initialize m scratch values at a |
| MATMUL | Accumulate A[m,k] B[k,n] into C[m,n]; flag1 accumulates. Auxiliary is the second accumulator dependency |
| EPILOGUE | Add a length-n bias; flag1 applies ReLU; flag2 adds a residual tile |
| WAIT | Explicit dependency wait in the timing model |
| DEPTHWISE | Optional native 3×3, stride-one, padded depthwise operation on HWC values in global memory; auxiliary locates bias; flag1 applies ReLU |
| HALT | Final instruction; must be last |

The functional C++ engine executes instructions in their serialized order. A separate resource/event model estimates overlap; functional execution is not a timed microarchitecture simulation. Double buffering uses different A/B slots and explicit reuse/accumulator dependencies. The initial model has one DMA engine, one matrix engine and one vector engine.

The serialization and C++ memory containers use float32. LOAD flag1 rounds operand values to IEEE binary16 (nearest, ties-to-even), while accumulation, bias and output remain float32. DEPTHWISE flag2 selects the same operand rounding. This is a logical precision model; it does not claim packed physical memory. Integer quantization is not supported. The native depthwise unit's direct-memory path is explicit in analysis and does not claim a realistic cache or SRAM implementation.

The loader validates size, version, limits, dependencies and finite input/constants. Execution validates bounds, initialization and supported flags. Invalid input returns a nonzero exit code with a diagnostic. The Python runner imposes a timeout and verifies output length/finite values. The binary format is intended for local experiments, not as a hardened general executable format.
