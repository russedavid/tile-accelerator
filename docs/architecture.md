# Architecture experiments

The compiler makes the dataflow explicit: compact A/B tiles are loaded, a C accumulator is initialized, reduction tiles are accumulated, an epilogue is applied, and the output is stored. A tile is legal only when A, B, C, bias and residual fit the scratch budget.

Double buffering adds a second A/B slot. Prefetch commands can overlap the previous tile's compute, while reuse dependencies prevent a slot from being overwritten too soon. The matrix instruction also waits for the preceding accumulator update. The functional binary engine validates request behavior; the event/resource model estimates overlap under a specified DMA and compute rate.

The native depthwise feature is a separate architecture choice. It computes the CNN's 3×3 depthwise stage before the matrix-based pointwise stage. The current functional model exposes its global-memory access and does not assume a hidden cache advantage.

The study sweeps scratch capacity, matrix throughput and output tile width. It reports a Pareto set for estimated latency, energy and an illustrative area proxy. Bandwidth and per-byte energy are varied independently to show sensitivity. The proxy is deliberately identified as such; a technology library or physical implementation would be needed for credible chip area and energy calibration.

The model is not a cycle-accurate simulator. It omits cache behavior, arbitration, bank conflicts, pipeline fill/drain and detailed interconnect. Conclusions are conditional on those assumptions. The compiler and instruction traces make those conditions inspectable.

The IREE VM bridge carries program/input buffers to the independent C++ engine. This establishes a host runtime integration. A physical accelerator driver and silicon validation would be a different hardware integration.
