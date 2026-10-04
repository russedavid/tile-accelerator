# Validation and measurement boundaries

The functional engine and the analytical timing model answer different questions. C++ binary execution checks the numerical result. The resource timeline estimates cycles from event dependencies and stated rates. Energy comes from action counts and supplied component costs. Area is an illustrative proxy. None is a fabricated-silicon measurement.

Tests cover aligned and partial tiles, different input values, reset state, independent NumPy references, a PyTorch-exported CNN, binary round trips, invalid dependencies, uninitialized reads, bad addresses, corrupt headers, unsupported versions, nonfinite values and seeded random shapes. Double-buffered programs must match their serial counterparts. FP16 rounding checks include ties, subnormals, zero and overflow.

The runtime recreates memory and event state for each request. The IREE VM native-module bridge follows the same contract. It is a host integration with a C++ simulator, not a hardware HAL driver. Subprocess/serialization overhead is not included in the modeled device latency.

The current models use original random inputs and weights. They establish arithmetic/encoding correctness, not perception accuracy. For a real model, evaluate its supported domain, source/data licenses, task metric and meaningful slices. A missing demographic annotation is not evidence that bias is absent.

Compiler scaling reports record sizes, command counts and elapsed compilation/encoding. Their initial timings are illustrative observations; they are not a statistically independent complexity proof. Architecture conclusions include bandwidth and energy sensitivity and retain assumptions beside the results.
