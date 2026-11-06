# Android performance batch after the 1.7 tok/s report

Baseline source: `85e9db1` / documentation `18bd2b2`. User measurements are
2.7 → 1.9 → 1.7 tok/s on S24 Exynos, EXL3 4.49 bpw, same prompt. No attached
ADB device is available. These negative results are retained; their precise
physical cause has not been measured. This batch does not establish 20 tok/s.

## Changes delivered together

- Scratch input/output and dense-head output buffers stay mapped. Copies keep
  their bounds checks; mapping lifetime ends before freeing memory. Fence and
  timeout handling remain. Weight buffers still map only for initial upload.
  With 153 scalar layer projections and an eight-chunk head, the old path used
  315 map plus 315 unmap calls per decode token. This is an API-call count,
  not a measured throughput improvement.
- Memory selection prefers DEVICE_LOCAL within the allowed HOST_VISIBLE and
  HOST_COHERENT types, keeping the first compatible fallback. Selected flags,
  device type/API/driver and logical buffer bytes are included in diagnostics.
  Logical bytes are distinct from Android process PSS and available system RAM.
- The codebook lookup buffer packs two exact FP16 codewords per uint32 instead
  of storing them as FP32. Working set **786,432 → 393,216 bytes**. This changes
  neither quantization nor decoded values; no bandwidth/speed multiplier claimed.
- Immutable SU/SV scales are rounded to FP16 once during loading; repeated
  conversions inside every projection are removed. Nonfinite/overflowing scales
  fail loading. Input/output rounding and Hadamard order remain unchanged.
- RoPE computes each angle/sine/cosine once per position/dimension, then reuses
  it across heads. The per-head numeric formula is unchanged.
- Automatic load-time calibration and tiled-kernel selection are removed from
  production. The original scalar reference kernel is used. Experimental tiled
  code remains covered by tests, but no app load chooses it or runs its probe.
- Prefill/decode diagnostics record host upload, command recording/submission,
  fence waiting, readback, projection CPU work, total forward time and head time.
  Head is nested in total/model/GPU time; these fields must not be added together.
  Waiting includes GPU execution/driver/scheduling, not isolated GPU timestamps.
  Decode wall time also includes tokenizer, greedy selection and callbacks.

After generation, **LFM3 model menu → “Copier le diagnostic de performances”**
copies the build SHA, GPU information, times, token counts and Android memory,
power-saving and thermal-status snapshots. Thermal status is the Android system
severity, not a GPU temperature. Collection is best effort, outside the native
decode timer; clipboard copy is user initiated. No prompt, answer, model path,
account identifier or device serial is included. A new request clears stale
diagnostics. The visible reply still has only its tok/s caption.

## Research findings and remaining architectural costs

The current EXL3 projection kernel still reads two packed words plus a lookup
per logical weight and uses scalar FP32 FMA/shared-memory reduction. Four-output
vectorization did not remove the two packed loads per weight. Caches can merge
requests, so this is not a count of physical DRAM traffic. There are still 154
synchronous submissions per logits-producing token on the 13-conv/11-attention
configuration, with CPU Hadamard/normalization/attention between projections.
Prefill remains token by token. The BF16 head alone has 500 MiB of weights.

The inspected [llama.cpp Vulkan graph implementation](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L14448)
accumulates graph work into submissions and supports fused operations. The
[EXLlama GEMV source](https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh)
uses cooperative tile loading/dequantization, matrix instructions and GPU
Hadamard stages. Our conservative Vulkan path lacks these architectural
features; copying a CUDA launch configuration would not implement them on S24.
[Khronos memory guidance](https://docs.vulkan.org/guide/latest/memory_allocation.html)
also distinguishes device locality and CPU visibility; one does not imply the
other. No hardware memory-type performance result is inferred from the guide.

The friend's LFM3/Q4_K_M/34 tok/s result is approximate; backend and exact build
are unknown. Termux alone does not establish GPU use. Hardware, head precision,
context and backend must be matched before treating that number as a target.
The new report is intended to separate projection latency, CPU work, head cost
and memory/thermal pressure on the real S24 before the next architectural change.

## Focused verification and limits

Local Rust 1.98.1, NDK27.2.12479018, Kani0.68/CBMC6.11, JDK17. Commands:
`cargo fmt`, strict `cargo clippy --locked --all-targets`, `cargo test --locked`,
explicit ignored Vulkan tests on emulator Mesa lavapipe, targeted `cargo kani`,
`actionlint`, `git diff --check`, Gradle Kotlin/test-Kotlin compilation and lint.
No local hardware GPU benchmark or full-checkpoint inference was run.

- **19 CPU tests pass**. New checks cover memory masks/locality/fallback/32-bit
  limits, scale rounding/overflow and bit-exact multi-head RoPE reference.
- **Seven functional Vulkan checks pass**: all 196,608 codewords bit-exact
  against independent procedural shader/CPU, 72 tiled/reference cases, persistent
  mapping reuse and copy errors, three projection/head tests, full reduced Engine
  fixture. Saved 128-logit oracle and tolerance are unchanged; diagnostic phase
  accounting and absence of prompt text are checked.
- A temporary first-compatible-memory mutation fails the regression, then
  restored code passes. Initial Clippy test-style failure was fixed.
- Kani: actual memory helper with four symbolic flag words and arbitrary mask,
  unwind7, **190 obligations/2 covers**, one unreachable formatting obligation.
  Earlier tuple/min comparator version failed a Kani assertion while its printed
  concrete inputs passed native Rust. An explicit loop avoids that discrepancy
  and passes the same unchanged harness. Counterexample/logs are preserved.
  Packed codebook offsets: arbitrary u32 ID/u16 state, **27 obligations/3 covers**.
  These do not prove Vulkan mappings, unsafe Send, concurrent execution, floats,
  Android layout or complete inference.
- Android Kotlin/test-Kotlin compilation and strict lint pass locally. CI also
  exercises the new copy-to-clipboard journey, previous UI/keyboard journeys,
  native ARM64 build, seven GPU checks and all CPU proof harnesses.

Temporary local evidence: ignored `build/perf10-*`. Source tests, regression
oracle and reports are tracked. S24 throughput, hardware mapping/memory effects,
full-checkpoint numerical parity and thermal causality remain unverified.
Exact source/CI/APK receipt is recorded in `opti.md`.
