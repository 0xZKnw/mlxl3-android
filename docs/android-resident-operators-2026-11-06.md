# Android resident operators and split-K EXL3 — 2026-11-06

Experiment ANDROID-PERF-2026-11-06-16, preregistered in [opti.md](../opti.md).
Baseline executable `45cc54a9e1cebabb585c205b9ed5ad86a2f2ec37`; documentation
head `b265a1b`. This report describes the bundled implementation, not a measured
phone speedup. Publication receipt is appended after inspecting CI and the APK.

## Evidence and research

The user's S24/Xclipse 940, LFM3-1.7B EXL3 nominal 4.49 bpw, returned 13 tokens
at 3.9060928 tok/s. [Selected exact JSON fields](measurements/s24-45cc54a-summary.json)
retain their provenance; the original full report remains in the conversation.
Per token: 102.20 ms EXL3 matvec, 144.76 ms all measured device intervals,
217.68 ms host fence wait, 41.78 ms overlapping host head latency. These intervals
exclude group barriers and queue boundaries, so their difference cannot all be
assigned to CPU wakeups. Thermal status was 0, but this is no controlled A/B.

Primary sources consulted:

- [EXL3 Vulkan PR #204](https://github.com/halo-box/strix-llama.cpp/pull/204),
  Nathanw1014, pinned `bd64c24736eab68a006c1f44f99da3b994ba098c`: register-window
  trellis decoding, paired FP16 codebook arithmetic and wave32. Earlier attribution
  remains in NOTICE. Its desktop Strix Halo results are not Android measurements.
- [AMD RDNA performance guide](https://gpuopen.com/learn/rdna-performance-guide/)
  and [occupancy explanation](https://gpuopen.com/learn/occupancy-explained/): wave
  width, register pressure and workgroup occupancy are workload-dependent. They
  motivate independent small groups, not a guaranteed speed multiplier.
- [Vulkan pipeline subgroup requirements](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageCreateInfo.html):
  request size32 only with enabled feature, supported compute stage and limits.
  The S24 report advertises min32/max64 and compute support. Software Mesa tests
  exercise the algorithm; they do not validate Samsung's size-control compiler.
- [Vulkan synchronization](https://docs.vulkan.org/spec/latest/chapters/synchronization.html):
  execution and memory dependencies must cover shared scratch, history, GPU-written
  activations and final host readback. Fences remain bounded at 30 seconds.
- [AMD shader statistics](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetShaderInfoAMD.html):
  VGPR/SGPR, LDS and scratch usage are optional, per pipeline. Unsupported queries
  are reported explicitly, not interpreted as zero spills.
- [vllm.cpp EXL3 Vulkan scope](https://github.com/mudler/vllm.cpp/blob/main/.agents/specs/backend-vulkan-exl3.md):
  portable native EXL3 and synthetic software Vulkan evidence; no applicable S24
  performance baseline established by the inspected material.
- [llama.cpp Xclipse memory report #7351](https://github.com/ggml-org/llama.cpp/issues/7351):
  user report on S24+ Exynos 2400, reporting memory pressure with Vulkan. This is
  relevant caution about expansion/copies, not evidence our current allocation is
  the bottleneck. We retain compressed EXL3 and add only bounded scratch/history.
- [Scaling On-Device GPU Inference, CVPRW2025](https://openaccess.thecvf.com/content/CVPR2025W/EDGE/papers/Tang_Scaling_On-Device_GPU_Inference_for_Large_Generative_Models_CVPRW_2025_paper.pdf):
  primary mobile GPU study covering quantization and fusion across Adreno/Mali.
  It does not benchmark EXL3 or this S24's Xclipse; no throughput claim is borrowed.

Searches for EXL3 Vulkan Android/phone benchmarks found no verified applicable
phone result. That is a search limitation, not a claim none exists. GGUF/llama.cpp
on an unspecified S25 Ultra backend is not a matched EXL3/S24 experiment.

## Implemented bundle

1. **Split-K groups32.** Four independent groups replace one128-thread group.
   Each lane retains exactly the previous FP32 FMA order. Sixteen partials per
   output preserve the previous sequential reduction order in post-Hadamard,
   before its FP16 boundary. Raw test/API multiplication uses a separate reduction
   dispatch, still one submission. Production adds no reduction dispatch. Extra
   scratch is 2 MiB at the maximum32768 outputs. Native arithmetic still undergoes
   the exhaustive GPU codebook self-test; unsupported size/pipeline creation uses
   the previous128-thread path. No autotuning or rejected tiled4 path is revived.
2. **Resident MLP.** One upload/submit/fence for W1, W3, SwiGLU and W2. Only W2
   returns to the CPU. Cached commands and existing global input scratch are
   reused; the activation overwrites it only after both input projections finish.
   Matrix ownership remains exclusive; no unsound Buffer Sync/Arc workaround.
3. **Resident convolution.** One upload/submit/fence for in-projection, seven-step
   FP32 ring history/FP16 boundaries, gating and out-projection. A completed fence
   precedes host position-header updates. Position0 logically resets the ring:
   old slots are ignored until overwritten. The full model adds about0.71 MiB of
   GPU history plus equal-sized FP32 convolution weights and tiny headers.
4. **Four-row dense kernel.** Shared activation loads, vector accumulators and
   reductions serve four output rows per workgroup. Each row retains the prior
   FP32 FMA and tree-reduction order, including BF16, odd inputs and row tails.
5. **Diagnostic schema3.** `device_queue_ns`/samples bracket whole submissions,
   including the final visibility command, independently of matrix/operator
   intervals. `device_swiglu_ns` and `device_convolution_ns` join device totals;
   queue times overlap them and must not be added again. Post-Hadamard now includes
   split-K reduction, so matvec intervals alone are not comparable across kernels.
   GPU history and CPU KV cache are reported separately. Optional AMD compiler
   statistics expose registers/LDS/scratch for the actual selected pipelines.
6. **CI.** New operator/hot-shape/dense regressions run normally and under actual
   Khronos synchronization validation. Shader validation now propagates any
   spirv-val failure through xargs instead of ignoring an individual find -exec
   failure. APK generation and existing small/large UI checks remain automatic.

For24layers/13conv/11attention, decode submissions are structurally108→71
(24MLP +13conv fences removed,34attention +1head retained). Prefill remains
per token:107→70 plus one final head. The reduced three-layer fixture measures
15→11decode and14→10prefill, not a phone throughput improvement.

## Verification and limits

Local Rust1.98.1, Kani0.68.0/CBMC6.11, NDK27.2.12479018 glslc, Mesa lavapipe.
No physical Apple GPU benchmark or simultaneous real model inference was run.

- `cargo test --locked`:21CPU tests pass (18lib,3checkpoint). Nine Vulkan lib tests,
  three Vulkan integration tests and one complete reduced Engine journey are
  explicitly selected on lavapipe. The optional microbenchmark remains unrun.
- EXL3:96 combinations (eight widths, three codebooks, four shapes) match the
  previous raw kernel bit-for-bit and independent CPU/f64 reconstruction. Actual
  MLP shapes2048→7680/K4 and7680→2048/K5 also match the prior kernel bit-for-bit.
- Dense:30 F16/BF16, odd/even input, output-tail combinations bit-exact to the
  frozen old kernel; existing independent CPU expectations and chunk tests pass.
- Resident operators: both native and forced portable paths; three MLP inputs
  compared to sequential projections + CPU activation at0.0005 +0.002relative;
  fifteen convolution steps including wrap/reset match the CPU boundary bit-exact.
  Invalid shapes, NaN/Inf, wrong input length and context position reject before
  submission; zero activation remains zero. Logical one-submit assertions pass.
- Full Engine fixture:128 nonzero finite logits retain the historical independent
  fixture and unchanged0.0005 +0.002relative tolerance; prefill/decode, repeated
  reset, cancellation and context errors pass. This is not full1.7B validation.
- Counterexample check: temporarily replacing convolution history count by1
  fails at position1. The shader was restored and the affected test rerun green.
- `cargo kani --lib`:14 harnesses,1426 obligations,32 cover properties satisfied;
  three unreachable standard-format checks. New production Rust helpers are
  checked over symbolic usize/u32 inputs: dense group bounds, capability gating,
  and convolution state allocation≤152909floats. This bounded source check does
  not prove GLSL, Vulkan synchronization, driver execution or entire inference.
  The previously disabled unsupported floating Kani harness remains unexecuted.
- Strict host and ARM64 Clippy, formatting, compilation, SPIR-V validation,
  actionlint and diff checks run; exact CI/packaging results are appended below.

Raw local logs: `build/perf16-*` (ignored, temporary, not a durable hosted archive).
The meaningful mutation failure is retained alongside the successful rerun.

QKV+gate fusion, GPU attention/norms, and batched prefill are not implemented in
this bundle. They require further numeric/state work. No20tok/s claim, phone
installation or full-checkpoint speed/correctness validation is made here.

## Publication receipt

Executable source **`2d4b1ecb4bb7d100208a1a10271c730fba64a446`**.
[Run42303987207](https://github.com/0xZKnw/mlxl3-android/actions/runs/42303987207):
all five jobs completed successfully, individually inspected.21CPU,13Vulkan and
12UI tests (six per emulator size); the grouped/operator/Engine checks were also
run under Khronos validation1.3.275 with synchronization validation enabled and
the layer actually loaded. No validation errors/hazards in its checked log.
JVM unit test/lint/build tasks succeed; unchanged JVM unit results are FROM-CACHE.
Kani14-harness full suite:1426 obligations/32covers, three unreachable formatting
checks; the separate setup harness is not counted twice. Optional benchmark skipped.

[APK artifact13544369936](https://github.com/0xZKnw/mlxl3-android/actions/runs/42303987207/artifacts/13544369936):
ZIP13,325,562bytes, SHA-256
`b9092d3c42d56e18cba8d4291d7309550a910df2571cfdde9afcf37972b060fb`.
Installable **`debug/app-debug.apk`**,34,289,948bytes, SHA-256
`a57435ce89c0df5a1f5efa348a4a297da5a379c1bcabba415c0764f75ad85fa2`.
The archive digest and SHA256SUMS manifest match. APK signaturev2, zipalign16K,
both ARM64 libraries' LOAD alignment0x4000, exact source SHA and new operator,
wave32, head and diagnostic markers in the native engine were verified.
Engine7,231,760bytes, SHA-256
`f403d86b6e51d199d9fd75051557e7b6a029c8d6121a743f2d22e8a44dde7640`.
[Machine-readable receipt](measurements/apk-2d4b1ec.json).

APK checks used `apksigner verify --verbose`, `zipalign -c -v -P16 4` and NDK
`llvm-readelf -h -lW`. A display-only parser initially tried to print the last line
of quiet zipalign's empty log; the actual check succeeded, and the verbose check
was rerun successfully. No package failure was hidden. Logs are retained at
`build/perf16-ci-*`, `build/perf16-apksigner.log`, `build/perf16-zipalign.log`;
local APK and receipt at `build/apk-2d4b1ec/`.

Status: **public CI-built APK**, functional software checks passed within their
stated domain, **phone performance unmeasured/inconclusive**. No release or phone
installation. No optimization experiment or physical benchmark left running.
