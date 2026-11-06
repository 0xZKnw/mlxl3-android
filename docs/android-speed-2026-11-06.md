# LFM3 Android speed investigation — 6 November 2026

The user reports **2.7 tokens/s on a Galaxy S24 Exynos with EXL3 4.49 bpw**.
The exact APK, prompt, context, driver, temperature and power conditions of that
measurement are unknown. There is no controlled on-device before/after result.

## Findings and implementation

The current engine returns to the CPU after each projection. AMD recommends
batching command submissions because each has CPU and GPU costs; Khronos also
recommends avoiding unnecessary synchronous GPU readbacks. These support a
specific hypothesis in this engine, rather than a predicted S24 speedup:

- [AMD RDNA performance guide](https://gpuopen.com/learn/rdna-performance-guide/)
- [Khronos CPU/GPU synchronization sample](https://docs.vulkan.org/samples/latest/samples/performance/wait_idle/README.html)

`Linear::forward_many` now groups the independent FFN gate/up projections and
attention Q/K/V/gate projections into one command submission and fence wait per
group. Each projection retains its own SU scaling, padded input, Hadamard,
weights, output and descriptor set. Dense attention gates can use a different
pipeline in the same submission. Kernels, reduction order, rounding and causal
cache updates are unchanged. Dependent operations still wait for their results.
Four lazy input buffers add at most **512 KiB of logical GPU allocation**;
output buffers are shared with the previously batched dense head. This is not a
measurement of process RAM.

For the supplied 13-convolution/11-attention architecture, projection submission
counts derive as follows (assuming the existing one-submission head):

| Per token | Before this change | After this change |
| --- | ---: | ---: |
| Convolution layers | 13 × 5 = 65 | 13 × 4 = 52 |
| Attention layers | 11 × 8 = 88 | 11 × 4 = 44 |
| Output head | 1 | 1 |
| Total decode | 154 | 97 |
| Prefill without head | 153 | 96 |

These are derived counts, **not a 37% throughput gain**. The reduced three-layer
integration fixture actually checks 13 submissions per logits-producing forward
and the prefill/decode counters emitted through `Engine::generate`.

## Focused verification

Baseline source: `38f2d17331ba52420f788b2f3481e3fd9afc0733`. Rust 1.98.1,
NDK 27.2.12479018 `glslc`, Mesa lavapipe supplied with the Android emulator.
Normal compilation is kept separate from the Mesa runtime environment. No
hardware GPU benchmark or 1.7B model load was run alongside the other chat.

- `cargo fmt`, `cargo clippy --locked --all-targets -- -D warnings`,
  `cargo test --locked`, `actionlint`, `git diff --check`: passed.
  CPU tests: **15 passed**; seven GPU/benchmark tests ignored by default.
- Explicit software-Vulkan checks: **six passed**: independent projection groups,
  codebook parity, three existing projection/head checks, full reduced engine.
  Mixed 3/4/5-bit EXL3, all three codebooks, unequal scales/padding/output sizes,
  F16/BF16 gates, invalid shapes/NaN/empty/oversized groups and retry are covered.
  Groups match sequential projections bit for bit and use one submission.
- Restoring the baseline's three engine source files, then comparing the grouped
  engine, gave **128/128 identical logit bits across eight sequential forwards**
  on lavapipe. The nonzero three-layer fixture exercises convolution, full and
  sliding attention. The checked-in regression oracle is
  `engine/tests/fixtures/lfm3-reduced-logits.json`; its portable CI comparison
  allows `0.0005 + 0.002 * abs(reference)` for cross-driver FP32 differences.
  The first log parser missed a test-runner prefix and rejected the count;
  extracting all eight records corrected the parser before any parity claim.
- A temporary mutation reusing projection 0's transformed input failed the new
  regression as intended. Restored production code and all affected checks pass.
- `cargo kani --lib --harness dense_batch_count_is_bounded_for_four_symbolic_chunks`:
  **131 obligations, zero failures, two covers**, unwind 6, arbitrary `usize`
  row sizes and batch lengths 0..4. One standard-library formatting obligation
  is unreachable. This bounds the production batch-size helper; it does not
  prove Vulkan, shaders, synchronization or complete inference. The existing CI
  runs all seven CPU harnesses separately.

Local raw logs are in ignored `build/independent-*`; they are temporary evidence,
not guaranteed permanent downloads. The regression oracle and tests are tracked.
No S24 throughput, thermal behaviour, peak RAM or full-checkpoint numerical
parity has been measured for this change. CI receipt is recorded in `opti.md`.

## Next candidates, not implemented or measured here

1. **Keep activations on GPU and chain dependent operators.** Port SU/Hadamard/SV,
   normalization and SwiGLU before removing their CPU waits. This addresses the
   remaining round trips; blindly deleting fences would introduce data hazards.
2. **True chunked prefill.** Current prefill runs one token at a time, rereading
   weights for every prompt token. Matrix-matrix work could reuse weights across
   tokens, but convolution state and causal/sliding attention need matching
   reference checks. Grouping independent projections is not batched prefill.
3. **Specialized EXL3 kernels.** Current QMV computes one output per workgroup,
   repeatedly unpacks tiles and uses runtime bit widths. Candidate work is
   multiple outputs per tile, fixed-bit specialization and supported subgroup
   reductions. The [llama.cpp Vulkan matvec source](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl)
   has subgroup/shared-memory variants and tunable block/row/column sizes.
   Its GGUF kernels are not drop-in EXL3 kernels. Capability checks, a portable
   fallback, logits parity and S24 timings are required before choosing a path.
4. **Measure the dense head and CPU attention separately.** The previous change
   removed seven waits from an eight-chunk head; it did not remove its roughly
   500 MiB of BF16 weight reads. Changing its quantization would be a separate
   model/quality decision. CPU KV-cache traversal also grows with context.

On-device phase timings should choose between these next steps. No target
tokens/s or gain is inferred from another model, backend or desktop GPU.
