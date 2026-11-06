# EXL3 kernel selection after the S24 regression

The user reports **1.9 tok/s with APK d7116bd**, versus **2.7 tok/s before**,
and confirms the same prompt and 4.49 bpw model. Thermal state, repeated trials
and exact context were not recorded. This is negative device feedback, not a
controlled attribution of the entire difference to one change.

Independent Q/K/V/gate and FFN batching is removed from the production path.
Its input buffers and unused grouping API/test are removed. The previously
qualified scalar path from `38f2d17` is restored; the separate dense-head batching
and tok/s caption remain. The three-layer fixture again checks **22 submissions
per logits-producing forward**, not 13. Previous numeric evidence is preserved.

## New kernel

`qmv_tiled.comp` processes four adjacent outputs with 128 threads. It reuses the
input and compressed tile, calculates bitstream offsets outside the inner loop,
and specializes the EXL3 bit width at pipeline creation. Every output keeps the
old scalar FMA sequence and the same 128-way reduction tree. It uses core 32-bit
types, 2 KiB shared memory and no subgroup/FP16 extension. It supports all 1..8
bit widths; **4.49 bpw is a mixture**, not a request to treat every layer as K4.
There is no dense reconstruction or checkpoint conversion.

The rationale is compiler constant propagation and neighbouring memory accesses:

- [Khronos specialization-constant sample](https://docs.vulkan.org/samples/latest/samples/performance/specialization_constants/README.html)
  explains why pipeline constants permit optimizations unavailable to push constants.
- [AMD memory-coalescing guide](https://gpuopen.com/learn/gcn-memory-coalescing/)
  explains the cost of scattered accesses. This is architectural guidance, not
  a measurement of the Samsung driver.
- [AMD occupancy guide](https://gpuopen.com/learn/occupancy-explained/)
  also explains why higher per-thread work/register use can hurt occupancy;
  therefore the tiled version is **not forced on every Android GPU**.

## Selection on the actual device

`Linear::load` calls `Gpu::prepare_exl3` once per padded input/output shape,
bit width and codebook in a session. Both kernels use the same resident weights,
nonzero deterministic input, validation, upload, submission, fence and readback.
Each gets one warmup. Nonempty finite results must match bit for bit first.
Four alternating pairs then measure **synchronous host latency**, including the
same wrapper costs; the numeric comparisons run outside the timed interval.

The tiled kernel is selected only when all checks match, samples are finite and
positive, each candidate's max/min is at most 1.5, and its median beats the
scalar median by more than 5%, with a >5% win in at least three of four pairs.
Otherwise the scalar kernel remains selected.
Unsupported tiled pipeline creation/readback falls back; an abandoned/timed-out
device remains an error. Repeated shapes reuse the session choice.

This qualification costs additional work **during model loading**, not decode.
It is not a full-model benchmark or a guarantee against thermal/context changes.
`done` events include `vulkan_tiled_shapes` / `vulkan_tuned_shapes` for diagnosis;
the visible UI continues to show only tok/s beneath the reply.

## Focused checks

- `cargo fmt`, strict host Clippy, `cargo test --locked`, `actionlint`, diff check,
  NDK `glslc -O` and `spirv-val --target-env vulkan1.1` pass. **16 CPU tests**;
  seven GPU/benchmark tests ignored by default, then six functional Vulkan
  checks run explicitly. Initial Clippy range-loop warning was fixed and rerun.
- Software Vulkan: **72 bit-exact tiled/scalar cases**, K1..8 × three codebooks ×
  128×128, 256×384 and 2048×512. Existing independent CPU reconstruction checks
  cover all bit widths/codebooks; codebook check covers 196,608 values. Tests
  exercise production selection, cached tuning, invalid dimensions and retry.
- A temporary wrong packed-tile stride fails at 256×384/K1/default; restoring
  the real shader makes the test pass. No mutation remains.
- Full reduced Engine fixture passes prefill/decode, the saved 128-logit oracle
  from the sequential source, reset, cancel and context errors. Cross-driver
  oracle tolerance remains `0.0005 + 0.002 * abs(reference)`; no relaxation.
- `cargo kani --lib --harness tiled_groups_cover_exactly_valid_outputs` checks
  arbitrary `usize` outputs: accepted iff a multiple of 16 in 16..32768;
  4..8192 groups cover exactly four outputs each. **105 obligations and two
  boundary covers pass**, without input assumptions or unwinding truncation.
  This is a source-level bounded model check of the Rust dispatch helper,
  not a proof of shader accesses, GPU drivers, timers or whole inference.
  Timing selection has deterministic failure/boundary tests, not a floating-point proof.

Local tools: Rust 1.98.1, Kani 0.68/CBMC 6.11, NDK27.2.12479018, emulator Mesa
lavapipe. Compilation and Mesa runtime environments were kept separate. Logs:
ignored `build/perf08-*` (temporary); regression tests and this report are tracked.
No hardware GPU benchmark, concurrent desktop profiling load or full 1.7B run
was performed locally. **New S24 tok/s, load overhead, thermal state and complete
checkpoint parity remain unmeasured.** CI/APK receipt is recorded in `opti.md`.

Published source: `85e9db1559b8d6e3d50e68539dc548fe21810300`.
[Run 42246844081](https://github.com/0xZKnw/mlxl3-android/actions/runs/42246844081)
passes all five jobs: 16 CPU tests, six explicit Vulkan checks, ten UI tests and
eight CPU Kani harnesses (686 obligations, 15 covers; two unreachable internal
formatting checks). The optional historical performance benchmark was skipped.
The software tuning test's internal timings were not retained as benchmark
evidence; no speedup is claimed from its selection.

[APK artifact](https://github.com/0xZKnw/mlxl3-android/actions/runs/42246844081/artifacts/13518731858):
`debug/app-debug.apk`, 34,014,540 bytes, SHA-256
`8151ba6655891263fde026db1d80677f972fc3654e83ba23806bc68874548768`.
Downloaded ZIP/APK digests, signature, 16 KiB zip alignment and ARM64 packaging
were checked. APK job 3m14s; no physical phone installation or release.
