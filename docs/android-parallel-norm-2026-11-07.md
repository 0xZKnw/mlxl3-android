# Parallel resident RMSNorm experiment

The user's fixed-input 32/128 benchmark on `903fcd2` reports 12.70056 tok/s,
78.7367 ms/decode step and seven submissions per step. Resident normalizations
and residuals account for 14.1716 ms/step: 96 commands at 147.621 us each.
These are instrumented device intervals, not isolated arithmetic timings.
[Selected exact baseline fields](measurements/s24-903fcd2-parallel-norm-baseline.json)
retain provenance; the complete original diagnostic was supplied in chat.

## Change

`rms_residual.comp` is compiled twice. The original serial reduction remains the
default. The optional variant uses 128 threads, coalesced strided input loads,
FP32 partial sums and a shared-memory tree reduction. It adds 512 bytes of shared
storage. No subgroup extension or fixed subgroup width is needed.

Weight rounding, the normalized FP16 output boundary, residual addition, layer
batching and EXL3 kernels are unchanged. FP32 summation order changes, so this is
not a claim of bit-exact whole-model equivalence. Q/K normalization, softmax and
the final CPU embedding norm are outside this experiment.

The model menu adds **Normalisations paralleles** (displayed with French accents
in the app). It defaults to off, persists independently and applies after model
reload. A startup guard compares 13,056 values against CPU serial RMSNorm at
widths 128/2048, including zero, varying and mixed-magnitude inputs, with and
without a residual. Tolerance is `2e-6 + 0.001 * abs(normalized)`, before residual
cancellation. A mismatch or candidate pipeline failure retains the serial
pipeline. A device timeout remains fatal. Diagnostics expose requested/active
state, guard result and available compiler statistics for both pipelines.

## Verification

Local Rust 1.98.1, shaderc 2025.2 and Mesa lavapipe 25.0.7:

- The focused Vulkan test performs 107,520 comparisons against an independent
  f64 reference, applying FP16 weight/output rounding. Widths 1, 127, 128, 129,
  2047 and 2048 cover both the actual layer size and reduction tails. Cases
  include zero, constant, varying, tiny and FP16 maximum-magnitude inputs, two
  epsilons and both residual modes. Passed; maximum observed absolute error was
  zero for this corpus on this software driver. This does not imply equality
  for all inputs/drivers.
- The startup guard passes and rejects an intentionally wrong copy pipeline.
- The existing complete reduced-model test passes with the parallel option
  enabled in its resident/batched path. Existing reference tolerances are
  unchanged. The fixture covers prefill/decode, reset, cancellation and context
  errors, not the real 1.7B checkpoint.
- Formatting, strict all-target Clippy and compiled SPIR-V validation pass.
- Normal CPU tests: 22 pass; GPU cases remain explicitly ignored there.
- All 13 resident Vulkan tests and the complete reduced-model test pass with
  Khronos synchronization validation enabled. The layer's insertion was checked;
  no validation errors, VUIDs or synchronization hazards were reported.
- The default serial shader compiles to exactly the same SPIR-V bytes as the
  baseline with this compiler (SHA-256
  `692f65c65b0088095cfd6dfedfa0061a3f14e781f316b788aed1f4f24cf4b427`).

Android CI is pending. No physical S24 speedup, sustained thermal result or full-checkpoint
logit/state parity is established by these local checks.

## S24 comparison

Keep the checkpoint, split-K 8, FFN K4 override 4, K5/cooperative kernels,
resident layers, batch 4, greedy and reference head identical to the baseline.
Only change **Normalisations paralleles**, then reload. Warm up and alternate
off/on/off/on under similar thermal conditions; retain at least five runs per
variant rather than selecting the fastest sample.

For each report retain source/APK/checkpoint hashes, the option's requested and
effective states, guard result, actual step count, decode seconds, normalization
intervals, queue time, submissions, temperature/thermal status and PSS. Compare
end-to-end decode latency first. Repeat at longer contexts and compare real
generated outputs/logits with the same checkpoint before qualifying numerical
behavior. The fixed benchmark feeds tokens; it does not test answer quality.

Halving the reported normalization intervals, with all other costs fixed, would
arithmetically change 12.70 to about 13.96 tok/s. This is a sensitivity example,
not a measured or predicted improvement.
