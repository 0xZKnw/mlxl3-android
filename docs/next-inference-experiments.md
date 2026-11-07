# Next S24 inference experiments

This change is stacked on PR #8. It does not promote previous experiments to
validated defaults and does not claim a phone speedup.

## Why these changes

The supplied 128-token profile has roughly 4.89 s in EXL3 matvec and 1.48 s in
the output head, versus 0.279 s in parallel RMSNorm. Prioritize matrix/head
work rather than adding more normalization variants. GPU stage timings and
CPU wall time overlap: these figures must not be summed into an end-to-end
speedup prediction.

- **Vector head (opt-in):** loads eight packed F16/BF16 weights per lane with
  a 128-bit load and uses four independent FP32 accumulation chains. This
  targets instruction/dependency overhead, not the number of weight bytes.
  Requires a 2048-wide head and supported subgroup arithmetic; startup checks
  compare F16/BF16 outputs with independent f64 dot products. Different
  summation order means this is not bit-exact full-model parity. On failure,
  the existing selected head remains available. If vector and wave head are
  both requested, vector takes precedence when supported; inspect diagnostics.
- **Hadamard shuffle (opt-in):** preserves ordered FP32 butterflies and FP16
  boundaries while moving the first five stages into wave32 shuffles. Shared
  memory handles only cross-wave stages. Capability and numerical checks must
  pass; otherwise the established shared-memory implementation remains active.
- **Baseline Hadamard cleanup:** omit the final butterfly's shared write and
  two barriers, because output is taken directly from each lane's register and
  no invocation reads the shared array afterward. Arithmetic is unchanged.

Hadamard work is separately timed in the diagnostic; this is not a claim that
shuffle optimizes the entire 4.89 s matvec bucket. More vectorization can increase
register pressure and regress on a given driver. Measure rather than assume.

## Menu and retained settings

Only the two new switches appear in the main experiments menu. Prior experiments
and established controls remain under **Réglages avancés et retour arrière**.
No existing preferences, device defaults or batch size are changed. A single
batch-12 result of 15.82 tok/s versus 15.74 at batch 8 does not establish a robust
winner; the batch-24 slowdown may include thermal/frequency effects.

## Phone validation

1. Keep the same model, prompt, split-K settings and chosen batch size throughout.
   Start with both new toggles off. Reload the model after every setting change.
2. Run the fixed 32-token prefill / 128-token decode benchmark with profiling off.
   Let the phone cool, alternate baseline/candidate order, and repeat each at
   least three times. Record token output, tokens/s, elapsed time, and thermals.
3. Enable **only vector head**, reload, inspect vector_head_requested,
   vector_head_active and vector_head_self_test. Run the same benchmark.
4. Disable vector head and enable **only Hadamard shuffle**, reload, inspect
   shuffle_hadamard_requested/active/self_test, and repeat.
5. Use full GPU profiling only for separate attribution runs. Do not compare
   their wall-clock throughput directly with profiling-off measurements.
6. Check full-checkpoint output/token parity, including multiple prompts and
   longer contexts. Only after individual checks, test both switches together.
   If either regresses or changes outputs unexpectedly, disable it and reload.

Software-Vulkan, CPU and emulator checks are necessary but cannot validate
Xclipse 940 throughput, sustained thermals or complete 1.7B checkpoint parity.
