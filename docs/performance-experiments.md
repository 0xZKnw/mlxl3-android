# Focused performance experiments

The measured S24 profile keeps GPU greedy, cooperative K5, K4 split4,
parallel RMSNorm and profiling off. Existing preferences are preserved after
the one-time profile migration. New experiments are opt-in, never claims of
measured speedup on Xclipse 940.

The main model menu exposes:
- RMSNorm subgroup: hierarchical subgroup reduction, guarded numerical test.
- Exact RMSNorm fusion: adjacent post-normalization/residual/pre-normalization,
  preserving both FP16 boundaries. Requires validated parallel RMSNorm;
  falls back when subgroup RMSNorm is active. Compare these independently.
- EXL3 multiwave: share activations across four wave32 groups for K4/K5,
  with bit-exact partial-output startup checks and baseline fallback.
- Larger GPU lots: 8/12/24 layers, retaining reset/cancel semantics between
  submissions. The final model output still crosses to CPU before the head.
- Wave head: previous experiment retained for comparison, off by default.

Reload the model after each setting change. Measure baseline and one experiment
at a time with profiling off, identical input and a cooled phone, then alternate
A/B order and repeat at least three times. Use the diagnostic active fields,
not only requested fields: unsupported or failed self-tests fall back.
Only compare combinations after individual checks. CPU/software-Vulkan tests
and small synthetic models do not replace full-checkpoint output parity or
real S24 throughput/thermal measurements.
