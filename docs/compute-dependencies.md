# Compute dependencies and adjacent RMSNorm

Every recorded compute command starts with a COMPUTE_SHADER-to-COMPUTE_SHADER
memory dependency (SHADER_WRITE to SHADER_READ | SHADER_WRITE). All producers and
consumers, including the layer copy, are compute dispatches. This preserves
execution order for write-after-read scratch reuse as well as write visibility
for read-after-write/write-after-write reuse between prepared command buffers.

A requested intermediate dispatch barrier is retained. This includes the
Hadamard/matvec/reduction phases and dense-head/argmax dependence. A terminal
compute barrier is redundant: the next command has its entry dependency, or the
submission ends with COMPUTE_SHADER-to-HOST / SHADER_WRITE-to-HOST_READ followed
by a fence wait. Standalone commands retain their own final host dependency.
Host writes use coherent storage and happen before queue submission.

This is not permission to remove the entry barrier or batch-end host dependency.
Any future transfer dispatch or async queue must revisit these stage scopes.
No speedup is asserted until measured without instrumentation on the target GPU.

## Validation

`resident_chained_norm_dependencies_preserve_half_boundaries` repeats a
post-RMSNorm + residual, pre-RMSNorm, and copy-back sequence using persistent
command buffers with instrumentation both on and off. It compares both hidden
residual state and next input to the CPU reference, exercising RAW, WAR and WAW
hazards. The existing CI `resident_` filter automatically includes it in both
normal and synchronization-validation runs. CI also runs full reduced-model
flow and grouped-projection tests. Local toolchains were unavailable when this
patch was authored; these checks must pass before publishing its APK.

## Adjacent normalization fusion constraints

A post-normalization followed by residual addition and pre-normalization is not
one RMSNorm. A valid fused implementation must compute:

1. scale1 from projected input, round normalized result to FP16;
2. add that result to the FP32 residual, retaining the hidden output;
3. scale2 from this updated residual, then round pre-normalized output to FP16.

It must keep both weight vectors, both epsilons, both reductions, and each FP16
boundary. A single workgroup can synchronize the stages through shared memory,
but this needs a dedicated shader, live-buffer lifetime validation and a matching
serial/parallel/subgroup reduction variant. Algebraic scale combination,
dropping the intermediate FP16 round, or relying on workgroup barriers across
multiple workgroups is invalid. The dependency patch deliberately leaves all
normalization arithmetic and outputs unchanged.

## Optional exact fused implementation

`set_fused_norm(true)` now requests the dedicated `rms_fused` shader. It is not
promoted as a speedup or default without S24 measurements. It requires validated
parallel normalization and retains unfused commands as the fallback. Shapes
larger than 2048 retain the unfused path. A startup check compares both outputs
bit-for-bit against two existing parallel GPU dispatches, and also checks the
CPU normalization reference, on counts 1/127/128/129/2048, zero, mixed signs,
small numbers and FP16 extremes. One case uses different epsilons to ensure the
two scales are independent. Startup failure leaves the ordinary pair active.

The shader uses one 128-thread workgroup, about 8.5 KiB of shared memory,
two parallel reductions in the same order as the reference, and explicit shared
barriers. It writes the residual hidden vector AND normalized next input. The
layer plans retain both weight buffers and unfused command buffers; graph
assembly substitutes only the adjacent validated pair and profiling reads only
the commands actually executed. The reduced full-model fixture enables this
experiment and covers convolution, attention, reset and cancellation. The
convolution fixture also compares fused and unfused modes across the ring cache.
