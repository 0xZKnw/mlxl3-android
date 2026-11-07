# EXL3 multiwave experiment

Opt-in only, via `set_exl3_multiwave` / `NativeEngine.setExl3Multiwave` before model load.

Four wave32 subgroups calculate four adjacent output tiles in a 128-invocation
workgroup. The first 16 invocations load an activation tile to 64 bytes of shared
memory. Two workgroup barriers per input tile protect write/read/reuse. Trellis
windows and FMA accumulation order are unchanged. Tail waves execute valid
clamped reads and skip output writes only after the last workgroup barrier.

K4 and K5 only. Native FP16 must already pass its codebook qualification and the
existing wave32/subgroup-shuffle capability gate must pass. On pipeline creation
or bit-exact self-test failure, retain the existing projection pipeline. Device
loss is propagated, never treated as a recoverable kernel failure.

Startup qualification compares all split partials bit-for-bit with the native
single-wave kernel: two shapes (128x48 tail case and 256x128), codebooks 0/1/2,
splits 1/2/4/8 when device-supported, and deterministic signed inputs. This is
kernel qualification, not full-model output parity. The CI Vulkan test exercises
qualification where wave32 is supported and explicit unsupported/disabled paths.
Software Vulkan passing does not establish Samsung performance.

This variant takes precedence over MCG/unroll selections for eligible matrices;
the ordinary prepared input/output Hadamard passes and split settings remain.
Diagnostics report requested state, active bitrates and distinct projection names.

Compare the same APK with this toggle alone off/on, profiling off and parallel
RMSNorm on, same fixed prompt, phone cooled between runs. Alternate run order and
repeat. Measure whole-model throughput and correctness; do not infer a win from
fewer activation loads. The extra barriers and reduced workgroup parallelism may
make it slower, particularly for small projections. Keep it off by default until
on-device measurements justify enabling it.
