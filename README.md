# MLXL3 Android

An experimental, custom **LFM3-1.7B EXL3 inference engine for Android**, using
Rust and Vulkan Compute. The Android UI follows MLXL3 Desktop's restrained
dark palette, model header, conversation layout and rounded composer.

**Initial target:** Galaxy S24 Exynos (Xclipse), with portable Vulkan kernels
for later qualification on Adreno and Mali. Android 9+, ARM64 and Vulkan 1.1
are required. No fixed subgroup width, shader FP16 or shader Int64 extension
is required. Support on a device is established by testing, not its GPU name.

## Status

Experimental implementation, with component checks passing in CI. Full-checkpoint numeric parity,
memory usage and sustained thermal behavior on the S24 have **not been validated**.
Exploratory fixed-input S24 measurements are available, but do not establish
full-model correctness or sustained production performance.
This is not a performance release. The APK contains no model weights and
requests no Internet permission.

The engine implements the LFM3 text path: cached EXL3 codebooks, register-window
matrix-vector multiplication and GPU Hadamard/scales, gated width-7 convolution,
full/sliding grouped attention, zero-tail partial RoPE, per-head sigmoid gates,
sandwich RMSNorm, SwiGLU and greedy generation. Projections run on Vulkan;
small operators and attention/cache control currently run on CPU. The codebook
lookup fallback uses an extra 384 KiB per loaded GPU; weights remain compressed.
On qualified shaderFloat16 devices, paired FP16 codebook additions and fixed-bit
shaders replace lookup gathers for Default/MCG. Mul1 keeps exact lookup values.
Startup checks every codebook state in both lanes against CPU before enabling
the native path; unsupported or mismatching devices keep the portable kernel. EXL3
commands are recorded at load time and reused; Q/K/V and FFN gate/up projections
share submissions. Prompts are processed token by token. The S24 reports and
the unmeasured performance limits of the current rework are recorded in
[the native-kernel investigation](docs/android-native-codebooks-2026-11-06.md),
[the grouped-projection investigation](docs/android-grouped-projections-2026-11-06.md)
and [the previous EXL3 investigation](docs/android-exl3-rework-2026-11-06.md).
After generation, the menu can copy timing/device diagnostics. Optional GPU
timestamps separate input Hadamard, EXL3 matvec and output Hadamard from host
fence waits. Samples count measured commands; submissions count shared fences.
The copied report also contains per-shape costs, the eight slowest projections,
CPU operator times, driver information and float/subgroup capabilities.

The model menu includes experimental split-K 1, 2, 4 and 8, an optional
paired dense-head kernel, optional GPU-resident convolution and attention layers, and
optional cooperative K5 and MCG-specialized kernels, and GPU greedy selection.
An optional subgroup-reduction head takes precedence over the paired head and
falls back when subgroup arithmetic is unavailable.
The experimental head row-count selector offers 1, 2, 4 and 8 outputs per
workgroup. A value other than 4 selects the variable-row kernel instead of the
paired or subgroup kernel, so compare these options separately.
Changes apply **only after reloading the model**;
on the measured **Samsung SM-S921B (Exynos S24)**, unset preferences now default to
split-K 8, FFN K4 override 4, cooperative K5, GPU convolution and attention,
4 layers per submission, and GPU greedy selection. The reference 4-row dense
head remains selected; MCG, the two-tile loop and alternative head kernels stay
off. This profile follows exploratory 32/128-token reports (up to 12.56 tok/s),
not full-checkpoint parity qualification. Other devices retain split-K 4, CPU
layer boundaries, batch 1 and disabled experimental variants. Existing saved
choices always win, including explicit `false` and the FFN `global` override;
no preferences are silently rewritten on upgrade. Missing values use the
hardware-specific defaults. Diagnostics identify the default profile, while the
engine section continues to report effective settings and kernel fallback.
Native codebook/cooperative self-tests remain in force. These do not validate
whole GPU layers: a load failure is surfaced rather than silently claiming a
successful optimized load; disable GPU layer options and reload if necessary. The copied report
records the effective split count, chosen head kernel, cooperative K4/K5
selection and its startup parity check, plus separate gate/head device times.
These timings overlap other reported scopes and must not be added together.
Different split counts, the paired head and GPU normalization may change
floating-point results. Compare output quality and repeated end-to-end measurements on the
target device before treating any option as an improvement. If a cooperative
kernel fails its startup parity check, the engine falls back to the prior path.
GPU greedy selection applies the FP16 boundary, checks all logits for non-finite
values, applies the tokenizer vocabulary mask, and resolves ties by lowest token ID.
The FFN K4 and K5 menu entries can override the global split count separately
for the 2048→7680 and 7680→2048 shapes. Tapping cycles through global, 1, 2,
4 and 8; reload the model to apply. The diagnostic reports effective choices.
The experimental layer-batch setting cycles through 1, 2 and 4 resident layers
per submission. It only combines adjacent layers whose GPU paths are enabled;
cancellation is checked between batches. The default is 4 on SM-S921B and 1 elsewhere.
The optional K4/K5 two-tile loop is an experiment, disabled by default. It does
not guarantee that the driver actually unrolls the loop or improves throughput.
The optional **Normalisations parallèles** setting distributes resident-layer
RMSNorm reduction across 128 threads. It is disabled by default and applies after
reloading. The FP16 boundary and residual addition are retained, but the FP32
reduction order changes. A startup comparison checks 13,056 values against the
CPU serial path and keeps the original shader on mismatch or pipeline failure.
Diagnostics report `parallel_norm_requested`, `parallel_norm_active` and
`parallel_norm_self_test`. This check does not establish full-checkpoint parity.
See [the experiment and S24 comparison protocol](docs/android-parallel-norm-2026-11-07.md).
The menu's fixed benchmark evaluates 32 prefill and 128 decode steps using a
repeatable sequence of vocabulary IDs. It is not a generated answer. Copy the
diagnostic afterward and alternate configurations under similar thermal conditions.
Android system traces include `MLXL3 upload`, `MLXL3 record commands`,
`MLXL3 queue submit`, `MLXL3 fence wait`, and `MLXL3 readback` slices. Capture
them with CPU scheduling and frequency tracks in Perfetto to distinguish CPU
sleep, scheduling, and GPU work; they do not directly align GPU timestamps.

One model, one conversation, context capped at 2048 tokens, up to 256 generated
tokens per UI request. Thinking is disabled through the official text-only
chat-template convention. No vision, speculative decoding, remote inference
or model downloader.

## Install and load a model

1. Open [Actions](https://github.com/0xZKnw/mlxl3-android/actions/workflows/android.yml),
   select a successful **Android APK** run and download its `MLXL3-Android-…`
   artifact. Install `app-debug.apk` on an ARM64 Android phone.
2. Download a complete `4bpw` variant folder from
   [LFM3-1.7B-EXL3](https://huggingface.co/0xzknw/LFM3-1.7B-EXL3).
   It must contain `config.json`, `tokenizer.json` and every `.safetensors` shard.
   The 8bpw variant is also supported by the format; it requires more memory.
3. In the app, choose **Importer mon modèle**, then that variant folder. The app
   copies its model files to private storage and loads them. Keep sufficient
   free storage for this second copy. Model files must remain immutable while
   loaded; replace them through the importer.
4. Send a short prompt. **Arrêter la génération** requests cancellation between layers and
   bounded GPU dispatches. Keep the app in the foreground during inference.

The debug APK is signed with a CI test key, not a production release key.
Separate CI runs can use different keys; Android may require uninstalling a
previous debug build before installing another, deleting its private model copy.

## Build locally

Use JDK 17, Gradle wrapper 9.3.1, Android SDK `platforms;android-37.0` /
build tools 36.0.0, NDK 27.2.12479018 and Rust 1.98.1. The native build
uses the NDK shader compiler by default; `GLSLC` can override it.

```sh
rustup target add aarch64-linux-android
export ANDROID_NDK_HOME=/path/to/android-sdk/ndk/27.2.12479018
scripts/build-native.sh
./gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
```

`local.properties` can set `sdk.dir=/path/to/android-sdk`.
The native build uses 16 KiB ELF segment alignment. The CI builds ARM64 only,
caches Cargo/Gradle, and never downloads the model. APK creation runs separately
from the formal checks and emulator UI tests.

## Verification

```sh
cargo fmt --manifest-path engine/Cargo.toml --check
cargo clippy --manifest-path engine/Cargo.toml --locked --all-targets -- -D warnings
cargo test --manifest-path engine/Cargo.toml --locked
# Explicit device tests (Mesa lavapipe in CI):
cargo test --manifest-path engine/Cargo.toml --lib vulkan::tests::codebooks -- --ignored --test-threads=1
cargo test --manifest-path engine/Cargo.toml --lib vulkan::tests::native_codebooks -- --ignored --test-threads=1
cargo test --manifest-path engine/Cargo.toml --lib vulkan::tests::stream_qmv -- --ignored --test-threads=1
cargo test --manifest-path engine/Cargo.toml --lib vulkan::tests::prepared_projection -- --ignored --test-threads=1
cargo test --manifest-path engine/Cargo.toml --lib vulkan::tests::grouped_projections -- --ignored --test-threads=1
cargo test --manifest-path engine/Cargo.toml --lib vulkan::tests::persistent_scratch -- --ignored --test-threads=1
cargo test --manifest-path engine/Cargo.toml --test vulkan -- --ignored --test-threads=1
cargo test --manifest-path engine/Cargo.toml --test flow -- --ignored --test-threads=1
cargo kani --manifest-path engine/Cargo.toml --lib
./gradlew :app:testDebugUnitTest :app:lintDebug :app:connectedDebugAndroidTest
```

The Vulkan tests check all 196,608 codebook values bit-for-bit and compare
nonempty finite projection outputs with CPU
reconstruction, for all 1–8-bit EXL3 layouts and all three codebooks. These
tests do not prove complete-model equivalence. Six emulator UI tests cover
contrast, import, sending, cancellation, diagnostic copying and the composer above the real keyboard.
The six UI cases run on two screen sizes and check real keyboard reopening
and cursor anchoring. See [the current engine verification record](docs/android-native-codebooks-2026-11-06.md),
[the previous record](docs/verification-2026-11-06.md)
and [the initial record](docs/verification-2026-11-05.md). Kani properties and limits,
failures and actual verification results are recorded in [opti.md](opti.md).
See [the S24 validation protocol](docs/s24-validation.md) before reporting results.

## Credits

- [MLXL3](https://github.com/0xZKnw/mlxl3): Rust EXL3 codec and Desktop design.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3): EXL3 format and reference algorithms.
- [Nathanw1014 and the ggml authors](https://github.com/halo-box/strix-llama.cpp/pull/204): Vulkan register-window trellis decoding algorithm.
- [Liquid AI](https://huggingface.co/LiquidAI/LFM3-1.7B): LFM3 model architecture and weights.
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers): tokenization.
- [ash](https://github.com/ash-rs/ash): Vulkan bindings.

Code is MIT licensed; see [third-party notices](THIRD_PARTY_NOTICES.md).
Model weights retain their original license and are distributed separately.
