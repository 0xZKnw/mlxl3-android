# Verification — 5 November 2026

Scope: portable EXL3 rounding, Android chat redesign and keyboard handling,
and reproducible Android build tools. Local checks below use the final source
of this change, Rust 1.98.1, JDK 17, NDK 27.2.12479018, Gradle 9.3.1,
AGP 9.1.1 / built-in Kotlin 2.2.10, SDK 37.0 / build tools 36.0.0.
Raw logs are local ignored `build/` artifacts, not durable CI evidence.
GitHub results must be inspected separately for the exact published commit.

## Regressions reproduced

- Previous shader on Mesa lavapipe: K=1 Default output 1 was -2.2867622,
  expected -2.2864646911621094. The new explicit integer FP16 nearest-even
  rounding passes without relaxing `1e-4 + 2e-5 * abs(expected)`.
  Logs: `build/lavapipe-before-retry.log`, `build/lavapipe-after.log`.
- Temporarily restoring the previous screen makes the assistant text contrast
  test fail. Restoring the approved design passes all five UI tests.
  Log: `build/ui-contrast-before.log`.

## Passing checks

- `cargo fmt --manifest-path engine/Cargo.toml --check`.
- `cargo clippy --manifest-path engine/Cargo.toml --locked --all-targets -- -D warnings`.
- `cargo clippy --manifest-path engine/Cargo.toml --locked --lib --target aarch64-linux-android -- -D warnings`.
- `cargo test --manifest-path engine/Cargo.toml --locked`: 12 CPU tests pass;
  four GPU tests are intentionally ignored here and run explicitly below.
- Explicit Vulkan lib codebook test: all 3 x 65,536 states match the Rust
  codec bit-for-bit on MoltenVK 1.4.2 and Mesa lavapipe LLVM 21.1.4.
- Explicit Vulkan integration tests: two tests pass on Mesa, covering all
  24 EXL3 bitrate/codebook combinations, dense F16/BF16 and invalid shapes.
- Explicit reduced-model flow test passes on Mesa: nonzero EXL3, hidden128,
  convolution/full/sliding layers, finite nonempty logits, tokenizer,
  reset, cancellation and context limits. This is not the 1.7B checkpoint.
- `scripts/build-native.sh`: ARM64 release build using the NDK shader tools.
  Generated SPIR-V passes `spirv-val --target-env vulkan1.1`.
- `./gradlew --no-daemon :app:testDebugUnitTest :app:lintDebug :app:assembleDebug :app:assembleDebugAndroidTest :app:connectedDebugAndroidTest`:
  three JVM tests, five UI tests pass, zero failures/skips; strict lint passes.
  Android 15/API35 ARM64 emulator, 1080x2400; real IME visible and multiline
  composer geometrically above its top, send control visible and enabled.
  Screenshots: `build/ui-approved/{welcome,conversation,keyboard}.png`.
- `actionlint .github/workflows/android.yml`; `shellcheck scripts/build-native.sh`.
- APK: `apksigner verify` and `zipalign -c -P 16 4` pass. Packaged native
  library matches the release output; no model weights bundled.

Local APK SHA-256:
`4b36f17ca2fbbb471dd04ccd71e8a236dccf781ed36faf392acc926cadbb860a`
(34,049,240 bytes). A CI rebuild has a different debug signing key/checksum.

## Source verification and limits

`cargo kani --manifest-path engine/Cargo.toml --lib`: five harnesses pass with
Kani 0.68.0 / CBMC 6.11.0. Domains: bitrate 1..8, pack logical index below
16*K / bit below16, unpack state below256, inverse permutation over all256
indices (unwind260), and checked context budgets over full symbolic usize64.
422 obligations succeed, eight cover checks are satisfied. One formatting
internals overflow assertion is unreachable. No production stub was used.

The optional floating-point greedy harness (four logits, two token IDs,
unwind6) hits a Kani compiler internal error; it remains **unverified** and
opt-in with `RUSTFLAGS='--cfg kani_float_checks'`. Logs and exact command are
recorded in opti.md. These bounded CPU checks do not prove Vulkan, JNI/Kotlin,
concurrency, memory usage, or complete-model generation.

Physical S24/Xclipse behavior, full 1.7B numeric state/logit parity, real process
memory, throughput and thermal behavior remain unverified. No speed campaign,
phone installation or release was performed. See s24-validation.md.
