# Android verification — 6 November 2026

Qualified source: `ee39235a7b39646e74a4bf54d775aefe4c3f419c`.
[GitHub run 42132302512](https://github.com/0xZKnw/mlxl3-android/actions/runs/42132302512):
all five jobs pass: APK, Vulkan, Kani, small UI and phone UI. The publication
adds documentation to this qualified application/engine source.

## Delivered behavior

On Android 11+, the Android content host fits the bottom IME/navigation insets
once. Compose consumes those applied insets. The focused Android view, chat
layout and cursor popup consequently use the same visible viewport. Android
9/10 retain the previous Compose inset path. The text field uses the existing
state-based Compose API; the approved palette and layout are preserved.

EXL3 codebooks are computed once and stored in a 786,432-byte (768 KiB) GPU
lookup buffer. Weights stay packed. Projection reduction/FMA order, FP16
boundaries, model operators, tokenizer and cache semantics are unchanged.
The procedural shader remains a test-only oracle, not an APK runtime option.

The user's S24 checkpoint is reported as **4.49 bpw exactly**. No complete
numerical or throughput qualification of that physical checkpoint is claimed.

## Commands and inspected results

Rust 1.98.1, NDK 27.2.12479018, JDK17, Gradle9.3.1, AGP9.1.1,
Compose BOM2025.04.01, SDK37.0/build-tools36.0.0; Vulkan1.1 SPIR-V.

- `cargo fmt --manifest-path engine/Cargo.toml --check` and strict Clippy
  (`--locked --all-targets -- -D warnings`, plus `--lib --target aarch64-linux-android`)
  pass in CI. `scripts/build-native.sh` builds the ARM64 release library.
- `cargo test --manifest-path engine/Cargo.toml --locked`: **13 CPU tests pass**;
  five GPU/benchmark tests are intentionally ignored by this command.
- Explicit `--lib vulkan::tests::codebooks -- --ignored --test-threads=1` checks
  all **196,608 values bit-for-bit** against both the prior procedural GPU
  shader and the Rust codec. Arrays have checked nonzero lengths.
- Explicit `--test vulkan -- --ignored --test-threads=1`: two tests pass,
  covering all 24 bitrate/codebook combinations, finite nonempty projection
  outputs, dense F16/BF16 and invalid dimensions/IDs. The CPU/f64 projection
  oracle retains absolute1e-4 + relative2e-5 tolerance.
- Explicit `--test flow -- --ignored --test-threads=1`: the nonzero reduced
  LFM3 EXL3 fixture passes load, prefill/decode, logits, tokenizer, reset,
  cancellation and context boundaries. This is not a 1.7B model parity test.
- `cargo kani --manifest-path engine/Cargo.toml --lib --harness codebook_offsets_are_aligned_bounded_and_reject_invalid_ids`
  passes first, then `cargo kani --manifest-path engine/Cargo.toml --lib`
  passes **six harnesses**. Full suite: 449 successful obligations, 11 satisfied
  covers, one unreachable standard-library formatting assertion. The targeted
  invocation separately repeats 27 obligations and three covers.
- New Kani property: every symbolic u16 state and u32 codebook ID; reject IDs
  >=3, preserve codebook/state slot identity, 4-byte alignment and offset+4
  within 786,432 bytes. No assumptions or loop bounds are needed for this helper.
  Prior properties cover K1..8 pack/unpack, inverse permutation256 (unwind260)
  and checked context budgets over usize64. These are bounded CPU source checks.
- Gradle `:app:testDebugUnitTest :app:lintDebug :app:assembleDebug :app:assembleDebugAndroidTest`
  passes; three JVM import tests pass, strict Android lint passes.
- `./gradlew --no-daemon -PuiTest :app:connectedDebugAndroidTest`: five UI cases
  pass on each API35 x86_64 AOSP display: 320x640/mdpi and1080x2400/dpi420
  (**10 executions**, no failures/skips). Keyboard tests type three lines,
  dismiss/reopen the real IME, preserve the draft, assert composer gap0..12dp,
  visible header/send, and cursor-popup screen center inside the composer.
  Screenshots for both profiles were visually inspected.
- `actionlint .github/workflows/android.yml`, `git diff --check`, APK checksum,
  `apksigner verify` and `zipalign -c -P 16 4` pass. No Safetensors in the APK.

Qualified CI APK: 33,975,708 bytes, SHA-256
`dedcc748615ae83122df524a30585e399feb7601dd712f297eefc533a03ac833`.
ARM64 engine SHA-256: `b4c6dfc13345ddf2485bd43f9231a58fa7a09adc2337e448559dc234c97a1509`.
A later CI rebuild uses its own debug signing key/checksum.

## Counterexamples and performance limits

The old screen is reproduced 190px above the keyboard on AOSP. Closing and
reopening the IME caught fixes that only passed the first opening. A later
screenshot exposed a cursor handle near the header despite passing composer
bounds; the new popup oracle detects it (center139,98 versus composer
30,215..298,330). Window flags, focus clearing, custom visibility propagation,
layout nudges and a Compose upgrade were tested and removed after failure.
The final change uses the native host viewport rather than cursor offsets.
History, commands and logs are retained in opti.md; local raw logs are ignored
build artifacts, not guaranteed durable evidence.

The earlier optional Mesa microbenchmark passes bit-exact output checks on
four LFM3-sized projection shapes x three codebooks at K4, with two warmups
and seven alternating pairs. Raw results are committed in
[mesa-353e7e8.json](measurements/android-codebook-lookup-20261106/mesa-353e7e8.json).
This uses synthetic packed weights and host wall times on llvmpipe, not the
user's 4.49bpw checkpoint. Significant drift in2048x7680 cases prevents an
aggregate gain claim. The lookup timing includes production validation while
the procedural reference calls the common compute wrapper directly; these
are descriptive end-to-end dispatch timings, not isolated GPU timestamps.
The benchmark is optional on workflow_dispatch and is not rerun for UI edits.

Physical S24 speed, process memory, thermals, full1.7B logits/cache parity,
Adreno/Mali compatibility and the legacy Android9/10 inset branch remain
unverified. The existing optional floating-point greedy Kani harness still
hits a compiler internal error, as recorded in the previous report. No proof
of Vulkan/JNI/Compose, concurrent behavior or complete-model inference is
claimed. No local engine compilation, prover, model load or benchmark was
started during the other chat's Desktop profiling; qualification ran remotely.
