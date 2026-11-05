#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ANDROID_NDK_HOME:?Set ANDROID_NDK_HOME to NDK 27.2.12479018}"
case "$(uname -s)" in
  Linux) host_tag=linux-x86_64 ;;
  Darwin) host_tag=darwin-x86_64 ;;
  *) echo 'Build native code on Linux or macOS' >&2; exit 1 ;;
esac
toolchain="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$host_tag/bin"
export GLSLC="${GLSLC:-$ANDROID_NDK_HOME/shader-tools/$host_tag/glslc}"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$toolchain/aarch64-linux-android28-clang"
export CC_aarch64_linux_android="$CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER"
export AR_aarch64_linux_android="$toolchain/llvm-ar"
# Android 15 supports 16 KiB pages; align native ELF segments accordingly.
export CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS='-C link-arg=-Wl,-z,max-page-size=16384'
cargo build --manifest-path engine/Cargo.toml --locked --release --lib --target aarch64-linux-android
mkdir -p app/src/main/jniLibs/arm64-v8a
cp engine/target/aarch64-linux-android/release/libmlxl3_android.so app/src/main/jniLibs/arm64-v8a/
