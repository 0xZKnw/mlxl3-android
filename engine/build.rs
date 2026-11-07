use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=GLSLC");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    for name in [
        "qmv",
        "qmv_stream",
        "hadamard",
        "dense",
        "dense_pair",
        "rms_residual",
        "reduce",
        "swiglu",
        "convolution",
        "dense_reference",
        "codebook",
        "qmv_reference",
        "codebook_reference",
    ] {
        let mut compiler = Command::new(env::var_os("GLSLC").unwrap_or_else(|| "glslc".into()));
        compiler
            .args(["--target-env=vulkan1.1", "-O", "-fshader-stage=compute"])
            .arg(format!(
                "shaders/{}.comp",
                if name == "dense_reference" {
                    name
                } else {
                    name.trim_end_matches("_reference")
                }
            ))
            .arg("-o")
            .arg(out.join(format!("{name}.spv")));
        if name.ends_with("_reference") {
            compiler.arg("-DEXL3_REFERENCE=1");
        }
        let status = compiler
            .status()
            .expect("install glslc (shaderc) or set GLSLC");
        assert!(status.success(), "{name} shader compilation failed");
    }
    for bits in 0..=8 {
        let (source, name) = if bits == 0 {
            ("codebook", "codebook_native".to_owned())
        } else {
            ("qmv_stream", format!("qmv_native_{bits}"))
        };
        let mut compiler = Command::new(env::var_os("GLSLC").unwrap_or_else(|| "glslc".into()));
        compiler
            .args([
                "--target-env=vulkan1.1",
                "-O",
                "-fshader-stage=compute",
                "-DEXL3_NATIVE=1",
            ])
            .arg(format!("shaders/{source}.comp"))
            .arg("-o")
            .arg(out.join(format!("{name}.spv")));
        if bits > 0 {
            compiler.arg(format!("-DEXL3_BITS={bits}"));
        }
        assert!(
            compiler.status().expect("glslc").success(),
            "{name} shader compilation failed"
        );
    }
    for bits in 1..=8 {
        let status = Command::new(env::var_os("GLSLC").unwrap_or_else(|| "glslc".into()))
            .args([
                "--target-env=vulkan1.1",
                "-O",
                "-fshader-stage=compute",
                "-DEXL3_NATIVE=1",
                "-DEXL3_PARTIALS=1",
            ])
            .arg(format!("-DEXL3_BITS={bits}"))
            .arg("shaders/qmv_stream.comp")
            .arg("-o")
            .arg(out.join(format!("qmv_wave_{bits}.spv")))
            .status()
            .expect("glslc");
        assert!(status.success(), "wave32 shader compilation failed");
    }
    for bits in [4, 5] {
        let status = Command::new(env::var_os("GLSLC").unwrap_or_else(|| "glslc".into()))
            .args([
                "--target-env=vulkan1.1",
                "-O",
                "-fshader-stage=compute",
                "-DEXL3_NATIVE=1",
                "-DEXL3_PARTIALS=1",
            ])
            .arg(format!("-DEXL3_BITS={bits}"))
            .arg(format!("-DEXL3_COOPERATIVE_K{bits}=1"))
            .arg("shaders/qmv_stream.comp")
            .arg("-o")
            .arg(out.join(format!("qmv_wave_{bits}_cooperative.spv")))
            .status()
            .expect("glslc");
        assert!(
            status.success(),
            "cooperative K{bits} shader compilation failed"
        );
        let status = Command::new(env::var_os("GLSLC").unwrap_or_else(|| "glslc".into()))
            .args([
                "--target-env=vulkan1.1",
                "-O",
                "-fshader-stage=compute",
                "-DEXL3_NATIVE=1",
                "-DEXL3_PARTIALS=1",
                "-DEXL3_FIXED_MCG=1",
            ])
            .arg(format!("-DEXL3_BITS={bits}"))
            .arg(format!("-DEXL3_COOPERATIVE_K{bits}=1"))
            .arg("shaders/qmv_stream.comp")
            .arg("-o")
            .arg(out.join(format!("qmv_wave_{bits}_mcg.spv")))
            .status()
            .expect("glslc");
        assert!(
            status.success(),
            "specialized MCG K{bits} shader compilation failed"
        );
    }
}
