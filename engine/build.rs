use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=GLSLC");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    for name in [
        "qmv",
        "dense",
        "codebook",
        "qmv_reference",
        "codebook_reference",
    ] {
        let mut compiler = Command::new(env::var_os("GLSLC").unwrap_or_else(|| "glslc".into()));
        compiler
            .args(["--target-env=vulkan1.1", "-O", "-fshader-stage=compute"])
            .arg(format!(
                "shaders/{}.comp",
                name.trim_end_matches("_reference")
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
}
