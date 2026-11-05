use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=GLSLC");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    for name in ["qmv", "dense"] {
        let status = Command::new(env::var_os("GLSLC").unwrap_or_else(|| "glslc".into()))
            .args(["--target-env=vulkan1.1", "-O", "-fshader-stage=compute"])
            .arg(format!("shaders/{name}.comp"))
            .arg("-o")
            .arg(out.join(format!("{name}.spv")))
            .status()
            .expect("install glslc (shaderc) or set GLSLC");
        assert!(status.success(), "{name} shader compilation failed");
    }
}
