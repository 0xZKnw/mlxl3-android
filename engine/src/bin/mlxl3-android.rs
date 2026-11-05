use anyhow::{Context, Result};
use mlxl3_android::{Engine, Message};
use std::{path::Path, sync::atomic::AtomicBool};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let path = args
        .get(1)
        .context("Usage: mlxl3-android MODEL_DIRECTORY [PROMPT]")?;
    let cancel = AtomicBool::new(false);
    let mut engine = Engine::load(Path::new(path), &cancel, &mut |s| eprintln!("{s}"))?;
    let prompt = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("Say hello in one sentence.");
    engine.generate(
        &[Message {
            role: "user".into(),
            content: prompt.into(),
        }],
        128,
        &cancel,
        &mut |event| {
            println!("{event}");
            Ok(())
        },
    )
}
