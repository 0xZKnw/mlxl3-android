use crate::{Engine, Message};
use anyhow::{Context, Result, ensure};
use jni::{
    JNIEnv,
    objects::{JClass, JObject, JString, JValue},
    sys::jstring,
};
use std::{
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);
static CANCEL: AtomicBool = AtomicBool::new(false);

fn emit(env: &mut JNIEnv, callback: &JObject, event: serde_json::Value) -> Result<()> {
    let text = env.new_string(event.to_string())?;
    env.call_method(
        callback,
        "onEvent",
        "(Ljava/lang/String;)V",
        &[JValue::Object(&text)],
    )?;
    env.delete_local_ref(text)?;
    Ok(())
}

fn boundary(env: &mut JNIEnv, action: impl FnOnce(&mut JNIEnv) -> Result<()>) -> jstring {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| action(env)));
    let message = match outcome {
        Ok(Ok(())) => String::new(),
        Ok(Err(err)) => format!("{err:#}"),
        Err(_) => "Native engine failed; restart the app".to_owned(),
    };
    if env.exception_check().unwrap_or(true) {
        return std::ptr::null_mut();
    }
    env.new_string(message)
        .map(JString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_tech_mlxl3_android_NativeEngine_load(
    mut env: JNIEnv,
    _class: JClass,
    path: JString,
    callback: JObject,
) -> jstring {
    boundary(&mut env, |env| {
        let path: String = env.get_string(&path)?.into();
        let mut engine = ENGINE
            .try_lock()
            .map_err(|_| anyhow::anyhow!("Engine is busy or must be restarted"))?;
        *engine = None;
        let mut callback_error = None;
        let loaded = Engine::load(Path::new(&path), &CANCEL, &mut |status| {
            if let Err(err) = emit(
                env,
                &callback,
                serde_json::json!({"type":"status","text":status}),
            ) {
                callback_error = Some(err);
                CANCEL.store(true, Ordering::Relaxed);
            }
        });
        if let Some(err) = callback_error {
            return Err(err);
        }
        let loaded = loaded?;
        emit(
            env,
            &callback,
            serde_json::json!({"type":"ready","device":loaded.model.gpu.name}),
        )?;
        *engine = Some(loaded);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_tech_mlxl3_android_NativeEngine_generate(
    mut env: JNIEnv,
    _class: JClass,
    messages: JString,
    max_tokens: i32,
    callback: JObject,
) -> jstring {
    boundary(&mut env, |env| {
        let messages: String = env.get_string(&messages)?.into();
        ensure!(
            messages.len() <= 2 * 1024 * 1024,
            "Chat request exceeds 2 MiB"
        );
        let messages: Vec<Message> = serde_json::from_str(&messages)?;
        ensure!(max_tokens > 0, "Invalid generation budget");
        let mut engine = ENGINE
            .try_lock()
            .map_err(|_| anyhow::anyhow!("Engine is busy or must be restarted"))?;
        engine.as_mut().context("Load LFM3 first")?.generate(
            &messages,
            max_tokens as usize,
            &CANCEL,
            &mut |event| emit(env, &callback, event),
        )
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_tech_mlxl3_android_NativeEngine_resetCancel(
    _env: JNIEnv,
    _class: JClass,
) {
    CANCEL.store(false, Ordering::Relaxed);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_tech_mlxl3_android_NativeEngine_cancel(_env: JNIEnv, _class: JClass) {
    CANCEL.store(true, Ordering::Relaxed);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_tech_mlxl3_android_NativeEngine_unload(_env: JNIEnv, _class: JClass) {
    CANCEL.store(true, Ordering::Relaxed);
    if let Ok(mut engine) = ENGINE.try_lock() {
        *engine = None;
    }
}
