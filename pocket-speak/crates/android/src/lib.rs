//! The desk's engine inside the Android app.
//!
//! `org.pockettts.android.engine.NativeEngine` is the only caller. An engine,
//! a voice and a cancel token are each a box handed to Kotlin as a `long`.
//!
//! The engine sits behind a mutex, and that mutex is load-bearing: Kotlin
//! serialises synthesis, but it loads voices outside that lock, so the mutex is
//! what keeps `load_voice` from running while `synthesize` holds the engine
//! mutably. A cancel token is separate from the engine, so cancelling never
//! waits for the frame in progress, and each request brings its own: a token
//! cancelled before its synthesis starts stops that synthesis at the first
//! frame, and cannot stop someone else's.
//!
//! ONNX Runtime is Microsoft's `onnxruntime-android` library, packaged by the
//! app and opened here by name (ort's `load-dynamic`), once per process.
//!
//! A panic never crosses into the JVM: each entry point that can fail catches
//! it and throws `IllegalStateException` instead, so a Rust panic in the
//! engine costs a failed read rather than the process. A fault inside ONNX
//! Runtime's native code is not a panic and is not caught. A panic inside
//! `synthesize` poisons the engine's mutex; [`lock`] takes the engine anyway,
//! because `Engine::synthesize` keeps no per-chunk state on the engine (its
//! doc says so), and refusing every later read would silence TalkBack until
//! the process died.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use jni::JNIEnv;
use jni::objects::{JClass, JFloatArray, JObject, JString, JValue};
use jni::sys::{jfloat, jint, jlong, jstring};
use pocket_tts_engine::{Cancel, Engine, Voice};

static RUNTIME: OnceLock<Result<(), String>> = OnceLock::new();

fn init_runtime(path: &str) -> Result<(), String> {
    RUNTIME
        .get_or_init(|| {
            ort::init_from(path)
                .map_err(|e| format!("ONNX Runtime at {path}: {e}"))?
                .commit();
            Ok(())
        })
        .clone()
}

/// Runs `body`, turning an error or a panic into a Java exception and
/// `fallback`.
fn guarded<T>(
    env: &mut JNIEnv,
    fallback: T,
    body: impl FnOnce(&mut JNIEnv) -> Result<T, String>,
) -> T {
    let outcome = catch_unwind(AssertUnwindSafe(|| body(env)));
    let message = match outcome {
        Ok(Ok(value)) => return value,
        Ok(Err(message)) => message,
        Err(panic) => {
            let what = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown".into());
            format!("panic in the engine: {what}")
        }
    };
    // A Java exception already pending (thrown by the audio sink) is the
    // better account; leave it rather than replace it.
    if !env.exception_check().unwrap_or(false) {
        let _ = env.throw_new("java/lang/IllegalStateException", message);
    }
    fallback
}

fn string(env: &mut JNIEnv, value: &JString) -> Result<String, String> {
    env.get_string(value)
        .map(Into::into)
        .map_err(|e| format!("string argument: {e}"))
}

/// The engine behind `raw`, locked, poisoned or not (see the module doc).
/// Every entry point locks it through here, so none refuses after a panic.
fn lock<'a>(raw: jlong) -> MutexGuard<'a, Engine> {
    // SAFETY: `raw` came from nativeLoad. The app loads one engine per process
    // and never frees it (NativeEngine.close has no caller), so it outlives
    // every call made on it.
    let engine = unsafe { &*(raw as *const Mutex<Engine>) };
    engine.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Loads the model in `model_dir`, opening ONNX Runtime from `ort_library`
/// first if this process has not.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pockettts_android_engine_NativeEngine_nativeLoad(
    mut env: JNIEnv,
    _class: JClass,
    model_dir: JString,
    threads: jint,
    ort_library: JString,
) -> jlong {
    guarded(&mut env, 0, |env| {
        let ort_library = string(env, &ort_library)?;
        init_runtime(&ort_library)?;
        let dir = string(env, &model_dir)?;
        let engine =
            Engine::load(Path::new(&dir), threads.max(1) as usize).map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(Mutex::new(engine))) as jlong)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pockettts_android_engine_NativeEngine_nativeFree(
    _env: JNIEnv,
    _class: JClass,
    raw: jlong,
) {
    if raw != 0 {
        // SAFETY: from nativeLoad; this is the one release.
        drop(unsafe { Box::from_raw(raw as *mut Mutex<Engine>) });
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pockettts_android_engine_NativeEngine_nativeSampleRate(
    mut env: JNIEnv,
    _class: JClass,
    raw: jlong,
) -> jint {
    guarded(&mut env, 24_000, |_| Ok(lock(raw).sample_rate() as jint))
}

/// Loads a voice embedding (`.safetensors`).
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pockettts_android_engine_NativeEngine_nativeLoadVoice(
    mut env: JNIEnv,
    _class: JClass,
    raw: jlong,
    path: JString,
) -> jlong {
    guarded(&mut env, 0, |env| {
        let path = string(env, &path)?;
        let engine = lock(raw);
        let voice = engine
            .load_voice(Path::new(&path))
            .map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(voice)) as jlong)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pockettts_android_engine_NativeEngine_nativeFreeVoice(
    _env: JNIEnv,
    _class: JClass,
    voice: jlong,
) {
    if voice != 0 {
        // SAFETY: from nativeLoadVoice, released once by Kotlin.
        drop(unsafe { Box::from_raw(voice as *mut Voice) });
    }
}

/// A cancel token for one request. Free it with nativeFreeCancel.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pockettts_android_engine_NativeEngine_nativeNewCancel(
    _env: JNIEnv,
    _class: JClass,
) -> jlong {
    Box::into_raw(Box::new(Cancel::new())) as jlong
}

/// Cancels the token: its synthesis stops at the next frame, or at the first
/// if it has not started.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pockettts_android_engine_NativeEngine_nativeCancel(
    _env: JNIEnv,
    _class: JClass,
    token: jlong,
) {
    if token != 0 {
        // SAFETY: from nativeNewCancel; Kotlin's CancelToken stops calling
        // this once it has freed the token.
        unsafe { &*(token as *const Cancel) }.cancel();
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pockettts_android_engine_NativeEngine_nativeFreeCancel(
    _env: JNIEnv,
    _class: JClass,
    token: jlong,
) {
    if token != 0 {
        // SAFETY: from nativeNewCancel, released once.
        drop(unsafe { Box::from_raw(token as *mut Cancel) });
    }
}

/// Synthesises one chunk, calling `sink.onAudio(float[])` as audio is decoded.
/// `seed` below 0 keeps the running generator; otherwise sampling restarts
/// from it. `token` is a cancel token from nativeNewCancel, or 0 for none.
/// Returns the chunk's stats as JSON, or null after throwing.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_pockettts_android_engine_NativeEngine_nativeSynthesize(
    mut env: JNIEnv,
    _class: JClass,
    raw: jlong,
    voice: jlong,
    text: JString,
    temperature: jfloat,
    seed: jlong,
    token: jlong,
    sink: JObject,
) -> jstring {
    guarded(&mut env, std::ptr::null_mut(), |env| {
        let text = string(env, &text)?;
        // SAFETY: from nativeLoadVoice; the app never frees a loaded voice.
        let voice = unsafe { &*(voice as *const Voice) };
        let cancel = if token == 0 {
            Cancel::new()
        } else {
            // SAFETY: from nativeNewCancel, and CancelToken frees it only
            // after the synthesis it was handed to has returned.
            unsafe { &*(token as *const Cancel) }.clone()
        };
        let mut engine = lock(raw);
        engine.set_temperature(temperature);
        if seed >= 0 {
            engine.set_seed(seed as u64);
        }
        let mut failure: Option<String> = None;
        let result = engine.synthesize(&text, voice, &cancel, |samples| {
            let call = (|| {
                let array: JFloatArray = env.new_float_array(samples.len() as i32)?;
                env.set_float_array_region(&array, 0, samples)?;
                let keep = env
                    .call_method(&sink, "onAudio", "([F)Z", &[JValue::Object(&array)])?
                    .z()?;
                env.delete_local_ref(array)?;
                Ok::<bool, jni::errors::Error>(keep)
            })();
            call.unwrap_or_else(|error| {
                failure = Some(format!("audio sink: {error}"));
                false
            })
        });
        drop(engine);
        if let Some(message) = failure {
            return Err(message);
        }
        let stats = result.map_err(|e| e.to_string())?;
        let json = format!(
            "{{\"tokens\":{},\"frames\":{},\"max_frames\":{},\"eos_frame\":{},\"end\":\"{:?}\",\"samples\":{},\"generation_ms\":{},\"first_audio_ms\":{}}}",
            stats.tokens,
            stats.frames,
            stats.max_frames,
            stats.eos_frame.map_or("null".into(), |f| f.to_string()),
            stats.end,
            stats.samples,
            stats.generation_ms,
            stats
                .first_audio_ms
                .map_or("null".into(), |f| f.to_string()),
        );
        env.new_string(json)
            .map(|s| s.into_raw())
            .map_err(|e| format!("stats string: {e}"))
    })
}
