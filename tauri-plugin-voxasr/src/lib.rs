//! VoxTerm on-device ASR plugin (Android). The phone records the mic, then transcribes the whole
//! clip at stop with an offline sherpa-onnx Whisper recognizer — full context, punctuation, fully
//! local (no pairing, no relay, no network). The webview drives it through three commands:
//! `start_transcribe` / `stop_transcribe`, and `poll_transcript`, which it polls on an interval to
//! read `{ phase, elapsed, level, durationSec, segments[], error? }` (polling — no plugin events).
//! `phase` is idle | recording | transcribing | done | error. Desktop/iOS get an honest
//! "unsupported" error (desktop uses the full Python engine; iOS is a future native port).

use tauri::{
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime,
};

/// Holds the Android plugin handle (the bridge to the Kotlin `VoxasrPlugin`). On non-Android
/// targets it just carries the app handle so the commands can return a clean error.
struct Voxasr<R: Runtime> {
    #[cfg(target_os = "android")]
    handle: tauri::plugin::PluginHandle<R>,
    #[cfg(not(target_os = "android"))]
    _app: AppHandle<R>,
}

#[tauri::command]
async fn start_transcribe<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    #[cfg(target_os = "android")]
    {
        // The native plugin uses the bundled, staged model — no args needed.
        app.state::<Voxasr<R>>()
            .handle
            .run_mobile_plugin::<serde_json::Value>("startTranscribe", ())
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = app;
        Err("on-device transcription is Android-only; use the desktop engine elsewhere".into())
    }
}

#[tauri::command]
async fn stop_transcribe<R: Runtime>(
    app: AppHandle<R>,
    stem: Option<String>,
    diarize: Option<bool>,
) -> Result<(), String> {
    #[cfg(target_os = "android")]
    {
        // Forward the session id (to persist <stem>.wav) + the opt-in diarize flag to the native side.
        let payload = serde_json::json!({ "stem": stem, "diarize": diarize.unwrap_or(false) });
        app.state::<Voxasr<R>>()
            .handle
            .run_mobile_plugin::<serde_json::Value>("stopTranscribe", payload)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (app, stem, diarize);
        Err("on-device transcription is Android-only".into())
    }
}

/// Delete a session's persisted audio (`<stem>.wav`) when the session is deleted in the GUI.
#[tauri::command]
async fn delete_audio<R: Runtime>(app: AppHandle<R>, stem: String) -> Result<serde_json::Value, String> {
    #[cfg(target_os = "android")]
    {
        app.state::<Voxasr<R>>()
            .handle
            .run_mobile_plugin::<serde_json::Value>("deleteAudio", serde_json::json!({ "stem": stem }))
            .map_err(|e| e.to_string())
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (app, stem);
        Ok(serde_json::json!({ "deleted": false }))
    }
}

/// Poll the recording/transcription state: `{ phase, elapsed, level, durationSec, segments[],
/// error? }` where `phase` is idle|recording|transcribing|done|error. The webview polls this on an
/// interval (simple + robust — avoids the plugin-event listener path).
#[tauri::command]
async fn poll_transcript<R: Runtime>(app: AppHandle<R>) -> Result<serde_json::Value, String> {
    #[cfg(target_os = "android")]
    {
        app.state::<Voxasr<R>>()
            .handle
            .run_mobile_plugin::<serde_json::Value>("pollTranscript", ())
            .map_err(|e| e.to_string())
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = app;
        Ok(serde_json::json!({ "phase": "idle", "elapsed": 0, "level": 0, "durationSec": 0, "segments": [] }))
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("voxasr")
        .invoke_handler(tauri::generate_handler![start_transcribe, stop_transcribe, poll_transcript, delete_audio])
        .setup(|app, _api| {
            #[cfg(target_os = "android")]
            {
                let handle = _api.register_android_plugin("site.nubs.voxterm.voxasr", "VoxasrPlugin")?;
                app.manage(Voxasr { handle });
            }
            #[cfg(not(target_os = "android"))]
            {
                app.manage(Voxasr { _app: app.clone() });
            }
            Ok(())
        })
        .build()
}
