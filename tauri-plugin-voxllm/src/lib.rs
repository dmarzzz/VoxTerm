//! VoxTerm on-device LLM plugin (Android). A generic, reusable on-device text generator: given a
//! prompt, it runs a small bundled LLM locally and returns the generated text — fully offline (no
//! network). The webview drives it through four commands:
//!   `llm_available`  -> `{ available: bool, model: string }`  (is a model loaded?)
//!   `start_generate` -> kicks off generation for a prompt (async; poll for the result)
//!   `poll_generate`  -> `{ phase: idle|running|done|error, text?, error?, elapsedMs? }`
//!   `cancel_generate`-> stop an in-flight generation
//! The conversation Graph / Interruptions modes use this to produce their JSON (prompting + parsing
//! live in JS); it's deliberately domain-agnostic so other on-device LLM features can reuse it.
//! Desktop/iOS get an honest "unavailable" so the GUI falls back to its offline heuristic analyzer.

use tauri::{
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime,
};

struct Voxllm<R: Runtime> {
    #[cfg(target_os = "android")]
    handle: tauri::plugin::PluginHandle<R>,
    #[cfg(not(target_os = "android"))]
    _app: AppHandle<R>,
}

#[tauri::command]
async fn llm_available<R: Runtime>(app: AppHandle<R>) -> Result<serde_json::Value, String> {
    #[cfg(target_os = "android")]
    {
        app.state::<Voxllm<R>>()
            .handle
            .run_mobile_plugin::<serde_json::Value>("llmAvailable", ())
            .map_err(|e| e.to_string())
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = app;
        Ok(serde_json::json!({ "available": false, "model": "" }))
    }
}

#[tauri::command]
async fn start_generate<R: Runtime>(
    app: AppHandle<R>,
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    top_k: Option<u32>,
) -> Result<(), String> {
    #[cfg(target_os = "android")]
    {
        let payload = serde_json::json!({
            "prompt": prompt,
            "maxTokens": max_tokens.unwrap_or(1024),
            "temperature": temperature.unwrap_or(0.2),
            "topK": top_k.unwrap_or(40),
        });
        app.state::<Voxllm<R>>()
            .handle
            .run_mobile_plugin::<serde_json::Value>("startGenerate", payload)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (app, prompt, max_tokens, temperature, top_k);
        Err("on-device LLM is Android-only".into())
    }
}

#[tauri::command]
async fn poll_generate<R: Runtime>(app: AppHandle<R>) -> Result<serde_json::Value, String> {
    #[cfg(target_os = "android")]
    {
        app.state::<Voxllm<R>>()
            .handle
            .run_mobile_plugin::<serde_json::Value>("pollGenerate", ())
            .map_err(|e| e.to_string())
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = app;
        Ok(serde_json::json!({ "phase": "idle" }))
    }
}

#[tauri::command]
async fn cancel_generate<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    #[cfg(target_os = "android")]
    {
        app.state::<Voxllm<R>>()
            .handle
            .run_mobile_plugin::<serde_json::Value>("cancelGenerate", ())
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = app;
        Ok(())
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("voxllm")
        .invoke_handler(tauri::generate_handler![
            llm_available,
            start_generate,
            poll_generate,
            cancel_generate
        ])
        .setup(|app, _api| {
            #[cfg(target_os = "android")]
            {
                let handle = _api.register_android_plugin("site.nubs.voxterm.voxllm", "VoxllmPlugin")?;
                app.manage(Voxllm { handle });
            }
            #[cfg(not(target_os = "android"))]
            {
                app.manage(Voxllm { _app: app.clone() });
            }
            Ok(())
        })
        .build()
}
