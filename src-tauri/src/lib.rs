// VoxTerm Tauri shell.
//
// Desktop: spawn the existing Python web engine (gui.server) as a child bound to a fresh
// loopback port with a per-launch token, then navigate the webview to it so the real GUI
// (gui/static) loads same-origin with every server-side gate armed. The engine is the
// existing app — Tauri is just the native window + lifecycle owner.
//
// Mobile: no engine (a phone can't run torch/onnxruntime); the bundled pairing page loads
// and the user connects to a desktop on the LAN.

#[cfg(desktop)]
use std::sync::Mutex;

// Dev spawns `python -m gui.server` directly (a known command — plain std::process, no ACL
// needed). Release spawns the frozen, per-triple sidecar through the scoped shell plugin.
#[cfg(all(desktop, debug_assertions))]
type EngineHandle = std::process::Child;
#[cfg(all(desktop, not(debug_assertions)))]
type EngineHandle = tauri_plugin_shell::process::CommandChild;

#[cfg(desktop)]
struct Engine(Mutex<Option<EngineHandle>>);

#[cfg(desktop)]
fn random_token() -> String {
    use rand::Rng;
    const CS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut rng = rand::thread_rng();
    (0..32).map(|_| CS[rng.gen_range(0..CS.len())] as char).collect()
}

/// Navigate the main window to the engine once its loopback port accepts connections.
/// A successful TCP connect means the server is listening; app.js retries its own API
/// calls, so this is enough to avoid the cold-start blank screen.
#[cfg(desktop)]
fn navigate_when_ready(app: &tauri::AppHandle, port: u16, token: String) {
    use tauri::Manager;
    let handle = app.clone();
    std::thread::spawn(move || {
        let addr = format!("127.0.0.1:{port}");
        for _ in 0..600 {
            if std::net::TcpStream::connect(&addr).is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        if let Some(win) = handle.get_webview_window("main") {
            let url = format!("http://127.0.0.1:{port}/?token={token}");
            if let Ok(u) = url.parse() {
                let _ = win.navigate(u);
            }
        }
    });
}

#[cfg(all(desktop, debug_assertions))]
fn start_engine(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    use tauri::Manager;
    let port = std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
    let token = random_token();
    // Dev: run the module from the repo (parent of src-tauri) using the venv interpreter
    // (VOXTERM_PYTHON) or python3 on PATH.
    let py = std::env::var("VOXTERM_PYTHON").unwrap_or_else(|_| "python3".into());
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("no repo root")?
        .to_path_buf();
    let child = std::process::Command::new(py)
        .args(["-m", "gui.server"])
        .current_dir(repo_root)
        .env("VOXTERM_GUI_PORT", port.to_string())
        .env("VOXTERM_GUI_TOKEN", &token)
        .spawn()?;
    app.manage(Engine(Mutex::new(Some(child))));
    navigate_when_ready(app.handle(), port, token);
    Ok(())
}

#[cfg(all(desktop, not(debug_assertions)))]
fn start_engine(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    use tauri::Manager;
    use tauri_plugin_shell::ShellExt;
    let port = std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
    let token = random_token();
    // Release: the frozen, per-target-triple engine, resolved + scoped by the shell plugin.
    // NOTE: a release build also needs `bundle.externalBin: ["binaries/voxterm-engine"]` in
    // tauri.conf.json plus a PyInstaller step producing binaries/voxterm-engine-<triple>. That
    // freeze/packaging is the deferred milestone (see ~/voxterm-plans/CEILING.md, R4); until it
    // lands, run the DEV build (`cargo tauri dev`), which spawns python directly (no externalBin).
    let (_rx, child) = app
        .shell()
        .sidecar("voxterm-engine")?
        .env("VOXTERM_GUI_PORT", port.to_string())
        .env("VOXTERM_GUI_TOKEN", token.clone())
        .spawn()?;
    app.manage(Engine(Mutex::new(Some(child))));
    navigate_when_ready(app.handle(), port, token);
    Ok(())
}

// std::process::Child::kill takes &mut self (dev); CommandChild::kill consumes self
// (release). Split so each branch is warning-clean.
#[cfg(all(desktop, debug_assertions))]
fn kill_handle(mut child: EngineHandle) {
    let _ = child.kill();
}
#[cfg(all(desktop, not(debug_assertions)))]
fn kill_handle(child: EngineHandle) {
    let _ = child.kill();
}

#[cfg(desktop)]
fn kill_engine(app: &tauri::AppHandle) {
    use tauri::Manager;
    if let Some(engine) = app.try_state::<Engine>() {
        if let Ok(mut guard) = engine.0.lock() {
            if let Some(child) = guard.take() {
                kill_handle(child);
            }
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default();

    // The shell plugin is desktop-only (mobile has no sidecar).
    #[cfg(desktop)]
    let builder = builder.plugin(tauri_plugin_shell::init());

    // On-device ASR is mobile-only — the phone transcribes locally via sherpa-onnx (no relay).
    #[cfg(mobile)]
    let builder = builder.plugin(tauri_plugin_voxasr::init());

    // On-device LLM (conversation graph + interruptions) is mobile-only and fully offline. If no
    // model is bundled it reports unavailable and the GUI keeps its heuristic analyzer.
    #[cfg(mobile)]
    let builder = builder.plugin(tauri_plugin_voxllm::init());

    let builder = builder.setup(|app| {
        if cfg!(debug_assertions) {
            app.handle().plugin(
                tauri_plugin_log::Builder::default()
                    .level(log::LevelFilter::Info)
                    .build(),
            )?;
        }
        #[cfg(desktop)]
        if let Err(e) = start_engine(app) {
            log::error!("failed to start VoxTerm engine: {e}");
        }
        Ok(())
    });

    builder
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_app_handle, _event| {
            // Tear the engine down with the app so no orphaned process keeps the mic open.
            #[cfg(desktop)]
            if let tauri::RunEvent::ExitRequested { .. } = _event {
                kill_engine(_app_handle);
            }
        });
}
