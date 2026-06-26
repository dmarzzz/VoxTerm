const COMMANDS: &[&str] = &["llm_available", "start_generate", "poll_generate", "cancel_generate"];

fn main() {
    tauri_plugin::Builder::new(COMMANDS)
        .android_path("android")
        .build();
}
