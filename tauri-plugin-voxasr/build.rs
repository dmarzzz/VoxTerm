const COMMANDS: &[&str] = &["start_transcribe", "stop_transcribe", "poll_transcript", "delete_audio"];

fn main() {
    tauri_plugin::Builder::new(COMMANDS)
        .android_path("android")
        .build();
}
