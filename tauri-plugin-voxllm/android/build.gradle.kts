plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "site.nubs.voxterm.voxllm"
    compileSdk = 36
    defaultConfig {
        minSdk = 24
    }
    buildTypes {
        getByName("release") {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.9.0")
    implementation(project(":tauri-android"))
    // Google MediaPipe LLM Inference (Google Maven). Resolves from the allprojects google() repo.
    // The AAR declares minSdk 23 (<= app minSdk 24) and ships a single, uniquely-named native lib
    // (libllm_inference_engine_jni.so) per ABI — no .so collision with sherpa-onnx / tauri-android.
    // The on-device model (.task) is bundled as an asset by ../fetch-deps.sh.
    implementation("com.google.mediapipe:tasks-genai:0.10.27")
}
