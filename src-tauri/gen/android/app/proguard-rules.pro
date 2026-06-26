# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile

# --- sherpa-onnx (on-device ASR) ---
# The sherpa-onnx native library reads config fields (e.g. decodingMethod, modelType, the nested
# whisper/feat configs) from the Kotlin config objects by NAME via JNI GetFieldID. R8 has no way to
# see those native accesses, so under minification it renames/removes the fields and the native
# OfflineRecognizer constructor fails at runtime with
#   "failed to get field id for decodingMethod".
# Keep every sherpa class and all of its members so the JNI field/method lookups resolve.
-keep class com.k2fsa.sherpa.onnx.** { *; }
-keepclassmembers class com.k2fsa.sherpa.onnx.** { *; }

# --- MediaPipe LLM Inference (on-device conversation analysis) ---
# The tasks-genai AAR ships NO consumer ProGuard rules, and MediaPipe uses JNI + protobuf + reflection
# internally. Keep the whole tasks namespace so the native LLM engine and its config protos resolve
# under release minification, and silence warnings about its optional compile-time-only deps.
-keep class com.google.mediapipe.** { *; }
-keepclassmembers class com.google.mediapipe.** { *; }
-keep class com.google.protobuf.** { *; }
-dontwarn com.google.mediapipe.**
-dontwarn com.google.protobuf.**
-dontwarn com.google.common.**
-dontwarn javax.annotation.**
-dontwarn javax.lang.model.**
-dontwarn autovalue.shaded.**