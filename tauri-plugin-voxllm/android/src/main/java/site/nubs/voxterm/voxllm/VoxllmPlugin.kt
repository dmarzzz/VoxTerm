package site.nubs.voxterm.voxllm

import android.app.Activity
import android.util.Log
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import java.io.File
import kotlin.concurrent.thread

// The bundled model lives in assets/voxllm-model/model.task and is staged to filesDir on first use,
// so generation is fully offline (no first-run download). Qwen2.5-0.5B-Instruct q8, MediaPipe .task,
// effective KV-cache 1280 tokens — keep prompts short (the JS side caps the transcript it sends).
private const val MODEL_ASSET_DIR = "voxllm-model"
private const val MODEL_FILE = "model.task"
private const val MAX_TOKENS = 1024            // total sequence (prompt + output); must be <= model's ekv 1280
private const val MAX_TOP_K = 40

@InvokeArg
class GenerateArgs {
    var prompt: String = ""
    var maxTokens: Int = MAX_TOKENS
    var temperature: Double = 0.2
    var topK: Int = MAX_TOP_K
}

/**
 * On-device LLM text generation via MediaPipe LLM Inference. A generic, domain-agnostic runner:
 * `startGenerate` kicks off generation for a prompt on a worker thread; the webview polls
 * `pollGenerate` for `{ phase, text?, error? }`. The conversation Graph / Interruptions modes build
 * their prompts + parse the JSON in JS — this stays a plain "prompt in, text out" engine so other
 * on-device LLM features can reuse it. Fully offline; the app strips INTERNET in its manifest.
 */
@TauriPlugin
class VoxllmPlugin(private val activity: Activity) : Plugin(activity) {
    @Volatile private var phase = "idle"        // idle | running | done | error
    @Volatile private var resultText = ""
    @Volatile private var errorMsg: String? = null
    @Volatile private var startedAt = 0L
    @Volatile private var elapsedMs = 0L
    // Bumped per request so a generation that outlives a cancel/restart can't clobber newer state.
    @Volatile private var generation = 0
    // Built once (heavy: loads ~550 MB, ~1.5 GB RAM) and reused across requests.
    @Volatile private var llm: LlmInference? = null

    // Stage the single model file from assets to filesDir. Temp-dir-then-rename so `out` is only ever
    // a complete dir (self-heals a half-copy), mirroring the ASR plugin's staging.
    @Synchronized
    private fun stagedModel(): File {
        val out = File(activity.filesDir, MODEL_ASSET_DIR)
        val model = File(out, MODEL_FILE)
        if (model.exists() && model.length() > 0) return model
        val tmp = File(activity.filesDir, "$MODEL_ASSET_DIR.tmp")
        tmp.deleteRecursively(); tmp.mkdirs()
        val am = activity.assets
        for (name in am.list(MODEL_ASSET_DIR) ?: arrayOf()) {
            am.open("$MODEL_ASSET_DIR/$name").use { input ->
                File(tmp, name).outputStream().use { input.copyTo(it) }
            }
        }
        if (!File(tmp, MODEL_FILE).exists()) {
            tmp.deleteRecursively()
            throw java.io.IOException("bundled LLM model is missing ($MODEL_FILE)")
        }
        out.deleteRecursively()
        if (!tmp.renameTo(out)) { tmp.deleteRecursively(); throw java.io.IOException("could not stage LLM model dir") }
        return File(out, MODEL_FILE)
    }

    private fun modelAssetPresent(): Boolean =
        try { (activity.assets.list(MODEL_ASSET_DIR) ?: arrayOf()).any { it == MODEL_FILE } }
        catch (e: Exception) { false }

    // Build the MediaPipe inference once, lazily. CPU backend for broad device compatibility (no GPU/
    // OpenCL delegate required). @Synchronized so two requests can't both build (and leak) an engine.
    @Synchronized
    private fun ensureLlm(): LlmInference {
        llm?.let { return it }
        val path = stagedModel().absolutePath
        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(path)
            .setMaxTokens(MAX_TOKENS)
            .setMaxTopK(MAX_TOP_K)
            .setPreferredBackend(LlmInference.Backend.CPU)
            .build()
        return LlmInference.createFromOptions(activity, options).also { llm = it }
    }

    // Is a model bundled? Cheap (asset listing only — no heavy init). The GUI uses this to decide
    // whether to swap its heuristic analyzer for the LLM one; if generation later fails, the GUI
    // falls back to the heuristic anyway.
    @Command
    fun llmAvailable(invoke: Invoke) {
        val res = JSObject()
        res.put("available", modelAssetPresent())
        res.put("model", "Qwen2.5-0.5B (MediaPipe)")
        invoke.resolve(res)
    }

    @Command
    fun startGenerate(invoke: Invoke) {
        val args = invoke.parseArgs(GenerateArgs::class.java)
        val gen = ++generation
        phase = "running"; resultText = ""; errorMsg = null
        startedAt = System.currentTimeMillis(); elapsedMs = 0
        invoke.resolve(JSObject())                 // resolve now; generation runs async, read via poll
        thread(start = true) {
            try {
                val engine = ensureLlm()
                val out = engine.generateResponse(args.prompt)
                if (generation == gen) {
                    resultText = out ?: ""
                    elapsedMs = System.currentTimeMillis() - startedAt
                    phase = "done"
                }
            } catch (e: Throwable) {
                Log.e("voxllm", "generation failed", e)
                if (generation == gen) { errorMsg = e.message ?: "generation failed"; phase = "error" }
            }
        }
    }

    @Command
    fun pollGenerate(invoke: Invoke) {
        val res = JSObject()
        res.put("phase", phase)
        if (phase == "running") res.put("elapsedMs", System.currentTimeMillis() - startedAt)
        if (phase == "done") { res.put("text", resultText); res.put("elapsedMs", elapsedMs) }
        errorMsg?.let { res.put("error", it) }
        invoke.resolve(res)
    }

    @Command
    fun cancelGenerate(invoke: Invoke) {
        generation++                               // abandon any in-flight result
        if (phase == "running") phase = "idle"
        invoke.resolve(JSObject())
    }
}
