package site.nubs.voxterm.voxasr

import android.Manifest
import android.app.Activity
import android.content.pm.ApplicationInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import android.webkit.WebView
import app.tauri.PermissionState
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.Permission
import app.tauri.annotation.PermissionCallback
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import com.k2fsa.sherpa.onnx.FastClusteringConfig
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerDiarization
import com.k2fsa.sherpa.onnx.OfflineSpeakerDiarizationConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerSegmentationModelConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerSegmentationPyannoteModelConfig
import com.k2fsa.sherpa.onnx.OfflineStream
import com.k2fsa.sherpa.onnx.OfflineWhisperModelConfig
import com.k2fsa.sherpa.onnx.SpeakerEmbeddingExtractorConfig
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import kotlin.concurrent.thread
import kotlin.math.sqrt

private const val SAMPLE_RATE = 16000
private const val WHISPER_WINDOW = 29 * SAMPLE_RATE   // just under Whisper's 30 s cap (it truncates >30 s)
private const val MAX_REPEAT_PHRASE = 12              // longest word-run de-looping scans for (see collapseRepeats)
private const val DIAR_THRESHOLD = 0.5f               // FastClustering cosine threshold; lower = more speakers

/**
 * On-device speech-to-text. Records the mic with AudioRecord (16 kHz mono PCM16) into a buffer and,
 * at stop, transcribes the whole clip with an OFFLINE sherpa-onnx Whisper recognizer — full
 * context, native casing + punctuation, no live/streaming output. Everything is local — no network.
 *
 * The webview polls `pollTranscript`, which reports a phase (idle/recording/transcribing/done/error)
 * plus, when done, the transcript segments. Whisper decodes <=30 s per pass, so a long clip is split
 * into 30 s windows and the texts joined.
 */
@InvokeArg
class StopArgs {
    var stem: String? = null      // session id → persist the take as <stem>.wav (null = don't keep audio)
    var diarize: Boolean = false  // run on-device speaker diarization at stop (opt-in: real time + memory)
}

@InvokeArg
class StemArgs { var stem: String = "" }

@TauriPlugin(
    permissions = [Permission(strings = [Manifest.permission.RECORD_AUDIO], alias = "microphone")],
)
class VoxasrPlugin(private val activity: Activity) : Plugin(activity) {
    @Volatile private var running = false
    private var worker: Thread? = null
    // Live-preview decoder: runs alongside the mic worker while recording, repeatedly decoding the
    // growing buffer so a rough transcript streams in near-real-time. The authoritative pass still
    // runs once at stop. Joined before that final pass so the two never decode concurrently.
    private var liveWorker: Thread? = null
    // Bumped per capture session so a worker that outlives stop's join can't run the mic alongside,
    // or clobber the phase of, a newer session.
    @Volatile private var generation = 0
    // Built once via ensureRecognizer(); shared by the mic worker and the debug self-test.
    @Volatile private var recognizer: OfflineRecognizer? = null
    private val modelFiles = listOf("encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt")

    // Raw PCM16 of the current take is spilled to a per-generation file on disk (filesDir/take-<gen>.pcm),
    // NOT held in RAM — a long take (45 min ≈ 86 MB) would otherwise OOM-crash the app, and the live
    // loop's old full-buffer copy churned the heap every pass. The mic worker appends; readers (live
    // loop, final decode) mmap-read bounded windows via RandomAccessFile. `samplesWritten` is the count
    // of PCM16 samples flushed so far — readers only read up to it so they never see unwritten bytes.
    @Volatile private var samplesWritten = 0
    private fun takeFile(gen: Int) = File(activity.filesDir, "take-$gen.pcm")
    // Polled state the webview reads. `phase` drives the GUI's record/transcribe/done state machine.
    @Volatile private var phase = "idle"          // idle | recording | transcribing | done | error
    @Volatile private var elapsedSec = 0.0
    @Volatile private var levelRms = 0.0
    @Volatile private var durationSec = 0.0
    @Volatile private var decodeProgress = 0.0    // 0..1 fraction of the take decoded (drives the GUI progress bar)
    @Volatile private var errorMsg: String? = null
    // Finalized transcript segments (one per <=30 s window): text + its start offset in seconds.
    private val segments = java.util.Collections.synchronizedList(mutableListOf<Pair<String, Double>>())

    // ---- speaker diarization (OPT-IN; runs at stop, costs real time + memory) ----
    // When enabled for a take, after Whisper decodes we run sherpa's OfflineSpeakerDiarization over the
    // whole clip to get [{start,end,speaker}] segments, then fill each with the Whisper TEXT whose token
    // timestamps fall inside it — producing real per-speaker utterances instead of one anonymous window.
    @Volatile private var diarizer: OfflineSpeakerDiarization? = null
    @Volatile private var diarizeEnabled = false       // set per-take from stop_transcribe's `diarize` arg
    @Volatile private var diarStatus = ""              // "" | "diarizing" — surfaced in the progress msg
    @Volatile private var speakerCount = 0
    private val diarFiles = listOf("segmentation.onnx", "embedding.onnx")
    // Speaker-attributed utterances built at stop (empty unless diarization ran). text/start/end seconds,
    // speaker = 0-based cluster id, gapBefore = silence before this utterance (null for the first).
    private data class Utt(val text: String, val start: Double, val end: Double, val speaker: Int, val gapBefore: Double?)
    private val utterances = java.util.Collections.synchronizedList(mutableListOf<Utt>())
    // One decoded window's result: collapsed text (authoritative transcript) + raw tokens & their
    // timestamps (seconds, relative to the window start) used only to place text into diar segments.
    private data class Decoded(val text: String, val tokens: Array<String>, val stamps: FloatArray)

    // Persisted per-session audio (so a session can be replayed / re-diarized): filesDir/voxterm-audio/<stem>.wav.
    // Saved at stop when a stem is known; deleted by the deleteAudio command when the session is deleted.
    @Volatile private var takeStem: String? = null
    private fun audioDir() = File(activity.filesDir, "voxterm-audio").also { it.mkdirs() }
    private fun audioFile(stem: String) = File(audioDir(), "${stem.replace(Regex("[^A-Za-z0-9_-]"), "_")}.wav")

    // ---- live-preview state (only meaningful while phase == "recording") ----
    // Completed <=29 s windows decoded DURING recording (rough, no boundary nudging) plus the
    // in-progress window's latest decode. The GUI streams these live; at stop they're superseded by
    // decodeTake()'s authoritative result, so the saved transcript is unaffected.
    private val liveSegments = java.util.Collections.synchronizedList(mutableListOf<Pair<String, Double>>())
    @Volatile private var livePartialText = ""
    @Volatile private var livePartialStart = 0.0
    // Serializes every native decode so the live loop, the final pass, and the debug self-test never
    // call rec.decode() on overlapping threads.
    private val decodeLock = Any()

    // The Whisper model is bundled in assets/voxterm-model and staged to filesDir on first use, so
    // transcription is fully offline (no first-run download). @Synchronized so a debug self-test and
    // a first record can't both stage concurrently and race the dir swap below.
    @Synchronized
    private fun stagedModelDir(): File {
        val out = File(activity.filesDir, "voxterm-model")
        // Sentinel = ALL required files present so a half-copied dir self-heals instead of wedging.
        if (modelFiles.all { File(out, it).exists() }) return out
        // Copy into a temp dir, then atomically swap it in: `out` is only ever a complete dir.
        val tmp = File(activity.filesDir, "voxterm-model.tmp")
        tmp.deleteRecursively()
        tmp.mkdirs()
        val am = activity.assets
        for (name in am.list("voxterm-model") ?: arrayOf()) {
            am.open("voxterm-model/$name").use { input ->
                File(tmp, name).outputStream().use { input.copyTo(it) }
            }
        }
        // Verify the staged dir is COMPLETE before swapping it in: a build shipping incomplete assets
        // fails loudly here (surfaced as a transcribe error) instead of as a cryptic native recognizer
        // crash later. Never promote a partial dir into `out`.
        val missing = modelFiles.filterNot { File(tmp, it).exists() }
        if (missing.isNotEmpty()) {
            tmp.deleteRecursively()
            throw java.io.IOException("bundled model is incomplete; missing: ${missing.joinToString()}")
        }
        out.deleteRecursively()
        if (!tmp.renameTo(out)) {
            tmp.deleteRecursively()
            throw java.io.IOException("could not stage model dir (rename ${tmp.name} -> ${out.name} failed)")
        }
        return out
    }

    private fun buildRecognizer(dir: File): OfflineRecognizer {
        // lang.txt (staged by fetch-deps.sh): "en" forces English (the *.en models), "auto"/empty/
        // absent means auto-detect (the multilingual models). sherpa-onnx treats language="" as detect.
        val langTag = File(dir, "lang.txt").let { if (it.exists()) it.readText().trim() else "en" }
        val whisperLang = if (langTag == "auto" || langTag.isEmpty()) "" else langTag
        val config = OfflineRecognizerConfig(
            featConfig = FeatureConfig(sampleRate = SAMPLE_RATE, featureDim = 80),
            modelConfig = OfflineModelConfig(
                whisper = OfflineWhisperModelConfig(
                    encoder = File(dir, "encoder.int8.onnx").absolutePath,
                    decoder = File(dir, "decoder.int8.onnx").absolutePath,
                    language = whisperLang,   // "" = auto-detect (multilingual); "en" for the *.en models
                    task = "transcribe",      // not "translate"
                    enableTokenTimestamps = true,   // per-token times → place text into diarization segments
                ),
                tokens = File(dir, "tokens.txt").absolutePath,
                numThreads = 2,
                modelType = "whisper",
            ),
        )
        // assetManager defaults to null → the recognizer is built from the file paths above.
        return OfflineRecognizer(config = config)
    }

    // Build the recognizer once, lazily. @Synchronized so the mic worker and the debug self-test
    // thread can't both pass the null check and each build (then leak) a native recognizer.
    @Synchronized
    private fun ensureRecognizer(dir: File): OfflineRecognizer =
        recognizer ?: buildRecognizer(dir).also { recognizer = it }

    // Stage the diarization models (pyannote segmentation + speaker embedding) from assets to filesDir,
    // same atomic-swap pattern as stagedModelDir. Throws if the build didn't bundle them (diarization
    // is an opt-in extra; a build without the models simply can't diarize).
    @Synchronized
    private fun stagedDiarDir(): File {
        val out = File(activity.filesDir, "voxterm-diar")
        if (diarFiles.all { File(out, it).exists() }) return out
        val tmp = File(activity.filesDir, "voxterm-diar.tmp")
        tmp.deleteRecursively(); tmp.mkdirs()
        val am = activity.assets
        for (name in am.list("voxterm-diar") ?: arrayOf()) {
            am.open("voxterm-diar/$name").use { input -> File(tmp, name).outputStream().use { input.copyTo(it) } }
        }
        val missing = diarFiles.filterNot { File(tmp, it).exists() }
        if (missing.isNotEmpty()) { tmp.deleteRecursively(); throw java.io.IOException("diarization models not bundled; missing: ${missing.joinToString()}") }
        out.deleteRecursively()
        if (!tmp.renameTo(out)) { tmp.deleteRecursively(); throw java.io.IOException("could not stage diar dir") }
        return out
    }

    @Synchronized
    private fun ensureDiarizer(): OfflineSpeakerDiarization {
        diarizer?.let { return it }
        val dir = stagedDiarDir()
        val config = OfflineSpeakerDiarizationConfig(
            segmentation = OfflineSpeakerSegmentationModelConfig(
                pyannote = OfflineSpeakerSegmentationPyannoteModelConfig(model = File(dir, "segmentation.onnx").absolutePath),
                numThreads = 2,
            ),
            embedding = SpeakerEmbeddingExtractorConfig(model = File(dir, "embedding.onnx").absolutePath, numThreads = 2),
            // numClusters = -1 → infer speaker count from the audio via the clustering threshold.
            clustering = FastClusteringConfig(numClusters = -1, threshold = DIAR_THRESHOLD),
            minDurationOn = 0.3f,
            minDurationOff = 0.5f,
        )
        return OfflineSpeakerDiarization(config = config).also { diarizer = it }
    }

    // Decode one <=30 s window in a single offline pass (no isReady/endpoint loop — that's online).
    // Serialized via decodeLock so the live loop and the final pass never decode concurrently. Returns
    // the collapsed (authoritative) text plus raw tokens/timestamps (used only to place text into
    // diarization segments — the saved transcript always uses `.text`).
    private fun decodeChunk(rec: OfflineRecognizer, samples: FloatArray): Decoded = synchronized(decodeLock) {
        val stream: OfflineStream = rec.createStream()
        try {
            stream.acceptWaveform(samples, SAMPLE_RATE)   // (samples, sampleRate)
            rec.decode(stream)
            val r = rec.getResult(stream)
            Decoded(collapseRepeats(r.text.trim()), r.tokens, r.timestamps)
        } finally {
            stream.release()                              // never leak the native stream
        }
    }

    // Whisper (like any autoregressive ASR) can fall into a repetition loop on ambiguous, noisy, or
    // near-silent audio, emitting the same word or phrase many times in a row ("But it's like. But
    // it's like. But it's like. …"). Collapse any phrase of 1..MAX_REPEAT_PHRASE words that repeats
    // 3+ times consecutively down to a single copy. Conservative by design: a phrase must recur at
    // least twice MORE than its first occurrence, so natural doubles ("no no", "very very") survive.
    // Applied to every decode result (live preview + final pass), so segments are clean at the source.
    private fun collapseRepeats(text: String): String {
        if (text.isEmpty()) return text
        val words = text.split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (words.size < 2) return text
        val out = ArrayList<String>(words.size)
        var i = 0
        while (i < words.size) {
            out.add(words[i])
            var collapsed = false
            val maxLen = minOf(MAX_REPEAT_PHRASE, out.size)
            for (len in 1..maxLen) {
                val tailStart = out.size - len            // the phrase just emitted (its first occurrence)
                var j = i + 1
                var reps = 0
                while (j + len <= words.size && phraseMatches(out, tailStart, words, j, len)) {
                    reps++; j += len
                }
                if (reps >= 2) { i = j; collapsed = true; break }   // keep the one copy in `out`, drop the repeats
            }
            if (!collapsed) i++
        }
        return out.joinToString(" ")
    }

    private fun phraseMatches(a: List<String>, aStart: Int, b: List<String>, bStart: Int, len: Int): Boolean {
        for (k in 0 until len) if (a[aStart + k] != b[bStart + k]) return false
        return true
    }

    @Command
    fun startTranscribe(invoke: Invoke) {
        // The plugin owns the mic, so it owns acquiring the permission: on a fresh install the first
        // Start prompts (and resumes in micPermissionCallback) instead of hard-failing.
        if (getPermissionState("microphone") != PermissionState.GRANTED) {
            requestPermissionForAlias("microphone", invoke, "micPermissionCallback")
            return
        }
        beginCapture(invoke)
    }

    @PermissionCallback
    fun micPermissionCallback(invoke: Invoke) {
        if (getPermissionState("microphone") == PermissionState.GRANTED) {
            beginCapture(invoke)
        } else {
            invoke.reject("microphone permission is required for on-device transcription")
        }
    }

    private fun beginCapture(invoke: Invoke) {
        if (running) {
            invoke.resolve(JSObject())
            return
        }
        val gen = ++generation
        // Make sure a prior take's worker (e.g. one that outlived stop's 2 s join) has fully exited
        // and released the single-owner mic before we open a new AudioRecord — its loop guard
        // (generation == its own gen) is already false, so this join returns promptly.
        worker?.join(3000)
        liveWorker?.join(3000)            // ditto for the prior take's live decoder
        running = true
        // Reclaim any orphaned take files (e.g. from a take whose process was killed before decode's
        // cleanup ran) so spilled PCM never accumulates on disk across sessions.
        activity.filesDir.listFiles { f -> f.name.startsWith("take-") && f.name.endsWith(".pcm") }
            ?.forEach { it.delete() }
        // Pre-create THIS take's file so the live reader can open it immediately — without this it could
        // race the mic worker's FileOutputStream and lose live preview to a FileNotFoundException. The
        // mic worker's FileOutputStream(...) truncates this empty file in place (same inode), so the
        // reader's open handle still sees every appended sample.
        try { takeFile(gen).createNewFile() } catch (_: Exception) {}
        samplesWritten = 0
        synchronized(segments) { segments.clear() }
        synchronized(liveSegments) { liveSegments.clear() }
        synchronized(utterances) { utterances.clear() }
        speakerCount = 0; diarStatus = ""
        livePartialText = ""; livePartialStart = 0.0
        phase = "recording"; elapsedSec = 0.0; levelRms = 0.0; durationSec = 0.0; errorMsg = null
        worker = thread(start = true) {
            var audio: AudioRecord? = null
            try {
                val minBuf = AudioRecord.getMinBufferSize(
                    SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
                )
                if (minBuf <= 0) { fail("audio buffer size unavailable ($minBuf)", gen); return@thread }
                audio = AudioRecord(
                    MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, minBuf * 2
                )
                if (audio.state != AudioRecord.STATE_INITIALIZED) {
                    fail("could not initialize the microphone", gen); return@thread
                }
                val buf = ShortArray(minBuf)
                val bytes = ByteArray(minBuf * 2)
                // Spill PCM straight to disk (unbuffered FileOutputStream → each write hits the OS file,
                // so a RandomAccessFile reader in another thread sees it as soon as samplesWritten is bumped).
                FileOutputStream(takeFile(gen)).use { out ->
                    audio.startRecording()
                    while (running && generation == gen) {
                        val n = audio.read(buf, 0, buf.size)
                        if (n <= 0) continue
                        var sumSq = 0.0
                        for (i in 0 until n) {
                            val s = buf[i].toInt()
                            bytes[2 * i] = (s and 0xff).toByte()
                            bytes[2 * i + 1] = ((s shr 8) and 0xff).toByte()
                            val v = s / 32768.0; sumSq += v * v
                        }
                        out.write(bytes, 0, n * 2)
                        samplesWritten += n          // publish AFTER the write so readers never read past it
                        levelRms = sqrt(sumSq / n)
                        elapsedSec = samplesWritten / SAMPLE_RATE.toDouble()
                    }
                }
            } catch (e: Exception) {
                fail(e.message ?: "recording error", gen)
            } finally {
                try { audio?.stop() } catch (_: Exception) {}
                audio?.release()
            }
        }
        liveWorker = thread(start = true) { runLiveLoop(gen) }
        invoke.resolve(JSObject())
    }

    private fun fail(msg: String, gen: Int) {
        if (generation == gen) { errorMsg = msg; phase = "error"; running = false }
    }

    @Command
    fun stopTranscribe(invoke: Invoke) {
        if (phase != "recording") { invoke.resolve(JSObject()); return }   // double-stop / never started
        val args = try { invoke.parseArgs(StopArgs::class.java) } catch (_: Exception) { StopArgs() }
        takeStem = args.stem
        diarizeEnabled = args.diarize
        running = false
        worker?.join(2000)                  // the mic worker exits on running=false; beginCapture re-joins
        liveWorker?.join(2000)              // stop the live decoder before the final pass (no concurrent decode)
        invoke.resolve(JSObject())          // resolve now; transcription runs async, reported via poll
        val gen = generation                // a new take bumps generation → decodeTake aborts cleanly
        phase = "transcribing"
        thread(start = true) { decodeTake(gen) }
    }

    // Delete a session's persisted audio (called from the GUI when a session is deleted on-device).
    @Command
    fun deleteAudio(invoke: Invoke) {
        val args = try { invoke.parseArgs(StemArgs::class.java) } catch (_: Exception) { StemArgs() }
        val res = JSObject()
        res.put("deleted", if (args.stem.isNotEmpty() && audioFile(args.stem).delete()) true else false)
        invoke.resolve(res)
    }

    // Transcribe a finished take by streaming <=29 s windows (under Whisper's 30 s cap) off the spilled
    // PCM file — only one window is ever resident in RAM during decode. Aborts quietly if a newer take
    // started (generation changed). Then, if diarization is enabled for this take, runs sherpa's
    // OfflineSpeakerDiarization over the whole clip and attributes the transcript to speakers; and if a
    // stem is known, persists the audio as a WAV. Deletes the spilled PCM when finished.
    private fun decodeTake(gen: Int) {
        val file = takeFile(gen)
        decodeProgress = 0.0; diarStatus = ""
        // Token (absoluteTimeSec -> tokenString) pairs, collected only when diarizing — used to fill the
        // speaker segments with the Whisper text whose tokens land inside them.
        val tokTimes = if (diarizeEnabled) ArrayList<Pair<Double, String>>() else null
        try {
            val total = (if (file.exists()) file.length() else 0L) / 2
            durationSec = total / SAMPLE_RATE.toDouble()
            if (total == 0L) { if (generation == gen) phase = "done"; return }
            val rec = ensureRecognizer(stagedModelDir())
            val decodeSpan = if (diarizeEnabled) 0.5 else 1.0   // leave the second half of the bar for diarization
            RandomAccessFile(file, "r").use { raf ->
                var off = 0L
                while (off < total && generation == gen) {
                    val hardEnd = minOf(off + WHISPER_WINDOW, total)
                    val win = ByteArray(((hardEnd - off) * 2).toInt())
                    raf.seek(off * 2)
                    raf.readFully(win)
                    val len = windowLen(win, hardEnd >= total)   // nudge the cut off a quiet frame
                    val samples = FloatArray(len) {
                        val b = it * 2
                        val s = ((win[b + 1].toInt() shl 8) or (win[b].toInt() and 0xff)).toShort()
                        s / 32768.0f
                    }
                    val base = off / SAMPLE_RATE.toDouble()
                    val dec = decodeChunk(rec, samples)
                    if (dec.text.isNotEmpty()) segments.add(dec.text to base)
                    if (tokTimes != null) {
                        val n = minOf(dec.tokens.size, dec.stamps.size)
                        for (i in 0 until n) tokTimes.add((base + dec.stamps[i]) to dec.tokens[i])
                    }
                    off += len
                    decodeProgress = (off.toDouble() / total * decodeSpan).coerceIn(0.0, 1.0)
                }
            }
            // Persist the audio for this session (replay / re-diarize) before we drop the spilled PCM.
            takeStem?.let { stem -> try { savePcmAsWav(file, audioFile(stem)) } catch (e: Exception) { Log.w("voxasr", "audio save failed: ${e.message}") } }
            // Speaker diarization (opt-in) — failures fall back to the single-speaker window transcript.
            if (diarizeEnabled && generation == gen) {
                try { diarize(file, total, tokTimes ?: emptyList(), gen) }
                catch (e: Exception) { Log.w("voxasr", "diarization failed (using single-speaker): ${e.message}") }
            }
            if (generation == gen) { decodeProgress = 1.0; diarStatus = ""; phase = "done" }
        } catch (e: Exception) {
            if (generation == gen) { errorMsg = e.message ?: "transcription error"; phase = "error" }
        } finally {
            file.delete()   // reclaim the spilled PCM; the transcript now lives in `segments`/`utterances`
        }
    }

    // Run sherpa's OfflineSpeakerDiarization over the whole take and build speaker-attributed utterances
    // by filling each [start,end,speaker] segment with the Whisper tokens whose timestamps fall inside.
    // Needs the whole clip in RAM as floats (the time+memory cost the user opted into).
    private fun diarize(file: File, total: Long, tokTimes: List<Pair<Double, String>>, gen: Int) {
        diarStatus = "diarizing"
        val samples = FloatArray(total.toInt())
        RandomAccessFile(file, "r").use { raf ->
            val buf = ByteArray(1 shl 16)
            var idx = 0
            while (idx < samples.size) {
                val want = minOf(buf.size, (samples.size - idx) * 2)
                val got = raf.read(buf, 0, want)
                if (got <= 0) break
                var b = 0
                while (b + 1 < got) { samples[idx++] = (((buf[b + 1].toInt() shl 8) or (buf[b].toInt() and 0xff)).toShort()) / 32768.0f; b += 2 }
            }
        }
        val diar = ensureDiarizer()
        val segs = diar.processWithCallback(samples, { done, totalc, _ ->
            if (generation == gen && totalc > 0) decodeProgress = (0.5 + 0.5 * done.toDouble() / totalc).coerceIn(0.0, 1.0)
            0   // 0 = keep going
        })
        var spkMax = -1
        var prevEnd: Double? = null
        synchronized(utterances) {
            utterances.clear()
            for (s in segs.sortedBy { it.start }) {
                val txt = tokTimes.filter { it.first >= s.start && it.first < s.end }.joinToString("") { it.second }
                val clean = collapseRepeats(txt.replace(Regex("\\s+"), " ").trim())
                if (clean.isEmpty()) continue
                val gap = prevEnd?.let { (s.start - it).coerceAtLeast(0.0) }
                utterances.add(Utt(clean, s.start.toDouble(), s.end.toDouble(), s.speaker, gap))
                if (s.speaker > spkMax) spkMax = s.speaker
                prevEnd = s.end.toDouble()
            }
        }
        speakerCount = spkMax + 1
        diarStatus = ""
    }

    // Write a take's spilled PCM16 (mono, 16 kHz) to a canonical 44-byte-header WAV at `dst`.
    private fun savePcmAsWav(pcm: File, dst: File) {
        val dataLen = pcm.length().toInt()
        dst.outputStream().use { o ->
            val hdr = java.nio.ByteBuffer.allocate(44).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            hdr.put("RIFF".toByteArray()); hdr.putInt(36 + dataLen); hdr.put("WAVE".toByteArray())
            hdr.put("fmt ".toByteArray()); hdr.putInt(16); hdr.putShort(1.toShort())          // PCM
            hdr.putShort(1.toShort()); hdr.putInt(SAMPLE_RATE); hdr.putInt(SAMPLE_RATE * 2)    // mono, byte rate
            hdr.putShort(2.toShort()); hdr.putShort(16.toShort())                             // block align, bits
            hdr.put("data".toByteArray()); hdr.putInt(dataLen)
            o.write(hdr.array())
            pcm.inputStream().use { it.copyTo(o) }
        }
    }

    // Samples of `win` (one candidate window, starting at the window's first sample) to actually
    // consume: the whole window if it reaches the take's end, otherwise nudged back to the quietest
    // 10 ms frame in the last 2 s so a word straddling the boundary isn't sliced.
    private fun windowLen(win: ByteArray, isLast: Boolean): Int {
        val samples = win.size / 2
        if (isLast || samples < WHISPER_WINDOW) return samples
        val frame = SAMPLE_RATE / 100                 // 10 ms
        var bestIdx = samples
        var bestEnergy = Double.MAX_VALUE
        var i = samples - 2 * SAMPLE_RATE
        while (i + frame <= samples) {
            var sum = 0.0
            for (j in i until i + frame) {
                val b = j * 2
                val s = ((win[b + 1].toInt() shl 8) or (win[b].toInt() and 0xff)).toShort()
                val v = s / 32768.0; sum += v * v
            }
            if (sum < bestEnergy) { bestEnergy = sum; bestIdx = i + frame }
            i += frame
        }
        return bestIdx
    }

    // Live preview: while recording, repeatedly decode the in-progress <=29 s window so a rough
    // transcript streams in. A window is finalized into liveSegments once it fills; the current
    // window's latest decode is livePartialText. Work per pass is bounded (<=29 s of audio) and there
    // is no boundary nudging (that's the final pass's job). Exits when running flips false or a newer
    // take starts. Whisper re-decodes the whole window each pass, so the partial can shift slightly
    // until the window finalizes — that's expected for an offline (non-streaming) model.
    private fun runLiveLoop(gen: Int) {
        val rec = try {
            ensureRecognizer(stagedModelDir())
        } catch (e: Exception) {
            // No live preview if the model can't load — the final pass at stop still runs and is the
            // one that surfaces a real error to the user. Don't kill the recording over a preview.
            Log.w("voxasr", "live preview disabled (recognizer init failed): ${e.message}")
            return
        }
        // Read the in-progress window straight off the spilled PCM file (only `samplesWritten` samples
        // are flushed), so the live loop never copies the whole growing take like it used to.
        val raf = try { RandomAccessFile(takeFile(gen), "r") } catch (e: Exception) {
            Log.w("voxasr", "live preview disabled (take file unavailable): ${e.message}"); return
        }
        try {
            var base = 0                    // first sample of the in-progress window
            var lastEnd = -1                // sample count at the last decode (skip if nothing new)
            val minNew = SAMPLE_RATE / 2    // re-decode once >=0.5 s of fresh audio has accumulated
            while (running && generation == gen) {
                val total = samplesWritten
                val end = minOf(base + WHISPER_WINDOW, total)
                val capped = (end - base) >= WHISPER_WINDOW && end != lastEnd   // window just hit 29 s
                if (end - base < minNew || (end - lastEnd < minNew && !capped)) {
                    try { Thread.sleep(150) } catch (_: InterruptedException) {}
                    continue
                }
                val win = ByteArray((end - base) * 2)
                try { raf.seek(base.toLong() * 2); raf.readFully(win) }
                catch (e: Exception) { try { Thread.sleep(150) } catch (_: InterruptedException) {}; continue }
                val samples = FloatArray(end - base) {
                    val b = it * 2
                    val s = ((win[b + 1].toInt() shl 8) or (win[b].toInt() and 0xff)).toShort()
                    s / 32768.0f
                }
                val text = try { decodeChunk(rec, samples).text } catch (e: Exception) { "" }
                if (generation != gen) break
                livePartialStart = base / SAMPLE_RATE.toDouble()
                livePartialText = text
                lastEnd = end
                if (end - base >= WHISPER_WINDOW) {              // window full → finalize, open the next
                    if (text.isNotEmpty()) liveSegments.add(text to base / SAMPLE_RATE.toDouble())
                    base = end
                    livePartialText = ""
                    lastEnd = -1
                }
            }
        } finally {
            try { raf.close() } catch (_: Exception) {}
        }
    }

    // The webview polls this: { phase, elapsed, level, durationSec, error?, segments:[{text,start}] }.
    @Command
    fun pollTranscript(invoke: Invoke) {
        val res = JSObject()
        res.put("phase", phase)
        res.put("elapsed", elapsedSec)
        res.put("level", levelRms)
        res.put("durationSec", durationSec)
        res.put("progress", decodeProgress)
        if (diarStatus.isNotEmpty()) res.put("stage", diarStatus)   // "diarizing" → the GUI shows it in the bar msg
        errorMsg?.let { res.put("error", it) }
        val arr = org.json.JSONArray()
        synchronized(segments) {
            for ((text, start) in segments) {
                arr.put(org.json.JSONObject().put("text", text).put("start", start))
            }
        }
        res.put("segments", arr)
        // Speaker-attributed utterances (populated only when diarization ran). The GUI prefers these
        // over `segments` so the saved transcript is multi-speaker.
        val uttArr = org.json.JSONArray()
        synchronized(utterances) {
            for (u in utterances) {
                val o = org.json.JSONObject().put("text", u.text).put("start", u.start).put("end", u.end).put("speaker", u.speaker)
                u.gapBefore?.let { o.put("gapBefore", it) }
                uttArr.put(o)
            }
        }
        res.put("utterances", uttArr)
        res.put("speakerCount", speakerCount)
        // Live preview (the GUI reads these only while phase == "recording"): finalized windows so
        // far plus the in-progress window's latest decode.
        val liveArr = org.json.JSONArray()
        synchronized(liveSegments) {
            for ((text, start) in liveSegments) {
                liveArr.put(org.json.JSONObject().put("text", text).put("start", start))
            }
        }
        res.put("liveLines", liveArr)
        val lp = livePartialText
        if (lp.isNotEmpty()) {
            res.put("livePartial", org.json.JSONObject().put("text", lp).put("start", livePartialStart))
        }
        invoke.resolve(res)
    }

    // Debug self-test: on debuggable builds, transcribe the bundled clip through the SAME offline
    // recognizer and log the result + xRT. Proves on-device decoding works without a mic. No-op on
    // release builds.
    override fun load(webView: WebView) {
        super.load(webView)
        val debuggable = (activity.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
        if (debuggable) Thread { runSelfTest() }.start()
    }

    private fun readWav16kMono(f: File): FloatArray {
        val bytes = f.readBytes()                       // canonical 16k mono PCM16 WAV: 44-byte header
        val n = (bytes.size - 44) / 2
        val out = FloatArray(n)
        var j = 44
        for (i in 0 until n) {
            val s = ((bytes[j + 1].toInt() shl 8) or (bytes[j].toInt() and 0xff)).toShort()
            out[i] = s / 32768.0f
            j += 2
        }
        return out
    }

    private fun runSelfTest() {
        try {
            val dir = stagedModelDir()
            val test = File(dir, "test.wav")
            if (!test.exists()) { Log.i("voxasr", "SELFTEST_SKIP no test.wav"); return }
            val rec = ensureRecognizer(dir)
            val samples = readWav16kMono(test)
            val audioSec = samples.size / 16000.0
            val t0 = System.nanoTime()
            val text = decodeChunk(rec, samples).text
            val decodeSec = (System.nanoTime() - t0) / 1e9
            // xRT < 1.0 = faster than real-time; logged so on-device latency is measured, not assumed.
            Log.i("voxasr", "SELFTEST_RESULT=[$text] xRT=%.2f (%.2fs decode / %.2fs audio)"
                .format(decodeSec / audioSec, decodeSec, audioSec))
        } catch (e: Exception) {
            Log.e("voxasr", "SELFTEST_ERROR", e)
        }
    }
}
