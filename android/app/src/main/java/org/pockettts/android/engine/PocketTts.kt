package org.pockettts.android.engine

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.pockettts.android.debug.Metrics
import org.pockettts.android.debug.VoiceTrace
import sonic.Sonic
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * Owns the one loaded copy of the model.
 *
 * The engine is pocket-speak's, the Rust crate the desk CLI runs, over JNI
 * ([NativeEngine]): upstream's own Rust implementation of the model since
 * 2026-10, and before that an ONNX export run on ONNX Runtime, which replaced
 * sherpa-onnx on 2026-10-03 (`android/docs/engine-swap.md`,
 * `android/docs/pure-rust-engine.md`).
 *
 * Loading takes about half a second on a Galaxy S25 (0.43-0.67 s measured)
 * and a few hundred megabytes of RAM, so the engine
 * is a process-wide singleton shared by the reader UI, the scratchpad and the
 * system TTS service - all three of which can be alive at once. Synthesis is
 * serialised: the engine keeps per-generation state, so two at once would
 * interleave into noise.
 */
class PocketTts private constructor(
    private val engine: NativeEngine,
    private val modelManager: ModelManager,
) {

    /** The [ModelCatalog] model this instance reads with. */
    val modelName: String get() = modelManager.modelName

    val sampleRate: Int = engine.sampleRate

    private val synthesisLock = Mutex()

    /**
     * Set once [retire] has freed the engine, under both [synthesisLock] and
     * the voice cache's monitor: every native call takes one of them and
     * checks this first, so none can reach a freed engine.
     */
    @Volatile
    private var closed = false

    /** A voice embedding, loaded into one engine once and reused there. */
    class LoadedVoice internal constructor(
        val id: String,
        internal val handle: Long,
        /** The [PocketTts] it was loaded into: a voice is one model's state. */
        internal val owner: PocketTts,
    ) {
        override fun equals(other: Any?): Boolean =
            this === other || (other is LoadedVoice && id == other.id && owner === other.owner)

        override fun hashCode(): Int = id.hashCode()
    }

    /** A read that outlived its model: another one was chosen meanwhile. */
    class ModelChangedException(model: String) :
        IllegalStateException("The voice model changed from $model; start the read again")

    // Written from Dispatchers.IO by both the reader and the system engine, so
    // not a plain map.
    private val voiceCache = ConcurrentHashMap<String, LoadedVoice>()

    suspend fun loadVoice(voice: VoiceCatalog.Voice): LoadedVoice = withContext(Dispatchers.IO) {
        voiceCache[voice.id]?.let { return@withContext it }
        val file = modelManager.ensureEmbedding(voice)
        synchronized(voiceCache) {
            if (closed) throw ModelChangedException(modelName)
            voiceCache[voice.id] ?: LoadedVoice(voice.id, engine.loadVoice(file.absolutePath), this@PocketTts)
                .also { voiceCache[voice.id] = it }
        }
    }

    /**
     * Frees the engine and its voices, after any synthesis in progress. Calls
     * made on this instance afterwards throw [ModelChangedException].
     */
    private suspend fun retire() = synthesisLock.withLock {
        synchronized(voiceCache) {
            closed = true
            voiceCache.values.forEach { engine.freeVoice(it.handle) }
            voiceCache.clear()
            engine.close()
        }
        Log.i(TAG, "Freed $modelName")
    }

    /**
     * A voice cloned from the user's own recording.
     *
     * Not available on this engine: pocket-speak is conditioned on precomputed
     * embeddings, and making one from a wav needs the Mimi encoder, which its
     * bundle does not carry. Callers fall back to a stock voice. Bringing
     * cloning back is a documented later step (`android/docs/engine-swap.md`).
     */
    @Suppress("UNUSED_PARAMETER")
    fun loadVoiceFile(id: String, file: File): LoadedVoice =
        throw UnsupportedOperationException("Cloned voices are not supported by this engine yet")

    /**
     * Synthesises [text] and hands audio to [onAudio] as it is produced.
     *
     * @param speed 1 is the voice's own pace. Anything else is applied after
     *   generation by Sonic, which changes tempo and keeps pitch; the model
     *   has no speed input of its own.
     * @param numSteps unused: this engine decodes the flow in one step, as the
     *   reference implementation does. Kept so the setting's callers compile.
     * @param temperature width of the neighbourhood the speaker is drawn from.
     * @param seed restarts sampling from this seed, or -1 to keep drawing.
     * @param cancel stops this generation at the next frame once cancelled, or
     *   at the first if it already was. Its owner makes it before calling, so
     *   a cancel during the wait for [synthesisLock] is not lost.
     * @param onAudio receives float samples, clamped to [-1, 1]; return false
     *   to abandon the rest of this utterance.
     * @return false if generation was stopped early.
     */
    @Suppress("UNUSED_PARAMETER")
    suspend fun synthesize(
        text: String,
        voice: LoadedVoice,
        speed: Float,
        numSteps: Int,
        temperature: Float,
        seed: Int,
        cancel: NativeEngine.CancelToken? = null,
        onAudio: (FloatArray) -> Boolean,
    ): Boolean = synthesisLock.withLock {
        if (closed || voice.owner !== this) throw ModelChangedException(modelName)
        withContext(Dispatchers.Default) {
            var completed = true
            val stretch = Stretch.forSpeed(speed, sampleRate)
            val startedAt = System.currentTimeMillis()
            var firstSampleMillis = -1L
            val emit: (FloatArray) -> Boolean = { samples ->
                if (samples.isEmpty() || onAudio(samples)) true else {
                    completed = false
                    false
                }
            }
            try {
                val stats = engine.synthesize(voice.handle, text, temperature, seed.toLong(), cancel) { samples ->
                    if (firstSampleMillis < 0) firstSampleMillis = System.currentTimeMillis() - startedAt
                    // The decoder can overshoot full scale. Sonic narrows
                    // floats to 16-bit without a clamp, so an overshoot would
                    // wrap into a full-scale click; clamped here for both paths.
                    for (i in samples.indices) samples[i] = samples[i].coerceIn(-1f, 1f)
                    emit(stretch?.process(samples) ?: samples)
                }
                val end = runCatching { JSONObject(stats).optString("end") }.getOrNull()
                if (end == "Cancelled") completed = false
                // Sonic's tail only for a chunk that finished.
                if (completed && stretch != null) emit(stretch.drain())
            } finally {
                VoiceTrace.generated(
                    voiceId = voice.id,
                    promptSamples = 0,
                    promptRate = sampleRate,
                    promptHash = voice.handle.hashCode(),
                    temperature = temperature,
                    seed = seed,
                    firstSampleMillis = firstSampleMillis,
                )
            }
            completed
        }
    }

    /** Sonic, set to one speed for one chunk. Null at speed 1. */
    internal class Stretch private constructor(private val sonic: Sonic) {
        fun process(samples: FloatArray): FloatArray {
            sonic.writeFloatToStream(samples, samples.size)
            return read()
        }

        fun drain(): FloatArray {
            sonic.flushStream()
            return read()
        }

        private fun read(): FloatArray {
            val out = FloatArray(sonic.samplesAvailable())
            val n = if (out.isEmpty()) 0 else sonic.readFloatFromStream(out, out.size)
            return if (n == out.size) out else out.copyOf(n)
        }

        companion object {
            fun forSpeed(speed: Float, sampleRate: Int): Stretch? =
                if (kotlin.math.abs(speed - 1f) < 0.01f) {
                    null
                } else {
                    Stretch(Sonic(sampleRate, 1).apply { this.speed = speed })
                }
        }
    }

    companion object {
        private const val TAG = "PocketTts"

        @Volatile
        private var instance: PocketTts? = null
        private val loadLock = Mutex()

        /**
         * Returns the shared engine for [wanted], the selected model unless a
         * request in another language asks for another, loading it if
         * necessary. Suspends for as long as the model takes to download on
         * first run. When another model has been selected since, the engine
         * for the old one is freed once its synthesis in progress is done.
         */
        suspend fun get(
            context: Context,
            wanted: String = Settings(context).modelName,
            progress: ModelManager.ProgressListener? = null,
        ): PocketTts {
            instance?.takeIf { it.modelName == wanted }?.let { return it }
            return loadLock.withLock {
                instance?.takeIf { it.modelName == wanted } ?: run {
                    // Freed before the next is loaded: two models at once may
                    // not fit in a phone's memory.
                    instance?.let { old ->
                        instance = null
                        old.retire()
                    }
                    create(context.applicationContext, wanted, progress).also { instance = it }
                }
            }
        }

        /** The already-loaded engine, if any. Never triggers a load. */
        fun peek(): PocketTts? = instance

        private suspend fun create(
            context: Context,
            modelName: String,
            progress: ModelManager.ProgressListener?,
        ): PocketTts = withContext(Dispatchers.IO) {
            // Through the shared installer rather than straight to the manager,
            // so a read that arrives while the download button's transfer is
            // running joins it instead of starting a second one over the same
            // files.
            val files = ModelInstall.ensure(context, modelName)
            val manager = ModelManager(context, files.modelName)
            val settings = Settings(context)

            Log.i(TAG, "Loading pocket-speak's engine from ${files.dir}")
            // Timed because the process is killed while cached routinely, and
            // this is paid again on the next read.
            val startedAt = System.currentTimeMillis()
            val engine = NativeEngine.load(files.dir.absolutePath, files.modelName, settings.numThreads)
            Metrics.modelLoadMillis = System.currentTimeMillis() - startedAt
            manager.removeRetiredModels()
            PocketTts(engine, manager)
        }
    }
}
