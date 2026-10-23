package org.pockettts.android.engine

/**
 * pocket-speak's engine, the Rust crate the desk CLI runs, over JNI.
 *
 * `pocket-speak/crates/android` builds `libpocket_tts_jni.so`; ONNX Runtime is
 * Microsoft's `onnxruntime-android`, loaded first so the Rust side can open it
 * by name. One instance per process, never closed: [PocketTts] holds it for
 * the process's life. Synthesis is serialised by [PocketTts]; the native side
 * also locks the engine, which is what keeps a voice load from overlapping a
 * synthesis. A [CancelToken] may be cancelled from any thread.
 */
class NativeEngine private constructor(private var handle: Long) : AutoCloseable {

    /** Receives decoded audio; return false to stop the chunk. */
    fun interface Sink {
        fun onAudio(samples: FloatArray): Boolean
    }

    val sampleRate: Int get() = nativeSampleRate(handle)

    /** A voice embedding loaded for this engine. Free it with [freeVoice]. */
    fun loadVoice(path: String): Long = nativeLoadVoice(handle, path)

    fun freeVoice(voice: Long) = nativeFreeVoice(voice)

    /**
     * Synthesises one chunk. [seed] below zero keeps the running generator.
     * [cancel], if given, stops it at the next frame once cancelled - or at
     * the first, if it was cancelled before this call.
     * @return the chunk's stats as JSON, as the engine reports them.
     */
    fun synthesize(voice: Long, text: String, temperature: Float, seed: Long, cancel: CancelToken?, sink: Sink): String =
        nativeSynthesize(handle, voice, text, temperature, seed, cancel?.raw() ?: 0L, sink)

    override fun close() {
        val h = handle
        handle = 0
        nativeFree(h)
    }

    /**
     * Stops one request, across all of its chunks. Its owner makes it before
     * waiting for the engine, so a cancel that arrives during the wait still
     * lands on its own synthesis, and closes it once the request is over. A
     * cancel after close does nothing.
     */
    class CancelToken : AutoCloseable {
        // Guarded by `this`: freed under the same lock cancel() takes, so a
        // cancel never touches a freed token.
        private var token = nativeNewCancel()

        @Synchronized
        fun cancel() {
            if (token != 0L) nativeCancel(token)
        }

        /** The native token. Valid until [close], which only the owner calls. */
        @Synchronized
        internal fun raw(): Long = token

        @Synchronized
        override fun close() {
            nativeFreeCancel(token)
            token = 0
        }
    }

    companion object {
        init {
            System.loadLibrary("onnxruntime")
            System.loadLibrary("pocket_tts_jni")
        }

        /** Loads the model files in [modelDir]. Throws if any is missing or wrong. */
        fun load(modelDir: String, threads: Int): NativeEngine =
            NativeEngine(nativeLoad(modelDir, threads, "libonnxruntime.so"))

        @JvmStatic private external fun nativeLoad(modelDir: String, threads: Int, ortLibrary: String): Long
        @JvmStatic private external fun nativeFree(handle: Long)
        @JvmStatic private external fun nativeSampleRate(handle: Long): Int
        @JvmStatic private external fun nativeLoadVoice(handle: Long, path: String): Long
        @JvmStatic private external fun nativeFreeVoice(voice: Long)
        @JvmStatic private external fun nativeNewCancel(): Long
        @JvmStatic private external fun nativeCancel(token: Long)
        @JvmStatic private external fun nativeFreeCancel(token: Long)
        @JvmStatic private external fun nativeSynthesize(
            handle: Long,
            voice: Long,
            text: String,
            temperature: Float,
            seed: Long,
            token: Long,
            sink: Sink,
        ): String
    }
}
