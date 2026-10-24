package org.pockettts.android.player

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.pockettts.android.R
import org.pockettts.android.engine.EngineTurn
import org.pockettts.android.engine.ModelCatalog
import org.pockettts.android.engine.ModelManager
import org.pockettts.android.engine.NativeEngine
import org.pockettts.android.engine.PocketTts
import org.pockettts.android.engine.Settings
import org.pockettts.android.engine.VoiceCatalog
import org.pockettts.android.engine.WavReader
import org.pockettts.android.ui.ModelNames
import java.io.File
import java.io.IOException
import kotlin.coroutines.cancellation.CancellationException
import kotlin.coroutines.coroutineContext
import kotlin.math.min

/**
 * Plays a few seconds of a voice, so it can be chosen by ear rather than by name.
 *
 * The sample is the voice's own reference audio, not a synthesised sentence.
 * That is not a shortcut: in Pocket TTS a voice *is* a few seconds of reference
 * audio the model conditions on, so this is the most direct answer to "what does
 * this one sound like". It also means a sample costs nothing but the prompt
 * download - no 125 MB model bundle, no inference, no waiting - and works on a
 * fresh install before the model has ever been fetched. The download it does is
 * the same one selecting the voice would trigger later, so browsing by ear warms
 * the cache instead of duplicating work.
 *
 * The trade-off is honest to state: this is the raw prompt, so it carries the
 * recording's own room and pacing. The synthesised voice tracks its timbre
 * closely, which is the thing being chosen, but not its background.
 *
 * Two cases have no recording to play. Reading in another language than
 * English, the recording would be the speaker in English (or in their own
 * language) when the question is how they sound reading this one. And Kyutai's
 * recordings of the native Italian, Spanish, German, Portuguese and Dutch
 * voices are behind a login. In both the sample is the voice reading the
 * language's sample sentence on this phone, which needs that model
 * downloaded, and says so when it is not.
 */
object VoiceSample {

    private const val TAG = "VoiceSample"

    /** A sample is for recognising a voice, not for listening to a recording. */
    const val MAX_SECONDS = 6f

    /** Long enough to remove the click of a hard cut, short enough not to be heard. */
    private const val FADE_SECONDS = 0.12f

    sealed interface State {
        data object Idle : State

        /** Fetching the prompt; the first play of a voice needs the network. */
        data class Loading(val voiceId: String) : State

        data class Playing(val voiceId: String) : State
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    /**
     * Failures, as one-shot events rather than state. A row's appearance and a
     * message that should be shown once are different things, and folding the
     * second into the first leaves the UI responsible for clearing it.
     */
    private val _errors = MutableSharedFlow<String>(extraBufferCapacity = 1)
    val errors: SharedFlow<String> = _errors.asSharedFlow()

    private val control = Mutex()
    private var job: Job? = null

    @Volatile
    private var player: StreamingPlayer? = null

    /** The spoken sample's generation, so stopping it stops the engine too. */
    @Volatile
    private var token: NativeEngine.CancelToken? = null

    /** The voice whose sample is loading or playing, if any. */
    val activeVoiceId: String?
        get() = when (val current = _state.value) {
            is State.Loading -> current.voiceId
            is State.Playing -> current.voiceId
            State.Idle -> null
        }

    /**
     * Starts [voiceId], or stops it if it is the one already sounding.
     *
     * @return false if a sample cannot start because the reader is speaking.
     *   Two voices at once is not a preview, it is a mess, and stopping someone's
     *   article to audition a voice is not ours to decide.
     */
    fun toggle(context: Context, voiceId: String): Boolean {
        if (activeVoiceId == voiceId) {
            stop()
            return true
        }
        if (Reader.isActive) return false
        play(context, voiceId)
        return true
    }

    fun play(context: Context, voiceId: String) {
        val appContext = context.applicationContext
        scope.launch {
            control.withLock {
                stopLocked()
                job = scope.launch { run(appContext, voiceId) }
            }
        }
    }

    fun stop() {
        scope.launch { control.withLock { stopLocked() } }
    }

    /** Whether [voiceId]'s sample is synthesised rather than its recording; see the class doc. */
    internal fun isSpoken(context: Context, voiceId: String): Boolean {
        val voice = VoiceCatalog.byId(voiceId) ?: return false
        return voice.url == null || Settings(context).readingLanguage != Settings.ENGLISH
    }

    private suspend fun run(context: Context, voiceId: String) {
        if (isSpoken(context, voiceId)) return speak(context, voiceId)
        var localPlayer: StreamingPlayer? = null
        try {
            _state.value = State.Loading(voiceId)
            val file = resolve(context, voiceId)
            coroutineContext.ensureActive()

            val audio = withContext(Dispatchers.IO) { WavReader.read(file) }
            val samples = trim(audio.samples, audio.sampleRate, MAX_SECONDS, FADE_SECONDS)
            if (samples.isEmpty()) throw IOException("The prompt for $voiceId is empty")
            coroutineContext.ensureActive()

            localPlayer = StreamingPlayer(audio.sampleRate).also {
                player = it
                it.start()
            }
            _state.value = State.Playing(voiceId)
            if (localPlayer.write(samples)) localPlayer.drain()
            _state.value = State.Idle
        } catch (cancelled: CancellationException) {
            _state.value = State.Idle
            throw cancelled
        } catch (error: Throwable) {
            Log.w(TAG, "Sample for $voiceId failed", error)
            _state.value = State.Idle
            _errors.tryEmit(error.message ?: error.javaClass.simpleName)
        } finally {
            localPlayer?.release()
            if (player === localPlayer) player = null
        }
    }

    /** [voiceId] reading the reading language's sample sentence. */
    private suspend fun speak(context: Context, voiceId: String) {
        var localPlayer: StreamingPlayer? = null
        val cancel = NativeEngine.CancelToken()
        token = cancel
        try {
            _state.value = State.Loading(voiceId)
            val settings = Settings(context)
            val manager = ModelManager(context)
            if (!manager.isModelInstalled) {
                val language = ModelNames.language(manager.model.language)
                throw IOException(context.getString(R.string.sample_needs_model, language))
            }
            val engine = PocketTts.get(context)
            val voice = engine.loadVoice(VoiceCatalog.byId(voiceId) ?: VoiceCatalog.default())
            val text = ModelCatalog.get(context).byName(engine.modelName)?.sampleText
                ?: throw IOException("No sample sentence for ${engine.modelName}")
            coroutineContext.ensureActive()

            localPlayer = StreamingPlayer(engine.sampleRate).also {
                player = it
                it.start()
            }
            val sink = localPlayer
            // A read asked for while this one is composing wins the engine.
            val turn = EngineTurn.take()
            engine.synthesize(
                text,
                voice,
                speed = 1f,
                numSteps = settings.decodeSteps,
                temperature = settings.temperatureFor(engine.modelName),
                seed = settings.voiceSeed,
                cancel = cancel,
            ) { samples ->
                if (EngineTurn.superseded(turn)) return@synthesize false
                _state.value = State.Playing(voiceId)
                sink.write(samples)
            }
            sink.drain()
            _state.value = State.Idle
        } catch (cancelled: CancellationException) {
            _state.value = State.Idle
            throw cancelled
        } catch (error: Throwable) {
            Log.w(TAG, "Spoken sample for $voiceId failed", error)
            _state.value = State.Idle
            _errors.tryEmit(error.message ?: error.javaClass.simpleName)
        } finally {
            localPlayer?.release()
            if (player === localPlayer) player = null
            if (token === cancel) token = null
            cancel.close()
        }
    }

    private suspend fun resolve(context: Context, voiceId: String): File {
        val models = ModelManager(context)
        VoiceCatalog.byId(voiceId)?.let { return models.ensureVoice(it) }
        // Not a stock voice, so it is one the user imported and already local.
        val imported = models.voiceFile(voiceId)
        if (imported.isFile) return imported
        throw IOException("No audio for voice $voiceId")
    }

    private suspend fun stopLocked() {
        // Stopping the player first unblocks a write parked on buffer space, so
        // the cancel below does not have to wait it out.
        player?.stop()
        token?.cancel()
        job?.cancelAndJoin()
        job = null
        _state.value = State.Idle
    }

    /**
     * Cuts [samples] to [maxSeconds] and fades the tail out.
     *
     * Some prompts run far longer than a preview should, and a hard cut in the
     * middle of a vowel is an audible click - the discontinuity is a step, and a
     * step is broadband. The fade only applies to audio that was actually cut;
     * a prompt already shorter than the limit is passed through untouched, so
     * its own natural ending is not clipped into a fade.
     */
    internal fun trim(
        samples: FloatArray,
        sampleRate: Int,
        maxSeconds: Float,
        fadeSeconds: Float,
    ): FloatArray {
        if (sampleRate <= 0) return samples
        val limit = (maxSeconds * sampleRate).toInt()
        if (limit <= 0 || samples.size <= limit) return samples

        val cut = samples.copyOf(limit)
        val fade = min((fadeSeconds * sampleRate).toInt(), cut.size)
        for (index in 0 until fade) {
            // Linear is enough over ~120ms; the point is to remove the step, not
            // to shape the decay. The ramp counts from one so that the final
            // sample lands on exactly zero - dividing by `fade` alone leaves
            // 1/fade of the amplitude behind, which is a step, just a small one.
            cut[cut.size - fade + index] *= 1f - (index + 1).toFloat() / fade
        }
        return cut
    }
}
