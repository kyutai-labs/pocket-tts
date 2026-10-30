package org.pockettts.android.engine

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.job
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Semaphore
import org.pockettts.android.debug.VoiceTrace
import org.pockettts.android.speech.TextChunker
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.cancellation.CancellationException

/**
 * Exposes Pocket TTS as a system-wide Android text-to-speech engine.
 *
 * This is the part that does the most work for the least ceremony. Once this
 * service is selected in Settings, every app that already knows how to read
 * text - Select to Speak, Chrome's read-aloud, ebook readers, accessibility
 * tools - speaks in a Pocket TTS voice, without any of them knowing this app
 * exists.
 *
 * Audio is handed back chunk by chunk as it is generated rather than at the
 * end, so the caller starts playing after the first sentence instead of after
 * the last one.
 */
class PocketTtsService : TextToSpeechService() {

    private val stopRequested = AtomicBoolean(false)

    private val catalog: ModelCatalog get() = ModelCatalog.get(this)

    /** The request being synthesised, so [onStop] cancels that one and no other. */
    @Volatile
    private var current: NativeEngine.CancelToken? = null

    // The framework declares these protected, and they are widened here so
    // they can be called from a test. This is the surface other apps drive,
    // in whatever order they like, and it had no tests at all - which is a
    // worse problem than a slightly larger public API on a service nothing
    // else in this app talks to.

    /**
     * A language is available once it can be read: it is the app's reading
     * language (whose model the app's first screen fetches), or another with a
     * model on the phone. One on offer but not downloaded is missing data, so
     * a caller can say so rather than read it in the wrong language.
     */
    public override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int {
        val language = SystemVoices.languageOf(lang, catalog.languages)
            ?: return TextToSpeech.LANG_NOT_SUPPORTED
        return when {
            Settings(this).modelFor(language) == null -> TextToSpeech.LANG_MISSING_DATA
            country.isNullOrEmpty() -> TextToSpeech.LANG_AVAILABLE
            language == Settings.ENGLISH && supportedCountries.any { it.equals(country, ignoreCase = true) } ->
                TextToSpeech.LANG_COUNTRY_AVAILABLE

            else -> TextToSpeech.LANG_AVAILABLE
        }
    }

    public override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int =
        onIsLanguageAvailable(lang, country, variant)

    public override fun onGetLanguage(): Array<String> {
        val settings = Settings(this)
        val language = settings.readingLanguage
        val voice = VoiceCatalog.byId(settings.voiceId) ?: VoiceCatalog.default()
        val locale = SystemVoices.localeOf(voice, language)
        return arrayOf(Locale(language).isO3Language, locale.isO3Country.orEmpty(), "")
    }

    public override fun onGetVoices(): MutableList<Voice> {
        val settings = Settings(this)
        val voices = mutableListOf<Voice>()
        for (language in catalog.languages) {
            val model = settings.modelFor(language)
            // Until a language's model is downloaded nothing in it can speak.
            // Saying so in the voice's own feature set is what makes a picker
            // offer to install it rather than let someone choose a voice and
            // hear silence.
            val features =
                if (model != null && ModelManager(this, model).isModelInstalled) {
                    emptySet()
                } else {
                    setOf(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED)
                }

            // The alias, first, because it is what onGetDefaultVoiceNameFor
            // returns. The framework resolves an engine's default voice name
            // once per client and then sends that name back on every request
            // thereafter, so returning a concrete id there pinned whichever
            // voice was selected when Select to Speak or Chrome first bound -
            // and changing the voice in this app did nothing for them until
            // they were force-stopped. A name that means "whatever is selected
            // now" is resolved late instead, on each request.
            voices += Voice(
                SystemVoices.selectedName(language),
                Locale(language),
                Voice.QUALITY_VERY_HIGH,
                Voice.LATENCY_NORMAL,
                false,
                features,
            )
            // Each voice, in each language that can be read: listing 27 voices
            // for a language that cannot speak would bury the ones that can.
            if (model == null) continue
            voices += VoiceCatalog.voices.map { entry ->
                Voice(
                    SystemVoices.nameOf(entry.id, language),
                    SystemVoices.localeOf(entry, language),
                    Voice.QUALITY_VERY_HIGH,
                    // Everything runs on the phone's CPU, so latency is real
                    // but there is no network round trip.
                    Voice.LATENCY_NORMAL,
                    false,
                    features,
                )
            }
        }

        // Voices cloned from the user's own audio are not listed: this engine
        // cannot condition on a wav (PocketTts.loadVoiceFile), and offering
        // one that then reads in another voice would be a false account.
        return voices
    }

    public override fun onIsValidVoiceName(name: String?): Int =
        if (SystemVoices.parse(name, catalog.languages) != null) TextToSpeech.SUCCESS else TextToSpeech.ERROR

    public override fun onLoadVoice(name: String?): Int = onIsValidVoiceName(name)

    /**
     * The language's alias, resolved on each request; see onGetVoices. A
     * language this engine does not read gets English's, as it always has.
     */
    public override fun onGetDefaultVoiceNameFor(lang: String?, country: String?, variant: String?): String =
        SystemVoices.selectedName(SystemVoices.languageOf(lang, catalog.languages) ?: Settings.ENGLISH)

    public override fun onStop() {
        stopRequested.set(true)
        // Between frames rather than at the next decoded block, about a
        // second of audio later. The token is this request's, so an in-app
        // read sharing the engine is not stopped by it.
        current?.cancel()
    }

    /**
     * What this engine will say, given what a caller asked for.
     *
     * Deliberately not stripped as Markdown, and separated out so that
     * decision can be held in place by a test.
     *
     * Text arriving here was chosen by another app - Select to Speak, a
     * reader, a browser - and is almost always prose already. Running it
     * through the Markdown pipeline rewrote it on the way past: asterisks
     * vanished from ordinary sentences, snake_case identifiers changed shape,
     * and links were replaced by nothing at all. An accessibility tool exists
     * to tell someone what is on the screen, and quietly deleting a URL from
     * that is not a formatting choice, it is a false account of the content.
     *
     * Passing it through also keeps the chunk offsets meaningful, which is what
     * makes `rangeStart` - and so highlight-as-it-reads in Chrome and Select to
     * Speak - possible at all. The app's own screens still strip, because there
     * the text is the user's own Markdown and they asked for it to be read as
     * prose.
     */
    internal fun speakableFor(request: SynthesisRequest?): String =
        request?.charSequenceText?.toString().orEmpty().trim()

    public override fun onSynthesizeText(request: SynthesisRequest?, callback: SynthesisCallback?) {
        if (request == null || callback == null) return
        stopRequested.set(false)

        val settings = Settings(this)
        val speakable = speakableFor(request)
        if (speakable.isBlank()) {
            callback.start(SAMPLE_RATE_FALLBACK, AudioFormat.ENCODING_PCM_16BIT, 1)
            callback.done()
            return
        }

        // Read in the request's language, with the model and the voice chosen
        // for it; see readingFor.
        val reading = readingFor(request.voiceName, request.language, settings)

        // Downloading 219 MB inside a synthesis request would look like a hang
        // to whichever app asked for the speech, so fail and let the app's own
        // screens fetch the model.
        if (reading.model == null || !ModelManager(this, reading.model).isModelInstalled) {
            Log.w(TAG, "Refusing to synthesise: no model for ${reading.language} is downloaded")
            callback.error(TextToSpeech.ERROR_NOT_INSTALLED_YET)
            return
        }

        // Claims the engine, so an in-app read standing on it gives way rather
        // than the two of them taking turns sentence by sentence in two
        // different voices. See EngineTurn.
        val turn = EngineTurn.take()
        var token: NativeEngine.CancelToken? = null

        try {
            // Made before waiting for the engine, so a stop during the wait
            // still lands on this request. Inside the try: loading the native
            // library can fail, and that should be a synthesis error.
            val cancel = NativeEngine.CancelToken().also { token = it; current = it }
            runBlocking {
                // Another language than the last request's loads its model in
                // place of the last one: one at a time fits a phone's memory.
                val engine = PocketTts.get(this@PocketTtsService, reading.model)
                val voice = loadRequestedVoice(engine, request.voiceName ?: SystemVoices.SELECTED, reading.voiceId)

                // SynthesisRequest reports the rate as a percentage, where 100
                // is the user's normal speed.
                // Applied by Sonic after generation (PocketTts.synthesize).
                // Android's own slider runs 10% to 600%; past 3x speech stops
                // being words, so the top is cut there.
                val speed = (request.speechRate / 100f)
                    .coerceIn(Settings.MIN_SPEED, SYSTEM_MAX_SPEED)
                val steps = settings.decodeSteps
                // The same speaker draw the in-app reader uses - and this is
                // the path most people spend their day in, since Select to
                // Speak and reader apps all arrive here.
                val temperature = settings.temperatureFor(reading.model)
                val seed = settings.voiceSeed

                callback.start(engine.sampleRate, AudioFormat.ENCODING_PCM_16BIT, 1)
                val maxBytes = callback.maxBufferSize

                // The engine runs ahead on another thread, as in Reader.play.
                // audioAvailable blocks once the framework holds about half a
                // second of audio, and composing inside it meant the next
                // chunk could not start until this one had all but played:
                // its text prefill, longer than that half second on a phone,
                // was heard as a silence at every chunk.
                val chunks = TextChunker.chunk(speakable)
                val pieces = Channel<Piece>(Channel.UNLIMITED)
                // The chunk being delivered plus one composed ahead.
                val permits = Semaphore(2)
                coroutineScope {
                    val producer = launch(Dispatchers.Default) {
                        val self = coroutineContext.job
                        try {
                            for ((index, chunk) in chunks.withIndex()) {
                                permits.acquire()
                                if (stopRequested.get() || EngineTurn.superseded(turn)) break
                                pieces.send(Piece(index, null))
                                // The chunk's speech, not its text: the same
                                // words, with a trailing run of short sentences
                                // joined so the model does not end early on
                                // them. Offsets stay the text's.
                                val completed = engine.synthesize(
                                    chunk.speech,
                                    voice,
                                    speed,
                                    steps,
                                    temperature,
                                    seed,
                                    cancel,
                                ) { samples ->
                                    val giveUp = !self.isActive || stopRequested.get() || EngineTurn.superseded(turn)
                                    if (!giveUp) pieces.trySend(Piece(index, samples))
                                    !giveUp
                                }
                                if (!completed) break
                                if (chunk.trailingPauseSeconds > 0f) {
                                    val silence = (chunk.trailingPauseSeconds * engine.sampleRate).toInt()
                                    pieces.send(Piece(index, FloatArray(silence)))
                                }
                            }
                            pieces.close()
                        } catch (cancelled: CancellationException) {
                            pieces.close()
                            throw cancelled
                        } catch (error: Throwable) {
                            // Delivered after the audio composed before it.
                            pieces.close(error)
                        }
                    }

                    var delivering = -1
                    for (piece in pieces) {
                        if (piece.chunk != delivering) {
                            if (delivering >= 0) permits.release()
                            delivering = piece.chunk
                        }
                        val samples = piece.samples
                        if (samples == null) {
                            // Where in the request this chunk is, so the caller
                            // can highlight along. Chrome and Select to Speak
                            // both use it, and it costs one call - the offsets
                            // are only meaningful because the text above is
                            // passed through unrewritten.
                            val chunk = chunks[piece.chunk]
                            callback.rangeStart(chunk.start, chunk.end, 0)
                        } else if (!deliver(callback, samples, maxBytes)) {
                            producer.cancel()
                            break
                        }
                    }
                }
                callback.done()
            }
        } catch (error: Throwable) {
            Log.e(TAG, "Synthesis failed", error)
            callback.error(TextToSpeech.ERROR_SYNTHESIS)
        } finally {
            current = null
            token?.close()
        }
    }

    /**
     * A chunk's audio on its way to the framework; null samples mark where
     * chunk [chunk] starts.
     */
    private class Piece(val chunk: Int, val samples: FloatArray?)

    /** What a request is read in: its language, that language's model and voice. */
    internal data class Reading(val language: String, val model: String?, val voiceId: String)

    /**
     * A concrete voice is read in its own language, whatever else the request
     * says. Otherwise the request's language decides - TalkBack moving from
     * English into a French passage sends the French alias and "fra" - and a
     * language this engine does not read falls back to the app's reading
     * language, as every request did before there was more than one.
     */
    internal fun readingFor(voiceName: String?, language: String?, settings: Settings = Settings(this)): Reading {
        val named = SystemVoices.parse(voiceName, catalog.languages)
        val asked = SystemVoices.languageOf(language, catalog.languages)
        val reading = when {
            named?.voiceId != null -> named.language
            asked != null -> asked
            named != null -> named.language
            else -> settings.readingLanguage
        }
        val voiceId = named?.voiceId?.takeIf { named.language == reading } ?: settings.voiceFor(reading)
        return Reading(reading, settings.modelFor(reading), voiceId)
    }

    private suspend fun loadRequestedVoice(
        engine: PocketTts,
        requested: String,
        voiceId: String,
    ): PocketTts.LoadedVoice {
        val manager = ModelManager(this, engine.modelName)

        VoiceCatalog.byId(voiceId)?.let { stock ->
            VoiceTrace.resolved("system", requested, voiceId, manager.voiceFile(voiceId).length(), stock.bytes)
            return engine.loadVoice(stock)
        }
        // An imported voice falls through: see onGetVoices.
        val fallback = VoiceCatalog.default()
        VoiceTrace.resolved(
            caller = "system",
            requested = requested,
            resolved = fallback.id,
            promptBytes = manager.voiceFile(fallback.id).length(),
            expectedBytes = fallback.bytes,
        )
        return engine.loadVoice(fallback)
    }

    /**
     * Converts float samples to the 16-bit PCM the framework expects and feeds
     * them to [callback] in pieces no larger than it allows.
     *
     * @return false if the caller has stopped consuming audio.
     */
    private fun deliver(
        callback: SynthesisCallback,
        samples: FloatArray,
        maxBytes: Int,
    ): Boolean {
        val maxSamples = (maxBytes / 2).coerceAtLeast(1)
        var offset = 0
        while (offset < samples.size) {
            if (stopRequested.get()) return false
            val count = minOf(maxSamples, samples.size - offset)
            val bytes = ByteArray(count * 2)
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until count) {
                val clamped = samples[offset + i].coerceIn(-1f, 1f)
                buffer.putShort((clamped * 32767f).toInt().toShort())
            }
            if (callback.audioAvailable(bytes, 0, bytes.size) != TextToSpeech.SUCCESS) {
                return false
            }
            offset += count
        }
        return true
    }

    private companion object {
        const val TAG = "PocketTtsService"

        const val SAMPLE_RATE_FALLBACK = 24000
        const val SYSTEM_MAX_SPEED = 3.0f
        val supportedCountries = listOf("USA", "GBR")
    }
}
