package org.pockettts.android.player

import android.content.Context
import org.pockettts.android.engine.ModelManager
import org.pockettts.android.engine.PocketTts
import org.pockettts.android.engine.Settings
import org.pockettts.android.engine.VoiceCatalog
import org.pockettts.android.debug.VoiceTrace

/** [SpeechEngine] backed by the real model. */
class PocketTtsEngine(
    private val context: Context,
    private val tts: PocketTts,
) : SpeechEngine {

    private val settings = Settings(context)

    private var voice: PocketTts.LoadedVoice? = null

    override val sampleRate: Int get() = tts.sampleRate

    /**
     * Who asked for the current read, for the trace.
     *
     * Held rather than passed down because [synthesize] can resolve a voice on
     * its own when nothing selected one, and a line in the trace attributed to
     * nobody is the line that starts the next wrong search.
     */
    private var caller: String = "reader"

    override suspend fun useVoice(voiceId: String, caller: String) {
        this.caller = caller
        voice = resolve(voiceId)
    }

    override suspend fun synthesize(
        text: String,
        speed: Float,
        onAudio: (FloatArray) -> Boolean,
    ): Boolean {
        val loaded = voice ?: resolve(VoiceCatalog.DEFAULT_VOICE_ID).also { voice = it }
        // Read per call rather than held, so moving a slider changes the next
        // chunk rather than the next read.
        return tts.synthesize(
            text,
            loaded,
            speed,
            settings.decodeSteps,
            settings.temperatureFor(tts.modelName),
            settings.voiceSeed,
            onAudio = onAudio,
        )
    }

    /**
     * Finds the prompt for [voiceId], falling back to the default voice.
     *
     * The fallback is silent by necessity - there is nothing sensible to do
     * mid-read about a voice that has gone missing - but silent is how "it
     * used the wrong voice" gets to be a mystery, so it is recorded.
     */
    private suspend fun resolve(voiceId: String): PocketTts.LoadedVoice {
        val manager = ModelManager(context)
        VoiceCatalog.byId(voiceId)?.let { stock ->
            // Recorded in a finally, after the load rather than before it.
            // loadVoice is what fetches the prompt, so asking the file its size
            // first reported "prompt file missing" for every first read after an
            // install - on a read that then went on to work perfectly. That is
            // the one line whose whole job is to make a genuinely missing prompt
            // unmistakable, and it was crying wolf. A finally rather than an
            // ordinary statement because a load that throws still has to leave
            // the breadcrumb, and in that case the file really is missing.
            try {
                return tts.loadVoice(stock)
            } finally {
                VoiceTrace.resolved(
                    caller = caller,
                    requested = voiceId,
                    resolved = voiceId,
                    promptBytes = manager.voiceFile(voiceId).length(),
                    expectedBytes = stock.bytes,
                )
            }
        }
        // Not a stock voice, so it is one the user imported. This engine
        // cannot clone (PocketTts.loadVoiceFile), so it reads in the default
        // voice, and the trace below says which was asked for.
        val fallback = VoiceCatalog.default()
        try {
            return tts.loadVoice(fallback)
        } finally {
            VoiceTrace.resolved(
                caller = caller,
                requested = voiceId,
                resolved = fallback.id,
                promptBytes = manager.voiceFile(fallback.id).length(),
                expectedBytes = fallback.bytes,
            )
        }
    }

    companion object : SpeechEngine.Factory {
        override suspend fun create(
            context: Context,
            progress: (Float) -> Unit,
        ): SpeechEngine = PocketTtsEngine(context, PocketTts.get(context, progress = progress))
    }
}
