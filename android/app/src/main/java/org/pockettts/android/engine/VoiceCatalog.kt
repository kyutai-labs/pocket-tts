package org.pockettts.android.engine

/**
 * The stock Pocket TTS voices.
 *
 * Pocket TTS has no fixed speaker table: a "voice" is just a few seconds of
 * reference audio that the model conditions on. So a voice here is a URL to a
 * wav, and adding your own is the same operation as picking a stock one.
 *
 * Paths are relative to the `kyutai/tts-voices` repository on Hugging Face,
 * whose per-voice licences are listed at
 * https://huggingface.co/kyutai/tts-voices.
 */
object VoiceCatalog {

    private const val BASE = "https://huggingface.co/kyutai/tts-voices/resolve/main/"

    data class Voice(
        val id: String,
        val displayName: String,
        /**
         * The speaker's own language and accent, as a BCP-47 tag. Every voice
         * reads every language the app offers; this is what it sounds like
         * doing so, and which language it is native to.
         */
        val language: String,
        /**
         * The reference recording, for the picker's preview; null where it is
         * not public (Kyutai's recordings of the native voices of Italian,
         * Spanish, German, Portuguese and Dutch need a Hugging Face login).
         * Those are previewed by reading a sentence in them instead.
         */
        val path: String?,
        /**
         * The prompt's exact size on Hugging Face.
         *
         * Here because a cached prompt is otherwise unidentifiable. An import
         * named like a stock voice used to overwrite that voice's file, and
         * nothing ever fetched it again - so the app would read every "Alba"
         * from that day on in a voice that was not Alba, with no error and no
         * way to tell from inside the app. A size is the cheapest check that
         * catches it, and the CDN serves an exact content-length.
         */
        val bytes: Long,
        // What the engine is conditioned on is not this wav, which is only the
        // preview the picker plays, but the voice's state for the selected
        // model: ModelCatalog pins one per model.
    ) {
        val url: String? get() = path?.let { BASE + it }

        /** The primary language subtag of [language]: "en", "fr", ... */
        val nativeLanguage: String get() = language.substringBefore('-')
        /** Stable filename for the cached wav. */
        val fileName: String get() = "$id.wav"
    }

    val voices: List<Voice> = listOf(
        Voice("alba", "Alba", "en-GB", "alba-mackenna/casual.wav", 958542),
        Voice("anna", "Anna", "en-GB", "vctk/p228_023_enhanced.wav", 804630),
        Voice("azelma", "Azelma", "en-GB", "vctk/p303_023_enhanced.wav", 823852),
        Voice("bill_boerst", "Bill Boerst", "en-US", "voice-zero/bill_boerst.wav", 955496),
        Voice("caro_davy", "Caro Davy", "en-US", "voice-zero/caro_davy.wav", 743528),
        Voice("charles", "Charles", "en-GB", "vctk/p254_023_enhanced.wav", 639272),
        Voice(
            "cosette", "Cosette", "en-US",
            "expresso/ex04-ex02_confused_001_channel1_499s.wav", 960044,
        ),
        Voice("eponine", "Eponine", "en-GB", "vctk/p262_023_enhanced.wav", 716330),
        Voice("eve", "Eve", "en-GB", "vctk/p361_023_enhanced.wav", 671872),
        Voice("fantine", "Fantine", "en-GB", "vctk/p244_023_enhanced.wav", 674852),
        Voice("george", "George", "en-GB", "vctk/p315_023_enhanced.wav", 642692),
        Voice("jane", "Jane", "en-GB", "vctk/p339_023_enhanced.wav", 759340),
        Voice("javert", "Javert", "en-US", "voice-donations/Butter.wav", 480044),
        Voice("jean", "Jean", "en-US", "ears/p010/freeform_speech_01_enhanced.wav", 640044),
        Voice("marius", "Marius", "en-US", "voice-donations/Selfie.wav", 480044),
        Voice("mary", "Mary", "en-GB", "vctk/p333_023_enhanced.wav", 639084),
        Voice("michael", "Michael", "en-GB", "vctk/p360_023_enhanced.wav", 751140),
        Voice("paul", "Paul", "en-GB", "vctk/p259_023_enhanced.wav", 717182),
        Voice("peter_yearsley", "Peter Yearsley", "en-US", "voice-zero/peter_yearsley.wav", 524448),
        Voice("stuart_bell", "Stuart Bell", "en-US", "voice-zero/stuart_bell.wav", 745776),
        Voice("vera", "Vera", "en-GB", "vctk/p229_023_enhanced.wav", 691416),
        // Kyutai's voices for the other languages, each the default for its
        // own (upstream's DEFAULT_VOICE_FOR_LANGUAGE, carried in models.json).
        Voice("estelle", "Estelle", "fr", "unmute-prod-website/developpeuse-3.wav", 1282324),
        Voice("juergen", "Jürgen", "de-DE", null, 0),
        Voice("lola", "Lola", "es", null, 0),
        Voice("giovanni", "Giovanni", "it", null, 0),
        Voice("rafael", "Rafael", "pt", null, 0),
        Voice("daan", "Daan", "nl", null, 0),
    )

    /** English's default, and the fallback for a voice that is gone. */
    const val DEFAULT_VOICE_ID = "alba"

    fun byId(id: String?): Voice? = voices.firstOrNull { it.id == id }

    fun default(): Voice = byId(DEFAULT_VOICE_ID) ?: voices.first()
}
