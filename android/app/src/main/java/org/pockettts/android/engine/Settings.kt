package org.pockettts.android.engine

import android.content.Context
import androidx.core.content.edit

/** User preferences, shared between the UI, the reader and the system engine. */
class Settings(context: Context) {

    private val app = context.applicationContext

    private val prefs = app.getSharedPreferences("pocket-tts", Context.MODE_PRIVATE)

    private val catalog = ModelCatalog.get(context)

    /**
     * The model in-app reads are made with, by name in [ModelCatalog]; its
     * language is the [readingLanguage]. A name that is no longer offered
     * reads as the default, so an update that drops a model falls back rather
     * than failing every read.
     */
    var modelName: String
        get() = prefs.getString(KEY_MODEL, null)?.takeIf { catalog.byName(it) != null } ?: DEFAULT_MODEL
        set(value) = prefs.edit {
            putString(KEY_MODEL, value)
            // Remembered per language too, for the system engine: English
            // read at the size last chosen for it while French is selected.
            catalog.byName(value)?.let { putString(modelKey(it.language), value) }
        }

    /**
     * The language in-app reads are in, as a BCP-47 tag: the selected model's.
     * The share sheet, a selection and the scratchpad carry no language of
     * their own, so this is the one they are read in.
     */
    val readingLanguage: String
        get() = catalog.byName(modelName)?.language ?: ENGLISH

    /**
     * The model a read in [language] is made with, or null when none is
     * installed: the selected model when it reads [language], whether or not
     * it is installed yet, else the one last chosen for [language], else the
     * first of its models on the device.
     */
    fun modelFor(language: String): String? {
        val selected = modelName
        if (catalog.byName(selected)?.language == language) return selected
        val remembered = prefs.getString(modelKey(language), null)
            ?.takeIf { catalog.byName(it)?.language == language }
        return (listOfNotNull(remembered) + catalog.modelsFor(language).map { it.name })
            .firstOrNull { ModelManager(app, it).isModelInstalled }
    }

    /** The voice in-app reads use: the [readingLanguage]'s. */
    var voiceId: String
        get() = voiceFor(readingLanguage)
        set(value) = setVoiceFor(readingLanguage, value)

    /**
     * The voice [language] is read in: the one chosen for it, else Kyutai's
     * for it - Estelle for French, Alba for English. Each language keeps its
     * own, so choosing a French voice leaves English as it was.
     */
    fun voiceFor(language: String): String =
        prefs.getString(voiceKey(language), null) ?: defaultVoiceFor(language)

    fun setVoiceFor(language: String, id: String) = prefs.edit { putString(voiceKey(language), id) }

    fun defaultVoiceFor(language: String): String =
        catalog.modelsFor(language).firstOrNull()?.defaultVoice ?: VoiceCatalog.DEFAULT_VOICE_ID

    /** Playback rate multiplier; 1.0 is the model's natural pace. */
    var speed: Float
        get() = prefs.getFloat(KEY_SPEED, 1.0f).coerceIn(MIN_SPEED, MAX_SPEED)
        set(value) = prefs.edit { putFloat(KEY_SPEED, value.coerceIn(MIN_SPEED, MAX_SPEED)) }

    /**
     * Pocket TTS is tuned for two cores and scales poorly past that, but
     * thread-limited phones sometimes do better with more. Read when the
     * engine first loads: a change takes effect when the app next starts.
     */
    var numThreads: Int
        get() = prefs.getInt(KEY_THREADS, DEFAULT_THREADS).coerceIn(1, 8)
        set(value) = prefs.edit { putInt(KEY_THREADS, value.coerceIn(1, 8)) }

    /**
     * Flow-decoding steps per audio frame.
     *
     * Each generated frame is produced by integrating a flow, and this is how
     * many Euler steps that integration takes. sherpa-onnx defaults to 5. The
     * reference implementation defaults to 1 - both in `lsd_decode` itself and
     * in `default_parameters.py` - so four fifths of that work may be buying
     * nothing at all, on a phone, per frame.
     *
     * The device has now said. At one step a listener reported the reading
     * closer to what the same model does on a desktop - "maybe a little more
     * expressive, or more inflection" - and no worse in any other respect, so
     * the default follows the reference rather than sherpa-onnx.
     *
     * It should also be cheaper: one Euler step per frame is one evaluation of
     * the flow graph rather than five. That prediction sits oddly beside the
     * same report saying it was "a tad slower to first word", which is either
     * run-to-run variance - chunk timings on that device move by ~90 ms between
     * identical runs - or something not yet understood. The `first=` figure in
     * the voice trace measures exactly this and can settle it.
     */
    var decodeSteps: Int
        get() = prefs.getInt(KEY_DECODE_STEPS, DEFAULT_DECODE_STEPS).coerceIn(MIN_STEPS, MAX_STEPS)
        set(value) = prefs.edit { putInt(KEY_DECODE_STEPS, value.coerceIn(MIN_STEPS, MAX_STEPS)) }

    /** Whether fenced code blocks in Markdown are read out. */
    var speakCodeBlocks: Boolean
        get() = prefs.getBoolean(KEY_CODE_BLOCKS, false)
        set(value) = prefs.edit { putBoolean(KEY_CODE_BLOCKS, value) }

    /**
     * Whether every sentence draws the same speaker.
     *
     * Pocket TTS has no speaker table: it is prompted with a few seconds of
     * reference audio and *samples* a speaker in that audio's neighbourhood.
     * The draw is a vector of Gaussian noise, and sherpa-onnx seeds it from a
     * random device on every sentence - not every read, every sentence, since
     * it re-splits whatever it is given on sentence-ending punctuation and
     * starts each one from a fresh generator. So a paragraph is read by a
     * succession of slightly different people, which is what "the voice keeps
     * changing" is.
     *
     * Pinning the seed makes that noise identical for every sentence, so the
     * same prompt yields the same speaker throughout - and, because the seed
     * is per-generation rather than per-anything-of-ours, on the system engine
     * as well as in the app.
     *
     * Off means a fresh draw each sentence, which is sherpa's own default and
     * worth keeping reachable: a fixed seed also fixes the reading, so the
     * same text always comes out identically, and someone re-reading a
     * paragraph to hear it differently wants the variation back.
     */
    var steadyVoice: Boolean
        get() = prefs.getBoolean(KEY_STEADY_VOICE, true)
        set(value) = prefs.edit { putBoolean(KEY_STEADY_VOICE, value) }

    /** The seed to hand the generator, or -1 for a fresh draw each sentence. */
    val voiceSeed: Int get() = if (steadyVoice) FIXED_SEED else RANDOM_SEED

    /**
     * How far from the voice prompt a speaker may be drawn, when set here
     * rather than left to the model; null leaves it to the model.
     *
     * The generator's noise has standard deviation sqrt(temperature), so this
     * sets the width of the neighbourhood the speaker is sampled from. Higher
     * is more varied and less like the prompt.
     *
     * Each model's config carries Kyutai's own value (`default_temperature`,
     * in models.json), 0.3 for every model offered today: "Human evaluations
     * consistently prefer the English model at temperature 0.3 over the
     * current default 0.7". Leaving it to the model means a later model, or a
     * later change of Kyutai's mind, reaches this app without anyone here
     * having to notice. Moving the slider sets a value of one's own, for
     * every model, until it is reset.
     */
    var temperatureOverride: Float?
        get() {
            migrateTemperature()
            return if (prefs.contains(KEY_TEMPERATURE)) {
                prefs.getFloat(KEY_TEMPERATURE, FALLBACK_TEMPERATURE).coerceIn(MIN_TEMPERATURE, MAX_TEMPERATURE)
            } else {
                null
            }
        }
        set(value) = prefs.edit {
            if (value == null) {
                remove(KEY_TEMPERATURE)
            } else {
                putFloat(KEY_TEMPERATURE, value.coerceIn(MIN_TEMPERATURE, MAX_TEMPERATURE))
            }
        }

    /** The temperature [model] reads at: the override, else the model's own. */
    fun temperatureFor(model: String): Float =
        temperatureOverride ?: defaultTemperatureFor(model)

    /** Kyutai's temperature for [model]. */
    fun defaultTemperatureFor(model: String): Float =
        (catalog.byName(model)?.defaultTemperature ?: FALLBACK_TEMPERATURE)
            .coerceIn(MIN_TEMPERATURE, MAX_TEMPERATURE)

    /**
     * Builds before the per-model default stored the slider's value whenever it
     * moved, and 0.3 was both the old default and every model's own today, so
     * a stored 0.3 is taken to mean "the default" and dropped, once. Any other
     * value was chosen and stays.
     */
    private fun migrateTemperature() {
        if (prefs.getBoolean(KEY_TEMPERATURE_MIGRATED, false)) return
        prefs.edit {
            val stored = prefs.getFloat(KEY_TEMPERATURE, Float.NaN)
            if (kotlin.math.abs(stored - LEGACY_DEFAULT_TEMPERATURE) < 0.005f) remove(KEY_TEMPERATURE)
            putBoolean(KEY_TEMPERATURE_MIGRATED, true)
        }
    }

    /** Whether text arriving from other apps is treated as Markdown. */
    var treatSelectionAsMarkdown: Boolean
        get() = prefs.getBoolean(KEY_SELECTION_MARKDOWN, true)
        set(value) = prefs.edit { putBoolean(KEY_SELECTION_MARKDOWN, value) }

    /**
     * Glass panel appearance. Exposed as settings rather than constants so the
     * values can be dialled in on a real device - the effect depends on the
     * wallpaper, the theme and whether the device supports blur at all, none of
     * which can be judged from source.
     */
    var glassAlpha: Float
        get() = prefs.getFloat(KEY_GLASS_ALPHA, DEFAULT_GLASS_ALPHA).coerceIn(0f, 1f)
        set(value) = prefs.edit { putFloat(KEY_GLASS_ALPHA, value.coerceIn(0f, 1f)) }

    var glassBlurDp: Float
        get() = prefs.getFloat(KEY_GLASS_BLUR, DEFAULT_GLASS_BLUR_DP).coerceIn(0f, MAX_BLUR_DP)
        set(value) = prefs.edit { putFloat(KEY_GLASS_BLUR, value.coerceIn(0f, MAX_BLUR_DP)) }

    var glassDim: Float
        get() = prefs.getFloat(KEY_GLASS_DIM, DEFAULT_GLASS_DIM).coerceIn(0f, 1f)
        set(value) = prefs.edit { putFloat(KEY_GLASS_DIM, value.coerceIn(0f, 1f)) }

    var glassCornerDp: Float
        get() = prefs.getFloat(KEY_GLASS_CORNER, DEFAULT_GLASS_CORNER_DP).coerceIn(0f, MAX_CORNER_DP)
        set(value) = prefs.edit { putFloat(KEY_GLASS_CORNER, value.coerceIn(0f, MAX_CORNER_DP)) }

    fun resetGlass() {
        glassAlpha = DEFAULT_GLASS_ALPHA
        glassBlurDp = DEFAULT_GLASS_BLUR_DP
        glassDim = DEFAULT_GLASS_DIM
        glassCornerDp = DEFAULT_GLASS_CORNER_DP
    }

    var scratchpad: String
        get() = prefs.getString(KEY_SCRATCHPAD, "") ?: ""
        set(value) = prefs.edit { putString(KEY_SCRATCHPAD, value) }

    fun voice(): VoiceCatalog.Voice = VoiceCatalog.byId(voiceId) ?: VoiceCatalog.default()

    private fun modelKey(language: String) = "${KEY_MODEL}_$language"

    /** English keeps the key every earlier build stored its one voice under. */
    private fun voiceKey(language: String) = if (language == ENGLISH) KEY_VOICE else "${KEY_VOICE}_$language"

    companion object {
        /**
         * The largest model, Kyutai's default: the best reading, and on the
         * engine this app now runs, faster on the same phone than the smaller
         * ONNX model it replaces. The small models are offered for phones
         * where it is not fast enough, and to save space.
         */
        const val DEFAULT_MODEL = "english_2026-10"

        const val MIN_SPEED = 0.5f
        const val MAX_SPEED = 2.0f
        private const val DEFAULT_THREADS = 2

        const val MIN_STEPS = 1
        const val MAX_STEPS = 8

        /**
         * The reference implementation's value, not sherpa-onnx's five.
         *
         * Changed on a device report rather than reasoning; see [decodeSteps].
         */
        const val DEFAULT_DECODE_STEPS = 1

        const val MIN_TEMPERATURE = 0.1f
        const val MAX_TEMPERATURE = 1.0f

        /** Upstream's value for a config that does not give one. */
        const val FALLBACK_TEMPERATURE = 0.3f

        /** What every build before per-model defaults stored as its default. */
        private const val LEGACY_DEFAULT_TEMPERATURE = 0.3f

        const val ENGLISH = "en"

        /**
         * Any value at all, as long as it never changes. It is not a quality
         * of the voice - it selects one draw out of the prompt's neighbourhood,
         * and all that matters is that every sentence selects the same one.
         */
        const val FIXED_SEED = 1

        /** sherpa-onnx's "seed from a random device", which it takes as -1. */
        const val RANDOM_SEED = -1

        const val MAX_BLUR_DP = 80f
        const val MAX_CORNER_DP = 48f

        /**
         * Dialled in on a device rather than guessed here, which is what the
         * appearance screen exists for. These are the values that came back.
         *
         * The alpha is low because the panel it governs sits over the app's own
         * content, where the blur behind it is real - it is the scratchpad
         * overlay that uses these, not the sheet over another app, which is
         * opaque by design because there is nothing behind it to frost.
         *
         * The alpha also has to land on the slider's 0.01 step grid: Material's
         * Slider throws during layout on a value that misses it, which took the
         * appearance screen down on open once already.
         */
        const val DEFAULT_GLASS_ALPHA = 0.37f
        const val DEFAULT_GLASS_BLUR_DP = 51f
        const val DEFAULT_GLASS_DIM = 0f
        const val DEFAULT_GLASS_CORNER_DP = 10f

        private const val KEY_MODEL = "model"
        private const val KEY_VOICE = "voice"
        private const val KEY_SPEED = "speed"
        private const val KEY_THREADS = "threads"
        private const val KEY_DECODE_STEPS = "decode_steps"
        private const val KEY_CODE_BLOCKS = "speak_code_blocks"
        private const val KEY_SELECTION_MARKDOWN = "selection_markdown"
        private const val KEY_STEADY_VOICE = "steady_voice"
        private const val KEY_TEMPERATURE = "temperature"
        private const val KEY_TEMPERATURE_MIGRATED = "temperature_follows_model"
        private const val KEY_SCRATCHPAD = "scratchpad"
        private const val KEY_GLASS_ALPHA = "glass_alpha"
        private const val KEY_GLASS_BLUR = "glass_blur_dp"
        private const val KEY_GLASS_DIM = "glass_dim"
        private const val KEY_GLASS_CORNER = "glass_corner_dp"
    }
}
