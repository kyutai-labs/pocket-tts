package org.pockettts.android.engine

import java.util.Locale
import java.util.MissingResourceException

/**
 * The names this app's voices go by in the system's voice list, and the
 * language each one reads.
 *
 * Every voice reads every language, so a voice in that list is a voice *in a
 * language*. English keeps the names earlier builds gave it - `selected`, and
 * each voice's bare id - because apps remember a voice by name and would
 * otherwise lose theirs on update. Every other language suffixes its tag:
 * `selected-fr`, `alba-fr`. Voice ids never contain a hyphen, so the two
 * cannot be confused.
 */
object SystemVoices {

    /**
     * A name that means "whatever voice is chosen for this language in the
     * app right now", resolved on each request rather than when a client bound.
     */
    const val SELECTED = "selected"

    /** A voice by name: a stock voice's id, or null for the selected one. */
    data class Named(val voiceId: String?, val language: String)

    fun selectedName(language: String): String =
        if (language == Settings.ENGLISH) SELECTED else "$SELECTED-$language"

    fun nameOf(voiceId: String, language: String): String =
        if (language == Settings.ENGLISH) voiceId else "$voiceId-$language"

    /** What [name] stands for, or null if it is not one of [languages]' names. */
    fun parse(name: String?, languages: Collection<String>): Named? {
        if (name.isNullOrEmpty()) return null
        val suffix = name.substringAfterLast('-', missingDelimiterValue = "")
        val (id, language) = if (suffix in languages && suffix != Settings.ENGLISH) {
            name.substringBeforeLast('-') to suffix
        } else {
            name to Settings.ENGLISH
        }
        if (language !in languages) return null
        return when {
            id == SELECTED -> Named(null, language)
            VoiceCatalog.byId(id) != null -> Named(id, language)
            else -> null
        }
    }

    /**
     * The offered language a request's ISO 639 code names (the framework
     * sends three letters, "fra"; some callers send two), or null.
     */
    fun languageOf(code: String?, languages: Collection<String>): String? {
        if (code.isNullOrEmpty()) return null
        return languages.firstOrNull { tag ->
            code.equals(tag, ignoreCase = true) || iso3(tag).any { code.equals(it, ignoreCase = true) }
        }
    }

    /**
     * The locale a voice reading [language] is listed under: the speaker's own
     * accent when it is native to [language] (Alba is en-GB), else the bare
     * language.
     */
    fun localeOf(voice: VoiceCatalog.Voice, language: String): Locale =
        if (voice.nativeLanguage == language) Locale.forLanguageTag(voice.language) else Locale(language)

    /**
     * Three-letter codes for a two-letter tag. German and Dutch have two each,
     * and Java knows only the terminological one ("deu", "nld"), where some
     * callers send the bibliographic ("ger", "dut").
     */
    private fun iso3(tag: String): List<String> {
        val terminological = try {
            Locale(tag).isO3Language
        } catch (missing: MissingResourceException) {
            ""
        }
        return listOf(terminological) + (BIBLIOGRAPHIC[tag]?.let(::listOf) ?: emptyList())
    }

    private val BIBLIOGRAPHIC = mapOf("de" to "ger", "nl" to "dut", "fr" to "fre")
}
