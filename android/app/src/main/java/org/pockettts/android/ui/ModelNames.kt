package org.pockettts.android.ui

import android.content.Context
import org.pockettts.android.R
import org.pockettts.android.engine.ModelCatalog
import org.pockettts.android.engine.Settings
import java.util.Locale

/**
 * What a person calls each model. The catalog's names are upstream's config
 * names, which say a version and a parameter count, not what to choose.
 *
 * English comes in sizes, named for what each is good at; every other language
 * has one model, named for its language. An English model the app has no
 * words for keeps its config name, so a model added to `models.json` shows up
 * before anyone writes a description for it.
 */
object ModelNames {

    private val ENGLISH_SIZES = mapOf(
        "english_2026-10" to (R.string.model_best to R.string.model_best_detail),
        "english_18m_2026-10" to (R.string.model_balanced to R.string.model_balanced_detail),
        "english_7m_2026-10" to (R.string.model_small to R.string.model_small_detail),
        "english_2m_2026-10" to (R.string.model_smallest to R.string.model_smallest_detail),
    )

    /** "Best quality" among English's; "French" for French's one. */
    fun title(context: Context, model: String): String {
        ENGLISH_SIZES[model]?.let { return context.getString(it.first) }
        val language = ModelCatalog.get(context).byName(model)?.language
        return if (language != null && language != Settings.ENGLISH) language(language) else model
    }

    fun detail(context: Context, model: String): String {
        ENGLISH_SIZES[model]?.let { return context.getString(it.second) }
        val language = ModelCatalog.get(context).byName(model)?.language ?: return ""
        if (language == Settings.ENGLISH) return ""
        return context.getString(R.string.model_language_detail, language(language))
    }

    /** "English, Best quality"; "French". For where the language is not already said. */
    fun full(context: Context, model: String): String {
        val language = ModelCatalog.get(context).byName(model)?.language
        return if (language == Settings.ENGLISH) {
            context.getString(R.string.model_full_name, language(Settings.ENGLISH), title(context, model))
        } else {
            title(context, model)
        }
    }

    /** A language's name in the phone's own language, capitalised: "French", "Französisch". */
    fun language(tag: String): String {
        val locale = Locale.getDefault()
        return Locale.forLanguageTag(tag).getDisplayLanguage(locale)
            .replaceFirstChar { if (it.isLowerCase()) it.titlecase(locale) else it.toString() }
    }

    /** Whether the app has words for [model] beyond its config name, for its tests. */
    internal fun hasWords(context: Context, model: String): Boolean =
        model in ENGLISH_SIZES || ModelCatalog.get(context).byName(model)?.language.let {
            it != null && it != Settings.ENGLISH
        }
}
