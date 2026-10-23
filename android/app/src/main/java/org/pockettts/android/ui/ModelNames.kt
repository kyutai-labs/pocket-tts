package org.pockettts.android.ui

import android.content.Context
import org.pockettts.android.R

/**
 * What a person calls each model. The catalog's names are upstream's config
 * names, which say a version and a parameter count, not what to choose.
 *
 * A model the app has no words for keeps its config name, so a model added to
 * `models.json` shows up before anyone writes a description for it.
 */
object ModelNames {

    private val NAMES = mapOf(
        "english_2026-10" to (R.string.model_best to R.string.model_best_detail),
        "english_18m_2026-10" to (R.string.model_balanced to R.string.model_balanced_detail),
        "english_7m_2026-10" to (R.string.model_small to R.string.model_small_detail),
        "english_2m_2026-10" to (R.string.model_smallest to R.string.model_smallest_detail),
    )

    fun title(context: Context, model: String): String =
        NAMES[model]?.let { context.getString(it.first) } ?: model

    fun detail(context: Context, model: String): String =
        NAMES[model]?.let { context.getString(it.second) } ?: ""

    /** The models the app has words for, for its tests. */
    internal val known: Set<String> get() = NAMES.keys
}
