package org.pockettts.android.engine

import android.content.Context
import org.json.JSONObject

/**
 * The models on offer and every file each one needs, pinned by sha256.
 *
 * This is `pocket-speak/models.json`, the file the engine itself compiles in,
 * packaged as an asset by Gradle: one list, so the app downloads exactly what
 * the engine reads. It is written by `pocket-speak/tools/pin-models.py`, and
 * the engine's tests fail when upstream moves a model's files without it being
 * rerun.
 */
class ModelCatalog private constructor(val models: List<Model>) {

    data class File(val name: String, val url: String, val sha256: String, val bytes: Long)

    data class Model(
        /** The config's name in upstream's `pocket_tts/config`. */
        val name: String,
        val files: List<File>,
        /** Precomputed voice states, by voice id. */
        val voices: Map<String, File>,
    ) {
        /** What installing it downloads, voices aside (each is 0.5-7 MB). */
        val downloadBytes: Long get() = files.sumOf { it.bytes }
    }

    fun byName(name: String): Model? = models.firstOrNull { it.name == name }

    companion object {
        const val ASSET = "models.json"

        @Volatile
        private var loaded: ModelCatalog? = null

        fun get(context: Context): ModelCatalog =
            loaded ?: synchronized(this) {
                loaded ?: context.applicationContext.assets.open(ASSET)
                    .bufferedReader().use { parse(it.readText()) }
                    .also { loaded = it }
            }

        fun parse(json: String): ModelCatalog {
            val models = JSONObject(json).getJSONArray("models")
            return ModelCatalog(
                (0 until models.length()).map { i ->
                    val model = models.getJSONObject(i)
                    val files = model.getJSONArray("files")
                    val voices = model.getJSONObject("voices")
                    Model(
                        name = model.getString("name"),
                        files = (0 until files.length()).map { j ->
                            val file = files.getJSONObject(j)
                            File(
                                name = file.getString("name"),
                                url = file.getString("url"),
                                sha256 = file.getString("sha256"),
                                bytes = file.getLong("size"),
                            )
                        },
                        voices = voices.keys().asSequence().associateWith { id ->
                            val voice = voices.getJSONObject(id)
                            File(
                                name = "$id.safetensors",
                                url = voice.getString("url"),
                                sha256 = voice.getString("sha256"),
                                bytes = voice.getLong("size"),
                            )
                        },
                    )
                },
            )
        }
    }
}
