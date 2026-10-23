package org.pockettts.android.engine

import android.content.Context
import android.os.Build
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

/**
 * The catalog is pocket-speak/models.json, packaged by Gradle. These hold the
 * app's side of it: what the engine's own tests cannot see.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [Build.VERSION_CODES.UPSIDE_DOWN_CAKE])
class ModelCatalogTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val catalog = ModelCatalog.get(context)

    @Test
    fun `the default model is on offer`() {
        assertNotNull(catalog.byName(Settings.DEFAULT_MODEL))
    }

    @Test
    fun `every voice the app offers exists for every model`() {
        // Switching models keeps the selected voice; a voice one model lacks
        // would fail every read the moment it was chosen.
        for (model in catalog.models) {
            val missing = VoiceCatalog.voices.map { it.id }.filterNot { it in model.voices }
            assertTrue("${model.name} has no state for $missing", missing.isEmpty())
        }
    }

    @Test
    fun `every file is pinned and named uniquely`() {
        for (model in catalog.models) {
            assertEquals(model.files.size, model.files.map { it.name }.toSet().size)
            for (file in model.files + model.voices.values) {
                assertTrue("${model.name}/${file.name}", Regex("[0-9a-f]{64}").matches(file.sha256))
                assertTrue("${model.name}/${file.name}", file.url.startsWith("https://huggingface.co/"))
                assertTrue("${model.name}/${file.name}", file.bytes > 0)
            }
        }
    }

    @Test
    fun `every model on offer has a name a person would choose by`() {
        // A model without one still shows, under its config name; this says
        // when the catalog gains a model the picker has no words for.
        assertEquals(catalog.models.map { it.name }.toSet(), org.pockettts.android.ui.ModelNames.known)
    }

    @Test
    fun `an unknown stored model reads as the default`() {
        val settings = Settings(context)
        settings.modelName = "a-model-a-later-build-dropped"
        assertEquals(Settings.DEFAULT_MODEL, settings.modelName)
        settings.modelName = "english_7m_2026-10"
        assertEquals("english_7m_2026-10", settings.modelName)
    }

    @Test
    fun `the retired ONNX bundle is deleted, and nothing else at its path`() {
        val root = File(context.filesDir, "pocket-tts")
        val onnx = File(root, "english_2026-04").apply { mkdirs() }
        File(onnx, "bundle.json").writeText("{}")
        val manager = ModelManager(context, Settings.DEFAULT_MODEL)
        manager.removeRetiredModels()
        assertFalse(onnx.exists())

        // Not the ONNX bundle: no bundle.json, so not this cleanup's to delete.
        onnx.mkdirs()
        File(onnx, "notes.txt").writeText("mine")
        manager.removeRetiredModels()
        assertTrue(File(onnx, "notes.txt").exists())
        onnx.deleteRecursively()
    }
}
