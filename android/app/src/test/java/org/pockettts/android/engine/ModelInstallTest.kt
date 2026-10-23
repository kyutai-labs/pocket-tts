package org.pockettts.android.engine

import android.content.Context
import android.os.Build
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.util.concurrent.atomic.AtomicInteger

/**
 * One install, however many things ask for it.
 *
 * Two callers used to reach the downloader independently - the button, and any
 * read that arrived before the model existed - sharing a partial-download file
 * and a staging directory without either knowing the other was there. Tapping
 * Download and then selecting text in another app had them writing the same
 * `.part` and deleting each other's unpacked files.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [Build.VERSION_CODES.UPSIDE_DOWN_CAKE])
class ModelInstallTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val downloads = AtomicInteger(0)

    private fun fakeFiles(modelName: String): ModelManager.ModelFiles {
        val dir = File(context.filesDir, "fake/$modelName").apply { mkdirs() }
        ModelManager(context, modelName).model.files.forEach { File(dir, it.name).writeText("x") }
        return ModelManager.ModelFiles(dir, modelName)
    }

    @Before
    fun setUp() {
        ModelInstall.resetForTesting()
        downloads.set(0)
        // Installs slowly, and without a network.
        ModelInstall.installer = ModelInstall.Installer { _, modelName, progress ->
            downloads.incrementAndGet()
            progress?.onProgress(0.5f)
            delay(200)
            fakeFiles(modelName)
        }
    }

    @After
    fun tearDown() = ModelInstall.resetForTesting()

    @Test
    fun `a model downloaded from the picker is used once it is in`() = runBlocking {
        val settings = Settings(context)
        settings.modelName = "english_2026-10"
        ModelInstall.installAndSelect(context, "english_2m_2026-10")
        // Not before: reads would wait on a download nobody may finish.
        assertEquals("english_2026-10", settings.modelName)
        withTimeout(5_000) { while (settings.modelName != "english_2m_2026-10") delay(10) }
        assertEquals(1, downloads.get())
    }

    @Test
    fun `two models never share one download`() = runBlocking {
        // A read for one model joining the download of another would load the
        // wrong files, or wait on a download it did not need.
        val first = async { ModelInstall.ensure(context, "english_7m_2026-10") }
        val second = async { ModelInstall.ensure(context, "english_2m_2026-10") }
        assertEquals("english_7m_2026-10", first.await().modelName)
        assertEquals("english_2m_2026-10", second.await().modelName)
        assertEquals(2, downloads.get())
    }

    @Test
    fun `two callers share one download`() = runBlocking {
        // The download button and a read that arrived before the model existed.
        val button = async { ModelInstall.ensure(context) }
        val read = async { ModelInstall.ensure(context) }

        withTimeout(5_000) {
            button.await()
            read.await()
        }
        assertEquals("the model was downloaded twice", 1, downloads.get())
    }

    @Test
    fun `progress is published where a screen can find it after rotation`() = runBlocking {
        val install = async { ModelInstall.ensure(context) }
        withTimeout(5_000) {
            while (ModelInstall.state.value !is ModelInstall.State.Working) delay(5)
        }
        // A recreated activity reads this rather than owning the transfer.
        assertTrue(ModelInstall.state.value is ModelInstall.State.Working)
        install.await()
        assertEquals(ModelInstall.State.Installed, ModelInstall.state.value)
    }

    @Test
    fun `a failure is reported rather than swallowed`() = runBlocking {
        ModelInstall.installer = ModelInstall.Installer { _, _, _ ->
            throw java.io.IOException("no network")
        }
        runCatching { ModelInstall.ensure(context) }
        val state = ModelInstall.state.value
        assertTrue("expected Failed, got $state", state is ModelInstall.State.Failed)
        assertEquals("no network", (state as ModelInstall.State.Failed).message)
    }

}
