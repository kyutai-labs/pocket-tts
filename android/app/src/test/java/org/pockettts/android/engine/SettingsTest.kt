package org.pockettts.android.engine

import android.content.Context
import android.os.Build
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * The stored values, and the range each of them is allowed to take.
 *
 * Ranges matter more here than they look: a Material slider handed a value off
 * its own step grid throws during layout rather than rounding, so a setting
 * written by one build and read by the next can take a whole screen down on
 * open. That has already happened once, to the appearance screen.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [Build.VERSION_CODES.UPSIDE_DOWN_CAKE])
class SettingsTest {

    private val settings = Settings(ApplicationProvider.getApplicationContext<Context>())

    @Test
    fun `decode steps default to the reference implementation's one, not sherpa's five`() {
        // This asserted 5 and said so: changing the default silently changes
        // how every existing install sounds, so the answer stayed the one in
        // use until a device answered. A device has now answered - one step was
        // reported closer to the same model on a desktop, and no worse - and
        // the reference implementation uses one, so the tripwire moves rather
        // than being worked around.
        assertEquals(1, settings.decodeSteps)
        assertEquals(Settings.DEFAULT_DECODE_STEPS, settings.decodeSteps)
    }

    @Test
    fun `decode steps stay inside the slider's range`() {
        settings.decodeSteps = 99
        assertEquals(Settings.MAX_STEPS, settings.decodeSteps)

        settings.decodeSteps = 0
        assertEquals(Settings.MIN_STEPS, settings.decodeSteps)

        settings.decodeSteps = 2
        assertEquals(2, settings.decodeSteps)
    }

    @Test
    fun `one voice across sentences is on unless it is turned off`() {
        assertTrue(settings.steadyVoice)
        settings.steadyVoice = false
        assertTrue(!settings.steadyVoice)
    }

    @Test
    fun `speed stays inside the slider's range`() {
        settings.speed = 9f
        assertEquals(Settings.MAX_SPEED, settings.speed, 0f)

        settings.speed = 0f
        assertEquals(Settings.MIN_SPEED, settings.speed, 0f)
    }

    @Test
    fun `temperature follows the model until it is set, and again once reset`() {
        assertEquals(null, settings.temperatureOverride)
        assertEquals(0.3f, settings.temperatureFor(Settings.DEFAULT_MODEL), 0f)
        assertEquals(0.3f, settings.temperatureFor("french"), 0f)

        settings.temperatureOverride = 0.45f
        assertEquals(0.45f, settings.temperatureFor(Settings.DEFAULT_MODEL), 0f)
        assertEquals(0.45f, settings.temperatureFor("french"), 0f)

        settings.temperatureOverride = 9f
        assertEquals(Settings.MAX_TEMPERATURE, settings.temperatureOverride!!, 0f)

        settings.temperatureOverride = null
        assertEquals(0.3f, settings.temperatureFor(Settings.DEFAULT_MODEL), 0f)
    }

    @Test
    fun `a stored 0_3 from before per-model defaults becomes the default, once`() {
        val prefs = ApplicationProvider.getApplicationContext<Context>()
            .getSharedPreferences("pocket-tts", Context.MODE_PRIVATE)
        prefs.edit().putFloat("temperature", 0.3f).commit()
        assertEquals(null, settings.temperatureOverride)

        // After the migration, 0.3 set on purpose stays set.
        settings.temperatureOverride = 0.3f
        assertEquals(0.3f, settings.temperatureOverride!!, 0f)
    }

    @Test
    fun `a value chosen before per-model defaults is kept`() {
        val prefs = ApplicationProvider.getApplicationContext<Context>()
            .getSharedPreferences("pocket-tts", Context.MODE_PRIVATE)
        prefs.edit().putFloat("temperature", 0.6f).commit()
        assertEquals(0.6f, settings.temperatureOverride!!, 0f)
    }

    @Test
    fun `each language keeps its own voice, starting on Kyutai's`() {
        assertEquals("en", settings.readingLanguage)
        assertEquals("alba", settings.voiceId)
        assertEquals("estelle", settings.voiceFor("fr"))
        assertEquals("juergen", settings.voiceFor("de"))

        settings.voiceId = "vera"
        settings.modelName = "french"
        assertEquals("fr", settings.readingLanguage)
        assertEquals("estelle", settings.voiceId)
        settings.voiceId = "alba"

        settings.modelName = "english_7m_2026-10"
        assertEquals("vera", settings.voiceId)
        assertEquals("alba", settings.voiceFor("fr"))
    }

    @Test
    fun `english keeps the voice stored before there were languages`() {
        val prefs = ApplicationProvider.getApplicationContext<Context>()
            .getSharedPreferences("pocket-tts", Context.MODE_PRIVATE)
        prefs.edit().putString("voice", "marius").commit()
        assertEquals("marius", settings.voiceFor("en"))
    }

    @Test
    fun `the selected model reads its language, downloaded or not, and others need a download`() {
        assertEquals(Settings.DEFAULT_MODEL, settings.modelFor("en"))
        // Nothing is downloaded in a test.
        assertEquals(null, settings.modelFor("fr"))
        settings.modelName = "french"
        assertEquals("french", settings.modelFor("fr"))
        assertEquals(null, settings.modelFor("en"))
    }
}
