package org.pockettts.android.engine

import android.os.Build
import android.speech.tts.SynthesisCallback
import android.speech.tts.TextToSpeech
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import androidx.test.core.app.ApplicationProvider
import org.pockettts.android.speech.TextChunker
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * The system engine, which is the surface this app controls least.
 *
 * Every other screen is driven by someone who chose to open it. This one is
 * driven by Select to Speak, Chrome, and ebook readers, in whatever order and
 * with whatever text they like - and it had no tests at all.
 *
 * The contract is unusual in that failing it is quiet. A caller that gets
 * silence, or gets its text back rewritten, has no way to tell this engine did
 * it, and the person listening is being told the screen says something it does
 * not.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [Build.VERSION_CODES.UPSIDE_DOWN_CAKE])
class PocketTtsServiceTest {

    private val service: PocketTtsService =
        Robolectric.buildService(PocketTtsService::class.java).create().get()

    /** Records what the framework would have been told. */
    private class RecordingCallback : SynthesisCallback {
        var started = 0
        var done = 0
        var error: Int? = null
        val ranges = mutableListOf<Pair<Int, Int>>()
        val audio = mutableListOf<Int>()
        private var maxBuffer = 4096

        override fun getMaxBufferSize() = maxBuffer
        override fun start(sampleRateInHz: Int, audioFormat: Int, channelCount: Int): Int {
            started++
            return TextToSpeech.SUCCESS
        }

        override fun audioAvailable(buffer: ByteArray?, offset: Int, length: Int): Int {
            audio += length
            return TextToSpeech.SUCCESS
        }

        override fun done(): Int {
            done++
            return TextToSpeech.SUCCESS
        }

        override fun error() { error = TextToSpeech.ERROR }
        override fun error(errorCode: Int) { error = errorCode }
        override fun hasStarted() = started > 0
        override fun hasFinished() = done > 0
        override fun rangeStart(start: Int, end: Int, frame: Int) { ranges += start to end }
    }

    private fun request(text: String) = android.speech.tts.SynthesisRequest(
        text,
        android.os.Bundle(),
    )

    @Test
    fun `refuses rather than hanging when the model is not downloaded`() {
        // Downloading 98 MB inside a synthesis request looks like a hang to
        // whichever app asked, and there is a correct code for this.
        val callback = RecordingCallback()
        service.onSynthesizeText(request("Anything at all."), callback)

        assertEquals(TextToSpeech.ERROR_NOT_INSTALLED_YET, callback.error)
        assertEquals("started an utterance it could not deliver", 0, callback.started)
    }

    @Test
    fun `empty text completes rather than erroring`() {
        // A caller that sends whitespace has not made a mistake worth an error;
        // it has asked for nothing, and nothing is a valid thing to deliver.
        val callback = RecordingCallback()
        service.onSynthesizeText(request("   "), callback)

        assertEquals(1, callback.started)
        assertEquals(1, callback.done)
        assertEquals(null, callback.error)
    }

    @Test
    fun `the voice list says so when the model is not installed`() {
        // Without this a picker offers voices that cannot speak, and choosing
        // one produces silence with no explanation.
        val voices = service.onGetVoices()
        assertTrue("no voices offered at all", voices.isNotEmpty())
        assertTrue(
            "voices did not report themselves as needing installation",
            voices.all { TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED in it.features },
        )
    }

    @Test
    fun `english is available and its countries are recognised`() {
        assertEquals(
            TextToSpeech.LANG_COUNTRY_AVAILABLE,
            service.onIsLanguageAvailable("eng", "GBR", ""),
        )
        assertEquals(
            TextToSpeech.LANG_COUNTRY_AVAILABLE,
            service.onIsLanguageAvailable("eng", "USA", ""),
        )
        assertEquals(
            TextToSpeech.LANG_NOT_SUPPORTED,
            service.onIsLanguageAvailable("jpn", "JPN", ""),
        )
    }

    @Test
    fun `another language is missing data until its model is downloaded`() {
        // Not "not supported": a caller can then say what is missing rather
        // than read French with the English model.
        assertEquals(TextToSpeech.LANG_MISSING_DATA, service.onIsLanguageAvailable("fra", "FRA", ""))
        assertEquals(TextToSpeech.LANG_MISSING_DATA, service.onIsLanguageAvailable("ger", "", ""))

        val settings = Settings(ApplicationProvider.getApplicationContext())
        settings.modelName = "french"
        try {
            // The reading language is available: the app's first screen
            // fetches its model, as it always has English's.
            assertEquals(TextToSpeech.LANG_AVAILABLE, service.onIsLanguageAvailable("fra", "FRA", ""))
            assertEquals(TextToSpeech.LANG_MISSING_DATA, service.onIsLanguageAvailable("eng", "GBR", ""))
            assertEquals(arrayOf("fra", "", "").toList(), service.onGetLanguage().toList())
        } finally {
            settings.modelName = Settings.DEFAULT_MODEL
        }
    }

    @Test
    fun `each language has its own alias and voices`() {
        assertEquals("selected", service.onGetDefaultVoiceNameFor("eng", "USA", ""))
        assertEquals("selected-fr", service.onGetDefaultVoiceNameFor("fra", "", ""))
        assertEquals("selected", service.onGetDefaultVoiceNameFor("jpn", "", ""))
        assertEquals(TextToSpeech.SUCCESS, service.onIsValidVoiceName("selected-de"))
        assertEquals(TextToSpeech.SUCCESS, service.onIsValidVoiceName("alba-fr"))
        assertEquals(TextToSpeech.SUCCESS, service.onIsValidVoiceName("estelle"))
        assertEquals(TextToSpeech.ERROR, service.onIsValidVoiceName("alba-ja"))

        val names = service.onGetVoices().map { it.name }
        assertTrue(names.containsAll(listOf("selected", "alba", "selected-fr", "selected-nl")))
        // Only the reading language's voices one by one: nothing else is downloaded.
        assertFalse("alba-fr" in names)
    }

    @Test
    fun `a request is read in its language, with that language's voice`() {
        val settings = Settings(ApplicationProvider.getApplicationContext())
        fun reading(voice: String?, language: String) = service.readingFor(voice, language)

        assertEquals("fr", reading("selected-fr", "fra").language)
        assertEquals(settings.voiceFor("fr"), reading("selected-fr", "fra").voiceId)
        assertEquals("estelle", reading("selected-fr", "fra").voiceId)
        // A concrete voice is read in its own language.
        assertEquals("fr", reading("alba-fr", "eng").language)
        assertEquals("alba", reading("alba-fr", "eng").voiceId)
        // The English alias with a French request: the request's language.
        assertEquals("fr", reading("selected", "fra").language)
        // A language with nothing to read it falls back to the reading language.
        assertEquals("en", reading(null, "jpn").language)
        assertEquals(Settings.DEFAULT_MODEL, reading(null, "jpn").model)
        // Nothing downloaded for French, so no model to read it with.
        assertEquals(null, reading("selected-fr", "fra").model)
    }

    @Test
    fun `a stock voice is a valid voice name and nonsense is not`() {
        assertEquals(TextToSpeech.SUCCESS, service.onIsValidVoiceName(VoiceCatalog.DEFAULT_VOICE_ID))
        assertEquals(TextToSpeech.ERROR, service.onIsValidVoiceName("not-a-voice"))
        assertEquals(TextToSpeech.ERROR, service.onIsValidVoiceName(null))
    }

    @Test
    fun `text from another app is not rewritten on its way through`() {
        // The engine used to run everything through the Markdown stripper, so
        // asterisks disappeared from ordinary sentences and links were replaced
        // by nothing at all. An accessibility tool exists to say what is on the
        // screen; deleting a URL from that is not formatting, it is a false
        // account of the content.
        val text = "See https://example.com for the *details* of file_name_here."
        assertEquals(text, service.speakableFor(request(text)))
    }

    @Test
    fun `only surrounding whitespace is taken off`() {
        assertEquals("Two  spaces  kept.", service.speakableFor(request("  Two  spaces  kept.  ")))
        assertEquals("", service.speakableFor(request("   ")))
        assertEquals("", service.speakableFor(null))
    }

    @Test
    fun `chunk offsets point into the text the caller sent`() {
        // What rangeStart reports, and what lets a caller highlight along. They
        // are only meaningful because the text goes through unrewritten -
        // stripping is what made them drift.
        val text = "First sentence here. Second sentence here. Third one."
        val speakable = service.speakableFor(request(text))
        TextChunker.chunk(speakable).forEach { chunk ->
            assertTrue("offset ${chunk.start}..${chunk.end} is outside the text", chunk.end <= text.length)
            assertEquals(chunk.text.trim(), text.substring(chunk.start, chunk.end).trim())
        }
    }


    @Test
    fun `the default voice name is resolved on each request, not at bind time`() {
        // Android asks an engine for its default voice name once per client and
        // then sends that name back on every request for as long as the client
        // lives. Returning a concrete voice id here pinned whichever voice was
        // selected when Select to Speak or Chrome first bound: changing the
        // voice in this app did nothing for them until they were force-stopped.
        val settings = Settings(ApplicationProvider.getApplicationContext())
        settings.voiceId = "jane"
        val name = service.onGetDefaultVoiceNameFor("eng", "GBR", "")

        settings.voiceId = "paul"

        assertEquals(
            "the default voice name is a snapshot of the selection",
            name,
            service.onGetDefaultVoiceNameFor("eng", "GBR", ""),
        )
        assertEquals(TextToSpeech.SUCCESS, service.onIsValidVoiceName(name))
        assertTrue(
            "the name a client caches is one of the voices offered",
            service.onGetVoices().any { it.name == name },
        )
    }
}
