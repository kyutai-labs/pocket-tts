package org.pockettts.android.engine

import android.content.Context
import android.os.Build
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

/**
 * Where imported voices live, and what happens when one is named like a stock voice.
 *
 * This is the only data in the app that cannot be fetched again. A model
 * re-downloads and a setting is retyped in seconds; a voice somebody recorded
 * exists in one directory and nowhere else. So the failure worth guarding
 * against is not a crash, it is a quiet overwrite - which is exactly what
 * happened: imports shared a directory with the downloaded prompts, so a wav
 * named `alba.wav` landed on the path the stock Alba prompt is cached at,
 * replaced it, and then vanished from the list because anything named like a
 * stock voice was filtered out of it.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [Build.VERSION_CODES.UPSIDE_DOWN_CAKE])
class VoiceStorageTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val manager = ModelManager(context)
    private val voices = File(context.filesDir, "pocket-tts/voices")

    /** A minimal readable 16-bit mono wav; import validates before storing. */
    private fun wav(): ByteArray {
        val samples = 16
        val data = ByteArray(samples * 2)
        val header = java.io.ByteArrayOutputStream()
        fun ascii(s: String) = header.write(s.toByteArray(Charsets.US_ASCII))
        fun int32(v: Int) = (0..3).forEach { header.write((v shr (it * 8)) and 0xFF) }
        fun int16(v: Int) = (0..1).forEach { header.write((v shr (it * 8)) and 0xFF) }
        ascii("RIFF"); int32(36 + data.size); ascii("WAVE")
        ascii("fmt "); int32(16); int16(1); int16(1); int32(24000); int32(48000); int16(2); int16(16)
        ascii("data"); int32(data.size); header.write(data)
        return header.toByteArray()
    }

    @Before
    fun clean() {
        voices.deleteRecursively()
    }

    @Test
    fun `importing a voice named after a stock one does not replace it`() {
        val stock = VoiceCatalog.default()
        val cached = File(voices, stock.fileName).apply {
            parentFile?.mkdirs()
            writeBytes("the downloaded prompt".toByteArray())
        }

        val imported = manager.importVoice(stock.id, wav())

        assertNotEquals("the import landed on the stock prompt", cached.absolutePath, imported.absolutePath)
        assertEquals("the stock prompt was overwritten", "the downloaded prompt", cached.readText())
    }

    @Test
    fun `an import named after a stock voice is still reachable under its own id`() {
        val stock = VoiceCatalog.default()
        val imported = manager.importVoice(stock.id, wav())
        val id = imported.nameWithoutExtension

        // Its id must not be one VoiceCatalog already answers to, or the
        // reader would resolve it to the stock voice and the import would be
        // unreachable.
        assertTrue("id $id collides with a stock voice", VoiceCatalog.byId(id) == null)
        assertTrue(manager.importedVoices().any { it.nameWithoutExtension == id })
        assertEquals(imported.absolutePath, manager.voiceFile(id).absolutePath)
    }

    @Test
    fun `two imports of the same name both survive`() {
        val first = manager.importVoice("recording", wav())
        val second = manager.importVoice("recording", wav())
        assertNotEquals(first.absolutePath, second.absolutePath)
        assertEquals(2, manager.importedVoices().size)
    }

    @Test
    fun `an import lists and resolves`() {
        val file = manager.importVoice("my_own_voice", wav())
        assertEquals("my_own_voice", file.nameWithoutExtension)
        assertEquals(listOf("my_own_voice"), manager.importedVoices().map { it.nameWithoutExtension })
        assertEquals(file.absolutePath, manager.voiceFile("my_own_voice").absolutePath)
    }

    @Test
    fun `voices imported by an older build are carried over, not lost`() {
        // The previous layout put them straight in the voices directory.
        voices.mkdirs()
        File(voices, "old_import.wav").writeBytes(wav())

        assertEquals(listOf("old_import"), manager.importedVoices().map { it.nameWithoutExtension })
        assertTrue(manager.voiceFile("old_import").isFile)
    }

    @Test
    fun `a downloaded stock prompt is not mistaken for an import`() {
        val stock = VoiceCatalog.default()
        voices.mkdirs()
        File(voices, stock.fileName).writeBytes(wav())

        assertTrue("the stock prompt was migrated as if imported", manager.importedVoices().isEmpty())
    }

    @Test
    fun `a stock prompt of the wrong size is not treated as cached`() {
        // The permanent one. An import named like a stock voice overwrote that
        // voice's prompt, and because any non-empty file counted as cached,
        // nothing ever fetched the real one again - so every "Alba" from that
        // day on was read in someone else's voice, silently, on every build.
        val alba = VoiceCatalog.byId("alba")!!
        voices.mkdirs()
        File(voices, alba.fileName).writeBytes(wav())

        assertFalse(
            "someone else's recording was accepted as Alba's prompt",
            manager.isCached(alba),
        )
    }

    @Test
    fun `the real prompt is treated as cached`() {
        val alba = VoiceCatalog.byId("alba")!!
        voices.mkdirs()
        // Not the real audio - the check is a size, which is what a stat can
        // see and what distinguishes a different recording.
        File(voices, alba.fileName).writeBytes(ByteArray(alba.bytes.toInt()))

        assertTrue(manager.isCached(alba))
    }

    @Test
    fun `every stock voice carries the size it ships as`() {
        // Read from Hugging Face rather than guessed. A zero here would make
        // the check above vacuous for that voice.
        VoiceCatalog.voices.filter { it.url != null }.forEach {
            assertTrue("${it.id} has no expected size", it.bytes > 0)
        }
        // Only these have no public recording, and are previewed by reading.
        assertEquals(
            setOf("juergen", "lola", "giovanni", "rafael", "daan"),
            VoiceCatalog.voices.filter { it.url == null }.map { it.id }.toSet(),
        )
    }
}
