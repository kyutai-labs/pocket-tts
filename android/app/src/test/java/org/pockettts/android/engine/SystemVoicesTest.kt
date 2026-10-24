package org.pockettts.android.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.util.Locale

/** The voice names the system list carries, which apps store and send back. */
class SystemVoicesTest {

    private val languages = listOf("en", "fr", "de", "es", "it", "pt", "nl")

    @Test
    fun `english keeps the names earlier builds gave it`() {
        assertEquals("selected", SystemVoices.selectedName("en"))
        assertEquals("alba", SystemVoices.nameOf("alba", "en"))
        assertEquals(SystemVoices.Named(null, "en"), SystemVoices.parse("selected", languages))
        assertEquals(SystemVoices.Named("alba", "en"), SystemVoices.parse("alba", languages))
    }

    @Test
    fun `other languages carry their tag`() {
        assertEquals("selected-fr", SystemVoices.selectedName("fr"))
        assertEquals("peter_yearsley-de", SystemVoices.nameOf("peter_yearsley", "de"))
        assertEquals(SystemVoices.Named("peter_yearsley", "de"), SystemVoices.parse("peter_yearsley-de", languages))
        assertEquals(SystemVoices.Named(null, "nl"), SystemVoices.parse("selected-nl", languages))
    }

    @Test
    fun `names that are not ours are refused`() {
        assertNull(SystemVoices.parse(null, languages))
        assertNull(SystemVoices.parse("", languages))
        assertNull(SystemVoices.parse("nobody", languages))
        assertNull(SystemVoices.parse("alba-ja", languages))
        assertNull(SystemVoices.parse("nobody-fr", languages))
        assertNull(SystemVoices.parse("alba-en", languages))
    }

    @Test
    fun `three-letter codes, both kinds, and two-letter ones name a language`() {
        assertEquals("en", SystemVoices.languageOf("eng", languages))
        assertEquals("fr", SystemVoices.languageOf("fra", languages))
        assertEquals("fr", SystemVoices.languageOf("fre", languages))
        assertEquals("de", SystemVoices.languageOf("deu", languages))
        assertEquals("de", SystemVoices.languageOf("ger", languages))
        assertEquals("nl", SystemVoices.languageOf("dut", languages))
        assertEquals("pt", SystemVoices.languageOf("POR", languages))
        assertEquals("es", SystemVoices.languageOf("es", languages))
        assertNull(SystemVoices.languageOf("jpn", languages))
        assertNull(SystemVoices.languageOf(null, languages))
    }

    @Test
    fun `a voice is listed under its accent only in its own language`() {
        val alba = VoiceCatalog.byId("alba")!!
        assertEquals(Locale.forLanguageTag("en-GB"), SystemVoices.localeOf(alba, "en"))
        assertEquals(Locale("fr"), SystemVoices.localeOf(alba, "fr"))
        val juergen = VoiceCatalog.byId("juergen")!!
        assertEquals(Locale.forLanguageTag("de-DE"), SystemVoices.localeOf(juergen, "de"))
    }
}
