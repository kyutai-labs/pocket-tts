package org.pockettts.android.engine

import android.content.Context
import android.os.Build
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.xmlpull.v1.XmlPullParser
import org.pockettts.android.R

/**
 * The backup rules name the models' directory as a literal string, because
 * XML cannot reference a Kotlin constant.
 *
 * That makes them a silent trap. Rename the bundle and the exclusion stops
 * matching: nothing fails to compile, no test that exists elsewhere notices,
 * and the next backup quietly tries to upload 200 MB against a 25 MB quota -
 * failing the whole backup, and taking the settings and the user's own
 * recorded voices with it. Exactly the shape of failure this project keeps
 * finding on a phone instead of in CI.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [Build.VERSION_CODES.UPSIDE_DOWN_CAKE])
class BackupRulesTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()

    private fun excludedPaths(resource: Int): List<String> {
        val paths = mutableListOf<String>()
        context.resources.getXml(resource).use { parser ->
            while (parser.next() != XmlPullParser.END_DOCUMENT) {
                if (parser.eventType == XmlPullParser.START_TAG && parser.name == "exclude") {
                    for (i in 0 until parser.attributeCount) {
                        if (parser.getAttributeName(i) == "path") paths += parser.getAttributeValue(i)
                    }
                }
            }
        }
        return paths
    }

    private val expected = "pocket-tts/${ModelManager.MODELS_DIR}"

    @Test
    fun `cloud backup excludes the model directory that actually exists`() {
        val paths = excludedPaths(R.xml.data_extraction_rules)
        assertTrue(
            "data_extraction_rules.xml excludes $paths, but the model lives at $expected",
            paths.contains(expected),
        )
    }

    @Test
    fun `the pre-Android-12 rules exclude it too`() {
        val paths = excludedPaths(R.xml.backup_rules)
        assertTrue(
            "backup_rules.xml excludes $paths, but the model lives at $expected",
            paths.contains(expected),
        )
    }

    @Test
    fun `nothing under the voices directory is excluded`() {
        // Voices are the one thing here that cannot be fetched again: a wav the
        // user recorded or imported exists nowhere but this directory.
        val everything = excludedPaths(R.xml.data_extraction_rules) +
            excludedPaths(R.xml.backup_rules)
        assertTrue(
            "a rule excludes voices, which cannot be recovered: $everything",
            everything.none { it.contains("voices") },
        )
    }
}
