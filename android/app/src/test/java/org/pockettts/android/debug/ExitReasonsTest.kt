package org.pockettts.android.debug

import android.app.ApplicationExitInfo
import android.os.Build
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * The exit reason is the one fact that decides what to do next: a native crash,
 * an ANR and a low-memory kill each demand a completely different fix, and all
 * three look identical from the outside. So the mapping from reason code to
 * words has to be right.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [Build.VERSION_CODES.UPSIDE_DOWN_CAKE])
class ExitReasonsTest {

    @Test
    fun `every reason code maps to a name`() {
        val codes = listOf(
            ApplicationExitInfo.REASON_ANR,
            ApplicationExitInfo.REASON_CRASH,
            ApplicationExitInfo.REASON_CRASH_NATIVE,
            ApplicationExitInfo.REASON_DEPENDENCY_DIED,
            ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE,
            ApplicationExitInfo.REASON_EXIT_SELF,
            ApplicationExitInfo.REASON_INITIALIZATION_FAILURE,
            ApplicationExitInfo.REASON_LOW_MEMORY,
            ApplicationExitInfo.REASON_OTHER,
            ApplicationExitInfo.REASON_PERMISSION_CHANGE,
            ApplicationExitInfo.REASON_SIGNALED,
            ApplicationExitInfo.REASON_USER_REQUESTED,
            ApplicationExitInfo.REASON_USER_STOPPED,
        )
        codes.forEach { code ->
            val name = ExitReasons.reasonName(code)
            assertTrue("code $code fell through to UNKNOWN", !name.startsWith("UNKNOWN"))
        }
    }

    @Test
    fun `an unrecognised code is reported rather than silently swallowed`() {
        assertEquals("UNKNOWN(9999)", ExitReasons.reasonName(9999))
    }

    @Test
    fun `native crash is named distinctly from a java crash`() {
        val native = ExitReasons.reasonName(ApplicationExitInfo.REASON_CRASH_NATIVE)
        val java = ExitReasons.reasonName(ApplicationExitInfo.REASON_CRASH)
        // These two send us to completely different places; conflating them
        // would send someone hunting Kotlin for a segfault in the native engine.
        assertTrue(native != java)
        assertEquals("CRASH_NATIVE", native)
        assertEquals("CRASH", java)
    }
}
