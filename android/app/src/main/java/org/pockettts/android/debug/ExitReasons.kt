package org.pockettts.android.debug

import android.app.ActivityManager
import android.app.ActivityManager.RunningAppProcessInfo
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import androidx.annotation.RequiresApi
import java.util.Date

/**
 * Reports why the previous process died, including deaths a crash handler cannot see.
 *
 * [CrashLog] only catches exceptions that unwind through the JVM. A native
 * segfault, an ANR, and a low-memory kill all bypass it entirely and leave the
 * app looking like it "just closed" - which is a symptom shared by about six
 * unrelated causes, so it identifies none of them.
 *
 * The platform already records the answer. `getHistoricalProcessExitReasons`
 * returns a reason code per death, and from Android 12 a native crash carries
 * its tombstone. Reading it turns a guess into a fact.
 */
object ExitReasons {

    /** A death worth telling the user about, already rendered for display. */
    data class Report(val summary: String, val detail: String, val timestamp: Long)

    /**
     * Deaths that always mean something went wrong, whatever the app was doing.
     */
    private val ALWAYS_REPORT = setOf(
        ApplicationExitInfo.REASON_CRASH,
        ApplicationExitInfo.REASON_CRASH_NATIVE,
        ApplicationExitInfo.REASON_ANR,
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE,
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE,
    )

    /**
     * Deaths that are a failure or a non-event depending entirely on what the
     * process was doing at the time.
     *
     * Android reclaims cached background processes constantly; that is the
     * memory manager working, not the app breaking. Several OEM task killers
     * report the same housekeeping as `SIGNALED`. Being killed *while reading
     * aloud* is a different matter and worth saying out loud.
     */
    private val REPORT_IF_FOREGROUND = setOf(
        ApplicationExitInfo.REASON_LOW_MEMORY,
        ApplicationExitInfo.REASON_SIGNALED,
    )

    /**
     * Whether an exit record deserves a dialog.
     *
     * Split out as a plain function of four numbers because this is the whole
     * decision, and getting it wrong is expensive in both directions: too
     * strict and a native crash goes unreported, too loose and the app cries
     * wolf about routine background trims until nobody reads the dialog.
     *
     * [importance] is the process importance at the moment of death, and lower
     * means more important - `IMPORTANCE_FOREGROUND` is 100, `IMPORTANCE_CACHED`
     * is 400.
     */
    internal fun isWorthReporting(
        reason: Int,
        importance: Int,
        timestamp: Long,
        lastReported: Long,
    ): Boolean {
        if (timestamp <= lastReported) return false
        if (reason in ALWAYS_REPORT) return true
        return reason in REPORT_IF_FOREGROUND &&
            importance <= RunningAppProcessInfo.IMPORTANCE_FOREGROUND_SERVICE
    }

    /** The most recent noteworthy death, or null if the last exit was unremarkable. */
    fun lastInterestingExit(context: Context): Report? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return null
        return runCatching { readLastExit(context) }.getOrNull()
    }

    /**
     * Records that [report] has been shown, so it is not shown again.
     *
     * Without this the platform's history is re-read on every resume and the
     * same record produces the same dialog forever - returning from the voice
     * picker was enough to raise it again.
     */
    fun markReported(context: Context, report: Report) {
        prefs(context).edit().putLong(KEY_LAST_REPORTED, report.timestamp).apply()
    }

    @RequiresApi(Build.VERSION_CODES.R)
    private fun readLastExit(context: Context): Report? {
        val activityManager = context.getSystemService(ActivityManager::class.java) ?: return null
        val history = activityManager.getHistoricalProcessExitReasons(
            context.packageName,
            /* pid = */ 0,
            MAX_RECORDS,
        )
        val lastReported = prefs(context).getLong(KEY_LAST_REPORTED, 0L)
        val exit = history.firstOrNull {
            isWorthReporting(it.reason, it.importance, it.timestamp, lastReported)
        } ?: return null

        return Report(
            summary = summarise(exit),
            detail = describe(exit, CrashLog.deviceMetadata(context)),
            timestamp = exit.timestamp,
        )
    }

    private fun prefs(context: Context) = context.applicationContext
        .getSharedPreferences("pocket-tts", Context.MODE_PRIVATE)

    /** One line naming what happened, which is usually the whole answer. */
    @RequiresApi(Build.VERSION_CODES.R)
    fun summarise(exit: ApplicationExitInfo): String = when (exit.reason) {
        ApplicationExitInfo.REASON_CRASH_NATIVE ->
            "Native crash - the failure was in the engine's native code (pocket-speak), not app code."

        ApplicationExitInfo.REASON_CRASH ->
            "Java crash - an uncaught exception."

        ApplicationExitInfo.REASON_ANR ->
            "Not responding - the main thread was blocked long enough for the system to kill it."

        ApplicationExitInfo.REASON_LOW_MEMORY ->
            "Killed for memory - the system needed the RAM back."

        ApplicationExitInfo.REASON_SIGNALED ->
            "Killed by signal ${exit.status}."

        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE ->
            "Killed for excessive resource use."

        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE ->
            "The process failed to start."

        ApplicationExitInfo.REASON_PERMISSION_CHANGE ->
            "Restarted because a permission changed."

        else -> "Process exited: ${reasonName(exit.reason)}"
    }

    @RequiresApi(Build.VERSION_CODES.R)
    fun describe(exit: ApplicationExitInfo, metadata: Map<String, String>): String = buildString {
        appendLine("Pocket TTS exit report")
        appendLine("when: ${Date(exit.timestamp)}")
        appendLine("reason: ${reasonName(exit.reason)} (${exit.reason})")
        appendLine("status: ${exit.status}")
        appendLine("importance: ${exit.importance}")
        appendLine("rss: ${exit.rss / 1024} MB")
        appendLine("pss: ${exit.pss / 1024} MB")
        exit.description?.let { appendLine("description: $it") }
        for ((key, value) in metadata) appendLine("$key: $value")

        // A native crash carries its tombstone from Android 12 onward, and that
        // is the only place the offending frame is recorded.
        tombstoneOf(exit)?.let {
            appendLine()
            appendLine("--- tombstone ---")
            append(it)
        }
    }

    @RequiresApi(Build.VERSION_CODES.R)
    private fun tombstoneOf(exit: ApplicationExitInfo): String? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return null
        if (exit.reason != ApplicationExitInfo.REASON_CRASH_NATIVE) return null
        return runCatching {
            exit.traceInputStream?.bufferedReader()?.use { reader ->
                // Tombstones are protobuf and can be large; the head carries the
                // signal, the fault address and the crashing frames.
                buildString {
                    var lines = 0
                    while (lines < TOMBSTONE_LINES) {
                        val line = reader.readLine() ?: break
                        appendLine(line)
                        lines++
                    }
                }
            }
        }.getOrNull()
    }

    fun reasonName(reason: Int): String = when (reason) {
        ApplicationExitInfo.REASON_ANR -> "ANR"
        ApplicationExitInfo.REASON_CRASH -> "CRASH"
        ApplicationExitInfo.REASON_CRASH_NATIVE -> "CRASH_NATIVE"
        ApplicationExitInfo.REASON_DEPENDENCY_DIED -> "DEPENDENCY_DIED"
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "EXCESSIVE_RESOURCE_USAGE"
        ApplicationExitInfo.REASON_EXIT_SELF -> "EXIT_SELF"
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "INITIALIZATION_FAILURE"
        ApplicationExitInfo.REASON_LOW_MEMORY -> "LOW_MEMORY"
        ApplicationExitInfo.REASON_OTHER -> "OTHER"
        ApplicationExitInfo.REASON_PERMISSION_CHANGE -> "PERMISSION_CHANGE"
        ApplicationExitInfo.REASON_SIGNALED -> "SIGNALED"
        ApplicationExitInfo.REASON_USER_REQUESTED -> "USER_REQUESTED"
        ApplicationExitInfo.REASON_USER_STOPPED -> "USER_STOPPED"
        else -> "UNKNOWN($reason)"
    }

    private const val KEY_LAST_REPORTED = "exit_last_reported"
    private const val MAX_RECORDS = 5
    private const val TOMBSTONE_LINES = 120
}
