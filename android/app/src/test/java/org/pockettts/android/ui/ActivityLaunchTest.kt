package org.pockettts.android.ui

import android.content.Intent
import android.os.Build
import androidx.appcompat.widget.Toolbar
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.CompletableDeferred
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import android.view.MotionEvent
import android.view.View
import android.widget.TextView
import com.google.android.material.slider.Slider
import org.pockettts.android.R
import org.pockettts.android.engine.Settings
import org.pockettts.android.player.FakeEngine
import org.pockettts.android.player.FakeSink
import org.pockettts.android.player.Reader
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config

/**
 * Boots every activity for real.
 *
 * These exist because of a specific escape: `Theme.PocketTts` inherited from
 * `Theme.Material3.DayNight`, which hands the activity a decor action bar, while
 * two screens also called `setSupportActionBar`. That combination throws
 * `IllegalStateException` the instant the activity is created, so opening the
 * voice picker or the scratchpad closed the app. It compiled, it packaged, it
 * passed every test - and it could not survive a single tap.
 *
 * A theme, a manifest entry and a layout are as capable of crashing an activity
 * as any function is, and none of them are type-checked. So each screen gets
 * driven through its real lifecycle here.
 *
 * Reaching RESUMED is not enough on its own. Views validate themselves during
 * layout, and `setup()` alone never lays anything out - which is how a Material
 * `Slider` handed an off-grid value (`170f/255f` against a `0.01` step) sailed
 * through these tests and then threw `IllegalStateException` from
 * `onSizeChanged` the moment the appearance screen was opened. [layOut] forces
 * a real measure and layout pass so that class of failure surfaces here.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [Build.VERSION_CODES.UPSIDE_DOWN_CAKE])
class ActivityLaunchTest {

    /**
     * A fake engine, held mid-sentence.
     *
     * Two reasons. Any screen that starts a read used to reach the real
     * engine, which meant `ensureModel` - so this suite quietly downloaded a
     * 98 MB model bundle from GitHub on every run. And holding the read open
     * is what lets a window be inspected while it is doing its job, rather
     * than after it has correctly closed itself.
     */
    private val engine = FakeEngine()
    private val gate = CompletableDeferred<Unit>()

    @Before
    fun useFakeEngine() {
        Reader.resetForTesting()
        engine.gate = gate
        Reader.engines = FakeEngine.Factory(engine)
        Reader.sinks = FakeSink.Factory()
    }

    @After
    fun releaseReader() {
        gate.complete(Unit)
        Reader.resetForTesting()
    }

    /**
     * Forces measure, layout and draw at a realistic size.
     *
     * Each phase catches a different class of failure, and reaching RESUMED
     * catches none of them: views validate their configuration while sizing
     * (this is where an off-grid `Slider` value throws), and custom views do
     * their real work in draw. Stopping at `setup()` let both through.
     */
    /** Presses the slider at [toFraction] of its width, as a finger would. */
    private fun drag(slider: Slider, toFraction: Float) {
        val x = slider.width * toFraction
        val y = slider.height / 2f
        listOf(MotionEvent.ACTION_DOWN, MotionEvent.ACTION_UP).forEach { action ->
            val event = MotionEvent.obtain(0L, 0L, action, x, y, 0)
            slider.dispatchTouchEvent(event)
            event.recycle()
        }
    }

    /**
     * Polls [check] on this thread until it holds.
     *
     * The reader runs on its own dispatchers, so what it does is not ordered
     * against an activity reaching RESUMED. Sleeping is right here: nothing
     * being waited for is delivered by the Robolectric main looper.
     */
    private fun waitFor(what: String, timeoutMillis: Long = 10_000, check: () -> Boolean) {
        val deadline = System.currentTimeMillis() + timeoutMillis
        while (System.currentTimeMillis() < deadline) {
            if (check()) return
            Thread.sleep(5)
        }
        fail("never became true: $what")
    }

    private fun layOut(activity: android.app.Activity) {
        val root = activity.findViewById<android.view.View>(android.R.id.content)
        root.measure(
            android.view.View.MeasureSpec.makeMeasureSpec(1080, android.view.View.MeasureSpec.EXACTLY),
            android.view.View.MeasureSpec.makeMeasureSpec(2400, android.view.View.MeasureSpec.EXACTLY),
        )
        root.layout(0, 0, 1080, 2400)

        val bitmap = android.graphics.Bitmap.createBitmap(
            1080,
            2400,
            android.graphics.Bitmap.Config.ARGB_8888,
        )
        root.draw(android.graphics.Canvas(bitmap))
        bitmap.recycle()
    }

    @Test
    fun `appearance screen survives layout with the stored glass values`() {
        Robolectric.buildActivity(AppearanceActivity::class.java).setup().use { controller ->
            // The sliders validate here, not at construction.
            layOut(controller.get())
            assertFalse(controller.get().isFinishing)
        }
    }

    @Test
    fun `main activity reaches resumed state`() {
        Robolectric.buildActivity(MainActivity::class.java).setup().use { controller ->
            layOut(controller.get())
            assertNotNull(controller.get().findViewById<Toolbar>(R.id.toolbar))
        }
    }

    @Test
    fun `the engine sliders show what is stored and store what is dragged`() {
        // The appearance sliders looked wired up for a whole release while
        // doing nothing, so a slider that is only laid out is not evidence of
        // anything. This reads the stored value back off the control and then
        // drives the control and reads the store.
        //
        // Decode steps meant something only to sherpa-onnx and are hidden
        // since the engine swap; speed is the engine slider that acts now.
        val settings = Settings(ApplicationProvider.getApplicationContext())
        settings.speed = 1.5f

        Robolectric.buildActivity(MainActivity::class.java).setup().use { controller ->
            val activity = controller.get()
            layOut(activity)
            assertEquals(View.GONE, activity.findViewById<Slider>(R.id.stepsSlider).visibility)
            assertEquals(View.GONE, activity.findViewById<TextView>(R.id.stepsLabel).visibility)
            val slider = activity.findViewById<Slider>(R.id.speedSlider)
            val label = activity.findViewById<TextView>(R.id.speedLabel)

            assertEquals("the slider ignored the stored value", 1.5f, slider.value, 0.001f)
            assertTrue("the label did not say the value: ${label.text}", "1.50" in label.text)

            // Through a touch rather than by calling the listener, because the
            // listener only acts on user changes.
            drag(slider, toFraction = 1f)

            assertEquals("dragging the slider did not reach the settings", 2.0f, settings.speed, 0.001f)
            assertTrue("the label did not follow: ${label.text}", "2.00" in label.text)
        }
    }

    @Test
    fun `voice picker reaches resumed state`() {
        Robolectric.buildActivity(VoicePickerActivity::class.java).setup().use { controller ->
            layOut(controller.get())
            assertNotNull(controller.get())
        }
    }

    @Test
    fun `model picker lists every model, each a radio button to TalkBack`() {
        Settings(ApplicationProvider.getApplicationContext()).modelName = "english_7m_2026-10"
        Robolectric.buildActivity(ModelPickerActivity::class.java).setup().use { controller ->
            val activity = controller.get()
            layOut(activity)
            val list = activity.findViewById<androidx.recyclerview.widget.RecyclerView>(R.id.modelList)
            val catalog = org.pockettts.android.engine.ModelCatalog.get(activity)
            // The explanation, then one row per model.
            assertEquals(catalog.models.size + 1, list.adapter?.itemCount)
            val rows = (0 until list.childCount).map { list.getChildAt(it) }
                .mapNotNull { it.findViewById<View>(R.id.row) }
            assertEquals(catalog.models.size, rows.size)
            for ((row, model) in rows.zip(catalog.models)) {
                val info = android.view.accessibility.AccessibilityNodeInfo.obtain()
                row.onInitializeAccessibilityNodeInfo(info)
                assertEquals(android.widget.RadioButton::class.java.name, info.className)
                assertTrue(info.isCheckable)
                assertEquals(model.name == "english_7m_2026-10", info.isChecked)
                val description = row.contentDescription.toString()
                assertTrue(description, description.startsWith(ModelNames.title(activity, model.name)))
                assertTrue(description, description.contains("MB"))
            }
        }
    }

    @Test
    fun `scratchpad reaches resumed state`() {
        Robolectric.buildActivity(ScratchpadActivity::class.java).setup().use { controller ->
            layOut(controller.get())
            assertNotNull(controller.get())
        }
    }

    @Test
    fun `read aloud activity survives the glass treatment`() {
        val intent = Intent(
            ApplicationProvider.getApplicationContext(),
            ReadAloudActivity::class.java,
        ).apply {
            action = Intent.ACTION_PROCESS_TEXT
            type = "text/plain"
            putExtra(Intent.EXTRA_PROCESS_TEXT, "Hello world.")
        }
        Robolectric.buildActivity(ReadAloudActivity::class.java, intent).setup().use { controller ->
            val activity = controller.get()
            layOut(activity)
            // Blur is best-effort, but the surface behind the controls is not:
            // without a window background the panel is invisible over the host
            // app, which is exactly what it looked like before.
            assertNotNull("read-aloud window has no background", activity.window.decorView.background)
            // Still open, because the read it started is still going. The
            // window closing once the read *ends* is the intended behaviour,
            // so the engine is held mid-sentence rather than asserting the
            // window never closes.
            assertFalse("window closed while still reading", activity.isFinishing)
        }
    }

    @Test
    fun `rotating mid-sentence does not start the paragraph again`() {
        // A configuration change recreates the activity with the same intent.
        // Handling it a second time stopped the read and began it from the
        // first word, so turning the phone restarted the paragraph.
        val intent = Intent(
            ApplicationProvider.getApplicationContext(),
            ReadAloudActivity::class.java,
        ).apply {
            action = Intent.ACTION_PROCESS_TEXT
            type = "text/plain"
            putExtra(Intent.EXTRA_PROCESS_TEXT, "One sentence. Two sentences. Three.")
        }

        Robolectric.buildActivity(ReadAloudActivity::class.java, intent).setup().use { controller ->
            // Waited for rather than read straight off: the reader hands a
            // chunk to the engine from its own thread, some time after the
            // activity has finished starting. Reading the count here used to
            // work only because the fake engine returned instantly.
            waitFor("the first chunk reached the engine") { engine.spoken.isNotEmpty() }
            val spokenBefore = engine.spoken.size

            controller.recreate()

            // Given a fair chance to speak again, and asserted that it did
            // not. Ten times the fake engine's own latency, so "it did not
            // happen" is not merely "it had not happened yet".
            val grew = (0 until 100).any {
                Thread.sleep(5)
                engine.spoken.size > spokenBefore
            }
            assertFalse("the read started again after rotation", grew)
        }
    }

    @Test
    fun `text selected inside the app is handed to the scratchpad, not a second window`() {
        // The PROCESS_TEXT item this app registers appears in the selection
        // toolbar of its own editor, so selecting text in the scratchpad used
        // to open a floating window over the screen that already had controls
        // for it: two sets of controls for one utterance, in two different
        // treatments. The window steps aside and tags the read instead.
        val intent = Intent(
            ApplicationProvider.getApplicationContext(),
            ReadAloudActivity::class.java,
        ).apply {
            action = Intent.ACTION_PROCESS_TEXT
            type = "text/plain"
            putExtra(Intent.EXTRA_PROCESS_TEXT, "A line from the scratchpad.")
        }

        val controller = Robolectric.buildActivity(ReadAloudActivity::class.java, intent)
        val activity = controller.get()
        shadowOf(activity).setCallingPackage(activity.packageName)
        controller.setup().use {
            assertTrue("window stayed open over our own screen", activity.isFinishing)
        }
        assertEquals(Reader.Source.Scratchpad, Reader.state.value.source)
    }

    @Test
    fun `text selected in another app keeps the window`() {
        val intent = Intent(
            ApplicationProvider.getApplicationContext(),
            ReadAloudActivity::class.java,
        ).apply {
            action = Intent.ACTION_PROCESS_TEXT
            type = "text/plain"
            putExtra(Intent.EXTRA_PROCESS_TEXT, "A line from somewhere else.")
        }

        val controller = Robolectric.buildActivity(ReadAloudActivity::class.java, intent)
        shadowOf(controller.get()).setCallingPackage("com.example.browser")
        controller.setup().use { c ->
            assertFalse("window closed over another app", c.get().isFinishing)
        }
        assertEquals(Reader.Source.Selection, Reader.state.value.source)
    }

    /**
     * The theme must not supply an action bar, or `setSupportActionBar` throws.
     * Asserting on the attribute catches a future parent-theme change directly,
     * rather than leaving it to whichever screen happens to be opened first.
     */
    @Test
    fun `app theme does not supply a decor action bar`() {
        Robolectric.buildActivity(MainActivity::class.java).setup().use { controller ->
            val activity = controller.get()
            val attrs = intArrayOf(androidx.appcompat.R.attr.windowActionBar)
            val typed = activity.theme.obtainStyledAttributes(attrs)
            try {
                assertFalse(
                    "Theme.PocketTts must be a NoActionBar theme",
                    typed.getBoolean(0, true),
                )
            } finally {
                typed.recycle()
            }
        }
    }

    @Test
    fun `read aloud with no text does not linger`() {
        val intent = Intent(
            ApplicationProvider.getApplicationContext(),
            ReadAloudActivity::class.java,
        ).apply {
            action = Intent.ACTION_PROCESS_TEXT
            type = "text/plain"
            putExtra(Intent.EXTRA_PROCESS_TEXT, "   ")
        }
        Robolectric.buildActivity(ReadAloudActivity::class.java, intent).create().use { controller ->
            assertTrue(controller.get().isFinishing)
        }
    }
}
