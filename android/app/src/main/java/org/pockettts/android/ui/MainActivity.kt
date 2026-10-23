package org.pockettts.android.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.net.Uri
import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pockettts.android.R
import org.pockettts.android.databinding.ActivityMainBinding
import org.pockettts.android.debug.CrashLog
import org.pockettts.android.debug.Metrics
import org.pockettts.android.debug.VoiceTrace
import org.pockettts.android.debug.ExitReasons
import org.pockettts.android.engine.ModelInstall
import org.pockettts.android.engine.ModelManager
import org.pockettts.android.engine.Settings
import org.pockettts.android.engine.VoiceCatalog

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var settings: Settings
    private var download: Job? = null

    private val notificationPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)
        Insets.apply(top = binding.appBar, bottom = binding.content)
        settings = Settings(this)

        binding.downloadButton.setOnClickListener { startDownload() }
        binding.timingsButton.setOnClickListener { showTimings() }
        binding.temperatureSlider.addOnChangeListener { _, value, fromUser ->
            if (fromUser) {
                settings.temperature = value
                binding.temperatureLabel.text =
                    getString(R.string.temperature, settings.temperature)
            }
        }
        binding.stepsSlider.addOnChangeListener { _, value, fromUser ->
            if (fromUser) {
                settings.decodeSteps = value.toInt()
                binding.stepsLabel.text = getString(R.string.decode_steps, settings.decodeSteps)
            }
        }
        observeInstall()
        // Decode steps meant something to sherpa-onnx; pocket-speak's engine
        // decodes in one step. Installing from a file took sherpa's archive.
        // Both hidden rather than left doing nothing.
        binding.stepsSlider.visibility = android.view.View.GONE
        binding.stepsLabel.visibility = android.view.View.GONE
        binding.installButton.visibility = android.view.View.GONE
        binding.installButton.setOnClickListener {
            // Any type: the bundle arrives as a .tar.bz2 that most file
            // providers report as application/octet-stream, or as nothing at
            // all, and filtering on that hides the file the user came to pick.
            pickModel.launch(arrayOf("*/*"))
        }
        binding.chooseVoiceButton.setOnClickListener {
            startActivity(Intent(this, VoicePickerActivity::class.java))
        }
        binding.appearanceButton.setOnClickListener {
            startActivity(Intent(this, AppearanceActivity::class.java))
        }
        binding.scratchpadButton.setOnClickListener {
            startActivity(Intent(this, ScratchpadActivity::class.java))
        }
        binding.ttsSettingsButton.setOnClickListener {
            // No public constant for this; it is the settings screen where a
            // user picks their preferred engine.
            runCatching {
                startActivity(Intent("com.android.settings.TTS_SETTINGS"))
            }.onFailure {
                startActivity(Intent(android.provider.Settings.ACTION_ACCESSIBILITY_SETTINGS))
            }
        }

        binding.speedSlider.addOnChangeListener { _, value, fromUser ->
            if (fromUser) {
                settings.speed = value
                binding.speedLabel.text = getString(R.string.speed, value)
            }
        }
        binding.markdownSwitch.setOnCheckedChangeListener { _, checked ->
            settings.treatSelectionAsMarkdown = checked
        }
        binding.codeBlocksSwitch.setOnCheckedChangeListener { _, checked ->
            settings.speakCodeBlocks = checked
        }
        binding.steadyVoiceSwitch.setOnCheckedChangeListener { _, checked ->
            settings.steadyVoice = checked
        }

        requestNotificationPermission()
    }

    /**
     * Without this the playback service still runs, but its notification - and
     * with it the only pause and stop controls once the floating window is
     * gone - is silently dropped on Android 13 and later.
     */
    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        val granted = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.POST_NOTIFICATIONS,
        ) == PackageManager.PERMISSION_GRANTED
        if (!granted) notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
    }

    override fun onResume() {
        super.onResume()
        refresh()
        offerLastCrash()
    }

    /**
     * A sideloaded app that dies has no logcat anyone can reach, so the evidence
     * is offered here instead. Shown once and then cleared.
     *
     * Two sources, because they see different failures. `CrashLog` catches
     * exceptions that unwind through the JVM; `ExitReasons` reads the platform's
     * own record, which is the only thing that sees a native crash, an ANR or a
     * low-memory kill. The platform record wins when both exist - it names the
     * kind of death, which is the part that decides what to do next.
     *
     * "Shown once" is enforced on dismissal rather than per button, because a
     * dialog tapped away is still a dialog that was read. It used to be cleared
     * only by the two buttons, and only for `CrashLog`, so the platform record
     * came back on every single resume - returning from the voice picker was
     * enough to raise the same report again.
     */
    private fun offerLastCrash() {
        val exit = ExitReasons.lastInterestingExit(this)
        val crash = CrashLog.lastCrash(this)
        val report = when {
            exit != null && crash != null -> "${exit.detail}\n\n--- java stack trace ---\n$crash"
            exit != null -> exit.detail
            crash != null -> crash
            else -> return
        }
        val title = if (exit != null) R.string.exit_title else R.string.crash_title

        MaterialAlertDialogBuilder(this)
            .setTitle(title)
            .setMessage(report.lineSequence().take(CRASH_PREVIEW_LINES).joinToString("\n"))
            .setPositiveButton(R.string.crash_share) { _, _ ->
                startActivity(
                    Intent.createChooser(
                        Intent(Intent.ACTION_SEND)
                            .setType("text/plain")
                            .putExtra(Intent.EXTRA_SUBJECT, getString(R.string.crash_title))
                            .putExtra(Intent.EXTRA_TEXT, report),
                        getString(R.string.crash_share),
                    ),
                )
            }
            .setNegativeButton(R.string.crash_dismiss, null)
            .setOnDismissListener {
                CrashLog.clear(this)
                exit?.let { ExitReasons.markReported(this, it) }
            }
            .show()
    }

    private fun refresh() {
        val voice = VoiceCatalog.byId(settings.voiceId)
        binding.voiceLabel.text = voice?.displayName ?: settings.voiceId
        // Through the step grid: Material's Slider throws during layout on a
        // value that misses it, which is how the appearance screen crashed on
        // open once already. A stored speed from another build is exactly the
        // sort of value that would miss.
        binding.speedSlider.value = GlassValues.snapToStep(
            settings.speed,
            binding.speedSlider.valueFrom,
            binding.speedSlider.valueTo,
            binding.speedSlider.stepSize,
        )
        binding.speedLabel.text = getString(R.string.speed, settings.speed)
        binding.temperatureSlider.value = GlassValues.snapToStep(
            settings.temperature,
            binding.temperatureSlider.valueFrom,
            binding.temperatureSlider.valueTo,
            binding.temperatureSlider.stepSize,
        )
        binding.temperatureLabel.text = getString(R.string.temperature, settings.temperature)
        binding.stepsSlider.value = settings.decodeSteps.toFloat()
            .coerceIn(binding.stepsSlider.valueFrom, binding.stepsSlider.valueTo)
        binding.stepsLabel.text = getString(R.string.decode_steps, settings.decodeSteps)
        binding.markdownSwitch.isChecked = settings.treatSelectionAsMarkdown
        binding.codeBlocksSwitch.isChecked = settings.speakCodeBlocks
        binding.steadyVoiceSwitch.isChecked = settings.steadyVoice

        val manager = ModelManager(this)
        val installed = manager.isModelInstalled
        binding.modelStatus.text = if (installed) {
            getString(R.string.model_ready)
        } else {
            getString(R.string.model_missing, (manager.model.downloadBytes / 1_000_000).toInt())
        }
        val busy = ModelInstall.isRunning
        binding.downloadButton.isEnabled = !installed && !busy
        binding.installButton.isEnabled = !installed && !busy
    }

    /**
     * The engine's own numbers, read off the device that produced them.
     *
     * There is no emulator in the build environment, so timings cannot be
     * measured anywhere but here. Showing them - and letting them be shared -
     * is what turns a question about performance into an answer, the same way
     * the exit report turned "it crashed" into a reason.
     */
    private fun showTimings() {
        // The voice trace goes in the same report because the two questions
        // are asked together: what it sounded like, and who was speaking.
        val report = "${Metrics.report()}\n\n" +
            "Voice trace\n${VoiceTrace.report()}\n\n" +
            CrashLog.deviceMetadata(this).entries.joinToString("\n") { "${it.key}: ${it.value}" }

        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.timings)
            .setMessage(report)
            .setPositiveButton(R.string.timings_share) { _, _ ->
                startActivity(
                    Intent.createChooser(
                        Intent(Intent.ACTION_SEND)
                            .setType("text/plain")
                            .putExtra(Intent.EXTRA_SUBJECT, getString(R.string.timings))
                            .putExtra(Intent.EXTRA_TEXT, report),
                        getString(R.string.timings_share),
                    ),
                )
            }
            .setNegativeButton(R.string.crash_dismiss, null)
            .show()
    }

    private val pickModel = registerForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri -> uri?.let { installModel(it) } }

    /** Installs the model from a bundle the user already has on the device. */
    private fun installModel(uri: Uri) {
        if (ModelInstall.isRunning) return
        binding.downloadButton.isEnabled = false
        binding.installButton.isEnabled = false
        binding.downloadProgress.isIndeterminate = true
        binding.downloadProgress.visibility = android.view.View.VISIBLE
        binding.modelStatus.setText(R.string.installing_model)

        download = lifecycleScope.launch {
            val result = runCatching { ModelManager(this@MainActivity).installFromArchive(uri) }
            binding.downloadProgress.visibility = android.view.View.GONE
            result.onFailure {
                binding.modelStatus.text = getString(
                    R.string.model_install_failed,
                    it.message ?: it.javaClass.simpleName,
                )
            }
            refresh()
        }
    }

    /**
     * Joins the shared install, rather than owning one.
     *
     * The download used to belong to this activity's lifecycleScope, so a
     * configuration change cancelled it - except the transfer has no suspension
     * point and carried on headless, while the recreated screen said "not
     * downloaded" and invited a second tap. Progress is observed instead, so
     * rotating mid-download shows the same download still going.
     */
    private fun startDownload() {
        if (ModelInstall.isRunning) return
        lifecycleScope.launch { runCatching { ModelInstall.ensure(this@MainActivity) } }
    }

    private fun observeInstall() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                ModelInstall.state.collect { state -> renderInstall(state) }
            }
        }
    }

    private fun renderInstall(state: ModelInstall.State) {
        val working = state is ModelInstall.State.Working
        binding.downloadProgress.visibility =
            if (working) android.view.View.VISIBLE else android.view.View.GONE
        binding.downloadButton.isEnabled = !working && !ModelManager(this).isModelInstalled
        binding.installButton.isEnabled = binding.downloadButton.isEnabled

        when (state) {
            is ModelInstall.State.Working -> {
                if (state.fraction < 0) {
                    binding.downloadProgress.isIndeterminate = true
                    binding.modelStatus.setText(R.string.downloading_unknown)
                } else {
                    binding.downloadProgress.isIndeterminate = false
                    binding.downloadProgress.progress = (state.fraction * 100).toInt()
                    binding.modelStatus.text =
                        getString(R.string.downloading, (state.fraction * 100).toInt())
                }
            }

            is ModelInstall.State.Failed ->
                binding.modelStatus.text = getString(R.string.error_generic, state.message)

            ModelInstall.State.Installed -> binding.modelStatus.setText(R.string.model_ready)

            ModelInstall.State.Idle -> refresh()
        }
    }

    private companion object {
        /** Enough of the report to recognise the failure without a wall of text. */
        const val CRASH_PREVIEW_LINES = 14
    }
}
