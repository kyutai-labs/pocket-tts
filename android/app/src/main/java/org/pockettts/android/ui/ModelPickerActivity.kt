package org.pockettts.android.ui

import android.content.Context
import android.net.ConnectivityManager
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.RadioButton
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.AccessibilityDelegateCompat
import androidx.core.view.ViewCompat
import androidx.core.view.accessibility.AccessibilityNodeInfoCompat
import androidx.core.view.accessibility.AccessibilityNodeInfoCompat.AccessibilityActionCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.pockettts.android.R
import org.pockettts.android.databinding.ActivityModelPickerBinding
import org.pockettts.android.databinding.ItemModelBinding
import org.pockettts.android.databinding.ItemModelHeaderBinding
import org.pockettts.android.databinding.ItemModelLanguageBinding
import org.pockettts.android.debug.Metrics
import org.pockettts.android.engine.ModelCatalog
import org.pockettts.android.engine.ModelInstall
import org.pockettts.android.engine.ModelManager
import org.pockettts.android.engine.PocketTts
import org.pockettts.android.engine.Settings
import org.pockettts.android.player.Reader

/**
 * Chooses the model reads are made with, and so the language: Kyutai's full
 * English model or one of the small ones, which start sooner, take less space
 * and keep up on older phones, or the model of another language.
 *
 * Every downloaded model is also used by the system engine for its own
 * language; the one chosen here is the one the app's own reads use.
 *
 * A model is downloaded only when asked, with its size shown first and a
 * warning on a metered connection, and is put to use once the download is
 * done, even if this screen has gone by then.
 */
class ModelPickerActivity : AppCompatActivity() {

    private lateinit var binding: ActivityModelPickerBinding
    private lateinit var settings: Settings
    private lateinit var adapter: ModelAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        binding = ActivityModelPickerBinding.inflate(layoutInflater)
        setContentView(binding.root)
        settings = Settings(this)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        binding.toolbar.setNavigationOnClickListener { finish() }
        Insets.apply(top = binding.appBar, bottom = binding.root)

        adapter = ModelAdapter(
            models = ModelCatalog.get(this).models,
            onChoose = ::choose,
            onDelete = ::confirmDelete,
            onHear = ::hear,
        )
        binding.modelList.layoutManager = LinearLayoutManager(this)
        binding.modelList.adapter = adapter
        refresh()

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                ModelInstall.state.collect { state ->
                    adapter.install = state
                    refresh()
                    if (state is ModelInstall.State.Failed) {
                        Toast.makeText(
                            this@ModelPickerActivity,
                            getString(R.string.error_generic, state.message),
                            Toast.LENGTH_LONG,
                        ).show()
                    }
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        refresh()
    }

    private fun refresh() {
        adapter.selected = settings.modelName
        adapter.installed = ModelCatalog.get(this).models
            .map { it.name }
            .filter { ModelManager(this, it).isModelInstalled }
            .toSet()
        adapter.measured = Metrics.generationRealTimeFactor
        adapter.notifyDataSetChanged()
    }

    private fun choose(model: String) {
        when {
            model == settings.modelName && model in adapter.installed -> return
            model in adapter.installed -> use(model)
            ModelInstall.installing == model -> return
            else -> confirmDownload(model)
        }
    }

    private fun use(model: String) {
        // The read in progress belongs to the old model: let it end now rather
        // than fail at its next chunk, once the old engine has been freed.
        Reader.stop()
        settings.modelName = model
        refresh()
        // A toast, so TalkBack says it too.
        val title = ModelNames.full(this, model)
        Toast.makeText(this, getString(R.string.model_now_in_use, title), Toast.LENGTH_SHORT).show()
        // Loaded now, so the next read does not wait for it. Frees the old one.
        val app = applicationContext
        lifecycleScope.launch(Dispatchers.IO) { runCatching { PocketTts.get(app) } }
    }

    private fun confirmDownload(model: String) {
        val title = ModelNames.full(this, model)
        val megabytes = megabytes(model)
        val metered = (getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager)
            .isActiveNetworkMetered
        MaterialAlertDialogBuilder(this)
            .setTitle(getString(R.string.model_download_title, title))
            .setMessage(
                getString(
                    if (metered) R.string.model_download_metered else R.string.model_download_body,
                    megabytes,
                ),
            )
            .setNegativeButton(R.string.cancel, null)
            .setPositiveButton(R.string.download) { _, _ ->
                ModelInstall.installAndSelect(this, model)
            }
            .show()
    }

    private fun confirmDelete(model: String) {
        val title = ModelNames.full(this, model)
        MaterialAlertDialogBuilder(this)
            .setTitle(getString(R.string.model_delete_title, title))
            .setMessage(getString(R.string.model_delete_body, megabytes(model)))
            .setNegativeButton(R.string.cancel, null)
            .setPositiveButton(R.string.delete) { _, _ ->
                // Never the model in use or one downloading: the button is not
                // offered for either, and this checks again rather than trust it.
                if (model != settings.modelName && ModelInstall.installing != model) {
                    ModelManager(this, model).deleteModel(model)
                    refresh()
                }
            }
            .show()
    }

    /** In the model's own language: Kyutai's sample sentence for it. */
    private fun hear() {
        val model = ModelCatalog.get(this).byName(settings.modelName)
        val text = if (model == null || model.language == Settings.ENGLISH) {
            getString(R.string.model_sample_text)
        } else {
            model.sampleText
        }
        Reader.speak(
            this,
            text,
            treatAsMarkdown = false,
            source = Reader.Source.ModelPicker,
        )
    }

    private fun megabytes(model: String): Int =
        ModelCatalog.get(this).byName(model)?.downloadMegabytes ?: 0

    private class ModelAdapter(
        models: List<ModelCatalog.Model>,
        private val onChoose: (String) -> Unit,
        private val onDelete: (String) -> Unit,
        private val onHear: () -> Unit,
    ) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

        var selected: String = ""
        var installed: Set<String> = emptySet()
        var install: ModelInstall.State = ModelInstall.State.Idle
        var measured: Float = 0f

        /** The list as shown: the explanation, then each language's heading and models. */
        private sealed interface Item {
            data object Intro : Item
            data class Language(val tag: String) : Item
            data class Row(val model: ModelCatalog.Model) : Item
        }

        private val items: List<Item> = listOf(Item.Intro) +
            models.groupBy { it.language }.flatMap { (tag, inLanguage) ->
                listOf(Item.Language(tag)) + inLanguage.map { Item.Row(it) }
            }

        class Header(val binding: ItemModelHeaderBinding) : RecyclerView.ViewHolder(binding.root)
        class LanguageHeading(val binding: ItemModelLanguageBinding) : RecyclerView.ViewHolder(binding.root)
        class Row(val binding: ItemModelBinding) : RecyclerView.ViewHolder(binding.root)

        override fun getItemCount() = items.size

        override fun getItemViewType(position: Int) = when (items[position]) {
            Item.Intro -> HEADER
            is Item.Language -> LANGUAGE
            is Item.Row -> ROW
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
            val inflater = LayoutInflater.from(parent.context)
            return when (viewType) {
                HEADER -> Header(ItemModelHeaderBinding.inflate(inflater, parent, false))
                LANGUAGE -> LanguageHeading(ItemModelLanguageBinding.inflate(inflater, parent, false)).also {
                    ViewCompat.setAccessibilityHeading(it.binding.root, true)
                }
                else -> Row(ItemModelBinding.inflate(inflater, parent, false))
            }
        }

        override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
            when (val item = items[position]) {
                Item.Intro -> bindHeader((holder as Header).binding)
                is Item.Language -> (holder as LanguageHeading).binding.language.text = ModelNames.language(item.tag)
                is Item.Row -> bindRow((holder as Row).binding, item.model)
            }
        }

        private fun bindHeader(binding: ItemModelHeaderBinding) {
            val context = binding.root.context
            binding.measured.visibility = if (measured > 0f) View.VISIBLE else View.GONE
            if (measured > 0f) {
                // Below 1.5x a phone has little to spare once playback, the
                // screen and everything else want the same cores.
                val text = if (measured < 1.5f) R.string.model_measured_slow else R.string.model_measured
                binding.measured.text = context.getString(text, measured)
            }
            // Hidden rather than disabled: a greyed button says nothing about
            // why, and the row below says the model is not downloaded.
            binding.hearButton.visibility = if (selected in installed) View.VISIBLE else View.GONE
            binding.hearButton.setOnClickListener { onHear() }
        }

        private fun bindRow(binding: ItemModelBinding, model: ModelCatalog.Model) {
            val context = binding.root.context
            val name = model.name
            val title = ModelNames.title(context, name)
            val megabytes = model.downloadMegabytes
            val working = (install as? ModelInstall.State.Working)?.takeIf { it.model == name }
            val isInstalled = name in installed
            val inUse = name == selected && isInstalled

            binding.title.text = title
            binding.detail.text = ModelNames.detail(context, name)
            binding.selected.isChecked = name == selected
            binding.state.text = when {
                working != null && working.fraction >= 0 ->
                    context.getString(R.string.model_state_downloading, (working.fraction * 100).toInt())
                working != null -> context.getString(R.string.model_state_downloading_unknown)
                inUse -> context.getString(R.string.model_state_in_use, megabytes)
                isInstalled -> context.getString(R.string.model_state_downloaded, megabytes)
                else -> context.getString(R.string.model_state_missing, megabytes)
            }
            binding.progress.visibility = if (working != null) View.VISIBLE else View.GONE
            if (working != null) {
                binding.progress.isIndeterminate = working.fraction < 0
                if (working.fraction >= 0) binding.progress.progress = (working.fraction * 100).toInt()
            }

            binding.row.setOnClickListener { onChoose(name) }
            // To TalkBack the row is one radio button: its name, what it is,
            // its state, and what a double tap will do.
            binding.row.contentDescription = listOf(title, binding.detail.text, binding.state.text)
                .filter { it.isNotBlank() }
                .joinToString(". ")
            ViewCompat.setAccessibilityDelegate(binding.row, object : AccessibilityDelegateCompat() {
                override fun onInitializeAccessibilityNodeInfo(host: View, info: AccessibilityNodeInfoCompat) {
                    super.onInitializeAccessibilityNodeInfo(host, info)
                    info.className = RadioButton::class.java.name
                    info.isCheckable = true
                    info.isChecked = name == selected
                }
            })
            ViewCompat.replaceAccessibilityAction(
                binding.row,
                AccessibilityActionCompat.ACTION_CLICK,
                context.getString(if (isInstalled) R.string.model_action_use else R.string.model_action_download),
                null,
            )

            val deletable = isInstalled && name != selected && working == null
            binding.delete.visibility = if (deletable) View.VISIBLE else View.GONE
            binding.delete.contentDescription = context.getString(R.string.model_delete, title)
            binding.delete.setOnClickListener { onDelete(name) }
        }

        private companion object {
            const val HEADER = 0
            const val LANGUAGE = 1
            const val ROW = 2
        }
    }
}
