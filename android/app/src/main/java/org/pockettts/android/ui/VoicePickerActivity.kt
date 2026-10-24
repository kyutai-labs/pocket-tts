package org.pockettts.android.ui

import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.launch
import org.pockettts.android.R
import org.pockettts.android.databinding.ActivityVoicePickerBinding
import org.pockettts.android.databinding.ItemVoiceBinding
import org.pockettts.android.engine.ModelManager
import org.pockettts.android.engine.Settings
import org.pockettts.android.engine.VoiceCatalog
import org.pockettts.android.player.VoiceSample
import java.util.Locale

/**
 * Picks the voice used for reading, and imports cloned ones.
 *
 * Also serves as the engine's settings screen: Android's text-to-speech
 * settings link here from the gear next to the engine name.
 */
class VoicePickerActivity : AppCompatActivity() {

    private lateinit var binding: ActivityVoicePickerBinding
    private lateinit var settings: Settings
    private lateinit var adapter: VoiceAdapter

    private val pickWav = registerForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri -> uri?.let { importVoice(it) } }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        binding = ActivityVoicePickerBinding.inflate(layoutInflater)
        setContentView(binding.root)
        settings = Settings(this)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.subtitle = getString(R.string.voices_reading_in, ModelNames.language(settings.readingLanguage))
        binding.toolbar.setNavigationOnClickListener { finish() }
        Insets.apply(top = binding.appBar, bottom = binding.root)

        adapter = VoiceAdapter(
            items = entries(),
            selectedId = settings.voiceId,
            onClick = { id ->
                settings.voiceId = id
                adapter.select(id)
            },
            onSample = { id ->
                if (!VoiceSample.toggle(this, id)) {
                    Toast.makeText(this, R.string.sample_while_reading, Toast.LENGTH_SHORT).show()
                }
            },
        )
        binding.voiceList.layoutManager = LinearLayoutManager(this)
        binding.voiceList.adapter = adapter
        observeSamples()

        // Hidden until cloning returns; see entries().
        binding.importButton.visibility = android.view.View.GONE
        binding.importButton.setOnClickListener {
            pickWav.launch(arrayOf("audio/wav", "audio/x-wav", "audio/*"))
        }
    }

    /**
     * A sample outlives this screen - it is a process-wide player, like the
     * reader - so it is stopped on the way out rather than left talking to an
     * empty room.
     */
    override fun onStop() {
        super.onStop()
        if (isFinishing) VoiceSample.stop()
    }

    private fun observeSamples() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                VoiceSample.state.collect { adapter.showSample(it) }
            }
        }
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                VoiceSample.errors.collect { message ->
                    Toast.makeText(
                        this@VoicePickerActivity,
                        getString(R.string.sample_failed, message),
                        Toast.LENGTH_LONG,
                    ).show()
                }
            }
        }
    }

    private data class Entry(val id: String, val name: String, val detail: String)

    /**
     * Every voice reads the reading language; its native speakers come first,
     * Kyutai's own voice for it at the top. Each says its accent, because Alba
     * reading French is a British speaker reading French.
     */
    private fun entries(): List<Entry> {
        val language = settings.readingLanguage
        val default = settings.defaultVoiceFor(language)
        val stock = VoiceCatalog.voices
            .sortedBy { voice ->
                when {
                    voice.id == default -> 0
                    voice.nativeLanguage == language -> 1
                    else -> 2
                }
            }
            .map { voice ->
                val accent = Locale.forLanguageTag(voice.language).displayName
                Entry(
                    voice.id,
                    voice.displayName,
                    if (voice.id == default) {
                        getString(R.string.voice_default_for, accent, ModelNames.language(language))
                    } else {
                        accent
                    },
                )
            }
        // Imported voices are not offered while the engine cannot clone
        // (PocketTts.loadVoiceFile); their files are kept for when it can.
        return stock
    }

    private fun importVoice(uri: Uri) {
        val result = runCatching {
            val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
                ?: error("Could not open that file")
            // The manager decides the final id, because it is the thing that
            // knows what is already taken - a name that collides is suffixed
            // rather than allowed to overwrite a voice that cannot be re-fetched.
            ModelManager(this).importVoice(displayNameOf(uri), bytes).nameWithoutExtension
        }

        result.onSuccess { name ->
            Toast.makeText(this, getString(R.string.imported_voice, name), Toast.LENGTH_SHORT).show()
            settings.voiceId = name
            adapter.replace(entries(), name)
        }.onFailure { error ->
            Toast.makeText(
                this,
                getString(R.string.voice_import_failed, error.message ?: ""),
                Toast.LENGTH_LONG,
            ).show()
        }
    }

    /**
     * A name a person would recognise.
     *
     * The last path segment of a document URI is usually a numeric id, so a
     * file picked from Downloads arrived in the list called something like
     * "1234". The provider knows its real name; ask for it.
     */
    private fun displayNameOf(uri: Uri): String {
        val fromProvider = runCatching {
            contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
        }.getOrNull()

        return (fromProvider ?: uri.lastPathSegment ?: "voice")
            .substringAfterLast('/')
            .substringBeforeLast('.')
            .replace(Regex("""[^A-Za-z0-9_-]"""), "_")
            .trim('_')
            .ifBlank { "voice" }
    }

    private class VoiceAdapter(
        private var items: List<Entry>,
        private var selectedId: String,
        private val onClick: (String) -> Unit,
        private val onSample: (String) -> Unit,
    ) : RecyclerView.Adapter<VoiceAdapter.Holder>() {

        private var sample: VoiceSample.State = VoiceSample.State.Idle

        class Holder(val binding: ItemVoiceBinding) : RecyclerView.ViewHolder(binding.root)

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) = Holder(
            ItemVoiceBinding.inflate(LayoutInflater.from(parent.context), parent, false),
        )

        override fun getItemCount() = items.size

        override fun onBindViewHolder(holder: Holder, position: Int) {
            val item = items[position]
            holder.binding.name.text = item.name
            holder.binding.language.text = item.detail
            holder.binding.selected.isChecked = item.id == selectedId
            holder.binding.root.setOnClickListener { onClick(item.id) }
            holder.binding.sample.setOnClickListener { onSample(item.id) }
            bindSample(holder, item.id)
        }

        private fun bindSample(holder: Holder, id: String) {
            val loading = (sample as? VoiceSample.State.Loading)?.voiceId == id
            val playing = (sample as? VoiceSample.State.Playing)?.voiceId == id
            holder.binding.sampleProgress.visibility = if (loading) View.VISIBLE else View.GONE
            holder.binding.sample.visibility = if (loading) View.INVISIBLE else View.VISIBLE
            holder.binding.sample.setImageResource(
                if (playing) R.drawable.ic_stop else R.drawable.ic_play,
            )
            holder.binding.sample.contentDescription = holder.itemView.context.getString(
                if (playing) R.string.stop_sample else R.string.play_sample,
            )
        }

        fun showSample(state: VoiceSample.State) {
            if (state == sample) return
            sample = state
            notifyDataSetChanged()
        }

        fun select(id: String) {
            selectedId = id
            notifyDataSetChanged()
        }

        fun replace(newItems: List<Entry>, id: String) {
            items = newItems
            selectedId = id
            notifyDataSetChanged()
        }
    }
}
