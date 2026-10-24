package org.pockettts.android.engine

import android.content.Context
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * The one place the model is installed from.
 *
 * There were two callers - the download button, and any read that reached
 * `PocketTts.get` before the model existed - and neither knew about the other.
 * They shared a partial-download file and a staging directory, so tapping
 * Download and then selecting text in another app had them writing the same
 * `.part` and deleting each other's unpacked files.
 *
 * The download also belonged to the activity that started it. A configuration
 * change cancelled the coroutine, but the transfer itself has no suspension
 * point, so it carried on headless while the recreated screen showed "not
 * downloaded" and invited a second tap - and the partial file was deleted on
 * the way out, so the second tap started from zero.
 *
 * So: one shared attempt per model, owned by the process. Everybody who asks
 * for a model joins the attempt in flight for it rather than starting
 * another, and a screen that goes away leaves it running. [state] follows the
 * latest attempt, which is the one the screen started.
 */
object ModelInstall {

    sealed interface State {
        data object Idle : State

        /**
         * [fraction] is 0..1, or -1 when the size is not known; [model] is the
         * model being installed, so a list of them can show which.
         */
        data class Working(val fraction: Float, val model: String = "") : State

        data object Installed : State

        data class Failed(val message: String) : State
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val lock = Mutex()
    /** By model name; guarded by [lock]. */
    private val attempts = HashMap<String, Deferred<ModelManager.ModelFiles>>()

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    val isRunning: Boolean get() = synchronized(attempts) { attempts.values.any { it.isActive } }

    /**
     * Installs the model if it is not already there, joining an attempt that is
     * already under way.
     *
     * Cancelling the caller does not cancel the download: it is shared, and one
     * screen navigating away is not a decision to abandon 219 MB that other
     * callers may still be waiting on.
     */
    suspend fun ensure(
        context: Context,
        modelName: String = Settings(context).modelName,
    ): ModelManager.ModelFiles {
        val app = context.applicationContext
        ModelManager(app, modelName).resolveModelOrNull()?.let {
            _state.value = State.Installed
            return it
        }

        val shared = lock.withLock {
            synchronized(attempts) {
                attempts[modelName]?.takeIf { it.isActive }
                    ?: scope.async { install(app, modelName) }.also { attempts[modelName] = it }
            }
        }
        return shared.await()
    }

    private suspend fun install(context: Context, modelName: String): ModelManager.ModelFiles {
        _state.value = State.Working(0f, modelName)
        return try {
            installer.install(context, modelName) { fraction ->
                _state.value = State.Working(fraction, modelName)
            }
                .also { _state.value = State.Installed }
        } catch (error: Throwable) {
            _state.value = State.Failed(error.message ?: error.javaClass.simpleName)
            throw error
        }
    }

    /** The model being installed, if any. */
    val installing: String? get() = (_state.value as? State.Working)?.model?.takeIf { isRunning }

    /**
     * Installs [modelName] and then makes it the one reads use, in the
     * process's scope rather than the caller's: leaving the screen that asked
     * must not leave the model downloaded but never chosen.
     */
    fun installAndSelect(context: Context, modelName: String) {
        val app = context.applicationContext
        scope.async {
            runCatching { ensure(app, modelName) }.onSuccess { Settings(app).modelName = modelName }
        }
    }

    /** How the model actually gets here. Swapped in tests, which have no network. */
    internal fun interface Installer {
        suspend fun install(
            context: Context,
            modelName: String,
            progress: ModelManager.ProgressListener?,
        ): ModelManager.ModelFiles
    }

    internal var installer = Installer { context, modelName, progress ->
        val manager = ModelManager(context, modelName)
        manager.ensureModel(progress).also {
            // A model with no voice cannot speak, and the first read may come
            // with no network; fetch the voice chosen for its language now.
            val settings = Settings(context)
            val voice = VoiceCatalog.byId(settings.voiceFor(manager.model.language)) ?: VoiceCatalog.default()
            manager.ensureEmbedding(voice)
        }
    }

    internal fun resetForTesting() {
        synchronized(attempts) { attempts.clear() }
        _state.value = State.Idle
        installer = Installer { context, modelName, progress ->
            ModelManager(context, modelName).ensureModel(progress)
        }
    }
}
