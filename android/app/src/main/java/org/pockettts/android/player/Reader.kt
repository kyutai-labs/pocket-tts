package org.pockettts.android.player

import android.content.Context
import android.util.Log
import androidx.annotation.VisibleForTesting
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.job
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import org.pockettts.android.debug.Metrics
import org.pockettts.android.engine.EngineTurn
import org.pockettts.android.engine.Settings
import org.pockettts.android.speech.MarkdownSpeech
import org.pockettts.android.speech.TextChunker
import kotlin.coroutines.cancellation.CancellationException
import kotlin.coroutines.coroutineContext

/**
 * Reads a body of text aloud, one chunk at a time.
 *
 * There is one reader for the whole process: starting a new utterance stops
 * whatever was playing, which is what you want when you select a second
 * paragraph while the first is still being read.
 *
 * Everything needed to *keep* reading survives the current chunk - the chunked
 * text, the loaded voice, the engine - so skipping back a sentence costs a
 * re-synthesis of that sentence and nothing else. No re-selection, no reload.
 */
object Reader {

    private const val TAG = "Reader"

    /**
     * Who asked for this utterance.
     *
     * The reader is process-wide, so a screen that wants to show its own
     * controls has to know whether the current read is *its* read. This used to
     * be a boolean on each screen, set in the click handler and cleared when
     * the state went Idle - which broke every time the state passed through
     * Idle on its way from one utterance to the next.
     */
    enum class Source(
        /**
         * How this source is named in [VoiceTrace].
         *
         * The trace called every in-app read "[reader]" regardless of who asked,
         * which made it useless for the one question it was reached for: whether
         * the read under investigation was the broadcast or the scratchpad.
         */
        val traceName: String,
    ) {
        /** Text selected in another app, or shared in. */
        Selection("selection"),

        /** The Speak button, or a selection inside the scratchpad. */
        Scratchpad("scratchpad"),

        /** Another app driving the system text-to-speech engine. */
        System("system"),

        /**
         * A coding agent or script, over [org.pockettts.android.SpeakReceiver].
         *
         * Neither a paste nor a selection: nobody was looking at this text in
         * an editor, and there is no screen anywhere that owns the read. It is
         * told apart from the other two because a read with no window behind
         * it fails differently - silently, with the phone in a pocket - and a
         * trace that calls it a selection sends the search somewhere there is
         * nothing to find.
         */
        Agent("agent"),

        /** "Hear the current model", on the model picker. */
        ModelPicker("model-picker"),
    }

    sealed interface State {

        /**
         * Which utterance this state belongs to, or 0 before there is one.
         *
         * A screen that started a read has to be able to tell that read's
         * ending from the ending of some earlier one it never asked for. The
         * floating window did not, and would close itself on a terminal state
         * left over from a previous utterance before its own had begun.
         */
        val utterance: Long

        /** Who asked for this utterance; null only before anything is asked for. */
        val source: Source?

        /** True once the utterance is over, however it ended. */
        val isTerminal: Boolean get() = false

        /**
         * Nothing has been asked for.
         *
         * Only ever the *starting* state. An utterance that ends reaches
         * [Finished], [Stopped] or [Failed] - all distinguishable from "not
         * started yet", which is the distinction three separate
         * `readingBegan`-style flags used to approximate, and get wrong every
         * time the state passed through Idle between two utterances.
         */
        data object Idle : State {
            override val utterance: Long get() = 0
            override val source: Source? get() = null
        }

        /** Model or voice is being fetched. [fraction] is -1 when size is unknown. */
        data class Preparing(
            override val utterance: Long,
            override val source: Source,
            val fraction: Float,
        ) : State

        data class Speaking(
            override val utterance: Long,
            override val source: Source,
            val chunkIndex: Int,
            val chunkCount: Int,
            /** Character range of the chunk within the speakable text. */
            val start: Int,
            val end: Int,
            val paused: Boolean,
            /**
             * Whether any sound has come out yet.
             *
             * The reader reaches this state when it starts *working on* a
             * chunk, and the model composes a whole sentence before it emits a
             * single sample. Reporting "Reading aloud" from the first moment
             * meant the app claimed to be reading through the several seconds
             * where it was silent - which is most of what the wait before the
             * first word actually feels like.
             */
            val audible: Boolean,
        ) : State

        /** Read to the end. */
        data class Finished(
            override val utterance: Long,
            override val source: Source,
        ) : State {
            override val isTerminal: Boolean get() = true
        }

        /** Ended early, by the user or by another utterance taking over. */
        data class Stopped(
            override val utterance: Long,
            override val source: Source,
        ) : State {
            override val isTerminal: Boolean get() = true
        }

        data class Failed(
            override val utterance: Long,
            override val source: Source,
            val message: String,
        ) : State {
            override val isTerminal: Boolean get() = true
        }
    }

    private val scope = CoroutineScope(SupervisorJob())

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    /** The text currently being read, after Markdown stripping. */
    @Volatile
    var speakableText: String = ""
        private set

    /**
     * Where the spoken text came from, for a screen that wants to follow along.
     *
     * A [State.Speaking] reports offsets into [speakableText], and stripping
     * deleted syntax, so those offsets are not positions in the document the
     * user is looking at. This turns one into the other.
     */
    @Volatile
    private var sourceMap: MarkdownSpeech.Speakable? = null

    /**
     * The range of [source] currently being spoken, or null if it cannot be
     * placed - because nothing is being read, or because [source] is no longer
     * the text that was read.
     */
    fun spokenRangeIn(source: String): IntRange? {
        val speaking = _state.value as? State.Speaking ?: return null
        val map = sourceMap ?: return null
        return map.sourceRange(speaking.start, speaking.end, source)
    }

    private val control = Mutex()
    private var job: Job? = null
    private var player: AudioSink? = null

    /**
     * The current utterance, minus its position.
     *
     * Held so that skipping does not repeat the expensive half of the work:
     * the engine is loaded, the voice is loaded, the Markdown is stripped and
     * the text is chunked exactly once per utterance.
     */
    private class Utterance(
        val id: Long,
        /** This utterance's claim on the engine; see [EngineTurn]. */
        val turn: Long,
        val source: Source,
        val speakable: String,
        val chunks: List<TextChunker.Chunk>,
        val engine: SpeechEngine,
        val speed: Float,
    )

    @Volatile
    private var utterance: Utterance? = null

    /**
     * An utterance that has been asked for but not started.
     *
     * Everything needed to begin a read, held until the one in front of it
     * ends. Deliberately the arguments to [run] rather than an [Utterance]:
     * the engine, the voice and the chunking are the expensive half, and
     * doing them at queue time would load a model for a read that a stop
     * arriving first means nobody will ever hear.
     */
    private class Pending(
        val id: Long,
        val context: Context,
        val text: String,
        val treatAsMarkdown: Boolean,
        val voiceOverride: String?,
        val source: Source,
    )

    /**
     * Reads waiting their turn. Only ever touched under [control].
     *
     * [EngineTurn] is not this. That arbitrates between the two things that
     * drive the one model - this reader and the system engine service - and
     * its rule is that the most recent request wins, which is the opposite of
     * a queue. It exists so two callers who cannot see each other do not
     * interleave; this exists so one caller can say "after that one", and
     * nothing about standing down for a stranger answers that.
     */
    private val queue = ArrayDeque<Pending>()

    /**
     * How many reads are waiting their turn.
     *
     * Read without [control], which is safe for what it is used for: a test
     * waiting for an enqueue to have landed before it does the thing it is
     * actually testing. Without it every queue test races the dispatch of the
     * coroutine that does the enqueuing, and races that are usually won are
     * the ones that fail in CI.
     */
    @VisibleForTesting
    internal val queued: Int get() = queue.size

    @Volatile
    private var chunkIndex: Int = 0

    /**
     * The source of the utterance currently being worked on.
     *
     * Tracked separately from [state] because it is known the moment `speak`
     * is called, while the state only catches up once the text has been
     * stripped - and a stop arriving in that window still has to be attributed
     * to the right screen.
     */
    @Volatile
    private var currentSource: Source? = null

    private val nextUtterance = AtomicLong(0)

    @Volatile
    private var currentUtterance: Long = 0

    val isActive: Boolean get() = job?.isActive == true

    /**
     * How the reader obtains an engine and a sink.
     *
     * Swapped in tests for fakes. Without this the reader can only be exercised
     * with a 125 MB model and real audio hardware, which is why its state
     * machine - the part that has repeatedly been wrong - had no tests, and why
     * `ActivityLaunchTest` was quietly downloading the model on every run.
     */
    @VisibleForTesting
    var engines: SpeechEngine.Factory = PocketTtsEngine

    @VisibleForTesting
    var sinks: AudioSink.Factory = StreamingPlayer

    /**
     * Returns the reader to its just-started state.
     *
     * Only for tests: this is a process-wide singleton, so without it one
     * test's terminal state is the next test's starting state.
     */
    @VisibleForTesting
    fun resetForTesting() {
        runBlocking {
            // The queued launches as well as the read in flight. speak() and
            // skip() return before their coroutine has taken the control lock,
            // so a read asked for by a test that ended early would otherwise
            // begin during the next one - against its engine, its sink and its
            // assertions. That is a whole class of confusing cross-test
            // failures, and it does not belong to the test that reports it.
            scope.coroutineContext.job.children.toList().forEach { it.cancelAndJoin() }
            control.withLock {
                queue.clear()
                stopLocked(ending = null)
            }
        }
        utterance = null
        currentSource = null
        currentUtterance = 0
        chunkIndex = 0
        speakableText = ""
        sourceMap = null
        _state.value = State.Idle
    }

    /**
     * Starts reading [text], stopping whatever was playing.
     *
     * @return the utterance id, so the caller can recognise its own read among
     *   the states of a reader the whole process shares.
     */
    fun speak(
        context: Context,
        text: String,
        treatAsMarkdown: Boolean = true,
        voiceOverride: String? = null,
        source: Source = Source.Selection,
    ): Long {
        val appContext = context.applicationContext
        val id = nextUtterance.incrementAndGet()
        // A voice sample auditioning in the picker must not talk over the thing
        // the user actually asked to hear.
        VoiceSample.stop()
        scope.launch {
            control.withLock {
                // A read that replaces what is playing replaces what was going
                // to play after it too. Queued follow-ups belong to the passage
                // being abandoned, not to the one taking over.
                queue.clear()
                // No terminal state on the way out: this is a handover, not an
                // ending, and anything watching for "the read is over" would
                // otherwise tear down and immediately rebuild.
                stopLocked(ending = null)
                startLocked(Pending(id, appContext, text, treatAsMarkdown, voiceOverride, source))
            }
        }
        return id
    }

    /**
     * Reads [text] after whatever is playing, rather than instead of it.
     *
     * For a caller sending a series of things to hear in order - an agent
     * handing over one reply while the last is still being read. [speak] would
     * make each one silence the one before, so a run of them plays only the
     * last few words of the last.
     *
     * With nothing playing this is exactly [speak]; there is no queue to join.
     *
     * @return the utterance id, which is allocated now even though the read
     *   starts later, so a caller can recognise its own read when it arrives.
     */
    fun enqueue(
        context: Context,
        text: String,
        treatAsMarkdown: Boolean = true,
        voiceOverride: String? = null,
        source: Source = Source.Selection,
    ): Long {
        val appContext = context.applicationContext
        val id = nextUtterance.incrementAndGet()
        scope.launch {
            control.withLock {
                val pending = Pending(id, appContext, text, treatAsMarkdown, voiceOverride, source)
                if (job?.isActive == true || queue.isNotEmpty()) {
                    queue.addLast(pending)
                } else {
                    VoiceSample.stop()
                    startLocked(pending)
                }
            }
        }
        return id
    }

    /**
     * Begins [pending]. The caller holds [control] and has stopped whatever
     * was playing.
     */
    private fun startLocked(pending: Pending) {
        currentSource = pending.source
        currentUtterance = pending.id
        val started = scope.launch {
            run(
                pending.context,
                pending.id,
                pending.text,
                pending.treatAsMarkdown,
                pending.voiceOverride,
                pending.source,
            )
        }
        job = started
        // Advancing the queue from inside the job would deadlock: a stop holds
        // `control` while it joins this very job, so the job cannot take the
        // lock it would need. A completion handler runs after the job is done
        // and the joiner has been released.
        //
        // A non-null cause means cancelled, and a cancel is either a handover
        // or a stop - both of which have already decided what plays next.
        started.invokeOnCompletion { cause ->
            if (cause == null) scope.launch { advance(started) }
        }
    }

    /** Starts whatever [finished] was holding up. */
    private suspend fun advance(finished: Job) {
        control.withLock {
            // Somebody took over while the completion handler was in flight;
            // the queue is theirs now.
            if (job !== finished) return@withLock
            // A failure here is almost never about the text - no model, no
            // voice, no network - so the rest of the queue would fail the same
            // way. Four more reads that each say nothing but the same error is
            // not a service to anyone.
            if (_state.value is State.Failed) {
                queue.clear()
                return@withLock
            }
            job = null
            advanceLocked()
        }
    }

    /** Starts the next queued read, if there is one. Caller holds [control]. */
    private fun advanceLocked() {
        startLocked(queue.removeFirstOrNull() ?: return)
    }

    /**
     * Moves [delta] chunks - roughly sentences - and carries on reading.
     *
     * Skipping back past the start replays the first chunk; skipping forward
     * past the end finishes the utterance, which is what a listener who keeps
     * tapping forward means by it.
     */
    fun skip(delta: Int) {
        if (delta == 0) return
        scope.launch {
            control.withLock {
                val current = utterance ?: return@withLock
                val target = chunkIndex + delta
                if (target >= current.chunks.size) {
                    stopLocked(ending = State.Finished(current.id, current.source))
                    // Skipping off the end is an ending like any other, and
                    // what was queued behind it is still wanted. The cancel
                    // inside stopLocked means the completion handler will not
                    // do this for us.
                    advanceLocked()
                    return@withLock
                }
                stopLocked(ending = null)
                val from = target.coerceAtLeast(0)
                job = scope.launch { play(current, from) }
            }
        }
    }

    fun skipForward() = skip(1)

    fun skipBack() = skip(-1)

    private suspend fun run(
        context: Context,
        id: Long,
        text: String,
        treatAsMarkdown: Boolean,
        voiceOverride: String?,
        source: Source,
    ) {
        val settings = Settings(context)
        val mapped = if (treatAsMarkdown) {
            MarkdownSpeech.toSpeakableWithSource(
                text,
                MarkdownSpeech.Options(
                    speakCodeBlocks = settings.speakCodeBlocks,
                    language = settings.readingLanguage,
                ),
            )
        } else {
            // Nothing was stripped, so the speakable text is the source and the
            // offsets need no translating.
            val trimmed = text.trim()
            val offset = text.indexOf(trimmed).coerceAtLeast(0)
            MarkdownSpeech.Speakable(
                trimmed,
                listOf(MarkdownSpeech.Span(0, trimmed.length, offset, offset + trimmed.length)),
            )
        }
        val speakable = mapped.text
        speakableText = speakable
        sourceMap = mapped

        if (speakable.isBlank()) {
            _state.value = State.Finished(id, source)
            return
        }

        val askedAt = System.currentTimeMillis()
        try {
            _state.value = State.Preparing(id, source, 0f)
            val engine = engines.create(context) { fraction ->
                _state.value = State.Preparing(id, source, fraction)
            }
            engine.useVoice(voiceOverride ?: settings.voiceId, source.traceName)
            val prepared = Utterance(
                id = id,
                turn = EngineTurn.take(),
                source = source,
                speakable = speakable,
                chunks = TextChunker.chunk(speakable),
                engine = engine,
                speed = settings.speed,
            )
            utterance = prepared
            play(prepared, from = 0, askedAt = askedAt)
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Throwable) {
            Log.e(TAG, "Reading failed", error)
            _state.value = State.Failed(id, source, error.message ?: error.javaClass.simpleName)
        }
    }

    /**
     * One piece of a chunk's audio on its way from the engine to the sink;
     * the end of a chunk when [samples] is null.
     */
    private class Piece(val chunk: Int, val samples: FloatArray?)

    /**
     * Reads [current] from chunk [from] to the end.
     *
     * Synthesis runs ahead of playback. The engine composes a whole chunk
     * before it emits a sample - every frame of the language model, then the
     * decoder - so a chunk's audio arrives all at once, after a silence as
     * long as the model took. Feeding the sink from inside the engine's
     * callback, as this used to, meant the next chunk could not begin until
     * this one had been *heard*: the blocking write held the callback, the
     * callback held the engine. The silence before every chunk was the model's
     * full pass over it, less whatever the sink had buffered, and on a phone
     * that was several seconds between paragraphs, charged for a table that
     * stripped to one line just as for a long one.
     *
     * Now a producer hands pieces to a channel and a consumer writes them,
     * and the producer may be [LOOKAHEAD_CHUNKS] chunks ahead of the one
     * being written. The model's pass over chunk N+1 happens while chunk N
     * plays, and as long as the model is faster than real time - it is, by a
     * comfortable margin on the device this was measured on - the gap is
     * gone. Memory is bounded by the lookahead: a chunk is at most a few
     * hundred characters, under thirty seconds of float audio.
     *
     * What is published follows the *listener*: [State.Speaking.chunkIndex]
     * is the chunk being written to the sink, not the one being composed,
     * so highlighting and skipping stay true to what is being heard.
     *
     * [askedAt] is when speech was asked for, so the wait before the first
     * sound can be measured rather than guessed at. A skip passes null: it
     * resumes an utterance whose first-audio time is already known.
     */
    private suspend fun play(current: Utterance, from: Int, askedAt: Long? = null) {
        if (from >= current.chunks.size) {
            _state.value = State.Finished(current.id, current.source)
            return
        }
        var localPlayer: AudioSink? = null
        try {
            localPlayer = sinks.create(current.engine.sampleRate).also {
                player = it
                it.start()
            }
            val sink: AudioSink = localPlayer

            // The sink refusing audio means it was stopped under us, which is
            // an interruption and not an ending: whoever stopped it owns the
            // state that follows. Publishing Finished here would put a
            // terminal state on the handover between two utterances - the same
            // bug the Idle-means-two-things design had, wearing a new hat.
            val interrupted = AtomicBoolean(false)
            var audible = false
            chunkIndex = from
            _state.value = speakingState(current, from, paused = sink.isPaused, audible = false)

            // Unbounded on purpose: the engine's callback cannot suspend, and a
            // callback that blocks waiting for room would hold the engine
            // exactly the way the blocking write used to - and, worse, would
            // have nothing to wake it once a stop had cancelled the consumer.
            // The bound is on chunks instead, below.
            val pieces = Channel<Piece>(Channel.UNLIMITED)
            // One permit per chunk the producer may be working on or holding
            // finished: the one being written plus the lookahead. Released as
            // the consumer finishes each chunk.
            val permits = Semaphore(LOOKAHEAD_CHUNKS + 1)

            coroutineScope {
                val producer = launch(Dispatchers.Default) {
                    val self = coroutineContext.job
                    try {
                        for (index in from until current.chunks.size) {
                            permits.acquire()
                            if (EngineTurn.superseded(current.turn)) {
                                interrupted.set(true)
                                break
                            }
                            val chunk = current.chunks[index]
                            val startedAt = System.currentTimeMillis()
                            var samplesProduced = 0L
                            val spoke = current.engine.synthesize(chunk.speech, current.speed) { samples ->
                                // The engine only looks at this between pieces,
                                // so a stop is felt within one piece rather than
                                // at the end of the chunk.
                                if (!self.isActive) return@synthesize false
                                // Checked per callback as well as per chunk: a
                                // request arriving mid-sentence should not have
                                // to wait out the rest of it.
                                if (EngineTurn.superseded(current.turn)) {
                                    interrupted.set(true)
                                    return@synthesize false
                                }
                                samplesProduced += samples.size
                                pieces.trySend(Piece(index, samples))
                                true
                            }
                            if (index == from) {
                                // The engine on its own, now that nothing in the
                                // callback waits on the speaker: audio seconds
                                // composed per wall second.
                                val elapsed = System.currentTimeMillis() - startedAt
                                if (elapsed > 0) {
                                    Metrics.generationRealTimeFactor =
                                        samplesProduced.toFloat() / current.engine.sampleRate / (elapsed / 1000f)
                                }
                            }
                            if (!spoke) {
                                interrupted.set(true)
                                break
                            }
                            pieces.trySend(Piece(index, null))
                        }
                        pieces.close()
                    } catch (cancelled: CancellationException) {
                        pieces.close()
                        throw cancelled
                    } catch (error: Throwable) {
                        // Delivered to the consumer in order, after the audio
                        // that was composed before it - so a failure on chunk
                        // three does not cut off chunk two mid-word.
                        pieces.close(error)
                    }
                }

                withContext(Dispatchers.IO) {
                    var writing = from
                    for (piece in pieces) {
                        coroutineContext.ensureActive()
                        if (piece.chunk != writing) {
                            writing = piece.chunk
                            chunkIndex = writing
                            _state.value = speakingState(current, writing, paused = sink.isPaused, audible = audible)
                        }
                        val samples = piece.samples
                        if (samples == null) {
                            if (!sink.writeSilence(current.chunks[piece.chunk].trailingPauseSeconds)) {
                                interrupted.set(true)
                                break
                            }
                            permits.release()
                            continue
                        }
                        if (EngineTurn.superseded(current.turn)) {
                            interrupted.set(true)
                            break
                        }
                        if (!sink.write(samples)) {
                            interrupted.set(true)
                            break
                        }
                        if (!audible) {
                            audible = true
                            askedAt?.let { Metrics.timeToFirstAudioMillis = System.currentTimeMillis() - it }
                            // Said once, when it becomes true.
                            (_state.value as? State.Speaking)
                                ?.takeIf { it.utterance == current.id }
                                ?.let { _state.value = it.copy(audible = true) }
                        }
                    }
                    // Nothing more will be heard, so nothing more should be
                    // composed; without this the scope would wait for the
                    // producer to finish every remaining chunk into a channel
                    // nobody reads.
                    if (interrupted.get()) producer.cancel()
                }
            }

            sink.drain()
            when {
                !interrupted.get() -> _state.value = State.Finished(current.id, current.source)
                // Losing the engine to a more recent request is an ending this
                // read has to own; a sink stopped from inside belongs to
                // whoever stopped it, and they publish their own state.
                EngineTurn.superseded(current.turn) ->
                    _state.value = State.Stopped(current.id, current.source)
            }
        } catch (cancelled: CancellationException) {
            // The state belongs to whoever cancelled us - a handover leaves it
            // alone, an explicit stop sets Stopped.
            throw cancelled
        } catch (error: Throwable) {
            Log.e(TAG, "Reading failed", error)
            _state.value = State.Failed(current.id, current.source, error.message ?: error.javaClass.simpleName)
        } finally {
            localPlayer?.let { Metrics.underruns = it.underruns }
            localPlayer?.release()
            if (player === localPlayer) player = null
        }
    }

    private fun speakingState(current: Utterance, index: Int, paused: Boolean, audible: Boolean): State.Speaking {
        val chunk = current.chunks[index]
        return State.Speaking(
            utterance = current.id,
            source = current.source,
            chunkIndex = index,
            chunkCount = current.chunks.size,
            start = chunk.start,
            end = chunk.end,
            paused = paused,
            audible = audible,
        )
    }

    /**
     * How many chunks synthesis may run ahead of the one being played.
     *
     * One is enough to hide the model's pass over the next chunk behind the
     * current one, which is the whole point; more would only add memory and
     * lengthen the work thrown away by a stop or a skip.
     */
    @VisibleForTesting
    internal const val LOOKAHEAD_CHUNKS = 1

    fun pause() {
        player?.pause()
        (_state.value as? State.Speaking)?.let { _state.value = it.copy(paused = true) }
    }

    fun resume() {
        player?.resume()
        (_state.value as? State.Speaking)?.let { _state.value = it.copy(paused = false) }
    }

    fun togglePause() {
        if (player?.isPaused == true) resume() else pause()
    }

    fun stop() {
        scope.launch {
            control.withLock {
                // Cleared before the early return: a stop with nothing playing
                // still means "and nothing after it either".
                queue.clear()
                val source = currentSource ?: return@withLock
                stopLocked(ending = State.Stopped(currentUtterance, source))
            }
        }
    }

    /**
     * Stops playback and waits for the reading coroutine to unwind.
     *
     * [ending] is the state to publish afterwards, or null when another
     * utterance is about to take over and will publish its own.
     */
    private suspend fun stopLocked(ending: State?) {
        // Stopping the player first unblocks any write that is parked waiting
        // for buffer space, so the cancel below does not have to wait it out.
        player?.stop()
        job?.cancelAndJoin()
        job = null
        if (ending != null) {
            utterance = null
            currentSource = null
            currentUtterance = 0
            chunkIndex = 0
            _state.value = ending
        }
    }
}
