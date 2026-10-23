# Pocket TTS for Android

A native Android app that reads text aloud in Pocket TTS voices, entirely on the
device. No server, no network at speaking time.

It exists to remove a specific papercut: wanting to hear a paragraph in a good
voice and having to paste it into a notes app or a pastebin to get there.

## What it gives you

**A "Read aloud" item in the tap-and-hold text selection menu.** Select text
anywhere - a browser, a chat app, a PDF reader - and Pocket TTS appears in the
selection toolbar next to Copy and Share. Tapping it starts reading immediately
in a small floating window with pause and stop. This is an
`ACTION_PROCESS_TEXT` activity; apps that draw their own selection toolbar and
suppress the system one will not show it, and for those there is a share-sheet
entry as well.

**A system-wide text-to-speech engine.** Once Pocket TTS is selected under
*Settings → Accessibility → Text-to-speech output*, every app that already knows
how to read text uses these voices: Select to Speak, Chrome's read-aloud, ebook
readers, accessibility tools. None of them need to know this app exists. Audio
is streamed back chunk by chunk as it is generated, so playback starts after the
first sentence rather than the last one, and each chunk is announced with
`rangeStart` so callers that highlight along can.

Text arriving through the engine is **not** rewritten. It was chosen by another
app and is nearly always prose already; running it through the Markdown stripper
took asterisks out of ordinary sentences, reshaped `snake_case` identifiers, and
replaced links with nothing at all. An accessibility tool exists to say what is
on the screen, and quietly deleting a URL from that is a false account of the
content rather than a formatting choice. It is also what makes the offsets
meaningful: strip the text and they no longer point anywhere the caller could
highlight. The app's own screens still strip, because there the Markdown is the
user's own and reading it as prose is the point.

**A read-aloud broadcast, for scripts and agents.** `am broadcast -n
org.pockettts.android/.SpeakReceiver --es text "..."` reads text with no window
and no foreground app, which is the one thing the selection window cannot do -
an activity start needs a foreground caller. It was built for a coding agent on
the same phone: a reply finishes and it is read, rather than copied, switched to
and pasted. See [reading Claude Code's replies aloud](docs/read-aloud-from-claude-code.md).

**A Markdown scratchpad.** Type or paste, hit speak. Markdown is stripped for
speaking, so `## Heading` is spoken as "Heading" rather than "hash hash
Heading", `**bold**` loses its asterisks, links are read as their text instead
of spelling out a URL, and code fences are skipped by default. Selecting a
passage in the editor and hitting speak reads just that passage, and the
sentence being spoken is highlighted as it goes.

**Any Pocket TTS voice, including your own.** Pocket TTS has no fixed speaker
table - a voice is a few seconds of reference audio the model conditions on - so
the stock voices and a wav you record yourself are the same kind of thing. The
voice picker lists the catalogue from
[`kyutai/tts-voices`](https://huggingface.co/kyutai/tts-voices) and can import a
local wav. Each row plays a short sample, because twenty one names tell you
nothing about which one you want.

The sample is the voice's own reference audio rather than a synthesised
sentence. That is the most direct answer to "what does this sound like": the
reference audio *is* the voice. It also costs nothing but the prompt download -
no model bundle, no inference, no wait - so voices can be auditioned on a fresh
install before the model has ever been fetched, and the download it does
is the same one selecting that voice would trigger later. The trade-off worth
knowing: it is the raw prompt, so it carries the recording's own room and
pacing. The synthesised voice tracks its timbre closely, which is the thing
being chosen, but not its background.

## Installing

Every build of `main` that touches the app publishes its release APK as a
GitHub release. The newest is always at
<https://github.com/mycosavant/pocket-tts/releases/latest/download/pocket-tts.apk>;
open it on the phone and allow the install. Every APK is signed with the same
committed key, so a new one installs over an old one and keeps its data.

## Building

Requires JDK 17+ and the Android SDK (compileSdk 35, build-tools 35.0.0).

```bash
cd android
echo "sdk.dir=$ANDROID_HOME" > local.properties
./gradlew :app:assembleDebug        # or: gradle :app:assembleDebug
./gradlew :app:testDebugUnitTest
```

The build compiles the engine from `../pocket-speak` first (`cargo ndk`, the
`buildNative` task), so it needs rustup's `aarch64-linux-android`,
`armv7-linux-androideabi` and `x86_64-linux-android` targets, `cargo-ndk` and
the NDK. Pass `-PskipNative` to package the libraries already in
`app/src/main/jniLibs`.

The debug APK lands in `app/build/outputs/apk/debug/`. It is about 28 MB (the
release APK 23 MB), most of it the engine for `arm64-v8a`, `armeabi-v7a` and
`x86_64`, 5 to 8 MB each. Dropping ABIs from `abiFilters` in
`app/build.gradle.kts` shrinks it. Until 2026-10 it was 85 MB, most of it ONNX
Runtime; see `docs/pure-rust-engine.md`.

## The debug key is committed, on purpose

`app/debug.keystore` is in the repository, and both build types are signed with
it. That is the opposite of the usual advice, and the usual advice is about a
key that ships an app to a store. This one is not that.

Left alone, the Android plugin signs debug builds with `~/.android/debug.keystore`
and generates one if it is missing. A CI runner has no home directory worth the
name, so every run generated a fresh key pair, and every APK this project ever
published was a different app as far as Android is concerned - identity is the
signing certificate, not the package name. The consequences were permanent
rather than intermittent:

- no build could be installed over another, ever: `INSTALL_FAILED_UPDATE_INCOMPATIBLE`,
- keeping app data on uninstall - which `hasFragileUserData` exists to offer, and
  which is the only thing standing between an uninstall and a lost voice clone -
  made the *next* install fail outright, because retained data remembers the
  identity that wrote it,
- so the way through was always uninstall-and-discard, which costs a 98 MB model
  download on every single sideload.

None of that is visible in the build file, which said "sign with the debug
config" before and after. It is a property of the artefact, so
`tools/check-signing.sh` reads the artefact: it pins the certificate's SHA-256
and CI runs it against both APKs. Change the key deliberately and that constant
has to change with it - and everyone holding the old build has to uninstall to
take the new one, which is exactly the cost worth being reminded of.

What committing it gives away is real but small: anyone with this repository can
build an APK that installs over yours. The platform's own debug key is public
and identical on every machine for the same reason.

## One install, however many things ask for it

Two callers reached the downloader independently - the button on the main
screen, and any read that arrived before the model existed - and neither knew
about the other. They shared a partial-download file and a staging directory, so
tapping Download and then selecting text in another app had them writing the
same `.part` and deleting each other's unpacked files.

The download also belonged to the activity that started it. A configuration
change cancelled the coroutine, but the transfer has no suspension point, so it
carried on headless while the recreated screen said "not downloaded" and invited
a second tap - and the partial was deleted on the way out, so that tap began
again from zero.

`engine/ModelInstall` owns one shared attempt per model. Everybody who asks for a
model joins the one in flight for it, a screen that goes away leaves it running,
and progress is a
`StateFlow` a recreated screen can pick up rather than something an activity
holds. Removing the sharing fails the test for it.

The download resumes. The largest model is 219 MB; without a `Range` request a
drop at 200 MB throws away 200 MB, and on a connection that drops regularly it never finishes at
all - each attempt just gets a different distance through the same first
stretch. The partial now survives a network failure and is deleted only when the
archive turns out not to unpack, because that one is not worth resuming.

## Keeping the model, and the voices

Everything lives in internal storage, so uninstalling threw all of it away: a
model download of up to 219 MB, and - the part that matters - every voice the user had recorded
or imported, which exists nowhere else. `android:hasFragileUserData` makes the
uninstall dialog offer to keep the data, which is the only way to offer that
choice at all.

Backup rules exclude the models and nothing else. Cloud backup allows an app
25 MB, so including up to 219 MB of regenerable weights does not make a partial
backup - it makes a failed one, taking the settings and the voices down with it.
Every model lives under `pocket-tts/models`, which the rules exclude as a whole,
so a model added later can never be backed up by accident. The rules name the
directory as a literal string because XML cannot reference a Kotlin constant, so
`BackupRulesTest` asserts the string still matches `ModelManager.MODELS_DIR`.

**Since 2026-10-03 "Install from a file" is hidden and refuses**: the paragraph
below describes sherpa-onnx's archive, which the new engine cannot read, and no
archive of pocket-speak's bundle is defined yet (`docs/engine-swap.md`).

"Install from a file" takes the same `.tar.bz2` as the download, streamed
straight from the content URI - the unpacked model is already 200 MB and there
is no reason to want another 98 MB beside it. It is checked for the expected
ONNX files before anything is replaced, so pointing at the wrong archive leaves
a working install working.

## First run

The model is not in the APK. On first launch the app downloads the selected
model - Kyutai's `english_2026-10` unless another is chosen, 219 MB, each file
checked against the sha256 `pocket-speak/models.json` pins - into app storage.
Each voice is a state precomputed for that model, 0.5 to 7 MB, fetched the first
time that voice is used. An upgraded install deletes the models earlier engines
read (sherpa-onnx's, then the ONNX export of `english_2026-04`) once the new one
has loaded. See `docs/pure-rust-engine.md`.

Inference runs on the CPU, on upstream's own Rust implementation of the model.
The previous engine generated 3.7 to 5 times faster than real time on a Galaxy
S25, with first audio 0.2 to 0.3 s into a chunk; on the same desktop CPU the new
one is 2.4 times faster on the larger `english_2026-10` model and starts in a
fifth of the time. Phone numbers for it are to come.

## How it works

**Since 2026-10-03 the engine is pocket-speak's Rust core, not sherpa-onnx**
(`docs/engine-swap.md`), and since 2026-10 that core runs upstream's own Rust
implementation of the model rather than ONNX Runtime
(`docs/pure-rust-engine.md`). The sections below that describe sherpa-onnx's
behaviour are the history of how the app got here, kept as written.

```
selected text ──▶ ReadAloudActivity ─┐
scratchpad ─────▶ ScratchpadActivity ─┼─▶ Reader ─▶ StreamingPlayer (AudioTrack)
                                      │      │
other apps ─────▶ PocketTtsService ───┘      ▼
                  (system TTS engine)   MarkdownSpeech ─▶ TextChunker ─▶ PocketTts
                                                                            │
                                                     NativeEngine (JNI) ─▶ pocket-speak/crates/engine
                                                                            │
                                                          xn-ptts + upstream's rust/src
```

- `speech/MarkdownSpeech` turns Markdown into speakable prose. A table is read
  as a screen reader reads one: announced ("Table, 2 rows."), then row by row
  under its column names; past four columns it is announced with its column
  names and not read (#10). Inline code is always read, as words; the setting
  covers fenced blocks only (#9).
- `speech/TextChunker` cuts it at sentence boundaries so the first audio arrives
  quickly and stopping is responsive.
- `engine/PocketTts` owns the single loaded model and serialises synthesis.
- `player/StreamingPlayer` writes float PCM to an `AudioTrack` with blocking
  writes, which back-pressures generation to the speed of playback.
- `player/PlaybackService` is a foreground service so reading survives the
  floating window being dismissed, and holds audio focus for as long as it is
  playing.
- `player/AudioFocus` asks the system for the right to be heard and gives it
  back. Without it the app read over whatever was already playing, a call did
  not pause it - muting the media stream during a call is at the vendor's
  discretion - and unplugging headphones carried on out loud from the phone's
  speaker. A transient loss pauses and resumes; a permanent one stops, because
  another app has taken over for as long as it likes and a paused reader nobody
  asked for helps no one. Ducking is declined: two voices at once are not two
  things you can listen to.

`MarkdownSpeech`, `TextChunker` and `WavReader` are plain JVM code and are
covered by unit tests in `app/src/test`.

## Stop has to be in the collapsed controls

The collapsed media player - the shade before it is expanded, the lock screen,
which cannot expand it - shows three controls. Stop was the fourth, on the
theory that swiping the notification away already ended the read; a media
notification for a read in progress cannot be swiped away, and on the device
the broadcast path was first tested on there was no way to end a read from the
shade at all. That is the one control a reader started from a pocket cannot be
without.

Two things draw those controls and both had to change. Before Android 13 the
notification's own actions are drawn, and the compact view now shows back,
pause and stop. From Android 13 the system draws the media session's actions
instead: previous, play/pause and next take the three slots, and a custom
action - which is the only way to offer Stop at all, since the system has no
button for `ACTION_STOP` - appears only once expanded. So `Transport` no longer
publishes `ACTION_SKIP_TO_NEXT`, which hands that slot to the first custom
action, and that is Stop. Forward becomes a custom action in the expanded
view, and the headset's "next" button - which the framework only routes to
`onSkipToNext` when the action is published - is taken by the callback itself.
Forward was the one to demote: it is the control that ends a read by accident.

## Saying the true thing while waiting

The model composes an entire sentence before it emits a single sample, so there
are several silent seconds between asking for speech and hearing any. The reader
reached `Speaking` when it started *working*, and every screen printed "Reading
aloud" through that silence - which is most of what the wait before the first
word actually feels like, and the reason it reads as broken rather than slow.

`Speaking.audible` is false until a sample has actually been written, and the
screens say "Composing the first sentence…" until it turns true. The wait did
not get shorter; it stopped being a lie.

## Following along

`Speaking` reports the character range of the chunk being read, but those
offsets are into the *stripped* text: stripping deletes syntax, so they drift
further from the document with every heading, link and asterisk passed.
`MarkdownSpeech.toSpeakableWithSource` records where each block came from, and
`Reader.spokenRangeIn` turns one into the other.

The map is per block, not per character - a block comes from a known run of
source lines, which is cheap to track. Within a block the range is narrowed to
the exact sentence when that sentence survived stripping unchanged, which is the
common case for prose; a rewritten line falls back to highlighting the whole
block, which is less precise but never wrong.

Everything hard about it is in the setup rather than the algorithm. Anything
that rewrites the string before the lines are counted silently shifts every
offset after it, so HTML comments are blanked with spaces of equal length rather
than removed, and line breaks are found by scanning rather than by normalising
CRLF to LF. Both have a test, and both tests fail if the normalisation comes
back. The CRLF one needed a second pass to be worth anything: the first version
used text that survived stripping unchanged, so the exact-sentence search
silently corrected the drift and the test stayed green against broken offsets.

## Two callers, one engine

There is a single model in the process and two things drive it: the in-app
reader, and the system text-to-speech service other apps call. `synthesize`
serialised them per *chunk*, so with both active they took turns sentence by
sentence and a passage came out read alternately in two different voices.
Neither side knew the other existed.

Serialising whole utterances instead would replace the alternation with a wait:
an app asking through Select to Speak would hang until a whole document finished
reading, which looks broken from the other side. So `engine/EngineTurn` makes
the most recent request win. A caller takes a turn before it starts and checks
it as it goes - per chunk and per audio callback, so a request arriving
mid-sentence does not have to wait out the rest of it. Finding the turn has
moved on means somebody asked more recently, and standing down is the only
thing that produces intelligible audio.

## A queue, where a turn will not do

`EngineTurn` arbitrates between the two things that drive the one model, and its
rule is that the most recent request wins. That is the right rule for two
callers who cannot see each other, and exactly the wrong one for a caller
sending several things to hear in order: an agent handing over one reply while
the last is still being read would have each one silence the one before it, and
a run of them would play as the last few words of the last.

So `Reader` has its own queue and `enqueue` joins it, while `speak` still
replaces. Nothing about standing down for a stranger answers "after that one",
and stretching `EngineTurn` to cover both would have made the same counter mean
two opposite things.

What the queue is careful about is whose it is. A `speak` that replaces the
current read drops what was queued behind it - those follow-ups belonged to the
passage being abandoned, and carrying them over would play a replacement
followed by the tail of the thing it replaced. A stop drops them too. Skipping
forward off the end of the last chunk does *not*: that is an ending like any
other, and it is the one ending that does not arrive through the reading job
completing, so it advances the queue by hand.

## The reader has one state machine, and it is testable

`Idle` used to mean two things - "nothing has been asked for" and "the read is
over" - so `stopLocked` published it between two utterances, and three screens
each kept a boolean to tell those apart. `StateFlow` conflates, so whether
anyone saw that transitional value was a race: the scratchpad overlay sometimes
failed to appear on a second Speak, and the foreground service sometimes
stopped and immediately restarted.

`Reader.State` now ends in `Finished`, `Stopped` or `Failed`, each carrying the
utterance it belongs to and who asked for it. The three flags are gone. Identity
matters as much as finality: the reader is process-wide, so a screen has to tell
its own read's ending from an earlier one it never asked for - the floating
window would otherwise close itself on stale news before its own read had begun.

`player/SpeechEngine` and `player/AudioSink` exist so this can be tested at all.
Before them, driving the reader meant a 98 MB model and an `AudioTrack`, which
is why the part that had actually been wrong had no tests - and why
`ActivityLaunchTest` reached `ensureModel` and quietly downloaded the model
bundle on every run of the unit suite.

Skipping is built on the same seam. An utterance keeps its chunks, its engine
and its loaded voice, so skipping back costs re-composing one chunk and nothing
else: no re-selection, no reload. Back at the first chunk replays it; forward
past the last one ends the read, which is what someone who keeps tapping
forward means by it - and starts the next queued read, if there is one.

A chunk is the unit, and a chunk is not a sentence: it is a paragraph, or up to
a few hundred characters of one, cut at sentence ends. The controls used to be
labelled "a sentence", and on a short reply - one paragraph, one chunk - a tap
forward near its second sentence ended the read, which reads as the wrong
button doing stop's job. The labels now say what the buttons do. A skip finer
than a chunk would need the engine to give audio back sentence by sentence,
and it composes a chunk whole.

`GlassPanelViewTest` asserts the two things that were missing when the sliders
went dead: where the panel thinks it is relative to its backdrop, and what
colour it actually put on the canvas. Its predecessor asserted only that drawing
did not throw, and Robolectric's software canvas made the panel bail before it
ever reached the geometry - so the broken function was never called by any test.

`ActivityLaunchTest` drives every activity through its real lifecycle under
Robolectric, and forces a measure and layout pass at a realistic size. A theme,
a manifest entry and a layout can crash an activity just as easily as a function
can, and none of them are type-checked — one build shipped a theme that supplied
a decor action bar to screens that also called `setSupportActionBar`, and
another handed a Material `Slider` a value off its step grid. Both compiled,
packaged, and threw the instant the screen opened.

The layout pass is the part that earns its keep. Views validate themselves while
sizing, so reaching RESUMED proves very little: the slider bug passed a version
of these tests that stopped at `setup()`, because nothing had been laid out yet.

## The one thing here that cannot be fetched again

**Since 2026-10-03 imported voices are hidden, not deleted.** The engine is
conditioned on precomputed embeddings and has no encoder to make one from a wav,
so the picker and the system voice list leave imports out until cloning returns
(`docs/engine-swap.md`). The files stay where this section says.

A model re-downloads and a setting is retyped in seconds. A voice somebody
recorded exists in one directory and nowhere else, so the failure worth
guarding against is not a crash but a quiet overwrite.

Imports used to share a directory with the downloaded prompts, so a wav named
`alba.wav` landed on exactly the path the stock Alba prompt is cached at,
replaced it, and then vanished from the list - because anything named like a
stock voice was filtered out of that list. One action, two losses, no message.

Imports now live in their own directory under `voices/imported`, and a name that
collides with a stock voice or an earlier import is suffixed rather than allowed
to win. Voices imported by an older build are carried across on first access;
one that was already named after a stock voice is indistinguishable from the
prompt it replaced and is left to be re-downloaded. `VoiceStorageTest` covers
all of it, and restoring the shared directory fails four of its six cases.

## The model decides when it is done, and gets the end wrong

The engine has no length for a chunk in advance. It runs the language model
frame by frame and stops when the model's own end-of-speech head crosses a
fixed threshold, plus three frames. That head is wrong in one particular
place: the end of a generation, when what remains to be said is a few very
short sentences. Found on the device - a reply ending `One. Two. Three. Four.
Five.` stopped after "Two" - and then reproduced here on the same engine build,
the same int8 bundle and the same Alba prompt, with a speech recogniser as the
judge, so the mechanism is measured rather than guessed:

- `... One. Two. Three.` lost "Three" in two runs of three; `... Patching.
  Testing. Done.` said "done" twice in all three; `One. Two. Three. Four.
  Five.` on its own lost "Five" every time. The same runs with ordinary prose
  after them were read in full every time: it is the *end* that is fragile.
- Letting generation run on past the end-of-speech mark does not recover the
  words. After a genuine ending the model does not fall silent, it repeats the
  last word - `five, five, five` for as long as it is allowed - so there is no
  silence to trim to.
- The reference implementation pads very short inputs with leading spaces
  because "the model does not perform well when there are very few tokens".
  sherpa-onnx strips them before the model sees them; padded and bare runs
  were identical to the sample.
- The same short sentences joined with commas into one - `One, Two, Three.` -
  were read in full in every run, for every shape tried.

So `TextChunker` gives each chunk a `speech` form: a run of two or more
sentences of two words or fewer at the end of a chunk is joined with commas.
What the user sees and what is highlighted is untouched; a period becoming a
comma is the whole of the difference in what is said. `MarkdownSpeech` already
makes every list item and heading its own paragraph, so a list is never a run
of short sentences inside one chunk; this is for prose that ends in a
countdown or a string of one-word statuses, which is how agent replies end.

What it does not cover, and is written down so nobody re-derives it: the same
head can fire early inside an ordinary sentence. `Tests 1, 2 and 3 are green.`
lost "green" in two runs of three here - an enumeration of numbers reads to the
model like a countdown ending - while `Files A, B and C are updated.` and plain
prose endings were never cut. The threshold is hard-coded in the engine, the
reference implementation uses the same value, and nothing at this layer can
tell a premature end from a real one without a recogniser listening. This is
the model's limit, and the tail rule is the part of it the app can reach.

## Measuring before optimising

Inference is CPU-only and near real time, and every performance question about
this app turns on timings that cannot be measured here - no emulator, no device.
So `debug/Metrics` collects three numbers and the main screen shows them, with a
share button, the way the exit report does.

- **Time to first audio**, which is the wait everybody feels.
- **Generation speed on the first chunk**: audio seconds composed per wall
  second, by the engine alone. Below 1.0 means the model cannot keep up with
  its own playback.
- **Underruns**, which now say one thing: whether composing the next chunk
  while the current one plays is enough to keep the speaker fed. They climb
  only when the model falls behind real time.

The channel between synthesis and the audio track, which an earlier version of
this section deliberately left unwritten until underruns proved it necessary,
is written - but the device proved something else first. Reading the engine's
source settled how it actually delivers audio: `GenerateSingleSentence` runs
every frame of the language model over a chunk *before* the decoder produces a
sample, and only then hands audio back, in pieces of just over a second. So
`first=NNNms` in the voice trace is not a prompt being primed; it is the whole
of the model's pass over that chunk. And because the sink was fed from inside
the engine's callback, chunk N+1 could not begin until chunk N had been *heard*:
the silence before every paragraph was the model's full pass over it, less a
couple of seconds of buffer, charged for a table that stripped to one line just
as for a long one. Underruns could not see that, because a track that is simply
not written to between chunks is not "running dry" in a way that counts.

`Reader` now composes a chunk ahead of the one being played. The bound is one
chunk of lookahead, which is enough to hide the model's pass behind the
previous chunk's audio whenever the model is faster than real time, and it is
on the device this was measured on. What the state publishes - the highlighted
chunk, what a skip counts from - follows the listener, not the engine.

An unmeasured value reports "not measured yet" rather than a confident zero, and
`MetricsTest` enforces that. "0 underruns" from a session that never played
anything reads as evidence and is not.

## The voice, and why it kept changing

The voice used to change from sentence to sentence, and sometimes was not the
voice that had been selected at all. Three separate faults, all of which sound
identical from the outside.

**The speaker is drawn, and the draw was uncontrolled.** Pocket TTS has no
speaker table. It is prompted with a few seconds of reference audio and samples
a speaker in that audio's neighbourhood; the draw is a vector of Gaussian noise.
In `offline-tts-pocket-impl.h`:

```cpp
float temperature = gen_config.GetExtraFloat("temperature", 0.7f);
float stddev = std::sqrt(temperature);
int32_t seed = gen_config.GetExtraInt("seed", -1);
NormalDataGenerator normal_gen(0, stddev, seed);
```

That sits inside `GenerateSingleSentence`, which sherpa-onnx calls once per
sentence after re-splitting whatever text it is handed on `.!?`. Per *sentence*,
not per call. A seed of -1 means a fresh draw from a random device every time,
so a paragraph is read by a succession of slightly different people.

Both knobs travel in `GenerationConfig.extra`, an untyped string map. This app
passed nothing, so it got sherpa-onnx's defaults. It now sends both:

- **A fixed seed**, when "Keep one voice across sentences" is on, so every
  sentence draws the same speaker. It is the whole of that feature. An earlier
  attempt kept the last six seconds of generated audio and fed it back as the
  next sentence's prompt - which could not work, because it acted at chunk
  boundaries while the re-draws happen per sentence *inside* a chunk, and
  because replacing the prompt with generated audio abandons the selected voice
  entirely and never returns to it. That code is gone.
- **Temperature 0.3** rather than sherpa-onnx's 0.7. Kyutai moved English to 0.3
  in this repository (`d108410`): "Human evaluations consistently prefer the
  English model at temperature 0.3 over the current default 0.7, and the change
  is free on every objective axis we measured." Their default is recorded
  against `english.yaml` and `english_2026-04.yaml`, while the bundle this app
  downloads is `english_2026-01` - a strong prior rather than a measured value
  for this snapshot, hence a slider.

**A stock prompt could be somebody else.** An import named like a stock voice
overwrote that voice's wav, and `ensureVoice` treated any non-empty file at the
right path as cached - so nothing ever fetched the real one again. Every "Alba"
from that moment on was read in a different person's voice, silently, on every
build. `VoiceCatalog` now carries each prompt's exact size and a mismatch is not
cached. (The comment claiming these "will be replaced by a fresh download" had
been there the whole time. Nothing did it.)

**The system engine served a voice chosen weeks ago.** Android resolves an
engine's default voice name once per client and then sends that name back on
every subsequent request. Returning a concrete voice id from
`onGetDefaultVoiceNameFor` therefore pinned whichever voice was selected when
Select to Speak or Chrome first bound, and changing the voice here did nothing
for them until they were force-stopped. The engine now advertises one alias
voice, `selected`, resolved on each request.

## The engine splits what has already been split

Measured against a reference implementation of this same model - a Rust adapter
over ONNX Runtime 1.23.2, in
[speech-kit-obsidian-plugin](https://github.com/mycosavant/speech-kit-obsidian-plugin)
(`native/src/adapters/pocket_tts.rs`) - the single biggest difference is not a
parameter. It is that the reference **never restarts**:

```rust
fn synthesize(&mut self, text: &str, ...) -> Result<SynthesisPcm, SynthesisError> {
    Ok(SynthesisPcm { samples: self.generate_chunk(text, voice_path, cancellation)?, ... })
}
```

The whole text goes into one autoregressive loop, which runs until the EOS logit
fires and a few frames after it, and the latents are then Mimi-decoded in
blocks. There is no sentence splitting anywhere in it.

sherpa-onnx does the opposite. `SplitByPunctuation` cuts on `.!?`,
`MergeShortSentences` re-accumulates to a 30-character floor, and each resulting
sentence is an independent generation with its own onset and its own ending,
knowing nothing of what came before. That is where the seams come from, and it
is why a paragraph here has never sounded like the same paragraph elsewhere:
sentences run into each other and first syllables sound clipped.

Both bounds are caller-configurable through the same `extra` map that already
carries the temperature and the seed:

```cpp
int32_t max_char_in_sentence = gen_config.GetExtraInt("max_char_in_sentence", 200);
int32_t min_char_in_sentence = gen_config.GetExtraInt("min_char_in_sentence", 30);
```

So both are set above `TextChunker`'s 400-character maximum, and a chunk is
generated whole. This is not a new policy - the text was already cut at sentence
boundaries, at a size chosen for time-to-first-audio - it is declining to have
that work undone. A chunk is comfortably inside the generation's own `max_frames`
cap of 500 frames: 400 characters is roughly 28 seconds of speech at the model's
frame rate.

What that leaves is one restart per chunk instead of one per sentence.

On a device, that change removed almost all of the clipped first syllable, at
both step counts. **And one flow-decoding step was then reported closer to the
same model on a desktop** - "maybe a little more expressive, or more
inflection" - so the default follows the reference's one rather than
sherpa-onnx's five. Fewer steps is also less work per frame, which sits oddly
beside the same report calling it a touch slower to the first word; the `first=`
figure in the trace is what settles that, and chunk timings on that device move
by about 90 ms between identical runs anyway.

Two differences from the reference remain. The seed is pinned to a constant here
against a fresh draw there. And the bundle is `english_2026-01` here against
`english_2026-04` there - which is not a version number so much as a different
package: sherpa-onnx's bundle carries a Mimi *encoder*, and the reference's does
not, because its voices are precomputed embeddings from
`kyutai/pocket-tts-without-voice-cloning`. The encoder is what turns a wav
somebody recorded into a voice, so that bundle cannot clone one. The temperature
is 0.3 in both.

## What the seed could not fix, and what did

Pinning the seed made every sentence draw the *same speaker*, and that is what
stopped a paragraph being read by a succession of different people. It left a
residue, reported from the device as "still drifting, not as bad as before".

The obvious explanation was that nothing generated crosses a sentence boundary,
so pitch, pace and energy reset at every full stop even when timbre holds. The
standard fix for chunked neural TTS follows from it: condition what is about to
be generated on what was just spoken. This repository has now tried that twice
and deleted it twice, and the second attempt is worth a paragraph because of how
it failed.

It worked, in the sense that it did what it said. `VoiceContinuity` made each
chunk's reference the voice prompt plus the last two seconds actually spoken,
with the prompt always the head - so identity was re-anchored on every
generation, which is the thing the first attempt got wrong. Three tests pinned
that. On a device it sounded *slightly worse*, and it cost a Mimi encode of
50-70 ms per chunk, because sherpa-onnx caches the voice embedding in a 50-entry
LRU keyed by a hash of the reference samples: a prompt that never changes is
encoded once per process, and one that moves every chunk is a miss every chunk.
"The same ten seconds either way" is exactly what makes it not free.

Then the splitting came to light, and the residue turned out not to be a missing
mechanism at all - it was sherpa-onnx cutting a chunk back into sentences and
generating each one independently. Four lines of `extra` map fixed it, the
clipped first syllables went away on the device, and what `VoiceContinuity` had
been built to work around no longer existed. It was cut before this merged.

Two things it found on the way out are worth keeping in mind, because both were
found by the device rather than by reasoning:

- **A sample-rate guard made it inert.** The stock prompts are 48 kHz, the model
  generates at 24 kHz, and the guard stood down on mismatch - silently, on the
  default voice, while the splitting it paid for went on happening. All of the
  cost and none of the effect. Two runs went into working out that a prompt hash
  which never changed meant "disabled" rather than "switched off", which is why
  the voice trace now says outright what it is doing.
- **Per-sentence conditioning cost a third of the generation speed.** The
  reference is re-encoded once per generation call, so splitting a chunk into
  three sentences pays it three times: 1.12x real time fell to **0.73x**, below
  playback, at which point gaps are arithmetic rather than bad luck. And it did
  not even decide what a sentence was, because sherpa-onnx re-splits whatever it
  is handed - `. . . .`, an ellipsis typed as spaced periods and ordinary in
  scripture and older prose, still reached the model as a request to speak three
  bare full stops however carefully this side folded them into their neighbours.

What survived is the instrumentation, which is the part that settled any of it:
each `[chunk n]` line carries `first=NNNms`, wall clock from asking for a chunk
to its first sample, so the encode is inside it and the blocking write is not.
`Metrics.generationRealTimeFactor` cannot see that interval - it is taken on the
first chunk of a read, which is always a cache hit.

## Who is actually speaking

`debug/VoiceTrace` records, per read: the voice asked for, the voice found,
whether that was a fallback, the prompt file's size against the size that voice
ships as, and then per chunk the prompt's length and content hash with the
temperature and seed in force.

```
[reader] asked for alba, got alba (958542 bytes as expected)
[chunk 0] alba prompt=9.98s hash=6f3a1c2e temp=0.30 seed=1
```

It exists because the faults above are mutually indistinguishable by ear and the
phone this runs on has no logcat within reach. Each line separates them: a
fallback names itself, a prompt that is not the one it claims to be shows a size
that does not match, a client sending a stale id shows a `requested` that is not
the selection, and a voice wandering with all of that correct is the draw. It
shares a screen with Timings, by the same reasoning as the exit report.

## Steps per frame, and the factor of five that was not there

`GenerationConfig.numSteps` is how many Euler steps the flow integration takes
for each generated frame. sherpa-onnx defaults it to 5. The reference
implementation defaults to 1 - in `lsd_decode` itself and again in
`default_parameters.py`. Four fifths of that work looked like it might be buying
nothing, per frame, on a phone.

It was shipped as a slider with the default left at 5, on the reasoning that one
side of the trade is only audible and nobody here can hear it. The device
answered: at one step the reading was reported closer to the same model on a
desktop, *"maybe a little more expressive, or more inflection"*, and no worse. So
the default is 1 now, and the slider stays for anyone who disagrees.

How much time that can buy is bounded, and worth knowing before reading the
number. The steps multiply one graph in the bundle - `lm_flow`, 10 MB - and not
the 76 MB `lm_main` that runs once per frame regardless. So the ceiling on the
saving is whatever fraction of a frame the small model accounts for, and that
fraction is still unmeasured. One report called one step a touch *slower* to the
first word, which cannot be a cost of doing less work; the `first=` figure on
each chunk trace line is what settles it, and chunk timings on that device move
by about 90 ms between identical runs anyway.

`docs/direct-ort.md` scopes what running these graphs ourselves would involve,
and starts from the same reading of the bundle. `docs/owning-the-pipeline.md`
asks the wider version of the same question.

## When it crashes

Sideloading onto a phone means no logcat within reach, and an app that dies
silently reports exactly one bit: "it crashed". `debug/CrashLog` writes the last
uncaught exception to app storage on the way down, and the next launch offers to
share it. It chains to the previous handler rather than swallowing the crash, so
the process still dies and the system dialog still appears - only the evidence
survives.

`CrashLog` alone cannot catch native crashes, ANRs or low-memory kills - none of
them unwind through the JVM - and those look identical from the outside to a
Kotlin exception. `debug/ExitReasons` closes that gap by reading the platform's
own record via `ApplicationExitInfo`, which names the kind of death and, for a
native crash on Android 12+, carries the tombstone. That reason code is the fact
that decides what to fix; the two sources are shown together on next launch.

## The JNI callback is not a lambda, on purpose

`PocketTts.audioCallback` wraps the streaming callback in an object expression
rather than passing a lambda. sherpa-onnx resolves that callback from native
code by name and exact signature - `invoke([F)Ljava/lang/Integer;`.

Kotlin 2.0 emits lambdas via `invokedynamic`, and D8 desugars them into
`$$ExternalSyntheticLambda` classes carrying only the erased
`invoke(Object)Object`. The specialised method is absent, `GetMethodID` returns
nothing, the JNI call proceeds with a pending exception, and the runtime aborts
the process. It surfaces as `REASON_CRASH_NATIVE` with no Java stack trace, from
code that is entirely ordinary Kotlin and compiles without a warning.

Nothing in Kotlin references that signature, so nothing in Kotlin breaks when it
disappears. `PocketTtsCallbackTest` asserts it by reflection and fails if the
callback becomes a lambda again.

It then disappeared a second time, by a different route. R8 inlined the
specialised method into its own bridge, leaving one method whose descriptor is
`(Ljava/lang/Object;)Ljava/lang/Object;` - the reflective test still passed,
because it reflects over the unshrunk class. Release builds have no CheckJNI, so
instead of aborting they synthesised entire utterances into nowhere and reported
a failure after a long silence. Nobody noticed for a simple reason: CI built
only `assembleDebug`, so every APK ever sideloaded was a debug build.

`proguard-rules.pro` now keeps the method, and `tools/check-jni-callback.sh`
reads the built APK with `dexdump` and fails if no method declares that
descriptor. Both APKs are built and checked in CI, and both are uploaded, so
the shrunk build is something that gets installed rather than something that is
assumed to work. The source is not the artefact, and only the artefact knows.

Since the engine swap the callback is `NativeEngine.Sink.onAudio([F)Z`, and the
engine also needs `NativeEngine`'s native methods under their own names.
`check-jni-callback.sh` now looks for those instead of sherpa-onnx's `invoke`.

## Glass, and where it cannot exist

Blurring depends entirely on what is behind the panel, and the two cases have
different answers.

Behind **another app**, only `Window.setBackgroundBlurRadius` can blur, because
an app may not read another app's pixels. Whether that works is not something to
reason about: `WindowManager.isCrossWindowBlurEnabled` answers it. On the device
this was built for it says no, and no value of any slider will produce a frost.

What it does not say is *why*, and this app no longer guesses. Accessibility →
"Reduce transparency and blur" turns it off system-wide, battery saver takes it
away too, and some devices never had it. The frosted panels the system draws for
*itself* - Samsung's quick settings and app-drawer folders, which are lovely -
are composited by a privileged process and settle none of it.

Most of those causes come and go, so the capability is followed with
`addCrossWindowBlurEnabledListener` rather than read once when the screen opens.
That also makes the appearance screen the test: toggle the accessibility setting
and the line changes under you, which answers the question without anyone having
to be right about it in advance.

The read-aloud sheet used to pretend otherwise. It drew a near-transparent
surface, found it could not blur, and painted a hard 0.94 instead - silently
overriding the opacity that had been dialled in, and looking like a blur that
had failed rather than like a decision. It is now an opaque Material card at the
bottom of the screen: bottom, because a centred dialog lands on the paragraph
that was just selected, which is the one thing the reader may want to keep
looking at. Where the platform genuinely supports cross-window blur it is
applied on top and the surface steps back to the tuned opacity - a bonus on the
devices that have it, not the design.

Behind **our own content**, `ui/GlassPanelView` gives a true
`backdrop-filter: blur()`: the blur is confined to the panel's own rounded
bounds and everything outside stays sharp. Each frame it records the backdrop
view into a `RenderNode`, offset so the slice behind the panel lands at the
panel's origin, hangs a blur `RenderEffect` on that node, and draws it clipped
to the panel - then the dim, then the tint, then the hairline. This works on any
Android 12 device with no vendor opt-in, so the scratchpad overlay is where the
glass lives.

That offset is the whole trick, and getting it wrong is silent. The panel has to
sit *outside* the backdrop - capturing an ancestor would draw the panel into its
own backdrop and recurse - so the backdrop is always a sibling. An earlier
version located the panel by walking *up* its parent chain looking for the
backdrop, which a sibling is never on: the walk ran off the top of the
hierarchy, the capture declined, and every panel fell back to a flat tint with
the configured opacity thrown away. The appearance sliders appeared to do
nothing, because for three of the four that was exactly true. Both views are now
located against the root they share and the difference is taken, which is
correct for siblings, cousins and ancestors alike.

The capture also has to start opaque, which is the second way this looked
broken. A backdrop view usually has no background of its own - it is letting the
window's background show through - so recording it captures its text on
transparency. Blur that, composite it, and the blurred copy lands *on top of*
the original, which is still there and still sharp underneath. Only the dim and
the tint appeared to do anything, and the panel read exactly like one whose blur
was not working.

That is why the appearance preview frosted convincingly while the real
scratchpad overlay did not: the preview's backdrop has a background and a row of
colour swatches, and the editor behind the overlay has neither. The panel now
fills the capture with the window background before drawing the backdrop into
it, so the blur has something solid to work on and the result covers what is
underneath instead of ghosting over it. Nothing about the backdrop's own layout
is required any more.

Because these failures have no visual signature - a panel that quietly gave up
looks like settings that were never wired - every frame records *why* it drew
what it drew in `lastDraw`, and the appearance screen prints it. A log line is
no use on a phone with no logcat within reach. It is worth saying that
`lastDraw` did *not* catch either of the two faults above: both reported
`BLURRED`, because the capture genuinely ran. It answers "did the blur path
execute", which is a narrower question than "did anything blur".

## The keyboard is not the window's problem any more

The scratchpad's Speak and Stop buttons sat under the keyboard. `adjustResize`
is in the manifest and does nothing: a window laid out edge to edge - which
`targetSdk 35` makes every window - is no longer resized when the keyboard
opens. It is handed the keyboard's size in the insets and expected to deal with
it. Nothing warns you, and the symptom is that reaching the control means
dismissing the keyboard you were typing with.

`ui/Insets` now pays back the maximum of the system bars and the IME, rather
than the system bars alone. The maximum and not the sum: the keyboard is drawn
over the navigation bar, so its inset already contains it, and adding them
leaves a gap exactly one navigation bar tall.

This is the technique [Haze](https://chrisbanes.github.io/haze/) uses on
Android. Haze is Compose Multiplatform and this app is Views, so the technique
is borrowed and the dependency is not.

`ui/AppearanceActivity` exposes opacity, blur, dim and corner radius as sliders,
because how this reads depends on the wallpaper-derived palette and the display
density - neither of which can be judged from source. Every label reports the
value in the form the source wants (`0.37f · alpha 94 · 37%`), and a button
copies the whole set as pasteable Kotlin. The defaults in `Settings` are the
values that came back from a device that way.

## Selecting text inside our own app

The `PROCESS_TEXT` item this app registers appears in the selection toolbar of
its own editor too, so selecting a line in the scratchpad opened the floating
window over the screen that already had controls for it - two sets of controls
for one utterance, in two different treatments, one of which could not frost.

`ReadAloudActivity` checks `callingPackage`: called from ourselves, it tags the
read as coming from the scratchpad, starts the service and finishes without ever
showing a window. The scratchpad claims any read tagged that way, whichever
route it arrived by.

## Edge-to-edge

`targetSdk 35` means Android 15 lays every window out edge to edge, so the
system bars are drawn over the content rather than beside it. `ui/Insets.kt`
pays those insets back as padding: each screen names the view that absorbs the
top inset (its app bar) and the one that absorbs the bottom. Without it the
toolbar wears the status bar and the first control is sliced in half.

## Licensing

Worth reading before shipping anything built from this.

- Since 2026-10 the models, their tokenizers and the voice states all come
  from Kyutai's own
  [`kyutai/pocket-tts-without-voice-cloning`](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning),
  CC-BY-4.0, pinned in `pocket-speak/models.json`. Until then the app read an
  ONNX export whose `LICENSE` file said CC-BY-4.0 while its README said
  non-commercial use only; that ambiguity went with it.
- The engine is upstream's Rust code and
  [xn-ptts](https://github.com/gradium-ai/xn-ptts), MIT (xn-ptts: MIT or
  Apache-2.0); `pocket-speak/crates/engine/NOTICE.md` has the details.
- `app/src/main/java/sonic/Sonic.java` is Bill Cox's Sonic, Apache 2.0, vendored
  unmodified.
- Voices have individual licences, listed per voice at
  [`kyutai/tts-voices`](https://huggingface.co/kyutai/tts-voices).
- Pocket TTS's own prohibited-use terms apply, in particular the ban on cloning
  someone's voice without their explicit and lawful consent. The import-a-wav
  feature makes that easy to do accidentally.

## Prior art

If you would rather not build anything:

- [NekoSpeak](https://github.com/siva-sub/NekoSpeak) (MIT) is an offline Android
  TTS engine that already supports Pocket TTS alongside Kokoro, KittenTTS and
  Piper, with voice cloning. It registers as a system engine but does not add a
  text-selection menu item or a scratchpad.
- [TTS Util](https://github.com/Danesprite/tts-util-app) (Apache 2.0) adds
  share-sheet entries and can read the clipboard through whichever system engine
  you have selected.
- [SherpaOnnxTtsEngineAndroid](https://github.com/jing332/SherpaOnnxTtsEngineAndroid)
  wraps sherpa-onnx as a system engine with in-app model downloads.

Pairing NekoSpeak (or this app's engine) with TTS Util gets you most of the way
without writing code; this app exists mainly to put all three pieces behind one
tap.
