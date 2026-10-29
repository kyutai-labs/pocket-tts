# Pocket TTS for Android

A native Android app that reads text aloud in Pocket TTS voices. Speech is generated on the device. Once the model and a voice are on the phone, speaking needs no network. The first use of a voice downloads its prompt.

The app exists to remove one annoyance: hearing a paragraph in a good voice meant pasting it into a notes app or a pastebin first.

## What it gives you

The text selection menu has a "Read aloud" item. Select text in any app (a browser, a chat app, a PDF reader) and Pocket TTS appears in the selection toolbar next to Copy and Share. Tapping it starts reading at once in a small floating window with pause and stop. This is an `ACTION_PROCESS_TEXT` activity. Apps that draw their own selection toolbar and suppress the system one will not show it, so the share sheet has an entry as well.

Once Pocket TTS is selected as the text-to-speech engine under *Settings → Accessibility → Text-to-speech output*, every app that reads text aloud can use these voices. That includes Select to Speak, Chrome's read-aloud, ebook readers and accessibility tools. None of them need to know this app exists. Audio is streamed back chunk by chunk as it is generated, so playback starts after the first chunk (a paragraph, or a few hundred characters of one) rather than after the whole text. Each chunk is announced with `rangeStart`, so callers that highlight along can.

Text arriving through the engine is not rewritten. Another app chose it, and it is nearly always prose already. Running it through the Markdown stripper removed asterisks from ordinary sentences, reshaped `snake_case` identifiers and deleted links entirely. An accessibility tool exists to say what is on the screen, so silently dropping a URL gives a false account of the content. Stripping also breaks the offsets: they would no longer point at text the caller can highlight. The app's own screens still strip Markdown, because there the Markdown is the user's own and reading it as prose is the point.

A read-aloud broadcast works for scripts and agents. `am broadcast -n org.pockettts.android/.SpeakReceiver --es text "..."` reads text with no window and no foreground app. The selection window cannot do this, because starting an activity needs a foreground caller. The receiver was built for a coding agent on the same phone: when a reply finishes, it is read aloud rather than copied, switched to and pasted. It takes three extras:

- `text`: the Markdown to read, up to 64 KB.
- `path`: a UTF-8 file to read instead of `text`, up to 1 MB.
- `append`: a boolean that queues the text behind the current read instead of replacing it.

The receiver is exported with no permission check and no intent filter. Any app on the device can send it a broadcast that starts a read, replaces the current one, or reads any file the app can open. Nothing is returned to the sender. See [reading Claude Code's replies aloud](docs/read-aloud-from-claude-code.md).

The scratchpad is a Markdown editor. Type or paste, then tap Speak. Markdown is stripped before speaking: `## Heading` is spoken as "Heading" rather than "hash hash Heading", `**bold**` loses its asterisks, and links are read as their text instead of spelling out a URL. Fenced code blocks are skipped unless "Read code blocks aloud" is on. Selecting a passage and tapping Speak reads just that passage, and the sentence being spoken is highlighted as it goes.

Pocket TTS has no fixed speaker table. A voice is a few seconds of reference audio that the model conditions on, so the stock voices and a wav you record yourself are the same kind of thing. The voice picker lists the 27 stock voices from [`kyutai/tts-voices`](https://huggingface.co/kyutai/tts-voices). Each row plays a short sample, because twenty-seven names say nothing about which one you want. Importing your own wav is hidden in this version (see below).

The sample is the voice's own reference audio rather than a synthesised sentence. That is the most direct answer to "what does this sound like", because the reference audio is the voice. It costs only the prompt download: no model bundle, no inference, no wait. A voice can be auditioned on a fresh install before the model has been fetched, and the download is the same one that selecting the voice would trigger later. The trade-off is that the sample is the raw prompt, so it carries the recording's room and pacing. The synthesised voice matches the timbre closely, which is the thing being chosen, but not the background. Where there is no recording to play, the sample is the voice reading that language's sample sentence on the phone instead. That covers reading in a language other than English, and the five native voices (Italian, Spanish, German, Portuguese and Dutch) whose recordings need a Hugging Face login. In that case the language's model must be downloaded, and the app says so when it is not.

The app reads in seven languages: English, French, German, Spanish, Italian, Portuguese and Dutch. Each language has a Kyutai model, which is the one the app reads with. The model chosen under "Choose language and model" sets the language the app reads in. The share sheet, a selection and the scratchpad carry no language of their own. As the system engine, the app reads each request in the language it asks for, using that language's model once it is downloaded. TalkBack moving into a French passage gets the French model. Each language keeps its own voice, starting on Kyutai's native one for it (Estelle for French). Every voice reads every language, in its own accent. Tables are announced in the language being read. Only one model is in memory at a time, so switching language costs a model load, about half a second.

How far each sentence's speaker may be drawn from the voice is set by the "Voice variation" slider. Until the slider is moved, the value is Kyutai's for the model in use (0.3 for every model offered today, from `pocket-speak/models.json`). "Use Kyutai's default" clears the override.

## Installing

Each push to `main` that changes the app or the engine builds a release APK and publishes it as a GitHub release. The newest one is always at <https://github.com/mycosavant/pocket-tts/releases/latest/download/pocket-tts.apk>. Open it on the phone and allow the install. Every APK is signed with the same committed key, so a new build installs over an old one and keeps its data. The release workflow keeps the ten newest `android-*` releases.

## Building

Requires JDK 17 and the Android SDK (compileSdk 35, build-tools 35.0.0).

```bash
cd android
echo "sdk.dir=$ANDROID_HOME" > local.properties
./gradlew :app:assembleDebug        # or: gradle :app:assembleDebug
./gradlew :app:testDebugUnitTest
```

The build compiles the engine from `../pocket-speak` first. The `buildNative` task in `app/build.gradle.kts` runs `cargo ndk`, so the build needs rustup's `aarch64-linux-android`, `armv7-linux-androideabi` and `x86_64-linux-android` targets, `cargo-ndk`, and the NDK. Pass `-PskipNative` to package the libraries already in `app/src/main/jniLibs` instead.

The debug APK lands in `app/build/outputs/apk/debug/`, and the release APK in the sibling `release/` directory. The debug APK is about 28 MB and the release APK about 23 MB. Most of that is the engine, built for `arm64-v8a`, `armeabi-v7a` and `x86_64`. Dropping ABIs from `abiFilters` in `app/build.gradle.kts` shrinks it. Until 2026-10 the debug APK was 85 MB, most of it ONNX Runtime; see `docs/pure-rust-engine.md`.

## The debug key is committed, on purpose

`app/debug.keystore` is in the repository, and both build types are signed with it. The usual advice is to keep signing keys out of the repository, and that advice is about keys that ship an app to a store. This key does not do that.

Left alone, the Android plugin signs debug builds with `~/.android/debug.keystore` and generates one if it is missing. A CI runner has no home directory worth the name, so every run generated a new key pair. Every APK this project published was therefore a different app as far as Android is concerned, because identity is the signing certificate, not the package name. The consequences were permanent rather than intermittent:

- No build could be installed over another: every attempt failed with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`.
- Keeping app data on uninstall, which `hasFragileUserData` offers, made the next install fail outright, because the retained data remembers the identity that wrote it.
- The only way through was to uninstall and discard the data, which meant re-downloading the model (98 MB for the sherpa-onnx bundle) on every sideload.

None of this is visible in the build file, which said "sign with the debug config" before and after. It is a property of the built artefact. `tools/check-signing.sh` reads the APK: it pins the certificate's SHA-256, and CI runs it against both APKs. Changing the key means changing that constant, and everyone holding the old build has to uninstall to take the new one.

Committing the key does give something away, though small. Anyone with this repository can build an APK that installs over yours. The platform's own debug key is public and identical on every machine for the same reason.

## One install, however many things ask for it

Two callers reached the downloader independently: the button on the main screen, and any read that reached the engine before the model existed. Neither knew about the other. They shared a partial-download file and a staging directory, so tapping Download and then selecting text in another app had them writing the same `.part` file and deleting each other's unpacked files.

The download also belonged to the activity that started it. A configuration change cancelled the coroutine, but the transfer has no suspension point, so it carried on in the background while the recreated screen said "not downloaded" and invited a second tap. The partial file was then deleted on the way out, so that tap started again from zero.

`engine/ModelInstall` now owns one shared attempt per model. Anything that asks for a model joins the attempt already in flight, and a screen that goes away leaves it running. Progress is a `StateFlow`, so a recreated screen can pick it up. Removing the sharing fails `ModelInstallTest`.

The download resumes. The largest model is 219 MB. Without a `Range` request, a drop partway through throws away everything already fetched. On a connection that drops regularly, the download never finishes, because each attempt starts over from the beginning. A partial file now survives a network failure. It is deleted only when its sha256 does not match the pinned value, because resuming a bad file would never succeed. Each file is checked against its sha256 before it replaces anything.

## Keeping the model, and the voices

Everything lives in internal storage, so uninstalling used to delete all of it: a model download of up to 219 MB and, more important, every voice the user had recorded or imported, which exists nowhere else. `android:hasFragileUserData` makes the uninstall dialog offer to keep the data. That is the only way to offer the choice.

The backup rules exclude the model directory and two retired model paths, and nothing else. Cloud backup allows an app 25 MB. Including a model of up to 219 MB would make the backup fail as a whole, taking the settings and the voices with it. Every model lives under `pocket-tts/models`, which the rules exclude as a whole, so a model added later is excluded too. The rules name the directory as a literal string, because XML cannot reference a Kotlin constant. `BackupRulesTest` checks that the string matches `ModelManager.MODELS_DIR`.

Until 2026-10-03, "Install from a file" accepted sherpa-onnx's `.tar.bz2` archive. The app streamed it straight from the content URI and checked for the expected ONNX files before replacing anything. The engine swap ended that. The button is hidden, and `ModelManager.installFromArchive` refuses with a message that says to use Download. An archive format for pocket-speak's bundle is not defined yet (`docs/engine-swap.md`).

## First run

The model is not in the APK. The app downloads the model when you tap Download on the main screen, or the first time a read needs it. The default is Kyutai's `english_2026-10`, 219 MB, unless another is chosen under "Choose language and model". Each file is checked against the sha256 that `pocket-speak/models.json` pins, and the files go into app storage.

Each voice is a state precomputed for that model, from 0.3 to 7.6 MB depending on the model. A voice is fetched the first time it is used. An upgraded install deletes the models that earlier engines read (sherpa-onnx's bundle, then the ONNX export of `english_2026-04`) once the new model has loaded. See `docs/pure-rust-engine.md`.

Inference runs on the CPU, using upstream's Rust implementation of the model through pocket-speak. The previous engine generated audio 3.7 to 5 times faster than real time on a Galaxy S25, with first audio 0.2 to 0.3 s into a chunk. On the same desktop CPU, the new engine is 2.4 times faster on the larger `english_2026-10` model and starts in a fifth of the time. Phone numbers for the new engine are still to come.

## How it works

As of 2026-10-03, the engine is pocket-speak's Rust core, not sherpa-onnx (`docs/engine-swap.md`). As of 2026-10, that core runs upstream's own Rust implementation of the model rather than ONNX Runtime (`docs/pure-rust-engine.md`). Where a section below describes sherpa-onnx's behaviour, it says so and is history.

```
selected text ──▶ ReadAloudActivity ─┐
scratchpad ─────▶ ScratchpadActivity ─┼─▶ Reader ─▶ StreamingPlayer (AudioTrack)
                                      │      │
other apps ─────▶ PocketTtsService ───┘      ▼
                  (system TTS engine)   MarkdownSpeech ─▶ TextChunker ─▶ PocketTts
                                                                            │
                                                     NativeEngine (JNI) ─▶ pocket-speak/crates/android
                                                                            │
                                                          pocket-speak/crates/engine ─▶ xn-ptts + upstream's rust/src
```

- `ui/ModelPickerActivity` chooses the model, and with it the language. It shows a heading per language. For English the choices are Best quality (`english_2026-10`, 219 MB), Balanced (`english_18m_2026-10`, 112 MB), Small (`english_7m_2026-10`, 67 MB) and Smallest (`english_2m_2026-10`, 47 MB). Each other language has one 219 MB model. A download asks first, showing the size and a warning on a metered connection. The model is put to use when its download finishes, even if the screen has gone. Each row is announced to TalkBack as a radio button with its state and what a double tap does. Deleting is a separate button.
- `speech/MarkdownSpeech` turns Markdown into speakable prose. A table is read the way a screen reader reads one: it is announced ("Table, 2 rows."), then read row by row under its column names. A table wider than four columns is announced with its column names and not read (issue #10). Inline code is always read, as words. The "Read code blocks aloud" setting covers fenced blocks only (issue #9).
- `speech/TextChunker` cuts text at sentence boundaries, so the first audio arrives quickly and stopping takes effect quickly. A chunk targets 200 characters and never exceeds 400.
- `engine/PocketTts` owns the single loaded model and serialises synthesis.
- `engine/PocketTtsService` is the system engine. `engine/SystemVoices` names its voices per language: `selected` and `alba` for English, as before, and `selected-fr` and `alba-fr` for French. A request is read in the language of its voice if it names one. Otherwise it is read in the language the request asks for, and failing that, in the app's reading language.
- `player/StreamingPlayer` writes float PCM to an `AudioTrack` with blocking writes. A blocking write slows generation to the speed of playback.
- `player/PlaybackService` is a foreground service, so reading continues after the floating window is dismissed. It holds audio focus while it plays.
- `player/AudioFocus` requests audio focus and gives it back. Without it, the app read over whatever was already playing, a phone call did not pause it (muting the media stream during a call is up to the vendor), and unplugging headphones kept the voice playing from the phone's speaker. A transient loss pauses and resumes. A permanent loss stops, because another app has taken over for as long as it likes, and a paused reader nobody asked for helps no one. The app declines ducking, because two voices at once cannot both be followed.

`MarkdownSpeech`, `TextChunker` and `WavReader` are plain JVM code and have unit tests in `app/src/test`.

## Stop has to be in the collapsed controls

The collapsed media player (the shade before it is expanded, and the lock screen, which cannot expand it) shows three controls. Stop used to be the fourth. The reasoning was that swiping the notification away already ended the read. A media notification for a read in progress cannot be swiped away, though, and on the device where the broadcast path was first tested there was no way to end a read from the shade at all. A reader started from a pocket needs a Stop control.

Two things draw those controls, and both changed. Before Android 13, the notification's own actions are drawn. The compact view shows back, pause and stop (`setShowActionsInCompactView(0, 1, 3)`). From Android 13, the system draws the media session's actions instead: previous, play/pause and next take the three slots. Stop can only be offered as a custom action, because the system has no button for `ACTION_STOP`. Custom actions appear only once the player is expanded. So `Transport` no longer publishes `ACTION_SKIP_TO_NEXT`. That leaves the next slot free for the first custom action, which is Stop. Forward becomes a custom action in the expanded view. The headset's "next" key is handled in `Transport`'s media button callback, because the framework forwards it to `onSkipToNext` only when that action is published. Forward was the control to demote, since it is the one that ends a read by accident.

## Saying the true thing while waiting

The first sound of a chunk takes the model a noticeable time to produce. During that time the reader was in the `Speaking` state, and every screen said "Reading aloud". That is most of what the wait feels like, and it is why the wait seems broken rather than slow.

`Speaking.audible` is false until a sample has actually been written. Until it turns true, the screens say "Composing the first sentence…". The wait is no shorter. The label stopped claiming that sound was playing.

## Following along

`Speaking` reports the character range of the chunk being read. Those offsets are into the stripped text, not the document. Stripping deletes syntax, so the offsets drift further from the document with every heading, link and asterisk. `MarkdownSpeech.toSpeakableWithSource` records where each block came from, and `Reader.spokenRangeIn` turns a range in the stripped text into one in the source.

The map is per block, not per character, because a block comes from a known run of source lines, which is cheap to track. Within a block, the range is narrowed to the exact sentence when that sentence survived stripping unchanged. That is the common case for prose. A rewritten line falls back to highlighting the whole block. That is less precise, but it is never wrong.

Most of the difficulty is in the setup rather than the algorithm. Anything that rewrites the string before the lines are counted shifts every offset after it. So HTML comments are blanked with spaces of the same length rather than removed, and line breaks are found by scanning rather than by normalising CRLF to LF. `MarkdownSourceMapTest` covers both. The CRLF test needed a second pass to be useful. The first version used text that survived stripping unchanged, so the exact-sentence search quietly corrected the drift, and the test passed against broken offsets.

## Two callers, one engine

One model lives in the process, and two things drive it: the in-app reader, and the system text-to-speech service other apps call. In the sherpa-onnx era, each side serialised its work per chunk, so with both active they took turns sentence by sentence, and a passage came out read alternately in two voices. Neither side knew the other existed.

Serialising whole utterances instead would replace the alternation with a wait. An app asking through Select to Speak would hang until a whole document had been read, which looks broken from the other side. So `engine/EngineTurn` makes the most recent request win. A caller takes a turn before it starts, and checks whether the turn has been superseded as it goes: per chunk, and in each audio callback, so a request that arrives mid-sentence does not wait for the rest of it. A superseded caller stands down, which is the only way to avoid two voices in one passage.

## A queue, where a turn will not do

`EngineTurn` arbitrates between the two things that drive the one model, and its rule is that the most recent request wins. That is the right rule for two callers that cannot see each other. It is the wrong rule for a caller sending several things to hear in order. An agent handing over one reply while the last is still being read would have each reply silence the one before it, and a run of replies would play as the last few words of the last.

So `Reader` has its own queue. `enqueue` joins it, while `speak` still replaces. Nothing about standing down for a stranger answers "after that one", and stretching `EngineTurn` to cover both would have made one counter mean two opposite things.

The queue is careful about whose it is. A `speak` that replaces the current read drops what was queued behind it, because those follow-ups belonged to the passage being abandoned. Carrying them over would play a replacement followed by the tail of the thing it replaced. A stop drops the queue too. Skipping forward off the end of the last chunk does not drop it. That is an ending like any other, and it is the one ending that does not come through the reading job completing, so the queue advances by hand.

## The reader has one state machine, and it is testable

`Idle` used to mean two things: "nothing has been asked for" and "the read is over". `stopLocked` published `Idle` between two utterances, and three screens each kept a boolean to tell the two apart. `StateFlow` conflates values, so whether anyone saw that transitional value was a race. The scratchpad overlay sometimes failed to appear on a second Speak, and the foreground service sometimes stopped and immediately restarted.

`Reader.State` now ends in `Finished`, `Stopped` or `Failed`, each carrying the utterance it belongs to and who asked for it. The three flags are gone. Identity matters as much as finality. The reader is process-wide, so a screen has to tell its own read's ending from an earlier one it never asked for. Otherwise the floating window would close itself on stale news before its own read had begun.

`player/SpeechEngine` and `player/AudioSink` exist so the reader can be tested. Before them, driving the reader meant loading a 98 MB model and an `AudioTrack`. That is why the part that had actually been wrong had no tests, and why `ActivityLaunchTest` reached `ensureModel` and quietly downloaded the model on every run of the unit suite.

Skipping is built on the same seam. An utterance keeps its chunks, its engine and its loaded voice, so skipping back costs re-composing one chunk and nothing else: no re-selection, no reload. Skipping back from the first chunk replays it. Skipping forward past the last chunk ends the read, which is what someone who keeps tapping forward means by it, and it starts the next queued read if there is one.

A chunk is the unit of skipping, and a chunk is not a sentence. It is a paragraph, or up to a few hundred characters of one, cut at sentence ends. The controls used to be labelled "a sentence". On a short reply (one paragraph, one chunk), a tap forward near its second sentence ended the read, which looks like the wrong button doing stop's job. The labels now say what the buttons do. A skip finer than a chunk would need the engine to return audio sentence by sentence, and the engine composes a chunk as a whole.

`GlassPanelViewTest` asserts the two things that were missing when the sliders went dead: where the panel thinks it is relative to its backdrop, and what colour it actually put on the canvas. Its predecessor asserted only that drawing did not throw. Robolectric's software canvas made the panel bail out before it reached the geometry, so the broken function was never called by any test.

`ActivityLaunchTest` drives every activity through its real lifecycle under Robolectric, and forces a measure and layout pass at a realistic size. A theme, a manifest entry or a layout can crash an activity as easily as a function can, and none of those are type-checked. One build shipped a theme that supplied a decor action bar to screens that also called `setSupportActionBar`. Another handed a Material `Slider` a value off its step grid. Both compiled, packaged, and threw as soon as the screen opened.

The layout pass is the part that earns its keep. Views validate themselves while sizing, so reaching RESUMED proves little. The slider bug passed an earlier version of these tests that stopped at `setup()`, because nothing had been laid out yet.

## The one thing here that cannot be fetched again

Since 2026-10-03, imported voices are hidden, not deleted. The engine is conditioned on precomputed embeddings and has no encoder to make one from a wav. So the picker and the system voice list leave imports out, and the import button is hidden, until cloning returns (`docs/engine-swap.md`). The files stay where they are.

A model re-downloads, and a setting is retyped in seconds. A voice somebody recorded exists in one directory and nowhere else. The failure worth guarding against is therefore not a crash but a quiet overwrite.

Imports used to share a directory with the downloaded prompts. A wav named `alba.wav` landed on exactly the path where the stock Alba prompt is cached. It replaced that prompt, and then vanished from the list, because anything named like a stock voice was filtered out. One action caused two losses, with no message.

Imports now live in their own directory, `pocket-tts/voices/imported` under the app's files directory. A name that collides with a stock voice or an earlier import gets a suffix rather than winning. Voices imported by an older build are moved across on first access. An import that was already named after a stock voice is indistinguishable from the prompt it replaced, so it is left to be downloaded again. `VoiceStorageTest` covers these cases.

A stock prompt is cached only if its size matches the size the voice ships as. `VoiceCatalog` records each prompt's exact size, and `ModelManager` treats a prompt of the wrong size as not cached. Before that check, an import named like a stock voice overwrote the stock prompt, and nothing fetched the real one again. Every "Alba" from then on was read in someone else's voice, silently.

## The model decides when it is done

The engine has no length for a chunk in advance. It runs the language model frame by frame and stops when the model's end-of-speech (EOS) head crosses a threshold. The EOS head is ignored for the first six frames, so a short chunk does not end before its first word. After EOS, the engine generates the frames the model recommends, or a text-length estimate plus two.

The head is wrong in one particular place: the end of a generation, when what remains to be said is a few very short sentences. This was found on the device in the sherpa-onnx era. A reply ending `One. Two. Three. Four. Five.` stopped after "Two". It was then reproduced on the same engine build and the same int8 bundle, with the stock Alba prompt, using a speech recogniser as the judge. The measurements below are from that sherpa-onnx build. The mechanism was measured rather than guessed:

- `... One. Two. Three.` lost "Three" in two runs out of three. `... Patching. Testing. Done.` said "done" twice in all three runs. `One. Two. Three. Four. Five.` on its own lost "Five" every time. The same runs with ordinary prose after them were read in full. It is the end of the text that is fragile.
- Letting generation run past the end-of-speech mark does not recover the words. After a genuine ending, the model does not fall silent. It repeats the last word (`five, five, five`) for as long as it is allowed, so there is no silence to trim to.
- The reference implementation pads very short inputs with leading spaces, because "the model does not perform well when there are very few tokens". The sherpa-onnx front end stripped that padding before the model saw it. Padded and bare runs gave identical results.
- The same short sentences joined with commas, as in `One, Two, Three.`, were read in full in every run and for every shape tried.

So `TextChunker` gives each chunk a `speech` form. A run of two or more sentences of two words or fewer at the end of a chunk is joined with commas. What the user sees and what is highlighted is unchanged, and a period becoming a comma is the whole of the difference in what is said. `MarkdownSpeech` already makes every list item and heading its own paragraph, so a list is never a run of short sentences inside one chunk. The join is for prose that ends in a countdown or a string of one-word statuses, which is how agent replies often end.

What it does not cover is recorded here so nobody has to re-derive it. The EOS head can also fire early inside an ordinary sentence. `Tests 1, 2 and 3 are green.` lost "green" in two runs of three. An enumeration of numbers reads to the model like a countdown ending. `Files A, B and C are updated.` and plain prose endings were never cut. The threshold is fixed in the engine, and nothing at this layer can tell a premature end from a real one without a recogniser listening. This is the model's limit. The tail rule is the part the app can reach.

## Measuring before optimising

Every performance question about this app depends on timings, and the build environment has no emulator or device to take them on. So `debug/Metrics` collects three numbers, and the Timings button on the main screen shows them. The dialog also shows the voice trace, and it has a share button.

- Time to first audio: the wait everybody feels.
- Generation speed on the first chunk: audio seconds composed per wall-clock second, measured by the engine alone. Below 1.0 means the model cannot keep up with its own playback.
- Underruns: whether composing the next chunk while the current one plays is enough to keep the speaker fed. Underruns rise only when the model falls behind real time.

`Reader` composes a chunk ahead of the one being played. The lookahead is one chunk (`LOOKAHEAD_CHUNKS`). That is enough to hide the model's work behind the previous chunk's audio whenever the model is faster than real time. What the state publishes, such as the highlighted chunk and the point a skip counts from, follows the listener and not the engine.

`Metrics.generationRealTimeFactor` is taken on a read's first chunk only, so it cannot show a cost that repeats on every chunk. The per-chunk `first=` figure in the voice trace can.

An unmeasured value reports "not measured yet" rather than a confident zero, and `MetricsTest` enforces that. A count of "0 underruns" from a session that never played anything is not evidence, so the report does not present it as one.

## The voice, and why it kept changing

The voice used to change from sentence to sentence, and sometimes was not the voice that had been selected at all. There were three separate faults, and all of them sound identical from the outside.

The speaker is drawn from noise, and the draw was uncontrolled. Pocket TTS has no speaker table. It is conditioned on a few seconds of reference audio, and it samples a speaker in that audio's neighbourhood. The sample is a vector of Gaussian noise. In the sherpa-onnx era, the generator's seed was -1 by default, which means a fresh draw from a random device on every call. sherpa-onnx split each chunk into sentences and made one call per sentence, so a paragraph was read by a succession of slightly different people.

The app now sends a seed with each chunk. Both the seed and the temperature travel in the engine's configuration:

- A fixed seed, when "Keep one voice across sentences" is on, so every chunk draws the same speaker. The seed is 1. It is the whole of that feature. An earlier attempt kept the last six seconds of generated audio and fed it back as the next chunk's prompt. That could not work: it acted at chunk boundaries, while the redraws happened per sentence inside a chunk. It also replaced the selected voice with generated audio and never returned to the voice. That code has been removed.
- Temperature 0.3, rather than sherpa-onnx's 0.7. Kyutai moved the English model to 0.3 in this repository (`d108410`), with the reasoning "Human evaluations consistently prefer the English model at temperature 0.3 over the current default 0.7, and the change is free on every objective axis we measured." Each model's `default_temperature` in `models.json` carries that value. The slider exists so a user can set a different value.

The system engine served a voice chosen weeks ago. Android resolves an engine's default voice name once per client, then sends that name back on every later request. Returning a concrete voice id from `onGetDefaultVoiceNameFor` therefore pinned whichever voice was selected when Select to Speak or Chrome first bound. Changing the voice in this app did nothing for them until they were force-stopped. The engine now advertises one alias voice, `selected`, which is resolved on each request.

## One generation per chunk

The app asks the engine for one chunk at a time, and the engine generates each chunk in a single pass. `TextChunker` cuts text at sentence boundaries, targets 200 characters and never exceeds 400, so a chunk is usually a sentence or two. Each chunk costs one start of the model.

In the sherpa-onnx era, the engine split each chunk again and started over once per sentence. On the device where it was measured, that produced clipped first syllables and seams between sentences. Sending whole chunks removed almost all of the clipping there.

## One decode step

In the sherpa-onnx era, `numSteps` was the number of Euler steps the flow integration took for each generated frame. sherpa-onnx defaulted to 5. The reference implementation defaults to 1, in `lsd_decode` and in `default_parameters.py`. pocket-speak's engine decodes in one step for every flow type. The setting therefore has no effect now. The app still stores a `decode_steps` value (default 1) and passes it to synthesis, but `PocketTts.synthesize` ignores it. The "Decode steps" slider on the main screen is hidden for the same reason.

The default was changed from 5 to 1 on a device report, not on reasoning. At one step, a listener described the reading as closer to what the same model does on a desktop ("maybe a little more expressive, or more inflection") and noticed no other difference. One report called one step a touch slower to the first word. Fewer steps should mean less work per frame, so the `first=` figure on each chunk trace line is the measure that settles it. Chunk timings on that device move by about 90 ms between identical runs, which is about the size of the difference reported.

`docs/direct-ort.md` scopes what running the graphs directly would involve. `docs/owning-the-pipeline.md` asks the wider version of the same question.

## Who is actually speaking

`debug/VoiceTrace` records, for each read, the voice asked for, the voice found, whether that was a fallback, and the prompt file's size against the size that voice ships as. For each chunk, it records the voice id, a hash, the temperature, the seed (or `random`), and the time to the first sample.

```
[selection] asked for alba, got alba (958542 bytes as expected)
[chunk 0] alba prompt=0.00s hash=1a2b3c4d temp=0.30 seed=1 first=412ms
```

The first line answers the question the trace was built for. The bracketed name is where the read came from: `selection`, `scratchpad`, `agent`, `system` or `model-picker`. A fallback names itself with `FELL BACK`. A prompt that is not the file it claims to be shows a size that does not match. A client that sends a stale voice id shows a requested name that is not the current selection.

Two fields on the chunk line do not yet carry what their names say. `prompt=` is always `0.00s`, because the call that records the chunk passes no prompt length. `hash=` is the identity hash of the loaded native voice handle, not a hash of the prompt audio. The example hash above is illustrative.

The trace exists because several separate faults sound the same: a fallback to another voice, a stock prompt that is a different speaker, a client holding a stale voice name, and the model drawing a different speaker from the right prompt. The phone has no logcat within reach, so the trace is the only record. It shares the Timings dialog with the metrics.

## When it crashes

Sideloaded apps on a phone have no logcat within reach, and an app that dies silently reports exactly one thing: "it crashed". `debug/CrashLog` writes the last uncaught exception to app storage on the way down, and the next launch offers to share it. It chains to the previous handler rather than swallowing the crash, so the process still dies and the system dialog still appears. Only the evidence survives.

`CrashLog` cannot catch native crashes, ANRs or low-memory kills, because none of them unwind through the JVM. These look identical from outside to a Kotlin exception. `debug/ExitReasons` closes that gap by reading the platform's record via `ApplicationExitInfo`. It names the kind of death and, for a native crash on Android 12 and later, includes the tombstone. The reason code decides what to fix, and the two sources are shown together on the next launch.

## The JNI callback has to survive shrinking

The engine calls back into the app from native code by name and exact signature. Two things resolve those names: the native methods of `NativeEngine`, which are bound by their exported `Java_…` symbols, and `NativeEngine.Sink.onAudio`, which the engine calls for each chunk of audio with the JNI signature `([F)Z`.

Nothing in Kotlin references that signature, so nothing in Kotlin breaks when R8 renames or inlines the method. The build passes, and the failure appears only on a phone. A failed JNI lookup leaves a pending exception, the process aborts, and the crash surfaces as `REASON_CRASH_NATIVE` with no Java stack trace. In a release build the failure can also look like a silent read, because the engine receives nothing to play.

Two guards cover this. `app/proguard-rules.pro` keeps the native methods and the `onAudio(float[])` method of every `Sink` implementation. `tools/check-jni-callback.sh` reads the built APK with `dexdump` and fails if any of the nine native methods, or `onAudio ([F)Z`, is not declared. CI runs it against both the debug and the release APK. Both APKs are also uploaded, so the shrunk build is something that gets installed and tested, not something that is assumed to work. The source is not the artefact, and only the artefact shows what R8 kept.

The check reads declarations, not behaviour. It shows that the method exists in the APK, and it does not show that a lambda or other `Sink` implementation calls it correctly. A device run is still the only test of that.

## Glass, and where it cannot exist

Blurring depends on what is behind the panel, and the two cases have different answers.

Behind another app, only `Window.setBackgroundBlurRadius` can blur, because an app may not read another app's pixels. Whether that works is not something to reason about. `WindowManager.isCrossWindowBlurEnabled` answers it. On the device this was built for, it says no, and no value of any slider will produce a frost.

The app does not guess why it is unavailable. Accessibility's "Reduce transparency and blur" turns it off system-wide, battery saver turns it off too, and some devices never had it. The frosted panels that the system draws for itself, such as Samsung's quick settings and app-drawer folders, come from a privileged process and do not show whether the app can blur.

Those causes come and go, so the capability is followed with `addCrossWindowBlurEnabledListener` rather than read once when the screen opens. The appearance screen therefore serves as the test. Toggle the accessibility setting and the status line changes while you watch.

The read-aloud sheet used to pretend otherwise. It drew a near-transparent surface, found that it could not blur, and painted a fixed 0.94 opacity instead. That silently overrode the opacity the user had set, and it looked like a blur that had failed rather than a decision. The sheet is now an opaque surface at the bottom of the screen. Bottom placement is deliberate: a centred dialog would cover the paragraph that was just selected, which is the one thing the reader may want to keep in view. Where the platform supports cross-window blur, the blur is applied on top and the surface steps back to the chosen opacity. That is a bonus on devices that have it, not the design.

Behind our own content, `ui/GlassPanelView` gives a real `backdrop-filter: blur()` effect. The blur is confined to the panel's rounded bounds, and everything outside stays sharp. Each frame, the view records the backdrop into a `RenderNode`, offset so that the slice behind the panel lands at the panel's origin. It applies a blur `RenderEffect` to that node and draws it clipped to the panel, then the dim, then the tint, then the hairline. This works on any Android 12 device with no vendor opt-in, so the scratchpad overlay is where the glass is.

The offset is the key part, and getting it wrong fails silently. The panel has to sit outside its backdrop. Capturing an ancestor would draw the panel into its own backdrop and recurse, so the backdrop is always a sibling. An earlier version found the panel by walking up its parent chain looking for the backdrop, which a sibling is never on. The walk ran off the top of the hierarchy, the capture was declined, and every panel fell back to a flat tint with the configured opacity discarded. The appearance sliders seemed to do nothing, because for three of the four they did nothing. Both views are now located against the root they share, and the difference is taken. That is correct for siblings, cousins and ancestors alike.

The capture also has to start opaque, which was the second way this looked broken. A backdrop view usually has no background of its own. It lets the window's background show through, so recording it captures its text on transparency. Blurring that and compositing it puts the blurred copy on top of the original, which is still there and still sharp underneath. Only the dim and the tint appeared to work, and the panel looked as if its blur had failed. The panel now fills the capture with the window background before drawing the backdrop into it. The blur has something solid to work on, and the result covers what is underneath instead of ghosting over it.

These failures leave no visual sign. A panel that quietly gave up looks like settings that were never connected. So every frame records why it drew what it drew in `lastDraw`, and the appearance screen prints it. `lastDraw` did not catch either of the two faults above, because both reported `BLURRED` when the capture ran. It answers whether the blur path executed, which is a narrower question than whether anything blurred.

## The keyboard is not the window's problem any more

The scratchpad's Speak and Stop buttons sat under the keyboard. `adjustResize` is set in the manifest, but it has no effect here. A window laid out edge to edge, which `targetSdk 35` makes every window, is not resized when the keyboard opens. The window is given the keyboard's size as an inset and must handle it. Nothing warns about this, and the symptom is that reaching a control means dismissing the keyboard you were typing with.

`ui/Insets` now pads the bottom of the view by the larger of the system bars and the IME inset. The maximum is used, not the sum. The keyboard is drawn over the navigation bar, so its inset already includes the navigation bar. Adding the two would leave a gap exactly one navigation bar tall.

This is the technique [Haze](https://chrisbanes.github.io/haze/) uses on Android. Haze is a Compose Multiplatform library and this app uses Views, so the technique is borrowed and the dependency is not.

`ui/AppearanceActivity` exposes opacity, blur, dim and corner radius as sliders. The right values depend on the wallpaper-derived palette and the display density, and neither can be judged from source. Each label reports its value in the form the source code wants (`0.37f  ·  alpha 94  ·  37%`), and a button copies the whole set as Kotlin. The defaults in `Settings` are the values that came back from a device this way.

## Selecting text inside our own app

The `PROCESS_TEXT` item this app registers also appears in the selection toolbar of its own editor. So selecting a line in the scratchpad opened the floating window over a screen that already had controls for the same thing. That gave two sets of controls for one utterance, in two treatments, one of which could not frost.

`ReadAloudActivity` checks `callingPackage`. When it is called from this app, it tags the read as coming from the scratchpad, starts the service and finishes without showing a window. The scratchpad claims any read tagged that way, whichever route it arrived by.

## Edge-to-edge

`targetSdk 35` means Android 15 lays every window out edge to edge, so the system bars are drawn over the content rather than beside it. `ui/Insets.kt` applies those insets as padding. Each screen names the view that absorbs the top inset (its app bar) and the view that absorbs the bottom. Without that, the toolbar sits under the status bar and the first control is cut in half.

## Licensing

Read this before shipping anything built from the app.

- The models, their tokenizers and the voice states all come from Kyutai's [`kyutai/pocket-tts-without-voice-cloning`](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning) repository, under CC-BY-4.0. They are pinned in `pocket-speak/models.json`. Before the engine swap, the app read an ONNX export whose `LICENSE` file said CC-BY-4.0 while its README said non-commercial use only. That ambiguity went away with the swap.
- The engine is upstream's Rust code and [xn-ptts](https://github.com/gradium-ai/xn-ptts), which is MIT (xn-ptts: MIT or Apache-2.0). The build uses a pinned fork of it, as `pocket-speak/crates/engine/Cargo.toml` shows. `pocket-speak/crates/engine/NOTICE.md` has the details.
- `app/src/main/java/sonic/Sonic.java` is Bill Cox's Sonic, under Apache 2.0.
- Each voice has its own licence, listed per voice at [`kyutai/tts-voices`](https://huggingface.co/kyutai/tts-voices).
- Pocket TTS's prohibited-use terms apply (see the root `README.md`), in particular the ban on cloning someone's voice without their explicit and lawful consent. The import-a-wav feature makes that easy to do by accident.

## Prior art

If you would rather not build anything:

- [NekoSpeak](https://github.com/siva-sub/NekoSpeak) (MIT) is an offline Android TTS engine. It supports Pocket TTS alongside Kokoro, KittenTTS and Piper, with voice cloning. It registers as a system engine but does not add a text-selection menu item or a scratchpad.
- [TTS Util](https://github.com/Danesprite/tts-util-app) (Apache 2.0) adds share-sheet entries and can read the clipboard through whichever system engine is selected.
- [SherpaOnnxTtsEngineAndroid](https://github.com/jing332/SherpaOnnxTtsEngineAndroid) wraps sherpa-onnx as a system engine, with in-app model downloads.

Pairing NekoSpeak (or this app's engine) with TTS Util gets most of the way there without writing code. This app exists mainly to put all three pieces behind one tap.
