# Eleven Utterances

*End-to-end test results, 11-12 Sep 2026, build 52 verified on device.*

*These results were measured on the sherpa-onnx engine. On 2026-10-03 the app
replaced it with pocket-speak's Rust engine (`engine-swap.md`), and on 2026-10-23
with upstream's pure-Rust engine (`pure-rust-engine.md`). The engine-level findings
below describe the sherpa-onnx engine. The script and proot findings still apply.*

Testing the Termux to Pocket TTS read-aloud path on the device for the first time.
Seven test cases, eleven reads. Everything that failed on first contact was in the
shell script or the proot environment, and none of it was in the app. The four app
defects found along the way were then traced against the engine's own source and
reproduced off the phone. The observations held, but the explanation for them did
not. Both PRs are merged, and build 52 has now been run on the phone: every fix
confirmed, one item untested.

| | |
|---|---|
| Device | SM-S938U1, Android 16 |
| Found on | build 46, d7b66ae |
| Verified on | build 52, merged main |
| Voice | alba, 958,542 bytes |
| Script fixes | [PR #7](https://github.com/mycosavant/pocket-tts/pull/7), merged |
| App fixes | [PR #8](https://github.com/mycosavant/pocket-tts/pull/8), merged |

## Verdict: the path works, the diagnosis needed the engine's source, and the device agrees

Transcript to script to broadcast to `Reader` to `PlaybackService` to audio, with a
notification, lock-screen controls, and the screen off. The four script bugs are
fixed in #7. All four app defects (3.1, 3.2, 3.4 and 3.5) are fixed in #8. The
fifth item, 3.3, turned out to be by design, and its labels were changed. Every fix
has now been confirmed on the phone. The headset "next" key remains untested.

## What the device showed on build 52

| Check | Build 46 | Build 52 |
|---|---|---|
| tail: `... One. Two. Three. Four. Five.` | stops after "Two." | full read |
| first audio, table-heavy piece | 4229 ms | 377 ms |
| generation speed | 0.83-1.38x | 2.29-3.41x |
| audible gap before a long paragraph | yes | none |
| audio underruns | 0-1 | 0 |
| stop in the collapsed shade | absent | present |
| stop on the lock screen | absent | present |
| headset "next" key | - | untested |

One prediction from the trace did not survive the listening. Chunks 4 and 5 of the
boundary read showed `first=3879ms` and `first=2346ms`, which looked like one-chunk
lookahead running out on an uneven chunk. There was no audible gap, though. A larger
chunk takes longer to produce its first sample, and that time sits under the
previous chunk's audio, which is exactly what the producer/consumer split is for.
`first=` is not silence.

## Status

| Test | Result | Note |
|---|---|---|
| 1. Script finds and extracts the reply | Pass | after fixing 1.1 |
| 2. App reads a fixed string | Pass | after fixing 1.2 |
| 3. `speak-last`, screen on | Pass | after fixing 1.3 |
| 4. Screen off, phone down | Pass | the case the receiver exists for |
| 5. Long reply, queued with `append` | Pass | 3 broadcasts, order kept, none dropped |
| 6. Transport controls | Pass | on build 52; headset key untested |
| 7. Stop hook registration | Unblocked | held until 3.1 was safe; it now is |

The `. . . .` ellipsis case, flagged in the handoff as a live prediction nobody had
checked, is verified fixed on the device. Full length, clean, correct inflection.

## Script bugs: 4 found, 4 fixed, merged in #7

All in `android/tools/speak-last.sh`. Patched in the installed copy first, then
verified again from the repo copy before the PR.

### 1.1: `tail` from a pipe hangs under proot

The blocker. The script never reached the app. The hang is unconditional, not
size-dependent, and specific to `tail` reading stdin.

```
printf 'a\nb\n' | tail -1      ->  hangs forever (two lines)
tail -1 file                   ->  fine
cat file | head -1             ->  fine
cat file | awk 'END{print}'    ->  fine
```

Fixed with `awk 'END{print}'`. A second change dropped the `grep -v` in favour of
`select(length > 0)` inside jq. That part was optional: grep from a pipe works fine,
and the misdiagnosis came before `tail` was isolated.

### 1.2: `am broadcast` needs an explicit `--user` on Android 16

```
SecurityException: Permission Denial: getIntentSender asks to run
as user -2 but is calling from uid u0a396; this requires
INTERACT_ACROSS_USERS_FULL or android.permission.INTERACT_ACROSS_USERS
```

`-2` is `USER_CURRENT`. termux-am passes it by default, and the platform now refuses
to resolve it from an app uid. `--user current` fails identically. In #7 the id is
`CLAUDE_TTS_USER`, default 0, so a work profile is not stuck. The same flag is
applied to `am start` in `pocket` mode, which was not exercised on the device.

### 1.3: Process substitution is broken under proot

```
speak-last.sh: line 143: /dev/fd/63: No such file or directory
Reading 1095 characters in Pocket TTS (0 broadcast(s)).
```

proot binds `/dev/fd` to the reader's `/proc/self/fd`, so the shell's fd 63 is not
there to open, and the loop body never ran. The dangerous part is the second line:
it reported success while sending nothing. Fixed with a temp file, and `sent=0` is
now an error.

### 1.4: Broadcast failures are invisible and the message misleads

```
am broadcast ... >/dev/null 2>&1 \
  || die "...am broadcast failed - is Pocket TTS installed?"
```

The `2>&1` swallowed the SecurityException in 1.2 entirely, and the message sent the
reader to check the install. The failure now prints the exception.

**What turned out to be true:** `am` cannot confirm delivery. It prints
`Broadcast sent without waiting for result` and exits 0 for a receiver that does not
exist, and also for a package that is not installed. It exits 1 only when the
platform refuses the broadcast outright. The message sent the reader to check the
install, but the script could not detect a missing install at all. `Timings`,
`utterances read`, is the only proof a read happened, and the doc now says so.

## App defects: what the engine showed

The follow-up read `offline-tts-pocket-impl.h` from sherpa-onnx 1.13.6, the build
the app shipped, and ran that same engine off the phone. It used the same int8
bundle, the same Alba prompt, decode steps 1, temperature 0.3, the app's own fixed
seed among three, and Whisper base.en as the judge. The numerics were x86, so the
exact seeds do not transfer to the device. The mechanisms do.

### 3.1: A run of short sentences at the end loses the tail (fixed, verified on device)

| Text | Tail | Build 46 | Build 52 |
|---|---|---|---|
| ... from Termux. One. Two. Three. Four. Five. | 5 runts | stops after "Two." | full |
| same, warm app | 5 runts | stops after "Two." | - |
| ... ordinary. One. ... Five. And this final ordinary sentence... | ordinary | full | - |
| ... plenty of audio to chew through. Done. | 1 runt | full | - |

Length is not the variable. Position is. Mid-text runts are harmless. One trailing
runt is harmless. A run of them loses the tail.

**What turned out to be true:** It is the model, not the reader. The engine has no
length for a chunk in advance. It runs the language model frame by frame and stops
when the model's own end-of-speech head crosses a hard-coded threshold, plus three
frames. On a trailing run of very short sentences that head fires early, inside a
single generation. Nothing in `Reader` concludes anything, so there was no wait to
add.

| End of chunk | Whole, as shipped | Joined with commas |
|---|---|---|
| ... One. Two. Three. | lost "Three" 2/3 | full 3/3 |
| ... Patching. Testing. Done. | "done" twice 3/3 | full 3/3 |
| One. Two. Three. Four. Five. (alone) | lost "Five" 3/3 | full 3/3 |
| ... It compiles. Yes. | lost "Yes" 1/3 | full 3/3 |
| same runs, prose after them | full 4/4 | - |
| ... Done. (one short sentence) | full 4/4 | left alone |

The obvious alternatives were measured and rejected. Letting generation run past the
end-of-speech mark recovers nothing. After a real ending the model does not fall
silent. It repeats the last word, `five, five, five...`, for as long as it is
allowed, so there is no silence to trim to. The reference implementation pads very
short inputs with leading spaces. sherpa-onnx strips them before the model sees
them, and padded and bare output were identical to the sample. One generation per
short sentence costs a full voice-conditioning pass per word.

**The fix:** `TextChunker` gives each chunk a `speech` form in which a trailing run
of two or more sentences of two words or fewer is joined with commas. What is shown
and highlighted is untouched. Lists and headings are already their own paragraphs,
so this only touches prose that ends in a countdown or a string of statuses, which
is how agent replies end.

**Known limit, in the README** (`android/README.md`): the same head fires early
inside an ordinary sentence when it holds a run of numbers. `Tests 1, 2 and 3 are
green.` lost "green" in 2 of 3 runs. `Files A, B and C are updated.` never did. The
threshold is the engine's, and the reference uses the same value. Nothing at the app
layer can tell a premature end from a real one.

### 3.2: Stop is missing from the notification and lock screen (fixed, verified on device)

No stop control was visible at all. The handoff predicted exactly this: a
MediaSession custom action added in PR #4, and the control most likely to be
missing.

**What turned out to be true:** #4's custom action was correct and bought no visible
button. Two things draw the controls, and both hid Stop. Before Android 13 the
notification's own actions are drawn, and the compact view showed back, pause,
forward. Stop was the fourth, dropped on the theory that swiping the notification
away already ends the read. A media notification for a read in progress cannot be
dismissed that way. From Android 13 the system draws the session's actions instead.
Previous, play/pause and next fill the three collapsed slots, and a custom action
appears only once the player is expanded, which the lock screen cannot do.

**The fix:** the compact view is back to back, pause, stop. `Transport` no longer
publishes `ACTION_SKIP_TO_NEXT`, which hands that slot to the first custom action,
and that is Stop. Forward is a custom action in the expanded view. The headset's
"next" key is taken in `onMediaButtonEvent`, because the framework only routes it
when the action is published.

**Confirmed on build 52:** back, pause and stop are the three drawn in the collapsed
shade and on the lock screen. Forward appears once expanded, and stop ends audio and
notification together. The headset "next" key was not tested. Bluetooth was off for
the audio work, and a false negative was not worth the risk.

### 3.3: Skip forward terminates the read (by design, relabelled in #8)

Pressed after skip back and several pause/resume cycles, near the start of the second
sentence. Audio and notification ended together, which is the outcome stop should
have produced, from the wrong button.

**What turned out to be true:** this was not a bug, and it was not related to 3.1. A
skip moves by chunk. A chunk is a paragraph or up to about 400 characters of one, cut
at sentence ends. The test text was one chunk, so forward from its second sentence
was forward past the last chunk. That ends the read by design, and starts the next
queued piece if there is one. The buttons said "a sentence", but they never moved by
one. They now say "Skip back" and "Skip forward", and the docs say what a chunk is.
A skip finer than a chunk would need the engine to give audio back per sentence, and
it composes a chunk whole.

### 3.4: A stripped table costs a four-second silence (fixed, verified on device)

```
[agent] ...                       <- queue piece 2, mostly a markdown table
[chunk 0] ... first=4229ms        <- against 786ms for piece 1
```

**What turned out to be true:** `first=` is not priming. It is the model's entire
pass over the chunk. `GenerateSingleSentence` runs every frame of the language model
over a chunk before the decoder produces a sample, then hands audio back in pieces of
just over a second. So the number scales with the chunk, and a chunk that is one
stripped table row still pays one full pass. That alone would only delay playback.
The pipeline change below is what made it a silence.

**On build 52:** the same heading-table-paragraph text gave *time to first audio:
377 ms*, against 4229 ms. The table is dropped silently, with no audible pause before
the paragraph behind it.

### 3.5: Generation speed is marginal on this device (fixed, verified on device)

Across eleven reads: 0.83x to 1.38x real time. The app flagged the low one itself:
`generation speed: 0.83x real time - slower than playback`.

```
[chunk 0] alba prompt=9.98s ...
```

**What turned out to be true:** The reader fed the speaker from inside the engine's
callback, and the write blocks when the buffer is full. So chunk N+1 could not begin
until chunk N had been *heard*. The silence before every paragraph was the model's
full pass over it, less about two seconds of buffer. Underruns could not see that,
because a track that is simply not written to between chunks is not "running dry" in
a way that counts. The 0.83x figure was also measuring the speaker as much as the
model. It timed a call that spent most of its length waiting on playback. Back-solved,
the model itself runs at roughly 1.4 to 4x real time on this device.

**The fix:** `Reader.play` is now a producer and a consumer. The engine's callback
drops pieces into a channel. A writer feeds the sink, and the producer may be one
chunk ahead of the one being written, bounded by a semaphore on chunks rather than a
bounded channel. The engine's callback cannot suspend, and a callback blocked waiting
for room would hold the engine exactly as the blocking write did, with nothing to
wake it after a stop. Published state follows the listener, not the engine.
`generation speed` now measures the engine alone, and underruns now mean one thing:
the model fell behind real time.

**On build 52:** `generation speed` read 2.29x and 3.41x on the same texts that gave
0.83 to 1.38x before. That brackets the back-solved 1.4 to 4x estimate. Underruns
stayed at 0.

## What was done

1. **Upstream the four script fixes.** [#7](https://github.com/mycosavant/pocket-tts/pull/7),
   plus the finding that `am` cannot confirm delivery.
2. **Compose the next chunk while the current one plays.** This closes 3.4 and 3.5.
   Three new `ReaderTest` cases pin the pipeline. The second chunk reaches the engine
   while the first is still being written, and the third does not. A stop writes
   nothing composed ahead. A failure on chunk two arrives after chunk one's audio.
3. **End a chunk where the model can.** `TextChunker.speech` joins a trailing run of
   short sentences. This closes 3.1 as observed. The numbers-in-a-sentence limit is
   documented, not fixed.
4. **Put Stop in the collapsed controls on both paths.** This closes 3.2.
5. **Say what the skip buttons do**, in their labels and in both docs. This is 3.3.

All of it is in [#8](https://github.com/mycosavant/pocket-tts/pull/8): 236 unit tests
and lint, green locally and in CI. Both PRs are merged, and build 52 was run on the
phone.

## Still open

- **The headset "next" key.** `ACTION_SKIP_TO_NEXT` is no longer published, so the
  key is taken in `onMediaButtonEvent` instead. That is untested, because Bluetooth
  was off. If Stop draws correctly but the key does nothing, the slot was freed and
  the handler did not land. Report those separately.
- **Test 7, the Stop hook.** It was held until 3.1 was safe to hit on every reply. It
  now is, so registering it is a one-line change to `~/.claude/settings.json`.
- **`CLAUDE_TTS=pocket`**, the `am start` activity path. #7 applies `--user` there
  too, unexercised.
- **A genuine reply over 32,000 characters.** The queue was exercised by lowering
  `CLAUDE_TTS_LIMIT` to 400, which runs the same code path.
- **Numbers inside an ordinary sentence.** Documented as an engine-level limit, not
  fixed. `Tests 1, 2 and 3 are green.` is the reproduction.

---

Source: `/root/dev/pocket-tts-test-findings.md` (outside this repository, not checked).
The companion *Read Aloud Wiring* is not in this repository either. Engine measured:
sherpa-onnx 1.13.6, pocket-tts-int8-2026-01-26, Whisper base.en as judge.

Committed from the published Claude artifact of the same name (2026-09-12), so this
account lives in the repo next to the code it describes, not only in a cloud
artifact.
