# Reading Claude Code's replies aloud

One command on the phone reads the reply that just finished, in the app's own
voice, with the app's own transport controls in the notification and on the lock
screen.

It replaces the manual loop: `/copy`, switch apps, paste into the scratchpad,
tap read.

## How it fits together

```
Claude Code (proot Ubuntu, under Termux)
  └─ speak-last.sh          finds the transcript, pulls the last reply
       └─ am broadcast      →  SpeakReceiver
                                 └─ Reader.speak(..., Source.Agent)
                                 └─ PlaybackService   notification, lock screen,
                                                      media session, audio focus
```

Nothing about the reading is new. `SpeakReceiver` takes the same route through
`Reader` and `PlaybackService` that the floating "Read aloud" window takes. So
`MarkdownSpeech` strips the Markdown exactly as it does for the scratchpad, and
the four transport controls are the same four.

The receiver exists for one reason: **it works with the screen off.** Starting an
activity needs a foreground caller, so a read triggered that way is only as
headless as whoever asked for it. A broadcast has no such requirement.

## Install, on the phone

```bash
# 1. the script
mkdir -p ~/.claude/hooks
cp android/tools/speak-last.sh ~/.claude/hooks/
chmod +x ~/.claude/hooks/speak-last.sh

# 2. the slash command
mkdir -p ~/.claude/commands
cp android/tools/claude/speak.md ~/.claude/commands/

# 3. the bare name (see below)
mkdir -p ~/.local/bin
ln -sf ~/.claude/hooks/speak-last.sh ~/.local/bin/speak-last
```

`jq` is the only dependency: `apt install jq` inside proot, or `pkg install jq`
in Termux.

## Using it

| what you type | notes |
|---------------|-------|
| `!speak-last` | shorter to type; the symlink in step 3 is what makes the bare name work |
| `/speak`      | the slash command |

**Both cost a turn.** An earlier version of this document claimed a leading `!`
ran the command without consuming one. That is wrong. It was corrected after
watching it on a real session: the command's output is appended to the transcript
and the assistant responds to it, exactly as a slash command does. Pick based on
typing comfort, not cost.

The symlink in step 3 is what makes the bare name work. A shell alias would not,
since the command does not run in an interactive shell.

`~/.claude/keybindings.json` rebinds keys within Claude Code. It does not run a
shell command, so there is no single-keystroke version of this today.

## The last reply is the one on screen

Typed at the prompt, the last `assistant` record in the transcript is the reply
that just finished, the one you are looking at. That is the intent, not an
off-by-one.

## Modes

Set `CLAUDE_TTS`:

| mode | what it does |
|------|--------------|
| `broadcast` | **default.** `SpeakReceiver`. No window, works with the screen off. |
| `pocket` | `ReadAloudActivity`, which opens the floating sheet. Needs the caller in the foreground. |
| `engine` | `termux-tts-speak`. For when the app is not installed. The only mode that strips Markdown in the script, because it is the only one whose far end will not. |
| `print` | Prints what it would read and stops. Start here when nothing comes out. |

## Long replies

A reply over about 32,000 characters (`CLAUDE_TTS_LIMIT`) is split on paragraph
boundaries and sent as a queue. The first piece replaces whatever is playing, and
the rest arrive with `append=true` and read in order. Without that, each piece
would silence the one before it, and a long reply would play as the last few words
of its last paragraph.

The receiver also takes a `path` extra, and that is the better answer on paper,
except that a file written by Termux is not a file Pocket TTS can open. The two
run as different uids with private storage, and neither `/data/local/tmp` nor
`/sdcard/Android/data` bridges that on a current Android without root. `path` is
there for the app's own files and for `adb`-driven testing. The queue is what the
phone actually uses.

## Also a Stop hook

The script reads hook JSON from stdin when it is given any, so the same file can
be registered as a `Stop` hook with no changes:

```json
{ "hooks": { "Stop": [ { "hooks": [ { "type": "command",
  "command": "~/.claude/hooks/speak-last.sh" } ] } ] } }
```

## Controls, and the battery setting they need

A read started this way begins with Pocket TTS in the background, and Android
refuses a foreground service started from there. The read still plays, because the
reader does not depend on the service. Without the service there is no
notification, no lock-screen or quick-settings controls, no audio-focus claim, and
nothing stops the system from killing the read mid-sentence.

Allowing Pocket TTS to run unrestricted in the battery settings is what grants it.
On a Samsung device: *Settings → Apps → Pocket TTS → Battery → Unrestricted*. With
that set, a broadcast read appears in the shade with skip back, pause and stop, and
forward once the player is expanded. The lock screen shows back, pause and stop as
well (`eleven-utterances.md`). The headset buttons are handled in code, including
"next", but the "next" key has not been tested on a device.

A skip moves by chunk: a paragraph, or up to about 400 characters of one, cut at a
sentence end where one is available. A short reply is one chunk, so forward on it
ends the read, and starts the next queued piece if there is one.

Timings says which of the two happened: `playback service: started`, or
`refused (...)` with the reason. So "it read but there were no controls" has an
answer on the device rather than in a log nobody can reach.

## When nothing comes out

The whole point of this path is that there is no window, which also means there is
nothing on screen to tell you what went wrong.

```bash
# 1. Is it the script or the app?
CLAUDE_TTS=print ~/.claude/hooks/speak-last.sh

# 2. Does the app read a fixed string?
am broadcast -n org.pockettts.android/.SpeakReceiver --es text "Receiver check."
```

`am` cannot confirm delivery. It prints `Broadcast sent without waiting for
result` and exits 0 whether the receiver handled it, does not exist, or the package
is not installed at all. A successful-looking run therefore proves only that the
intent left Termux. `utterances read` in Timings is the confirmation.

It *does* exit 1 when the platform refuses the broadcast outright. On Android 16
that happens if the user id is left implicit:

```
SecurityException: Permission Denial: getIntentSender asks to run as user -2
but is calling from uid u0aNNN; this requires INTERACT_ACROSS_USERS_FULL
or android.permission.INTERACT_ACROSS_USERS
```

`-2` is `USER_CURRENT`, which termux-am passes by default and which the platform
no longer resolves from an app uid. The script sends an explicit `--user 0`.
Override it with `CLAUDE_TTS_USER` on a work profile or secondary user. `--user
current` is not an alternative, because it maps to the same -2.

Then open Pocket TTS and tap **Timings**. That is the instrument, not `logcat`.
Since Android 4.1, an app cannot read other apps' logs, and Termux is an ordinary
app. From Termux, `logcat` shows Termux and nothing else. Reading the app's own log
needs adb, over wireless debugging to the phone itself if you want it.

Timings answers most of it directly:

| Line | What it settles |
|------|-----------------|
| `utterances read` | Whether the broadcast reached `Reader` at all. Force stop the app first so the count starts from zero and there is no doubt which read you are looking at. |
| `time to first audio` | A cold start pays for loading the model and the voice prompt inside the read. In the 2026-10-03 debug run the model load was 0.62 s and first audio came at 1.1 s. The second read gives the warm number. |
| `generation speed` | Below 1.0x real time means the model cannot keep up with its own playback, and gaps are arithmetic rather than bad luck. Measured on the first chunk of a read only. |
| `[chunk n] ... first=NNNms` | Wall clock from asking for that chunk to its first sample. That is the model's whole pass over the chunk. The engine composes every frame before the decoder produces a sample, so the figure scales with the chunk's length, and a chunk that is one stripped table row still pays a full pass. The reader composes each chunk while the previous one plays, so this only becomes a silence when it is longer than the previous chunk's audio. Expect the first chunk of a cold read to be far larger than the rest. |
| `playback service` | Whether the read got its notification and controls; see above. |
| `[agent] asked for X, got Y` | Which source asked, and whether the voice that answered is the one that was asked for. `FELL BACK` or a prompt size that is not the expected one is the whole diagnosis. |

A voice trace line reading `prompt file missing` on the *first* read after an
install is the prompt being fetched during that read. It is recorded after the
fetch, so seeing it on a warm read means the file really is gone.
