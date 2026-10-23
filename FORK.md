# This fork

[kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) with two things on top:

- **`android/`**: an Android app that reads text aloud in Pocket TTS voices, from the
  share sheet, a text selection, a broadcast, or as the phone's system TTS engine.
  [android/README.md](android/README.md).
- **`pocket-speak/`**: the engine the app runs, and `pocket-speak`, the same engine as a
  desktop command that reads text from stdin. [pocket-speak/README.md](pocket-speak/README.md).

Everything else is upstream's, unchanged. `README.md` and `AGENTS.md` describe upstream's
Python package and are upstream's files.

**Install the app:** <https://github.com/mycosavant/pocket-tts/releases/latest/download/pocket-tts.apk>
is the newest build of `main`.

## One rule: upstream's files stay upstream's

The fork owns the paths in [.github/fork/owned-paths](.github/fork/owned-paths) and
nothing else:

| path | what |
|---|---|
| `android/` | the app |
| `pocket-speak/` | the engine, the desk CLI, their tools |
| `FORK.md`, `.github/fork/` | this file and the scripts below |
| `.github/workflows/android.yml`, `fork-*.yml` | the fork's CI |

Upstream never touches these, so merging upstream cannot conflict on them. Every other
file stays identical to upstream's, and the **Fork checks** workflow fails a PR that
changes one. When fork code needs something upstream has, it builds on upstream's file
from a fork path rather than editing it. If an upstream file truly must change, list it
in [.github/fork/upstream-edits](.github/fork/upstream-edits) with the reason; expect
it to conflict whenever upstream edits that file.

## Merging upstream

The **Upstream watch** workflow runs every Monday. When upstream has commits the fork
lacks, it keeps one issue labelled `upstream-sync` up to date: how many commits, and
whether they merge cleanly or which files conflict. It closes the issue once the fork
has caught up. Run it on demand from the Actions tab, or locally with
`.github/fork/upstream-status.sh`.

To merge:

```bash
git switch -c upstream-sync origin/main
git fetch https://github.com/kyutai-labs/pocket-tts main
git merge --no-ff FETCH_HEAD
.github/fork/check-boundary.sh
git push -u origin upstream-sync   # then open a PR
```

**Merge that PR with "Create a merge commit", never "Squash and merge".** A squash
records upstream's changes as the fork's own, forgets which upstream commit they came
from, and the next merge then conflicts on every file upstream has changed since.

## Releases

Each push to `main` that touches the app or the engine builds the APK and publishes it as
a release tagged `android-<run>-<commit>`, marked latest. Releases made by that workflow
do not trigger other workflows. A release made by hand would also start upstream's
release workflows (PyPI publish, docs, Rust binaries), which fail or do nothing useful in
this fork.
