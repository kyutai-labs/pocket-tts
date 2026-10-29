# This fork

[kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) with two additions:

- **`android/`**: an Android app that reads text aloud in Pocket TTS voices. It
  takes text from the share sheet, a text selection, or a broadcast, and it can
  be the phone's system TTS engine. [android/README.md](android/README.md).
- **`pocket-speak/`**: the engine the app runs, and `pocket-speak`, the same
  engine as a desktop command that reads text from stdin.
  [pocket-speak/README.md](pocket-speak/README.md).

Everything else is upstream's and stays identical to it. `README.md` and
`AGENTS.md` describe upstream's Python package.

**Install the app:** <https://github.com/mycosavant/pocket-tts/releases/latest/download/pocket-tts.apk>
always points at the newest release the Android workflow built from `main`.

## One rule: upstream's files stay upstream's

The fork owns the paths in [.github/fork/owned-paths](.github/fork/owned-paths) and
nothing else:

| path | what |
|---|---|
| `android/` | the app |
| `pocket-speak/` | the engine, the desk CLI, their tools |
| `FORK.md`, `.github/fork/` | this file and the scripts below |
| `.github/workflows/android.yml`, `fork-*.yml` | the fork's CI |

Upstream never touches these paths, so a merge of upstream cannot conflict on
them. Every other file must stay identical to upstream's. The **Fork checks**
workflow fails a PR that changes one. When fork code needs something upstream
has, it builds on upstream's file from a fork path instead of editing it. If an
upstream file must change, list it in
[.github/fork/upstream-edits](.github/fork/upstream-edits) with the reason. Expect
a conflict whenever upstream edits that file.

## The engine builds on upstream's Rust code

The engine in `pocket-speak/crates/engine` compiles in four of upstream's
`rust/src` files unchanged: `config.rs`, `defaults.rs`, `text.rs`, and
`weights.rs`. It also embeds the model configs in `pocket_tts/config`. A merge of
upstream that changes any of these runs the Android workflow, whose `engine` job
runs the unit tests in `config.rs`, `defaults.rs`, and `text.rs`. Upstream has
no tests in `weights.rs`.

Two more files are ported rather than compiled in: `rust/src/model.rs` and
`rust/src/rng.rs`, listed in [.github/fork/ported](.github/fork/ported) with the
git blob each port follows. When upstream changes either file, **Fork checks**
fails the PR. The failure prints the `git diff` to review. After porting the
change, record the new blob in `.github/fork/ported`. The same check also fails
when the engine's xn-ptts pin differs from upstream's `rust/Cargo.toml`.
[android/docs/pure-rust-engine.md](android/docs/pure-rust-engine.md) has the
details.

## Merging upstream

The **Upstream watch** workflow runs every Monday at 06:17 UTC, and it can be run
on demand from the Actions tab. When upstream has commits the fork lacks, it keeps
one issue labelled `upstream-sync` up to date. The issue gives the number of
commits and says whether they merge cleanly or which files conflict. The workflow
closes the issue once the fork has every upstream commit. To run the same report
locally, use `.github/fork/upstream-status.sh`.

To merge:

```bash
git switch -c upstream-sync origin/main
git fetch https://github.com/kyutai-labs/pocket-tts main
git merge --no-ff FETCH_HEAD
.github/fork/check-boundary.sh && .github/fork/check-ported.sh
git push -u origin upstream-sync   # then open a PR
```

**Merge that PR with "Create a merge commit", never "Squash and merge".** A squash
records upstream's changes as the fork's own and forgets which upstream commit
they came from. The next merge then conflicts on every file upstream has changed
since.

## Releases

Each push to `main` that touches `android/` or `pocket-speak/` (or the other
paths in the workflow's trigger) builds the APK and publishes it as a release
tagged `android-<run>-<commit>`, marked latest. The workflow keeps the ten newest
of these releases and deletes older `android-*` releases along with their tags.
Releases made by that workflow do not trigger other workflows, because they use
the default `GITHUB_TOKEN`.

A release made by hand would start upstream's `publish-package.yml`,
`gh-pages.yml`, and `rust-release.yml`, which are upstream's PyPI, docs, and
binary workflows.
