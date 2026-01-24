# Task Context: NumPy vs Rust Parity Audit

Session ID: 20260123-numpy-parity
Created: 2026-01-23
Status: in_progress

## Current Request
Perform an exhaustive, end-to-end comparison of every NumPy function against the pure Rust port in this repo. Use the same real-world datasets for each feature in both implementations, record parity/gaps, and update `rust-numpy/GAP_ANALYSIS.md`. Create GitHub issues for gaps without lane labels and avoid duplicate issues.

## Requirements
- Maximum search coverage: multiple explore + librarian agents in parallel.
- Use direct tools: Grep, ripgrep (rg), and ast-grep (sg) for exhaustive search.
- Compare behavior using identical real-world data for each feature.
- Save findings to `rust-numpy/GAP_ANALYSIS.md` in table form.
- Search existing issues before creating any new ones (no duplicates).
- Do not add lane labels to created issues.

## Decisions Made
- Track work under issue #348.
- Worktree: `ralph/issue-348-gap-analysis-validation` in `.worktrees/issue-348-gap-analysis-validation/`.

## Files to Modify/Create
- rust-numpy/GAP_ANALYSIS.md - parity table and findings.
- .tmp/context/20260123-numpy-parity/bundle.md - delegation context.

## Static Context Available
- /home/grant/.config/opencode/context/core/standards/documentation.md
- /home/grant/.config/opencode/context/core/standards/code-quality.md
- /home/grant/.config/opencode/context/core/standards/test-coverage.md
- /home/grant/.config/opencode/context/core/workflows/task-delegation.md

## Constraints/Notes
- Follow repo rules: no commits on main; all changes map to issue.
- Keep comparisons deterministic and reproducible.

---
**Instructions for Subagent:**
Load this context file first. Identify NumPy APIs and their Rust equivalents, note mismatches, and return findings with file paths and suggested table entries for `rust-numpy/GAP_ANALYSIS.md`.
