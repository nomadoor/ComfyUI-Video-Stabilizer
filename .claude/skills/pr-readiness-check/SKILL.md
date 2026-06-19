---
name: pr-readiness-check
description: Use before committing, pushing, or opening a PR to catch missing files, mixed staging, validation gaps, and PR metadata mistakes.
---

# PR Readiness Check

Use this skill right before commit/push/PR creation, especially after adding new
files, validation scripts, nodes, docs, or workflows.

## Checklist

1. Inspect branch and file state:

```bash
git status -sb
git diff --stat
git diff --cached --stat
git ls-files --others --exclude-standard
```

2. Verify new files are intentionally included.

- If code imports a new module, confirm the module is tracked or staged.
- If validation references a new script, confirm the script is tracked or staged.
- Do not rely on `git diff` alone; it does not show untracked files.

3. Check staged scope before commit.

```bash
git diff --cached --name-only
```

- Stage explicit paths when the worktree is mixed.
- Do not use broad staging if unrelated user changes are present.

4. Run validation in the environment that has runtime dependencies.

```bash
python3 scripts/validate_repo.py
python3 -m compileall nodes scripts
```

If plain `python3` lacks declared runtime dependencies such as OpenCV, also run
the checks with the available ComfyUI venv and report the environment difference.

5. Before opening a PR:

- Confirm the PR title has no `[codex]`, `[claude]`, or similar tool label.
- Link related issues without closing them unless the user explicitly asks:
  use `Related: #N`, not `Fixes #N` / `Closes #N`.
- Confirm the pushed branch contains the latest local commit.

## When Reporting

Mention:

- branch and latest commit
- whether untracked files remain
- validation commands and results
- any skipped review finding and why
