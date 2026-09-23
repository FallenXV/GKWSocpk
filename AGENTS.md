# Working guidelines

- Preserve unrelated changes and snapshot history.
- Verification is one command: `python tests/smoke.py`. Run it once before
  reporting a code change as done and report its last line. It takes a few seconds.
- Do not write, extend, or restore tests, test plans, fixtures, or evidence files
  (including under `tests/`, `docs/`, or `artifacts/`) unless the user asks.
  If you think a change needs a dedicated test, say so in one line and move on.
- Do not edit `tests/smoke.py` to cover new features. Change it only if it breaks
  because of an intentional change to how the app starts or serves files.
- Pause and ask before deleting or rewriting data files, changing CSV/snapshot
  formats or CLI flags, large refactors, or when the requirement is ambiguous.
  Otherwise proceed.
- Keep reports short: what changed, the smoke result, anything you could not check.
