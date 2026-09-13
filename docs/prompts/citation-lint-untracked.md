# goinfer task: the citation lint must not red on files git does not track

> **Box:** any. No hardware, no models, no measurement — one Python file, one test.
> **Priority:** ahead of the queue and web-UI work, because both will produce new task docs and
> every new task doc spends its first hours untracked.
> **Read first:** `scripts/queue_citation_lint.py` (all of it — the module docstring is the design
> argument and you are extending it, not overriding it), `scripts/install-git-hooks.sh` (the
> TRACKED source that generates `.git/hooks/pre-push` — `.git/` itself is never committed, so the
> hook you can see in your own checkout is a per-clone artifact; edit the generator, never the
> generated file, or the change works on your machine and nowhere else, invisibly), and
> [`../tasks/task-work-queue-2026-09.md`](../tasks/task-work-queue-2026-09.md) §"How this doc was
> lost" / J0, which is the incident this closes.

## What happened

`docs/task-work-queue-2026-09.md` was written on 2026-09-12, left untracked for a few hours — the
normal resting state for a fresh design record in this repo — and **deleted on 2026-09-13**. Not by
a decision: by a workaround. The pre-push hook refused a push because the lint reds on that file's
citations, so the file was moved out of the tree to let the push through, and never moved back.
Commit `3e01502c` records the procedure in its own message, including the words "then restored".
It was not. The doc is gone from every branch, stash, dangling object and backup.

## Why the lint reds on it

`live_docs()` walks `ROOT.rglob("*.md")` — the filesystem, not git. An untracked markdown file
carrying `path:line` citations is linted like any other, its citations are absent from
`docs/QUEUE.md`'s generated index, and the run exits non-zero. The pre-push hook then refuses.

The only remedy the lint offers is to remove the file from the repository. So "take the new design
record out of the tree" becomes the standard move, performed under time pressure, by whoever is
trying to push something unrelated. That is the failure mode to close.

## The argument for the fix is already in the file

`live_docs()` skips `docs/internal/` with this reasoning, which you should read in full before
writing anything:

> That directory is DELIBERATELY UNCOMMITTED (.gitignore), so every path inside it resolves on
> exactly one machine and on none of the clones CI or anyone else runs. Linting it produces
> failures that are structurally unfixable from anywhere but here … and a lint that reds on files
> CI cannot see is a lint people stop reading.

Every clause of that applies to **any** untracked file, not to one directory. The exclusion was
written for a path when it is a property of tracking. Generalise it to the property.

## What to build

1. **`live_docs()` skips paths git does not track.** One `git ls-files -z` call against `ROOT` up
   front, into a set; skip anything not in it. Not one subprocess per file. Ignored files fall out
   for free (they are untracked), which means the existing `docs/internal/` special case becomes
   redundant — **leave it in place anyway**, and add a line saying it is now belt-and-braces, so
   the reasoning stays attached to its example.

2. **Print what was skipped. Do not skip silently.** One line, always, when the set is non-empty:

   ```
   note: 2 untracked doc(s) not linted — commit them to bring their citations under the gate:
     docs/tasks/task-something-new.md
     docs/prompts/some-brief.md
   ```

   Silence would trade a false red for an invisible hole, which is the shape this repo keeps
   naming. The note is not a warning and must not affect the exit status.

3. **`--update` indexes only tracked files too.** Same set, same call. A generated index must never
   carry rows that resolve on one machine — which is exactly what happened on 2026-09-12: the index
   absorbed 11 citations from an untracked file (`fd538dc6`), then dropped them again on the next
   run (`6c019eb5`).

4. **Handle the no-git case** — if `git ls-files` fails (not a repository, git absent), lint
   everything as today and say so in one line. Degrading to the old behaviour is right; degrading
   silently is not.

5. **The pre-push refusal message gains one line — edited in `scripts/install-git-hooks.sh`'s
   heredoc (around its `"Re-point the citation, or run --update..."` line), NOT in
   `.git/hooks/pre-push` directly.** `.git/hooks/pre-push` is that heredoc's OUTPUT, regenerated
   fresh each time `bash scripts/install-git-hooks.sh` runs and never committed — an edit made
   straight to the generated file looks like it worked (your own next push shows the new line) and
   commits nothing, so nobody else's clone, and no future re-run of the installer on this one, ever
   sees it. Add: *"If the red is a doc you have not committed yet, commit it — do not move it out
   of the tree."* The hook is where someone reads advice at exactly the moment this went wrong, but
   the FILE that ships that advice to every clone is the installer script.

## Gate — mutation-checked, red before green

A test that goes red against the current script and green after:

1. Write a temp `.md` under `docs/` containing a citation that cannot resolve (a real file at an
   absurd line number, or an unresolvable SHA). Leave it untracked.
2. Run the lint. **Today: non-zero.** After the fix: **exit 0**, and stdout carries the note naming
   that file.
3. `git add` the file (no commit needed — `ls-files` sees the index).
4. Run again. **Exit non-zero**, and the failure names that citation. This half matters as much as
   the first: the fix must not become a way to keep a bad citation permanently invisible.
5. Confirm `--update` before step 3 does not write a row for the untracked file, and after step 3
   does.

Put it wherever the other script tests live; if there is no home for Python script tests yet, a
`scripts/test_queue_citation_lint.py` runnable by `python3 -m unittest` is fine — do not add a test
dependency for this.

## Constraints

- **No new Python dependency.** stdlib and `subprocess` git, as the script already does.
- **Do not weaken any existing check.** Tracked files lint exactly as before, including the
  reachability check, the sibling-repo resolution, and the `sha-lint: allow` escape.
- **Do not add a flag to opt out of the skip.** A `--lint-untracked` switch would restore the
  failure mode for anyone who reads a flag list.
- Keep the module docstring's register: it explains *why*, with the incident that motivated each
  rule. Add the 2026-09-13 loss as a short paragraph in that docstring — it is the second incident
  in this file's history and it belongs next to the first.

## Out of scope

- Anything about the *content* of the recovered work-queue doc. It has been rebuilt and committed
  separately at `docs/tasks/task-work-queue-2026-09.md`.
- Any change to what counts as a citation, or to the index format.
- Any attempt to recover the original file. It is gone; the rebuild is the replacement.

<!-- doc-reviewed: 2026-09-13 -->
