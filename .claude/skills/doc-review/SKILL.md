---
name: doc-review
description: "Review a goinfer docs/*.md task, scoping, plan or spec doc: verify its claims against the tree, archive it if complete or superseded, list unblocked remaining work, and give a worth-it opinion."
---

# goinfer doc review

Input: one path under `docs/` (a `task-*.md`, `scoping-*.md`, `plan-*.md`, `spec/*.md`, or a
reference doc). Output: a verdict, the disk changes made, the remaining unblocked work, and an
opinion on whether that work is worth doing for goinfer. Francis commits; you do not.

Repo: `~/tmcode/goinfer` on the Mac (`aikit` is the sibling). In Cowork it is mounted at
`$HOME/mnt/goinfer` — use `device_bash`, never the cloud shell. In Claude Code it is the cwd.

## 0. Rules that override the doc's own words

- **Never trust a doc's status line.** `scoping-lfm2.md` said "build NOT started" on the day the
  build landed; `roadmap.md` said "v0.3.0 is tagged" at v0.17.2; a framing note claimed "a landed
  v1.0" when no such tag exists. Date the status line (`git log -1 -- <doc>`) and check every
  claim against the tree.
- **The generated matrices are the truth for what runs where**: `docs/capability-matrix.md`
  (families) and `docs/hardware-matrix.md` (per-backend residency; columns are
  Family|CPU|WebGPU|CUDA|Metal). `testdata/parity_manifest.json` is the truth for validation
  status. `git tag --sort=-v:refname | head -1` is the truth for the version.
- Read the doc in full before deciding anything. Read `CLAUDE.md` §citation lint before touching
  any file the lint scans.

## 1. Ground the doc (do all of these; they are cheap)

```
git log -5 --format='%h %ad %s' --date=short -- <doc>
grep -rn '<basename>' --include='*.md' docs README.md CLAUDE.md | grep -v '^docs/completed'   # live refs
grep -c '<doc path>|' docs/citation-index.md                                              # citation-index rows keyed to it
python3 scripts/doc_review_staleness.py                                                   # is this doc even due for a look?
```

The last one is a cheap triage step, not a substitute for the rest of this skill: if the doc
already carries a `<!-- doc-reviewed: YYYY-MM-DD -->` footer (§3b below) and the script reports
it FRESH, nothing the doc cites has moved since it was last checked — skip it and spend the time on
a doc the script flags STALE or with NO FOOTER instead. A FRESH result is not proof the doc is
correct, only that nothing it points at has changed; still ground and verify normally on anything
the script can't cover (a claim with no `path:line` citation behind it, a verdict about hardware or
a decision, an aikit-side citation — see the script's own docstring for why those aren't checked).

Then, per claim in the doc:
- **"X shipped / is open"** → grep the code for the symbol, kernel file, flag or env var; check
  the matrices; check `docs/queue-*.md` and `docs/completed/queue-*.md` for the item's G/P/E/Q
  entry and whether it was archived; check `docs/completed/` for a sibling that already closed it.
- **"X is blocked on Y"** → confirm Y still holds (hardware, upstream binding, a decision, an
  aikit version — `grep aikit go.mod`).
- **"X is impossible / not our fight"** → grep for a later doc that says "corrects", "supersedes",
  "withdrawn", "reversed" and names this doc. Verdicts get overturned within days here.
- **Numbers** → find the current row in `docs/benchmarks.md` (TL;DR table at the top) or
  `docs/ollama-chase.md` §1; note the date on both.
- **Path:line citations** → the lint owns them; do not hand-edit line numbers.

## 2. Classify

| verdict | meaning | action |
|---|---|---|
| **COMPLETE** | every item shipped, killed, measured, or moved to a doc that owns it | archive to `docs/completed/` |
| **SUPERSEDED** | the verdict or premise was overturned elsewhere | archive to `docs/completed/`, with a retraction status block naming what overturned it and where |
| **LIVE, STALE** | still the owner of open work, but carries wrong claims | correct in place (dated notes, retraction-in-place); move to `docs/tasks/` if not there already |
| **PARKED** | open, blocked, trigger named | correct in place if needed; move to `docs/tasks/parked/` if not there already; restate the trigger in the report; check the trigger still holds |

A doc with one live section and the rest complete: check whether the live section is a summary of
something that already has a canonical home (`spec/`, an audit R-item, a queue entry). If so,
correct it in place, then archive and re-point citers to the canonical home.

**Where a doc lives, by verdict — this repo's migration convention, started 2026-09-13.**
`docs/task-*.md` files no longer sit flat in `docs/` root: a COMPLETE/SUPERSEDED verdict archives
to `docs/completed/<name>.md` exactly as step 3 below always did; a LIVE-STALE or
verified-accurate-but-open verdict moves to `docs/tasks/<name>.md`; a PARKED verdict moves to
`docs/tasks/parked/<name>.md`. `docs/prompts/*.md` briefs follow the same three-way split (archive
delivered/superseded ones, leave genuinely open ones in `docs/prompts/`) but there is no
`docs/prompts/parked/` — an open prompt just stays in `docs/prompts/`. If the doc is already in the
right place for its verdict, no move is needed. Do not invent a fourth location.

## 3. Archive procedure (COMPLETE or SUPERSEDED — house conventions, follow exactly)

1. Prepend the standard archival header, copied verbatim from the top of
   `docs/completed/task-cuda-cgofree-spike.md` (the blockquote starting **ARCHIVED — a record, not
   instructions**). 21 archived docs were missing it as of 2026-09-12; do not add a 22nd.
2. Below it, a dated **Status** block: what closed it (commit, date, doc), what the tree says now,
   and the remainders the doc did not own. If the doc's verdict was wrong, say so plainly and name
   the correcting doc — retraction in place, at full value.
3. Strip imperatives per `docs/parity-coverage-policy.md` §"archiving a doc strips its
   imperatives". Lightest correct touch: one sentence in the status block saying every instruction
   below is a record of what was asked, not a task.
4. `git mv docs/<name> docs/completed/<name>`. If the Cowork VM refuses (stale `.git/index.lock`,
   or `rm` returns "Operation not permitted"), write the new file, move the original to
   `goinfer/_to_delete/`, and tell Francis; git pairs them as a rename at commit.
5. **References.** Live docs that name it: light load → fix each to the `completed/` path (the
   2026-09-12 precedent); heavy load or many `path:line` cites into it → leave a pointer stub at
   the old path in `docs/plan-still-slow.md`'s shape (title, **Moved.**, new path, one line on what
   became of it, "pointer kept because other pages link here"). A stub is a live doc: **no
   `path:line` citations in it.** Archived docs that name it: leave alone.
6. `docs/README.md`: if the doc had a map row, move or remove it; set the archive heading count to
   `ls docs/completed | wc -l` (others archive concurrently — recount, never increment).
7. **Citation lint**: `python3 scripts/queue_citation_lint.py`, exit code read directly, never
   through a pipe. `--update` for drifted line numbers. Never delete a citation to make a red go
   green. Rows keyed to a moved doc retire on their own. The Cowork VM's Python is 3.10 and the
   script needs 3.12 — if it fails to parse, say so; the pre-push hook on the Mac runs it.
8. Stage by path. The tree almost always has unrelated uncommitted work.

## 3a. Move procedure — LIVE-STALE or PARKED, relocating within the live tree

Apply corrections in place first (dated notes, retraction-in-place, per the doc's own established
convention — a blockquote correction beside the stale claim, not a silent rewrite). Then, if the
doc is not already at the right location for its verdict:

1. `git mv docs/task-<name>.md docs/tasks/task-<name>.md` (LIVE-STALE / accurate-but-open) or
   `docs/tasks/parked/task-<name>.md` (PARKED). No ARCHIVED header, no status-block-strips-
   imperatives step — this is still a live, actionable doc, just relocated.
2. **Fix every downstream reference, thoroughly — this is the part most likely to be skipped.**
   Grep the WHOLE repo (not just `docs/`) for the basename and any bare prose mentions of it —
   markdown links, backtick prose citations, and Go/shell code comments (the citation lint does
   not check code comments pointing at docs, so they rot silently if missed). Fix each one to the
   new path:
   - A same-directory `docs/*.md` markdown link becomes `tasks/task-<name>.md` (or
     `tasks/parked/task-<name>.md`).
   - A `docs/prompts/*.md` or `docs/measurements/*.md` file, one level down, needs `../tasks/...`.
   - If the moved doc itself has outbound relative links (to `measurements/`, sibling `task-*.md`
     docs, `examples/`), they need `../` prepended since the doc is now one level deeper — and if a
     link target has ALSO already moved into `docs/tasks/` or `docs/tasks/parked/`, that link
     becomes same-directory (or a `parked/` subpath) instead of `../`.
   - **Leave these alone even if they cite the old path** — frozen/historical by this repo's own
     convention: `docs/completed/*.md`, `docs/measurements/*.md`, anything under `testdata/`,
     `CHANGELOG.md`. Editing a frozen record to match a live move defeats the point of freezing it.
3. `python3 scripts/queue_citation_lint.py`, exit code read directly. `--update` if the move
   drifted a citation's index key. **A `--update` run from inside a fresh git worktree can
   mis-resolve aikit SHA citations** it can't find via the default sibling-checkout lookup, flipping
   SHAs that were fine before your edit to UNRESOLVED — that is a worktree-location artifact, not a
   real regression; set `GOINFER_SHA_LINT_REPOS=<path to the aikit checkout>` before `--update`, or
   revert just those rows, rather than letting the artifact get written into the file.
4. Do not touch `docs/README.md`'s counts on every single pass — if several docs are moving in the
   same sweep, recount once at the end rather than after each one.

## 3b. Every pass, regardless of verdict — the doc-reviewed footer

`docs/tasks/*.md`, `docs/tasks/parked/*.md`, and `docs/prompts/*.md` each carry a trailing
`<!-- doc-reviewed: YYYY-MM-DD -->` footer, checked by `scripts/doc_review_staleness.py` (step 1).
**Add it if missing, update the date if present, on every pass that verifies the doc — including a
pass that changes nothing at all.** A doc confirmed accurate today and left untouched still needs
its footer bumped to today, or the staleness script has no way to know it was actually looked at;
this genuinely means a no-op review still produces a one-line commit (`git add` the file, commit
just the footer). `docs/completed/` files do **not** get this footer — they are frozen and excluded
from the staleness check the same way they are excluded from the citation lint.

## 4. Remaining work

For every open item in the doc: **unblocked / blocked (by what) / superseded**, and which live doc
owns it now. The finding that matters most is usually an **orphan** — an open item no live doc
owns — or a **false block** ("tested and negative" where the test measured a different mechanism;
check what was actually measured before repeating the claim). Name orphans explicitly and say
where they should land (a `queue-*.md` entry, an existing task doc, or nowhere with a reason).

## 5. Worth-it opinion

Judge against the repo's stated priorities, in this order, and say which one decided it:

1. **The promotion gate** — goinfer is not promoted until it is at least as fast as Ollama on the
   machines people have (owner decision 2026-09-11; `docs/roadmap.md`). Performance leads, breadth
   is the tiebreaker.
2. **The niche** — single-user, batch-1, consumer hardware, one cgo-free static binary
   (`docs/positioning.md`). Server-scale features are the wrong weight class.
3. **Campaign conventions** (`CLAUDE.md`, `docs/ollama-chase.md`): projection bands before
   measuring; kill-or-earn measurement before a task doc; split-before-build; a family must get one
   whole forward on hardware we own or it is parked, not queued; weights posted, not announced.

Output one of: **do now / measure first (name the one number and its kill threshold) / park (name
the trigger) / kill (name what replaced it)**. Give the fact that would flip the call.

## 6. Report format

Short, bottom line first, prose with a table only where items are compared:

1. Verdict line (COMPLETE / SUPERSEDED / LIVE-STALE / PARKED) and the one-sentence reason.
2. What happened to each item the doc scoped (table if more than three).
3. Remaining work, unblocked vs blocked, with owners and orphans.
4. Worth-it call and the deciding priority.
5. Disk changes made (paths), what Francis must do (commit; run the lint if it could not run
   here), and anything noticed but deliberately left alone.

Register: humble, direct, no "honestly"/"honest". vscode-claude is "them", never "it". If the
request was a question ("should we…?", "is this obsolete?"), answer it and offer to act; if it
was an instruction ("archive it", "fix it"), act and report.

## Known drift to expect

- `docs/README.md` counts (families, archive size, prompts) drift constantly; recount from source.
- `docs/queue-performance.md` had two different P24s as of 2026-09-12.
- Old docs say "27 families", "three attention axes", "hybrids stay staged", "no fp8 support",
  "the resident path is stateless", "MatmulBTW4A8Batch doesn't exist" — all false; the live
  `docs/roadmap.md` §Superseded has the replacements.
- Metal is cgo-free via purego `objc`; WebGPU is the one cgo backend. Docs from before August
  get this backwards.
