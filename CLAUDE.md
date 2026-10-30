# goinfer — agent notes

## Working in the tree

**Seven Go modules**: the root, `gpu/`, `cuda/`, `metal/`, `demo/agent/`, `sdktest/` (tests only, the official OpenAI and
Anthropic SDKs against the in-process server; never tagged) and `site/` (the website generator, built by `site.yml`).
`go.work` is gitignored and **mandatory** for cross-module work. A `GOWORK=off` build of a submodule resolves the root from
the proxy at its last published tag, so "method not found" on good code is expected between releases, not a bug.

**Build tags gate real code:** `realckpt` (heavy, real checkpoints), `goinfer_testhooks` (cross-module test seams),
`cuda` / `gpu` / `metal`. A change that compiles untagged can be broken under a tag. Vet the variants your change touches:
`go vet -tags realckpt ./...`, `go vet -tags 'cuda goinfer_testhooks' ./cuda/`.

**Two CI gates nothing auto-runs for you.** `gofmt -l .` across every module you touched. staticcheck: run
`go run ./cmd/lint`, which checks the binary is CI's pinned build (and prints the build command if not), proves it analyses
with a canary, runs CI's targets under `GOOS=linux` itself, and ends with a VERDICT line. An empty staticcheck result is
indistinguishable from one that never ran. Never combine `go run …staticcheck` with `GOOS=linux`: it builds a Linux binary
the Mac cannot exec, and the shell still reports exit 0.

**Product code reaches `main` through a pull request that merges itself; do not watch CI** (Francis, `docs/tasks/task-subsystems-2026-10.md` §12 PW1).
- Push your branch, open a PR, and at once, checks still pending, run `gh pr merge <n> --repo <owner/repo> --auto --merge`
  as its own bare command (no `cd`, no `&&`); then print `BRANCHES READY: #<n> <head sha>`. Never a plain `gh pr merge`
  without `--auto`. If `--auto` is denied, print `BRANCHES READY` and say so; the coordinator takes it to Francis. **Never
  route around a denial** (no other merge command, no API call, no asking another session).
- The PR need not be rebased onto the newest `main`; a conflict leaves it open. Nobody holds a turn open on its checks or main's CI, polls them, or
  schedules wake-ups for them. Read the result at the next natural point (`gh pr view <n> --json state`, `gh run list
  --limit 5`: your next push, the next task, always before a tag) and report a red the moment you see it; a main run can
  cover several merges, each green on its own.
- Branch protection does not enforce this (admins are exempt); the rule does. Only two pushes go straight to `main`: a
  docs-only change (`scripts/ci_docs_only.sh`: `docs/**`, `RELEASING.md`, `.claude/**`) and a release, per `RELEASING.md`.
- **If a hook holds a push you believe is wrong, stop and report it to Francis. Never `--no-verify`.** A background push job
  pushes a named commit (`git push origin <sha>:<branch>`), never `HEAD`, from a worktree no one else edits.

**No new production env reads.** `testdata/env_reads.txt` lists every `GOINFER_*` variable production code reads, and
`TestEnvVars_docAndCodeAgree` fails on one not listed. A new operator choice is a `decoder.Options` field with a per-model
accessor; a new diagnostic is a test hook. The list only shrinks (`docs/env-vars.md` § "Policy — where a new setting goes").

**A change to any guard threshold (swap, fit, memory) needs Francis's word, the same as a bypass flag.** The swap guard's
default is 512 MB. Nobara's plans set `swap_guard_mb` 4096 only as Francis's decision of 2026-10-17, and only until a build
carries the per-process guard (dba04134 or later); a plan then drops `swap_guard_mb` and takes the default. No other plan,
script or box takes 4096 on its own, and only Francis writes `swap_guard_approved_by` (the timed-run skill, § "Guard thresholds").

**Never `git add -A` / `git add .`.** It sweeps generated fixture metadata into git, and dir-only skip-guards then read a
fixture as present. Stage explicit paths.

**`main` moves under you.** `git fetch` and rebase before pushing. Changes you did not make in a file you are editing are
someone else's in-flight work: commit *your* hunks, leave theirs in the tree.

## Models and benchmarks

**Every timed run reads its checkpoint from `~/models` on the machine doing the timing; a row measured from `/srv/models`
or `/Volumes/` (the archive) is void.** `bench_peer.py` refuses those paths; `docs/benchmarks.md` § "Model storage" is the authority.

`docs/benchmarks.md` is provenance-gated: a number enters a table only with machine, checkpoint+quant, greedy/seed, pinned
versions, date, thermal note and local-disk path; read its Methodology first. Peer comparisons go through
`scripts/bench_peer.py`, same-session interleaved; `scripts/bench_compare.sh` is for goinfer-vs-goinfer only (it times
in-process kernels, so beside a peer it divides a kernel throughput by an end-to-end one). A cross-session ratio is not a
ratio (`docs/measurements/noise-registry.md`, TE3). CUDA rows are anchored to an NVIDIA driver version; a driver change
needs a deliberate re-anchor, not a carry-forward.

## Code comments

**A Go comment says what is true now:** what the code does, its contract (ownership, units, concurrency, what a caller must
check), the reason in a sentence or two where the shape is not obvious, and every guardrail, with a pointer to its evidence.
**It does not carry the history:** how a bug was found, dates, commit ids, sessions, reviewers, tracker ids as content,
measured figures, retractions and open work go to the commit message, the measurement or task doc, or the queue. Point at
docs by path and heading and at code by declaration name, never `file.go:NNN`. The test for a sentence: would someone
changing this code next week act differently because of it? Keep it. Does it explain how we came to know? Move it.

**Length is part of the rule.** Say each thing once, at the declaration it binds (an accessor comment is about six lines).
Do not restate the signature or a sibling's comment, list which backends or callers take a path, or narrate the body; a
longer design explanation goes to the doc it cites or `docs/code-notes/<package>.md`. The comments around you do not set
the style: do not rewrite comments in a file you are only passing through, or add to the history in one. Directive comments
(`//go:build`, `//go:embed`, `//nolint`, `// Output:`) are code. `scripts/comment_census.py diff` (pre-push, a warning)
lists added comments that carry a history marker.

## Tests

Each rule's story is in the model-bring-up skill draft, `docs/skills-draft/model-bring-up/SKILL.md`.
- **A SKIP IS NOT A PASS.** `go test` prints `ok` when every test skipped; confirm with `-v` and read for `--- PASS`.
- **A new fixed-resolution real-checkpoint golden, or any over ~1 MB, is written `.json.gz`** (Python `gzip.open(OUT, "wt")`)
  and read via `decoder.ReadGoldenJSONForTest` (in-package: `readGolden`); small `.json` goldens stay. Measurement raw data
  over ~1 MB (ncu CSVs, sweep dumps) is committed `.gz` from the start, and the record names the `.gz` path.
- **A unit test that supplies its own calling convention proves nothing about the callers:** when a contract depends on how
  often or under what condition a component is called, test it through its caller.
- **A doc comment claiming coverage is not coverage:** the body must contain an assertion naming the thing claimed.
- **`go test -v ./a/ ./b/` prints nothing until `./a/` finishes:** one package per invocation, `tee` to watch.
- **A minimal repro can be minimal in exactly the dimension that hides the bug:** ask what is identically zero at that size
  (n=1 kills interaction terms, softmax over one element is 1.0, RoPE at position 0 is trivial).
- **Difference per layer against HF's `output_hidden_states`, not from final logits.** Its last row is after the final
  norm, and a matching argmax means nothing.
- **A per-family struct literal will omit a field, and zero looks legal:** check at the chokepoint every family passes
  (`resolveArchitecture` → `validateResolved()`), and test that guard directly.
- **A tight real-checkpoint bar can catch a bug in the reference:** sanity-check `persistent=False` buffers computed in
  `__init__` (RoPE tables, chiefly) for NaN and magnitude before calling a disagreement goinfer's.
- Heavy tests need `GOINFER_HEAVY_TESTS=1` and their assets (`testdata/assets.json`); `go run ./cmd/gate` is the runner.

## Testing tiers

**Register every gate at the lowest tier that can answer it,** pre-registered at every tier: tier 0 (seconds; deterministic
counters, bit-identity on tiny fixtures) by default; tier 1 (one small real model, by day, wall time stated first); tier 2
only for a published number or a small-effect decision. Never re-measure an unchanged arm; stop when the answer is clear;
fit-check every load before a timed run or night job. **Tier 1 GPU runs hold the timing lock exclusively**
(`timing_lock.py run`), so two sessions' GPU checks do not share the card. Never-swap gates, nobara's fit checks, the desktop
compositor's CUDA context and `cuMemAlloc`'s 2 MiB rounding: the timed-run skill, § "Testing tiers".

## Day loop

**The day loop is seconds:** `python3 scripts/touched.py` builds, vets and tests what the diff touches, runs every
source-scan test that reads a changed file (a new one goes in `scripts/touched_scans.py`, or `scripts/test_touched.py`
fails), and ends with a VERDICT line. No `go run ./cmd/gate quick` (the pre-tag tier), whole-package `go test` or CUDA suite
by day. Once, right before a push and after the last merge of `main`: the citation lint, and the parity refresh
(`scripts/refresh_parity_hashes.sh`) when a non-test `decoder/*.go` file's code changed. Fix a red CI run by its one test.
A change that forces a long run by day is a modularity bug: say so, naming the edge.

## Run budget

**Read the timed-run skill draft, `docs/skills-draft/timed-run/SKILL.md`, before any timed run, night job or gate
registration;** it has the full rules and the gate-tiers table. In short: timed runs go by day, through the timing lock
(`timing_lock.py run`, or a harness that takes it), one per box, with the start and expected finish time stated first.
Night (`night.py`) is only for a Mac job the fit guard refuses while Francis's apps are open, or one over about 2 hours.
Any other build or test you run holds the lock shared (`timing_lock.py run --shared --label <what> -- <cmd>`); never delete
the lock file; a command that touches the other box checks its lock in the same command and ends if held. Comparisons run
A B B A in one session on one box; a cell started under heavy load is void (re-run, listed, never averaged). "Run it now"
from Francis means run it now.

## Gate tiers

Every gate names its tier, instrument, stopping rule and cost basis (`docs/measurements/noise-registry.md` §8); the table
is in the timed-run skill, and the release tier is `RELEASING.md`'s.

## Long-running work

Anything over a few minutes you launch is detached (`setsid nohup … </dev/null &`, PPID 1); on macOS, `launchctl submit -l
<label> -- /bin/bash -lc "…; launchctl remove <label>"` (without the remove, the job restarts on exit and overwrites its
log). Archive logs somewhere durable, never `/tmp`. Prefer committed increments to one big commit.

## Citations and the pre-push hook

`scripts/queue_citation_lint.py` is a pre-push hook that refuses a red push (index: `docs/citation-index.md`); it ends with
a VERDICT line, so never read its exit code through a pipe. A backtick-quoted path under a **gitignored** directory is a
forbidden destination, in any repo: describe the record in prose. **Cite code by declaration, never by line**
(`chat/chat.go:Template.Render`); a dated record (audit, review, measurement write-up) is instead pinned with
`<!-- citations-at: <commit> -->` (the commit its numbers were true at) and its `path:line` citations are checked there;
`path:line` in an unpinned doc is red. `CANNOT SEARCH — no module cache` is a missing download (after an aikit bump, or in
a fresh worktree): `python3 scripts/queue_citation_lint.py --pinned-modules | xargs go mod download`, never `--update`.

<!-- citation-lint: allow-path file.go the placeholder in this repo's own descriptions of the citation forms (file.go:NNN, file.go:NN), not a citation -->
<!-- citation-lint: allow-path path.go the placeholder in the symbol-citation examples (path.go:Name, path.go:Type.Method), not a citation -->

## Task docs: the ledger, and closing as you go

A live task doc carries one ledger under its status block: `<!-- ledger -->`, then `| id | state | next | evidence |`, one
row per item (open, active, blocked, decision, done, killed, moved: the row names where; prose below is history). **Change
the row in the same commit as the work;** a move leaves a link back in the destination. **The commit that closes the last
row archives the doc, in the same push:** `python3 scripts/task_docs.py archive <doc> --status <file>` prints the plan; read
it, then add `--apply --stage`. A new task doc starts with a ledger, and commit subjects name items by an ID the doc owns
(`KV4`, not a bare `S4`). `task_docs.py rank` and `check` list recent and closable docs.

## Measurement discipline

The measurements are the product; each rule's evidence is in the timed-run skill, § "Measurement discipline".
- **Difference matched observations; do not pool them** when variants can be interleaved on the same input.
- **A bit-identical speed change ships even under its registered bar** when all four hold: bit-identical outputs; scratch and
  allocated memory not higher; faster on at least 9 of 10 timed pairs; it replaces code rather than adding a second path (a
  test-only switch back is not one). The registered bar still decides any change that alters outputs.
- **A lossless opt-in feature merges at median ≥ 1.05× with at least 9 of 10 timed pairs faster** (bit-identical, or inside a
  registered fidelity gate); its PR states device and host bytes and any effect on the longest context that fits.
  **Changing a default** also needs the gain judged against its memory and Francis's call. An investment bar is not a merge bar.
- **Include the do-nothing arm. Pre-register the decision rule** for anything that will be argued about, with an explicit
  *ambiguous → parked* band. **Commit negative results** with the same care; move a bar only with a mechanism.
- **Read prior art before re-deriving,** for what it concluded: a doc may say something narrower than its headline.
- **A marginal stopping rule assumes a curve shape:** when it is unknown, run the full ladder, and pre-register two things
  that can disagree. **A guard that inverts under its own condition is worse than none:** key a budget on a quantity you
  compute, never the OS's account of what remains. **Quote a slice as a slice,** not as the model.
- **A retraction is not done until it reaches every page quoting the figure:** `grep` the figure with its unit, then and there.

## Releases

`RELEASING.md` is the authority: five modules, a two-step tag, and a GitHub Release on the root tag only as a non-optional
final step. Do not restate its version numbers; read them with the command it gives. `gh` uses the **townsendmerino** account.
