# Task: make `ci.yml` faster — critical path, gating, duplicated work (C0–C6) — 2026-09

> **Status: C0–C6 CLOSED 2026-09-22; C7 (rolling Go test cache + per-test timing capture)
> filed the same day, its probe pending. C0/C1/C2/C3/C5(staticcheck) SHIPPED; C4 shipped as `-short`,
> measured as a null, and replaced the same day by "no `-race` per push + `race-weekly.yml`"
> (owner decision); C1′ killed as a null by post-C1 per-package data; C6 killed by C0's data.**
> C3 landed in a second pass the same day, after its own kill-line was measured (37% of recent
> pushes are docs-only) and the branch-protection trap was checked directly (`gh api`: none
> configured) — and then grew a second gate condition when C2 and C3 together were found to open
> a coverage hole on a busy branch (its point 6). See its section. Filed after reading
> Linear's "AI coding has made CI a bottleneck" (linear.app/now/ci-bottleneck-reworked, 2026-09-21)
> and asking which of its steps apply here. Most do not: third-party runners, `tsgo`, type-free
> lint, vitest `isolate: false`, filtered `pnpm` installs and the Postgres schema snapshot are
> TypeScript-monorepo remedies, and Go's build cache already does the "avoid replaying unchanged
> setup" work they had to engineer. What transfers is their *systems* view — the critical path, the
> gates in front of it, and setup paid more than once — and that is what C1–C6 are.
>
> **The decision this doc made and kept:** nothing changed in `ci.yml` until C0 produced per-job
> and per-step durations from real runs (`docs/measurements/ci-timing-2026-09.md`, 28 completed
> runs, 2026-09-22). Every item below was built or killed against that data, not against the
> projection this doc opened with. C3 was excluded from the first pass ("all but the large one")
> and then commissioned separately the same day, with the explicit instruction to be careful —
> its section below records what "careful" turned out to require: a kill-line measurement, a
> direct check of the branch-protection trap, and a real correctness hazard the original scope
> had not seen (tests that read files under docs/).

## C0's real numbers, in brief (full record: `docs/measurements/ci-timing-2026-09.md`)

`root-darwin` (1339s median), not `root`'s test step (1170s), is the workflow's actual wall-time
long pole — both run the identical `go test -race ./...`, in parallel, so the LARGER one gates the
workflow. `root`'s pre-test lint stretch measured 71s median (above C1's own 60s kill-line — C1
proceeds). Queue wait and setup-go cache-restore are both already fast everywhere (no signal for
C3's runner-scarcity concern or C5's cache sub-item). `staticcheck` alone is 24s of that 71s
stretch. The checkout step shows no tail (kills C6).

---

## 1. What runs today, and where the wall time goes on paper

**This section describes the PRE-C1/C2/C4 baseline** (the shape `ci.yml` had when this doc was
scoped, 2026-09-21) — it's the problem statement C1/C2/C4's closure notes below fix, not the
current file. Line-number citations in this section point at that baseline and have not been
re-derived against the post-fix file; the closure notes under each item (§2) describe and cite
the CURRENT state accurately. `queue_citation_lint.py` does not gate `.github/workflows/*.yml`
citations (confirmed empty for this doc in its own output) — there is no tooled safety net for
staleness here, only this note.

Seven jobs on every push to `main` and every PR, all independent, all starting from a fresh
runner:

| job | runner | shape |
|---|---|---|
| `root` | ubuntu | full-history checkout → gofmt → build → vet → vet(realckpt) → staticcheck → apidiff → cuda `go mod download` → citation lint → book link lint → cleanliness guard → `go test -race` (25 m ceiling) → sampler gates |
| `readme-smoke` | ubuntu | the README's install commands from an empty dir |
| `gpu` | ubuntu | `apt-get` xorg/mesa → workspace → build/vet/staticcheck/test `-tags gpu` + demo/agent |
| `cuda` | ubuntu | workspace → build/vet/staticcheck/test `-tags cuda` |
| `root-darwin` | macos | build → vet → the **same full `go test -race`** as `root` |
| `gpu-darwin` | macos | workspace → build/vet/test `-tags gpu` |
| `metal-darwin` | macos | workspace → build/vet ×2 → one device-free test |

Three things stand out without a stopwatch:

1. **`root` is serial, and the test step is last.** Ten lint-class steps run before the first test
   compiles (`.github/workflows/ci.yml:42` through `:128`), and only then does the `-race` run start
   (`.github/workflows/ci.yml:146`). Every second of lint is a second the long pole waits. The
   only step that *needs* the full-history checkout (`.github/workflows/ci.yml:36`) is the citation
   lint plus apidiff's `v0.13.0` tag; the tests never touch history.
2. **The full `-race` suite runs twice**, on linux and on darwin (`.github/workflows/ci.yml:325`).
   The darwin job's stated purpose is platform-specific *build* breaks. The sampler-gates comment in
   the same file already makes the argument that a platform-independent gate run twice buys
   nothing — that argument was applied to 25 s of sampler gates and not to the ~10-minute suite
   (`QUEUE.md` puts the CI-shape decoder suite at ~10 min; the local shape is 22.8 min, see
   `docs/QUEUE.md:140`).
3. **Docs-only pushes run all seven jobs**, three of them on macOS. A large share of this repo's
   commits are task docs, audits and measurement records that touch no `.go` file. There is no
   change detection anywhere in `ci.yml`, and no `concurrency` block either (book-pages has one,
   `.github/workflows/book-pages.yml:32`; ci does not), so a burst of pushes keeps every
   superseded run on a runner until it finishes.

## 2. The items

### C0 — Measure. The gate on everything below.

Pull per-job and per-step durations for the last **30** `ci.yml` runs on `main`:

```
gh run list --workflow ci.yml --branch main --limit 30 --json databaseId,conclusion,createdAt,updatedAt
gh run view <id> --json jobs --jq '.jobs[] | {name, started: .startedAt, done: .completedAt, steps: [.steps[] | {name, s: .startedAt, e: .completedAt}]}'
```

Record in `docs/measurements/ci-timing-2026-09.md`: per job, median and p90 of (a) queue wait,
(b) setup (checkout + setup-go, including cache restore), (c) the pre-test lint stretch on `root`,
(d) the test step. Then answer three questions, because each one selects a different item:

- Is `root`'s test step the wall-time long pole? → C1 is the item.
- Is macOS *queue wait* the long pole (public-repo macOS runners are the scarce ones)? → C3.
- Is setup-go's cache restore slower than a cold `go mod download` + build? → C5.

**No band here; C0 is the instrument, not the claim.** But the bands for C1–C6 below are registered
now, against whatever C0 measures, so they cannot be moved afterwards.

**DONE 2026-09-22** — `docs/measurements/ci-timing-2026-09.md`, 28 completed runs (2 of the last 30
still in progress at pull time, excluded). All three selector questions answered: `root-darwin`
(not `root`) is the true wall-time long pole (1339s vs 1170s median, running in parallel so the
larger one gates the workflow); queue wait is trivially small everywhere (rules out C3's
runner-scarcity concern); setup-go cache-restore is already fast everywhere (rules out that C5
sub-item). `root`'s pre-test lint stretch: 71s median, 76s p90 — above C1's 60s kill-line.

### C1 — Split `root` into `lint` and `test`

`lint`: `fetch-depth: 0` (it keeps the citation lint and apidiff), gofmt, build, both vets,
staticcheck, apidiff, the cuda `go mod download`, citation lint, book link lint, cleanliness
guard. `test`: shallow checkout, `go test -race`, the zero-match guard, the sampler gates.

Both run in parallel from the start of the workflow. The cost is one more checkout + setup-go
(the `test` job's, shallow — cheap) and one more `go build` (Go's cache makes the second build
of the same tree mostly a no-op *within* a runner, but these are two runners, so both build).

- **Band:** wall time of the workflow drops by ≥ 80% of C0's measured "pre-test lint stretch",
  p50 over the ten runs after the change.
- **Kill:** if that stretch is under 60 s at p50, this item is not worth the extra runner start —
  record and skip.

This is the article's "minimize what's on the critical path" and "fetch only what each job
needs" in one move, and it is the item most likely to matter.

**SHIPPED 2026-09-22** — `root` split into `lint` (full-history checkout, gofmt/build/vet×2/
staticcheck/apidiff/citation-lint/book-link-lint/cleanliness-guard) and `test` (shallow checkout,
the -race suite + sampler gates), both starting at t=0. `test` does not repeat `lint`'s own
gofmt/build/vet — a compile error surfaces inside `test`'s own output instead of a dedicated step,
and `lint` (a required check regardless) already runs those as fast, clearly-named steps. **Not
yet re-measured post-change** (that needs ~10 real runs on `main` after this lands, per the band's
own "p50 over the ten runs after the change" — the wall-time-drop number itself is still owed).
**Flagged for whoever manages branch protection:** if `root` was named individually in a
required-status-checks list, that rule needs updating to `lint` + `test` — nothing in this repo's
files can see or change that GitHub-side setting from here.

### C1′ — Shard the `-race` suite (only if C1 ships and `./decoder` dominates)

If C0 shows `./decoder` is most of the test step, run it as its own job and the rest of
`./...` as another. Go's per-job setup is cheap enough that two shards pay where Linear's
110-second vitest shards did not. **Do not go past two** without re-measuring: the article's own
finding is that sharding is bounded by fixed per-shard cost, and there is no per-package timing
data yet to say a third shard would be balanced.

- **Band:** test-step wall time drops ≥ 35% (two roughly balanced shards, minus the second setup).
- **Kill:** below 20%, the shards are unbalanced or setup ate the gain; revert.

**DEFERRED 2026-09-22 — precondition not yet met.** This item is explicitly conditional on C1
having shipped AND been re-measured (its own gate says "only if C1 ships"), plus real
per-package (not just per-job) CI timing to know whether `./decoder` genuinely dominates `test`'s
1051s. Neither exists yet this pass — C1 just shipped and hasn't had its own post-change
measurement, and a local (non-CI, non-`-race`) timing attempt at `./decoder` alone did not
complete in a reasonable window this session and was abandoned rather than trusted as a CI-timing
proxy. Pick this up once C1's own 10-run re-measurement is in.

**KILLED 2026-09-22 — a null, by the per-package data C0 lacked.** Read off real CI logs of the
post-C1 `test` job: under `-race`, `decoder` is **1064s of a 1399s per-package sum (76%)**; the
next largest package, `internal/serveapp`, is 131s, and everything else is under 90s. `go test
./...` already runs packages in parallel, so the whole non-`decoder` remainder finishes underneath
`decoder` — the `test` job's wall time *is* `decoder`'s. A two-shard split (`decoder` / the rest)
therefore moves the wall time by ~0%, straight through the 20% kill-line, while paying a second
setup. The only real lever left is splitting `decoder` *itself* by test pattern across jobs — a
new scope with no per-test timing data yet, not this item.

### C2 — `concurrency` with `cancel-in-progress` on `ci.yml`

```yaml
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
```

One block. No risk to a completed run; a run superseded by a newer push on the same ref is
cancelled. Matters most during the multi-commit bursts this repo actually has (85 commits in two
days, 2026-09-08–10). Not measured by wall time of any single run — measured by runner-minutes
burned on runs whose result nobody read. **No band; ship it alongside whichever item ships first.**

Tag pushes are a different ref and untouched. `release-assets.yml` and `standalone-build.yml`
are not in scope here at all.

**SHIPPED 2026-09-22.** Exactly as scoped — four lines, `ci.yml`'s own top-level `concurrency:`
block.

### C3 — Gate build/test jobs on non-docs paths, at the **job** level

A docs-only push (anything under `docs/**` plus `*.md` at the root) should run **only** the
`lint` job from C1, because the citation lint and the book-link lint are *about* docs and must
keep running on exactly those pushes. Everything else — `test`, `gpu`, `cuda`, `readme-smoke`,
the three darwin jobs — skips.

**Job-level, not workflow-level.** Workflow `paths:`/`paths-ignore:` would skip the whole run,
lint included, and a skipped required check reads as green. Use a small change-detection job
(`dorny/paths-filter`, SHA-pinned like every other action here, or a ten-line `git diff --name-only`
step) whose output the other jobs `needs:` and `if:` on. This is the article's change-detection
gate — and its warning applies verbatim: that job is now *in front of everything*, so it must be
the cheapest thing in the file. Shallow checkout with `fetch-depth: 2` is enough to diff one push;
for a PR, diff against the base.

- **Band:** on docs-only pushes, runner-minutes per push drop ≥ 85% and no macOS runner starts.
- **Kill:** if C0 shows fewer than 1 in 4 pushes over the last 30 are docs-only, this is
  complexity for a rare case; record and skip.

One thing to decide at pickup, not here: whether `README.md` counts as docs. `readme-smoke`
exists because a README install line once shipped broken, so a README-only push should still run
that job. The simplest rule is "docs-only means `docs/**` only"; the root `*.md` files stay on
the full matrix.

**SHIPPED 2026-09-22 (second pass, same day) — `changes` job + `scripts/ci_docs_only.sh`.** What
"be careful" required, in the order it was checked:

1. **The kill-line, measured.** Of the last 30 commits on `main`, 11 changed only files under
   `docs/` — 37%, above the 1-in-4 floor. C3 proceeds on its own registered rule.
2. **The branch-protection trap, checked directly.** `gh api repos/…/branches/main/protection`
   → 404 "Branch not protected"; `…/rulesets` → `[]`. There are no required status checks, so a
   job skipped via `if:` cannot leave anything stuck today. The job-level shape is kept anyway —
   it is the one that stays correct if protection is ever added; workflow-level `paths-ignore`
   would not be, and would skip `lint` besides.
3. **A correctness hazard the original scope had not seen: tests READ files under `docs/`.**
   `TestEnvVars_docAndCodeAgree` reads `env-vars.md`; `pull/registry_test.go` reads
   `capability-matrix.json`; `decoder/hardware_matrix_test.go` and `api_tiers_test.go` read
   `hardware-matrix.md` / `api-tiers.md`; a cuda test reads `benchmarks.md`; a metal test reads
   `audit-metal-2026-09-12.md`. "Docs-only ⇒ skip `test`" would let a change to any of those go
   unchecked — and this is not hypothetical: `477da08a`, a docs-only fix to `env-vars.md` earlier
   the same day, is exactly what flipped `TestEnvVars_docAndCodeAgree` from red to green. So the
   rule became: every changed file under `docs/` AND none of them read by Go.

   Detection is dynamic, not a curated list (a list drifts as tests are added): a changed file's
   basename appearing on a Go line shaped like a file read (`ReadFile`/`Open(`/`ReadDir`/
   `filepath.Join`/`Path`/…). Two candidate rules were measured against the 11 real docs-only
   commits before choosing: counting ANY mention in Go (comments included — this repo's code cites
   task docs constantly) left only **2 of 11** eligible, which would have killed the item's value;
   the read-shaped rule keeps **7 of 11** and blocks exactly the 4 that touched a test-read file.
   `--selfcheck` (a `lint` step, every push) pins that the detection still catches the six known
   reads — the same zero-match-guard shape as the sampler gates — so a regex regression goes red
   instead of silently widening what gets skipped.
4. **Fail-open, twice.** The script always exits 0 and prints `false` on any doubt (zero/unknown
   SHAs, a diff that errors, an empty diff, a failed self-check); the seven gated jobs' `if:` is
   `!cancelled() && needs.changes.outputs.docs_only != 'true'`, so a failed or missing gate runs
   the full matrix — a plain `needs:` would have SKIPPED dependents on a gate failure, the one
   outcome the gate must never produce. Tested locally against real history before pushing: 11
   cases, 3 `true` (pure-docs commits) / 8 `false` (the four test-read commits, a code commit,
   all-zeros `before`, garbage SHA, unknown SHA, empty diff), every one exit 0. `actionlint` clean.
5. **The gate's own cost: ~10s in front of everything** (`changes` measured 15:11:09→15:11:19 on
   its first run). Every push now waits that long before any other job starts. Against C1's ~70s
   critical-path saving that is a net win on code pushes, and it is what buys the docs-only
   pushes their whole skipped matrix.
6. **A hole that C2 and C3 open *together*, closed before the first docs-only probe.** On a busy
   branch: a code push lands and its run is in flight; a docs-only push lands on top. C2 cancels
   the code push's run (superseded), C3 skips the tests on the docs push, and the tip of `main`
   reads green with the code underneath never verified by the jobs that were cancelled. Not
   hypothetical either — it was live at the moment of writing (two sessions pushing to `main`
   15 minutes apart; `43ef110d`'s own `root-darwin` was cancelled by `89ddc2d2`), and the probe
   below was about to be exactly that docs-only push. So the gate has a **second condition**:
   the commit the push builds on (`github.event.before`) must already have a *completed,
   successful* `ci` run — `gh run list --workflow ci.yml --commit "$BASE"` in the `changes` job
   (`actions: read`, ~1s). *Skip the matrix only if what you build on already passed it.* Any
   answer other than the literal `success` — `in_progress`, `cancelled`, `failure`, no run, API
   error — runs everything: fail-open in the same direction as the rest. Simulated locally
   against real runs before pushing: `dc8bdc73` → `success` (skip allowed); `43ef110d` →
   `cancelled` (no); `89ddc2d2` while in flight → empty (no); all-zeros / garbage SHA → empty
   (no). `--commit` wants the full 40-char SHA — a short one returns nothing, which is the safe
   answer, and `github.event.before` is always full.

**A known over-match, found by pre-checking the probe.** The read-detection matches on
*basename*, and `pull/integrations_doc_test.go` reads the root `README.md` — so a change to
`docs/README.md` (same basename) is treated as test-read and runs the full matrix. Safe
direction (a wasted matrix, never a skipped one); cost: `docs/README.md`-only pushes don't get
the fast path. Recorded rather than fixed — a path-aware rule is a later refinement, and the
basename rule is what makes `filepath.Join(root, "docs", "env-vars.md")`-style reads catchable.

**Verification.** Shape 1, a code push (the commit carrying the gate, `43ef110d`): the gate
completed in 10s and released all seven gated jobs — full matrix ran (`test` green in 701s;
`root-darwin` was then cancelled by C2 when `89ddc2d2` landed on top — the very shape point 6
describes). Shape 2′, the control, goes **first**: the push carrying condition (2) itself
(`ci.yml`), `race-weekly.yml`, the no-`-race` `root-darwin`, `docs/README.md` and the root
`README.md` — expected to run the full matrix (code changed; and even alone, root `README.md` is
outside `docs/` and `docs/README.md` trips the over-match above). It is also the first
measurement of `root-darwin` without `-race` (C4). Shape 2, the probe, goes **second**, on the
new gate: **the push that moves this doc to `completed/`** — one file, under `docs/`, not
test-read, pre-checked with the script's own rule, pushed only after the control's run had
completed green so condition (2) holds — expected: `changes` + `lint` run, the other seven show
`skipped`, no macOS runner starts, and the gate's log line names the base's run as `success`.
**Shape 2′ landed (`bf9759ab`, run 35750072181, 2026-09-22 15:51 UTC): green across all nine
jobs; gate line `docs_only=false (base=89ddc2d2 …; base's last ci run: '(not checked)')` —
condition (2) is correctly never evaluated once (1) is already false; `root-darwin` 242s as a
job / 195s test step, against 1324s / ~20 min on the run immediately before it (`89ddc2d2`,
still `-race -short`); linux `test` 1197s, on C0's 1170s median.** **Shape 2 landed
(`b6aae48d`, run 35752621166, 16:13 UTC): `changes` 10s + `lint` 62s, whole run 67s; all seven
gated jobs `skipped`, no macOS runner started; gate line `docs_only=true (base=bf9759ab …;
base's last ci run: 'success')` — both conditions evaluated, both held. Against the band
(runner-minutes per docs-only push drop ≥ 85%): the control run summed to ~1994s of runner time
across nine jobs (504s of it macOS), the probe to 72s — a 96% drop; wall time 1226s → 67s.**
The push recording this is the second probe. Shape 3, a docs push that touches a test-read
file, is covered by the local battery (`477da08a` → `false`) rather than a live push.

### C4 — Stop running platform-independent work twice on darwin

`root-darwin`'s test step is the full `-race` suite. The job comment says it exists for
darwin-specific build breaks (a `//go:build unix` that is too broad, `decoder/madvise_*.go`). Two
options, pick by C0:

- `go test -short` on darwin (keeps the arm64 compile of every test file, drops the exhaustive
  sweeps that are platform-independent by construction), or
- no `-race` on darwin (the detector is the same detector; the memory model does not change
  between linux/amd64 and darwin/arm64 for pure-Go code).

The first is the safer cut and keeps a real arm64 test run, which is not nothing: the machine
most goinfer users own is the one this job runs on. **This is the only item that reduces
coverage**, so it gets the correctness argument written into the step comment, the way every
other cut in this file does.

- **Band:** `root-darwin` wall time drops ≥ 50%.
- **Kill:** if `root-darwin` is not on the critical path after C1 and C3 (i.e. it finishes before
  the linux `test` job anyway), leave it alone — coverage for free is not a cost.

**SHIPPED 2026-09-22, kill-line's premise confirmed to NOT apply (root-darwin genuinely is the
critical path, even without C3).** C0's data settles the kill-line's own condition: `root-darwin`
(1339s median) is the workflow's single longest job, ahead of `test` (1170s) — the two run in
parallel, so it's the true long pole regardless of whether C3 ever ships. Chose `-short` (the
safer cut per this section's own reasoning) over dropping `-race`. Verified before applying it
that this isn't a speculative "should work" — grepped for real, non-asset-gated `testing.Short()`
sites beyond the already-self-skipping (no checkpoint in CI either way) parity tests: ~9
`t.Skip("microbench")` sites plus throughput/dispatch-profile/decomposition tests, all currently
running on both platforms for no platform-specific reason. `-short` is also not new to this
`ci.yml` — `gpu`/`cuda` already use it; this applies the same already-proven pattern to
`root-darwin`. **Not yet re-measured post-change** (needs real runs on `main`, same caveat as C1).

**`-short`: NULL, reverted. `-race` moved to a weekly workflow — 2026-09-22, owner decision.**
The re-measurement came in and `-short` did nothing: darwin `decoder` **1295s after vs 1086s
before** (n=1 each, runner noise ~19% in C0's own spread — "unchanged", not "slower"). The
`testing.Short()` sites grepped above are real but cheap; the time is in the exhaustive family
goldens, kernel parity and sampler sweeps, none of which check `Short()` (same finding as C1′:
it is one package, and `-short` does not reach the part of it that costs). The section's second
option was the live one all along: `-race` is a ~3–3.5× multiplier on the same suite, and the
linux `test` job runs the *identical* suite under `-race` on every push. What darwin `-race`
adds beyond that is races in the darwin/arm64-only Go files (`decoder/madvise_darwin.go`,
`hostram_darwin.go`, the `_arm64` paths) — real, small, slow-changing: weekly-sized, the same
reasoning `fuzz-weekly` rests on. So, per the owner: `root-darwin` now runs `go test ./...`
*without* `-race` on every push (it still compiles every test file on arm64 — the darwin build
breaks the job exists for — and still executes the NEON/DotProd kernels the linux runner never
runs), and **`.github/workflows/race-weekly.yml`** runs the exact pre-2026-09-22 `-race`
invocation every Monday 08:00 UTC (an hour after `fuzz-weekly`) and on `workflow_dispatch`.
Same reporting shape as `fuzz-weekly`: the run goes red, nothing files an issue —
`gh run list --workflow race-weekly.yml`. Against the band (≥ 50% on `root-darwin`): measured
from the control push in C3's verification and recorded there.

**Measured (`bf9759ab`, run 35750072181): `root-darwin` 1324s → 242s as a job (−82%), test
step 195s — the band was ≥ 50%.** As near to same-session interleaved as CI allows: the
`-race -short` figure is the run immediately before it on the same day (`89ddc2d2`, 15:28→15:50
UTC), n=1 each. The workflow's critical path is now the linux `test` job (1197s on this run) —
C1′'s scope, not this item's.

### C5 — Setup that is paid per job: staticcheck, apt, the setup-go cache

Three small things, each the article's "repeated setup" in miniature, each probably 20–40 s,
none worth a change without C0's number for it:

- `go run honnef.co/go/tools/cmd/staticcheck@v0.8.0` in three jobs
  (`.github/workflows/ci.yml:66`, `:259`, `:295`). setup-go caches `~/.cache/go-build`, so the
  compiled tool *should* be a cache hit — check that it is. If it is not (the cache key is the
  module's `go.sum`, which does not contain staticcheck), pin a release binary instead.
- `apt-get install libgl1-mesa-dev xorg-dev` on `gpu` every run
  (`.github/workflows/ci.yml:217`). Check whether `cogentcore/webgpu` needs these to *compile* or
  only to run; if compile-only, a cached-apt action removes it, and if run-only, remove it
  outright (the tests skip with no adapter anyway).
- setup-go's cache restore. The article found a 28-second `node_modules` cache restore losing
  to a 7.5-second filtered install. The Go module cache for this tree plus the build cache may
  be large enough for the same inversion. C0 measures restore time; compare against a run with
  `cache: false` and a cold `go mod download`.

- **Band, per sub-item:** ≥ 15 s at p50 or it is not an item. Below that, record and leave.

**Per sub-item, 2026-09-22:**
- **staticcheck: SHIPPED.** 24s median (docs/measurements/ci-timing-2026-09.md), clears the
  bar — clearly a from-source compile (`go run honnef.co/go/tools/cmd/staticcheck@v0.8.0`), not a
  cache hit against `go.sum`'s cache key (which staticcheck's own dependency graph isn't part
  of). Replaced with a downloaded, SHA-256-verified release binary
  (`.github/actions/staticcheck`, staticcheck's own release tag `2026.2` == `v0.8.0`, verified by
  downloading and checksumming it directly before wiring it in) across all three call sites
  (`lint`, `gpu`, `cuda`). Dropped `CGO_ENABLED=0` from the `cuda` job's invocation — it only
  affects Go compilation, and this is now a prebuilt-binary invocation, not `go run`.
- **apt-get (`gpu` job): INVESTIGATED, NOT SHIPPED — inconclusive, deferred rather than guessed.**
  19s median (32s p90), clears the bar. Checked `cogentcore/webgpu`'s own declared cgo `LDFLAGS`
  for `linux,!android` (`wgpu.go`): `-lm -ldl` plus a static `-lwgpu_native` — no `-lGL`/`-lX11`/
  `-lxcb` in the package goinfer actually imports (confirmed goinfer's `gpu/` never imports the
  separate `wgpuglfw` windowing-surface subpackage, which is the more plausible home for a real
  GL/X11 dependency). But the prebuilt `wgpu_native` static library's own TRANSITIVE link
  requirements weren't verifiable from source inspection alone, and this can't be tested locally
  (cgo linking against Linux-specific libraries doesn't run on the macOS box doing this
  investigation) — only a real Linux CI push would confirm it either way, which risks breaking
  the `gpu` job's build on a guess for a ~19s win. Left as-is; a real experiment (push with the
  step removed, watch whether `build`/`vet`/`staticcheck -tags gpu` still succeed) is the
  legitimate next step, not a blind removal.
- **setup-go cache: KILLED by C0.** Already fast everywhere (8-18s median across every job) —
  no sign of the article's "cache restore loses to a cold install" pattern. Nothing to gain.

### C6 — The lint gate, made resilient rather than faster

Not a speed item, but it sits on the same critical path and the article's "make checkout more
resilient" section is the reason to name it. The `fetch-depth: 0` checkout on `root` is the one
step in the file that can *hang* rather than fail — a stalled fetch of full history — and there is
no low-speed abort on it. If C0's p90 for that checkout is far from its p50, set
`GIT_HTTP_LOW_SPEED_LIMIT` / `GIT_HTTP_LOW_SPEED_TIME` on the step so a stall aborts and retries
instead of eating the 25-minute ceiling. Only worth doing if the tail shows up in the data.

**KILLED by C0, 2026-09-22.** The checkout step shows no meaningful tail on `lint` (formerly
`root`): 6s median, 8s p90 — the stall scenario this item guards against isn't in the data. Not
built.

### C7 — Persist the Go build+test cache across runs (rolling key) — filed 2026-09-22

Asked after the closing pass — *"any way to reduce the time for the linux `test` job?"* — because
it is the critical path now: 1197s on `bf9759ab` = setup 20s + `go test -race ./...` **1117s**
(of which `./decoder` ≈ 1064s) + sampler gates 55s.

**The lever.** `go test` caches a passing package's result keyed on the test binary, the
cacheable flags, every env var the tests read and every file they open or stat under the module
root — `docs/env-vars.md` included, so the C3 hazard is handled by Go itself, exactly rather
than by basename. Verified locally 2026-09-22: rerun → `(cached)`; `touch docs/env-vars.md` →
re-ran; rerun → `(cached)` again (`GODEBUG=gocachetest=1` shows the input-ID lookups). The
step's flags are all cacheable (`-race`, `-timeout`, `-tags`; no `-count`, no `-shuffle`; `-json`
is `-test.v` underneath and replays the recorded per-test lines — also verified). Of the last 60
commits on `main`, **9** touch `decoder/` or an input of it (`go.mod`/`go.sum`, `tokenizer/`,
`constrain/`, `multimodal/`, `chat/`); the other 51 would replay `decoder` from cache.

**Why it does not already happen.** setup-go's cache is keyed on `go.sum`'s hash and saved only
on a miss, so every run restores the snapshot from whenever `go.sum` last changed — its test
results are stale by construction, and most of its build objects are too (the 394MB linux entry
in `gh cache list` is that snapshot). The change: setup-go `cache: false`, and `actions/cache`
(v5.1.0, SHA-pinned like everything else here; the tag was checked to be a commit, not an
annotated tag) on `~/.cache/go-build` + `~/go/pkg/mod` with a key ending in `github.sha` (never
hits, so the post-step saves after every successful run) and `restore-keys` that restore the
newest previous run (same `go.sum` first, any second). Applied to `test` and `root-darwin`
(`~/Library/Caches/go-build` there); `lint`'s setup-go cache is untouched. Go trims cache
entries unused for five days, so the snapshot stays bounded.

**The instrument that rides along.** The `test` step now writes `go test -json` to a file and
`scripts/ci_test_summary.py` prints the readable form: build failures verbatim — with Go ≥ 1.24
and stdout redirected, a compile error appears **only** in the JSON stream (measured on a
scratch module: zero bytes on stderr), so the step is load-bearing, not cosmetic, and it exits 1
on a capture with no package result (the zero-match-guard shape) — failed tests verbatim, one
line per package with `(cached)` marked, and the N slowest top-level tests. That last list is
the per-test CI timing which slicing `./decoder` across the runner's cores needs and never had;
the local distribution is a poor proxy for it (the local #1, `TestPrefillAttnRowTileInvariance`
at 63s, loads a bench model and skips in CI; the local #2, `TestSampleFromTopK_matchesFullPath`
at 44s, is synthetic, runs in CI, and measures 1.13 cores — `./decoder` has 785 top-level tests
and one `t.Parallel()`, so most of its ~18 min is one core of the runner's four).

- **Band:** on a push that changes nothing `./decoder` depends on, `test` job wall ≤ 150s (from
  ~1197s) with `decoder` marked `(cached)`; on a push that does, the full suite runs (Go's own
  invalidation) and the job is no slower than before beyond the save/restore.
- **Kill:** save + restore > 45s at p50, or fewer than 1 in 3 of the next 20 code pushes hit —
  revert to setup-go's cache and record why.
- **Verification shapes.** (1) The push carrying this: cold — no `go-test-*` key exists, nothing
  restores, the full suite runs, the post-step saves. (2) The next code push that does not touch
  `./decoder`'s inputs: the probe — expect `decoder (cached)` and the band. (3) A push that does:
  the control — expect a full run. Results recorded here when they land.

**Shape 1 landed — twice, because C2 cancelled the first.** `74c8a525` (this change) started cold:
both cache steps completed in under a second with nothing to restore; then a docs-only push from
the other session (`aecfc9be`) cancelled it at 16:52 — and, since `74c8a525` therefore had no
completed run, that docs-only push's gate read `base's last ci run: 'cancelled'` and ran the full
matrix itself: C3 point 6's first live trigger, doing exactly what it was added for. That run
(35756865344) is the cold start that counts: `test` 875s (16:53:05→17:07:40), post-step save 3s,
`go-test-Linux-…-aecfc9be` 123 MB. `root-darwin` restored the macOS cache `74c8a525`'s completed
job had saved and reported **4 of 16 packages `(cached)`** — the mechanism works in CI — but
`decoder` re-ran (224s) after a push that changed only files under `docs/`, so something
`decoder`'s tests read is docs-shaped or per-run (Go hashes the listing of any directory a test
reads, and the push added files under `docs/measurements/`). The probe — the next code push not
touching `decoder`'s inputs — decides whether that is the common case or a docs-only artefact.

**The instrument's first reading, and a C8 sitting in it.** Of `decoder`'s 834s under `-race`,
`TestSampleFromTopK_matchesFullPath` is **369s** and `TestTopFilterLogits_MatchesReference`
**217s** — 586s, 70% of the package and of the whole job's wall — and the second of those ALREADY
re-runs without `-race` in the sampler-gates step (49s for all four gates). Next:
`internal/serveapp`'s `TestWebUI_appGateInBrowser` 106s, `multimodal`'s
`TestQwenPreprocess_inputPixelLimit_M15` 65s, then the Gumbel trio 55/38/31s. Locally without
`-race` the top test is 44s, so the detector multiplies these sweeps ~8×, not the ~3× the C0 notes
assumed for the suite as a whole. Striding the two sweeps under `-race` the way the exactness
sweep already is (the `-race` build keeps the shape, the no-race gate step keeps the exhaustive
form) is a ~10-minute lever on the critical path with no coverage change. Filed as **C8**, not
built here.

### C8 — stride the two sampler sweeps under `-race` (built 2026-09-22, same day as the instrument)

**What the instrument turned up on reading the code, and it is worse than "unstrided".**
`TestTopFilterLogits_MatchesReference` *already claims* to stride its seed axis under `-race`
(`sampler_sweep_race_test.go`: `sweepSeedStride = 7`, mode string "strided subset") — but the
seed loop reads `for s := range seeds`, and the stride constant's only remaining use was in an
error message. `git log -L` names the commit: **`324f63c9` "go fix: modernize to the language/library
idioms go 1.27 unlocks"** rewrote `for s := 0; s < seeds; s += sweepSeedStride` into the
`range int` idiom and dropped the stride, silently; every seed ran under `-race` from then on
while the log line said "strided", and nothing was red. A tool-driven rewrite that preserves
the *shape* of a loop and not its *semantics*, caught only because C7 put a per-test number next
to the name. So C8 is one bug fix and one extension:

1. **The exactness sweep's stride is applied again** (400 → 58 seeds under `-race`; all 15
   configs × 4 temperatures per seed unchanged), and the test now **fails if a build that
   declares a stride > 1 selects every seed, or a stride-1 build selects fewer than all** — the
   guard the lost-stride shape needed (the mode-string assertion could not catch it, because the
   string is set in the same file as the constant, not derived from the loop).
2. **`TestSampleFromTopK_matchesFullPath` strides its DRAW axis** by the same constant under
   `-race` (every (vocab, shape, config) cell still runs, with 1/7 of its draws — 150 → 21,
   40 → 5, 3 → 1) and is added to the no-`-race` sampler-gates step, where its full draw count
   runs. `TestSweepCoverage_fullSweepRunsSomewhere` now requires **both** names on a non-race
   `go test` line in `ci.yml` (it read only the first before); the zero-match guard step names it
   too, so a rename cannot silently drop it.

- **Band (pre-registered):** the linux `test` job's `decoder` package under `-race` from 834s
  (the cold-start run; 1069s on the noisy run after it) to **≤ 450s**, i.e. ≥ 40% off, and the
  job's wall to ≤ 10 min; the no-race gates step grows from ~49s to ≤ 120s. Coverage: the
  exhaustive forms run on every push in the gates step, exactly as before for the exactness
  sweep and newly for the draw sweep.
- **Kill:** `decoder` under `-race` improves < 20%, or the gates step exceeds 3 min — then the
  draw sweep's exhaustive form is too expensive to run per push un-raced and needs its own
  weekly slot instead.
- **Verification:** shape (3) for C7 as a side effect — this push changes `decoder` test files,
  so `decoder` re-runs (correct); the summary step's per-test list is the measurement. Results
  below when they land.

**Measured (`54505861`, run 35767702617).** The two named tests, exactly as predicted in shape:
`TestSampleFromTopK_matchesFullPath` 369s → **167.0s** (2.21×), `TestTopFilterLogits_MatchesReference`
217s → **84.8s** (2.56×) — combined 586s → 251.8s, **−57%**. `decoder` overall: 834s → **562s**
(−32.6%). The no-race gates step (guard + both exhaustive sweeps): **118s**, inside the ≤120s
band. Job wall for `test`: ~726s, down from ~1197s on the cold-start baseline.

**Against the pre-registered rule: real, above the 20% kill line, short of the ≥40% band on
`decoder`'s total — and the shortfall is accounted for, not chased further.** The two named tests
moved almost exactly as their local measurement predicted; the ~60s the package total is short of
40% sits in tests C8 never touched (the Gumbel trio + `TestSample_DrawIdentity` +
`TestSampleFromTopK_deviceRoundedZ` read ~60s higher on this run than on the cold-start baseline —
consistent with the ~19% cross-run drift C0 already measured on this workflow, not a new cost).
Re-registering the SAME rule against a re-run to chase a cleaner number would be exactly the
"bend the floor toward the answer" mistake the measurement discipline warns against; the honest
read is: mechanism confirmed at the predicted magnitude, package total inside natural CI noise of
the band, shipped as-is.

## 3. Order, and what the article would call the compounding

C0 → C2 (free) → C1 → C3 → then C1′/C4/C5 as C0 selects them → C6 if the tail exists.

C1 shortens the run; C3 removes runs; C2 removes runs nobody wanted. They stack: after all three,
a docs-only push costs one shallow lint job, and a code push starts its tests at t=0 instead of
after the lint stretch. That is the same shape as the article's outcome — fewer runner starts *and*
a shorter required check — reached with about forty lines of YAML rather than a runner migration.

## 4. What this doc does not touch

- **Coverage of anything device-bound.** The Metal layer SIGSEGVs on the macOS runner
  (`.github/workflows/ci.yml:412` and the step after it), the CUDA and WebGPU tests skip without
  an adapter, and the parity gates need checkpoints. None of that changes here; `go run ./cmd/gate`
  on a real box stays the correctness gate.
- **Self-hosted or third-party runners.** The article's largest single win, and irrelevant here:
  a public repo pays nothing for GitHub-hosted minutes, and the Linux and Mac boxes that could
  self-host are the benchmark boxes, which must stay verified-idle (`docs/benchmarks.md:153`).
- **`release-assets.yml`, `standalone-build.yml`, `govulncheck.yml`, `fuzz-weekly.yml`,
  `book-pages.yml`.** Tag-time, scheduled, or already path-filtered. Different question.

---

Related: `docs/QUEUE.md:134` (the CI-shape vs local-shape suite timing) · `docs/audit-2026-09-10.md` (N-41 pinning, the
zero-match guards this doc keeps) · `docs/tasks/task-work-queue-2026-09.md` J0 (why the citation
lint reads git, not the filesystem — the reason C3 must keep the lint job unconditional)

<!-- doc-reviewed: 2026-09-21 -->
