# Task: make `ci.yml` faster — critical path, gating, duplicated work (C0–C6) — 2026-09

> **Status: C0/C1/C2/C4 SHIPPED, C5 partial, C1′/C6 killed by C0's own data, C3 explicitly OUT OF
> SCOPE for this pass (deliberately deferred — see below) — 2026-09-22.** Filed after reading
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
> projection this doc opened with. **C3 (the docs-only fast path) was explicitly excluded from
> this pass by the person who commissioned it** — everything else ("all but the large one") was
> in scope. C3 remains unbuilt and unmeasured; its own band/kill-line stand as originally
> registered, for whoever picks it up next.

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

**EXPLICITLY OUT OF SCOPE for the 2026-09-22 pass — excluded by the person who commissioned the
rest of this doc's items ("do all but the large one"), not killed by C0's data.** C0 did measure
its own selector question (macOS queue wait is trivially small, so runner-scarcity isn't the
concern C3 would address) but did NOT measure the docs-only-push fraction C3's own kill-line
needs ("fewer than 1 in 4 pushes over the last 30 are docs-only"). This remains the single
largest, highest-complexity item in the doc — the real trap it doesn't name explicitly: GitHub
branch-protection "required status checks" semantics around a job skipped via `if:` need
verifying against this repo's actual protection rules, not just the workflow file, before
trusting that a skipped `test`/`gpu`/`cuda`/etc. reads as passing rather than leaving a PR stuck.
Whoever picks this up next should start from that trap, not from the YAML alone.

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
