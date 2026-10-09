# Task — coverage for the modules Codecov cannot see

**Status:** DONE 2026-10-08 for the Linux modules: demo/agent uploads under its own flag; gpu and cuda measured under
the 25% bar and recorded as left out; metal/ is the separate decision this doc names; site/ out of scope (results below).
**Venue:** CI only. No local measurement, no night-queue job, no hardware.
**Scoped in:** this doc. Prompted by the Codecov dashboard reading 67.17% (27,780 of 41,356
lines, commit `fa21bcc`) and that number covering one module out of six.

---

## What is actually happening

`.github/workflows/coverage.yml:39` is the whole data source:

```
go test -timeout 40m -tags goinfer_testhooks -covermode=atomic -coverprofile=coverage.out ./...
```

Two independent reasons that can only ever produce the root module.

1. **`./...` is a package pattern and stops at module boundaries.** It does not walk into a
   directory with its own `go.mod`. There are six: the root, `cuda/`, `demo/agent/`, `gpu/`,
   `metal/`, `site/`.
2. **The only tag passed is `goinfer_testhooks`.** Everything behind `//go:build cuda`, `gpu` or
   `metal` does not compile, so it is not instrumented and does not appear in the profile as
   zero — it is simply absent.

`.github/codecov.yml` is not involved. It sets informational statuses and `comment: false`; it
governs how Codecov displays an upload and cannot add modules that were never in one. The
workflow is already accurate about its own scope — its header says "Test coverage of the root
module" and the job summary writes "(root module, per-package)". Nothing here is broken. The
number means less than its placement on the README badge implies, and that is the thing to fix.

Scale of what is missing, by test-file count: `cuda/` 263, `metal/` 306, `gpu/` 139, `site/` 9,
`demo/agent/` 4.

## The trap to handle first

`go.work` is gitignored, so a checkout has none. A bare `go test` inside a submodule therefore
resolves the root from the module proxy **at its last published tag**, not from the checkout —
the failure `CLAUDE.md` describes under "Working in the tree", which can both fail on good code
and, worse here, silently measure the published root instead of the one under test.

This repo already solves it everywhere else, with an ephemeral stitch:

- `ci.yml:585` — `go work init . ./gpu ./demo/agent`
- `ci.yml:646` — `go work init . ./cuda`
- `ci.yml:888` — `go work init . ./gpu`
- `ci.yml:930` — `go work init . ./metal`
- `govulncheck.yml:101`, `:123`, `:145` — the same pattern
- `release-assets.yml:64-71` — a `GOWORK=` file in `RUNNER_TEMP`, which is the tidiest variant

Use that. A coverage job that skips it is measuring the wrong tree.

## What compiles where, already proven by existing jobs

Do not rediscover this; the build jobs establish it.

- `gpu/` builds on `ubuntu-24.04` under `-tags 'gpu goinfer_testhooks'` (`ci.yml:588`).
- `cuda/` builds on `ubuntu-24.04` under `CGO_ENABLED=0 -tags 'cuda goinfer_testhooks'`
  (`ci.yml:649`) — no CUDA toolkit needed, the backend is cgo-free.
- `metal/` is darwin-only and gets its own macOS job (`ci.yml:930`).
- `demo/agent/` needs no tag at all and already builds inside the workspace (`ci.yml:591-595`,
  which records that it "is the fifth module and had ZERO CI coverage").

## The open question — measure it, do not assume it

**How much of `gpu/`, `cuda/` and `metal/` actually executes on a runner with no device?**

The workflow header predicts "GPU backends whose tests skip without a device count as uncovered
here". That may be right, but 263 CUDA test files is a lot of files for all of them to be
device-gated, and plenty of what lives in those modules is device-free: layout and packing math,
quantization helpers, descriptor construction, error paths. Nobody has counted.

So the first step produces a number, not a pull request. For each of `gpu` and `cuda`, on an
ordinary Linux runner, with the `go work init` stitch:

```
go test -C <mod> -tags '<tag> goinfer_testhooks' -covermode=atomic -coverprofile=cov-<mod>.out ./...
go tool cover -func=cov-<mod>.out | tail -1
```

Report, per module: the statement percentage, how many tests ran versus skipped (`-v` and count
`--- PASS` against `--- SKIP`, per **"A SKIP IS NOT A PASS"**), and the wall time. That table
decides everything below, and it is cheap — it is a CI run, not a measurement that needs a quiet
machine.

## The change, once the numbers are in

Extend `coverage.yml` rather than adding workflows. Shape:

```yaml
      - name: stitch the workspace
        run: go work init . ./gpu ./cuda ./demo/agent
      - name: coverage, root
        run: go test -timeout 40m -tags goinfer_testhooks -covermode=atomic -coverprofile=cov-root.out ./...
      - name: coverage, demo/agent
        run: go test -C demo/agent -covermode=atomic -coverprofile=../../cov-agent.out ./...
      # gpu and cuda added here only if the measurement above justifies them
      - uses: codecov/codecov-action@<pinned sha>
        with:
          files: cov-root.out,cov-agent.out
          flags: root,agent
          token: ${{ secrets.CODECOV_TOKEN }}
          fail_ci_if_error: false
```

**Flags are not optional here.** A 4% `cuda` flag sitting beside a 67% `root` flag is readable
and each number means something on its own. Blended into one figure, both become uninterpretable
and the headline moves for reasons nobody can attribute. One flag per module.

**Leave `-coverpkg` off.** `coverage.yml:10` explains why: per-package credit means a package's
number reflects its own tests. Cross-package attribution would inflate every number and hide
exactly the gaps this is meant to expose.

**Watch the job budget.** `timeout-minutes: 60` with the root run already at `-timeout 40m`.
Adding `cuda` and `gpu` may not fit. If the measured wall times do not leave headroom, split the
submodules into a second job uploading under its own flags — Codecov merges uploads for a commit
from any number of jobs — rather than raising the timeout and hoping.

**`metal/` needs a macOS runner.** Treat it as a separate decision after the Linux modules land;
it adds a runner and a cost, for a module whose tests are the most likely of the three to be
device-gated.

## Decision rules, written before the numbers

- `demo/agent` is added regardless of what the measurement shows. It is CPU-only, four test
  files, no tags, and `ci.yml:591` already names its absence as a known gap.
- A submodule whose device-less run clears **25% statements** is added under its own flag.
- Below 25%, it is left out and `coverage.yml`'s header gains one line naming the module, the
  measured figure and the date — so the exclusion is a recorded result rather than an oversight
  someone rediscovers next year.
- If the combined wall time exceeds ~50 minutes, split jobs before dropping any module.
- `site/` is out of scope for this task: decide separately whether a website generator belongs
  in an engine's coverage number at all. Note `CLAUDE.md`'s "Working in the tree" says **five Go
  modules** and lists the root, `gpu/`, `cuda/`, `metal/`, `demo/agent/` — `site/go.mod` makes
  six. Correct that line in the same change, whichever way the `site/` question goes.

## What this does not change

The headline percentage will likely fall, because modules now counted as absent would be counted
as partly covered. That is the point: a lower number describing six modules is worth more than a
higher one describing one. Nothing here gates a build — `.github/codecov.yml` keeps both statuses
informational, and `fail_ci_if_error: false` stays, so Codecov can never turn a run red.

Line coverage also remains a map of what CI executes, not of what is verified. The
`a3_divergence_test.go` case in `CLAUDE.md` — a doc comment promising MoE coverage over a body
that asserted nothing about MoE — would score as fully covered. Treat this number as a way to
find code CI never reaches, and nothing more.

---

## Results, 2026-10-08

**Step 1, the measurement** (the `coverage-probe` workflow, a branch-only run deleted after the reading; ubuntu-24.04, no
device, each module's own tags, the workspace stitched with `go work init . ./gpu ./cuda ./demo/agent`; run 37881448770):

| module | statements | top-level tests PASS / SKIP / FAIL | incl. subtests PASS / SKIP | wall |
|---|---|---|---|---|
| `gpu/` (`gpu goinfer_testhooks`) | **3.2%** | 34 / 168 / 0 | 34 / 186 | 29 s |
| `cuda/` (`cuda goinfer_testhooks`, `CGO_ENABLED=0`) | **12.2%** | 87 / 329 / 0 | 155 / 379 | 23 s |
| `demo/agent/` (no tags) | **16.7%** | 13 / 0 / 0 | 13 / 0 | 16 s |

- **The workflow header's prediction holds for the GPU modules.** Most of their tests skip without a device: 168 of 202 top-level
  on `gpu/`, 329 of 416 on `cuda/`. Each module is one package (`cuda/s4siglip` has no statements), so there is no
  device-free sub-package to credit separately.
- **demo/agent's 16.7%** is one tested package (`demo/agent/agent`, 27.2%) blended with four `cmd/` and `internal/` packages
  that have no tests.

**Step 2, applied by the pre-registered rules:**
- `demo/agent` is added regardless: `coverage.yml` runs it through a go.work in `RUNNER_TEMP` (the release-assets.yml variant,
  which leaves the root run in module mode: in workspace mode the root's `./...` would also match nested workspace modules)
  and uploads it under flag `agent`; the root uploads under flag `root`. Two uploads, because codecov-action applies its
  flags to every file in an upload.
- `gpu/` (3.2%) and `cuda/` (12.2%) are under 25%: left out, and `coverage.yml`'s header names each with its figure and
  the date.
- The wall time question does not arise: the three modules together are about 70 s against the root's ~8-10 min.
- `CLAUDE.md` now says six modules and names `site/`. Whether `site/` belongs in the number stays out of scope.

**Not done here:** `metal/` (a macOS runner, its own decision).
