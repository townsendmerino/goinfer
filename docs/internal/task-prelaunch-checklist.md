# Task (goinfer): pre-launch checklist — close the release train

> **For:** Claude Code, in `~/tmcode/goinfer` (with `~/tmcode/aikit` in the
> workspace). Mechanical close-out, in order. The engineering is done; this is
> finishing work. **This file is itself internal — delete it in Step 5.**
> Parity (`TestDecodeParity`) stays green after every step.

## Step 0 — preconditions (aikit first; the train is dependency-ordered)

The held goinfer perf wiring uses aikit v0.5.0 APIs (`Workspace`,
`MatmulBTW8A8Into`, `MatmulBTW8A8Batch`, `SetParallelThreshold`). It only builds
via `go.work` today.

- [ ] Confirm **aikit v0.5.0 is tagged AND pushed** to the proxy:
      `git -C ../aikit ls-remote --tags origin | grep v0.5.0`. If absent, that's
      blocking — the aikit side must push first (lower its default
      `parallelThreshold` per `perf-campaign.md` if that was the agreed change,
      or leave it — goinfer sets its own threshold in `main.go`, so a goinfer
      bump does not require the aikit default to change).

## Step 1 — land the held perf commit, drop go.work

- [ ] `go get github.com/townsendmerino/aikit@v0.5.0` (bump goinfer `go.mod`).
- [ ] Confirm `decoder/tune.go` (the untracked `SetDecodeParallelThreshold` /
      `DefaultDecodeParallelThreshold` API `main.go` already calls) is included.
- [ ] Commit the modified decoder files + `main.go` + `tune.go` together.
- [ ] Remove the cross-repo `go.work` reliance: `go build ./...` and
      `go test ./...` must pass **without** `go.work` (temporarily `mv go.work
      go.work.off` and verify a clean resolve against the published aikit v0.5.0,
      then restore for local dev — `go.work` stays gitignored).
- [ ] `gofmt`/`go vet`/`go test ./...` green; `BenchmarkDecode` reproduces ~68
      tok/s on the 0.5B.

## Step 2 — reconcile EVERY tok/s number to one measurement

The campaign measured **~68 tok/s (0.5B)**; the **1.5B was never re-measured
post-perf — re-run its sweep now** and use the real number. Then update every
occurrence (currently all say ~44 / ~20):

- [ ] `demo/chat/README.md` — banner cold-start line + both tier rows + the
      tradeoff paragraph.
- [ ] `demo/chat/RELEASE_TEMPLATE.md` — the two "Download — two sizes" lines.
- [ ] `docs/ARCHITECTURE.md` — the two-tier table (line ~99–100) and any prose.
- [ ] `CHANGELOG.md` — the perf entry (already ~68 in places; make consistent).
- [ ] `docs/demo-plan.md` — if it survives Step 5, else N/A.
- [ ] Grep to confirm no stragglers: `grep -rn '44 tok\|~44\|20 tok\|~20 tok' .`
      returns nothing stale.
- [ ] Also reconcile the banner sample in the README (it shows `0.48s` load —
      keep, that's still right) so cold-start + tok/s are internally consistent.

## Step 3 — gpu submodule version skew

- [ ] `gpu/go.mod` pins aikit v0.4.0 while main is now v0.5.0. Bump it:
      `cd gpu && go get github.com/townsendmerino/aikit@v0.5.0 && go mod tidy`.
- [ ] `go build -tags gpu ./...` (cgo/WebGPU present on this Mac) compiles + its
      tests pass. If gpu needs a goinfer require bump too, do it.

## Step 4 — verify cmd/prequant

- [ ] CHANGELOG references `cmd/prequant` but a survey couldn't find `/cmd`.
      Either it exists (confirm `ls cmd/prequant`) or the build path lives
      elsewhere. If the tool is real, ensure it's committed; if the build uses a
      different entry point, fix the CHANGELOG + `build-embed.sh` references so
      the documented `.giw` build story is accurate.

## Step 5 — docs/ cleanup (public repo hygiene)

7 of 10 `docs/` files are internal execution scratchpads. Before a public tag:

- [ ] **Keep public:** `ARCHITECTURE.md`.
- [ ] **Reframe + keep (optional, good marketing):** `perf-campaign.md` → a
      "how we made decode ~40% faster" writeup, OR move it to internal.
- [ ] **Move to `docs/internal/` (add `docs/internal/` to `.gitignore`) or
      delete:** `migration-plan.md`, `demo-plan.md`, `task-inmemory-load.md`,
      `task-prequant-embed.md`, `task-two-model-tiers.md`,
      `task-1.5b-demo-prompts.md`, `task-perf-phase0-profile.md`,
      `task-perf-aikit-linalg.md`, **and this file (`task-prelaunch-checklist.md`)**.
- [ ] Confirm the public repo's `docs/` now reads as a curated set, not a sprint
      log.

## Step 6 — minimal CI (`.github/workflows/` is empty)

- [ ] Add `test.yml`: `go test ./...` (model-asset tests skip cleanly), `gofmt`,
      `go vet`. Add a job that builds `gpu` with `-tags gpu`.
- [ ] Add a core-cleanliness guard (mirror aikit's): the default module graph
      must not pull `cogentcore/webgpu` —
      `go list -deps ./... | grep -i webgpu` is empty (it lives only in `gpu`).

## Step 7 — cut the release

- [ ] Bump CHANGELOG `[Unreleased]` → `[v0.1.3]` (or next) with date.
- [ ] Build both tiers, all five platforms, via `build-embed.sh --name …`
      (prequant default).
- [ ] `shasum -a 256 dist/goinfer-chat-* > dist/checksums.txt`.
- [ ] `gh release create v0.1.3 demo/chat/dist/* --notes-file demo/chat/RELEASE_TEMPLATE.md`
      (after RELEASE_TEMPLATE's numbers are reconciled in Step 2).
- [ ] Set the GitHub repo **Topics** (golang, llm-inference, local-llm, pure-go,
      gguf, …) and the About one-liner.

## Definition of done (the launch gate)

- [ ] Clean checkout (no `go.work`) builds + tests green against published aikit
      v0.5.0; `gpu` builds under `-tags gpu`.
- [ ] **One** 0.5B tok/s number and **one** (freshly measured) 1.5B number across
      README, RELEASE_TEMPLATE, ARCHITECTURE, CHANGELOG.
- [ ] `docs/` curated (only ARCHITECTURE + optionally a reframed perf writeup are
      public); internal task docs gone from the tree.
- [ ] CI present and green; core graph webgpu-free.
- [ ] Release tagged with both tiers' assets + checksums; repo Topics + About set.
