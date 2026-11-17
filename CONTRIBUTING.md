# Contributing to goinfer

This page is for anyone changing the tree, human or working through an AI tool. It adds no rule of its own: the rules live in
[`CLAUDE.md`](CLAUDE.md) (the project's instruction file, which binds every tool) and [`AGENTS.md`](AGENTS.md) (the short list of
what is skipped most often). Read `CLAUDE.md` before a non-trivial change. [`docs/layout.md`](docs/layout.md) maps every
directory and module.

## Build

The repository is seven Go modules: the root, `backend/webgpu/`, `backend/cuda/`, `backend/metal/`, `demo/agent/`,
`apps/sdktest/` (tests only: the official OpenAI and Anthropic SDKs against the in-process server; never tagged) and
`apps/site/` (the website generator).

```sh
bash scripts/make_work.sh   # writes go.work from go.work.example
go build ./...
```

`go.work` is gitignored and **required** for any change that crosses modules. A build of a submodule with `GOWORK=off`
resolves the root from the module proxy at its last published tag, so "method not found" on code that is right is expected
between releases, not a bug.

**Build tags gate real code:** `realckpt` (real checkpoints, heavy), `goinfer_testhooks` (cross-module test seams), and
`cuda`, `gpu` and `metal` for the backends. A change that compiles untagged can still be broken under a tag, so vet the
variants your change touches, for example `go vet -tags realckpt ./...` and `go vet -tags 'cuda goinfer_testhooks' ./backend/cuda/`.

## Test

Run the smallest check that can answer the question; the day loop is seconds, not minutes.

- `python3 scripts/touched.py` builds, vets and tests what your diff touches, runs every source-scan test that reads a changed
  file, and ends with a `VERDICT:` line. Do not run whole-package `go test` of `decoder/` or a GPU suite as a routine check.
- `gofmt -l .` in every module you touched must print nothing.
- `go run ./cmd/lint` runs the same pinned staticcheck as CI, proves it analysed something with a canary, and ends with a
  `VERDICT:` line. An empty staticcheck result by itself proves nothing, since it looks the same when the tool never ran.
- Before a push, run the citation lint (`python3 scripts/queue_citation_lint.py`, ends in a `VERDICT:` line; the pre-push hook
  runs it too). If the lint reports a missing module cache, download the modules it names; do not pass `--update`.
- Register a new check at the lowest tier that can answer it: tier 0 (seconds, deterministic counters, bit-identity on tiny
  fixtures) by default, tier 1 for one small real model, tier 2 only for a published number or a small-effect decision.
  A bug fix gets a test that fails first.
- **A skip is not a pass.** `go test` prints `ok` when every test skipped. Run with `-v` and read for `--- PASS`.
- Heavy tests need `GOINFER_HEAVY_TESTS=1` and their assets (`testdata/assets.json`); `go run ./cmd/gate` is the runner.
- A new production environment variable is not the way to add a setting: `testdata/env_reads.txt` lists every `GOINFER_*`
  variable production code reads and a test fails on one not listed. A new setting is a `decoder.Options` field.

**CI runs no CUDA or Metal device test.** A green CI run, or goldens run on another architecture, does not verify a change to a
GPU backend. Say in the pull request which device tests ran, on what hardware, or say that none did.

## Numbers

A number enters a doc only with its provenance: machine, checkpoint and quantization, greedy or seeded, pinned versions, date,
thermal state and the local-disk path of the checkpoint. Read the methodology in [`docs/benchmarks.md`](docs/benchmarks.md)
first. Comparisons with another engine go through `scripts/bench_peer.py`, interleaved in one session; a ratio across two
sessions is not a ratio. A guard threshold (swap, fit, memory), a default and a parity gate are the maintainer's to change.

## Comments

A Go comment says what is true now: what the code does, its contract (ownership, units, concurrency, what a caller must
check), the reason in a sentence or two where the shape is not obvious, and every guardrail with a pointer to its evidence.
It does not carry history (how a bug was found, dates, commit ids, measured figures, open work); that belongs in the commit
message or a doc. Say each thing once, at the declaration it binds, and cite code by declaration name, never `file.go:NNN`.

## Pull requests

- A product change goes through a pull request; a docs-only change (`scripts/ci_docs_only.sh` says `true`) or a release may go
  straight to `main`.
- Branch from `origin/main`. Keep a pull request small. To pick up a newer `main`, merge it into your branch; do not rebase and
  do not force-push.
- Never skip a hook (`--no-verify`). If a hook holds a push you believe is wrong, stop and say so.
- Stage explicit paths. Never `git add -A` or `git add .`: it sweeps generated fixture metadata into git.
- Changes you did not make in a file you are editing are someone else's in-flight work: commit your hunks and leave theirs.
- A golden is never regenerated to turn a test green. A ratchet or pin that may only shrink is not raised without the
  maintainer's agreement.
- A change that lands structure or closes tracked work updates the ledger row of its task doc when it lands
  (`CLAUDE.md`, "Task docs: the ledger").
- Releases follow [`RELEASING.md`](RELEASING.md).

## Using an AI tool

The same rules apply, and the tool must load `CLAUDE.md` (Claude Code reads it; other tools are pointed at it by
`AGENTS.md`). Commits made with a tool's help may carry its `Co-Authored-By:` trailer. You remain responsible for what is
pushed: read the diff, run the checks above, and report what actually ran.
