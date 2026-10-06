# v0.21.0 — the Metal device gate (RELEASING.md §C1-M), 2026-10-05

**PASS — metal on Darwin @ `85c50766`.** All 9 declared check groups reported; 10 pass, 2 skip, 0 fail.

- **Machine:** MacBook Pro M1 Pro 16 GB, macOS (darwin/arm64), go1.27.0, Apple M1 Pro GPU; checkpoints from `~/models`.
- **How it ran:** the Mac night queue, job `release-v0.21.0-metal-gate`, 18:35:01–18:40:13 PDT, by
  `docs/measurements/release-v0.21.0/run-metal-gate.sh` (`REV=85c50766`): a worktree pinned at that commit with no
  vendor/ directory, 22 gitignored fixtures linked from the main checkout, its own go.work, `git status` clean.
- **What it does NOT cover (its own skip list):** the clean-GPU check (no `nvidia-smi` on a Mac), and 21 Linux-only CI
  hygiene steps. CUDA is the other box's run: no machine has both GPUs.
- **Scope:** the verdict names `85c50766`. Commits after it up to the tag are docs only unless this record says
  otherwise; a code change before the tag means re-running this gate.
- The full log is in the night run's log directory on the Mac (`~/goinfer-logs/release-v0.21.0/`, outside the repo).

The gate's output, verbatim except two things: ANSI colours stripped, and each test-log prefix `file.go:NN:` written
`file.go line NN:`, so the citation lint does not read a log line as a citation of a path it cannot resolve.

```
rev:      85c50766c77ab24abbab9e8fdb649efdf53b19f1
started:  2026-10-05 18:35:01 PDT
fixtures: 22 gitignored entries linked from /Users/francistownsend-merino/tmcode/goinfer
tree:     0 entries in git status (0 = clean)
go:       go version go1.27.0 darwin/arm64, GOWORK=/Users/francistownsend-merino/goinfer-bench/release-v0.21.0/wt-metal/go.work

== provenance ==
  repo        85c50766
  date (UTC)  2026-10-06T01:35:05Z
  host        Darwin arm64
  backend     metal
  models      /Users/francistownsend-merino/models
  webgpu      adapter present (backend=metal)
  gpu         Apple M1 Pro

== 0. clean GPU ==
  SKIP  clean-GPU check (no nvidia-smi; on Metal check Activity Monitor by hand)

== 1. seam (runs anywhere — no GPU, no model download) ==
  PASS  serve↔decoder↔backend seam: residency is actually reached, backend names validate

== 2. Metal suite ==
   ... metal-suite: 1m0s elapsed, 111 tests finished, last: TestPrefillGemmW4 | in flight: TestGemmW8Tile_matchesReference (13s)
   ... metal-suite: 2m0s elapsed, 196 tests finished, last: TestLayerA_vectorAddDispatch | in flight: TestMMAPrefill (7s)
   ... metal-suite: 3m0s elapsed, 196 tests finished, last: TestLayerA_vectorAddDispatch | in flight: TestMMAPrefill (1m7s)
  PASS  full metal suite
      ok  	github.com/townsendmerino/goinfer/metal	192.804s

== 2c. resident PARITY gates (-tags goinfer_testhooks — the forward is asserted here) ==
  PASS  resident parity gates (dense, gemma3, gemma4 MoE, qwen3.5, mellum, gpt-oss, paging bit-exactness)

== 3. cgo-free ==
  PASS  metal/cmd/serve builds CGO_ENABLED=0 and links no Metal framework (dlopen'd via purego-objc)

== 4. lifecycle ==
  PASS  Close() frees — sawtooth not staircase, and frees with a second model resident
          close_leak_test.go line 95: trajectory: base 884 MB → peak 1027 MB → end 884 MB (growth +0 MB over 4 cycles); CurrentAllocatedSize 655360 → 1703936 bytes, one resident 558104576
          close_leak_test.go line 299: A+B alive → closed A: A ledger now 0 buf/0 obj (want 0/0), B ledger 542 buf/78 obj (unchanged from 542/78); device CurrentAllocatedSize 1116733440 → 559349760 bytes; B bit-identical. RSS 2054 MB (informational)

== 4b. prefill (f16-MMA TTFT — a shipped path, and it shipped NaN) ==
  PASS  prefill matches sequential decode and emits finite logits (no NaN)
          prefill_gemma_test.go line 91: gemma prefill last-token argmax matches sequential ✓ (cosine 0.9998)
          prefill_moe_parity_test.go line 31: moe prefill last-token argmax matches sequential ✓ (cosine 0.9999)
          prefill_moe_parity_test.go line 47: moe prefill last-token argmax matches sequential ✓ (cosine 0.9998)
          prefill_moe_parity_test.go line 47: moe prefill last-token argmax matches sequential ✓ (cosine 0.9994)
          prefill_parity_test.go line 75: prefill last-token argmax matches sequential ✓ (cosine 0.9980)
          prefill_parity_test.go line 96: prefill 140 tokens: sequential 867.30 ms vs PrefillLast 72.65 ms => 11.9x faster TTFT

== W. WebGPU suite + resident parity (adapter-detected, -tags 'gpu goinfer_testhooks') ==
  PASS  webgpu suite (backend=metal)
      ok  	github.com/townsendmerino/goinfer/gpu	24.528s
  PASS  webgpu resident parity gates (qwen3.5 DeltaNet dense; MoE/granite/nemotron opt-in fixtures)
      ok  	github.com/townsendmerino/goinfer/gpu	17.212s

== 5. repo hygiene (derived from .github/workflows/ci.yml) ==
  PASS    GITIGNORED DESTINATIONS: none. A committed document citing a path under a gitignored directory is red without an existence test — the gitignore status decides, because an uncommitted target has no history to audit.
  SKIP  21 linux-only CI hygiene step(s) — wrong platform for this host
  PASS  9 CI hygiene check(s) reproduced locally, derived from ci.yml

== verdict ==
  check groups: 9 declared -> 9 reported   |   verdicts within them: 10 pass, 2 skip, 0 fail

  Skipped — a skip is not a pass; this gate does NOT cover:
    - SKIPPED: clean-GPU check (no nvidia-smi; on Metal check Activity Monitor by hand)
    - SKIPPED: 21 linux-only CI hygiene step(s) — wrong platform for this host

  PASS — metal on Darwin @ 85c50766 (2026-10-06T01:40:13Z)
  Paste this block for the tag. The OTHER box must pass its own run: no machine has both GPUs.
finished: 2026-10-05 18:40:13 PDT
```
