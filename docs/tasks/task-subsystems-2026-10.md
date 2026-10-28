# Task: subsystems with contracts — validations that carry over, tests that take seconds, one memory manager (VK0–VK6, MM0–MM6, AX0–AX7, SB1–SB14, PW1–PW6) — 2026-10

> **Status 2026-10-26: SCOPED.** Every owner decision (SD1–SD8) was made on 2026-10-26, and SD4's settings are applied. **SD4:** auto-merge,
> with the checks required on `main` (owner, 2026-10-26: "yes to SD4"; PW1). **SD1:** the towers and the GPU backends
> move from aikit into goinfer, and `linalg` stays in aikit, worked on through `go.work` and tagged weekly (owner,
> 2026-10-26: "yes to this - move the towers and the gpu backends into goinfer."). That is Track AX (§7). Every gate
> below is a sketch: whoever starts a phase registers its gates first, with the rule, void conditions, stopping rule and
> cost basis (`CLAUDE.md`, "Testing tiers").
>
> Written on the owner's ask, in a Cowork conversation the same day: "long runs/gates/tests/? of more than half an
> hour, sometimes multiple hours, are killing development velocity. In my experience that is a bad smell, meaning we
> need to modularize so only modules need testing and e2e tests happen only once a day or so. what are the major
> components that we could refactor into modules that rarely change but still support the needs of the product, with a
> focus on quick testing when the code does change." Then: "also wondering if we could extract a clean
> memory-management subsystem with a clean contract", "what other subsystems should you figure out", and "yes, detailed
> doc(s) please".
>
> **Facts about the code were read at goinfer main `c4cfb4cb` and at aikit main (root v1.75.0, `gpu/v0.35.0`) on
> 2026-10-26.** The census in §1 is [`../measurements/subsystems-2026-10.md`](../measurements/subsystems-2026-10.md),
> with the scripts and outputs that produced it.
>
> **This builds on, and does not replace:** the test-efficiency campaign
> (`docs/completed/task-test-efficiency-2026-09.md`, TE0–TE12), the 2026-10-13 test-wait record and its moves
> (`docs/measurements/test-wait-2026-10.md`), the forward architecture (`docs/tasks/task-forward-architecture-2026-10.md`,
> OP and RF), the KV architecture (`docs/tasks/task-kv-architecture-2026-10.md`, KV0–KV11) and the memory campaign
> (`docs/tasks/task-memory-campaign-2026-10.md`, MEM0–MEM17). Where an item here is the same work as an item there, the
> row says so, and the other doc keeps it.

**Ledger, 2026-10-26.** One row per item, the current state; it supersedes the text below where they disagree, and is
updated in the same commit as the work. `python3 scripts/task_docs.py check` reads it.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| VK0 | done | 2026-10-26 (mac-vk01): 234 of 4,052 commits re-staled 8,895 family validations in 30 days; 8,775 (98.7%) came with no moved golden and no changed recorded metric, an upper bound on what a behavior key saves | §5.3; `docs/measurements/vk0-2026-10.md` |
| VK1 | done | 2026-10-26 (mac-vk01): `internal/kernelprint`, 65 kernels; K1, K2 and K3 PASS on arm64, arm64 `aikit_nodotprod`, amd64 AVX2, `aikit_noavx2`, `aikit_nopopcnt` (`aikit_noavx512` VOID here); all 8 aikit bumps the suite builds against (v1.70.0 to v1.75.0) left every fingerprint unchanged, 385 family re-stalings a fingerprint key would not have made; found: embed's AVX2 Q4_K/Q6_K dequant fallback is forced by no tag | §5.4; `docs/measurements/vk1-2026-10.md` |
| VK2 | done | 2026-10-26 (mac-vk01): part 1, one file per family adapter (#277, A1–A4 PASS); part 2, `core` covers the forward path (Francis 15:31, option 1; B1–B4 PASS, the census roots amended before grading): `model_forward.go` replaces `model.go` in `core`, 26 forward-path files gain a home, `TestParityManifest_coreClosed` holds it; VK0's window: 52 plumbing-only commits (2,089 re-stalings) no longer re-stale, at most 24 forward-path commits newly do | §5.5; `docs/measurements/vk2-2026-10.md` |
| VK3 | done | 2026-10-26 (mac-vk01, Francis 17:37): one record per family, `testdata/parity/<family>.json` and `shared_sets.json`; every reader through `paritykey.ManifestJSON` (the old single file still read at older commits, so `gate shadow` compares across the change); R1–R4 PASS, two families refreshed in parallel now change two files | §5.6; `docs/measurements/vk3-2026-10.md` |
| VK4 | done | 2026-10-26 (mac-vk01): `programFingerprint` per fixture config (37, `testdata/program_fingerprints.json`) and per real-checkpoint config present; F1–F3 PASS, F2 found and fixed the head tie (`tie_word_embeddings`); 33 of 43 families have a fixture program; in no key yet: it enters VK5's shadow key in week 2 (Francis 16:27) | §5.7; `docs/measurements/vk4-2026-10.md` |
| VK5 | active | week 1 graded 2026-10-27 (31 runs, all green; the aikit v1.76.0 bump re-staled 43 families by today's key and none by the fingerprint key, each explained; caveat: no kernel changed); the switch to the kernel-fingerprint key, S1–S5 all PASS, merged (#308); week 2 (option A, Francis 05:52: the program builders out, programs in): W1–W5 all PASS (#310); amended after Gemini's review of #310 (§5.8), and the week starts at the first `vk-shadow` run after that fix merges (mac-vk01, branch `vk5-review310-2026-10`), graded a week later; a possible week 3 (option C, decoder's own ops fingerprinted) noted, not started; M | §5.8; `docs/measurements/vk5-2026-10.md` |
| VK6 | blocked | VK5; the daily canary and the coverage-of-reality audit; M | §5.9 |
| MM0 | open | either box: every OS and device memory reading behind one probe package, with a fake for tests; S | §6.5 |
| MM1 | open | either box: the pure planner `mem.Solve` out of the fit code, with a decision census against today's; M | §6.5 |
| MM2 | blocked | MM1; backends report the bytes they will allocate instead of being predicted; M | §6.5 |
| MM3 | blocked | MM2; a ledger and leases in each backend's allocator; M | §6.5 |
| MM4 | blocked | KV4 and KV7; the arbiter, which is KV11's one budget for KV blocks, expert slots and the prefix cache; L | §6.5; KV11 |
| MM5 | blocked | OP4; demand derived from the program, each operation declaring its buffers; M | §6.5 |
| MM6 | blocked | MM1; `night.py add` and `goinfer fit` call `Solve` against the box's recorded profile at queue time; S | §6.5 |
| AX0 | done | aikit PR #21 merged 2026-10-26 (`e262e12`): the README freezes the moving paths at root v1.75.0 and `gpu/v0.35.0`, the newest tags, and records `gpu/webgpu` as retired; S | §7.3 |
| AX1 | done | PR #274 merged 2026-10-26 (the device package split into `cuda/internal/gpudev` and `metal/internal/gpudev`; supersedes §7.1 and §9's `internal/gpudev`); Mac gates §7.6; nobara's CUDA device run (#279) 0 regressions against aikit `gpu/v0.35.0`, and `TestCUDA_launchDirect` segfaults on both sides alike (open, aikit's at the tag; it joins goinfer's CUDA suite at AX2d); `gpu/webgpu` retired, not copied (owner, 2026-10-26); M | §7.3, §7.6; `docs/measurements/ax1-2026-10/README.md` |
| AX2 | done | AX2a #278, AX2b #282, AX2c #283, AX2d #284; every gate PASS (§7.8–§7.10), the CUDA legs on nobara 2026-10-26 (0 differ in every tower-identity pair; the CUDA suite identical by name). Found on the way: `cuda/s4siglip`'s silent registry split (fixed in AX2b), `TestCUDA_launchDirect`'s segfault (aikit's at the tag), and four CUDA tests red on main (`TestCUDAKVSlots_pricedAgainstWhatIsLeft`, `TestResidentDenseBytes_matchesCUDADevice`), none caused by AX; M | §7.7–§7.10; `docs/measurements/ax2-2026-10/README.md` |
| AX3 | active | on branch `ax3-drop-aikit-gpu-2026-10`: no goinfer `go.mod` requires `aikit/gpu`; `internal/movedpaths` refuses the moved paths; gates §7.11 PASS on the Mac except AX3-G, which waits for the next release tag; the citation lint needed no fallback; S | §7.3, §7.11 |
| AX4 | open | aikit side: deprecate or remove by tier, trim the GPU tooling, tag; S–M | §7.3 |
| AX5 | open | SD2 decided 2026-10-26: the ANN and text-encoder GPU backends stay in aikit on the frozen `gpu/v0.35.x`; `gpupins` stops bumping them; S | §7.3, §14 |
| AX6 | open | SD3 decided 2026-10-26: goinfer's main may require an aikit pseudo-version between weekly tags; a goinfer release requires a tagged aikit; S | §7.3, §14 |
| AX7 | blocked | AX4; re-count aikit bumps, tags and pin-only commits 30 days later; S | §7.4 |
| SB1 | open | either box: backend kernel tests out of packages that import `decoder` (as `metal/kernels` did), a ratchet on the count, then the op-level backend interface with OP4 and RF1; L in all, the first step S–M | §8 |
| SB2 | open | either box: admission and routing as one pure function, tested as a table, the matrices generated from it; M | §8 |
| SB3 | moved | VK2 and VK4 (the family adapters) | §8 |
| SB4 | open | formats and the model store: loader identity by per-tensor hashes; M | §8 |
| SB5 | open | engine policies over a forward interface, tested on a scripted fake model; M | §8 |
| SB6 | open | the scheduler as a policy on a virtual clock, out of the HTTP package; M | §8 |
| SB7 | open | the serving shell over an `Engine` interface, tested on a fake engine; M | §8 |
| SB8 | active | census done (#338, #341, #345); the move to `internal/codec` (Francis: option 1) is C1 #346, C2 #347 and C3; mac-ax01 | §8 SB8 |
| SB9 | blocked | AX2; the towers with per-tower goldens; S | §8 |
| SB10 | open | one typed configuration schema, resolved by a pure function, its docs generated; M | §8 |
| SB11 | open | cost counters per operation and phase, one schema for tests, gates and the wire (with X1); M | §8 |
| SB12 | open | gate infrastructure: thin runners, graders as pure functions of records; M | §8 |
| SB13 | done | 2026-10-26 (mac-vk01, Francis 19:52): `internal/layout/imports.json`, 167 arrows, 132 test arrows and five protected rules (the towers, the gpudev halves, the kernel packages, the memory probe, the root's backends); `TestLayout_arrows` fails on a new arrow, a stale one or a rule break; L1–L3 PASS | §9; `docs/measurements/sb13-2026-10.md` |
| SB14 | done | 2026-10-26 (mac-vk01, Francis 21:42): `TestPackageDocs_namedTestsExist` (77 package docs, 21 named tests, invariant lines) and `TestArchitecture_importAppendix` (`docs/ARCHITECTURE.md`'s import appendix generated from SB13's data); D1–D3 PASS; fixed on the way: the hand-kept imports column (stale since AX3) and two stale parity-manifest mentions | §4.1; `docs/measurements/sb14-2026-10.md` |
| PW1 | active | SD4 decided 2026-10-26: branch protection on `main` (19 required checks, not strict, admins not enforced) and "Allow auto-merge" turned on that day; left: sessions use `gh pr merge --auto`, and retire `scripts/main_ci_guard.py`'s wait and its `CLAUDE.md` rule; S | §12, §14 |
| PW2 | done | decided 2026-10-27: OS jobs stay required; revisit if one job becomes the long pole by several minutes (Francis 13:19; after PW2a the slowest PR job was about 14 minutes and spread across Windows, `test-rest`, `test-decoder-d` and `gpu-darwin`, so moving Windows to nightly would save 0–2 minutes); S | §12; `docs/measurements/ci-tail-2026-10-27.md` §4a |
| PW2a | done | 2026-10-27 (nobara-ci01, Francis 11:14): MM1's two census tests moved from the decoder tail to their own -race steps in shards a and d; no test dropped, none renamed; G1–G6 PASS; the first PR run 21.9 → 10.1 min to its last job (the tail 19.9 → 8.0), `test-rest` now the slowest; S | §12.2; `docs/measurements/ci-tail-2026-10-27.md` |
| PW2b | active | 2026-10-27 (nobara-ci01; a PW2 follow-up, coordinator 17:30): every `go test` summary step in `ci.yml` runs only when its test step ran, so a failed earlier step no longer adds a missing-file error; no test, job or check changes; S | §12.4 |
| PW3 | done | `docs/README.md`'s counts no longer stored (#324); `CHANGELOG.md`'s unreleased entries are one file each in `changelog.d/`, collected at release by `scripts/changelog.py` (Francis 12:14, option 1; PW3-C1..C4 PASS); `docs/citation-index.md` left as is (2 commits since CC0); the parity manifest split by VK3; S–M | §12 |
| PW4 | done | 2026-10-26 (mac-vk01, Francis 21:42): `night.py wait NAME [--timeout MIN]` follows one queued job through its re-queues and exits 0 / 1 / 2 / 124; W1–W3 PASS in a temporary `NIGHT_DIR`, three mutations shown red | §12; `docs/measurements/pw4-2026-10.md` |
| PW5 | done | retired 2026-10-27 (Francis: option A): `touched.py` passes `-count=1` everywhere again, and the test-cache scanner, its data and wiring are removed; the saving was about 9 s on a 133 s re-run and three review rounds each found a new hole | §12.3; `docs/measurements/pw5-2026-10.md` |
| PW6 | blocked | the first VK, MM and PW items; re-run §1's census 30 days after they land; S | §12 |
| SD1 | done | decided 2026-10-26: the towers and the GPU backends move into goinfer; linalg stays in aikit | §14 |
| SD4 | done | decided 2026-10-26: auto-merge, with the checks required on `main` | §14 |
| SD2, SD3, SD5–SD8 | done | decided 2026-10-26, one at a time (§14) | §14 |

## 0. Summary

**Where the time goes** (§1; the Mac's sessions, 2026-10-13 to 10-26; nobara's were not read, its timing lock was held):
- **By wall clock, waiting on CI after a push is the largest wait: 20.3 h, 16.9% of 120 active hours.** Local tests
  and gates are 10.9%, timed runs 5.8%, nobara and the night queue 4.5%. The share of tests, gates and CI fell from
  37.5% (week of 2026-10-12) to 22.5% (week of 10-19), against 38% in the 30 days before 2026-10-13. The October moves
  worked.
- **The runs of half an hour to hours are timed gates and real-checkpoint gates, not unit tests.** KV8a's M5 passes
  ran 1 h 51 min each, KV1's speed gate 1 h 02 min, served gates up to 2 h.
- **They fail on the box's state far more than on code.** Of the Mac's 131 night jobs since 2026-10-06, 49 did not end
  ok: 33 of those were the box's state (16 memory-guard refusals, 9 voids, 8 re-queues or a busy lock), 4 a failing test.
- **The day loop is minutes, not seconds.** `scripts/touched.py`'s p90 is 4.8 min, and 1,485 of 1,641 session
  `go test` calls passed `-count=1`, so the test cache never replayed a result.
- **CI is 8.6 min at the median and up to 45 min when sessions push together,** from runner queueing, not test time.

**Five structural causes** (§2):
1. **Validations are keyed on source files and a version string, not on behavior.** One 2,996-line file holds every
   family's adapter and sits in the hash set 42 of 43 families use. `decoder/model.go` is in the same set and took 86
   commits in 30 days. aikit's version is mixed into every family's hash, and goinfer changed it 37 times in 30 days.
2. **Integration tests stand where contract tests should.** 240 of cuda's 347 test files, 245 of metal's 401 and 87 of
   gpu's 172 import `decoder`; their production code uses about 100 decoder identifiers in 44 files. Every decoder edit
   re-runs them.
3. **Machine state is a hidden input.** The memory decisions read the live machine, so their tests do; Go's test
   cache cannot see that state, so the day tools disable the cache; and a busy desktop refuses a night job at 2 AM.
4. **A release train runs inside the inner loop.** aikit cut 35 root tags and 8 `gpu/*` tags in 30 days; each bump
   touches every goinfer module and re-stales every family; about 40 pin-only commits a month keep aikit's GPU modules
   on the newest root tag.
5. **Shared hot files and waiting.** At about 110 commits a day on main, a file every change touches (the citation
   index, the changelog, the parity manifest, the docs map) is a queue, and a product push waits for the previous
   push's CI.

**What this doc proposes.** Subsystems whose contracts can be checked in seconds. Each publishes a fingerprint that
the validations of what is built on it key on, so a change re-validates only what its contract says it can change.
Real checkpoints, timed runs and the peer matrix move to a daily run that nobody waits on.

| track | what | first step | size |
|---|---|---|---|
| VK (§5) | validations keyed on behavior: kernel and program fingerprints, per-family adapters and records, a daily canary | VK0–VK3 | S–M each |
| MM (§6) | the memory-management subsystem: a pure planner, a ledger with leases, one arbiter | MM0, MM1 | M |
| AX (§7) | the towers and GPU backends move into goinfer (decided) | AX0–AX2 | M |
| SB (§8) | the other subsystems: backend interface, admission, formats, engine policies, scheduler, serving shell, codec, towers, config, counters, gate infrastructure, layout | SB1's first step, SB13 | S–L |
| PW (§12) | the waiting: auto-merge, the CI matrix, hot files, cross-box completions, the test cache | PW1 (SD4), PW2 | S |

## 1. Where the time goes

The full record, instruments and re-run commands: [`../measurements/subsystems-2026-10.md`](../measurements/subsystems-2026-10.md).

### 1.1 Wall clock (the Mac, 2026-10-13 to 10-26)

`docs/measurements/test-wait-2026-10/session_wall.py` gives every gap between two transcript events to one bucket, so the
buckets add up to wall clock (the owner's time, waits over 3 h and model stalls over 30 min are left out).

| bucket | hours | share |
|---|---:|---:|
| model thinking and writing | 59.1 | 49.4% |
| **waiting on CI after a push** | **20.3** | **16.9%** |
| waiting on local tests and gates | 13.1 | 10.9% |
| tool work (read, edit, git) | 7.1 | 5.9% |
| waiting on benchmarks and measurements | 7.0 | 5.8% |
| waiting on nobara and night jobs | 5.3 | 4.5% |
| build, vet, lint | 4.9 | 4.1% |
| other polling | 2.9 | 2.4% |
| active total | 120 | |

Week of 2026-10-12: tests, gates and CI 37.5% of active time, all waiting 51.7%. Week of 2026-10-19: 22.5% and 40.9%.

**A long run costs latency more than session time.** A 1 h 51 min pass holds a decision for two hours even while the
session does other work, which is why its share of wall clock understates what it costs.

### 1.2 Which commands (2026-10-12 to 10-26; call time, which overlaps)

- `go test`: 1,641 calls, 13.9 h. **1,485 of them (11.9 h) passed `-count=1`.** `scripts/touched.py` passes it for
  every package it tests.
- `touched.py`: 334 runs; median 0.4 min, **p90 4.8 min**, p99 12.7 min; 29 runs of 5 min or more.
- Watching PR checks (`gh pr checks --watch`): 34 calls, 5.6 h, median 9.3 min.
- Polling nobara for remote and night runs: 47 calls, 15.4 h of call time, nearly all of it in the background beside
  other work (the 5.3 h of wall clock above).
- Timed served gates: 7.9 h. `staticcheck`: 2.5 h. Parity refreshes: 159 calls, 1.7 h.

### 1.3 The night queue (the Mac, 2026-10-06 to 10-26)

131 job rows (retries included), 82 ok. Of the 49 that were not:

| cause, from each job's log | rows |
|---|---:|
| the fit or swap guard refused the load | 16 |
| voided (a cell started under heavy load, or a guard declined) | 9 |
| re-queued by its own check, or stopped | 5 |
| the other box or the timing lock was busy | 3 |
| timed out | 3 |
| a test failed | 4 |
| a harness or script error | 2 |
| other | 7 |

The longest jobs were timed gates: KV8a's M5 passes (1 h 51 min each), its M2b (1 h 47 min, failed), tack's Gemma 4 E4B
and E2B Metal runs (timed out at 1 h 20 min and 55 min), KV1's speed gate (1 h 02 min), KV2's share gate (1 h 00 min).

### 1.4 CI

The last 60 `ci` runs: median 8.6 min, p90 12.3 min, max 44.7 min. The three slowest were queueing: in one, every job
started 32 minutes after the run was created; in the other two, one Windows job started 37 and 38 minutes in, after
everything else had finished. When nothing queues, `gpu-darwin` is the longest job (7–9 min). The decoder tail shard
runs 1,205 top-level tests in 249 s under `-race`; its slowest is `TestPrefillChunked_bitIdentical` at 51 s.

### 1.5 The repository's rate

3,260 non-merge commits on main in the 30 days to 2026-10-26; 113 to 164 a day from 2026-10-21 to 10-25. The most
touched files: `docs/citation-index.md` 241, `CHANGELOG.md` 229, `testdata/parity_manifest.json` 225,
`docs/queue-engineering.md` 162, `internal/serveapp/main.go` 116, `docs/README.md` 106, `metal/model.go` 99,
`decoder/model.go` 86.

### 1.6 Limits

- nobara's transcripts were not read. At 09:27 PDT its timing lock was held by a settling night job, and a read from
  the Mac would have run beside it. The both-box figures are the 2026-10-13 record's.
- Call time (§1.2) overlaps; only §1.1 adds up to wall clock.
- Classes and causes are regexes over commands and logs; the record lists them.

## 2. Why: five structural causes

### 2.1 Validations keyed on source files and a version string

- `testdata/parity_manifest.json` validates 43 families. A family's validation goes stale when
  `decoder/parity_manifest_test.go:freshDepsHash` changes: the Go tokens (`goCodeTokens`) of every file in the family's
  sets, then `aikit_version` (`rootAikitVersion`). The `core` set is 11 files; 42 families use it, 42 use `loaders`,
  41 use `quant`.
- **One family's adapter re-stales 42.** `decoder/registry.go` (2,996 lines) holds all 40 `…Architecture(cfg *Config)`
  adapters and is in `core`.
- **Plumbing re-stales 42.** `decoder/model.go` (2,577 lines, 60 functions, 86 commits in 30 days) is in `core`. Most of
  it is load, options and backend plumbing (`Load`, `modelFromOptions`, `parseQuant`, `checkHalf16Load`, `closeBackend`,
  `warnPrefillDeclined`), beside a few numeric helpers (`addResidual`, `normalize`, `normalizeInto`,
  `softcapParallel`).
- **Every aikit bump re-stales every family,** including bumps whose own subject says the output did not change
  ("output unchanged", "bit-identical" and "weights unchanged" appear in several of the 37).
- **The manifest is one shared file** (225 commits in 30 days).
- What exists makes re-staling cheaper, not rarer: token hashing (2026-10-13), the goldens-gated refresh
  (`scripts/refresh_parity_hashes.sh`, about 4 min) and `go run ./cmd/gate identity`, which carries a validation over on
  byte-identical full logits.

### 2.2 Integration tests where contract tests should be

| module | test files | of which import `decoder` | non-test files importing `decoder` | distinct `decoder.X` used |
|---|---:|---:|---:|---:|
| `cuda` | 347 | 240 | 21 | 93 |
| `metal` | 401 | 245 | 16 | 103 |
| `gpu` | 172 | 87 | 7 | 35 |
| `internal/serveapp` | 162 | 39 | 20 | 52 |

- The production seam is narrow (about 100 identifiers per backend); the test seam is the whole decoder, because the
  tests drive a backend through `decoder.Load` and a model fixture. Go re-runs a package when anything it imports
  changes, so every decoder edit re-runs these suites. The 2026-10-13 record named this "the next edge".
- Where the seam is used, it is wide: CUDA's resident has 18 methods that run the model, Metal's 18 and WebGPU's 7, and
  the decoder type-asserts a resident against 34 optional interfaces (the forward-architecture doc, §0, read
  2026-10-13).
- `internal/serveapp`'s tests call `decoder.Load` 43 times; the package takes about two minutes.

### 2.3 Machine state as a hidden input

- Fit, residency and pager decisions read the live machine (free memory, swap, free VRAM) when they decide, and the fit
  guard predicted what each backend would build (`metalWillBeResident` and `cudaWillBeResident`, which MM2a replaced with
  each backend's own report, `decoder/builddemand.go:RegisterBuildDemand`). Their tests read the machine too, so a verdict
  depends on what else is open.
- Go's test cache keys on the test binary, the cacheable flags, the environment variables a test reads and the files it
  opens inside the module (§16). It cannot see a sysctl, a device or a checkpoint under `~/models`. The day tools pass
  `-count=1`, reasonably, and pay for it on every run.
- The night queue shows the cost: 33 of 49 jobs that did not end ok (§1.3).

### 2.4 A release train inside the inner loop

- aikit cut 35 root tags (v1.48.0 to v1.75.0) and 8 `gpu/*` tags in 30 days. goinfer's root `go.mod` changed its aikit
  requirement 37 times, each time in every module. By subject (approximate): 16 for linalg kernels, 14 for the vision
  and audio towers, 5 for embed, formats or mmap, 1 for the GPU module.
- Each root tag runs aikit's ritual (its `RELEASING.md`: releasegate with golangci-lint, apidiff and a CHANGELOG line
  for every new export; perfgate on a real box; vulncheck; CI; the tag; `gpupins --fix`), then goinfer's bump and a
  parity re-arm.
- aikit's eight `gpu/*` modules took 39 to 41 commits each in 30 days, nearly all requirement bumps; their Go code
  changed 0 to 2 times each.
- One bump pinned a pre-release pseudo-version rather than wait for a tag (2026-10-24, KS-O1).
- aikit's own line puts the towers on goinfer's side: kernels in aikit, model semantics in goinfer (the premise of
  aikit's goinfer-kernel-moves task doc). SigLIP, Qwen2.5-VL, Pixtral and Gemma 4's conformer are model semantics.

### 2.5 Shared hot files and waiting

- At this rate the files every change touches are queues (§1.5), and each merge of main is a reason to re-run the
  parity refresh and the citation lint (`CLAUDE.md`, "Day loop").
- A product push to main waits for main's previous CI run (`scripts/main_ci_guard.py`, owner rule 2026-10-18). A PR
  merges only after a session watches its checks: main has no branch protection, and none of the last 30 merged PRs
  used auto-merge.
- Several sessions pushing at once queue the runners (§1.4).

## 3. What must not change

1. **Bit-identity at every migration step,** per backend, on every family fixture and real checkpoint the backend runs
   (the forward-architecture doc, §2 item 1).
2. **The CPU stays the reference;** hand-written kernels stay; this is not a graph compiler.
3. **`decoder`'s public API stays as it is.** A public type is defined in `decoder` with an internal twin, never
   aliased to an internal package: aliasing hid the sampler types' fields from `go doc` on 2026-10-13
   (`docs/measurements/test-wait-2026-10.md` §4).
4. **No new production environment reads** (`testdata/env_reads.txt` only shrinks).
5. **A change to any guard threshold (swap, fit, memory) needs Francis's word.**
6. **The measurement standards and the run budget** (`CLAUDE.md`, "Measurement discipline" and "Run budget").
7. **`RELEASING.md`'s two-step tag** for any change to a seam between modules.

## 4. Design rules for a subsystem

1. **Its contract is narrower than its code, and written down** in its package doc: inputs, outputs, what it may
   assume, what it promises.
2. **It owns a fast oracle:** a reference implementation, a fake, a fixture set or a simulator, in its own package, so
   its tests import nothing above it.
3. **It publishes a fingerprint:** a hash of its outputs on a canonical input set that reaches every dispatch path. A
   validation of something built on it keys on that fingerprint, not on its source.
4. **Packages, not modules.** A new `go.mod` only for a separate toolchain or a separate tag train. Every module is a
   release train; §2.4 is the cautionary tale.
5. **Import direction is a test** (SB13), with ratchets that only fall.
6. **Probes live at the edge.** One package reads the OS or a device; everything else takes a snapshot, as data.
7. **A contract test is checked against how the system calls it.** `CLAUDE.md`'s rule that a unit test supplying its
   own calling convention proves nothing about its callers is enforced by the daily coverage audit (VK6), not by
   reading test names.
8. **The day loop checks contracts; the daily run audits them.** A change that cannot be judged without the daily run
   is a modularity bug (`CLAUDE.md`, "Day loop").

### 4.1 How each subsystem is documented (owner, 2026-10-26)

Four homes, one kind of fact each, and each one checked so it cannot drift:

| where | holds | checked by |
|---|---|---|
| the package's `doc.go` | the contract: what it owns, inputs and outputs, the invariants it promises (each naming its test), what it may import, the one command that tests it in seconds; plus `Example` functions | `go doc`; SB14's test (each named test exists); the import gate (SB13) |
| `docs/code-notes/<package>.md` | the design: why this shape, what was considered and rejected, known limits, links to the task docs and measurements | the citation lint (code cited by declaration) |
| `docs/ARCHITECTURE.md` | the one system map: a section per subsystem, its allowed imports, links to its `doc.go` and code-notes | SB14's test: its import table equals the SB13 gate's data |
| a short `CLAUDE.md` in the module or package folder | working rules for the sessions: the day-loop command, what not to import, which fixtures to use, the gate before a push | review; it stays short, and the root `CLAUDE.md` keeps only repo-wide rules |

- **No `README.md` per package.** `go doc` does not show it, so it becomes a second copy of `doc.go`. READMEs stay at module roots
  (`cuda/`, `metal/`, `gpu/`), where GitHub shows them.
- **No ADR files.** Decisions live in the task docs, dated and in the owner's words (SD1–SD8 here); code-notes link to them.
- **History stays out of all four**, per `CLAUDE.md`'s comment rule: it goes in commit messages, task docs and measurement records.

**The template, for `internal/mem` (MM0 starts from it):**

```go
// Package mem plans, accounts for and arbitrates every byte goinfer holds: weights, KV blocks, recurrent state,
// expert slots, the prefix cache and scratch, on every tier (VRAM, unified memory, host, pinned, file-mapped).
//
// Contract. Solve(Demand, Machine, Policy) is pure and returns a Plan or a named Decline. At run time every
// allocation is a Lease from the Ledger that names its plan line; the Arbiter is the one budget for the
// evictable classes. Machine state arrives as data: mem/probe is the only package that reads the OS or a device.
//
// Invariants, each with the test that holds it:
//
//	I1 nothing is allocated without a plan line        TestLedger_noUnplannedBytes
//	I2 every lifetime closes to zero                   TestLedger_lifetimesClose
//	I3 the ledger agrees with the measured process     TestLedger_closesAgainstProbe (per backend)
//	I4 Solve is pure                                   TestSolve_pure
//	I5 Solve is monotone                               TestSolve_monotone
//	I6 one decider                                     TestNoOtherAllocationEstimate (source scan)
//	I7 thresholds carry the owner's approval           TestPolicy_approvalStamp
//	I8 budgets key on computed bytes                   TestSolve_ignoresOSAccount
//
// Imports: the standard library only (mem/probe adds golang.org/x/sys).
//
// Test it: go test ./internal/mem/... (seconds; no model, no device). The machine zoo is testdata/machines/*.json.
//
// Design: docs/code-notes/mem.md. Plan and phases: docs/tasks/task-subsystems-2026-10.md, Track MM.
package mem
```

And its folder's `CLAUDE.md`, in full:

```markdown
# internal/mem — working rules
- Day loop: `go test ./internal/mem/...` (seconds). Never load a model or touch a device here.
- Import nothing from goinfer. Read the OS or a device only in `mem/probe`.
- A guard threshold changes only in the committed Policy, with Francis's approval stamp.
- A new invariant goes in `doc.go` with its test name, or SB14's test fails.
```

## 5. Track VK — validations keyed on behavior

### 5.1 Today's key

| trigger | re-stales | how often, 30 days |
|---|---|---|
| a code edit to any of the 11 `core` files | 42 families | `decoder/model.go` alone 86 commits |
| an adapter edit in `decoder/registry.go` | 42 families | 20 commits |
| an aikit bump | 43 families | 37 |
| a loader edit (`decoder/weights.go`, `decoder/gguf.go`) | 42 families | 27 and 22 commits |

VK0 turns this into an exact before number.

### 5.2 The target

A validation of family F on backend B carries over while its key is unchanged:

    key(F, B) = H( program fingerprint of F's configs,
                   numerics fingerprints of the kernels F's operations use on B's architecture,
                   loader fingerprint of F's fixtures,
                   walker fingerprint of B )

Each fingerprint is a hash of outputs on canonical inputs and takes seconds. When the key changes, the change is
re-validated at the lowest tier that can answer it, a tiny fixture first, and the real checkpoint runs in the daily run
or before a tag. `gate identity` stays the instrument when a key changes and a real checkpoint is needed.

### 5.3 VK0 — the before number

Replay `freshDepsHash` over the last 30 days of main: for each commit, how many families went stale, which set
triggered it, and whether any golden moved. The output is the count of re-stalings a behavior key would not have made.
Tier 0; minutes. S.

### 5.4 VK1 — a numerics fingerprint for the aikit kernels goinfer calls

- For each `linalg` and `embed` kernel goinfer calls, a canonical vector suite that reaches every dispatch path: M=1
  and M>1 tiles, tails, thresholds, and each forced fallback (CI's `noavx2`, `noavx512`, `nopopcnt` and `nodotprod`
  cells already exist).
- The fingerprint is the hash of the output bits, per kernel and architecture. `freshDepsHash` mixes in the
  fingerprints for the architecture the validation ran on, in place of `aikit_version`.
- First computed in goinfer by a test (seconds); later published by aikit with each tag, so a bump can be judged
  without running anything.
- **Gates (sketch):** a planted one-ULP change in one kernel changes that kernel's fingerprint and only that one
  (mutation, both directions); a release that changes no kernel leaves every fingerprint unchanged; the suite reaches
  every path the dispatch census lists. S–M.

#### VK1's gates, registered 2026-10-26 before the code (mac-vk01)

All tier 0: no model, no device, no timed run. The instrument is the package `internal/kernelprint`: `Compute` runs each
aikit `linalg` and `embed` kernel goinfer's production code calls over a fixed, generated input suite and returns one
SHA-256 per kernel, keyed by the dispatch configuration (GOARCH, `linalg.ActiveKernels().Active`, `ForcedFallbacks()`).
"The dispatch census" is the `cpu-*` entries of `docs/hardware-coverage.json` and the predicates they name.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| K1 stable | two `Compute` calls in one process give identical fingerprints for every kernel | never | one run per configuration | one `Compute`: seconds (estimate, measured in the record) |
| K2 mutation, both directions | on a copy of aikit v1.75.0 with a planted one-ULP change in one kernel's output (+1 ULP and −1 ULP, in two kernels: `MatmulAVAcc64` and GGUF Q5_0 dequantization), exactly the planted kernel's fingerprint changes and every other kernel's stays; the same copy with no plant reproduces the module cache's fingerprints for every kernel (removing a plant restores) | a plant whose text replacement does not apply (a no-op mutation): fix the plant, not the rule | one run: the unplanted copy and four plants | five builds of `linalg`, `embed` and the test, about 1–2 min on the Mac (estimate) |
| K3 census reach | in each configuration run, every `if` in aikit's `linalg` and `embed` (the files built for that configuration) whose condition names a census predicate has its active side executed by the suite (read from a coverage profile of the suite), unless it is on a content-keyed exclusion list that says why goinfer's production code never reaches it; a predicate site the condition does not decide needs both sides | a configuration run under a forcing tag that forced nothing (`ForcedFallbacks()` empty); it is re-run on hardware where the tag forces | one run per configuration | one coverage build and run: under a minute (estimate) |

Configurations run on the Mac: arm64 as built, arm64 `aikit_nodotprod`, and amd64 under Rosetta (`ROSETTA_ADVERTISE_AVX=1`)
as built, `aikit_noavx2` and `aikit_nopopcnt`. No machine here has AVX-512 VNNI, so `aikit_noavx512` forces nothing on
it and the AVX-512 side of K3 is not run here; the record says so, and CI's mixed runner pool or nobara under SDE is
where it can run. `freshDepsHash` and the parity manifest do not change in VK1 (VK5 is the shadow mode).

### 5.5 VK2 — one file per family adapter, and only numeric code in `core`

- Move each `…Architecture` adapter out of `decoder/registry.go` into a file of its family's own. The registry map,
  `resolveArchitecture` and `Architecture.validateResolved` stay where they are. Each family's `own` list names its
  adapter file; `core` keeps the registry mechanism only.
- Move `decoder/model.go`'s numeric helpers into a numeric file in `core`; `decoder/model.go` leaves `core`.
- **Gates (sketch):** no family's state changes at the commit that narrows the sets (every validation carried by
  identity); a planted edit in one adapter stales exactly one family. S–M, mechanical.

#### VK2 part 1's gates (the adapters), registered 2026-10-26 before the code (mac-vk01)

Part 1 moves each `…Architecture` adapter, and each helper only one adapter uses, from `decoder/registry.go` into
`decoder/arch_<family>.go`, unchanged. It is done by a tool that cuts each declaration with its doc comment, then
`goimports`. Each manifest family's `own` list gains its adapter's file and the files of the adapters it calls
(voxtral→llama; qwen3_asr and qwen3_vl→qwen3; qwen2_5_vl→qwen2; qwen3_vl_moe→qwen3_moe and qwen3_vl's m-RoPE helper).
`registry.go` keeps the map, `resolveArchitecture`, `validateResolved`, the bounds check and the two helpers several
adapters share, and stays in `core`. Part 2 (`model.go` and what `core` covers) waits on a decision. Measured while
planning it: 19 files holding forward-path code are in no hash set, among them `program.go` and `program_cpu.go`.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| A1 unchanged code | every top-level declaration in `decoder` has the same Go token stream before and after the move; only the file holding it changes | never | one run | seconds |
| A2 carried by identity | `scripts/refresh_parity_hashes.sh` (goldens-gated) refreshes `deps_hash`; the manifest diff changes `own` lists and `deps_hash` and nothing else: no status, `validated_at`, date or metrics | a golden red before the move (then it is not this change's) | one run | about 4 min (the script's own figure) |
| A3 a planted adapter edit | on a throwaway commit after the move, `gate shadow` stales exactly: `{phi3}` for an edit in `arch_phi3.go`; `{llama, voxtral}` for one in `arch_llama.go`; every family that uses `core` for one in `registry.go` | — | one run each | about a minute in all |
| A4 the census holds | a new test, `TestParityManifest_adapterFilesOwned`, derives each family's adapter files from the registry and the adapters' calls and fails when a family's `own` list lacks one; `gate mutation` turns it red by dropping one entry | — | one run | seconds |

#### VK2 part 2's gates (`core` covers the forward path), registered 2026-10-26 before the code (mac-vk01)

Francis (15:31, relayed): option 1. Move `model.go`'s forward functions into a forward file in `core`, drop its load,
option and generation plumbing from `core`, give every uncovered forward-path file to `core` or to its one family's
`own` list, and add the test that fails when forward-path code sits in a file no set covers.

The rule the test holds: **`core` is closed under calls.** From every function in a `core` file, a type-checked call
graph of `decoder` (the files this GOOS and GOARCH build, calls resolved by `go/types`, interface and func-value calls not
followed) reaches only functions in files some shared set or some family's `own` list covers. Rooting at `model.go`'s
forward reaches 19 uncovered files today. Rooting at all of `core` also reaches the batched prefill, the VL prefill and
config loading: 31 in all. The 12 beyond the 19 follow the same rule. A file only one family subset reaches goes to
that subset's set or `own` lists: KDA → bailing_hybrid, Gemma 4's MoE FFN → gemma4, MLA int8 → deepseek_v2, deepseek_v3
and kimi_k2, the DeltaNet batched path → the `deltanet` set, Qwen3.5's VL prefill → qwen3_5 and qwen3_5_moe. Everything
else goes to `core`.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
**Amended 2026-10-26, before B2–B4 were graded (mac-vk01):** the rule's roots. Rooting at every function in a `core` file
does not converge: each file it adds brings its plumbing in as new roots (`blockspec.go`'s `GenerateStream`,
`lora.go`'s adapter loading), and the closure passed 65 files and was still growing, most of `decoder`. The test roots at
the forward's entry points instead: every function in `model_forward.go` (the per-token forward) and `forwardn.go` (the
batched one), both in `core`. A file the forward reaches is covered, but its other functions are not roots. That is
Francis's wording, "forward-path code". Under it, 26 files were uncovered (`config_nonfinite.go` and
`recommended_sampling.go` drop out, `lifecycle.go` comes in). The test is `TestParityManifest_coreClosed`.

| B1 unchanged code | as A1: every top-level declaration in `decoder` keeps its token stream; only `model.go`'s forward functions change file | never | one run | seconds |
| B2 the forward census | `TestParityManifest_coreClosed` passes after the change and fails on the tree before it (it names the uncovered files); `gate mutation` taking `program_cpu.go` out of `core` turns it red | — | one run each | seconds (one type-check of `decoder`) |
| B3 carried by identity | the goldens-gated refresh; the manifest diff changes the sets, `own` lists and `deps_hash` and nothing else | a golden red before the change | one run | about 4 min |
| B4 planted edits | `gate shadow` on throwaway commits stales: every `core` family for a token in `program_cpu.go` (none before) and in the new forward file; none for a token in `model.go`'s `Load`; exactly `{bailing_hybrid}` for one in `kda.go` | — | one run each | about a minute in all |

### 5.6 VK3 — one parity record per family

`testdata/parity/<family>.json`, plus one file for the shared sets; `TestParityManifest_fresh` reads the directory. Two
sessions working on different families stop colliding, and a refresh touches only the records it changes. S.

#### VK3's gates, registered 2026-10-26 before the code (mac-vk01)

Francis (17:37, relayed): VK3 while VK5's shadow week runs. The layout:
- `testdata/parity/shared_sets.json` holds the shared sets.
- `testdata/parity/<family>.json` holds one family's row, with the same fields and values as today.
- `testdata/parity_manifest.json` is removed.

Every reader goes through one function, `paritykey.ManifestJSON`. It assembles the manifest's shape from the records,
or from the old single file at a commit that predates them. That keeps `gate shadow` and the VK0 replay working across
the change, and it errors where the day-loop selector would silently find no manifest. The decoder writes only the
records whose content changed.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| R1 carried unchanged | the records, assembled, equal the old manifest: every family's fields (`uses`, `own`, `deps_hash`, `status`, `validated_at`, `date`, `reference`, `machine`, `method`, `metrics`) and the shared sets, as JSON values; `TestParityManifest_fresh` passes with no refresh | never | one run | seconds |
| R2 the readers read the records | no Go, script or workflow outside history tools names `testdata/parity_manifest.json`; the decoder's parity tests, `gate quick`'s golden-file map, `gate parity`'s qualifying paths, `gate identity`'s families and `scripts/refresh_parity_hashes.sh` all run on the records; the refresh, run on a planted stale family, rewrites that family's record and nothing else | — | one run | the refresh's goldens, about 20 s warm |
| R3 the same families re-stale | `gate shadow` across the layout change itself (before: the single file; after: the records) classes every family agree-fresh; on a throwaway commit after it, a token planted in `arch_phi3.go` stales exactly `{phi3}`, as VK2's A3 did on the single file | — | one run each | about a minute |
| R4 parallel refreshes touch different files | two families made stale on two branches and refreshed separately change disjoint files (`phi3.json`, `llama4_text.json`), and the branches merge with no conflict | — | one run | two refreshes, under a minute |

### 5.7 VK4 — a program fingerprint

`decoder/program.go:buildProgram` takes an `*Architecture` and an experts predicate, not weights, so a program can be
built from a config alone. Serialize the op kinds and their parameters and hash them: one fingerprint per family fixture
config and per real checkpoint config in `testdata/assets.json`. A refactor that leaves every program unchanged leaves
every family's key unchanged. M.

#### VK4's gates, registered 2026-10-26 before the code (mac-vk01)

Francis (15:55, relayed): VK4 next. Nothing in `program.go` or `program_cpu*.go` changes; the code is new files, so the
nobara sessions working near the program walker see no collision. `programFingerprint` hashes a program by a
reflection walk of its operations and their parameters: kinds, registers, every parameter field by name and value, and
floats by their bits. It writes no type names, so renaming a type is not a program change. `programForConfig` builds
the program from a `config.json` alone. Its experts predicate is the config's: layers `[FirstKDense, NumLayers)` of an
MoE architecture, or Llama 4's own per-layer MoE flags. Every committed fixture `config.json` that resolves to an
architecture gets a fingerprint in `testdata/program_fingerprints.json`; each real-checkpoint directory in
`testdata/assets.json` gets one in the record, computed where its `config.json` is present.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| F1 the table | `TestProgramFingerprint_table` recomputes every committed fixture's fingerprint and fails on any difference from the table (`-update` rewrites it); two computations in one process agree | never | one run | seconds |
| F2 config equals weights | for every fixture whose weights are present on the machine, the program built from its config has the same fingerprint as the program `Load` builds from its weights | — | one run (fixtures absent are listed, not counted) | a few seconds per tiny fixture |
| F3 refactor-blind, change-sensitive | (a) VK2's moves left every program unchanged: the table computed at `f8e248b7` (before part 2) equals the one at this branch; (b) `gate mutation` changing one parameter in one family's layer ops (`gemma4LayerOps`' v-norm epsilon) turns F1 red, and the diff names only Gemma 4 fixtures | — | one run each | seconds |
 — the key, in shadow mode first

On every push to main, compute both today's staleness and the new key, and log every disagreement with its reason.
Switch after one week (SD6, decided 2026-10-26) with no disagreement left unexplained. M.

#### VK5's gates, registered 2026-10-26 before the code (mac-vk01)

Order: VK5 shadow mode first, then VK2 (owner, 2026-10-26 13:06, relayed). Until VK4 exists, the shadow key is today's
key with one input replaced: `aikit_version=<pin>` becomes `kernelprint=<digest>`, the SHA-256 of the dispatch
configuration and every kernel's VK1 fingerprint. The source files are hashed as today. The instrument is
`internal/paritykey` (today's key as a library, and the shadow key), `go run ./cmd/gate shadow <before> <after>`, and a
workflow, `.github/workflows/vk-shadow.yml`, that runs it on every push to main. It is not a required check and has no
concurrency group, so collapsed pushes are not skipped. It writes the comparison to the job summary and to an artifact.
Both fingerprints of a push come from one runner and one suite: the after side builds the suite against the after
pin, the before side the same suite against the before pin.

Across a push, each family is one of: agree-stale, agree-fresh, old-only (today's key re-stales it, the shadow key does
not), new-only (the reverse). Old-only is **explained** when the aikit pin is the only input of today's key that moved
and every fingerprint is unchanged. New-only cannot happen with one suite and one toolchain unless something the key
does not see moved, so every new-only is **unexplained** until someone explains it in the shadow record.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| P1 equivalence | `paritykey`'s today key equals `freshDepsHash` for every family in the tree (a decoder test calling both) | never | one run | seconds |
| P2 classifier | a table test covers every class and reason; removing the "every fingerprint unchanged" condition turns it red (`gate mutation`) | never | one run | seconds |
| P3 history | `gate shadow` on three real pushes from VK0's window: an aikit-only bump the suite builds both sides of (all 43 families old-only, explained), a commit that only edits `decoder/model.go` (all agree-stale), and one that touches no dependency (all agree-fresh) | the suite does not build against a side's pin | one run each | one child build per pin change, about 5 s; under a minute in all |
| P4 the job | on a push to main it finishes and writes its summary; a push whose before commit is missing (a new branch, a force push) or whose before pin the suite cannot build is logged VOID, never as agreement | — | read at the next natural point, never watched (`CLAUDE.md`, "Push finished work") | under 3 min a push (estimate, checked on the first runs) |

The shadow period's stopping rule is SD6's: one week of pushes to main from the workflow's first run, with no
unexplained disagreement. The switch is its own PR and its own decision.

**Two shadow weeks (Francis, 2026-10-26 16:27, option 1).** Week 1, the one running, grades only the swap of the aikit
pin for the kernel fingerprints. When it ends, week 2 grades the key built from VK4's program fingerprints plus the
kernel fingerprints, with the forward's source files out of it. Each week then explains its disagreements by one
change. Week 2's gates are registered before its code, like week 1's.

**Week 1 graded (Francis, 22:40: end it early once a real aikit bump has gone through).**
- **The runs.** 31 `vk-shadow` runs on main, from the workflow's first at 15:01 PDT (2026-10-26) to 23:20. All green,
  none VOID. 17 ran on AVX2 runners and 14 on AVX-512 runners.
- **The one aikit bump** (#305, v1.75.0 → v1.76.0, comments, docs and tools only, on an AVX-512 runner): today's key
  re-staled all 43 families and the kernel-fingerprint key none. All 43 verdicts were old-only, each explained "aikit
  v1.75.0 -> v1.76.0 changed no kernel fingerprint".
- **No other disagreement in the week.** Two pushes (VK2's two parts) re-staled all 43 under both keys; the other 28
  left every family fresh under both.
- **Caveat.** This bump changed no kernel, so the week shows the key carrying a validation over, not catching a change.
  That the fingerprint catches a one-ULP kernel change rests on VK1's K2 mutation test.

Run by run: `docs/measurements/vk5-2026-10/week1-runs-2026-10-27.txt`.

#### The switch's gates, registered 2026-10-27 before the code (mac-vk01)

The switch puts the kernel fingerprints in today's key in place of `aikit_version`. The digest has to be the same on
every runner, so the key mixes in a committed table, not the runner's own fingerprints.
- **The table.** `testdata/kernel_fingerprints.json` holds every kernel's fingerprint per dispatch configuration, for
  the three CI runs on:
  - `arm64[dotprod]`, generated on this Mac;
  - `amd64[avx2,f16c,popcnt]`, generated under Rosetta. Its digest equals the one 17 week-1 runs on AVX2 runners
    recorded;
  - `amd64[avx2,avx512vnni,avx512vnni+vl,f16c,popcnt]`, #303's SDE table. Its digest equals the one 14 week-1 runs on
    AVX-512 runners recorded, so the emulation matched the hardware.
- **The key.** `freshDepsHash` and `paritykey`'s today key end in `kernel_fingerprints=<digest of the table>`.
- **The table's own check.** `internal/kernelprint`'s `TestCompute_table` compares the runner's own fingerprints with
  its configuration's entry, and reports a configuration with no entry as not checked. `-update` writes the entry.
- **The shadow job.** `gate shadow` gains one explained reason: the key's definition changed at this commit.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| S1 the table | `TestCompute_table` passes here on arm64 and amd64-AVX2; the AVX-512 entry's digest equals the real AVX-512 runners' | never | one run each | seconds |
| S2 a one-time re-stale | the goldens-gated refresh changes the 43 `deps_hash` lines and nothing else; P1 (`TestParityKey_matchesFreshDepsHash`) still holds | a golden red before the change | one run | about a minute |
| S3 the pin is out of the key | on a throwaway commit that changes only the aikit requirement, `TestParityManifest_fresh` stays green and `gate shadow` classes every family agree-fresh | — | one run | about a minute |
| S4 a kernel change re-stales | `gate mutation` changing one fingerprint in the table turns `TestCompute_table` red (this config's entry) and `TestParityManifest_fresh` red (every family; VK4's per-family narrowing is week 2's) | — | one run each | seconds |
| S5 the shadow job across the switch | `gate shadow` on the switch commit classes all 43 old-only, each explained by the key's change of definition, with none unexplained | — | one run | about a minute |

**The switch, graded 2026-10-27: S1–S5 all PASS** (`docs/measurements/vk5-2026-10.md` §5). The table's three digests
equal week 1's per configuration; the one-time refresh moved only the 43 `deps_hash` lines, with goldens 33/31/0; an
aikit-only commit leaves every family fresh; one changed fingerprint re-stales all 43; the shadow job explains all 43
across the switch by the change of definition. Added beyond the registration: the shadow job also explains an old-only
family when the table moved only for configurations other than the runner's own.

#### Week 2's gates, registered 2026-10-27 before the code (mac-vk01; Francis, 05:52: option A)

Week 2 shadows a key in which the program builders leave and VK4's program fingerprints take their place.
- **Builders, out:** `decoder/program.go`, `blockspec.go`, `arch.go`, `config.go`, `registry.go` and every
  `decoder/arch_*.go` adapter.
- **Kept in:** the executor, loader/quant/serialize and supporting files, and the kernel table's digest (today's tail).
  The program fingerprint cannot see an executor, so executors stay (the brief's option B is rejected for that reason).
- **The shadow key** for a family with programs is its dependency files less the builders, then
  `programs=<digest>`, then today's tail. The digest is over the family's fixtures' entries in
  `testdata/program_fingerprints.json`, for every GOARCH in it, so every runner computes the same key.
- **A family with no program keeps today's key, unchanged.**
- **The family's programs are a new `programs` field in its parity record:** the fixture directories whose config's
  adapter is defined in that family's own adapter file. Where one adapter file serves several families (`arch_deepseek.go`,
  `gemma12.go`, `arch_llama.go`, `arch_qwen3_moe.go`), the family the fixture's `model_type` names takes it. That gives
  31 families with programs and 12 without: `deepseek_v2`, `kimi_k2`, `gpt-oss`, `gpt2`, `granitemoehybrid`,
  `internlm2`, `laguna`, `qwen2`, `qwen2_moe`, `qwen3`, `qwen3_vl`, `spark2_5`.
- **The shadow job** compares the two keys from the committed tables, so it no longer builds the kernel suite. The
  kernel table moves both keys alike.
- **Explained disagreements:**
  - an old-only family whose every moved input is a builder file;
  - a new-only family whose only moved input is its program list (fixtures added to its key, as at this PR).
- **Unexplained:** every other disagreement. In particular, a family whose programs move while today's key does not
  means a file outside its dependencies changed a program.
- **The limit, stated before the week.** A builder change on a path no fixture exercises leaves every program
  unchanged, so it reads as an explained old-only verdict, exactly like a refactor. Week 2's grade therefore reads
  each such push's builder diff.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| W1 equivalence | a decoder test computes the week-2 key itself (`familyDepFiles` less the builders, the program table, the kernel table) and finds it equal to `paritykey`'s for all 43 families; for the 12 without programs it equals today's key | never | one run | seconds |
| W2 the mapping | every fixture in the program table is in exactly one family's `programs`; each listed fixture has an entry for every GOARCH; its config's adapter is defined in a file of that family's `own` list. Mutation: moving `testdata/llama-tiny` to `voxtral`'s list turns it red | never | one run each | seconds |
| W3a refactor-blind | a throwaway commit adding an unused constant to `decoder/arch.go`: today's key re-stales all 43; `gate shadow` gives 31 old-only, each explained by `decoder/arch.go`, and 12 agree-stale | — | one run | under a minute |
| W3b change-sensitive | a throwaway commit changing one parameter in `arch_gemma3.go` (with the program table's `-update`): `gemma3` agree-stale, the other 42 agree-fresh | — | one run | under a minute |
| W3c a program moved from outside | a throwaway commit changing only `testdata/llama-tiny`'s entry in the program table: `llama` new-only, unexplained, exit 1 | — | one run | seconds |
| W4 classifier | `TestCompare_classes` covers every week-2 case; `gate mutation` dropping the "every moved input is a builder" condition turns it red | never | one run | seconds |
| W5 the job across this PR | `gate shadow origin/main <this PR's head>`: 31 new-only, each explained by its program list; 12 agree-fresh; none unexplained | — | one run | under a minute |

**Week 2's gates, graded 2026-10-27: W1–W5 all PASS** (`docs/measurements/vk5-2026-10.md` §6).
- 31 families have programs, from 37 fixtures; 12 keep today's key.
- W3a: a builder-only edit gives 30 old-only, all explained, and 12 agree-stale. `bailing_hybrid`, which uses no shared
  set, stays agree-fresh, a departure from the registered count that is recorded there.
- W3b: a Gemma 3 parameter change re-stales `gemma3` alone.
- W3c: a program moved without a dependency is new-only and unexplained.
- W5: 31 new-only, each explained by the program list, across this PR.
- W2's test gained the registered tie-break before its plant was run; the record says why.

The week starts at the first `vk-shadow` run on main after this PR merges.

**Amended 2026-10-27 after Gemini's review of #310, before any week-2 run is graded** (`docs/measurements/vk5-2026-10.md` §8):
- `decoder/gemma12.go`, the `gemma` and `gemma2` adapters, is a builder.
- A fixture listed on both sides whose fingerprints moved is a moved input, even when the program list also changed.
- An empty program table is an error.
- `bailing_hybrid`'s record now names the core, loader, quant and MoE sets and deepseek's MLA forward, which its forward
  runs through.
- **Week 2 starts at the first `vk-shadow` run on main after that PR merges;** earlier runs are not graded.

**Week 2's stopping rule** is SD6's: one week of pushes to main from the week-2 job's first run, with no unexplained
disagreement. In addition, every push with an explained old-only verdict has its builder diff read in the record and
classed either as a refactor or as a behavior change on a path no fixture exercises. A behavior change of the second
kind is a finding: the switch waits until a fixture covers that path. The switch is its own PR and its own decision.

**Possible week 3 (option C), not started.** A numerics fingerprint of decoder's own ops (a VK1-style suite over the
executor's op kinds on fixed inputs) would let the executor files leave the key the way the kernel table let the aikit pin
leave it. The hand-written forwards are whole-model paths, so they would need family-level fixtures. It needs its own
gates (reach, mutation). Size L.

### 5.9 VK6 — the daily canary and the coverage audit

- **The canary.** Each day, re-validate a rotating sample of carried-over (family, backend) pairs on real checkpoints,
  so every pair is re-checked within two weeks. A blind spot in a fingerprint then shows up as a canary failure, at
  most two weeks late.
- **The coverage audit.** The daily runs record which (operation, shape class, dispatch path) each real model reaches,
  from SB11's counters, and fail when VK1's suite or SB1's kit does not reach one. That holds the fast tier to the way
  the system actually calls it. M.

### 5.10 What it buys

Most commits stop re-staling anything; an aikit bump with unchanged numerics costs nothing; the manifest stops being a
merge point; real-checkpoint runs move to the daily run and to tags.

## 6. Track MM — the memory-management subsystem

### 6.1 Today

Memory decisions are spread over 34 decoder files (7,100 lines, touched by 141 commits in 30 days) and four internal
packages:

| group | files | lines | commits, 30 days | role |
|---|---:|---:|---:|---|
| `decoder/residency.go` | 1 | 2,019 | 58 | admission and resident builds, with memory checks inside the chain |
| `decoder/fitguard.go` | 1 | 1,022 | 28 | fit arithmetic, weight and KV pricing, the backend predictions |
| `decoder/fitplan.go`, `decoder/fitquery.go`, `decoder/metal_fitctx.go` | 3 | 656 | 11 | fit plans, queries, the Metal context cap |
| `decoder/prefill_budget.go`, `decoder/scratch.go`, `decoder/prefill_chunk_scratch.go` | 3 | 653 | 21 | prefill and step scratch |
| `decoder/weightbytes.go`, `decoder/residentneed.go` | 2 | 427 | 25 | weight and resident byte counts |
| `decoder/kvresolve.go`, `decoder/kvgeometry.go`, `decoder/kvpolicy.go`, `decoder/kvscope.go` | 4 | 529 | 23 | the KV plan |
| `decoder/moepool.go`, `decoder/moepaging.go`, `decoder/moeworkingset.go`, `decoder/layerpaging.go` | 4 | 890 | 5 | expert and layer pager policies |
| probes: `decoder/hostram_*.go`, `decoder/memwatch_*.go`, `decoder/swapwatch.go`, `decoder/faultcount_*.go`, `decoder/coldload*.go`, `decoder/giwmap_*.go` | 16 | 904 | 12 | reading the OS |
| `internal/hostmem` | 6 | 458 | 17 | the available reading (`cmd/hostmem`) |
| `internal/swapguard` | 2 | 235 | 2 | the swap guard |
| `internal/fitcmd` | 1 | 312 | 5 | `goinfer fit` |
| `internal/kvmgr` | 5 | 1,344 | 11 | KV blocks and the prefix index |

- `decoder/fitguard.go:fitCheckFor` reads the checkpoint's tensor headers and the machine while it decides;
  `metalWillBeResident` and `cudaWillBeResident` predict a backend's build instead of asking the backend.
- Because decisions read the live machine, their tests do. The same load passes by night and is refused by day with the
  owner's apps open (2026-10-20: a 5.1 GB budget against a 5.6 GB need), and the night queue spends rows on it (§1.3).
- **`internal/kvmgr` is the model to copy:** it imports only the standard library (`crypto/sha256`, `encoding/binary`,
  `errors`, `slices`), its tests are oracle tests, and it took 15 commits in 60 days while `decoder/residency.go` took
  58 in 30.

### 6.2 The contract (sketch)

```go
// Package mem plans, accounts for and arbitrates every byte goinfer holds. It imports nothing from goinfer.
// mem/probe is the only package that reads the OS or a device; everything here takes its readings as data.
package mem

type Tier uint8  // VRAM, Unified (Apple), HostAnon, HostPinned, FileMapped
type Life uint8  // Process, Model, Session, Request, Step
type Class uint8 // Weights, Experts, KV, RecurrentState, PrefixCache, Scratch, Staging, Tower, Aux

// A Need is one thing someone will allocate, declared before it is allocated.
type Need struct {
	Owner     string // "layer 12 kv", "experts", "prefill scratch", "vision tower"
	Class     Class
	Life      Life
	Bytes     int64  // as requested; a tier's rounding is the allocator's (CUDA rounds 2 MiB and over up to 2 MiB)
	Tiers     []Tier // acceptable placements, best first
	Evictable bool   // may be shed under pressure: experts, cached KV blocks, the prefix cache
}

// Demand is what one load asks for.
type Demand struct {
	Fixed, PerSession []Need
	PerPosition       int64 // KV and recurrent state per token, summed over layers
}

// Machine is a snapshot of one box, taken by mem/probe or read from a recorded profile.
type Machine struct {
	Name       string
	Free       map[Tier]int64 // goinfer's available reading, per tier
	SwapUsed   int64
	Compressed int64
}

// Policy holds the guard thresholds as data. A threshold change needs the owner's word, recorded here.
type Policy struct {
	Margin      map[Tier]int64
	SwapGuardMB int64
	ApprovedBy  string // only Francis writes this
	ApprovedOn  string
}

// Solve is pure: the same inputs give the same plan. A Decline names the need that does not fit, and why.
func Solve(d Demand, m Machine, p Policy) (Plan, *Decline)

// Plan lists the reservations per tier and line, the context cap, the KV blocks, the expert slots, the prefix-cache
// budget, and the arithmetic line the load banner prints.
type Plan struct{ /* … */ }

// Ledger is the runtime account: every allocation is a lease that names its plan line.
type Ledger interface {
	Reserve(n Need, line LineID) (Lease, error)
	Usage() Usage // by tier, class and life; Usage.Unplanned counts bytes with no plan line
}

type Lease interface{ Release() }

// Arbiter is the one budget for the evictable classes (KV11): it grants bytes and sheds leases under pressure.
type Arbiter interface {
	Grant(c Class, bytes int64) int64
	Shed(p Pressure) []Lease
}
```

### 6.3 Invariants, each a fast test

| # | invariant | how it is checked |
|---|---|---|
| I1 | nothing is allocated without a lease that names a plan line | every test asserts `Usage().Unplanned == 0`; tier 0 |
| I2 | every lifetime closes to zero: after a request, a session, a model's `Close` | a lifecycle test per backend on a tiny fixture; tier 0 (MEM2's class of leak) |
| I3 | the ledger agrees with the measured process, within a stated per-backend tolerance | one tiny-model test per backend: `VmHWM` on Linux, the task's footprint on macOS, the device's allocated bytes on CUDA and Metal; seconds |
| I4 | `Solve` is pure | property tests: same inputs, same plan, across runs and goroutines; microseconds |
| I5 | `Solve` is monotone | more free memory never gives a smaller plan; a larger demand never gives a larger context cap |
| I6 | there is one decider | every decline is a `*Decline` from `Solve`; a source scan finds no other estimate of a backend's allocation |
| I7 | thresholds are data with an approval stamp | a test fails when `Policy` differs from the committed one without `ApprovedBy` and `ApprovedOn` (the `night.py add` rule, carried by a type) |
| I8 | budgets key on computed bytes, never on the OS's account of them | darwin's UBC makes RSS report what survived rather than what was asked for (`CLAUDE.md`, "A guard that INVERTS"); the OS reading enters only as `Machine.Free` and in I3 |

### 6.4 How it is tested

- **The machine zoo.** Recorded snapshots, as data: the M1 Pro with the owner's apps open (7.0 GB available and a 4.9 GB
  budget in the 2026-10-15 G2-R log; a 5.1 GB budget against a 5.6 GB need on 2026-10-20), the M1 Pro idle, nobara's
  RTX 2070 SUPER with and without `kwin_wayland`'s 16 MiB context, and a CI runner with no GPU. `cmd/hostmem` already
  makes the available reading; MM0 adds a form that writes a snapshot. `Solve` runs over the zoo × every family config
  × the option grid in seconds.
- **The trace simulator.** Replay recorded request traces (Claude Code's 19,692–20,199-token first requests and its
  8,000–17,000-token thread switches, `docs/measurements/kv0-traffic-2026-10-13.md`) and recorded expert routing
  (`decoder/routercapture.go:routerCapture` captures it) through the arbiter, and read hit rates, evictions and peak
  bytes per class. This answers KV11's open question, a fixed split against one budget, without a GPU.
- **Ledger-closes, per backend** (I3): one tiny model, seconds.
- **Daily calibration.** Each box loads at the plan's edge (the largest model `Solve` says fits) and records plan
  against measured. Drift past a stated bound is a finding, not anyone's red.

### 6.5 Phases

| phase | what moves | gate (sketch) | size |
|---|---|---|---|
| MM0 | the probes (`hostram_*`, `memwatch_*`, `swapwatch`, `faultcount_*`, the cold-load probes) and `internal/hostmem` and `internal/swapguard`'s readings behind `mem/probe`; a fake probe for tests | identical readings through the interface; a source scan finds no OS memory read outside `mem/probe` | S |
| MM1 | `Solve`, out of `fitguard.go`, `fitplan.go`, `fitquery.go`, `prefill_budget.go`, `weightbytes.go`, `residentneed.go`, `kvresolve.go` and `metal_fitctx.go`; tensor-header reads move to the caller, which hands `Solve` numbers | a decision census: for every family config × zoo machine × option cell, the same decision and the same arithmetic line as today's code, byte for byte; tier 0 | M |
| MM2 | backends report their own `Demand` for a build, their rounding included; `metalWillBeResident` and `cudaWillBeResident` go | the census unchanged; I3 per backend | M |
| MM3 | a ledger and leases in the allocators: CUDA (`cuMemAlloc`'s 2 MiB rounding priced), Metal (buffers; no-copy aliasing as `FileMapped`), WebGPU (`gpu/bufaccount.go`) | I1 and I2 per backend at tier 0; I3 | M |
| MM4 | the arbiter, KV11: one budget for KV blocks, expert slots and the prefix cache; `internal/kvmgr` asks it for blocks | the simulator's policy registered and graded first, then device runs | L; after KV4 and KV7 |
| MM5 | demand from the program: each operation declares its buffers (weights, per-sequence state, per-step scratch), and `Demand` is their sum plus the backend's deltas | the census unchanged; a planted new family with no hand-written memory line is priced correctly | M; after OP4 |
| MM6 | `night.py add` and `goinfer fit` call `Solve` against the target box's recorded profile at queue time | replay the last three weeks' refused night jobs: each would have been refused at queue time | S; after MM1 |

#### MM0's gates, registered 2026-10-26 before the code (nobara-mm01)

What moves: the OS memory readings in `decoder/hostram_*`, `decoder/memwatch_*`, `decoder/faultcount_*`, the cold-load
residency sample (`giwResidentFraction`), `decoder/swapwatch.go`'s sampler, all of `internal/hostmem` (which `internal/mem/probe`
absorbs; `cmd/hostmem` prints the same lines), `internal/swapguard`'s readings, the Metal module's direct `hw.memsize` reads and
`cmd/gate`'s `/proc/<pid>/status` read. `decoder`'s exported readers (`HostRAMBytes`, `HostRAMAvailableBytes`, `SwapUsedBytes`,
`SelfSwapReader`, `SelfSwapBytes`, `StartSwapWatch` and its types) stay, as delegates (§3 item 3). Out of scope, named so the
gap is visible: the device readings (`cuMemGetInfo`, Metal's `recommendedMaxWorkingSetSize` and `currentAllocatedSize`) need a
device handle and the device library, which §9 keeps out of the probe's imports; MM2 has the backends report them.

All four are tier 0: no model, no device, no timed run, so the timing lock is not taken beyond `touched.py`'s shared hold.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM0-1 identical readings | (a) every parser test that existed moves with its parser and passes with its expected values unchanged; (b) a frozen copy of each pre-move reader (from `origin/main` at bc365a837), kept in a test file, agrees with the probe on the live box: `Physical`, `SelfSwap`'s availability and `FaultCounts`' ok exactly; `Available`, `SwapUsed` and `SelfSwap` equal, or between two reference reads that bracket the probe's, within 5 tries; (c) every decoder seam (`hostRAM`, `hostRAMAvailable`, `coldAvailBytes`, `coldRAMBytes`, `coldResidentFraction`, `sidecarHostRAM`) and the exported readers return the probe's value | the box has no swap configured and (b)'s swap leg compares 0 with 0 (reported, not passed) | all run to completion | seconds; Linux natively, darwin and windows in CI's root jobs |
| G-MM0-2 a fake drives the fit guard | with `probe.Fake` installed through one test helper, each outcome is reached and named: `guardFit` (through `fitCheckFor` on a tracked tiny fixture) proceeds on an unknown reading, proceeds when it fits, auto-pins, refuses a pinned context, refuses at the floor; `decideGIWFit` reaches the same five; `planColdLoad` through `prepareColdLoad` reaches read-first, no-read-around and nothing | an outcome is reached only by setting a non-probe input (each case changes the fake's readings alone, plus the pin) | all cases run | seconds |
| G-MM0-3 one reader | a source scan (go/ast, not text, so comments do not count) over every non-test `.go` file in all seven modules, outside `testdata/`, finds none of the OS memory reads (the literals `/proc/meminfo`, `MemTotal:`, `MemAvailable:`, `SwapTotal:`, `SwapFree:`, `VmSwap`, `VmHWM`, `VmRSS`, `statm`, `smaps`, `hw.memsize`, `vm.swapusage`, `kern.memorystatus_level`, `vm.page_`, `GlobalMemoryStatusEx`, `host_statistics64`, `task_info`; the selectors `Getrusage`, `Sysinfo`, `SYS_MINCORE`, `Mincore`) outside `internal/mem/probe`. It must go red first: a planted source per pattern is flagged. And `internal/mem/probe` imports the standard library and `golang.org/x/sys` only | the scan reads zero files, or fewer than the tree's count of non-test `.go` files less the excluded ones (a scan that read nothing passes vacuously) | runs to completion | seconds |
| G-MM0-4 builds | `go build` and `go vet` of the root for GOOS linux, darwin and windows; `metal` for darwin; `gpu` and `cuda` for linux with their tags (`cuda goinfer_testhooks`, `gpu goinfer_testhooks`); CI's pinned `staticcheck` on the touched modules after its U1000 canary goes red; `gofmt -l` empty on every touched module | the staticcheck canary does not go red (then nothing was checked) | all run | about a minute |

**MM0 read, 2026-10-26 (nobara, linux/amd64): all four pass.** G-MM0-1: `TestHost_matchesFrozenReaders` and
`TestMemProbe_seamsReadTheProbe` pass on nobara (swap in use was nonzero, so the swap leg was a real comparison); the darwin and
windows legs run in CI's root jobs. G-MM0-2: `TestFitGuard_fakeProbeReachesEveryOutcome` reaches all five `guardFit` outcomes (on
internlm2-tiny with its context raised to 131,072 in a temp copy), all five `decideGIWFit` outcomes and the three cold-load
actions. G-MM0-3: `TestOnlyProbeReadsOSMemory` scans 720 non-test files, flags every planted pattern and passes;
its first run flagged `cmd/gate`'s `/proc/<pid>/status` field names, which now come typed from `probe.ReadProcMemory`.
G-MM0-4: builds and vets clean on all three systems and the tagged modules; the pinned staticcheck's canary went red, then
the touched packages were clean on all three systems (Metal's untagged run lists unused test hooks in files this change does
not touch; CI does not run staticcheck on Metal). The parity manifest stayed fresh: no moved file is in a family's deps.

#### MM1's decision census, registered 2026-10-26 before any code (nobara-mm01)

MM1 moves the fit arithmetic into `mem.Solve`. Its bar: every decision today's code makes, on every cell below, comes out the
same from `Solve` and its callers, byte for byte, the printed and returned lines included. This step records the census
from today's code (`origin/main` at the commit this branch starts from) and commits it. It changes no product code.

**The decisions** (each with every line it prints or returns):

| id | decision | outputs recorded |
|---|---|---|
| D1 | the load-time resident guard: `fitCheckFor` and `decide`, as `guardFit` and `FitQuery` use them | the pin, the refusal and its text, `DenseStreamable`, need, budget, weight, KV and source bytes, `known`, `ratio`, the line `guardFit` would print (cap note or warning), `arithmetic()` |
| D2 | the `.giw` guard: `decideGIWFit` | every `giwFit` field, the refusal text, the note |
| D3 | `Model.Plan`, for cpu, cuda, metal and webgpu | every `Plan` field, `Reason`, `NeedBytes()` |
| D4 | `Model.MetalFitCtx` | every `MetalFit` field |
| D5 | request-time admission: `AdmitPrefillMemoryShare`, and `FitBudgetSummary` | the error text (or none), the summary's five values |
| D6 | `resolveWeightCacheBudget` | the budget |
| D7 | `ResolveKV` for cpu, cuda, metal and webgpu, and `WebGPUCtxCeiling` | delivered precision and reason; the ceiling |

**The cells.**
- **Families:** every tracked tiny checkpoint that loads on the CPU in the plain test binary (directories with a `config.json`
  and tracked weights, and tracked `.gguf` files), each at its own window and, for directories, at a long variant (a temp
  copy with `max_position_embeddings` 131,072) so the auto-pin and floor branches are reachable. D2 and D7 also take the tracked
  real-shape configs (`decoder/testdata/laguna_configs/`, `testdata/nemotron-h-configs/`, `testdata/glm_ocr_real_config.json`),
  which need no weights.
- **Options:** for D1 and D2, backend cpu, cuda or metal, with the GPU backend linked or not (a stub in the registry, as
  `withFakeCUDACompiled` does), × quant int4, int8, f32, int4mix (D1) × context unpinned, 4,096 or 32,768 × KV f32, f16 or i8
  × stream-weights off or on (D1) × the CUDA expert cache off or on (cuda only) × the no-fit-guard knob off or on (one cell
  each). D3–D5 run on each family loaded at int4 and at f32, over a request grid named in the test.
- **Machines:** a zoo of recorded snapshots in `testdata/machines/*.json`. Each names its source, and an entry not read from
  a log is marked as an assumption: the M1 Pro with Francis's apps open (7.0 GB available, the 2026-10-15 G2-R log); the M1 Pro
  under the 2026-10-20 load (a 5.1 GB budget); the M1 Pro near idle (`kern.memorystatus_level` 79%, the B41 record); nobara
  with and without `kwin_wayland`'s CUDA context (host and VRAM read on 2026-10-26); a GitHub runner with no GPU; and a
  machine that reads unknown. Every cell also runs at its own edges: for each decision that changes with free memory, the
  lowest reading at which each outcome holds, and one byte under it, found by search and stored as plain numbers.
- **The host class.** Weight bytes depend on the CPU, because the loader repacks int4 differently on AVX2, AVX-512 VNNI and
  arm64. So the census is keyed by host class (GOARCH plus the measured bytes-per-element table), as `parityWantInt4ByArch` is
  keyed by architecture. This step records nobara's class natively and the VNNI class under Intel SDE, as the int4 parity
  golden was (the read below: they are one class). A class with no entry, arm64 included, is a capture run: the test logs the census hash and skips the comparison.
  The Mac records arm64 when a session there takes it.

**Gates.**

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM1-C1 coverage | every outcome of every decision is reached by at least one cell. D1 and D2: proceed unpriced, fits, warns, auto-pins, refuses a pin, refuses at the floor (D1 also a refusal with `DenseStreamable`, and a CUDA build-peak need). D3: each `Placement` the code can return for each backend. D4: each `Why`. D5: admitted and refused, at least once each. A missing outcome fails the test, naming it | fewer families load than the tracked fixture count less the ones named as not loadable on the CPU (a census that loaded nothing covers nothing) | runs to completion | seconds |
| G-MM1-C2 determinism | two runs in one process, and a run in a second process, give the same census hash | the probe fake is not installed (a live reading leaked in) | runs to completion | seconds |
| G-MM1-C3 the golden | the census equals `testdata/mm1_census.json.gz` for this host class, cell for cell and byte for byte. It is written once by `-update` in this PR and then frozen: MM1's `Solve` PRs reproduce it without `-update`. A cell added, removed or changed is a red run, and the PR that changes it says why | the host class has no entry (a capture run, reported as such, not a pass) | runs to completion | seconds |
| G-MM1-C4 tier | the census test runs in under 30 s on nobara and under a minute in CI, with no model asset, no device and no network | — | — | measured on the first run and recorded here |

**MM1 census read, 2026-10-26 (nobara, Ryzen 7 3700X; timing lock free, shared hold): all four pass.** The census is 29,637
groups. Per decision: D1 8,435, D2 3,948, D3 11,760, D4 2,310, D5 2,730, D6 2 and D7 451, plus the loads group. It covers 105
family variants, and 210 family loads succeeded. The loads that fail are the towers, encoders and ASR models, which the loads
group lists (a family that stops loading is a red run). G-MM1-C1: every listed outcome is reached. G-MM1-C2: two in-process runs
and a second process agree (`cce976c3…`). The first run did not agree: the long variants' temp-dir names reached the fit guard's
line, and they now keep the fixture's name. G-MM1-C3: the golden is written for the amd64 class (bytes per element f32 4,
int8 1.015625, int4 0.5625). Under SDE `-hsw` the census matches it, and under `-icx` the class and the census hash are the
same, so VNNI and AVX2 hosts share it. arm64 is a capture run until the Mac records it. G-MM1-C4: `TestDecisionCensus` takes 16.3 s
and the second-process test 16.6 s. One build of the census took 37 s on the first run and 22 s after the cuts below, and takes 8 s with D1, D3 and D4 in parallel
(D2 stays serial at 5 s, because `decideGIWFit` reads the probe seam on every evaluation; `-race` is clean). Two cuts to the registered grid came first, and they change no
decision's reach: stream-weights only for a `.gguf` (it moves nothing for a directory), and the CPU's KV f16 cell dropped (the
CPU's precision is `KVQuant`'s). The golden holds a 16-hex SHA-256 prefix per group (199 KB); `-census.dump` writes the full text
(92 MB) to diff a red run against its base.

**CI on the PR, root-windows (run 40433980269): red, then fixed on the branch.** Every group differed because Windows built the
keys from OS paths (`..\internal\vision\…`), which the golden holds as `internal/vision/…`. The host class matched the
golden. The keys now use forward slashes on every OS, and the long variants hard-link their weights, falling back to a copy
(a symlink needs a privilege on Windows that the runner happened to have). **G-MM1-C4 is not met on Windows:**
`TestDecisionCensus` took 166 s there, against a minute registered for CI. D2, serial, took 58 to 66 s of each build there,
against 5 s on nobara. The fix belongs to the next MM1 step: once `Solve` takes the available reading as data, D2 no longer
goes through the shared probe seam and runs in parallel like D1, D3 and D4. **D2's part done 2026-10-26:** `decideGIWFit` delegates to `decideGIWFitAt`,
which takes the reading as an argument, so D2 runs in parallel (1 s against 5 s on nobara; each census test 8 s against 16 s).
The golden did not move, `-race` is clean, and a serial check holds `decideGIWFit` through the probe fake equal to
`decideGIWFitAt` at the zoo's readings. G-MM1-C4 on Windows is still unmeasured: CI runs `go test` there without `-v`, so it
reports only the decoder package's time (410 s on 2026-10-27, run 40520519837), not the census's.

#### MM1's `Solve`, registered 2026-10-26 before the code (nobara-mm01; Francis approved all of MM1 at 17:35)

**What moves, in three PRs, each held to the census.**
- **MM1a, this PR: the host guards.** D1, D2, D5 and D6 move into a new `internal/mem`: the resident guard's need, budget,
  ratio, auto-pin search, refusal, warning and cap lines, the `.giw` guard, request admission, the load banner's budget, and
  the weight-cache budget. `mem` takes numbers and returns decisions with their lines. A load's KV comes in as a curve, the
  bytes at a context, because KV is not linear in the context (sliding windows flatten it). Machine state comes in as a
  `mem.Machine`. The thresholds come in as a `mem.Policy` whose default holds today's values (0.70, 0.75, the 2,048 floor, the
  1 GB `.giw` margin, the 256 MiB attention scratch); no value changes. The decoder keeps the reads: tensor headers, configs,
  the probe. It also keeps the types its tests and callers use (`fitCheck`, `giwFit`, `FitDeclineError`), as adapters over
  `mem`. I7's approval stamp is not added here: only Francis writes it, and no threshold moves.
- **MM1b: device planning.** D3 (`Plan`) and D4 (`MetalFitCtx`), with the byte counts they read (`weightbytes.go`,
  `residentneed.go`) handed in as numbers.
- **MM1c: the KV precision rules.** D7 (`ResolveKV`), and I6's source scan: no estimate of a load's memory outside `mem`.

**MM1a's gates.**

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM1-S1 the census | `TestDecisionCensus` passes without `-census.update`, on nobara and on CI's amd64 jobs (linux and windows): every group byte for byte | the census reports a capture run on an amd64 host (the golden's class not found) | runs to completion | 16 s on nobara |
| G-MM1-S2 imports | `internal/mem` imports the standard library only (an import test, as `internal/mem/probe` has) | the test reads no source file | runs to completion | seconds |
| G-MM1-S3 pure (I4) | each `mem` decision, on 10,000 generated inputs, gives the same answer twice in one goroutine and once each in 16 concurrent ones; `mem` holds no package-level variable | — | runs to completion | seconds |
| G-MM1-S4 monotone (I5) | on the same inputs, more available memory never makes the resident or `.giw` guard's outcome worse (refusal, then auto-pin, then fits) or its pinned context smaller, and never turns an admitted request into a refusal | — | runs to completion | seconds |
| G-MM1-S5 the existing tests | the decoder's fit tests pass unchanged: `fitguard*_test.go`, `giwfitguard_test.go`, `fitguard_giw_pinned_test.go`, `prefill_budget_test.go`, `memprobe_test.go`, `fitquery*_test.go`; the `cuda`, `metal` and `gpu` modules build and vet with their tags | — | runs to completion | about a minute |
| G-MM1-S6 builds | build and vet for linux, darwin and windows; CI's pinned staticcheck after its canary goes red; gofmt; `touched.py`; the parity manifest stays fresh, or is refreshed once with its goldens before the push | the staticcheck canary does not go red | runs to completion | about two minutes |

**MM1a read, 2026-10-26 (nobara): all six pass.** G-MM1-S1: `TestDecisionCensus` passes without `-census.update`, every group
byte for byte, and so does the second-process test (16.5 and 16.8 s). G-MM1-S2: `internal/mem` imports the standard library only.
G-MM1-S3: 10,000 generated inputs give the same answers twice in one goroutine and in each of 16 at once, and the package
declares no variable. G-MM1-S4: over a 13-reading ladder, no decision gets worse with more memory. It goes red on a planted
defect (a refusal above 64 GiB), on its first input. G-MM1-S5: the 59 decoder fit tests named in the gate pass unchanged, as do
`internal/fitcmd` and `internal/swapguard`; `cuda`, `metal` and `gpu` vet with their tags. G-MM1-S6: vet on linux, darwin and
windows; the pinned staticcheck after its canary went red (it found `fitCheck.remedy` left unused, now removed); gofmt;
`touched.py`; the citation lint. The parity manifest stayed fresh. The decoder keeps `fitCheck`, `giwFit` and
`FitDeclineError` as adapters: a `fitCheck` becomes a `mem.HostCheck` (its KV as a curve over `estimateKVBytes`), and
`guardFit`, `decideGIWFit`, `AdmitPrefillMemoryShare`, `FitBudgetSummary` and `resolveWeightCacheBudget` call `mem`.


#### MM1b's gates (`Plan` and `MetalFitCtx`), registered 2026-10-26 before the code (nobara-mm01)

**What moves.** `Model.Plan`'s placement arithmetic (the context search toward the 4,096 floor, the resident and expert-cached
fits, the slot count, every reason line, and the CPU's weight-paged fallback) becomes `mem.SolvePlacement`. `MetalFitCtx`'s
search becomes `mem.SolveMetalCtx`, which calls `SolvePlacement` itself. The decoder keeps the eligibility checks: missing
features, `ResidentEligible`, and WebGPU's decline of a KV flag on the families that do not implement it. Those read the
model's architecture, not its bytes. The decoder also keeps the byte counts it hands in: dense, experts, the host copies, and
KV as a curve per backend. `mem` gets its own `Placement`, which the decoder maps to its public one (§3 item 3: never an
alias). `Policy` gains the plan floor (4,096) and Metal's fit margin (384 MiB) at today's values.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM1b-1 the census | `TestDecisionCensus` passes without `-census.update`, D3 and D4 included, byte for byte | the golden's class not found on an amd64 host | runs to completion | 16 s |
| G-MM1b-2 pure and monotone | `TestSolve_pure` and `TestSolve_monotone` cover `SolvePlacement` and `SolveMetalCtx` on generated inputs. More free memory never moves a placement down the order decline, weight-paged, expert-cached, resident; never gives fewer slots at the same placement or a smaller context at the same placement; and never gives `MetalFitCtx` a smaller context. The monotone test goes red on a planted defect | — | runs to completion | seconds |
| G-MM1b-3 the existing tests | the decoder's `Plan` and `MetalFitCtx` tests (`fitplan*_test.go`, `metal_fitctx*_test.go`, every test calling `.Plan(` or `MetalFitCtx(`) and `metal/`'s fit tests (vet on darwin) pass unchanged; `cuda`, `metal` and `gpu` vet with their tags | — | runs to completion | about a minute |
| G-MM1b-4 builds | as G-MM1-S6 | as G-MM1-S6 | runs to completion | about two minutes |

**G-MM1b-2 amended before its pass, 2026-10-26 (nobara-mm01), with the evidence.** The registered rule ("never fewer slots at
the same placement") fails on today's planner, by design: it takes context before experts. Measured on a generated MoE (8
experts, top-2; first written as top-8, which is impossible with two slots) at 16 GiB free, it was expert-cached at context 6,080 with 8 slots; at 32 GiB, expert-cached at context
83,328 with 2. The census shows the same arithmetic before and after the move, so the rule, not the move, was wrong. Holding
slots monotone would change placements, which is Francis's call; it is not done here. The amended rule: placement never moves
down the order; at the same resident or expert-cached placement, the context never shrinks, nor the slots at the same context;
`MetalFitCtx`'s context never shrinks. A refusal's context only echoes the attempt and is not compared. It goes red on a
planted defect (a refusal above 32 GiB free).

**MM1b read, 2026-10-26 (nobara): all four pass, G-MM1b-2 as amended.** G-MM1b-1: `TestDecisionCensus` passes without
`-census.update`, D3 and D4 included, every group byte for byte. G-MM1b-2: `TestSolvePlacement_pure` (10,000 inputs, twice and
in 16 goroutines) and `TestSolvePlacement_monotone` pass. G-MM1b-3: the 18 decoder `Plan` and `MetalFitCtx` tests pass
unchanged. `TestPlan_extraBytesReservedAheadOfExperts` skips for want of `testdata/gemma4-moe-tiny`, as it does on main. `cuda`,
`metal` and `gpu` vet with their tags. G-MM1b-4: vet on three systems, the pinned staticcheck (it found `gb` left unused, now
removed), gofmt, the layout test, the citation lint; the parity manifest stayed fresh.

#### MM1c's gates (`ResolveKV`, and I6's scan), registered 2026-10-26 before the code (nobara-mm01)

**What moves.** `ResolveKV`'s resolution order becomes `mem.ResolveKV`: no KV for a model without attention; the CPU cache's
rules (batched decode reads f32 only; f16 and i8 where the family allows it); Metal's (MLA keeps f32; auto is f16); CUDA's
(int8 not yet; the drafter keeps f32; a family the policy declines keeps f32; auto per the policy); and the rest (auto is
f32). The decoder keeps the policy registry and its public types (`KVFacts`, `KVF16Policy`, `RegisterKVF16Policy`,
`CUDAKVF16Decline`, `CUDAKVF16AutoDecline`), because the cuda module registers against them. The policy reaches `mem` as
closures bound to the load's options. The decoder also keeps `resolveKVHook`, a test seam, and the architecture predicates,
which `mem` takes as booleans.

**I6, as far as MM1 reaches.** A source scan over every non-test Go file outside `internal/mem` finds each reference to a
guard threshold: `WeightsMemFraction`, `fitMemFraction`, `prefillAvailFraction`, `giwMemMargin`, `residentMemFraction`,
`MetalFitMarginBytes`, `ctxPlanFloor`, `fitWarnRatio`, `ctxFloor`, and `mem`'s own constant names. Each must be a constant
declaration that aliases `mem` (directly or through one decoder alias), or a site on a ratchet list whose per-file count may
only fall. The list today: `gpu/kv_slots.go` 2 and `metal/backend.go` 3, the backends' own guards, which MM2 removes. The
scan must go red on a planted reference.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM1c-1 the census | `TestDecisionCensus` passes without `-census.update`, D7 and every decision that reads `ResolveKV` (D1, D2 through `kvPricingFor`) included | the golden's class not found on an amd64 host | runs to completion | 16 s |
| G-MM1c-2 the existing tests | every decoder test that calls `ResolveKV`, `kvRefusal`, `KVPrecisionReports` or `KVDelivered`, and the `cuda` module's KV policy tests (vet with tags; the device ones run on nobara), pass unchanged | — | runs to completion | about a minute |
| G-MM1c-3 I6's scan | `TestNoGuardThresholdOutsideMem` passes on the tree, reads every non-test file, and goes red on a planted reference | it reads fewer files than the tree holds | runs to completion | seconds |
| G-MM1c-4 builds | as G-MM1-S6 | as G-MM1-S6 | runs to completion | about two minutes |

**MM1c read, 2026-10-26 (nobara): all four pass.** G-MM1c-1: `TestDecisionCensus` passes without `-census.update`. G-MM1c-2: the
decoder's six KV-resolution tests pass unchanged, the generated precision matrix among them. The `cuda` module's KV16 and KV4
identity and default tests pass on the RTX 2070 SUPER under the timing lock (47 cases with subtests), and so does the `gpu`
module's `ResolveKV` test. G-MM1c-3: `TestNoGuardThresholdOutsideMem` reads 726 non-test files. It finds exactly the ratchet
(`gpu/kv_slots.go` 2, `metal/backend.go` 3) and counts a planted reference. G-MM1c-4: vet on three systems and with the
module tags, the pinned staticcheck, gofmt, `test_touched.py`; the parity manifest stayed fresh. MM1 is done with this:
D1–D7 run through `internal/mem`, and the census holds byte for byte.

#### MM6's gates, registered 2026-10-26 before the code (nobara-mm01; Francis's answers of 22:37)

**Francis's decision (22:37):**
- 1a: refuse at queue time, with an override flag.
- 1b: the idle profile prices night jobs, the apps-open profile day jobs.
- 1c: new repeatable `--fit-model PATH --fit-opts "..."` flags, priced through `goinfer fit`. A job queued without them is not
  checked and gets a warning.

**What is built.**
- **`fit -profile FILE`** prices a checkpoint as `Load`'s own memory check would on that recorded machine, without loading it.
  It is `decoder.FitQueryAvailable`, which is `FitQuery` with the available reading taken from the profile instead of the
  probe. It prints one line (or `-json`) and exits 0 when the load proceeds (fits, auto-pins, or cannot be priced), or 4 when
  it would be refused.
- **`night.py add --fit-model PATH [--fit-opts "..."]`**, repeatable, runs that check for each declared load:
  - the profile is the box's night profile for a job with `--not-before`, else its day profile;
  - the binary is the chat build for the job's `-backend` (`cuda/cmd/chat`, `metal/cmd/chat`, else `demo/chat`), because
    what `Load` prices depends on the backends linked in;
  - a refusal stops the queueing unless `--fit-override REASON`, which the job records;
  - a job with no `--fit-model`, or a box with no recorded profile, is queued with a warning.
- The box-to-profile table: `nobara-pc` uses `nobara-no-desktop` at night and `nobara-desktop` by day; `macbookpro.lan` uses
  `m1pro-near-idle` and `m1pro-apps-open`.
- **Out of scope here:** Metal's own resident guard (`metal/backend.go`, against physical RAM) and CUDA's device-side
  context fit, which run in the backend at build time.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM6-1 the profile check | on tracked tiny fixtures, `fit -profile` against generated profiles answers as `decoder.FitQueryAvailable` does: exit 0 and `fits` at ample memory, exit 0 and `capped` at the auto-pin edge (long-context copy), exit 4 with `Load`'s own refusal text below it, exit 0 and `unpriced` at an unknown reading; and `FitQueryAvailable` at the probe's reading equals `FitQuery` | — | runs to completion | seconds |
| G-MM6-2 `night.py add` | with a scripted checker: a refused load is not queued and the message names the model, the profile and the refusal; `--fit-override` queues it and records the reason; a job without `--fit-model` queues with a warning; a `--not-before` job is priced on the night profile, any other on the day profile; an unknown host queues with a warning; the backend picks the binary | the tests do not reach `cmd_add` (they must queue into a temp queue dir) | runs to completion | seconds |
| G-MM6-3 end to end | `night.py add` with its real checker on nobara: a tracked tiny checkpoint against a profile too small for it is refused, and against `nobara-no-desktop` is queued (into a temp queue dir), through `go run ./demo/chat fit` | the build fails (that is a red run, not a void) | runs to completion | about a minute (the build) |
| G-MM6-4 the census | `TestDecisionCensus` passes without `-census.update` (`FitQuery` is refactored, not changed) | as G-MM1-S1 | runs to completion | 8 s |
| G-MM6-5 the replay (the Mac) | every night job on the Mac refused by the fit or swap guard from 2026-10-06 to 10-26, re-checked with `fit -profile` on the night profile for its model and options as queued, is refused (exit 4). A job whose refusal was the swap guard's (not priced by `FitQuery`) is listed apart, not counted against the rule. The commands are in the PR; the result follows the PR. | the job's model is no longer on the Mac's local disk (listed, not counted) | every refused job | the Mac session's estimate, minutes |
| G-MM6-6 builds | as G-MM1-S6, and `python3 scripts/test_night.py` | as G-MM1-S6 | runs to completion | about two minutes |

**MM6 read, 2026-10-26 (nobara): G-MM6-1, 2, 3, 4 and 6 pass; G-MM6-5, the Mac's replay, follows the PR.**
- G-MM6-1: `TestProfile_answersAsFitQueryAvailable` reaches all four verdicts, each with its exit code and with
  `FitQueryAvailable`'s answer. `TestFitQueryAvailable_isFitQueryAtThatReading` holds it equal to `FitQuery` at the probe's
  reading, for a directory and a `.gguf`.
- G-MM6-2: eight `FitCheckAtQueueTime` tests pass, and the whole `test_night.py` (62).
- G-MM6-3: through the real `go run ./demo/chat fit`, llama-tiny is refused on a 64 KiB profile and fits on
  `nobara-no-desktop`. The first try found that `go run` exits 1 for the program's exit 4, so `night.py` reads the JSON
  verdict, not the exit code. The scripted checker in G-MM6-2 now exits 1 as `go run` does.
- G-MM6-4: `TestDecisionCensus` passes without `-census.update`.
- G-MM6-6: vet on three systems, the chat builds' vet with their tags, the pinned staticcheck (untagged and `realckpt`),
  gofmt, the parity manifest fresh.
- The replay's commands and its pass rule are in the PR. Because the night profile is the idle Mac, the replay also
  measures whether the Mac's nights are idle: a refusal that printed about 7 GB available would not be refused on a 12.2 GB
  profile. That would be a finding about which profile prices night jobs (Francis's 1b), and it is reported as one.

**G-MM6-5 read, 2026-10-27 (mac-vk01, the Mac, `origin/main` `014cbd54`): FAIL by its rule. The finding is about the
night profile, not the code** (`docs/measurements/mm6-replay-2026-10-27.md`).
- Nine fit-guard refusals in the window. One model is no longer on local disk, so 8 are counted. No `.giw` or swap-guard
  refusals.
- `m1pro-near-idle` refuses 1 of the 8 (the 7B at 12.6 GB). It fits the other 7, which need 2.7–6.7 GB against its 8.5 GB
  budget.
- `fit -profile` prices each load within 0.1 GB of the need the real refusal logged (rows 2–8). The real refusals printed
  2.9–7.7 GB available, median 6.6, against the profile's 12.2: the Mac's nights were not idle.
- `m1pro-apps-open` refuses 7 of the 8. The 1.5B fits there; its night had 2.9 GB.
- **DECISION for Francis (1b):** which profile prices the Mac's night jobs, given these readings.

#### MM2a's gates (backends report their build demand), registered 2026-10-26 before the code (nobara-mm01; Francis's answers of 22:40)

**Francis's decision (22:40):**
- 2a: decoder keeps a default copy of each backend's demand report, and each backend registers its own at init, replacing
  it (the `ResolveKV` pattern), with a tier-0 test in each backend module that its report equals decoder's default.
- 2b: CUDA's 2 MiB allocation rounding stays out of MM2. MM2 is a pure move held to "census unchanged"; the rounding is
  proposed afterward as its own census-changing step, with a DECISION NEEDED and the census diff.

**What moves.** `cudaWillBeResident` and `metalWillBeResident` go. In their place is a registered report:
`RegisterBuildDemand(backend, func(BuildQuery) BuildDemand)`, with `DefaultBuildDemand(backend, q)` as decoder's copy. A
`BuildDemand` says:
- whether the build goes resident;
- whether its KV is device memory (the host fit then prices none);
- the host the build adds while it packs (CUDA: one packed layer and the staging blocks);
- the host peak of building an expert cache from a `.gguf` (CUDA C': weights, a packed copy, a pinned expert copy);
- the pinned expert copy a `.giw` load holds.

The fit guard (`kvPricingFor`, `fitCheckFor`, `priceCtxAndKV`, `giwPinnedExpertBytes`) and `FitQuery`'s decode path read the
report for the load's backend. The `cuda` module registers its report with its own staging constant and expert packing; the
`metal` module registers Metal's. Which architectures a backend runs resident stays decoder's feature table (SB1's to move),
read through the `FeaturesCovered` field that `BuildQueryFor` fills.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM2a-1 the census | `TestDecisionCensus` passes without `-census.update` | as G-MM1-S1 | runs to completion | 8 s |
| G-MM2a-2 the reports agree | in `cuda` (tag `cuda`) and in `metal` (darwin), a tier-0 test loads tracked tiny fixtures on the CPU, and for each config × the option cells (backend, expert cache, stream-weights) × a weight/expert ladder, the registered report equals `DefaultBuildDemand`. It goes red when a planted change moves one term. Metal's test runs in CI's darwin job (no Mac here); nobara vets it for darwin | the test compares fewer than 10 configs | runs to completion | seconds |
| G-MM2a-3 the existing tests | the decoder's fit tests pass, with the two predicate calls in `fitguard_cuda_test.go` and `fitguard_metal_test.go` read through the report (mechanical); `cuda`, `metal` and `gpu` vet with their tags | — | runs to completion | about a minute |
| G-MM2a-4 builds | as G-MM1-S6 | as G-MM1-S6 | runs to completion | about two minutes |

**MM2a read, 2026-10-26 (nobara): G-MM2a-1, 3 and 4 pass; G-MM2a-2 passes for CUDA and is checked for Metal in CI's darwin job.**
- G-MM2a-1: `TestDecisionCensus` passes without `-census.update`.
- G-MM2a-2: CUDA's registered report equals `DefaultBuildDemand` on 36 tracked configs and 6,912 queries, and a planted
  change is seen. Metal's test is the same; nobara vets it for darwin, and CI's darwin job now runs it, with its own zero-match
  guard. Both tests live in existing test files that import decoder (`cuda/resident_cap_test.go`,
  `metal/resident_memguard_test.go`), because SB13's ratchet on backend test files importing decoder only falls.
- G-MM2a-3: the decoder's 60 fit tests pass, with the two predicate calls read through the report. `cuda`, `metal` and `gpu`
  vet with their tags.
- G-MM2a-4: vet on three systems and with `realckpt`, the pinned staticcheck (untagged, `realckpt`, and `cuda` with
  testhooks), gofmt, the layout tests, the citation lint.
- The PR also regenerates `docs/ARCHITECTURE.md`'s import appendix. `TestArchitecture_importAppendix` is red on main since
  #301 added `cuda/internal/launchgeom`.

#### MM6's night profile, registered 2026-10-27 before the code (nobara-mm01; Francis's option B of 05:56)

**Francis's decision (05:56), replacing 1b after the replay:** option B. Each box gets a recorded "at night" profile, priced at
the 25th percentile of real nights. On the Mac the memory in use at night is the setup itself: the Claude app, background
sessions, their builds and tests. It persists. The replay's real refusals saw 2.9–7.7 GB available (median 6.6) against the
idle profile's 12.2.

**What is built.**
- **Every job start records the fit guard's available reading.** It goes in the `=== START` line and in the box's
  `memory.jsonl` under `~/goinfer-logs/night`, marked night or day by the job's `--not-before` (as registered; since the follow-up below, by the start's time alone). Today the runner logs only the
  load average at a start, so this record starts now.
- **The night profile is computed from those readings.** It is the 25th percentile (linear interpolation) of the night-job
  readings in the last 14 nights. A reading belongs to the night that began at or before it, so 02:00 counts with the evening
  before. The profile records its percentile, window, reading count, night count and dates. It is written to
  `~/goinfer-logs/night/profile-night.json`, a per-box file, not committed. A committed profile would change the census's
  machine zoo (`testdata/machines/*.json` is census input) and its golden every night, and each box's file is that box's own
  data.
- **Night jobs (`--not-before`) price against it; day jobs keep the apps-open profile.** With fewer than 5 readings or 3 nights
  in the window, a night job falls back to the apps-open profile with a warning.
- **`night.py profile`** prints the profile. `--backfill` seeds the history once from existing night logs: the fit guard's own
  "currently has X GB of memory available" lines, in logs whose job started in a night window. That sample is biased low,
  because those lines come from refusals and warnings, and it is marked `backfill` in the record.
- **nobara is the same,** from its own readings.

**The expected replay count, stated from the data before running.** The Mac's recorded night readings are the refusals'
printed figures: 2.9, 3.9, 4.6, 6.4, 6.4, 6.6, 6.9, 7.0, 7.4, 7.5 and 7.7 GB. Their 25th percentile is about 5.5 GB, a budget
of about 3.85 GB. The 8 counted jobs need 5.1, 12.6, 6.7, 5.5, 5.6, 5.1, 5.0 and 2.7 GB. So the night profile refuses rows
1–7 and fits row 8: **expected 7 of 8**. Row 7 (5.0 GB) fits only if the 25th percentile reaches 7.2 GB. Refined before any run, from
the window rule: the 14 nights ending 2026-10-27 start on 10-14, so the 10-08 and 10-12 readings fall outside it. The
in-window readings (2.9, 3.9, 4.6, 6.4, 6.6, 6.9, 7.0, 7.7) give about 4.4 GB, a 3.1 GB budget: still 7 of 8.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-NP-1 the percentile and window | a fake-history test: the profile's figure is the 25th percentile of exactly the night readings in the last 14 nights (older readings and day readings excluded), with linear interpolation checked against a hand computation; the profile records percentile, window, count, nights and dates | — | runs to completion | seconds |
| G-NP-2 the fallback | with fewer than 5 readings or 3 nights, a night job is priced on the apps-open profile and `add` prints a warning naming the count | — | runs to completion | seconds |
| G-NP-3 the record | a job start through the real runner writes the available reading to the START line and to `memory.jsonl`, with the night flag | — | runs to completion | seconds |
| G-NP-4 the replay (the Mac) | with the profile computed from the Mac's backfilled readings, the 8 counted jobs re-priced on it: **at least 7 refused** (the expectation above); the profile's figure and count are reported beside it | the profile falls back (too few readings), which is reported as such | every counted job | minutes |
| G-NP-5 builds | `python3 scripts/test_night.py`; `touched.py`; the citation lint | — | runs to completion | about a minute |

**Night-profile read, 2026-10-27 (nobara): G-NP-1, 2, 3 and 5 pass; G-NP-4, the Mac's replay, follows the PR**
(the commands are in the PR).
- G-NP-1: the profile equals a hand computation of the 25th percentile over exactly the in-window night readings, and records
  its percentile, window, readings and nights.
- G-NP-2: 6 readings over 2 nights fall back to `m1pro-apps-open` with a warning naming the counts.
- G-NP-3: a job through the real runner writes `available 4321 MB` to its START line and the reading to `memory.jsonl`.
- G-NP-5: `test_night.py` (74 tests at this read; 78 after the fixes before merge), `touched.py`, the citation lint.
- nobara: a dry run of `--backfill` on its night logs finds no fit-guard availability lines (this was the `START_RE` bug, fixed below, not an empty history). Its night
  profile starts from the START readings, and its night jobs are priced on `nobara-desktop`, with the warning, until it has 5
  readings over 3 nights.

**Fixed before merge, 2026-10-27 (nobara-mm01), from Gemini's review of #312 and the Mac's G-NP-4 (VOID: no profile).**
- `--backfill` found nothing: `START_RE` anchored `^` without `re.M`, and every real log has header lines before `=== START`.
  nobara's "no fit-guard lines" above was this bug, not an empty history. A test now backfills a log in the real shape.
- A box is priced only on its own readings: `night_profile` keeps readings whose `host` is the box priced. A night job
  queued for another box falls back to that box's day profile with the warning, and this box's `profile-night.json` is not
  overwritten.
- A 0 MB reading counts (it was dropped as false).
- One definition of night: a window from 22:00 to the deadline. A job is a night job when its `--not-before` start falls in
  it (a 14:00 start is a day job), and a start reading is a night reading only inside it, as backfill already required.
- The window is the 14 nights that have begun. Before, between noon and 22:00 it counted tonight, which had not begun, and
  so held 13. So the replay's "14 nights ending 2026-10-27, from 10-14" reads, before 22:00 on the 27th, as 10-13 to 10-26.
- `avail_mb` lived on `Server` while `run_job` is the base `Runner`'s, so `night.py start` would have failed at every job
  start; it is now on `Runner`, and it returns None when `hostmem` fails. A test runs the real one.
- Not changed: Gemini's 1000-against-1024 point (the fit guard prints GiB, so backfill's `* 1024` is right; the test's
  values are MB labels), and its import-time binding of `MEMORY_LOG` (the tests set `NIGHT_DIR` before importing `night`,
  so they never touch the live log).
- `test_night.py`: 78 tests pass. G-NP-4 is to be re-run on the Mac after the merge.

#### MM2b's gates (CUDA's 2 MiB allocation rounding in the plan), registered 2026-10-27 before the code (nobara-mm01; Francis's option A of 05:50)

**Francis's decision (05:50):** option A, "one set of numbers that is accurate". `Plan("cuda")` prices the driver's 2 MiB
rounding through CUDA's report. The CUDA build reads the slack from the plan instead of adding its own copy. `goinfer fit`,
MM6's queue-time check and the build all use that one figure.

**The design: one copy.**
- **The computation.** It is the CUDA build's own planner. `BuildResident` becomes `buildResident(m, planOnly)`. Plan-only
  stops right after `packedAllocSlack` over the planned buffers, before any device allocation, and validates experts instead
  of packing them, so it holds no host copy.
- **The report.** `RegisterPlanSlack` is a per-model report, memoized on the model. Through it, CUDA hands that figure to
  decoder's `Plan`, which prices it as a fixed term beside the dense weights: `Plan.AllocSlackBytes`, in `NeedBytes()` and in
  every fit check.
- **The build.** `resolveCtxCapFitSlack` stops adding the slack to the extra bytes it hands `Plan`; `Plan` already holds it.
- **No mirror.** Decoder keeps no mirror of CUDA's buffer layout, which would be a second copy. Its default slack is 0, which
  is the true figure on every tracked tiny fixture (measured 2026-10-26: 60 resident builds, all 0).
- **Out of scope.** The KV pools' rounding is priced by the build already (`kvRoundingBound`); it is not part of this step.

**Amended from the proposal, before the code.** Gate 1's "shape-only configs" assumed a path into `Plan` that decoder does not
have: `Plan` reads a loaded model's matrices, and decoder has no shape-only model. Building one would re-implement `Plan`'s
inputs, which is not today's code path. So the real-size cells use the real checkpoints on nobara's local disk, planned through
CUDA's report. The test that does it loads each model on the CUDA device, so it needs one. That is a tier-1 test in the `cuda` module, which skips as a capture run where
the models are absent.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM2b-1 the real-size diff | for Qwen2.5-Coder 0.5B and 1.5B and Qwen2.5-7B (int4, from `~/models`), D3's `Plan("cuda")` cells (the request grid and a free ladder with each cell's change points) are computed with the slack and with it forced to 0. The diff is reviewed group by group and listed in the PR body (groups changed, placements moved, every cell that moves to a decline). Then their golden is written. The decoder census passes without `-census.update`, all of its cells' slack being 0 | a model path under `/srv/models` or `/Volumes/` (CLAUDE.md), or a model absent | runs to completion | about a minute on nobara |
| G-MM2b-2 the device check | on nobara's RTX 2070 SUPER, for 0.5B and 1.5B: `Plan("cuda").AllocSlackBytes` equals the real build's `packedAllocSlack` to the byte, and plan dense plus slack equals the build's dense plus slack. `TestResidentDenseBytes_matchesCUDADevice` still holds. The CUDA KV-slot and margin tests (`TestCUDAKVSlots_*`, `TestKSH6C_marginAtTheReturnedContext`, `TestKSB8_widthChooser`) still pass | a timed run holds the card (the run takes the timing lock) | runs to completion | about two minutes |
| G-MM2b-3 one copy | the build no longer adds the slack beside the plan: a source test finds `packedAllocSlack` called only inside `buildResident`, and `resolveCtxCapFitSlack` takes no slack argument. The MM2a agreement test (`TestBuildDemand_matchesDecoderDefault`) gains the slack report: CUDA's equals decoder's default (0) on every tracked tiny fixture | — | runs to completion | seconds |
| G-MM2b-4 builds | as G-MM1-S6 | as G-MM1-S6 | runs to completion | about two minutes |

**MM2b read, 2026-10-27 (nobara): all four pass.**
- G-MM2b-1, the real-size diff (`docs/measurements/mm2b-rounding-2026-10-27/README.md`): rounding 138.5 MB (0.5B), 122.2 MB
  (1.5B) and 405.8 MB (7B). All 63 groups change. At nobara's two real readings, no cell moves to a decline, and only the 7B's
  unpinned 32,768 cells shrink (29,696 to 25,984 at 7.63 GB). 462 cells move to a decline, all inside the band one rounding
  wide below the old fit edge; the README lists them by group. The real-size golden (`cuda/testdata/mm2b_realsize.json.gz`)
  was written after that review. The decoder census passes without `-census.update`.
- G-MM2b-2: the plan's slack equals the real build's `packedAllocSlack` to the byte (145,276,928 and 128,122,880). (The
  read first said "as does dense plus slack": that check compared the plan's figures with themselves and was removed in the
  MM1 follow-up; the dense figure against the device is `TestResidentDenseBytes_matchesCUDADevice`'s.) `TestResidentDenseBytes_matchesCUDADevice`, the `TestCUDAKVSlots_` tests,
  `TestKSH6C_marginAtTheReturnedContext` and `TestKSB8_widthChooser` pass, and so do `TestBuildScratchAccounting` (six models)
  and its planted defect with `GOINFER_HEAVY_TESTS=1`.
- G-MM2b-3: `TestPlanSlack_oneCopy`; the agreement test's slack report is 0 against decoder's 0 on every tracked family. The
  new tests live in existing test files, because SB13's ratchet counts every test file of a package that reaches decoder.
- G-MM2b-4: vet on three systems and with the module tags, the pinned staticcheck (untagged, `realckpt`, cuda), gofmt,
  `touched.py`, the citation lint.
#### MM1 follow-up: the Gemini review's findings, registered 2026-10-27 (nobara-mm01)

Gemini reviewed #296, #300, #302, #307 and #311 (relayed by the coordinator, 2026-10-27 07:24). Each finding was checked
against origin/main (a22bd928d). The guard fixes below (the divide, the nil policy, the enum sync, the scan, the lookup order)
were written before this section; the gates were registered before the rest (the generators, the agreement tests, #311's
items, the census build) and before any of it ran as a gate.

**What changes, none of it a placement or a default.**
- `SolvePlacement` no longer divides by an unpriced expert size (`NExperts` with `ExpertBytesFull` under a byte each): such
  experts cost nothing, so every slot fits. Before, that load panicked; no caller hands it in today (the census holds).
- `ResolveKV`'s CUDA branch treats a nil `Decline` as no decline and a nil `Auto` as false (before: a panic).
- `decoder.Placement` and `mem.Placement` are held equal at compile time, all five values both ways.
- I6's scan: `RequestFraction` joins the thresholds, and a constant that aliases a different threshold counts as a use.
- `buildDemandFor` looks up the backend's report before building its query.
- `doc.go`'s contract says what is true: the guards proceed on unknowns; the planner declines on a 0 reading. I5 is stated
  as far as it holds (below).
- Tests only: the generators, the monotone test, `ResolveKV`'s branch and purity tests, the agreement tests, #311's tests.

**I5 does not hold for an MoE plan's placement, measured before any change.** `chooseCtx` grows the context before it prices
the experts, so one more KV step can leave less than top-k experts' room: on a generated CUDA load (8 experts, top-2), the
plan was expert-cached at context 4,480 and declined at 4,608, at 16,160 MiB. Changing that (reserving top-k experts in
`chooseCtx`) moves placements, so it is Francis's call and is not in this PR. The task-doc example in G-MM1b-2's amendment
(top-8, two slots) was misreported: the load was top-2.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-GF-1 the census | `TestDecisionCensus` passes without `-census.update`, every group byte for byte | the golden's class not found on an amd64 host | runs to completion | 16 s |
| G-GF-2 the planner tests | `TestSolvePlacement_pure`, `TestResolveKV_eachBranch`, `TestResolveKV_pure` and `TestSolvePlacement_monotone` pass. The monotone test walks free memory in 16 MiB steps to 64 GiB on generators that keep `DeviceLoad`'s invariants (tier per backend, a host copy only on Metal, Metal's linear KV), and asserts: a dense placement never worsens; a placed context never shrinks; slots never fall at the same context; the Metal context never shrinks (MoE included, as registered; withdrawn in the second follow-up below: an MoE Metal context does fall). At least a third of Metal draws reach a context past the floor. A planted defect (a refusal above 32 GiB free) turns it red | — | runs to completion | under a minute |
| G-GF-3 the guard fixes | an unpriced-expert load and a zero `CUDAPolicy` return without a panic; the scan reads every non-test file and counts a planted `.RequestFraction` read and a planted cross-alias | the scan reads fewer files than the tree holds | runs to completion | seconds |
| G-GF-4 the agreement tests | `TestBuildDemand_matchesDecoderDefault` (cuda, metal) has no duplicate backend; it asserts that a nonzero `HostPackBytes`, `HostBuildPeak` and (cuda) `PinnedExpertBytes` each occur in the corpus, or names the one that does not; a planted change per field turns it red; and decoder's registry returns the backend's own function | — | runs to completion | seconds |
| G-GF-5 #311's items | the tautological dense-plus-slack check is removed with its claim; the archive guard resolves symlinks; `TestPlanSlack_oneCopy` checks the call site's argument; `planSlackOffForTest` is atomic; a plan-only build that stops early is not memoized | — | runs to completion | seconds; the device tests run once on nobara under the timing lock |
| G-GF-6 builds | as G-MM1-S6 | as G-MM1-S6 | runs to completion | about two minutes |

**Read, 2026-10-27 (nobara): all six pass.**
- G-GF-1: `TestDecisionCensus` and its second-process twin pass without `-census.update`. The serial probe check now covers
  every config, not the first eight, and holds.
- G-GF-2: the planner tests pass. 919 of the 1,500 Metal draws reach a context past the floor, and the planted refusal is
  found.
- G-GF-3: the zero-policy rows of `TestResolveKV_eachBranch` return; the scan reads 734 files and counts the planted
  cross-alias and `.RequestFraction` read (3). The unpriced-expert half was not tested: no test drew such a load (the
  generators price every expert at a byte or more), so this line first claimed a check that did not exist. The second
  follow-up adds it.
- G-GF-4: 36 configs and 6,912 queries, now with no duplicate backend. Every field's planted change is seen, so a nonzero
  `HostPackBytes`, `HostBuildPeak` and `PinnedExpertBytes` each occur. `TestBuildDemand_registered` passes on cuda. Its
  metal twin is vetted for darwin but runs only on a Mac with `goinfer_testhooks`, because CI's metal job runs untagged.
- G-GF-5: `TestPlanSlack_oneCopy` goes red on a planted `+allocSlack` at the call site. On the RTX 2070 SUPER (timing lock
  held) these pass: `TestPlanSlack_matchesTheBuild` (145,276,928 and 128,122,880 again), `TestPlanSlack_realSizeDiff`,
  `TestResidentDenseBytes_matchesCUDADevice` and the `TestCUDAKVSlots_` tests (log: `~/goinfer-logs/mm1/` on nobara,
  gf-cuda-device).
- G-GF-6: vet untagged and tagged (testhooks; cuda with testhooks; darwin metal both ways); the pinned staticcheck with its
  canary red (U1000) on the same variants, clean; gofmt.

#### MM1 follow-up 2: Gemini's review of #316, registered 2026-10-27 (nobara-mm01)

Relayed by the coordinator at 08:47. The code below was written before this section; the gates were registered before they
ran.

**Withdrawn from #316: "an MoE Metal context never falls".** #316 listed Gemini's Metal case as not holding, on a probe that
stepped 16 MiB. The window it lives in is under one 8-position KV step (a few MiB), so the probe stepped over it. At a 64 KiB
step a constructed Metal MoE load (8 experts, top-2, 512 KiB a KV step) falls from 4,104 to the 4,096 floor 1,535 times between
5.6 and 9 GiB. The cause is the MoE planner's (context before experts), so it joins that decision
(the decision note for Francis); the monotone test now checks the Metal context for dense loads only.

**What changes.**
- An expert cache whose experts could not be priced (`ExpertBytesFull` under a byte each) declines with a reason. #316
  sized every slot on 0 bytes there; before #316 that line panicked, so no caller depended on either. Not changed: an unpriced
  MoE load whose dense weights fit goes resident with 0 expert bytes, as `Model.Plan` did before MM1. Changing that moves a
  placement, so it is in the decision note.
- `PlanWithoutSlackForTest` passes "no slack" to one call (`Model.plan`'s argument) instead of setting a global, so a
  concurrent `Plan` never sees it and two overlapping calls cannot restore it early.
- The monotone test also holds an MoE plan's placement at an unchanged context; Metal's agreement comment says what it plants.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-GF2-1 the census | `TestDecisionCensus` passes without `-census.update` | the golden's class not found on an amd64 host | runs to completion | 16 s |
| G-GF2-2 the planner | `go test ./internal/mem/` passes, with `TestSolvePlacement_unpricedExperts` (0 and 7 bytes over 8 experts decline naming the unpriced experts); reverting the guard turns it red | — | runs to completion | under a minute |
| G-GF2-3 the slack hook | the rounding diff (`TestPlanSlack_realSizeDiff`) holds its golden on nobara's card; vet with the testhooks and cuda tags | a timed run holds the card | runs to completion | about a minute |
| G-GF2-4 builds | as G-MM1-S6 | as G-MM1-S6 | runs to completion | about two minutes |

**Read, 2026-10-27 (nobara): all four pass.** G-GF2-1: the census and its second-process twin pass. G-GF2-2: `internal/mem`
passes; the unpriced-expert test goes red with the guard reverted. G-GF2-3: `TestPlanSlack_realSizeDiff` holds its golden and
`TestPlanSlack_matchesTheBuild` passes on the RTX 2070 SUPER (exclusive timing lock). G-GF2-4: vet (testhooks, cuda, darwin
metal), the pinned staticcheck with its canary red, gofmt, `touched.py`, the parity refresh, the citation lint.

#### MM1's MoE reserve (option A), registered 2026-10-27 before the code (nobara-mm01; Francis's option A of 09:09)

**Francis's decision (09:09):** "option A for the MoE planner. Reserve the top-k experts' bytes before growing the context in
chooseCtx, so more memory never gives a worse MoE placement. Gate it as MM2b was." The question and the measured census effect
are in the decision note (`MM1 follow-up 2` above names the cause).

**The change.** `SolvePlacement`'s `chooseCtx` prices an MoE's top-k experts (top-k × the per-expert bytes) beside the dense
weights, the slack and the host copy before it grows the context, on both its paths (the per-position rate and CUDA's block
search). The expert-cache and resident fits after it are unchanged. A load whose experts could not be priced reserves nothing
(it declines at the expert cache, as #320 made it).

**Expected before the run.** The census moves in D3 only, on MoE fixtures: 984 of 29,637 groups, measured on a scratch copy at
e934b8ce plus #320 (10,336 readings declined → expert-cached, 5,028 weight-paged → expert-cached, 80 declined → resident, 24
resident → expert-cached). D1, D2 and D4–D7 do not move.

**Pre-registered contingency.** Reserving top-k alone may leave a fall between resident and expert-cached, where every
expert's extra bytes are under one KV step, and Metal's context fit needs a resident answer. If the monotone test finds either,
the planner tries the resident placement first with every expert (and Metal's host copy of them) reserved, and the expert cache
second with top-k reserved, each choosing its own context. That is the same rule (reserve what the placement needs before
the context grows) one placement at a time. The census diff is then re-measured and listed the same way; the context invariant
holds within a placement (G-MM1b-2 as amended), not across an upgrade from expert-cached to resident.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MA-1 the census diff | the census is computed with and without the change; the diff is listed in the PR body group by group (each changed group, its backend and family, and every reading whose placement moves, by direction), before `-census.update`. Any group outside D3, or any non-MoE group, that moves is a failure of the change, not a diff to bless | the golden's class not found on an amd64 host | runs to completion | about a minute |
| G-MA-2 monotone, MoE included | `TestSolvePlacement_monotone` asserts, for every load, MoE included: the placement never moves down the order; within a placement the context never shrinks, nor the slots at the same context; the Metal context never shrinks. The walk adds a fine ladder (64 KiB steps across each draw's sawtooth window) to the 16 MiB one. A planted defect (the reserve removed) turns it red, and so does the planted refusal | — | runs to completion | under a minute |
| G-MA-3 the probe | the constructed Metal MoE load of the follow-up (1,535 falls at 64 KiB steps) has none | — | runs to completion | seconds |
| G-MA-4 real sizes and the device | `TestPlanSlack_realSizeDiff` holds its golden (dense models: no move expected), and the CUDA KV-slot tests pass on nobara's card | a timed run holds the card | runs to completion | about a minute |
| G-MA-5 builds | as G-MM1-S6 | as G-MM1-S6 | runs to completion | about two minutes |

**Read, 2026-10-27 (nobara): G-MA-2 to G-MA-5 pass. G-MA-1 fails its registered rule's letter, and the golden is updated with
that stated; Francis decides whether it stands.**
- **The contingency was taken.** The top-k reserve alone passed the generated walk, but on a constructed CUDA load whose experts
  are under one KV block it still fell from resident to expert-cached 449 times at 64 KiB steps. So the planner tries resident
  first (every expert reserved), then the expert cache (top-k reserved), as registered.
- **G-MA-1, the census diff** (`docs/measurements/mm1-moe-reserve-2026-10-27/`, groups listed one by one in `groups.tsv.gz`):
  - 3,534 of 29,637 groups change; 1,068 of them move a placement or class (D3 984, D4 84).
  - By reading: 9,676 declined → expert-cached, 5,028 weight-paged → expert-cached, 732 declined → resident, 92 expert-cached →
    resident, and 1,232 Metal fits leave the floor. **None moves down.** Every moved group is an MoE fixture.
  - **The registered rule called any move outside D3 a failure, and D4 moves.** Those are Metal's context fit, which the
    contingency exists to fix (it needs a resident answer). The expectation and the rule were written for the top-k reserve
    alone and were not amended when the contingency was registered. The golden is updated because the diff was listed and every
    move is upward on an MoE fixture, but the PR is not set to merge by itself: that is Francis's call.
  - The registered expectation (984 groups, 24 resident → expert-cached) was the top-k-only measurement; the contingency removed
    the 24 downward readings.
- G-MA-2: `TestSolvePlacement_monotone` asserts I5 for every load, MoE included, with 64 KiB windows at each draw's first change
  points. It finds the planner without the reserve (input 364: resident, then declined) and the planted refusal.
- G-MA-3: `TestSolvePlacement_moeSawtooth`. The CUDA load has no fall (the planner without the reserve falls), and the Metal
  load climbs from the floor without a fall.
- G-MA-4: `TestPlanSlack_realSizeDiff` holds its golden (dense models). The CUDA KV-slot, margin, dense-bytes and rounding
  tests pass on the RTX 2070 SUPER (exclusive timing lock; log `~/goinfer-logs/mm1/` on nobara, ma-cuda-device). Real MoE
  checkpoints, which the census does not hold: `goinfer fit` on CUDA said the 26B (gguf and .giw) and Qwen1.5-MoE declined;
  it now says expert-cached (8/128 at context 11,264; 4/60 at 15,104), as the CUDA build runs them.
- G-MA-5: listed in the PR.

**G-MA-1 re-registered, 2026-10-27 (Francis's option C of 10:36), before the re-run.** The rule above is replaced by this
one. The census is computed with and without the change: "without" is this branch with `internal/mem/placement.go` taken from
main. The diff is graded as before: placements and classes at every reading either run records, contexts only at readings
both runs record.
- **Pass:**
  - D1, D2, D5, D6 and D7 are byte-identical;
  - every changed group is in D3 or D4 and on an MoE fixture (one whose D3 groups price experts above 0 in the base census);
  - **no move is downward anywhere**: no placement moves down the order (decline < weight-paged < expert-cached < resident),
    no Metal fit falls to the floor from above it, and no context is smaller at a reading both runs record within the same
    placement or class. A decline's context is excluded, because it only echoes the attempt.
- **A D4 move is allowed only when it is upward on an MoE fixture.** Any downward move, or any move on a non-MoE fixture, is
  still a failure.
- **Expected, from the mechanism (not from the earlier run):**
  - D1, D2, D5–D7: 0 groups change. They do not call `SolvePlacement`'s MoE path.
  - Non-MoE groups: 0. The reserve is 0 without experts.
  - Downward: 0. The reserve only moves the context choice toward where the placement's experts fit. Where the old planner
    placed an MoE at context c, the experts already fit at c, so the new context at the same placement is at least c, and a
    placement it reached it still reaches.
  - D3: at most the 4,592 MoE groups, all upward.
  - D4: at most the 168 Metal-fit groups that can leave the floor in the base census (an MoE fixture, unpaged, known memory,
    a context above the floor at some reading; 14 fixtures, `lower` 0 and 8192, 84 each). A group with `lower=2048`, paged
    experts or unknown memory never leaves the floor, so it cannot move.
- **Then:** the device tests (G-MA-4) re-run on nobara's card. On a pass, auto-merge is set on #321. On a failure of this rule:
  stop, DECISION NEEDED.

**Re-run read, 2026-10-27 (nobara), against the re-registered rule: PASS.**
- The census was dumped without the change (this branch, `placement.go` from main at efa85df15's second parent) and with it,
  and graded by `docs/measurements/mm1-moe-reserve-2026-10-27/grade.py`: **0 failures.**
- D1, D2 and D5–D7 are byte-identical. Every changed group is on an MoE fixture: 3,366 D3 and 168 D4.
- The D4 changes are exactly the 168 groups the mechanism allowed: 14 fixtures, `lower` 0 and 8192, 84 groups each.
- 960 D3 groups and 84 D4 groups move a placement or class. By reading: 9,676 declined → expert-cached, 5,028 weight-paged →
  expert-cached, 732 declined → resident, 92 expert-cached → resident, and 1,232 Metal fits leave the floor.
- No placement moves down, no Metal fit falls to the floor, and no context is smaller within a placement at a reading both runs
  record.
- The committed golden holds (`TestDecisionCensus` passes) and `go test ./internal/mem/` passes.
- G-MA-4 re-run on the RTX 2070 SUPER under the exclusive timing lock: 15 of 15 pass (the rounding diff's golden, the
  KV-slot, margin, dense-bytes and agreement tests; log `~/goinfer-logs/mm1/` on nobara, ma-cuda-device-rerun).

#### MM6 night profile and the unpriced MoE: Gemini on #312 and #320, registered 2026-10-27 before the code (nobara-mm01)

Relayed by the coordinator at 10:02. #320's two test findings (an MoE decline unchecked at the same context; MoE Metal draws
counted as searched) are on the monotone test that #321 rewrites: it asserts every load, MoE included, so they are fixed there.

**What changes.**
- **The night profile is weighted by night, not by reading:** each night's lowest reading, then the 25th percentile over the
  nights in the window (still 14 nights, at least 3 of them, and at least 5 readings). One night with many short jobs no longer
  outweighs the others. **Amended before merge (coordinator, 10:27, from Gemini on #322):** per night alone can raise the
  figure (a night with many low readings becomes one), so the figure is the lower of the per-night and the per-reading 25th
  percentile. The change can then only lower the figure or keep it, so the queue-time check can refuse more night jobs, never
  fewer; that is listed for Francis. The registration first claimed that of per night alone, which was wrong.
- **A reading is a night reading by its time alone** (inside 22:00 to the deadline), in the runner as in backfill, because a
  backfilled log does not record its job's `--not-before`. The profile describes the box at night, whichever job read it.
  Pricing still goes by `--not-before`.
- **Backfill skips a job that already has a reading** of any source at the same start, so a refused night job is not counted
  twice.
- **A failed reading is logged** in the runner's log, instead of only `available ?` in the job's.
- **An MoE whose experts could not be priced is declined on a device** (paged on the CPU), resident included, before any
  context is chosen. Before, with no slot count asked, it went resident with 0 expert bytes. That changes what the planner
  admits; the census diff is listed in the PR.
- Tests: a job run through the real runner at 23:00 records a night reading; the queue-time test seeds past nights only;
  the hostmem test asserts the reading is a number where hostmem works.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-NP2-1 per night | a fake history with one night of ten readings at 1 GB and four nights of one reading at 6 GB: the profile is the 25th percentile of the five nights' minimums, checked against a hand computation | — | runs to completion | seconds |
| G-NP2-2 the record | through the real runner at 23:00 a job records `night: true`; at 20:00 `night: false`; a failed reading logs a line naming it; backfill adds nothing for a job that has a start reading | — | runs to completion | seconds |
| G-NP2-3 the unpriced MoE | on loads that keep `DeviceLoad`'s invariants (CUDA with no host copy; Metal whose host copy is its experts'), an unpriced MoE declines on CUDA and Metal and pages on the CPU, at slot counts 0 and 2; `go test ./internal/mem/` passes | — | runs to completion | seconds |
| G-NP2-4 the census | `TestDecisionCensus`: the diff with and without the change is listed in the PR body group by group before `-census.update`. Expected: no group moves, since no fixture's experts are unpriced. Any move is listed for Francis | the golden's class not found on an amd64 host | runs to completion | about a minute |
| G-NP2-5 builds | `test_night.py`, `touched.py`, vet, the pinned staticcheck, the citation lint | — | runs to completion | about two minutes |

**Read, 2026-10-27 (nobara): all five pass.**
- G-NP2-1, as amended: `test_a_busy_night_counts_once` (per reading is the lower figure there, and is taken) and
  `test_per_night_weighting_never_raises_the_figure` (per night is the lower figure, and is taken).
- G-NP2-2: the real runner records `night: false` at 20:00 and `night: true` at 23:00. A failed reading logs
  "the available reading failed" and records nothing. Backfill adds nothing for a job with a start reading. The hostmem test
  now requires a number: it failed in a shell without `go` on its PATH. nobara's `goinfer-night.service` has
  `/usr/local/go/bin` on its PATH, so its runner reads.
- G-NP2-3: `TestSolvePlacement_unpricedExperts` passes on CUDA, Metal and the CPU at 0 and 2 slots. With the guard
  disabled, it goes red (resident with 0 expert bytes).
- G-NP2-4: the census dump is byte-identical with and without the change. No fixture's experts are unpriced, so nothing moves.
- G-NP2-5: `test_night.py` (81 tests), `go test ./internal/mem/`; the rest is in the PR.

#### MM3a's gates (the ledger, and leases in CUDA's allocator), registered 2026-10-27 before the code (nobara-mm01; Francis's go of 11:08)

**Francis's go (11:08):** "a ledger and leases in each backend's allocator, so every allocation is a lease naming its plan
line. CUDA first (cuMemAlloc's 2 MiB rounding priced, now one figure after MM2b), then WebGPU, then Metal. Gates before code:
I1 … and I2 … per backend at tier 0; I3 … with one tiny model per backend, seconds. No change to what any guard admits."

**The design.**
- **`internal/mem` gets the ledger**, its one stateful type. It records reservations; it reads nothing and decides nothing.
  - `Ledger.Reserve(tier, line, requested, charged)` returns a `*Lease`, and `Lease.Release` is idempotent.
  - `Usage()` reports, per tier and per line, the open leases and their requested and charged bytes, plus the peak charged.
    `Unplanned` is the charged bytes of open leases whose line is empty or not a known plan line.
  - The plan lines name the `DevicePlan` terms: `dense`, `experts`, `kv`, `state` (recurrent and conv state), `slack`
    (rounding the plan prices), `companion` (a drafter's or tower's bytes, `ExtraBytes`), `scratch` (per-pass activations, the
    margin's share), `staging` (the build's host staging), `pinned-experts`, and `test` (a test's own buffers).
- **CUDA's `gpudev.Device` keeps a `Ledger`.**
  - Every allocation leases its bytes: requested, and charged by the driver's rule (a buffer of 2 MiB or more rounds up to
    2 MiB). That rule is one function, which `allocRoundSlack` (the plan's slack) also calls.
  - `ReleaseBuf`, `ReleaseAll` and `ReleaseObjects` release the leases.
  - Pinned host buffers lease on `HostAnon` (the open question in §6.7, "CUDA pinned host memory as its own tier", stays open;
    the line names them).
- **Every production allocation names its line.**
  - `gpudev` gains lined constructors. The unlined ones remain for tests and lease on the line `gpudev.UnlinedLine`: "" (so
    unplanned) unless a test package sets it to `test`.
  - The resident's helpers (`af`, `akv`, `ai`, `au32`) take the line from the build phase in progress (`r.onLine`). The build
    and request paths run on the resident's executor goroutine, so the phase needs no lock.
- **Nothing reads the ledger to decide.** No guard, plan or default changes.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM3a-1 I1, static | a source scan over every non-test file of the `cuda` module (`internal/qwencuda` and `internal/visioncuda` included) finds no call to an unlined `gpudev` constructor (`NewBufferLenOf`, `NewBufferOf`, `NewBufferLen`, `NewBufferBytes`, `MustBuf`, `NewHostBuffer`, `NewMappedHostBuffer`, `RegisterMappedHostBuffer`); it reads every file and goes red on a planted call | it reads fewer files than the module holds | runs to completion | seconds |
| G-MM3a-2 I1, dynamic | the `cuda` package's `TestMain` installs a hook on `ReleaseObjects` and fails the run if any released device's `Usage().Unplanned` was above 0; run with the device suite once on nobara (tags `cuda goinfer_testhooks`, the tests MM2b and MM1 ran: KV slots, dense bytes, plan slack, build scratch, KS-B8, KS-H6C) and `-v`. A planted allocation on the empty line is reported | a timed run holds the card | one run | about two minutes on the card |
| G-MM3a-3 I2 | tier 0: `internal/mem`'s ledger tests (reserve and release in any order, idempotent release, peak, by line and tier, concurrent use under `-race`); on the device: every released device's ledger has 0 open leases and 0 charged bytes after `ReleaseObjects`, and `ReleaseBuf` closes its one lease (the same hook as G-MM3a-2) | as G-MM3a-2 | as G-MM3a-2 | seconds, and the same run |
| G-MM3a-4 I3 | on the RTX 2070 SUPER under the exclusive timing lock, for llama-tiny and Qwen2.5-Coder-0.5B q4_k_m (`~/models`): the device's free VRAM just after the device is created, against free VRAM after the build and one 8-token generate, gives the measured bytes. **Pass: charged ≤ measured ≤ charged + 64 MiB.** Small buffers are charged at their size, while the driver hands them out of shared 2 MiB pages; the 64 MiB also covers the modules the build loads and its streams. The record reports both numbers and the gap | another process's allocation on the card changes during the cell (the desktop's context is constant and noted) | one run per model | under a minute |
| G-MM3a-5 one rounding rule | `TestPlanSlack_oneCopy` still passes; `allocRoundSlack` calls `gpudev`'s rule (a scan finds the 2 MiB constant once); `TestPlanSlack_matchesTheBuild` still matches to the byte | as G-MM3a-2 | runs to completion | seconds |
| G-MM3a-6 no guard change | `TestDecisionCensus` passes unchanged; the CUDA fit, KV-slot and margin tests pass unchanged; a scan finds no non-test read of `Usage()` outside tests and the ledger's own package | — | runs to completion | about a minute |
| G-MM3a-7 builds | vet with the module tags, the pinned staticcheck with its canary, gofmt, `touched.py`, the citation lint | — | runs to completion | about two minutes |

If I3 misses its band, the finding is reported first and the band is not widened.

**Two changes to the design, made before any gate ran.**
- `gpudev` may not import goinfer (SB13's rule "gpudev-standalone"). So `gpudev` defines a `Leaser` interface and takes plan
  lines as strings, and the cuda package attaches a `mem.Ledger` adapter when it creates a device (`createDevice`). The plan
  lines are untyped constants, so they pass as strings unconverted.
- The I1 hook reads the ledger through that adapter.

**MM3a read, 2026-10-27 (nobara, RTX 2070 SUPER, driver 595.91.07, kwin_wayland's 16 MiB context present): all seven pass.**
- **G-MM3a-1:** `TestLedger_everyAllocationNamesALine` reads all 58 non-test files outside `gpudev` and finds no unlined call. It
  counts the planted file's 2.
- **G-MM3a-2 and G-MM3a-3:** in the device run (tags `cuda goinfer_testhooks`, `GOINFER_HEAVY_TESTS=1`, exclusive timing lock,
  `-v`), 96 devices are released, none with an unplanned byte and none with an open lease. 27 tests pass; 7 skip on their own
  gates (KS-B8's timing and KS-H6C's breakdown and margin, which need their variables). `TestLedger_unplannedIsReported` sees
  the planted 3 MiB, charged as 4, and the planted open lease. `internal/mem`'s ledger tests pass under `-race`.
  - `touched.py`'s wider selection first found 4 devices with unplanned bytes: three pack-stream tests and
    `TestWindowedKV_view`, which drive the build's helpers outside a build. A stack per unplanned lease showed every one came
    from those tests and none from production. They now name the test line, and the selection passes.
- **G-MM3a-4, I3 (inside the band):**

  | model | measured | charged | gap |
  |---|---:|---:|---:|
  | llama-tiny | 8,388,608 | 4,479,240 | 3.9 MB |
  | Qwen2.5-Coder-0.5B | 991,952,896 | 988,562,064 | 3.4 MB |

  By line, the 0.5B is dense 483.6 MB, kv 503.3 MB and scratch 2.3 MB. Both ledgers hold no lease after `Close`.
- **G-MM3a-5:** `allocRoundSlack` is `gpu.ChargedBytes(n) - n`, and the 2 MiB constant is defined once
  (`gpu.AllocQuantumBytes`). `TestPlanSlack_oneCopy` and `TestPlanSlack_matchesTheBuild` pass (145,276,928 and 128,122,880
  again).
- **G-MM3a-6:** `TestDecisionCensus` passes unchanged. The only non-test reader of the ledger is the test seam's call site.
- **G-MM3a-7:** vet (cuda with and without testhooks, gpu, darwin metal); the pinned staticcheck with its canary red, clean;
  gofmt; the layout tests, with `imports.json` and the ARCHITECTURE appendix regenerated for the new `→ internal/mem`
  arrows; `touched.py`; the citation lint.
- Logs: `~/goinfer-logs/mm1/` on nobara (mm3a-device2).

**Fixed before merge, 2026-10-27 (coordinator, 12:01; CI's lint and Gemini's review of #328).**
- **CI lint:** the new source-scan test was not in `scripts/touched_scans.py`. It is listed now.
- **Leases where none were made (Gemini 1, real).** A device created other than through the cuda package's wrapper (the
  `qwencuda` and `visioncuda` towers call `gpu.CreateSystemDefaultDevice` directly) had no leaser, and the release check skipped
  it. `gpudev.NewLeaser` is now set by the cuda package at init, so every device a process with cuda linked creates is leased.
  A released device with no ledger is now a break. Only a test (`cuda/s4siglip`) imports those two towers today.
- **I2 after release was nearly tautological (Gemini 2, real).** `ReleaseObjects` frees and releases everything it tracked
  before the check, so only an untracked pin could show there. I2 is now also held within a model's life: the charged bytes
  after a second and a third identical request equal those after the first, which a buffer leaked per request would grow.
  Passes on llama-tiny and the 0.5B.
- **The plants did not exercise the suite's check (Gemini 3, real).** `TestLedger_unplannedIsReported` now runs the suite's
  own `ledgerBreak` on an unplanned allocation, an open lease, a clean device and a device with no ledger. The static scan
  now reads `gpudev` too (68 files), counting any unlined call outside the unlined constructors' own chain. `UnlinedLine =
  test` in the tests stays: production's unlined calls are excluded statically, which is what the static gate is for.
- **`onLine` off the executor (Gemini 4, real).** `prefillCore`, `SetAdapter`, `AttachDrafter` and the drafter's four
  methods set the line on the caller's goroutine before their `r.do`. They now pass it to `r.doOn`, which sets it inside the
  job. Every remaining `onLine` is in a function that allocates directly, so it already runs on the executor (allocation needs
  the context current there).
- **llama-tiny was skipped without the 0.5B (Gemini 5, real).** It runs alone now.
- **Gemini 6:**
  - A `NewBufferLenOn` twin is added.
  - `slack` and `staging` stay as lines: they name plan terms, though CUDA leases neither. Its rounding is in each buffer's
    charged bytes, and its pack staging is Go memory the allocator never sees.
  - `ByLine` summing tiers is by design; `ByTier` splits them.
- **Re-run:** the device suite under the exclusive lock (34 pass, 102 devices released, no break; I3 unchanged at 3.9 MB and
  3.4 MB), `touched.py`, the pinned staticcheck, `test_touched.py` and the citation lint. Log: mm3a-device3.

#### MM3b's gates (leases in WebGPU's allocator), registered 2026-10-27 before the code (nobara-mm01; Francis's go of 11:08)

**What is different from CUDA.**
- WebGPU has no allocation chokepoint. The kernels package calls `wgpu.Device.(Try)CreateBuffer(Init)` directly at 133 sites,
  and the gpu package at 3 more. Buffers are freed by `Release` or `Destroy` at 168 sites, mixed in with releases of
  pipelines, bind groups and encoders.
- The existing `liveBufferBytes` counts only the caller-owned wrapper types, process-wide.

**The design.**
- **Every creation goes through a lined Context wrapper** (`c.createBuffer(line, …)` and its siblings, exported as
  `kernels.CreateBufferOn` for the gpu package). It leases the buffer's `GetSize()` bytes on the Context's `mem.Ledger`
  (tier VRAM) and records the lease against the buffer.
- **Every buffer release goes through `releaseBuf` (or `destroyBuf`)**, which closes the lease.
- **The rewrite is mechanical.** A typed scan (go/types) found the release sites, so only `*wgpu.Buffer` releases are
  rewritten. Pipelines and bind groups are untouched.
- **Lines by enclosing function:**
  - `kv`: the KV caches and MLA's latent store;
  - `dense`: the weight uploads;
  - `experts`: the stacked experts;
  - `companion`: vision, towers and LoRA;
  - `state`: Mamba's state;
  - `test`: the test hooks;
  - `scratch`: the per-call ops and the runners' buffers.
- **`liveBufferBytes` stays as it is**, a separate gauge.
- **Nothing reads the ledger to decide.**

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM3b-1 I1, static | a typed scan over the gpu module's non-test code (`gpu` and `gpu/internal/kernels`, tags `gpu goinfer_testhooks`) finds no `CreateBuffer`, `CreateBufferInit`, `TryCreateBuffer` or `TryCreateBufferInit` call outside the lined wrappers, and no `Release` or `Destroy` of a `*wgpu.Buffer` outside `releaseBuf` and `destroyBuf`. It goes red on a planted call of each kind | it reads fewer files than the packages hold | runs to completion | seconds |
| G-MM3b-2 I1 and I2, dynamic | a release check on every Context the gpu module's tests close: no unplanned byte ever, and no open lease after `Close`. Run once on nobara's card (Vulkan; tags `gpu goinfer_testhooks`; the KV-slot, residency, leak and parity tests), with `-v` | a timed run holds the card | one run | minutes |
| G-MM3b-3 I2 per request | on llama-tiny, the charged bytes after a second and a third identical request equal those after the first | as G-MM3b-2 | one run | seconds |
| G-MM3b-4 I3 | after the build and one request on llama-tiny (resident on WebGPU), the ledger's open VRAM leases equal wgpu-native's registry count of buffers the user holds (`GenerateReport().Hub.Buffers.NumKeptFromUser`), exactly. The bytes are reported beside the process's VRAM from `nvidia-smi`; that comparison is context only, not graded, because the Vulkan allocator sub-allocates buffers from large blocks | the registry's count is shown to include buffers the runtime holds for itself: reported, not re-banded | one run | seconds |
| G-MM3b-5 no guard change | `TestDecisionCensus` passes unchanged; the WebGPU KV-slot and fit tests pass unchanged; nothing outside tests reads the ledger | — | runs to completion | about a minute |
| G-MM3b-6 builds | vet with the gpu tags, the pinned staticcheck with its canary, gofmt, `touched.py`, the layout tests (`imports.json` updated for any new arrow), the citation lint | — | runs to completion | about two minutes |

**Found while building, before the gates' read.**
- **Freeing by method value.** The runners keep buffer releases as method values (`r.keep = append(r.keep, b.Release)`), and
  so do 16 other sites, which free a buffer later without a call at the site. The typed scan counts a method value as a free,
  and those sites take `releaser(b)`, which binds `b` when it is evaluated, as the method value did.
- **The ledger must not keep a buffer alive.** Some buffers are never released explicitly; the binding's finalizer frees them
  when the GC collects them. A first version keyed leases by the buffer itself. That kept those buffers, and the Vulkan devices
  they pin, alive, and the kernels suite ran out of devices ("Not enough memory left"; main passes all 110). Leases are now
  keyed by a weak pointer, and a `runtime.AddCleanup` closes a collected buffer's lease.
- **The check must not force collection inside `Close`.** A first version of the check ran the GC inside `Close` and so started
  finalizer-driven device releases beside the caller's. `TestContextClose_lateWrapperClose` deadlocked. The hooks now keep each
  closed Context's ledger, and `LedgerBreaks` judges them once after the run, letting the GC collect first.

**MM3b read, 2026-10-27 (nobara, RTX 2070 SUPER over Vulkan; exclusive timing lock): all six pass.**
- **G-MM3b-1:** `TestLedger_everyBufferIsLeased` type-checks the 61 non-test files of `gpu` and `gpu/internal/kernels` and
  finds no create, release or destroy outside the wrappers. The plant gives 4: a create, a release, a destroy and a method
  value; a pipeline's `Release` is not counted.
- **G-MM3b-2:**
  - The kernels suite passes all 111 tests (main's 110 and the scan) in 227 s, against main's 216 s. 703 Contexts closed, and
    none broke the ledger.
  - The gpu package's selection (KV slots, residency, parity, ledger) passes: 24 pass, and 30 skip on their own gates (heavy
    checkpoints, opt-in variables, missing fixtures, darwin-only). 81 Contexts closed with no break.
- **G-MM3b-3:** the charged bytes after a second and a third identical request equal the first.
- **G-MM3b-4, I3 (exact):** 196 open leases against wgpu-native's 196 held buffers. The ledger charges 246,568 bytes (dense
  102,704, kv 131,072, scratch 12,792). The process's VRAM from `nvidia-smi` is 202–212 MiB, for context only.
- **G-MM3b-5:** `TestDecisionCensus` passes unchanged. Nothing outside tests reads the ledger, except `BuffersHeld`, a
  diagnostic.
- **G-MM3b-6:**
  - vet passes with the gpu and testhooks tags.
  - The pinned staticcheck is clean with those tags (CI's variant), and its canary goes red. It flagged two wrappers nothing
    called, which are removed.
  - gofmt, `touched.py`, `test_touched.py` (the scan listed in `touched_scans.py`) and the citation lint pass.
  - The layout tests pass, with `imports.json` and the ARCHITECTURE appendix regenerated for the `→ internal/mem` arrows.
- Logs: `~/goinfer-logs/mm1/` on nobara (mm3b-kernels3, mm3b-gpu2, and main's baseline mm3b-baseline-main).

#### MM3b follow-up: Gemini's review of #334, registered 2026-10-27 before the code (nobara-mm01; coordinator, 14:18)

**Findings checked against main.**
- **Lines (real).**
  - `UploadF32` charged `dense`, but its callers upload per-call activations (scratch) and vision inputs (companion); only the
    resident's `up32` uploads weights.
  - The `kvslot-headroom` probe is created and freed at once, but leased on `kv`.
  - `MLALatentStore` is a synchronous op that frees its buffers before returning, but leased them on `kv`.
- **The dynamic I2 check can pass vacuously (real).** A collection cleanup closes any lease whose buffer the GC frees, and the
  check runs after the tests, when every test's locals are unreachable. A buffer that production code forgets to release is
  then counted as closed.
- **The typed scan ignores `CgoFiles` (real in principle).** No kernels file has `import "C"` today.
- **`destroyBuf` and `ReleaseDestroyBuffer` are dead (real).** Nothing destroys a buffer, and a destroy before a release would
  break I3's count. The line lists in the PR and the doc omit `state` and `test`.

**What changes.**
- `UploadF32` leases `scratch`. A lined `UploadF32On` is used by the vision paths (`companion`) and the resident's weights
  (`dense`). The probe and `MLALatentStore` lease `scratch`.
- The collection cleanup counts what it closes. A lease closed by the GC, rather than by `releaseBuf`, is a buffer production
  code never released, and the I2 check reports it with its count and bytes. Before any change, the cleanup's count on the kernels
  suite and the gpu selection is measured, with the creation sites of what it closed. That measurement decides the rule:
  - if every one comes from test code, the check fails on a count above 0 from production code;
  - otherwise, the production sites are listed, and each is fixed (an explicit release) or, if it is by design, named in an
    allow-list with why. No such entry is added without saying so in the PR.
- The scan reads `CgoFiles` too. `destroyBuf` and `ReleaseDestroyBuffer` go. The doc's line list gets `state` and `test`.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM3bf-1 lines | the three sites lease as above; the census unchanged | — | runs to completion | seconds |
| G-MM3bf-2 I2 not vacuous | the kernels suite and the gpu selection pass with the collection count checked by the rule above. A planted buffer that production code leaves unreleased is reported (a unit test creates one, drops the reference, and sees the count) | a timed run holds the card | one run | about four minutes |
| G-MM3bf-3 the scan | `TestLedger_everyBufferIsLeased` reads `GoFiles` and `CgoFiles`; its plant still gives 4 | — | runs to completion | seconds |
| G-MM3bf-4 builds | as G-MM3b-6 | — | runs to completion | about two minutes |

**Read, 2026-10-27 (nobara, RTX 2070 SUPER over Vulkan; exclusive timing lock): all four pass, after the measurement found three
real faults.**
- **The measurement.** My first count of GC-closed leases ran without `-v`. `go test` discards a passing package's output, so
  it reported none, and that reading was wrong. Run with `-v`, it found 625 in the kernels suite and 448 in the gpu package:
  - **Production, `FusedMLP`.** It kept three raw buffers in a `[]interface{ Release() }`. Their `Release` reached wgpu by
    interface dispatch, bypassing `releaseBuf`, which the typed scan did not see. They are now `leasedBuf`, whose `Release`
    goes through `releaseBuf`. The scan also flags any `*wgpu.Buffer` converted to an interface with a `Release` method (an
    append, a slice literal or an argument); on the old `FusedMLP` it reports all three.
  - **Production, `DecodeRunner.release`.** It freed the runner's buffers but not a LoRA adapter still bound (four buffers and
    two bind groups per projection). They were left to the GC. The runner now frees them with itself.
  - **Tests.** The two Context-close reproducers leak a matrix per round on purpose, and they already set `leakExempt`. A
    Context closed while it is set is not judged; my plant sets it too.
- **G-MM3bf-1:** `UploadF32` leases `scratch`. `UploadF32On` takes `companion` in the vision paths and `dense` for the
  resident's weights. The headroom probe and `MLALatentStore` lease `scratch`. The census is unchanged.
- **G-MM3bf-2:** `LedgerBreaks` counts a buffer the GC freed with its lease open as a break. Both suites pass with none:
  - the kernels suite passes all 112 tests, judging 103 Contexts (the reproducers' 600 and the plant are exempt);
  - the whole gpu package passes, with 251 Contexts.
  - `TestLedger_anUnreleasedBufferIsCounted` sees its one dropped buffer, 512 bytes.
  - I3 stays exact: 196 against 196.
- **G-MM3bf-3:** the scan reads `GoFiles` and `CgoFiles`, and its plant now gives 5.
- **G-MM3bf-4:**
  - The pinned staticcheck passes with the gpu and testhooks tags, as do `touched.py` (the whole gpu package) and the
    citation lint.
  - The layout tests pass; the arrow data drops a test import the tree no longer makes.
  - `destroyBuf` and `ReleaseDestroyBuffer` are removed.
  - The doc's line list is now `kv`, `dense`, `experts`, `companion`, `state`, `test` and `scratch`.

#### MM3c's gates (leases in Metal's allocator), registered 2026-10-27 (nobara-mm01; Francis's go of 11:08)

The lease code in `metal/internal/gpudev` (the `Leaser`, `MustBufOn`, the lined constructors) was written before this
section, while surveying, and is not yet committed. Everything else is written after it, and every gate is graded after it.
The device gates run on a Mac: their commands are in `/tmp/mm3-metal.md`, handed to a Mac session by the coordinator.

**The design (CUDA's, with Metal's differences).**
- **gpudev leases every MTLBuffer** through a `Leaser` that the metal package installs at init (`gpudev.NewLeaser`). gpudev
  imports nothing of goinfer's ("gpudev-standalone").
  - A buffer's requested bytes are its `length`, its charged bytes its `allocatedSize`.
  - It leases on `Unified`; a `NewBufferNoCopy` alias leases on `FileMapped` (Francis's go).
- **The metal package has no allocation chokepoint.** Model code calls gpudev's constructors and the package's own helpers
  (`NewBufferU32`, `NewBufferFloats`, `byteBuf`, the int4, int8 and q4k packers, the alias helpers) at 546 sites.
  - Each helper gets a lined twin (`…On(line, …)`); the unlined name stays as a wrapper on `gpudev.UnlinedLine` for the 826
    test call sites.
  - Every production call site names its line, chosen by its enclosing function (with per-line overrides where a function
    mixes lines), by a typed rewrite (go/types, darwin).
  - Lines: `dense` (weights, norms, RoPE tables), `experts` (expert stacks and their biases, the expert pool), `kv`, `state`
    (recurrent and conv state, checkpoints), `companion` (towers, encoders, LoRA), `test` (test hooks), `scratch` (the
    rest: activations and the per-op uniforms).
- **Nothing reads the ledger to decide.**

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM3c-1 I1, static | a scan over every non-test file of the metal module finds no call to an unlined gpudev constructor (`NewBufferLen`, `NewBufferBytes`, `NewBufferNoCopy`, `MustBuf`, `NewBufferOf`, `NewBufferLenOf`) or an unlined metal helper outside its own wrapper; it reads every file and goes red on a planted call. It runs on darwin; the metal-darwin CI job runs it, and it is also run from Linux with a darwin type-check before the push | it reads fewer files than the module holds | runs to completion | seconds |
| G-MM3c-2 I1 and I2, dynamic (Mac) | a release hook in the metal package's `TestMain` fails the run if any released device had an unplanned byte, or a lease open after release; run with the Mac's device suite (the commands name the tests) | the Mac's apps hold the GPU (the run notes `CurrentAllocatedSize` at its start) | one run | the Mac session's estimate |
| G-MM3c-3 I2 per request (Mac) | on llama-tiny resident on Metal, the charged bytes after a second and a third identical request equal those after the first | as G-MM3c-2 | one run | seconds |
| G-MM3c-4 I3 (Mac) | `CurrentAllocatedSize` just after the resident's device is created, against after the build and one request on llama-tiny: **charged (Unified) ≤ measured ≤ charged + 64 MiB** (pipelines, libraries, the command queue). The FileMapped leases are reported apart, with whether the measured delta includes them | another Metal allocation in the process during the cell | one run | seconds |
| G-MM3c-5 no guard change | `TestDecisionCensus` passes unchanged; the Mac's Metal fit tests (`MetalFitCtx`, the memory guard) pass unchanged; nothing outside tests reads the ledger | — | runs to completion | about a minute |
| G-MM3c-6 builds | vet for darwin (untagged and `goinfer_testhooks`), the pinned staticcheck for darwin with its canary, gofmt, `touched.py`, the layout tests, the citation lint | — | runs to completion | about two minutes |

If I3 misses its band on the Mac, the finding is reported first and the band is not widened.

**MM3c read on nobara, 2026-10-27 (a darwin type-check; the device gates follow from the Mac, `/tmp/mm3-metal.md`).**
- **G-MM3c-1:**
  - The typed rewrite found 546 calls: 147 to gpudev's methods, 36 to its generic constructors and 363 to the package's
    helpers. 38 of them are inside the helpers themselves.
  - Each call now names its line. Each helper has a lined twin, and an unlined wrapper where tests call it.
  - staticcheck flagged the ten wrappers no test calls, and they were removed.
  - A replay of the scan's logic on Linux reads all 71 non-test files and finds no unlined call.
  - The scan joins CI's device-free metal-darwin step, behind its zero-match guard.
- **G-MM3c-5 (the part that runs here):** `TestDecisionCensus` passes unchanged. Nothing outside tests reads the ledger.
- **G-MM3c-6:** `go vet` for darwin passes, untagged and with `goinfer_testhooks`. The pinned staticcheck for darwin with the
  test hooks is clean. Also pass: gofmt, `touched.py`, `test_touched.py` (the scan listed), the layout tests (`imports.json` and
  the ARCHITECTURE appendix updated) and the citation lint.
- **G-MM3c-2, 3 and 4:** to be run on the Mac.

**Amended before merge and before the Mac run (coordinator, 14:28, Gemini's review of #335).**
- **G-MM3c-1's scan is tightened.** It now also flags a lined call handed `UnlinedLine` or `""` as its line, and it skips only
  true wrappers: a function whose body hands `UnlinedLine` to its own `…On` twin. Before, it skipped any function with a listed
  name. `newBufferLenUntracked`, a probe allocator with no production caller, takes its line from its caller.
- **G-MM3c-4 gets an aliasing cell, because llama-tiny never aliases.** `gemma4-dense-scaled` is transcoded to a Metal `.giw`
  and loaded with `GOINFER_METAL_ALIAS=1`, about 53 MB of no-copy windows. Pass:
  - its FileMapped leases are above 0 (aliasing was exercised); and
  - measured is within [U, U+64 MiB], or within [U+F, U+F+64 MiB], where U is charged Unified and F is charged FileMapped.
  - The record says which of the two held, which answers whether `CurrentAllocatedSize` counts no-copy buffers.
  - A FileMapped charge counts each window's pages, so a page shared by two adjacent windows is counted twice (at most a page
    per tensor). That is stated, not changed.
  - Each I3 cell runs in its own process (`-run` per cell), because the system device is shared and earlier tests' frees can
    land during a cell.
- **#2 holds as designed.** `ReleaseAll` at `Close` is how a Metal resident frees its buffers, so a lease closed there is not a
  leak. The leak check within a model's life is G-MM3c-3: charged bytes do not grow across identical requests.
- **#1 holds.** `UnlinedLine = test` makes a test's own unlined calls planned. Production's unlined calls are excluded by the
  static scan, which is what I1's static half is for.
- **Lines (#5):** MLA's RoPE table leases `dense`; the q4k segment counts and `uHasSink` (uniforms) lease `scratch`. The counts
  in this section read 546 calls throughout.

### 6.6 What it absorbs, and what stays outside

- **Absorbs:** KV11 (MM4); the fit guard's byte and KV accounting, which the owner put on Track OP on 2026-10-21 (MM5);
  MEM12-scratch's tier-0 counters (I1); MEM2's real `Close` and MEM6's returning memory after each chunk (I2); MEM7's
  sizing to the machine (`Solve`); the guard-threshold rule (I7).
- **Stays outside:** the paging mechanics (the pagers keep their I/O and ask the arbiter), the allocators' own code
  (they report to the ledger), and the kernels.

### 6.7 Open questions

- macOS: which reading the ledger closes against (the task's footprint, not RSS), and how Metal residency sets and
  no-copy buffers count.
- CUDA pinned host memory as its own tier, or `HostAnon` with a flag.
- Whether the per-process `VmSwap` reading (owner, 2026-10-17) stays a guard or becomes I3's calibration input on Linux.

#### MM1's first item: the two CUDA device tests that fail on main, registered 2026-10-26 before the fix (nobara-mm01)

Francis put these in MM1's scope (2026-10-26 17:35): `TestCUDAKVSlots_pricedAgainstWhatIsLeft` (the free-VRAM probe runs twice,
the test wants once) and `TestResidentDenseBytes_matchesCUDADevice` on the 0.5B and the 1.5B (479 MB and 998 MB more on the card
than Plan expects). Both fail the same way at origin/main (the AX2d logs on nobara).

**The cause, as found in the code before any run:** KS-B8's `choosePrefillWidth` (commit 1622062be) reads `cudaFreeVRAM` a second
time, as the build's last step, after the KV pools are allocated. Both tests stub `cudaFreeVRAM` and treat every call as
`checkKVFits`' read, which comes before the KV. The probe-count test fails on the count. The dense-bytes test keeps the last
call's reading, so its "device before KV" figure includes the KV pools. That would explain its surplus: Qwen2.5's KV at the
default context comes to roughly what is reported. Production is unaffected: `checkKVFits` still reads first and alone prices the KV,
and no fit decision or default moves. So the fix is to the tests: each records the build's first read.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-CU-1 the cause | in the dense-bytes test on the 0.5B and the 1.5B, the second read's drop exceeds the first's by at least the KV pools' charged bytes (`kvChargedBytes` at the build's context and slots), and the first read is within Plan + slack + margin | a timed job holds the card (the run takes the timing lock), or the build reads the probe other than twice | one run | under a minute of GPU on nobara, both models from `~/models` |
| G-CU-2 the fix | both tests pass at the fix on llama-tiny, the 0.5B and the 1.5B, with `-v` showing `--- PASS` for each case; the rest of `cuda`'s `cudaFreeVRAM`-stubbing tests (`TestCUDAKVSlots_`, `TestKSB8`, `TestKSH6`, `TestBuildScratch`) still pass | as G-CU-1 | one run | under two minutes |

If G-CU-1 fails (the second read does not account for the surplus), the cause is something else and the fix is not made: the
finding is reported first.

**Read, 2026-10-26 (nobara, RTX 2070 SUPER, driver 595.91.07; kwin_wayland's 17 MiB context present; timing lock held): both pass.**
G-CU-1: the build reads the probe twice, and the second read sits 480 MB (0.5B) and 1,008 MB (1.5B) above the first, exactly
the KV pools' charged bytes (one slot at context 16,384). That accounts for main's 479 MB and 998 MB surplus. At the first read
the device holds 466 MB and 1,054 MB, 5 and 6 MB above Plan + slack, inside the 214 and 248 MB margins. G-CU-2: both tests pass
on llama-tiny, the 0.5B and the 1.5B, with every `TestCUDAKVSlots_` test. With `GOINFER_HEAVY_TESTS=1`, `TestBuildScratchAccounting`
(six models), its planted-defect test and the three KS-B8 build tests pass too. Logs: `~/goinfer-logs/mm1/` on nobara
(cuda-probe-tests and cuda-probe-heavy, 2026-10-26).

## 7. Track AX — the towers and GPU backends move into goinfer (SD1, decided 2026-10-26)

### 7.1 What moves (aikit main on 2026-10-26)

| aikit package (module) | files, lines | tests | goinfer non-test importers today | lands in goinfer |
|---|---|---:|---|---|
| `github.com/townsendmerino/aikit/vision` (root module; Hard tier) | 23, 6,323 | 30 | 33 files: 18 in the root module, 1 in `demo/agent`, cuda 6, metal 3, gpu 5 | `internal/vision` (SD5) |
| `github.com/townsendmerino/aikit/audio` (root module) | 8, 2,227 | 6 | root 10, cuda 1, metal 1 | `internal/audio`, beside `internal/whisper` |
| `github.com/townsendmerino/aikit/gpu` (its own module, v0.35.0): the device substrate (Metal and CUDA behind one `Device`/`Buffer`/`Queue`/`Pipeline`/`Encoder` vocabulary), copies and uploads, streams, residency sets, the ViT kernels | 15, 5,125 | 38 | cuda 31, metal 8, root 1 | `internal/gpudev` |
| `…/gpu/qwencuda`, `…/gpu/visioncuda` (modules) | 1 each, 484 and 330 | 1 each | `visioncuda` in one cuda test | the `cuda` module |
| `…/gpu/qwenmetal`, `…/gpu/visionmetal` (modules) | 1 each, 527 and 398 | 2 and 3 | none | the `metal` module |
| `…/gpu/webgpu` (module) | 1, 43 | 1 | none | the `gpu` module, or retired |

- **Stays in aikit:** `linalg`, `embed`, `encoder`, `mmap` and the RAG packages (`chunk`, `bm25`, `ann`, `fuse`,
  `hybrid`, `late`, `sparse`, `topk`). The edges run one way: `vision` and `audio` import `embed` and `linalg`, and no
  RAG package imports `vision`, `audio` or `gpu`.
- **The edge cases (SD2):** `…/gpu/anncuda` and `…/gpu/annmetal` (ANN on the GPU, used by aikit's own `examples/gpu-ann`)
  and `…/gpu/enccuda` and `…/gpu/encmetal` (the text encoder's matmuls on the GPU; no goinfer, ken or tack import)
  depend on the device substrate.

### 7.2 Where it lands, and why internal

- **Internal packages add no public API,** so no apidiff and no tier promise (SD5). The `internal` rule is by import
  path, so the `cuda`, `metal` and `gpu` modules, whose paths sit under `github.com/townsendmerino/goinfer`, can import
  the root module's internal packages; `golang.org/x/tools/gopls` does the same with `golang.org/x/tools/internal`.
- **`multimodal` (public) keeps the goinfer side of each tower** (block sentinels, preprocessing configs, splicing) and
  exposes what the examples and the demo need, so no example imports an internal package.
- **Version skew follows the existing rule:** a backend module's tag pairs with the root tag it was built with
  (`RELEASING.md`'s two-step tag).

### 7.3 Steps

| step | what | gate (sketch) |
|---|---|---|
| AX0 | freeze the moving paths in aikit at the tags the move starts from (root v1.75.0 or later, `gpu/v0.35.0` or later); a note in aikit's README | no aikit commit touches a moving path after its freeze tag |
| AX1 | copy into goinfer with provenance: the commit message names the aikit tags; NOTICE and THIRD_PARTY_LICENSES lines move with the code; tests, goldens and fixtures move; build tags kept | the moved tests pass in goinfer, by name, with `-v` (a skip is not a pass) |
| AX2 | switch the imports one module at a time: root, then gpu, metal, cuda | **tower identity:** for every tower family's fixture, the tower's output bytes are identical before and after (raw bits, mutation-checked once); every family's image and audio parity tests unchanged; build, vet and staticcheck under every tag |
| AX3 | drop `github.com/townsendmerino/aikit/gpu` and `…/gpu/visioncuda` from every goinfer `go.mod`; a source scan in `scripts/touched_scans.py` refuses an import of a moved aikit path | `GOWORK=off` builds of each module after the two-step tag |
| AX4 | aikit side. `vision` is in the Hard tier: a `Deprecated:` package comment pointing to goinfer, frozen, removed only at a v2. `audio` is not: deprecated for one minor, then removed. The GPU module and the four tower backends: deprecation comments in their `go.mod` files and no further tags. `gpupins` and the GPU gate tools trimmed to what stays; aikit's README tiers updated | aikit's releasegate and consumer gate (ken, tack) green |
| AX5 | SD2: the ANN and text-encoder GPU backends | per SD2 |
| AX6 | SD3: `linalg` and `embed` worked on through `go.work` (goinfer's gitignored `go.work` lists `../aikit` for cross-repo work); aikit tagged weekly; with VK1, a bump with unchanged numerics re-stales nothing | — |
| AX7 | 30 days after AX4: aikit bumps in goinfer, aikit tags, pin-only commits and parity re-stales caused by aikit, against §2.4 | — |

### 7.4 What it should buy

- About 15 of the last 30 days' 37 bumps (the towers and the GPU module) stop being bumps; the linalg and embed changes
  batch into about four tags a month.
- The pin-only commits for the moved modules stop.
- Tower and device work becomes an ordinary goinfer edit: the test cache, `touched.py`, one PR.

### 7.5 Risks

- aikit keeps a frozen copy of the device substrate for the ANN and encoder backends (SD2); the two copies diverge by
  design.
- The history stays in aikit; goinfer's copy cites the freeze tags.
- Cross-repo citations in goinfer's docs that point at moved aikit files must be re-pointed in the same commits, or the
  citation lint goes red.
- The CUDA half of the substrate builds on Linux only; the move keeps its build tags.

### 7.6 AX1 gates (registered 2026-10-26 11:40 PDT by mac-ax01, before the copy was made or any test ran; committed here once this doc reached the branch)

Tier 0 throughout: no model, no timed run. The Metal device tests touch the Mac's GPU, so every test command holds the
timing lock shared (`timing_lock.py run --shared`).

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| AX1-T (tests carry over) | Baseline: each moved package's tests at its freeze tag (a `git archive` of root `v1.75.0` and of `gpu/v0.35.0`, tracked files only), on this Mac, `go test -v -count=1`. The copy: the same packages in goinfer, same flags. **Pass:** every test that reads `--- PASS` in the baseline reads `--- PASS` in goinfer, by name, and no test reads `--- FAIL` in goinfer. A test that skips in the baseline and skips in goinfer for the same reason (a real checkpoint, an untracked bench fixture, a CUDA device) is listed, not counted as a pass. | a baseline test fails at the tag (it is listed and excluded, not fixed here); the timing lock is held exclusive by another session (wait, do not overlap) | each side runs once, to completion; a red is re-run by name only | not yet measured; estimated under 10 min a side on the Mac (vision and audio CPU tests plus the Metal ViT tests); the baseline's wall time becomes the record |
| AX1-S (source identity) | Every moved `.go` file equals its file at the tag after the scripted rewrite (import paths, the package clause and doc line of `gpudev`, the fixture paths), and nothing else differs; every non-Go file (`.cu`, `.ptx`, `testdata/`) is byte-identical. A script diffs the two trees. | — | complete | seconds |
| AX1-B (builds under every tag) | `go build` and `go vet` of: the root module (darwin), `internal/gpudev` with `GOOS=linux`, the `metal` module (`metal`, `metal goinfer_testhooks`), the `cuda` module with `GOOS=linux GOARCH=amd64 CGO_ENABLED=0` (`cuda`, `cuda goinfer_testhooks`), the `gpu` module; `gofmt -l` empty; CI's pinned staticcheck on the new packages, after its canary prints U1000 | — | complete | about a minute |
| AX1-N (adds only) | `git diff origin/main --name-status` shows only added files, plus `go.mod`/`go.sum` lines, `.gitignore` exceptions for the moved fixtures, NOTICE and THIRD_PARTY_LICENSES lines, and docs; no existing `.go` file changes | — | complete | seconds |

What this gate cannot see, and who runs it: the CUDA device tests (`internal/gpudev`'s `cuda_*_test.go`, `cuda/internal/visioncuda`,
`cuda/internal/qwencuda`) build here with `GOOS=linux` and run only on nobara; they are listed for nobara in the PR.

**Result, 2026-10-26 (mac-ax01):** AX1-S, AX1-N, AX1-B and AX1-T PASS on the Mac; the measured cost was 11 s for the aikit baseline and 8 s for goinfer. The record, the logs and the two scripts are in [`../measurements/ax1-2026-10/`](../measurements/ax1-2026-10/README.md). Open: the CUDA device tests on nobara.

**Amended after CI, 2026-10-26:** the device package does not land at `internal/gpudev`. In the root module it put purego and
gocudrv into the root's packages and module graph, which `ci.yml`'s cleanliness guard refuses; a nested module would
have been a sixth module against rule 4 (§4). It splits by backend instead: `cuda/internal/gpudev` (the Linux files, the
`.cu` sources and PTX) and `metal/internal/gpudev` (the darwin files), shared files in both. That supersedes the
`internal/gpudev` landing spot in §7.1 and §9. Gates re-run on the split, all PASS; the record has the detail.

### 7.7 AX2 plan (proposed by mac-ax01 at AX1's stop point; AX2 registers these gates in full, with void, stopping and cost terms, before its code)

**`vision` and `audio` switch in every module in one commit; the device package switches one module at a time after**
(each backend to its own half).
The towers cannot go one module at a time: the backends register their resident encoders into `vision`'s package-level
hooks (`vision.RegisterResident`, `vision.RegisterQwenResident`) and receive the root's `*vision.Encoder` values, so a
root on `internal/vision` beside a backend on aikit's `vision` either fails to compile (the types) or compiles and
silently runs the CPU tower (the hooks). Importers on origin/main `4a91080e`, non-test + test files:
`vision` 18 + 22 root, 6 + 23 cuda, 3 + 16 metal, 5 + 8 gpu, 1 `demo/agent`; `audio` 10 + 18 root, 1 + 3 cuda, 1 + 3
metal; aikit's `gpu` 31 + 52 cuda, 8 + 10 metal (the root imports none: `cmd/gate`'s release check and
`internal/hwcensus`'s census name it as a module string, and AX3 points both at the two `gpudev` halves).

| step | what | gate |
|---|---|---|
| AX2a | a tower-identity test, landed before any import moves: for every tower family's committed tiny fixture (SigLIP, Gemma 4, Qwen2.5-VL, Qwen3.5-VL, Qwen3-VL DeepStack, GLM-OCR, Pixtral, SigLIP2 NaFlex, Whisper, Gemma 4 audio, Qwen3-ASR, Voxtral) and every backend present (CPU; Metal on the Mac; CUDA on nobara), it writes the tower's output bytes' sha256 and whether the resident (device) path ran, to a JSON record | records at the base commit; tier 0 |
| AX2b | `vision` and `audio`: rewrite every importer in root, `gpu`, `metal`, `cuda` and `demo/agent` to `internal/vision` and `internal/audio`, one mechanical commit (a script, as AX1's); `examples/invoice` moves to what `multimodal` exposes (§7.2) | **tower identity:** every (family, backend) hash equal to AX2a's record, and the resident flag unchanged; mutation-checked once by leaving one backend on aikit's `vision` (its resident flag must flip and the gate go red); every family's image and audio parity test unchanged; build, vet and staticcheck under every tag; apidiff clean |
| AX2c | aikit's `gpu` to `metal/internal/gpudev` in the `metal` module (with `metal/kernels`; `metal/internal/*` already use it) | the same identity record on Metal; the metal suite's device tests by name |
| AX2d | aikit's `gpu` to `cuda/internal/gpudev` in the `cuda` module, and its one test's `visioncuda` import to `cuda/internal/visioncuda` | the same identity record on CUDA, on nobara; the CUDA suite's device tests by name |

Then AX3 drops `aikit/gpu` and `aikit/gpu/visioncuda` from every `go.mod`. Docs citing a moved aikit file stay valid
while aikit keeps the frozen copy (AX4 removes `audio` after one minor); each is re-pointed in the AX2 commit that moves
its importer, or the citation lint goes red at AX4.

### 7.8 AX2a and AX2b gates (registered 2026-10-26 15:22 PDT by mac-ax01, before AX2a's code)

Tier 0: tiny committed fixtures, f32 towers, no model, no timed run. Every Mac test command holds the timing lock shared.
The test-only environment variable below is read by tests only, so `testdata/env_reads.txt` does not change.

**The instrument (AX2a).** `internal/towerid` runs each tower family on its committed tiny fixture with fixed synthetic
inputs: SigLIP, Qwen2.5-VL, Qwen3.5-VL, Qwen3-VL with DeepStack, GLM-OCR, Gemma 4 vision, Pixtral, SigLIP2 NaFlex,
Gemma 4 audio, Whisper, Voxtral and Qwen3-ASR. On a device backend it attaches each tower through the registry production
uses: the unnamed slots (`EnableResident`) for SigLIP and Qwen2.5-VL, and the named ones (`multimodal.NewQwen3Tower`,
`NewGlmOcrTower`, `NewGemma4Tower`, `embeddinggemma2.NewAudioAccelerator`, `multimodal.UseWhisperDevice`) for the rest. It
covers every family the backend serves. A record holds, per (backend, family, case), the SHA-256 of the output's float32
bits and whether the device path attached. One test per backend calls it: the root module (CPU), `metal` (Metal), `gpu`
(WebGPU, `-tags gpu`) and `cuda` (CUDA, on nobara). `GOINFER_TOWER_ID_OUT=<file>` writes the record.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| AX2a-T (the instrument holds) | Each backend's test passes. Every case runs twice with equal hashes, or the identity gate below could not be read. Every family the backend serves attaches its device path; an attach error is a FAIL, not a skip. | the backend's device is absent (a named skip, VOID for that backend, never a pass) | once per backend, to completion | not measured; estimated under a minute a backend on the Mac. The first run's wall time becomes the record |
| AX2b-ID (tower identity) | A record from the AX2a test at AX2b's base commit and one at AX2b's head, same machine and flags. **Pass:** every (backend, family, case) hash and device flag is equal. Any difference is a FAIL, with no tolerance band: the switch changes no numeric code, because `internal/vision` and `internal/audio`'s production files are aikit v1.75.0's, byte for byte. Backends: CPU, Metal and WebGPU on the Mac; CPU and CUDA on nobara (relayed, by day). | as AX2a-T | each record once; a difference is diagnosed by name, never re-run until it agrees | AX2a-T's measured time, twice per backend |
| AX2b-M (the identity gate can go red) | Once, at AX2b's head, plant the failure it exists for. One backend's registration for an unnamed slot is left on aikit's `vision`. If that compiles, its family's device flag must flip and AX2b-ID read FAIL. If no such plant compiles, the record says so: the split cannot ship silently there. The gate is then planted at the test instead, with the registration name changed, and must read FAIL. | — | once | minutes |
| AX2b-N (the mechanical commit is the script) | The import switch is one commit made by a script. Re-running the script on that commit's parent reproduces it byte for byte. The commit contains nothing else. The script also runs gofmt. | — | complete | seconds |
| AX2b-S (nothing left on aikit's towers) | `go list -deps` of every module, under every build tag it uses (root with `realckpt` and `goinfer_testhooks`; `metal`; `gpu`; `cuda` with `GOOS=linux`; `demo/agent`), names neither `github.com/townsendmerino/aikit/vision` nor `.../aikit/audio`, test packages included | — | complete | seconds |
| AX2b-B (builds) | Build and vet under every tag. The cleanliness guard is run from the root as CI runs it. gofmt, and CI's staticcheck after its canary. `touched.py`. CI's full suite on the PR, apidiff included. The image and audio parity tests are in that suite, and none of their goldens may change. | — | complete | minutes; CI as usual |

**AX2a result, 2026-10-26 (mac-ax01):** AX2a-T PASS on the CPU (12 families, 22 cases), Metal (6 families, 17 cases) and
WebGPU (5 families, 15 cases), each under a second; CUDA builds and waits for nobara with AX2d. A device backend with no
registration linked in errors on its first declared tower instead of returning CPU bits. The record is
[`../measurements/ax2-2026-10/`](../measurements/ax2-2026-10/README.md).

**AX2b result, 2026-10-26 (mac-ax01):** AX2b-N, -S, -ID (CPU, Metal, WebGPU: 54 cases, 0 differ) and -M PASS on the Mac;
AX2b-S caught `cuda/s4siglip`'s blank import of aikit's `gpu/visioncuda`, which compiles and would have registered into a
slot nothing reads, and the script now maps it to `cuda/internal/visioncuda`. AX2b-M's fallback form: no single
registration file left on aikit compiles, and two plants at the registration each turned the Metal identity test red.
CUDA's and nobara's CPU records wait for nobara. Detail in the record.

**What AX2b changes outside the import lines.** First, `multimodal` and `embeddinggemma2`'s exported functions that take
tower types now take `internal/vision` and `internal/audio` types. Both packages are Experimental tier
(`docs/api-tiers.md`; apidiff checks `decoder`, `tokenizer`, `chat` and `constrain`), so this is allowed, and it gets a
CHANGELOG line. Second, `examples/invoice` and `demo/agent` reach the towers through new `multimodal` functions instead
of importing an internal package (§7.2). That is a separate commit after the mechanical one, gated by AX2b-B, with each
example's own test.

### 7.9 AX2c gates (registered 2026-10-26 15:45 PDT by mac-ax01, before AX2c's code)

AX2c switches the `metal` module (with `metal/kernels`) from aikit's `gpu` to `metal/internal/gpudev`, the device
package's Metal half from AX1. It is one scripted commit (`ax2cd_rewrite.py metal`): the import path becomes
`gpu "github.com/townsendmerino/goinfer/metal/internal/gpudev"`, so no body changes. Tier 0 and tier 1 (the Mac's GPU,
no model), by day, under the shared timing lock.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| AX2c-N | Re-running the script on the commit's parent reproduces the commit byte for byte. The commit contains nothing else. | — | complete | seconds |
| AX2c-S | `go list -deps -test -tags goinfer_testhooks ./...` in `metal` names no `github.com/townsendmerino/aikit/gpu`. | — | complete | seconds |
| AX2c-ID | `TestTowerIdentity_metal`'s record at AX2c's head equals AX2b's head record, case by case. | no Metal device | once | a second |
| AX2c-T (the Metal suite carries over) | The `metal` module's tests, one package per `go test -v -count=1 -tags goinfer_testhooks` call, at AX2b's head (the base) and at AX2c's head, compared by name with `ax1_compare.py`. **Pass:** every base PASS is a PASS at the head, and no FAIL at the head. A test that fails at the base is listed and excluded, not fixed here. | another session holds the timing lock exclusive (wait); the Mac's free memory is under 4 GB at a run's start (a device suite under memory pressure fails for reasons no code owns; the run is re-made) | each side once, to completion; a red is re-run by name only | the Metal suite measured 4 min 12 s in a quick-full run (`CLAUDE.md`, "Gate tiers"); estimate 5 min a side, 10 min in all |
| AX2c-B | Build, vet and staticcheck of `metal` under its tags (plain, `goinfer_testhooks`); gofmt; the root cleanliness guard | — | complete | a minute |

### 7.10 AX2d gates (registered 2026-10-26 15:45 PDT by mac-ax01, before AX2d's code)

AX2d switches the `cuda` module from aikit's `gpu` to `cuda/internal/gpudev`, the device package's CUDA half from AX1,
in one scripted commit (`ax2cd_rewrite.py cuda`). The Mac compiles it (`GOOS=linux`); the device legs run on nobara,
by day, through nobara's own session (relayed), never over ssh from the Mac.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| AX2d-N | Re-running the script on the commit's parent reproduces the commit byte for byte. The commit contains nothing else. | — | complete | seconds |
| AX2d-S | `GOOS=linux go list -deps -test -tags 'cuda goinfer_testhooks' ./...` in `cuda` names no `github.com/townsendmerino/aikit/gpu`. | — | complete | seconds |
| AX2d-B | Build, vet and staticcheck of `cuda` with `GOOS=linux GOARCH=amd64 CGO_ENABLED=0`, tags `cuda` and `cuda goinfer_testhooks`; gofmt; the root cleanliness guard | — | complete | a minute |
| AX2d-ID (nobara) | `TestTowerIdentity_cuda` and `TestTowerIdentity_cpu` records at AX2b's base, AX2b's head and AX2d's head, equal case by case (one nobara session covers AX2b-ID's CUDA leg too). | no CUDA device answers | once each | seconds each |
| AX2d-T (nobara) | The `cuda` module's tests (`-tags 'cuda goinfer_testhooks'`, one package per `go test -v -count=1` call) at AX2c's head (the base) and at AX2d's head, compared by name with `ax1_compare.py`. **Pass:** every base PASS is a PASS at the head, and no FAIL at the head. `TestCUDA_launchDirect` segfaults on both sides at AX1 (#279), so it runs alone and the rest with `-skip '^TestCUDA_launchDirect$'`; a test failing at the base is listed and excluded, not fixed here. | a desktop session holds a CUDA context (`kwin_wayland`), which shifts plans and skips the drain-group tests (`CLAUDE.md`, "Testing tiers"): the record notes it, and drain-group skips on both sides are not counted as passes | each side once, to completion | nobara's session states the estimate from its last CUDA suite record before starting |

**AX2c and AX2d results, 2026-10-26 (mac-ax01):** AX2c-N, -S, -ID, -T (775 base passes, 0 regressions) and -B PASS on
the Mac. AX2d-N, -S and -B PASS on the Mac (compiled for linux/amd64); AX2d-ID and -T wait for nobara. Detail in the record.

### 7.11 AX3 gates (registered 2026-10-26 16:54 PDT by mac-ax01, before AX3's code)

AX3 drops `github.com/townsendmerino/aikit/gpu` and `…/gpu/visioncuda` from every goinfer `go.mod`. A test then keeps the
moved paths out: it fails on a `.go` file importing aikit's `vision`, `audio` or `gpu` (or anything under it), and on a
`go.mod` requiring the `gpu` module or one under it. Three things outside the imports move with it.

- **`internal/hwcensus`'s census** scans goinfer's two `gpudev` halves in place of the aikit module.
- **`cmd/gate release`'s row 2 and `decoder/aikit_pin_test.go`** stop pinning `aikit/gpu`; no module requires it.
- **The citation lint.** It resolves aikit `gpu/...` citations through `cuda/go.mod`'s require. With the require gone,
  it falls back to the frozen tag `gpu/v0.35.0`, which aikit keeps (SD2, AX0), so those citations stay valid. §7.5
  expected every citation to be re-pointed instead. CI's cache step downloads that tag.

One mechanical commit first: `cuda/kernels/ptx_test.go` (SB1's batch 1, merged after AX2d's rewrite) still imports aikit's
`gpu`, and `ax2cd_rewrite.py cuda` switches it.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| AX3-S (the scan) | The new scan passes on the tree. Planted once, it reads red twice: an aikit `gpu` import added to a test file, then an `aikit/gpu` require added to `metal/go.mod`. It reads green again after each plant is removed. It is listed in `scripts/touched_scans.py`. | — | complete | seconds |
| AX3-M (the module graphs) | No goinfer `go.mod` or `go.sum` names `aikit/gpu` or `aikit/gpu/visioncuda`. `go list -m all` names neither in `metal` (workspace), nor in `cuda` (workspace, `GOOS=linux`). | — | complete | seconds |
| AX3-B (builds and the moved checks) | Build, vet and staticcheck of every module under its tags. The root cleanliness guard. `TestHardwareCensus`, `cmd/gate`'s release tests and `decoder`'s pin test, by name with `-v`. `test_touched.py` and the citation lint, exit code read directly, with every aikit `gpu/...` citation resolving. `TestTowerIdentity_metal` against AX2c's record, as a smoke check. | — | complete | minutes |
| AX3-G (standalone builds) | §7.3's `GOWORK=off` build of each module. It needs the two-step tag, so it is deferred to the next release, where `standalone-build.yml` runs it; nothing here grades it. | — | — | — |

## 8. The other subsystems (SB1–SB13)

Each: what it is today, its contract, its fast check, what stops re-running, its first step and size.

### SB1 — the backend interface, at the level of operations

- **Today:** §2.2.
- **Contract:** a backend implements operations, not families. It receives a program, a view of the weights and a
  memory plan, and returns a resident with one call shape (RF1) that runs ragged rows. It declines at build time, by
  operation and shape, never at run time.

  ```go
  // package backendapi (internal): imported by decoder and by the cuda, metal and gpu modules. A sketch; OP4 and RF1 own the design.
  type Backend interface {
      Name() string
      Ops() OpSet                                                // the operation kinds and shape classes it implements
      Build(p Program, w Weights, plan mem.Plan) (Resident, []Decline, error)
  }

  type Resident interface {
      Forward(ctx context.Context, b Batch, out Outputs) error // decode, verify, prefill and image rows
      Demand() mem.Demand                                      // MM2
      Close() error
  }
  ```
- **Fast check:** an op conformance kit per backend, in the shape of llama.cpp's `test-backend-ops` and PyTorch's OpInfo
  (§16). Every operation kind runs on random small tensors across shape classes (head dims 64, 128, 256 and 512, GQA
  ratios, tails, thresholds) against the CPU reference: bit-identical where the operation is declared exact, within a
  declared tolerance where it is not. Synthetic two-layer programs exercise the walker. Nothing imports `decoder`.
- **Stops re-running:** backend suites on decoder edits, and family × backend runs (a pair carries over by VK's key).
- **First step, before OP4:** move the kernel-level tests in `cuda/` and `gpu/` into packages that do not import
  `decoder`, the way `metal/kernels` did on 2026-10-13. A ratchet test: the count of backend test files importing
  `decoder` (240, 245 and 87 today) may only fall.
- **Then** OP4 (residents walk the program) and RF1 (one call shape) land with the kit as their gate. L in all; the
  first step S–M.

#### SB1 step 1 — gates, registered 2026-10-26 before the code (nobara-sb01)

- **What a file-level count misses.** Package `cuda`'s own non-test files import `decoder` (21 of them), so every test in
  package `cuda` is rebuilt and re-run on a decoder edit whether or not its own file imports `decoder`. The same holds
  for `metal` and `gpu`. So the ratchet counts two things per backend module: **direct**, the test files whose own
  imports name `decoder` (the §6 census figure: 240, 245, 87), and **reach**, the test files in a directory whose test
  build reaches `decoder` through any chain of goinfer imports. Reach is the one a decoder edit invalidates; a kernel
  test leaves it only by leaving the package, as `metal/kernels` did.
- **Ratchet (tier 0).** `internal/layout`'s `TestBackendTests_decoderImportsOnlyFall` parses the imports of every Go
  file in the tree (build tags ignored, so a count is the union over tags; `//go:build ignore` files skipped) and fails
  when either count of any of the three modules differs from the committed figure: above it is a regression; below it,
  the figure is lowered in the same commit. It lands at the counts of `origin/main` at `bc365a837`, in its own commit:
  direct 246, 245 and 87, reach 352, 395 and 172 for `cuda`, `metal` and `gpu`.
  `-v` prints the census. Cost: milliseconds; no device, no model. Void: none (it is deterministic over the tree).
- **Each batch of moved tests (tier 0 on the device, seconds).** The batch's files move from `cuda/` to `cuda/kernels/`,
  a package that imports nothing from goinfer (aikit `gpu` and gocudrv only); it reads the PTX the moved tests load
  from `cuda/testdata/*.ptx`, the same files `cuda/kernels.go` embeds. A moved file's `goinfer_testhooks` constraint is
  dropped: it gates package `cuda`'s test seams, which a moved test does not use, so the moved test runs under `-tags
  cuda` alone, in `gate gpu`'s 2a suite.
  - **Rule:** every moved `Test`/`Benchmark` function is listed before the move and found after it (no test deleted);
    each moved test is run by name with `-v` on nobara's RTX 2070 SUPER before the move (in `./cuda/`, under `cuda
    goinfer_testhooks`) and after (in `./cuda/kernels/`, under `cuda`), and its outcome (PASS, FAIL or SKIP with the same reason) is the same in both;
    `go build` and `go vet` of the cuda module pass under each of no tag, `cuda`, and `cuda goinfer_testhooks`;
    `go list -deps -test ./cuda/kernels/` contains no `goinfer/decoder`; the ratchet's reach count for `cuda` falls by
    the number of moved files, and is lowered.
  - **Void:** a run while `timing_lock.py status` shows a timed holder (the runs hold the lock shared, so they refuse
    instead); a run on a card with another process's context (`nvidia-smi` names it), which can fail a test on memory
    for reasons not in the test.
  - **Stopping rule:** one before-run and one after-run per batch; a mismatch stops the batch and is reported, not
    re-run until it agrees.
  - **Cost basis:** the kernel tests in `gate gpu`'s 2a suite take about 12 s together
    (`docs/measurements/cuda-device-suite-2026-10-18/run-cuda-device-suite.sh`'s header); a batch is a fraction of it.
    Heavy-gated (`GOINFER_HEAVY_TESTS`) and env-gated (`GOINFER_A13_*`) tests skip in both runs, and the record says so.
- `gate gpu`'s 2a kernel-level suite lists `./cuda/kernels/` beside `./cuda/`, so the moved tests stay in it; the night
  `cuda-device-suite` and CI's cuda job run `./cuda/...` and need no change.
- **Batch 1 (2026-10-26):** 25 files (35 tests, 16 benchmarks) moved to `cuda/kernels/`, every outcome the same before
  and after; cuda reach 352 → 327. `gate gpu`'s heavy tier also runs `./cuda/kernels/` as one more process, so its two
  heavy-gated bandwidth tests stay in it. Record: `docs/measurements/sb1-kernel-tests-2026-10-26/README.md`.
- **Batch 2 (2026-10-26):** 5 files (6 tests) moved, outcomes the same; cuda reach 327 → 322. Same record. The next dozen
  kernel tests check kernels against package `cuda`'s production packers, so they wait for the packers to move.

#### SB1 step 1 — metal's batches: gates, registered 2026-10-26 21:41 PDT before the code (mac-ax01)

Francis's 17:36 decision: metal and gpu, the test-only method. gpu has no test-only move (nobara-sb01's record, "metal and gpu").
metal has 62 candidate files (67 `Test` and 20 `Benchmark` functions; 39 tagged `darwin`, 23 `darwin && goinfer_testhooks`).
Each file's only package-`metal` dependencies are `metal/kernels` aliases, `device.go`'s aliases and the pure helpers in
`testshared_test.go`. None imports `decoder`.

- **The move.** The files go from `metal/` to `metal/kernels/` (`package kernels`), in batches.
  - Each file keeps its build line. `goinfer_testhooks` is dropped only where the file uses none of package `metal`'s test
    seams; that is always so for these, since none uses `metal`'s test hooks.
  - `metal/kernels` gains test-only files: lowercase aliases for the MSL names and pack wrappers `metal/msl.go` and
    `metal/pack.go` give package `metal`, and copies of `argmaxF`, `cosF`, `parityStats` and `assertParity` (it already has
    `mustFinite`, `f16ToF32` and the device aliases).
  - No `decoder` file changes. If one would have to, the batch stops with DECISION NEEDED.
- **Rule, per batch:**
  - every moved `Test`/`Benchmark` function is listed before and found after (none deleted);
  - every moved `Test` runs by name with `-v` on the Mac's Metal device before the move (`./metal/`, tags
    `goinfer_testhooks`) and after (`./metal/kernels/`, the same tags), with the same outcome (PASS, FAIL, or SKIP for the
    same reason), compared with `ax1_compare.py`;
  - `go build` and `go vet` of the metal module pass with and without `goinfer_testhooks`;
  - `go list -deps -test ./metal/kernels/` names no `goinfer/decoder`;
  - `internal/layout`'s metal reach figure falls by the files moved (Direct stays 245) and is lowered in the same commit.
- **Void:** a run while another session holds the timing lock exclusive (the runs hold it shared and wait); a run with
  under 4 GB available (Metal tests fail on memory for reasons not in the test).
- **Stopping rule:** one before-run of all 67 and one after-run per batch; a mismatch stops the batch and is reported,
  never re-run until it agrees.
- **Cost basis:** kernel tests on tiny synthetic inputs, seconds each; the before-run is the measurement. Benchmarks are
  not run (`-run` only), and nothing is timed.

#### SB1 step 1 — moving cuda's weight packers: the bit-identity gate, registered 2026-10-26 before the code (nobara-sb01)

Francis's decision (17:36): move `packWeight` and its packers into a small decoder-free package, with a bit-identity gate
on the packed output registered first; stop with DECISION NEEDED if the move needs a `decoder` change or changes a byte.

- **The move.** `cuda/internal/wpack` (imports aikit `linalg` only) takes the packing bodies: `packWeight`'s per-kind
  packing, `packWeightStack`'s concatenation and its kind and K checks, `packI8`, `packI8Into`, `packInt4Into`,
  `permuteFast`, `nibblePosFast`, `f32tof16`. `decoder.Int4F32` is not imported: its body (`Int4F16`, then
  `linalg.F16ToF32Slice`) is the same two calls, made in `wpack`. Package `cuda` keeps every name as a thin wrapper, and
  keeps `planWeight`'s admission ahead of each pack, member by member, so its errors come in the same order. `hostW`
  stays in package `cuda`: it is the resident's plan (planned sizes, row readers, the int8 alternative, the stack's
  sources), and only its packed fields come from `wpack`.
- **Gate (tier 0, no device).** `cuda`'s `TestWeightPackers_bitIdentity` hashes (SHA-256) the output of every entry
  point above, through package `cuda`'s names, on seeded inputs: `packWeight` on every kind `planWeight` admits (int4 from
  `QuantizeInt4` and from `WrapInt4F16` with arbitrary f16 scales, subnormal and negative included; int8 with and without
  W8A8; q4k from raw blocks; f32; bf16), at more than one shape; `packWeightStack` on stacks of each; `packI8`,
  `packI8Into`, `packInt4Into` on random data; `nibblePosFast` on 0–7; `permuteFast` on 2^22 seeded words plus every
  single-nibble word; `f32tof16` on every one of the 2^32 float32 bit patterns. Each hash covers the kind, N, K, the
  three packed slices, and the error text where one is expected (a mixed-kind and a ragged-K stack, an unsupported
  kind). The hashes are written from the tree before any code moves and committed; after the move the same test, with no
  edit, must reproduce every one.
- **Void:** none; it is deterministic. **Stopping rule:** one run before (writes the hashes), one after; any differing
  hash stops the move with DECISION NEEDED. **Cost basis:** seconds, most of it the 2^32 sweep; measured: 28 s writing the
  32 hashes at `91fe36e7b`, 30 s reproducing all 32 on an unchanged tree.
- **Then** the kernel tests that needed only the packers move to `cuda/kernels` by batch 1's gate (each by name with
  `-v` on the device, before and after), using `wpack` directly, and the ratchet falls.
- **Done (2026-10-26):** all 32 hashes match after the move, and the gate went red on two planted changes (reverted). Six
  test files left package `cuda` (four to `cuda/kernels`, two to `cuda/internal/wpack`), every outcome the same; cuda
  reach 322 → 317. gpu has no test-only move (its kernels are package `gpu`'s production code); metal has 62 candidates
  that need a Mac run. Record: `docs/measurements/sb1-kernel-tests-2026-10-26/README.md`.

#### SB1 — launch geometry out of package cuda: the identity gate, registered 2026-10-26 before the code (nobara-sb01)

Francis's decision (21:42), item 1. The relayed text names `decoder/prefill.go` and `decoder/flash_decode.go`; neither file
exists, and no `decoder` file declares these constants. They are `cuda/prefill.go`'s and `cuda/flash_decode.go`'s, which the
STOP POINT it answers named. This work reads it that way and touches no `decoder` file; its PR waits for Francis to confirm
the reading before auto-merge is set.

- **The move.** `cuda/internal/launchgeom` (standard library only) takes the launch geometry the attention-digest tests
  need: `faWarps`, `faMaxRows`, `faRowsPerCTA`, `faRun` and `faRowRuns` (`flash_decode.go`); `attnFusedBM`,
  `attnFusedThreads`, `attnFusedKPAD`, `attnFusedMinRows`, `imgBlockMaxNWin`, `imgBlocksMaxNWin` (`prefill.go`). The KV
  cache's f16 converters `f32ToF16RNE` and `f32SliceToF16RNE` (`kvhalf.go`) go to `cuda/internal/wpack`. Package `cuda`
  keeps every name, as a constant or a thin wrapper.
- **Gate 1, values (tier 0).** `cuda`'s `TestLaunchGeometry_identity` hashes every moved constant's value and every moved
  function's output over a grid (`faRowsPerCTA` over head dims 32-512 and GQA ratios 1-16; `faRowRuns` over n0, m,
  window and S; `imgBlockMaxNWin` and `imgBlocksMaxNWin` over windows and blocks; `f32ToF16RNE` on all 2^32 float32 bit
  patterns; `f32SliceToF16RNE` on a seeded slice), through package `cuda`'s names, into
  `cuda/testdata/launchgeom_hashes.txt`, written before any code moves and committed; after the move the same test, with
  no edit, must reproduce every hash.
- **Gate 2, outputs (device, short, under the lock shared).** Before and after, by name with `-v`: the moved tests
  (`TestAttnKernelDigests` compares every attention kernel's output digest with `cuda/testdata/attn_digests.json`), and
  `gate gpu`'s 2b parity group (`go test -tags 'cuda goinfer_testhooks' ./cuda/`, the resident parity tests). Same outcome
  for every test. Void: a run beside a timed holder (the shared lock refuses). Stopping rule: one run each side; a
  difference stops the move with DECISION NEEDED.
- **Then** the attention-digest family (`attn_digest_rig`, `attn_digest`, `attn_softcap_kernel`, `mamba_kernels`) and
  `prefill_splitkv_img` move to `cuda/kernels`, and the ratchet falls.

#### SB1 — gpu's kernel layer: decided, and its gates, registered 2026-10-27 before the code (nobara-sb01)

Francis's decisions: the alias split (05:46), contained in `gpu/internal/kernels` (06:34), after `decoder`'s adapter types
moved to `internal/adapter` (06:57, #314). The cut, from a type-checked declaration graph of package `gpu` with its tests
(tags `gpu goinfer_testhooks`): a declaration stays in `gpu` if it uses `decoder`, uses a declaration that stays, or is a
method of a type that stays; an exported method that stays keeps its type in `gpu`, and an unexported one becomes a
function. That leaves in `gpu` the residency glue (`backend.go` with the `decoder.RegisterBackend` call, so WebGPU still
registers through the same import, `hwinfo.go`, `kv_slots.go`, `kvcopy.go`, `prefill_scratch.go`, `residency.go`,
`resident_sample.go`, `testhooks_kvscale.go`, `deprecated_release.go`'s backend check) and 75 test files; 48 non-test and
78 test files move. Two in-package pre-steps first: `Context.uploadProj` becomes the function `uploadProj(c, w)`, and
`deprecated_release.go`'s `Release` aliases move to `release_aliases.go`.

- **Mechanics.** `gpu` keeps an alias for each of its 12 exported types and a wrapper for each of its 3 exported
  functions, so callers compile unchanged. A name the staying code uses from the moved code is exported inside the internal
  package, except a field or method of one of the 12 public types: that is reached through a package-level accessor in
  `gpu/internal/kernels` (a field as `*kernels.ContextDevice(c)`, a method as `kernels.ContextEnsureAttn(c, ...)`), so no
  name becomes public. A test helper used across the line is copied. A type-checked rewriting tool makes every edit; its
  rename table is committed with the record.
- **G1, declaration identity (tier 0, no device).** A standard-library tool compares the package at the base with `gpu`
  plus `gpu/internal/kernels` at the head: every base declaration (functions, methods, types, vars, consts, `init`s and
  `_` checks) appears exactly once, and its code, printed without comments, is identical after mapping back the rename
  table, the `kernels.` qualifiers, the accessor calls and the two pre-step conversions. The only other declarations
  allowed are the generated aliases, wrappers and accessors, each in its generated shape, and the copied test helpers,
  each equal to its source. Any other difference fails it.
- **G2, the gpu suite by name (WebGPU on Vulkan, under the lock shared).** `go test -count=1 -p 1 -v -tags 'gpu
  goinfer_testhooks' ./gpu/...` at the base and at the head: every test the base reports has the same outcome (PASS,
  FAIL, or SKIP with the same reason) at the head, wherever it now lives.
- **G3, logits (`go run ./cmd/gate identity <base> <head> -backend webgpu -assets tiny`).** Every family the base runs
  is byte-IDENTICAL; a family NOT RUN on both sides is listed with its reason.
- **G4, towers.** `TestTowerIdentity_webgpu` with `GOINFER_TOWER_ID_OUT` at the base and the head: every case's hash
  equal. The comparison with AX2a's Mac record (`docs/measurements/ax2-2026-10/records-ax2a/webgpu-darwin-arm64.json`)
  is a Mac run; its commands go to mac-ax01.
- **G5, API.** apidiff on package `gpu`: the only incompatible changes are the 12 types' identity (and the signatures
  that name them); no name removed, none added. CHANGELOG names the moved types, the `%T` and reflection change, and no new
  public package.
- **Void:** a device run beside a timed holder (the shared lock refuses it). **Stopping rule:** one run each side; any
  difference stops the move with DECISION NEEDED. **Cost basis:** G1 seconds; G2 measured at the base run; G3 families ×
  steps, stated before it runs; G4 3 s (measured 2026-10-26).
- **Then** the freed tests are already in `gpu/internal/kernels` (they move with the code they test, which keeps their
  bodies unedited), and `internal/layout`'s gpu reach falls by the moved test files.
- **Done (2026-10-27):** G1 PASS (1141 declarations), G2 PASS (281 tests, every outcome and skip reason the same), G3
  PASS (39 families byte-identical on WebGPU, 4 with no tiny checkpoint of their kind), G4 PASS on nobara (15 tower cases;
  the Mac comparison with AX2a's record is mac-ax01's), G5 PASS (the 12 types' identity only). 48 non-test and 75 test
  files moved; gpu reach 172 → 99. Record: `docs/measurements/sb1-gpu-split-2026-10-27/README.md`.

#### SB1 — gpu's kernel layer (item 2 of 21:42): parked, DECISION NEEDED (nobara-sb01, 2026-10-26)

A type-checked declaration graph of package `gpu` (tags `gpu`) shows the shape the split would take. 15 declarations
use `decoder` directly and 37 in all reach it; the other 557 of 594, the runners included, do not. So the cut that frees tests
is everything clean into `gpu/kernels`, with package `gpu` keeping the residency glue (`residency.go`, `kv_slots.go`,
`kvcopy.go`, `lora_resident.go`, `resident_sample.go`, `prefill_scratch.go`, `testhooks_kvscale.go`) and the backend
registration. It would free 80 of the 152 in-package test files. Its cost: 129 files move; 111 unexported names the glue
and the 72 remaining tests use must be exported (32 methods, and test seams such as `g4DropVNormForTest`); `gpu`'s public
API needs a wrapper or alias per moved exported name, and a mutable exported var cannot be aliased without splitting its
state; the four vision-tower files (AX's area) move too. A `metal/kernels`-style move (the WGSL sources only) is small but
frees no test, because every gpu kernel test dispatches through `Context`. Both readings are in the DECISION below; the
item is parked until Francis picks one.

### SB2 — admission and routing

- **Today:** `decoder/admit.go:Admit`, pure over `AdmitFacts`: the backend and precision checks, the runner shape
  (`admitRunner`), the activation-group gate, then `decoder/features.go:residentGateReasons` against `ResidentBackendFeatures`;
  then the backend's own build through `DeclineResident`; the option-path admission grid of 2026-10-08 (`docs/completed/task-option-path-admission-2026-10.md`).
  The owner's note of 2026-10-21 in the forward-architecture doc: 20 of the 71 registration and bound findings counted on
  2026-10-08 were one family that one path did not know about, six of them Critical.
- **Contract:** `Admit(program, request shape, options, backend op set, plan) → Route or Decline`, pure.
- **Fast check:** the whole table, every family × backend × shape × option, in milliseconds, as golden output;
  `docs/capability-matrix.json` and `docs/hardware-matrix.md` generated from it, so they cannot drift from the code.
- **Stops re-running:** finding out in a real run which path a request took. M.

#### SB2 — the admission and routing census, registered 2026-10-27 before the code (nobara-sb01)

Francis's decision (11:08): census first; the pure function must reproduce it exactly; the matrices generated from it must
equal today's committed ones byte for byte; memory questions go through `internal/mem`; no change to what is admitted or
routed (one that appears stops the work with DECISION NEEDED).

Where today's decisions live (surveyed 2026-10-27): the decoder's load path (backend choice, the fit guards, which MM1's
`TestDecisionCensus` already records, the resident gates, KV resolution, the path reports, CPU batching and speculative
eligibility); each backend's `BuildResident` (CUDA alone has about 60 declines, several reading free VRAM or device
attributes); and request-time routing. The census is layered to match, each layer a golden in the module that owns the
decisions:

- **Layer D (decoder; this step, tier 0).** `decoder`'s `TestAdmissionCensus` records, deterministically and without a
  device:
  - **model-free, per registered family (`representativeConfig`, as the hardware matrix does) × webgpu, cuda, metal:**
    the runner-shape decline and every resident gate, for the default activations and for per-32 activations;
  - **per tiny fixture × option cell × backend:** the outcome of a real `Load` with that backend name, a fake resident
    backend standing in for cuda, metal and webgpu that builds whatever it is asked to, so the record is the decoder's
    own decision: the load's error or success, whether the resident build was reached (admitted) or the decline's
    reason, `DecodePath` and the KV precision reports; on cpu also `PrefillPath`, `canBatchN(2)`, the CPU-batch model
    eligibility, `CPUConcurrentSafe` and `specRollbackSafe`. Option cells: Quant int4, int8int8 and f32, each with
    ActQuantGroup 0 and 32; int4 with the `GOINFER_SSM_RESIDENT` knob set; int4 with KVPrecision f16; int4 with KVQuant
    i8. The startup self-test is off for these loads (it runs a backend's kernels on its device: layer B);
  - **backend choice:** `autoBackend` for each `testdata/machines` profile (its own goarch and free bytes) × each
    linked set (cpu; cpu+cuda; cpu+metal; cpu+cuda+metal; cpu+webgpu).
  The golden is the full text, gzipped, one per architecture (`testdata/sb2_admission_census.<goarch>.txt.gz`: the
  CPU's int4 layout is a repack on arm64, a decision that differs by GOARCH), so a diff names the cell; the run is
  repeated in-process and must agree with itself; an architecture with no golden fails and prints its census for
  recording. No reading of the machine enters a line: backend choice reads the profile's goarch and free bytes, and the
  loads read a fixed fake probe (64 GiB, 48 GiB available), so the fit guards proceed.
- **Layers B (later steps, in the backend modules):** each backend's `BuildResident` decisions and device routes, with
  every live read replaced by a profile input through the backend's existing seams: CUDA and WebGPU on nobara, Metal on
  the Mac. Registered here; each gets its own registration of cells before its code.
- **Then the pure function.** `Admit` must reproduce layer D with the census unchanged, and `docs/hardware-matrix.md` and
  `docs/capability-matrix.json` generated from it must equal the committed files byte for byte. **Void:** none
  (deterministic). **Stopping rule:** any change to a recorded decision stops with DECISION NEEDED. **Cost basis:**
  measured: about 9 s for the census built twice (2,711 lines).
- **Layer D landed (2026-10-27).** `decoder`'s `TestAdmissionCensus`, goldens
  `testdata/sb2_admission_census.<goarch>.txt.gz` (amd64 and arm64): 156 model-free lines (52 registered families ×
  webgpu, cuda, metal), 35 backend-choice lines (7 machine profiles × 5 linked sets), and 2,520 load lines (70 committed
  fixtures × 9 option cells × cpu, cuda, metal, webgpu; the 15 that are not decoder checkpoints, towers and encoders,
  record their load error). Only fixtures whose weights git tracks are read, so a checkout with local-only fixtures
  censuses what CI does. Shown red: dropping the q4k condition from the ActQuantGroup gate changes 145 cells (reverted).
  The arm64 golden was recorded from CI's own run of the census at `343630740` (the test prints an unrecorded
  architecture's census and fails): root-linux-arm64, root-darwin and root-windows-arm64 printed the same 2,711 lines,
  and they differ from amd64's in 212 cells, every one only in the CPU's int4 layout note (a repack on arm64).
- **Step 1 landed (2026-10-27): `Admit`.** `decoder/admit.go:Admit` is the decoder's resident admission as a pure function
  of `AdmitFacts` (the arch, the backend's name and whether it can build residents, the quant, -backend auto, the
  activation group, whether the projections are int4, the SSM knob): the residency, precision, runner-shape,
  activation-group and capability gates in the order the load applied them. `Model.residencyGate` gathers the facts and
  calls it; `DecodeRunnerEligible` and `ResidentEligible` read its pieces, and the hardware matrix prints its runner shape
  and capability gates. The census is unchanged on both architectures' goldens, and both matrices regenerate byte for
  byte (`-update` leaves no diff).

#### SB2 step 2 — the backend layers, registered 2026-10-27 before the code (nobara-sb01)

Each backend's own `BuildResident` decides after the decoder admits. Its static gates (shapes, weight kinds, activation
groups, KV layouts) read the model alone; its memory and device decisions read free VRAM, host RAM or device attributes.
The static gates are censused now, on the same fixtures and option cells as layer D, with the machine out of the record:

- **Layer B-cuda (`cuda` module, `-tags 'cuda goinfer_testhooks'`, no device; without `goinfer_testhooks` the file is not built).** `TestBackendAdmissionCensus_cuda`: per committed fixture ×
  layer D's 9 option cells, a `Load` under a fake backend named `cuda` (the decoder's admission is the real one); where
  the decoder admits, the fake runs CUDA's own build in plan-only mode (`cudaBackend.buildResident(m, true)`, which runs
  every static decline and returns before it allocates) and records the decline or the planned allocation slack, then
  declines so nothing is uploaded. Golden `cuda/testdata/sb2_cuda_census.<goarch>.txt.gz`. Runs on CI's deviceless cuda
  job.
- **Layer B-webgpu (`gpu` module, `-tags 'gpu goinfer_testhooks'`, no device; likewise).** WebGPU's static gates move, in their order, into one function
  `BuildResident` calls first (a refactor: same checks, same reasons); `TestBackendAdmissionCensus_webgpu` records it the
  same way. Golden `gpu/testdata/sb2_webgpu_census.<goarch>.txt.gz`.
- **Layer B-metal (the Mac).** Commands in `/tmp/sb2-metal.md`: `residentPreBuild`'s static gates and its memory decisions
  with `metalLiveAvailable` and `metalWorkingSetCap` set from each `testdata/machines` profile.
- **Memory and device decisions on cuda and webgpu.** The context and slot plan is `Model.Plan`, which MM1's census already
  records for cuda and webgpu across the machine zoo. The build's own reads (CUDA's `checkWeightsFit` and `checkKVFits`,
  expert slots, prefill width, SM shape; WebGPU's slot probe) come after allocation and are the next layer: each has a pure
  helper over a free-memory figure, to be fed the profiles' readings.
- **Gate.** Each layer is deterministic (built twice, equal), shown red by a planted change, and changes no decision: the
  WebGPU refactor must leave layer D unchanged and its own census equal to the same census run on the code before it.
  **Cost basis:** measured on the first run; tiny loads, seconds.

**Step 2 read (2026-10-27, nobara-sb01).** Record: `docs/measurements/sb2-backend-census-2026-10-27/README.md`.

- **B-cuda.** 412 cells reach the build, including every cell layer D builds resident. 279 plan and 133 decline. Both builds agree,
  in 1.7 s for the two. The planted MoE per-32 decline turns exactly its 42 cells red.
- **B-webgpu.** The static gates are now `gpu/residency.go:staticResidentDecline`. 218 cells reach it: 207 pass and 11 take the
  quantized-KV decline. The plant turns exactly those 11 red. The census before the refactor (the real build on nobara's device)
  agrees with the one after it on all 218 cells, and the real build gives byte-identical answers before and after.
- **B-metal.** The Mac session's commands are in `/tmp/sb2-metal.md`.
- **The ratchet.** `internal/layout:TestBackendTests_decoderImportsOnlyFall` exempts the three census files by exact path
  (`decoderRatchetExempt`), and every backend's limits stay where they were. This was Francis's decision of 2026-10-27, option 1
  of three, the others being to raise each limit by one or to offset each file by moving out a kernel test. Under any other
  name the same file turns the ratchet red.
- **Next layer.** The builds' memory and device reads, fed from the machine profiles.

#### SB2 — Plan asks Admit, registered 2026-10-27 before the code (nobara-sb01)

Francis's decision (a) of 2026-10-27, option 1. `Model.Plan`'s eligibility check (`planIneligible`) asked `ResidentEligible`,
which is the arch's answer. It reads neither the precision policy nor WebGPU's Nemotron-MoE override, so Plan called Nemotron 3
Nano "not eligible" on webgpu while the load builds it resident. Plan will ask `Admit` with the model's own facts, and the
backend it is asked about.

- **What may change.** Plan's eligibility verdict, only where it disagreed with the decoder's admission. Its pricing does not
  change. WebGPU's lossy-KV mirror (Nemotron, Qwen3.5 and MLA at `--kv f16` or `--kv i8`) is unchanged. The `fit:` message for a
  declined backend names `Admit`'s refusal.
- **Gate 1 (decoder, tier 0).** `TestPlanAgreesWithAdmission` runs on layer D's fixtures and option cells, on cuda, metal and
  webgpu:
  - Plan's verdict comes from a CPU load with the cell's options and the KV request dropped (admission does not read the KV).
  - It must equal whether the decoder's admission let the load reach the backend's build, under layer D's fake backend.
  - Every census line is compared, and a disagreement is red.
- **Gate 2 (gpu, tier 0).** For each cell layer B-webgpu records, Plan's webgpu verdict with the cell's KV request must equal
  `staticResidentDecline`'s.
- **Plant.** The same comparison, fed the old verdict (`MissingResidentFeatures` plus `ResidentEligible`), must go red. It must
  name the Nemotron 3 Nano webgpu cells among its disagreements.
- **MM1's census.** It records Plan, so its golden moves. Every changed line must be a backend and cell where the old Plan
  disagreed with the decoder's admission, and the record lists them. A changed line outside that set stops the work with
  DECISION NEEDED.
- **Out of scope, reported.** CUDA's own build-time declines (`buildResident` in plan-only mode) are not the decoder's admission,
  and Plan does not read them. Layer B-cuda counts the cells where they decline. Closing that is a separate decision.
- **Cost basis:** tiny loads, seconds; the MM1 census is about a minute.

**Read (2026-10-27, nobara-sb01).**
- **Gate 1.** `decoder/plan_admit_test.go:TestPlanAgreesWithAdmission` passes: Plan and the load's admission agree on all 1,890
  cells (70 fixtures × 9 cells × 3 backends). 405 of those cells have no CPU model. They are fixtures no backend loads, and none of
  them reached a build.
- **Plant.** The old arch-only verdict disagrees with the load on 260 cells:
  - 216: per-32 activations on metal and webgpu, where Plan said eligible and the load declines;
  - 8: dense Nemotron-H at int8 or f32 without the knob, where Plan said eligible and the load declines;
  - 4: Nemotron 3 Nano on webgpu, where Plan said not eligible and the load builds it;
  - 2: Phi-3 at per-32 W8A8 on cuda, where the activation hazard is cleared, Plan said not eligible and the load builds it.
- **MM1's census.** It was dumped on main and on this branch and compared group by group.
  - 2,520 lines are wording only: a decline that stays a decline and now quotes `Admit`'s sentence.
  - The groups whose verdict changed are all `testdata/nemotron-tiny` at f32 (D3 on cuda, metal and webgpu, and D4, Metal's
    context fit, which reads the same eligibility) or `testdata/nemotron3nano-tiny` at int4 on webgpu (D3).
  - Every one of them is in the plant's disagreement set, so nothing outside it moved. The amd64 golden is regenerated; it is the
    only class the golden records.
- **Gate 2.** `gpu/sb2_census_test.go:TestPlanAgreesWithLoad_webgpu` passes: Plan's webgpu verdict, with each cell's KV
  request, agrees with the load (the decoder's admission, then `staticResidentDecline`) on all 630 cells. 135 of those cells have
  no CPU model, and none of them reached the build.
  - With main's old `planIneligible` in place, the same test is red on 113 cells.
  - Plan's KV request is the serving flag's precision (`KVPrecision`). The census's `kv-i8` cell sets `KVQuant` alone, which
    webgpu does not deliver.
- **Amendment after Gemini's review of #348 (2026-10-27, the coordinator's relay).** These are within Francis's decision that
  Plan agrees with the load.
  - WebGPU's routed-expert kind check (int8 or int4 only) read the weights alone but ran mid-upload. 16 f32 MoE cells passed
    `staticResidentDecline`, so Plan placed them while the build failed.
    - The check moves into `staticResidentDecline` as a typed decline, reading the new `decoder.Model.MoEExpertKind`. Plan's
      webgpu check mirrors it.
    - Layer B-webgpu's golden changes on exactly those 16 cells.
    - The real build on nobara's device changes on exactly those 16 cells too, from `gpu: MoE residency expert kind "f32"` to
      the decline. Both fall back to the CPU, and every other answer is byte-identical.
    - MM1's census changes in 297 groups, all D3 webgpu at f32 on those 16 fixtures, and all now the expert-kind decline.
  - Gate 1 now holds Plan's admission step (`Model.planAdmitDecline`) to the admission on all three backends. On cuda and metal
    it also checks that Plan's whole eligibility check equals that step. webgpu's mirrors are gate 2's.
  - Gate 2 adds the serving flag `--kv=i8` (both KV options) as a tenth cell, and passes on all 700 cells. Its in-test plant, a
    Plan asked without the KV request, turns 22 cells red, all lossy-KV cells of the families webgpu declines.
  - Both census builders and gate 2 clear `GOINFER_GEMMA4_RESIDENT` too, and unregister their fake backends
    (`decoder.UnregisterBackendForTest`).

### SB3 — family adapters

Moved to VK2 (one file each) and VK4 (program fingerprints). With SB1 and SB2, adding a family is a file, a fixture and
a program fingerprint, and nothing shared changes.

### SB4 — formats and the model store

- **Today:** the GGUF and safetensors readers, the `.giw` container (v17; the reader refuses any other version, per the
  owner's v16 decision), transcodes and `pull`. `decoder/weights.go` (3,106 lines) and `decoder/gguf.go` (3,028) are
  in 42 families' hash set.
- **Contract:** what a loader returns for a file: per tensor, its layout kind and a hash of the loaded bytes.
- **Fast check:** synthetic GGUF and safetensors files written in-test, round trips, per-tensor hashes; on real
  checkpoints, the hashes of the first tensors of each kind.
- **Stops re-running:** forward parity for loader-only edits (identical tensors give an identical forward; the loader
  fingerprint is one input to VK's key).
- **Open question for the format's owner:** whether a layout kind can carry its own version inside the single-container
  rule, so a new kind does not re-transcode every cached `.giw`. M.

### SB5 — engine policies (sessions, prefix reuse, speculative decoding, sampling, constraints)

- **Today:** 26 `spec_*` test files and the session and reuse tests in `decoder` run through loaded fixtures;
  `internal/sampling` and `internal/kvmgr` are already packages of their own.
- **Contract:** each policy runs over a `Forward` interface; an exact policy produces the same token stream as plain
  decoding.
- **Fast check:** a fake model with scripted logits: acceptance patterns, rollbacks, branch points. M.

### SB6 — the scheduler

- **Today:** queueing, fair admission, jobs and batches live in the HTTP package (`internal/serveapp/batches.go`,
  `internal/serveapp/sessions.go` and the job files; `docs/tasks/task-work-queue-2026-09.md`, J1–J9).
- **Contract:** a policy over a queue and an engine's capacity: admission order, fairness, priorities, deadlines,
  prefix-aware placement (J6), and later RF's batch building.
- **Fast check:** a simulator on a virtual clock over recorded arrivals. M.

### SB7 — the serving shell

- **Today:** `internal/serveapp`, 76 source files and 19.9k lines; `main.go` took 116 commits in 30 days; 39 of its
  162 test files import `decoder`.
- **Contract:** an `Engine` interface (generate, stream events, cancel, stats, models).
- **Fast check:** a fake engine; the OpenAI and Anthropic shapes and SSE checked with no model. The `sdktest` module
  stays the end-to-end check. M.

### SB8 — the conversation codec

- **Today:** `chat` (templates), `tokenizer`, `constrain`, and tool-call parsing in the serving package.
- **Contract:** messages to tokens to tool calls and back, per family template.
- **Fast check:** golden renders and round trips per template. Its open contract question is
  `docs/tasks/task-tool-grammar-union-2026-09.md`'s: two or more tools under `tool_choice` auto. S–M.

#### SB8 — the codec census, registered 2026-10-27 14:25 PDT before the code (mac-ax01)

Francis's decision (14:16): census first, recorded at `origin/main`; the move to one package must reproduce every golden
byte for byte; tool-call parsing gets its own table of model outputs and parsed calls, malformed ones included; nothing
about what a model sees may change (a change stops the work with DECISION NEEDED).

The census is a test in `chat` (it pins the clock Llama 3's and Harmony's date lines read, which only the package can) that
calls only the API the serving routes call, in the order they call it. It adds test files and goldens under
`chat/testdata/codec/` and changes no other file, so what it records is `origin/main`'s code.

- **Template sources.** Every distinct chat template the repository tracks (`testdata/chat_goldens/` and
  `testdata/chat_think_goldens/`), every chat template a local model under `~/models` carries that `Detect` accepts
  (snapshotted into the census's testdata, so CI renders them too), and each family's constructor as the bare-checkpoint
  fallback returns it. Each source is resolved through `Detect` exactly as a load does.
- **Variants per source:** each `ThinkMode` where the template has a reasoning spec, each `ToolFormat` where it declares a
  native tool form, each reasoning effort for Harmony.
- **Conversations,** fixed: system plus user; user alone; multi-turn with non-ASCII text; a replayed assistant reasoning
  turn; tools offered; a tool call and its result; two calls and two results; a finished tool loop and a follow-up; a user
  message carrying the family's own control-token text (it must stay text); an image placeholder where the family has a
  block (Gemma 3, Gemma 4, Qwen-VL ChatML, GLM-OCR, Pixtral, LFM2-VL) and an audio placeholder for Gemma 4, built and
  spliced as `serveapp` builds them.
- **Recorded per case:** the segments (text and the Special flag), the rendered string's equality with their concatenation,
  the family name, its stop strings, its tool-call opener, whether the prompt opens a think block; and, with the source's
  tokenizer, the token IDs (`EncodeSegments`, no added BOS, as the routes encode), the stop strings' IDs, and whether the IDs
  decode back to the rendered text.
- **Tokenizers.** Tracked ones (Qwen2.5-Coder's GGUF metadata, Mistral v0.3's `tokenizer.json`) are checked everywhere; a
  local one is named by its `~/models` path and SHA-256: present with that hash, its IDs are checked; present with another,
  the test fails; absent, the test logs a named skip and prints how many ID checks ran and how many were skipped. Llama 3,
  Mellum 2 and Harmony tokenizers are not on the Mac: their IDs are recorded on nobara by the same `-update`, which keeps
  recorded IDs whose tokenizer is absent, and that commit's diff must add IDs only.
- **Tool-call parse table** (its own golden): about 40 model outputs across every family's dialect (Hermes JSON, Qwen 3.5
  XML, Mistral, Llama 3, Gemma 4, Harmony, fenced, bare), well-formed and malformed (bad JSON, unterminated, unknown tool,
  arguments of the wrong type or not an object, a call inside a think block, prose that mentions the opener), each through
  `ParseToolCalls` and `ParseToolCallsFor` with a fixed tool list, for each family and tool format, lenient off and on.
- **Tier 0 throughout.** The goldens are JSON, gzipped if one exceeds 1 MB. **Void:** none (deterministic). **Shown red:** a
  planted one-byte change to a template and to a parser must fail the census, then be reverted. **Stopping rule:** a census
  that differs from its golden after the move stops with DECISION NEEDED; there is no tolerance. **Cost basis:** seconds;
  the Gemma tokenizers load in about a second each.
- **The move** (one package for the codec) is registered after the census lands, with its package shape; `chat`,
  `tokenizer` and `constrain` are in apidiff's Hard tier, so a shape that changes their exported API is the owner's call.
- **Census recorded (2026-10-27, the Mac, at `origin/main` `92a64e92`).** `chat/codec_census_test.go`:
  `TestCodecCensus`, one golden per source under `chat/testdata/codec/sources/` (gzipped JSON, 4.8 MB as text, 0.2 MB
  stored), and `TestToolCallParseCensus`, `chat/testdata/codec/tool_call_parse.json`. 55 sources: 19 tracked templates,
  5 tracked tokenizers, 20 local models, 11 constructors; 1,780 render cases and 715 token-ID checks on the Mac, of which
  40 (the tracked tokenizers) run anywhere, the rest only where the model is on disk. Without the local models every
  render is still checked: Detect and the renderers ask the vocabulary about 6 to 13 markers per source, and the answers are
  recorded with it. The vocabulary fingerprint is a SHA-256 over every token's text in ID order, not the file's hash: a
  GGUF is the weights too. It takes about 3 s with the local tokenizers and under 1 s without. Recording twice gives
  identical files. **Shown red:** a one-byte change to Gemma 3's generation prompt fails 5 sources (the tracked template,
  the constructor, 3 local models); dropping the lead's trim in the Hermes parser fails the parse table. Both reverted.
  What the census shows today, recorded as it is, unchanged: Phi-3's GGUF tokenizer does not round-trip (its IDs decode
  without the newline after each control token); `granite-dbrx-meta.gguf`'s template is declined (raw-completion
  fallback). **Owed:** the Llama 3, Mellum 2 and Harmony token IDs, recorded on nobara by
  `go test ./chat -run TestCodecCensus -codec.update`, whose diff must only add sources and IDs.
  Each source renders the nine conversations above plus its family's image and audio placeholders, 9 to 11 cases per
  variant. Token IDs are on the 24 sources whose tokenizer is there and whose template `Detect` accepts (the tracked
  tokenizers but `granite-dbrx-meta`, and the local models); the tracked templates and the constructors carry renders only.
- **Gemini's review of #338, answered (2026-10-27, mac-ax01).** The vocabulary probes are now checked and recorded on
  every return, including a template `Detect` declines. `-codec.update` runs the local-model guard (vocabulary hash and
  chat template) on a model recorded before, so a different quant or template at the same `~/models` path fails instead
  of replacing another machine's record. A tokenizer that fails to load for any reason but a missing file fails the
  census. Both guards were shown red in check and update mode, with a corrupt `tokenizer.json` and another model's
  tokenizer under a recorded name, and no golden was rewritten. The parse table adds the five tracked templates that
  declare a native tool form, each under the three tool formats (37 entries, from 22). It records `supports_tools` and
  asserts the contract directly: a family without tool support reads no call from anything (Gemma 3, GLM-OCR, both
  Phi-3s, and Ministral, whose `[TOOL_CALLS]name[ARGS]` form `Ministral()` documents as not wired); each family with it
  reads every well-formed call of its own dialect; no family reads a call from prose, a malformed call, an empty or
  missing name, a Harmony final answer, or Ministral's form; and the tool format does not change a parse. The 22 earlier
  entries are unchanged, row for row. **Shown red:** dropping `mistral` from `SupportsTools` fails the contract;
  accepting an empty name fails it for ChatML and Llama 3. That second plant passed before the review, because no output
  had an empty name, so the table gained one (43 outputs). Both reverted.
- **nobara's token IDs, recorded (2026-10-27 15:57 PDT).** On nobara (Go 1.27.3 linux/amd64), from a detached worktree
  of `16e21d7f`, under the shared timing lock (load 1.3; the lock had been free since 15:56):
  `go test ./chat -run 'TestCodecCensus$' -codec.update -codec.models '^(llama-3\.2-1b-instruct|mellum2|gpt-oss-20b)'`.
  `-codec.models` limits discovery to the owed families; nobara holds about 80 models. The run added five sources and
  changed no recorded file: `llama-3.2-1b-instruct-q4_k_m.gguf` (Llama 3), `mellum2-unq` and `mellum2.1-thinking`
  (Mellum 2, the second with its four think modes), and gpt-oss-20b as both its HF `tokenizer.json` and its MXFP4 GGUF
  (Harmony, three efforts). That makes 108 token-ID cases, every one round-tripping. The same run also checked 680 of
  the Mac's recorded IDs against nobara's copies of the 18 Mac models it also holds: their vocabulary, template and
  every ID matched. The census has 60 sources; the Mac checks 715 IDs and skips the 108 recorded on nobara, and renders
  all 60.

#### SB8 — the move to `internal/codec`, registered 2026-10-27 16:10 PDT before the code (mac-ax01; Francis's option 1)

Francis chose option 1: a new `internal/codec` owns the path from messages to token IDs and back, with `chat` and
`tokenizer` as its parts, their public APIs unchanged. Three pull requests, each green on its own:

- **C1, the encode half.** `internal/codec` gains template detection from a tokenizer (the one `Detect` call the
  routes make), a prompt's segments (with tools or without), the image and audio block splice (one block or several),
  and their encoding (`EncodeSegments` with no added BOS, as every route encodes). No caller moves. The census gets a
  seam for its render, splice and encode steps; `TestCodecCensus_throughCodec` (an external test of `chat`, so it can
  import the codec) runs the whole census through `internal/codec` against the same goldens.
- **C2, the callers' encode path.** `serveapp` (the text, tool, overflow and vision routes), `chatapp` (text, image,
  batch) and `internal/decide` call the codec where they composed `Detect`, render, splice and encode themselves.
  `serveapp`'s carried-ID regions (`encodeRegions`) and its KV-boundary encodes encode pieces of a prompt, not a prompt,
  and stay where they are.
- **C3, the reply half.** The codec gains the reply reader: reasoning splitting (`ReplySplitter`), settling a finished
  turn's tool calls (`settleCalls`: the inline parse, a Harmony router's calls, a call read from the reasoning when there
  is no answer) and reconciling streamed prose (`reconcileProse`), moved from `serveapp`; the census's reply and
  parse-table steps get the same seam, and `serveapp` and `chatapp` call the codec.
- **Out of scope, by design.** `demo/agent` (its own module: a standalone build resolves the root at its last tag, which
  has no `internal/codec`) and `examples/` (they show the public API) keep calling `chat` and `tokenizer`.

Gates, per pull request:

| gate | rule | void | stopping rule | cost basis |
|---|---|---|---|---|
| G-CX-goldens | `TestCodecCensus` and `TestToolCallParseCensus` pass with no file under `chat/testdata/codec/` in the diff | none: deterministic | any golden change stops the move with DECISION NEEDED | seconds |
| G-CX-arm | the census through the codec passes on every source, variant and case the census holds (renders, IDs, and from C3 the replies and the parse table), and a planted change in the codec (an added BOS) fails it, then is reverted | none | as above | seconds |
| G-CX-api | no non-test file under `chat/`, `tokenizer/` or `constrain/` changes; CI's `apidiff_check.sh` reports nothing new | none | a change stops with DECISION NEEDED | seconds |
| G-CX-callers (C2, C3) | each replaced call site composes the same calls in the same order as before (read in review against the codec's body); `touched.py` green; CI's tests of `serveapp`, `chatapp` and `decide` pass unchanged, no golden of theirs in the diff | none | a red test is fixed in the move or the move is reverted, never a golden re-recorded | `touched.py` seconds; the rest is CI's |

- **C1 landed on its branch (2026-10-27, the Mac).** `internal/codec/codec.go`: `Template`, `Segments`,
  `ImageSegments`, `ImagesSegments`, `Encode`, `EncodePrompt`. The census's render, splice and encode steps go through a
  `censusPath`; `directPath` makes the same calls as before, and `chat`'s `TestCodecCensus_throughCodec` (an external
  test, `chat/census_through_codec_test.go`, reaching the census through `chat/export_test.go`) runs it through the codec.
  **G-CX-goldens:** pass, no golden in the diff. **G-CX-arm:** pass on all 60 sources, with the local models (715 IDs)
  and without (renders only); an added BOS in `codec.Encode` fails 16 sources (those whose tokenizer has a BOS token),
  reverted. **G-CX-api:** no non-test file under `chat/`, `tokenizer/` or `constrain/` changed. The layout data records
  the new arrows (`internal/codec` to `chat`, `multimodal`, `tokenizer`; `chat`'s test to `internal/codec`), and
  `scripts/touched_scans.py` runs the arm when `internal/codec`, `tokenizer` or `multimodal` changes.
- **C2 landed on its branch (2026-10-27, the Mac).** Eleven sites call the codec. In `serveapp`: `promptForT`,
  `encodePrompt`'s render and whole-prompt encode, the tools route, `fixedPromptTokens`, `encodeVisionSegments`,
  `imagesPrompt` and the model load's `Detect`. In `chatapp`: the session's `Detect`, the batch encode and the image
  prompt. In `decide`: `NewPlainTokenizer` and `EncodeChat`. Each replaced expression is the codec function's body
  (`RenderToolsSegments` with no tools delegates to `RenderSegments`, so `Segments` is the same call where the tools
  route passed none). **G-CX-callers:** `touched.py` green; the `serveapp` tests named for the changed paths (vision,
  tools, overflow, carried IDs, think, prompt, image, audio, and the per-family image tests) pass, 167 run and 11
  skipped (real checkpoints or env-gated assets); no test or golden changed. **G-CX-api:** still no non-test change under
  `chat/`, `tokenizer/` or `constrain/`.
- **C3 landed on its branch (2026-10-27, the Mac).** `internal/codec/reply.go`: `NewReplySplitter`, `SettleCalls`,
  `ReconcileProse`, `ErrProseDiverged`. `serveapp`'s `settleCalls` and `reconcileProse` are one-line delegates (its tests
  call them), `errProseDiverged` is the codec's error, and the three reply-splitter constructions (`newThinkOut`,
  goinfer-chat's turn and batch) call the codec. `SettleCalls` takes the router's calls and reasoning as values: the
  getters it replaces (`thinkOut.toolCalls`, `reasoningText`) return none for a nil router, which leaves the fallbacks
  unreachable exactly as the `th != nil` test did. The census's reply and settle steps get the seam;
  `TestToolCallParseCensus_throughCodec` checks the settle column through `codec.SettleCalls` with no router.
  **G-CX-arm:** all four census tests pass, with the local models and without. Shown red, each reverted: settling without
  the tool list fails the parse arm; no reply splitter fails the census arm; an untrimmed lead on a routed (Harmony) call
  is outside the census (it has no router) and fails `serveapp`'s `TestSettleCalls_harmony`. **G-CX-callers:**
  `touched.py` green; `serveapp`'s reply, tool, think and prose tests pass, 73 run and 5 skipped. **G-CX-api:** no
  non-test change under `chat/`, `tokenizer/` or `constrain/`.

### SB9 — the towers

After AX2: each image and audio tower with per-tower goldens (output bytes on fixed inputs), and later its blocks
expressed as programs, so SB1's kit covers them on every backend. S, then M.

### SB10 — configuration

- **Today:** `decoder.Options` (30 fields), the 42 `GOINFER_*` reads listed in `testdata/env_reads.txt`, and the flags in
  `internal/loadflags` and `docs/flags.meta.json`.
- **Contract:** one typed schema, and `Resolve(flags, env, defaults, machine)` returning the resolved settings, pure.
- **Fast check:** table tests; the docs generated; `TestEnvVars_docAndCodeAgree` generalized to flags and options. M.

### SB11 — cost counters and telemetry

- **Contract:** per operation and per phase: bytes read and written, kernel launches or command buffers, syncs,
  allocations, time. One schema for tier-0 tests, gates and the wire (X1, `docs/tasks/task-turn-telemetry-2026-09.md`).
- **Fast check:** counters on tiny fixtures, asserted exactly. The program walker (OP4) emits them for every backend. M.

### SB12 — gate and measurement infrastructure

- **Today:** `cmd/gate` (18,847 lines of Go, 13,321 of them non-test) and `scripts/` (289 Python files, touched by 426
  commits in 60 days).
- **Contract:** runners launch, lock and record; graders are pure functions of records.
- **Fast check:** graders run on recorded fixtures with plants, as the CUDA device-suite judge already does with fake
  streams. M.

### SB14 — documentation that cannot drift

§4.1's four homes, held by one test: each invariant named in a `doc.go` points at a test that exists, and the
architecture doc's import table is the import gate's data. It is `CLAUDE.md`'s rule that a doc comment claiming coverage
is not coverage, made mechanical. S.

#### SB14's gates, registered 2026-10-26 before the code (mac-vk01)

Francis (21:42, relayed): SB14, then PW4, overnight, tier 0. The test lives beside SB13's, in `internal/layout`, and
reads every package doc comment in the tree, `doc.go` or not.
- **Every test a package doc names must exist.** A name with a package before it ("decoder's TestDecisionCensus") must
  be in that package; a bare name must be in the doc's own package.
- **Every line of an "Invariants" block must name a test.**
- **`docs/ARCHITECTURE.md` gains a generated appendix between markers,** built from `internal/layout/imports.json`: one
  row per importing package with its imports and its test-only imports, then the protected rules. The test fails when
  the appendix differs from the data; `-update` regenerates it.

Measured before the code: 8 package docs name tests today, and every named test exists.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| D1 today's tree | the doc and table tests pass on main's tree once the appendix is generated | never | one run | seconds |
| D2 drift in a doc | `gate mutation` renaming a test that a `doc.go` names turns it red, naming the doc and the test; one adding an invariant line that names no test turns it red; one naming a test in the wrong package ("decoder's TestSolve_pure") turns it red | never | one run each | seconds |
| D3 the table is the data | `gate mutation` adding an arrow to `imports.json` without regenerating turns the table test red; one editing a row of the appendix by hand turns it red | never | one run each | seconds |

### SB13 — the package layout, as a test

§9's table as a test that reads `go list -deps` for every module and fails on a forbidden edge, with ratchets (backend
test files importing `decoder` may only fall). S.

#### SB13's gates, registered 2026-10-26 before the code (mac-vk01)

Francis (19:52, relayed): SB13 while VK5's shadow week runs. The data is `internal/layout/imports.json`, which SB14
and `docs/ARCHITECTURE.md` can read. It holds two things:
- **The arrows.** Every import from one goinfer package to another, in every module, as repository-relative package
  paths. Non-test and test files are kept as two lists.
- **The protected separations, as rules:**
  - `internal/vision` and `internal/audio` import no goinfer package and, outside the standard library, only aikit's
    `linalg` and `embed`.
  - The two gpudev halves (`cuda/internal/gpudev`, `metal/internal/gpudev`) import no goinfer package.
  - `cuda/kernels` and `metal/kernels` never import `decoder`, in tests either.
  - `internal/mem/probe` imports no goinfer package; that it is the only OS memory reader stays MM0's
    `TestOnlyProbeReadsOSMemory`.
  - No root-module package imports the `cuda`, `metal` or `gpu` modules or aikit's GPU modules, and the root `go.mod`
    requires none of them.

`TestLayout_arrows` parses imports the way the existing decoder ratchet in the same package does (no `go list`, so
seconds, across all modules). It fails on:
- an arrow the data lacks: a new arrow, named with the file that makes it;
- an arrow the data has but the tree no longer makes: the data must be lowered to match;
- any arrow, made or recorded, that breaks a rule.

So an arrow, once removed from the data, comes back only through an edit to the data, and an arrow that breaks a rule
cannot be put back at all. nobara-sb01's `decoderRatchet` in the same package is not touched.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| L1 today's tree | `TestLayout_arrows` passes on origin/main's tree with the data written from it; every protected rule holds today | never | one run | seconds |
| L2 a planted forbidden import | `gate mutation` adding a `decoder` import to a `cuda/kernels` test file turns it red and the message names `cuda/kernels → decoder` and the rule; one adding an unrecorded but permitted arrow turns it red and names the arrow | never | one run each | seconds |
| L3 the ratchet only falls | `gate mutation` deleting a recorded arrow from the data while the tree still makes it turns it red (a removed arrow cannot be re-added without editing the data); one recording an arrow the tree does not make turns it red; one recording `internal/vision → decoder` in the data turns it red as a rule violation even with no code making it | never | one run each | seconds |

## 9. The target layout

| package | owns | may import |
|---|---|---|
| `decoder` (public) | the facade: `Load`, `Model`, `Options`, generation; the per-family adapter files (VK2) | everything in this table except the backend modules |
| `internal/program` (out of `decoder` when SB1 needs it) | the operation list and the CPU reference | aikit `linalg` |
| `internal/backendapi` (SB1) | the backend interface | `internal/program`, `internal/mem` |
| `internal/mem`, `internal/mem/probe` (MM) | planning, the ledger, the arbiter; the probes | the standard library (the probes also `golang.org/x/sys`) |
| `internal/kvmgr` | KV blocks and the prefix index | the standard library |
| `internal/gpudev` (AX) | the device substrate | `golang.org/x/sys`, purego |
| `internal/vision`, `internal/audio` (AX) | the towers | aikit `linalg` and `embed` |
| `multimodal` (public) | the goinfer side of each tower | `internal/vision`, `internal/audio` |
| the `cuda`, `metal` and `gpu` modules | backends | `internal/backendapi`, `internal/program`, `internal/mem`, `internal/gpudev`, aikit; **never** `decoder` once SB1 lands |
| `internal/scheduler` (SB6) | the scheduling policy | the standard library |
| `internal/serveapp`, `internal/chatapp`, `cmd/*` | the shell | `decoder` through an `Engine` interface (SB7) |

## 10. Performance: counters per change, timed runs once a day

- Most performance questions about a change are questions about counts: bytes moved, launches, syncs and allocations
  per token (SB11). They are answered at tier 0.
- A kernel's speed is an in-process interleaved A/B (TE5(b): every resolved kernel-local pair agreed in sign with its
  served A/B, 37 of 37 same-session and 45 of 45 overall,
  `docs/measurements/test-efficiency-2026-09/te5b-concordance-2026-09-28.md`). It is carried to end to end with the
  counters' category shares; those projections held within 1.5% (`docs/completed/task-prefill-gap.md` §6.1).
- Served and peer gates become the daily calibration on a quiet box: a fixed cell set, the existing rules (A B B A, the
  void rule, two arms for a new-over-old ratio), a trend over days, and a step in the trend bisected with the counters
  first.
- A change whose speed cannot be judged without a served gate is the modularity bug of `CLAUDE.md`'s "Day loop".

## 11. The daily run

- **What runs:** VK6's canary and coverage audit; the performance calibration cells; MM's plan against measured on
  each box; the product smoke (serve, the SDK tests, a tool-call turn, the web UI's fetch); tower identity spot checks.
- **Where and when:** both boxes at night, on the night queue (SD7, decided 2026-10-26), within its limits (no job over
  3 hours, none started that would end after 06:30).
- **Nobody waits on it.** It writes a record and a short summary; a failure opens a queue entry; no session holds a turn
  on it.

## 12. Track PW — the waiting

- **PW1 (SD4, decided and applied 2026-10-26).** Applied: branch protection on `main` requires 19 checks (`changes`,
  `lint`, the five decoder shards, `test-rest`, `readme-smoke`, `gpu`, `sdktest`, `cuda`, the four `root-*` jobs, the
  arm64 forced-fallback job, `gpu-darwin`, `metal-darwin`), not strict, with admins not enforced so the owner's account
  can still push docs-only changes; "Allow auto-merge" is on. The three matrix forced-fallback jobs are left out
  because a docs-only run reports them under the unexpanded name `forced fallbacks (${{ matrix.force }})`, which would
  leave a required check waiting forever. Left to do: the rule for sessions and the guard's retirement. The design as
  scoped: turn on auto-merge for goinfer (available for public repositories on GitHub Free) with a branch rule
  or ruleset on `main` that requires the fast checks (lint and the Linux test jobs), and let the owner's account push
  docs-only changes directly. A session runs `gh pr merge --auto` and moves on. Main's own CI still runs once per merge
  and a push to main does not cancel the run before it (`.github/workflows/ci.yml`'s concurrency rule), so a red run
  points at one PR, which is the purpose of `scripts/main_ci_guard.py`, served without anyone waiting. Retire the
  guard's wait when PW1 is on.
- **PW2.** Run `root-darwin`, `root-windows`, `root-windows-arm64`, `root-linux-arm64`, `gpu-darwin` and
  `metal-darwin` after merge, or only when their paths change, and keep them out of the required checks. The
  40–45-minute runs were these runners queueing (§1.4). They still gate tags through `gate release`.
  **Decided 2026-10-27 (Francis, 13:19): the OS jobs stay required; revisit if one job becomes the long pole by several
  minutes.** After PW2a, the five PR runs' last jobs finished 11.5–17.1 minutes in (median 14.6), with the last job
  varying between `root-windows`, `test-rest`, `test-decoder-d` and `gpu-darwin` according to which runners queued,
  so taking the Windows jobs out would save 0–2 minutes (`docs/measurements/ci-tail-2026-10-27.md` §4a).
- **PW2a (2026-10-27, Francis 11:14).** `test-decoder-tail` finished 19–22 minutes into a PR run, against 11–12 for the
  Windows jobs: MM1's two census tests (`TestDecisionCensus`, `TestDecisionCensus_secondProcess`) landed in it and take
  658 s of its 1,130 s under -race (`docs/measurements/ci-tail-2026-10-27.md`). They move to their own steps in
  `test-decoder-a` and `test-decoder-d`, whose sampling tests take one to three minutes. Gates in §12.2.
- **PW3.** Shard or generate the hot files. The citation index records each cited commit's subject in one file, and
  every new citation edits it: keep the check, and store the expected subjects per citing doc. CHANGELOG fragments per
  change, assembled at release. Parity records per family (VK3). The `docs/README.md` counts written by
  `scripts/task_docs.py`, which already recounts them (on 2026-10-26 it found the design-record count one short).
  **PW3, measured and gated (registered 2026-10-27 11:11 PDT by mac-ax01, before PW3's code; Francis 11:08: "next for you is PW3"):**

  | file | commits, last 30 days | last 7 days | what drives them now |
  |---|---:|---:|---|
  | `docs/citation-index.md` | 240 | 1 | since CC0 closed the door (2026-10-09) it holds only `docs/QUEUE.md`'s commit and bare-file citations: 2 commits since |
  | `CHANGELOG.md` | 280 | 12 | an entry per user-visible change |
  | `docs/README.md` | 90 | 9 | a doc's entry and the two counts `task_docs.py` rewrites |
  | `testdata/parity_manifest.json` | — | — | gone: VK3 split it into `testdata/parity/` |

  - **`docs/citation-index.md`: no change.** It stopped being hot when the path index went (CC0); sharding it would add
    machinery for two commits a fortnight.
  - **`CHANGELOG.md`: a decision for Francis.** Fragments per change, assembled at release, change how the file reads on
    `main` between releases.
  - **`docs/README.md`'s counts: removed from the file.** Two PRs that each add a task doc both rewrite the count to N+1.
    Git merges that without a conflict, because the edits are identical, and the result is one short. An add beside an
    archive conflicts. The two headings drop their numbers, and `task_docs.py check` reports the counts from the tree
    instead of comparing them with the README.

  | gate | rule | void when | stopping rule | cost basis |
  |---|---|---|---|---|
  | PW3-R1 (parallel adds merge) | In a scratch clone, two branches off the same base each add a design record the way a session does (the doc, then `task_docs.py`'s recount where it applies). Merged one after the other: at the base, the README's count comes out wrong (or the merge conflicts). After the change, both merges are clean and `task_docs.py check` is clean. | — | once each side | seconds |
  | PW3-R2 (nothing else moves) | `task_docs.py check` exits 0 and prints the counts it used to compare. The README's only diff is the two headings' numbers. `archive` still runs. The citation and book-link lints pass. | — | complete | seconds |

  **PW3 results (2026-10-27, mac-ax01):**
  - **PW3-R1 PASS.** In a scratch clone, two branches off one base each added a design record and recounted the way a
    session did. At the base both merges were clean and the README said 71 design records against the tree's 72. After the
    change both merges are clean and `task_docs.py check` counts 72 from the tree; no stored count is left to go wrong.
  - **PW3-R2 PASS.** `docs/README.md` changes only in the two headings. `check` exits 0 and prints 70 (61 + 9) and 140, the
    figures the README carried. `--strict` exits 1 at origin/main too, for other sessions' uncommitted docs. The citation,
    book-link and touched-scan checks pass.
  - **Left outside this repo's code:** the doc-review skill (`.claude/skills/doc-review/SKILL.md`) still tells an
    archiving session to recount the README. That file is under `.claude`, which this session does not write.
  - **`CHANGELOG.md` waits for Francis** (DECISION NEEDED in mac-ax01's report).

  **PW3, the CHANGELOG: fragments (Francis, 12:14: option 1; gates registered 2026-10-27 12:16 PDT by mac-ax01, before the code).**
  - **Design.** Each change adds one file to `changelog.d/`, named `YYYY-MM-DD-<slug>.md`, holding its entry exactly as it
    would have appeared in `[Unreleased]`. `CHANGELOG.md`'s `[Unreleased]` becomes one line pointing there.
    - `scripts/changelog.py render` prints the entries, newest first (descending by file name), joined verbatim.
    - `scripts/changelog.py release vX.Y.Z --date D` writes `## [vX.Y.Z] — D` with that text under `[Unreleased]` and
      deletes the fragments: today's manual `[Unreleased]` → `[vX.Y.Z]` move, done by the script.
    - `gate changelog` (and so `gate release`'s CHANGELOG row) reads `[Unreleased]` as the pointer line plus every fragment.
    - RELEASING.md's step says to run the script. Today's 33 entries move into fragments whose names keep their order.

  | gate | rule | void when | stopping rule | cost basis |
  |---|---|---|---|---|
  | PW3-C1 (today's text, byte for byte) | `scripts/changelog.py render` on the migrated fragments equals origin/main's `[Unreleased]` body byte for byte | — | complete | seconds |
  | PW3-C2 (a release reads as today) | On a scratch copy, `release v0.24.0 --date 2026-10-27` gives a `[v0.24.0]` section whose body is origin/main's `[Unreleased]` body, byte for byte, under the header today's move writes. It leaves no fragment behind. `gate changelog -section v0.24.0` reports what origin/main's `gate changelog` reports for `[Unreleased]`, line for line after the section name. | — | complete | seconds |
  | PW3-C3 (parallel adds merge) | Two branches off one base each add a fragment. Both merge cleanly, and `render` and `gate changelog` see both entries. At origin/main the same two adds, made in `CHANGELOG.md`, conflict. | — | once each side | seconds |
  | PW3-C4 (nothing else moves) | `gate changelog` with no flags reads the same at this branch as at origin/main. `cmd/gate`'s changelog and release tests pass, as do the script's own tests (`scripts/test_changelog.py`), `touched.py`, the citation lint and `test_touched.py`. | — | complete | minutes |

  **PW3 CHANGELOG results (2026-10-27, mac-ax01):**
  - **PW3-C1 PASS:** `render` on the 33 migrated fragments equals origin/main's `[Unreleased]` body, 17,135 bytes.
  - **PW3-C2 PASS:** in a scratch clone, `release v0.24.0 --date 2026-10-27` writes a `[v0.24.0]` section whose body is
    that same text byte for byte, under `## [v0.24.0] — 2026-10-27`. It leaves no fragment, and `gate changelog
    -section v0.24.0` reads as origin/main's `gate changelog` did (the same eight owed items).
  - **PW3-C3 PASS:** two parallel fragment adds merge cleanly and `render` lists both; the same two adds made in
    `CHANGELOG.md` conflict.
  - **PW3-C4 PASS:**
    - `gate changelog` reads identically to origin/main. With `changelog.d/` moved aside it would owe one more item
      (`POST /admin/models/fit`), so it does read the fragments.
    - `touched.py` is green, `cmd/gate`'s 79 s run included.
    - `scripts/test_changelog.py` passes 6 tests and runs in CI. The citation lint and `test_touched.py` pass.

  **PW3 review of #330 (Gemini; coordinator 12:49; registered 2026-10-27 12:50 PDT by mac-ax01, before the code).** Four
  findings, all real.
  - (1) A `changelog.d/` change voids a quick-full record: `quickRecordVoidedBy` treats every path with a `/` as voiding.
  - (2) An entry under `CHANGELOG.md`'s `[Unreleased]` passes `gate changelog` and is stranded at release.
  - (3) Only a synthetic repo exercises `render`'s name and format rules; the live `changelog.d/` never meets them before a
    release.
  - (4) `render` accepts a file holding two entries.

  Gate PW3-R330, one rule per finding, each shown red first and green after, with a test or a planted input:
  - (1) `quickRecordVoidedBy("changelog.d/x.md")` is false, and `decoder/x.go` still voids.
  - (2) `gate changelog` (and so `gate release`'s row) owes an entry left under `[Unreleased]`, and `scripts/changelog.py
    release` refuses one.
  - (3) CI runs `render` on the live tree.
  - (4) `render` refuses a two-entry file.

  The checks: `cmd/gate`'s tests, `scripts/test_changelog.py`, `touched.py` and the lints. Cost: seconds. Stopping rule: complete.

  **PW3-R330 results (2026-10-27, mac-ax01): all four PASS, each shown red first.**
  - (1) `quickRecordVoidedBy` read `changelog.d/…` as voiding; now inert, and `TestQuickRecord_whatVoidsARun` has the case. RELEASING.md
    names the path.
  - (2) A stray entry under `[Unreleased]` now owes in `gate changelog`. A plant in the real CHANGELOG read `OWED entry - **a
    planted stray entry**`, and `TestUnreleasedStray` holds the rule. `release` refuses one too
    (`test_release_refuses_an_entry_left_under_unreleased`).
  - (3) CI's step now runs `render` on the live tree.
  - (4) `render` refuses a two-entry file (`test_a_fragment_with_two_entries_is_refused`).

  `gate changelog` on the clean tree reads exactly as before. `touched.py` is green (`cmd/gate` 81 s), and the lints and
  `test_touched.py` pass.

- **PW4.** `night.py wait <job>`: block until a queued job ends and exit with its status. A session on the other box
  then runs one background `ssh` and gets one completion notification, in place of sleeping and polling over ssh.

  **PW4's gates, registered 2026-10-26 before the code (mac-vk01; Francis 21:42, relayed).** `night.py wait NAME`
  [`--timeout MIN`] follows one queued instance of the job: the one it finds queued or running, by its `seq`, through
  its re-queues (a retry or a stop) until its final record. It then prints one line with the job's outcome and log, and
  exits:
  - 0 when the outcome is `ok`;
  - 1 for any other final outcome (FAILED, timeout, VOID-refused);
  - 2 when no such job is queued, running or finished;
  - 124 when its own `--timeout` passes first.

  A job already finished when `wait` starts returns its newest record at once. Nothing else in `night.py` changes, and
  the tests run in a temporary `NIGHT_DIR`, never either box's live queue.

  | gate | rule | void when | stopping rule | cost basis |
  |---|---|---|---|---|
  | W1 backward-compatible | `scripts/test_night.py`'s existing tests pass unchanged; no other subcommand's output or file changes | never | one run | seconds |
  | W2 the semantics | new tests drive the real `Runner` in a thread under a temporary `NIGHT_DIR`: a job that ends ok → 0; one that fails → 1; one that exits 75 and is re-queued, then ends ok → `wait` waits through the retry and returns 0; one already finished → its status at once; an unknown name → 2; `--timeout` → 124 | never | one run | seconds |
  | W3 the command line | `python3 scripts/night.py wait NAME` run as a subprocess under a temporary `NIGHT_DIR` exits with the job's status while a runner in the test runs it: what a session on the other box runs as its one `ssh` | never | one run | seconds |
- **PW5 (after MM0).** `scripts/touched.py` drops `-count=1` for packages with no hidden inputs. A package whose tests
  read a probe, a device or a file outside the module is listed as such, and a source scan keeps the list true.
- **PW6.** Re-run §1's census 30 days after the first VK, MM and PW items land.

### 12.1 PW1's gates (registered 2026-10-26, before the change)

Every gate is tier 0: seconds, no model, no device, no timed run. Each runs once on the finished branch.

| gate | rule (PASS iff) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| PW1-G1 the guard's own tests | `python3 -B scripts/test_main_ci_guard.py` exits 0, all tests `ok` (the CI step of the same name) | a test is skipped | one run | stdlib, under a second (the CI step's own note) |
| PW1-G2 the guard on a docs-only range | `scripts/main_ci_guard.py 980f449b^ 980f449b` (a docs-only commit on main) exits 0 with "a docs-only push", with `MAIN_CI_GUARD_RUNS` naming a fake in-progress run, so the pass cannot come from an idle main | `ci_docs_only.sh` classifies the range as code | one run | about a second |
| PW1-G3 the guard on a product range | the same with `4f9f07aa^ 4f9f07aa` (a Go change on main) exits 1 and names the fake run | `ci_docs_only.sh` classifies the range as docs-only | one run | about a second |
| PW1-G4 the whole hook, both ranges | the hook as `install-git-hooks.sh` writes it, fed each range as a `refs/heads/main` stdin line (no push), runs the guard, the citation lint, `TestParityManifest_fresh` and the census warning, and its exit matches G2 and G3 | the citation lint or the parity test is red on the tree for a reason unrelated to the range (then G4 reports the guard's verdict only) | one run per range | the lint and the parity test, under a minute warm |
| PW1-G5 the text | `queue_citation_lint.py` exits 0 (read directly, not through a pipe) and `book_link_lint.py` exits 0 | — | one run | seconds |

If the hook itself changes after DECISION PW1-D1, G1 to G4 are re-run on the changed hook, with G3's expected exit set by the decision.

**Results, 2026-10-26, on the branch after the `CLAUDE.md` change (the hook's logic unchanged).** G1 PASS (9 tests `ok`, none
skipped). G2's first run was void by its own rule: it passed `980f449b^`, and `ci_docs_only.sh` accepts only hex SHAs, so it
failed open to "code"; the hook always passes SHAs. Re-run with both ends resolved to full SHAs: classifier `true`, guard exit 0,
"a docs-only push". G3 PASS: classifier `false`, exit 1, naming fake run 99 (against the real `gh run list` it also held,
naming #264's merge run). G4 PASS: the docs-only range exits 0 through the guard, the citation lint,
`TestParityManifest_fresh` and the census warning; the product range exits 1 at the guard. G5 PASS: both lints exit 0.

**DECISION PW1-D1, Francis, 2026-10-26 11:47 (relayed by the coordinator): remove `scripts/main_ci_guard.py` and its pre-push
step** (option (b); recommended was to turn the wait into a refusal of direct product pushes). The guard, its tests, the hook
step, its CI step and the `CLAUDE.md` sentence go. Amended gates, registered before that change, run once on the changed branch:

| gate | rule (PASS iff) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| PW1-G1' nothing names the guard | `git grep -n -E 'main_ci_guard\|MAIN_CI_GUARD'` outside frozen records (`docs/completed/`, `docs/measurements/`) and this section's own history matches nothing that runs: no script, hook, workflow or `CLAUDE.md` line | — | one run | a second |
| PW1-G2' the hook on the docs-only range | the hook extracted from the changed `install-git-hooks.sh`, fed `980f449b`'s range as full SHAs on a `refs/heads/main` line, exits 0 | the citation lint or the parity test is red on the tree for a reason unrelated to the range | one run | under a minute |
| PW1-G3' the hook on the product range | the same for `4f9f07aa`'s range exits 0, with `MAIN_CI_GUARD_RUNS` still naming a fake in-progress run (it must now be ignored: nothing waits on main's CI) | as G2' | one run | under a minute |
| PW1-G4' the workflow and the installer parse | `ci.yml` loads as YAML with the `main CI guard` step gone and every other step of its job kept; `bash -n` on `install-git-hooks.sh` and on the extracted hook | — | one run | a second |
| PW1-G5' the text | as G5, after the deletion (a citation of the deleted file in a live doc is red and is fixed, not pinned away) | — | one run | seconds |

**Results, 2026-10-26.** G1' PASS: the only matches left are this doc's ledger, its §2.5 diagnosis, the PW1 plan text and this
section. G2' PASS: the docs-only range exits 0. G3' PASS: the product range exits 0 with the fake in-progress run named,
so nothing waits on main's CI. G4' PASS: `ci.yml` parses, and against `origin/main` the only change is the `lint` job's
`main CI guard` step, with every job kept; both `bash -n` checks pass. G5' PASS: the citation lint and the book link lint exit 0.
The `CLAUDE.md` merge step also took the coordinator's correction of 11:52: a session prints `BRANCHES READY` and the
coordinator or Francis sets auto-merge, because a background session's own `gh pr merge` is denied as a merge without
review and a session never routes around a denial.

**Amended 2026-10-26 evening (Francis, 13:04 and 21:42).** A session's own `gh pr merge --auto`, run as a bare command
while the checks are pending, passes the classifier (every PR from #274 on merged this way), so the session sets auto-merge
itself right after `gh pr create`; only a plain merge without `--auto` stays forbidden. `CLAUDE.md` says so.

### 12.2 PW2a's gates (registered 2026-10-27 11:55 PDT by nobara-ci01, before `ci.yml` changes)

The measurement that sets the target is `docs/measurements/ci-tail-2026-10-27.md`: on the slow runner class the tail's
`go test` step is 1,169 s over a 1,130 s package, of which the two census tests are 364 s and 295 s and the rest 472 s.
The change: the tail's `-skip` adds both census names; `test-decoder-a` gains a step running `^TestDecisionCensus$` on
`./decoder/` and `test-decoder-d` one running `^TestDecisionCensus_secondProcess$`, each under the same
`go test -race -timeout 25m -tags goinfer_testhooks -json` line as every other shard, each with the zero-match guard
and the summary step the other shards carry. No test's content changes, no job is added or renamed.

Purity, for the -race clause: neither census test is pure. `buildCensus` runs D1, D2 and the loads through
`parallelEach`, which starts `runtime.NumCPU()` goroutines (read in `decoder/mm1_census_build_test.go`, confirmed by
`grep -n 'go func' decoder/mm1_census*_test.go`), and the census's own mutex guards their writes. So both keep -race.

| gate | rule (PASS iff) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| PW2a-G1 every test still runs, by name | `docs/measurements/ci-tail-2026-10-27/shard_cover.py` reads every `go test` line of the five decoder shards and `test-rest` in `ci.yml` at `origin/main` and at the branch, lists the top-level tests of `./decoder` and `./internal/sampling` (`go test -list`, `-tags goinfer_testhooks`), and maps each to the steps that select it, applying `-run`, `-skip` and the package list. PASS iff, on both sides, every listed test is selected by exactly one -race step; the two sides select the same tests (counts printed, equal); the only tests whose step changed are the two census tests; and the no-race sampler-gates step selects the same tests on both sides | the two sides' test lists differ (main moved under the branch: re-run on a merged tree) | one run | `go test -list` compiles two packages, under a minute |
| PW2a-G2 -race kept | in G1's map, every test that ran under -race on `origin/main` runs under -race on the branch, the census tests included (no exception is claimed: they are not pure, above) | as G1 | one run | in G1 |
| PW2a-G3 check names | the set of job ids and of job `name:`s in `ci.yml` (loaded as YAML) is identical at `origin/main` and the branch, so the 19 required checks keep existing and branch protection needs no change | — | one run | a second |
| PW2a-G4 the census steps ran their tests | in the first PR run, each new census step's summary lists its test with an elapsed time over 60 s (a skip or a zero-match reads near 0 s), and the tail's top-level test count plus the two census steps' counts equals the count main's tail reported at the PR's merge base | a decoder test was added or removed between the merge base and the PR head (then compare against `go test -list` on the PR head) | the first PR run | read from its logs |
| PW2a-G5 the slowest required job | in the first uncontended PR run, every one of the 19 required jobs finishes under 12.0 minutes from the run's first job creation; recorded with each job's duration, the tail's and shards a and d's step times, and the runner class (`TestPrefillChunked_bitIdentical` reads about 50 s on the fast class and 85 s on the slow one). 12.0–13.0 is ambiguous: read the next run (the merge run on `main`) and record both | any required job started more than 2 minutes after the run's first job was created (runner queueing, not our jobs), or the run is a re-run attempt; then the next run is read | the first non-void run; if it is on the fast class, the next slow-class run on `main` is also read when it arrives, not triggered | read from `gh api .../jobs` |
| PW2a-G6 the scans that read `ci.yml` | `python3 scripts/touched.py` on the branch passes, which runs every source scan reading `ci.yml` (the sampler sweep-coverage check, the multimodal and kvmgr race-subset checks, the release-CI check), and `python3 -B scripts/test_touched.py` passes | the timing lock is held by a timed run (wait for it) | one run | seconds to a minute |

**Results, 2026-10-27** (`docs/measurements/ci-tail-2026-10-27.md` §3–4). G1 PASS: 1,252 tests, each in exactly one -race
step on both sides, only the two census tests moved; two plants red. G2 PASS: none lost -race. G3 PASS: 20 jobs identical,
so protection is unchanged. G4 PASS: the census steps ran 230.4 s and 228.1 s, and the tail's 1,202 plus 2 equals main's
1,204 at the merge base. G5 PASS on PR run 40560349006, slow runner class, not void: the slowest required job, `test-rest`,
finished 10.1 minutes in. G6 PASS by name only: `touched.py` ran none of the four scans that read `ci.yml`, because
`scripts/touched_scans.py` has no row for it, so they were run directly; the missing row was then added (below).

**Amended 2026-10-27, after the review of #325 (relayed 11:49).** G6's route is corrected: `scripts/touched_scans.py` lists the
four `ci.yml` readers, and `touched.py --base origin/main` runs them, each `ok`. The census guards list with
`-tags goinfer_testhooks` and report a failed listing as a build failure; their summaries skip when the test step did not
run. G1–G3 re-run on the fixed head: PASS. The review's other points and what held are in the record's §5.

### 12.3 PW5's gates (registered 2026-10-27 before the code; mac-vk01; Francis, 11:08)

**Why Go's test cache cannot be trusted everywhere.** Go replays a cached test result when the test binary, its cacheable
flags, the environment variables it read through `os.Getenv`, and the files inside its module that it opened are all
unchanged (`go help test`; `computeTestInputsID` re-checks no file outside the module root). Anything else a test
depends on is a hidden input, and a replayed pass would hide a change in it.

**The hidden inputs, each found by a source scan:**

| hidden input | how the scan finds it |
|---|---|
| OS memory | an import of `internal/mem/probe`, the only OS memory reader since MM0 |
| a device | `metal`, `cuda` or `gpu` code, or an aikit `gpu` import |
| a subprocess | `exec.Command`; its environment and its reads are untracked |
| the whole environment | `os.Environ()`, which `testlog` does not record |
| files outside the module | the home directory, the asset registry's checkpoints, absolute `/Users`, `/home`, `/proc`, `/sys`, `/tmp` or `/dev` paths, and `"../` in a nested module |
| the network | `net.Dial` or `http.Get` |
| timing | a test that reads the clock: `time.Since` or `testing.Benchmark` |

**Where a scan looks.** A package has a hidden input when one of these is found in any of:
- its own files, tests included;
- the non-test files of any goinfer package its code or tests import, transitively over `internal/layout/imports.json`'s
  arrows.

Most packages that import `decoder` are therefore not cacheable, because `Load`'s fit guard reads the probe.

**The data and how touched.py uses it.**
- The classification is data, `internal/layout/test_cache.json`: every package in the seven modules, cacheable or not,
  with its reasons.
- A `declared` section holds reasons a scan cannot see, added by hand; nothing is in it at the start.
- `TestTestCache_classified` regenerates the classification and fails on any difference (`-update` rewrites it), so a
  cacheable package that gains a hidden read goes red in the change that adds it.
- `touched.py` drops `-count=1` only for a package the data marks cacheable. An unlisted package, or a missing or
  unreadable data file, keeps it.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| PW5-G1 complete and true | `TestTestCache_classified` passes; every directory holding a Go file in the seven modules has an entry; every non-cacheable entry names at least one reason | never | one run | seconds |
| PW5-G2 the scan bites | three `gate mutation` plants turn it red, naming the package and the reason: `exec.Command` in a test of `internal/f16`; `time.Since` in a test of `internal/giw`; an import of `decoder` in a test of `internal/towerid` ("OS memory", reached through `decoder`) | never | one run each | seconds |
| PW5-G3 touched.py honors it | `scripts/test_touched.py`: a cacheable package's `go test` has no `-count=1`; a non-cacheable one, an unlisted one, and every package when the data file is missing or malformed all keep it | never | one run | seconds |
| PW5-G4 the cache replays only what it should | in G5's second run, every package the data marks cacheable whose inputs the edit did not touch prints `(cached)`, and no non-cacheable package does | the edit touches every cacheable package | one run | from G5 |
| PW5-G5 the measurement | `touched.py`'s wall time on the diff `0158bdc0..3d01e4a3` (`decoder` 3 files, `internal/mem` 3, `metal` 1, and the scan tests they trigger), on the PW5 branch, under a private `GOCACHE` warmed once by an untimed run. Sequence: before A1, then a one-line comment edit to one `decoder` file in the diff, A2, a run with no change A3; the edit reverted; after B1, the same edit, B2, B3. Each run records its wall time, the packages that ran and replayed, and the 1-minute load average at its start. Reported as measured, with no pass bar: Francis asked for the number | a run starts with the load average above 4.0 (re-run it, and list it) | one sequence | about 6 runs × a minute or two |

**PW5 graded, 2026-10-27: G1–G3 PASS, G4 VOID in its registered run and shown by B3, G5 measured**
(`docs/measurements/pw5-2026-10.md`).
- 14 of 82 packages are cacheable (16 as first built; Gemini's two reviews of #327 widened the scan); touched.py drops
  `-count=1` for those only.
- On `0158bdc0..3d01e4a3`, a re-run with nothing changed fell from about 133 s to 124.3 s. Three cacheable tests
  replayed, saving about 9 s. Most of the time is `metal`'s device suite (about 100 s) and `decoder` (13–18 s), both
  uncached by their hidden inputs.
- G2's first plant moved to `internal/kvmgr` (`internal/f16` has no test), and timing counts only test files, as
  registered. Both are recorded there.

**Amended 2026-10-27, after Gemini's second look at #327** (`docs/measurements/pw5-2026-10.md` §5):
- Clock reads count in every file, not only in tests, failing safe; the registered "a test that reads the clock" is
  widened.
- Paths are judged on string literal values, and a climb past the module root counts in any module.

**PW5 after Gemini's third look (2026-10-27, `docs/measurements/pw5-2026-10.md` §6):**
- Climbs are counted on cleaned literals and summed per package. A declaration counts only once applied.
- The patterns are wider, and randomness is a hidden input.
- **Decided 2026-10-27: A, retire the cache** (Francis). `touched.py` passes `-count=1` for every package again; the
  scanner (`internal/layout`'s test-cache classification), `internal/layout/test_cache.json`, its `-update` and
  `touched.py`'s cache wiring are removed. Reason: about 9 s saved on a 133 s re-run, and three review rounds each found a
  new hole.

### 12.4 PW2b's gates (registered 2026-10-27 17:40 PDT by nobara-ci01, before `ci.yml` changes)

The change: each of the eight summary steps that read a `go test -json` file under `if: always()` (shards a to d, the tail,
shard c's sampling tail, `test-rest`) gets the condition the census summaries already have,
`always() && steps.<id>.outcome != 'skipped'`, and the test step that writes its file gets that `id`. Nothing else in
`ci.yml` changes. All tier 0, each run once on the finished branch.

| gate | rule (PASS iff) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| PW2b-B1 only the conditions change | `ci.yml` at `origin/main` and at the branch, loaded as YAML, differ only in: each summary step's `if` (now naming its writer's `id`), and an `id` added to each writer; every summary that reads a `.json` names the step that writes it, and that step carries the `id`; every other key of every step and job is equal | — | one run | a second |
| PW2b-B2 the tests and checks are unchanged | `shard_cover.py origin/main HEAD` (no movers) prints PASS: the same 1,252 tests in the same steps, the same jobs | as PW2a-G1 | one run | under a minute |
| PW2b-B3 the scans that read `ci.yml` | `python3 scripts/touched.py` runs the four `ci.yml` readers, each `ok` | the timing lock is held by a timed run | one run | seconds |
| PW2b-B4 the normal path | in the PR's first run, every one of the eight summary steps ran (conclusion `success`, not `skipped`) and every job is green | the run is cancelled | the first completed run | read from `gh api .../jobs` |

The skipped path (a summary skipped because its test step was) rests on GitHub's documented `steps.<id>.outcome` values and
is not planted in CI: a plant costs a full PR run of shared runners. The census steps have used the same condition since #325.

## 13. Order, sizes, and what each step is worth alone

| order | items | worth alone |
|---|---|---|
| 1 | VK0, VK1 (goinfer side), VK2, VK3 | most commits stop re-staling parity; the manifest stops being a merge point |
| 1 | PW1 (SD4), PW2, PW3 | sessions stop waiting on CI; CI stops queueing |
| 1 | AX0, AX1, AX2 | tower and device work leaves the release train |
| 2 | MM0, MM1, MM6 | night jobs stop being refused for a reason knowable at queue time; memory tests stop reading the machine |
| 2 | SB1's first step, SB13 | decoder edits stop re-running backend kernel suites |
| 3 | AX3, AX4, AX6, VK4, VK5 | aikit bumps become rare and, with unchanged numerics, free |
| 3 | MM2, MM3, SB2, SB5, SB6, SB7 | the memory account closes; admission is a table; the shell tests with no model |
| 4 | VK6, SB4, SB8–SB12, PW4, PW5 | the daily run audits the contracts |
| later | MM4 (after KV4 and KV7); MM5 and SB1's interface (after OP4 and RF1) | one budget; demand from the program; one call shape |

## 14. Owner decisions

| id | decision | recommendation |
|---|---|---|
| SD1 | move the towers and the GPU backends into goinfer; linalg stays in aikit, through `go.work`, tagged weekly | **decided 2026-10-26** |
| SD2 | the ANN and text-encoder GPU backends (`…/gpu/anncuda`, `…/gpu/annmetal`, `…/gpu/enccuda`, `…/gpu/encmetal`) | stay in aikit on the frozen `gpu/v0.35.x`; `gpupins` stops making them track the newest root tag. **Decided 2026-10-26** as recommended |
| SD3 | between weekly aikit tags, may goinfer's main require an aikit pseudo-version (a commit on aikit's main)? | yes for main; a goinfer release still requires a tagged aikit, which keeps aikit's own rule that goinfer never ships against an untagged aikit. **Decided 2026-10-26** as recommended |
| SD4 | auto-merge, and a rule on `main` requiring the checks; the pre-push wait retired | **decided 2026-10-26** (owner: "yes to SD4"); applied the same day |
| SD5 | the moved towers as internal packages, with what the examples and the demo need exposed through `multimodal` | **decided 2026-10-26:** internal, as recommended |
| SD6 | how long VK5 runs in shadow mode before it replaces today's hash | **decided 2026-10-26: one week**, with no unexplained disagreement (recommended was two weeks or 200 pushes) |
| SD7 | the daily run's boxes and hours | **decided 2026-10-26: both boxes at night** (recommended was nobara by day), within the night queue's limits |
| SD8 | guard thresholds as `mem.Policy` data with an approval stamp (I7) | **decided 2026-10-26:** yes, as recommended; it keeps "a threshold change needs Francis's word" and makes it a test |

## 15. Risks

- **A fingerprint misses a path,** and a stale validation carries over. VK6's canary bounds the delay to two weeks, and
  the coverage audit names the path.
- **Shadow-mode disagreements take time to explain.** SD6 keeps the old hash until they are explained.
- **The planner drifts from the machine.** I3 per backend and the daily calibration catch it.
- **More modules.** Rule 4: packages, not modules.
- **Public API creep.** Internal packages for everything new (SD5).
- **Mechanical moves collide with in-flight branches** at about 110 commits a day. Announce each move, land it as one
  mechanical commit per module, and keep logic changes out of it.
- **Habits outlive rules.** PW6's census checks whether the waiting stopped.

## 16. Prior art (read 2026-10-26)

- **Go's test cache** ([`go help test`](https://pkg.go.dev/cmd/go/internal/test)): "The rule for a match in the cache
  is that the run involves the same test binary and the flags on the command line come entirely from a restricted set
  of 'cacheable' test flags"; "Tests that open files within the package's module or that consult environment variables
  only match future runs in which the files and environment variables are unchanged"; "The idiomatic way to disable test
  caching explicitly is to use -count=1." **Take:** pure packages make the cache safe (PW5).
- **Bazel's test caching** ([VirtusLab's Bazel book](https://bazel.virtuslab.com/book/1~2~1/)) keys a result on the test
  binary, its runfiles, flags and environment, replays a passing result while they are unchanged, always re-runs a
  failure, and has an `external` tag for tests that depend on state outside the build graph. **Take:** declare hidden
  inputs instead of disabling the cache everywhere (PW5).
- **Meta's predictive test selection** ([Machalica et al., ICSE-SEIP 2019](https://arxiv.org/abs/1810.05286)) reports
  "over 95% of individual test failures and over 99.9% of faulty changes" while selecting fewer than a third of the tests
  build dependencies would select. **Take:** dependency-based selection over-selects badly, which is the case for
  behavior keys. **Leave:** a learned model; fingerprints and tiers are exact where it is statistical.
- **llama.cpp's `test-backend-ops`** ([its testing page](https://mintlify.wiki/ggml-org/llama.cpp/development/testing))
  "checks that different backend implementations of ggml operators produce consistent results". **Take:** the op
  conformance kit (SB1).
- **PyTorch's OpInfo** ([OpInfos FAQ](https://github.com/pytorch/pytorch/wiki/OpInfos-FAQ)): an operator's metadata and
  sample inputs; "The `ops` decorator will create a set of tests for each `device` and `OpInfo`", each compared with a
  reference. **Take:** one operation database drives every backend's tests (SB1, SB9).
- **vLLM** ([cache config](https://docs.vllm.ai/api/vllm/config/cache.html)) sizes the KV cache after a profiling run,
  inside `gpu_memory_utilization`, "the fraction of GPU memory to be used for the model executor"; its block count
  "will be set after profiling". **Take:** measure the non-KV peak, then plan, which is I3's calibration. **Leave:**
  profiling at every start; `Solve` computes and the daily run calibrates.
- **ggml's graph allocator** ([`ggml-alloc.h`, a mirror of ggml-org's](https://git.mitjafelicijan.com/r/llmnpc/blob/llama.cpp/ggml/include/ggml-alloc.h))
  reserves buffers from a worst-case measure graph and "does not allocate or modify the graph". **Take:** demand derived
  from the program (MM5).
- **llama.cpp's `--fit`** ([fit-params README, a mirror](https://huggingface.co/datasets/echodict/llama.cpp/blob/main/tools/fit-params/README.md))
  fits a model's projected memory use to the free device memory at run time (`llama_params_fit`) by changing the
  context size, the layers on the GPU and expert tensor placement, with a free-memory margin. **Take:** a decline
  that proposes the fitting change, as `Solve`'s `Decline` should.
- **MLX's memory controls** ([its `memory.h`, as vendored by go-mlx](https://forge.lthn.ai/core/go-mlx/raw/branch/main/dist/include/mlx/memory.h))
  separate a memory limit (a guideline during graph evaluation; by default 1.5 times the recommended working set on
  Metal), a cache limit, and a wired limit, "the total size in bytes of memory that will be kept resident". **Take:**
  tiers and limits as separate knobs (`Tier`, `Policy`).
- **GitHub auto-merge** ([GitHub Docs](https://docs.github.com/en/enterprise-cloud@latest/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/automatically-merging-a-pull-request))
  "is available in public repositories with GitHub Free", and merges once the required reviews and status checks pass.
  **Take:** PW1.

## 17. When this closes

Each track closes when its ledger rows are done, killed or moved. The doc closes when PW6's re-count is graded against §1.

<!-- doc-reviewed: 2026-10-26 -->
