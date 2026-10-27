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
| VK5 | active | shadow week 1 (SD6) runs from the workflow's first run on main, 2026-10-26, and grades the aikit-pin → kernel-fingerprint swap alone; week 2 then grades the key built from program fingerprints (VK4) plus kernel fingerprints, with the forward source files out of it (Francis 16:27: option 1); the switch after each week is its own PR and decision; M | §5.8; `docs/measurements/vk5-2026-10.md` |
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
| SB8 | open | the conversation codec (templates, tokenizer, tool calls), golden per template; S–M | §8 |
| SB9 | blocked | AX2; the towers with per-tower goldens; S | §8 |
| SB10 | open | one typed configuration schema, resolved by a pure function, its docs generated; M | §8 |
| SB11 | open | cost counters per operation and phase, one schema for tests, gates and the wire (with X1); M | §8 |
| SB12 | open | gate infrastructure: thin runners, graders as pure functions of records; M | §8 |
| SB13 | open | the package layout as an import-direction test with ratchets; S | §9 |
| SB14 | open | the doc-drift test: every invariant a `doc.go` lists names a test that exists, and `docs/ARCHITECTURE.md`'s import table equals SB13's data; S | §4.1 |
| PW1 | active | SD4 decided 2026-10-26: branch protection on `main` (19 required checks, not strict, admins not enforced) and "Allow auto-merge" turned on that day; left: sessions use `gh pr merge --auto`, and retire `scripts/main_ci_guard.py`'s wait and its `CLAUDE.md` rule; S | §12, §14 |
| PW2 | open | the macOS, Windows and arm64 CI jobs after merge or path-filtered, out of the required checks; S | §12 |
| PW3 | open | the shared hot files sharded or generated; S–M | §12 |
| PW4 | open | `night.py wait <job>`, so a session on the other box gets one completion instead of polling; S | §12 |
| PW5 | blocked | MM0; `touched.py` drops `-count=1` for packages with no hidden inputs; S | §12 |
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
  guard predicts what each backend will build (`decoder/fitguard.go:metalWillBeResident`,
  `decoder/fitguard.go:cudaWillBeResident`). Their tests read the machine too, so a verdict depends on what else is
  open.
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

### SB2 — admission and routing

- **Today:** `decoder/residency.go:Model.residentAdmission`: the runner shape (`Model.decodeRunnerDecline`), then
  `decoder/features.go:residentGateReason` against `ResidentBackendFeatures`, then the backend's own build through
  `DeclineResident`; the option-path admission grid of 2026-10-08 (`docs/completed/task-option-path-admission-2026-10.md`).
  The owner's note of 2026-10-21 in the forward-architecture doc: 20 of the 71 registration and bound findings counted on
  2026-10-08 were one family that one path did not know about, six of them Critical.
- **Contract:** `Admit(program, request shape, options, backend op set, plan) → Route or Decline`, pure.
- **Fast check:** the whole table, every family × backend × shape × option, in milliseconds, as golden output;
  `docs/capability-matrix.json` and `docs/hardware-matrix.md` generated from it, so they cannot drift from the code.
- **Stops re-running:** finding out in a real run which path a request took. M.

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

### SB13 — the package layout, as a test

§9's table as a test that reads `go list -deps` for every module and fails on a forbidden edge, with ratchets (backend
test files importing `decoder` may only fall). S.

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
- **PW3.** Shard or generate the hot files. The citation index records each cited commit's subject in one file, and
  every new citation edits it: keep the check, and store the expected subjects per citing doc. CHANGELOG fragments per
  change, assembled at release. Parity records per family (VK3). The `docs/README.md` counts written by
  `scripts/task_docs.py`, which already recounts them (on 2026-10-26 it found the design-record count one short).
- **PW4.** `night.py wait <job>`: block until a queued job ends and exit with its status. A session on the other box
  then runs one background `ssh` and gets one completion notification, in place of sleeping and polling over ssh.
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
