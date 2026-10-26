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
| VK2 | active | mac-vk01, branch `vk2-adapters-2026-10`: part 1 (one file per adapter) under gates A1–A4 registered in §5.5; part 2 (`model.go` out of `core`) waits on a decision, because 19 files holding forward-path code are in no hash set; S–M | §5.5 |
| VK3 | open | either box: one parity record per family in place of one shared manifest; S | §5.6 |
| VK4 | open | either box: a program fingerprint per family config, built from the config with no weights; M | §5.7 |
| VK5 | blocked | VK1–VK4; the behavior key in shadow mode beside today's hash, then in its place (SD6); M | §5.8 |
| VK6 | blocked | VK5; the daily canary and the coverage-of-reality audit; M | §5.9 |
| MM0 | open | either box: every OS and device memory reading behind one probe package, with a fake for tests; S | §6.5 |
| MM1 | open | either box: the pure planner `mem.Solve` out of the fit code, with a decision census against today's; M | §6.5 |
| MM2 | blocked | MM1; backends report the bytes they will allocate instead of being predicted; M | §6.5 |
| MM3 | blocked | MM2; a ledger and leases in each backend's allocator; M | §6.5 |
| MM4 | blocked | KV4 and KV7; the arbiter, which is KV11's one budget for KV blocks, expert slots and the prefix cache; L | §6.5; KV11 |
| MM5 | blocked | OP4; demand derived from the program, each operation declaring its buffers; M | §6.5 |
| MM6 | blocked | MM1; `night.py add` and `goinfer fit` call `Solve` against the box's recorded profile at queue time; S | §6.5 |
| AX0 | open | freeze the moving aikit paths at the tags the move starts from; S | §7.3 |
| AX1 | open | copy into goinfer with provenance, NOTICE lines, tests and fixtures; M | §7.3 |
| AX2 | blocked | AX1; switch goinfer's imports one module at a time, each step bit-identical; M | §7.3 |
| AX3 | blocked | AX2; drop the `aikit/gpu` requires and add a source scan that keeps them out; S | §7.3 |
| AX4 | blocked | AX3; aikit side: deprecate or remove by tier, trim the GPU tooling, tag; S–M | §7.3 |
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

### 5.6 VK3 — one parity record per family

`testdata/parity/<family>.json`, plus one file for the shared sets; `TestParityManifest_fresh` reads the directory. Two
sessions working on different families stop colliding, and a refresh touches only the records it changes. S.

### 5.7 VK4 — a program fingerprint

`decoder/program.go:buildProgram` takes an `*Architecture` and an experts predicate, not weights, so a program can be
built from a config alone. Serialize the op kinds and their parameters and hash them: one fingerprint per family fixture
config and per real checkpoint config in `testdata/assets.json`. A refactor that leaves every program unchanged leaves
every family's key unchanged. M.

### 5.8 VK5 — the key, in shadow mode first

On every push to main, compute both today's staleness and the new key, and log every disagreement with its reason.
Switch after one week (SD6, decided 2026-10-26) with no disagreement left unexplained. M.

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
