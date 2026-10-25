# SP2 — speculation on a recurrent hybrid: Qwen3.5-9B with DFlash on nobara (2026-10-24)

SP2 of track SP. The gates were registered before any code, in `docs/tasks/task-forward-architecture-2026-10.md` §6a, "SP2 on
nobara" (commit `7d5a96cf`, pushed before the first line of SP2 code). This page records each reading against that rule.

- **The mechanism:** `cuda/recurrent_spec.go` (the arena, `VerifyLastNArgmax`, `FoldVerified`), the twins in `cuda/deltanet.cu`
  (`delta_conv_rows_spec`, `delta_rule_rows_128_spec`, `delta_rule_rows_spec`), and the loop in `decoder/blockspec.go`
  (`BlockSpec.generate`, `Model.specRollbackSafeVia`).
- **The gates:** tier 0 in `cuda/recurrent_spec_test.go` and `decoder/recurrent_spec_test.go`; the 9B in
  `cuda/recurrent_spec_real_test.go`, driven by `run-sp2.sh` (`MODE=identity|cost|speed`).
- **Logs:** `~/goinfer-logs/sp2-2026-10/` on nobara; the JSON lines each 9B run writes are copied under `raw/`.

## The kernels: what changed in `deltanet.ptx`

Rebuilt with `cuda/build_ptx.sh deltanet` (NVRTC 12.9.86; the committed PTX first rebuilt byte-identically at main `d9a1cb4d`,
so the toolchain reproduces it). As merged, `cuda/deltanet.cu` gains only additions: every committing kernel's source is main's,
and its PTX is main's apart from label numbers. The three verify twins are standalone kernels:
- `delta_conv_rows_spec` folds the previous verify's committed rows into the window (the slide alone), then computes each row from
  the window and the pass's earlier rows with `delta_conv_rows`'s FMA chain, in the same floating-point order, and slides nothing
  for its own rows;
- `delta_rule_rows_128_spec` runs one loop over the folded rows and its own with `delta_rule_rows_128`'s row arithmetic, stores the
  state once after the folded rows, and writes output for its own rows only;
- `delta_rule_rows_spec`, the generic twin, holds the state row in a predicated 128-float register array where `delta_rule_rows`
  walks it in global memory, with the same ascending-kd FMA chains.

G1a shows each bit-identical to a prefill of the same rows on the models that take it. None declares local memory in its PTX; at
the driver's JIT the 128-wide twin spills 96 B/thread, the committing kernel's own figure (`TestKernelLocalMemoryCensus` pins
both). Earlier forms of the twins (templated with the committing kernels, before the fold moved into the scan) are described where
their readings are recorded below.

## Readings

### Tier 0 (G1a, G1b, G5, G3's pricing): PASS on all three models, both plants caught (2026-10-24, 23:39–23:55 PDT)

Run on nobara under the timing lock (shared), driver 595.91.07: `go test -tags 'cuda goinfer_testhooks' -run TestRecurrentSpec
./cuda/` with `GOINFER_HEAVY_TESTS=1`, and the decoder's `TestSpecRollbackSafeVia_admitsOnlyTheFoldPath`,
`TestRecurrentSpecArenaBytes_formula` and `TestBlockSpecGenerate_foldsTheEmittedRows`. Final run at `f3d4b1b2`; log
`~/goinfer-logs/sp2-2026-10/tier0-1024-2339.log` and the run after the lint fixes.

| gate | qwen35-tiny (generic scan, MoE FFN) | qwen3next-tiny (generic scan) | qwen3.5-0.8b (128-wide scan) |
|---|---|---|---|
| G1a: verify leaves the state; fold n = prefill n, byte for byte, w ∈ {2, 4, 8, 16}, every n | PASS (30 cases) | PASS | PASS |
| G1a under P1 (fold skips the conv window) | caught, 30 of 30 cases | caught, 30 of 30 | caught, 30 of 30 |
| G1a under P2 (verify writes the state) | caught, 56 failures | caught, 56 | caught, 56 |
| G1b: scripted drafter, tokens and end state equal plain greedy, w ∈ {2, 4, 8, 16}, 2 prompts | PASS | PASS | PASS (128 tokens) |
| G1b with the guard tripped (plain steps after folds) | PASS | PASS | PASS |
| G1b under P1 / P2 | caught, 8 of 8 / 8 of 8 | caught, 8 / 8 | caught, 8 / 8 |
| G5: checkpoint untouched; kvmgr reuses all of A; state after A = cold prefill; B warm = cold, plain and speculative | PASS | PASS | PASS |
| G3: the reserve grows by the arena's price, which is the allocated bytes rounded up to 2 MiB | PASS | PASS | PASS |
| a seed longer than one pass keeps every captured row, bit-equal to one pass | — | PASS | — |

**Two defects that predate SP2, found by these gates and fixed on this branch:**
- **BlockSpec at the context cap.** On qwen35-tiny's 64-position context, a speculative turn emitted one token more than plain
  decode (a round's last, never forwarded), and with the guard tripped its fallback stepped past the cap and failed where plain
  decode finishes. `BlockSpec.generate` now takes plain decode's token budget (the context left after the prompt).
- **A long prompt's seed went out of device memory** (G1c's first run, below): with the drafter's capture armed, the CUDA prefill
  ran the whole prompt as one pass. It now takes `PrefillLast`'s passes and joins each pass's captured rows, so `serve --drafter`
  on a long prompt no longer needs one 3k-row pass beside the drafter.

**The source-scan gates:** `touched.py` green at `f3d4b1b2` (gofmt, build and vet of both modules, the cuda kernel lints, the
gate's heavy partition, which now classifies Qwen3.5-9B). The local-memory census pinned the twin as it stood then,
`delta_rule_rows_128_spec`, at 80 B/thread; the final twin (the fold in the scan, below) spills 96 B, the committing scan's own
figure, and is pinned at 96. staticcheck (CI's v0.8.1 build, its unused-field canary red first) is clean on
`./decoder/` and on `./cuda/` under `cuda goinfer_testhooks`.

### G1c, first run (2026-10-24, 23:41–23:45 PDT): stopped by an out-of-memory in the seed, not a divergence

All 36 cells it reached passed: 18 prompts (SP0's 12 thinking-off, its 6 thinking-on) at w = 4 and w = 16, tokens and every
DeltaNet layer's state byte-equal to plain greedy. The 19th prompt (2,924 tokens) went out of device memory in BlockSpec's seed
prefill, the defect above. The run is kept as recorded (`raw/identity-2026-10-24-2340.jsonl`); G1c is graded on the re-run.

### G1c, the re-run: PASS (2026-10-24 23:54 – 2026-10-25 00:02 PDT, at `f3d4b1b2`)

`MODE=identity bash run-sp2.sh`, the test binary run directly under the timing lock (shared), at load 1.6 or below with 58 GB
available. Peak resident 15.3 GB (`VmHWM`), peak swap 0. Raw: `raw/identity-2026-10-24-2353.jsonl`.

- **The load:** Qwen3.5-9B's `.giw`, resident context 4,608 (SP0's, held with the arena priced in), f32 KV, the drafter
  attached through `NewBlockSpec`. The arena allocates 31,817,728 B (30.3 MiB: 16 rows × 24 DeltaNet layers, the registered
  formula), and the fit guard prices it at 33,554,432 B (32 MiB, rounded up to 2 MiB as `cuMemAlloc` rounds it). The companion
  reserve: 1,326,350,336 B. Free VRAM with everything attached: 1.15 GB of 8.16.

| cell | prompts | w = 4 | w = 16 |
|---|---|---|---|
| code, thinking off | 6 | 6 of 6 | 6 of 6 |
| math, thinking off | 6 | 6 of 6 | 6 of 6 |
| math, thinking on | 6 | 6 of 6 | 6 of 6 |
| long (2,924 and 2,922 tokens) | 2 | 2 of 2 | 2 of 2 |

- **The bar:** every cell's speculative tokens equal plain greedy's (`Model.Generate` under the exact-attention scope), to the
  end, and every DeltaNet layer's conv window and state after the turn are byte-equal to plain's. **40 of 40.**
- **The guard's fallback** (tripped at its window, 4 code prompts at w = 4): 4 of 4.
- **The plants** at w = 4 on the 12 thinking-off prompts: P1 (fold skips the conv window) diverged on 12 of 12; P2 (verify
  writes the state) on 12 of 12. Registered expectation: at least one each. Met.
- **The served default's plain decode** (no exact-attention scope; recorded, no bar): equal to the speculative tokens on all 40.
- **G5 on the 9B:** the checkpoint's 50.2 MiB untouched by a speculative turn; the next turn reused 119 of 119 positions (A's
  prompt and output); the state after A byte-equal to a cold prefill of A's prompt and output; warm B equals cold B.

**Verdict: G1 passes (G1a, G1b, G1c), and G5 passes.** The path is correct on the real pairing; the PR opens on it.

**An observation, not a gate:** on the two long prompts the acceptance guard stopped speculating after its 6-round window (the
turns finished on plain steps, and identically). A diagnostic on two prompts of about 1,000 tokens (a source file and a question)
showed the same, and rules out the chunked seed: seeding in 512-row and in 256-row passes gave the same rounds and tokens. The
drafter accepted 1–4 tokens a round behind 1,000 tokens of code (mean about 2) against about 4 on the bare question. So DFlash's
acceptance on this pairing falls with context; G4's long cell reads what that costs.

### Review fixes before G2 and G4 (2026-10-25, from the PR's second-opinion review)

- **The fold path is prepared before the drafter attaches.** `NewBlockSpec` now asks the resident to prepare the arena
  (`PrepareRecurrentSpec`) and checks admission before `AttachBlockDrafter`, so a recurrent model the path refuses (Mamba-2,
  heads past 128, a sliding window, an arena that does not fit) gets no drafter weights on its device. The arena's price
  (`RecurrentSpecArenaBytes`) is 0 for each model the path refuses, so the fit guard reserves nothing for them.
  `TestPrepareRecurrentSpec_refusesBeforeAttach` pins each refusal.
- **G4's window and void rule.** Both arms time from the first token out to the end of the turn. BlockSpec folds the prompt into
  its drafter's context before its first token (`BlockSpec.generate`), so that work is outside the window in both arms; the last
  forward and the commit are inside it in both, since plain decode forwards its last token before its channel closes. The
  registered void rule is enforced in G2 and G4: a cell or pair whose start reads load above 8.0, or available memory under 15.3
  GB (the identity run's peak resident memory for this load) plus 2 GB, is recorded as void and re-run in order.
- **Plants and the guard trip** are armed through helpers that disarm them with `defer`, so a failing helper cannot leave a later
  test planted.

**G1c re-run on the fixed tree: PASS** (2026-10-25, 00:57–01:05 PDT, at `2df07347`, the arena now prepared before the drafter's
weights): 40 of 40 cells token- and state-identical, the guard's fallback 4 of 4, P1 and P2 each caught on 12 of 12 prompts, G5
passing; the same 1.15 GB free with everything attached; peak resident 15.3 GB, peak swap 0. Raw: `raw/identity-2026-10-25-0057.jsonl`.

### Second review's fixes, and a G4 smoke run (2026-10-25, 01:30–02:10 PDT)

- **No arena for a refused architecture.** The admission check (`RecurrentSpecArenaBytes`, 0 for every model the path refuses)
  now runs before the resident is asked to prepare, so a Gated-DeltaNet model with a sliding window or wide heads allocates
  nothing. `TestPrepareRecurrentSpec_refusesBeforeAttach` counts the preparations asked for: 0 for Mamba-2, a sliding window and
  value heads of 256. Mutation-checked: moving the check after the preparation fails those three cases.
- **G4's plain arm runs under the exact-attention scope** BlockSpec holds (the plain greedy G1 compares with), and every
  prompt's tokens are compared between the arms of each pair; a difference fails the test. The long cell's prompts (behind a
  3,800-token preamble) are not in G1c's set, so this is their identity check. The load record notes whether the multi-row
  flash-decode lane serves verify rows (then the scope is a no-op in both arms).
- **The void rule's memory term** is the 2 GB margin alone: every cell starts with the load already resident in the process.
- **A third pre-existing BlockSpec limit, found by the smoke run:** the seed's capture went to the drafter in one `FuseContext`
  call, whose device scratch grows with the rows, so a 3,900-token prompt went out of device memory. The loop now fuses in
  512-row pieces; the fusion is per row and `ExtendContext` appends, so the context is the same.
- **The G4 smoke run** (`SP2_SMOKE=1`, one pair over two prompts per context; not G4's reading) ran both contexts to the end with
  tokens equal in both arms.
- **Checks on the fixed tree:** tier 0 on all three models, `touched.py` and the scan-table test green; the dense pairing's
  gates on Qwen3-4B pass (`TestDFlashLoop_lossless`, `TestGenerateBlockSpec_production`: 96 tokens identical, 5.65 tokens a
  round, 1.49×; `TestBlockSpecStream`).

### Third review's follow-ups (2026-10-25, 02:30 PDT)

- **The 512-row fuse pieces at tier 0:** `TestBlockSpecGenerate_fusesALongSeedInPieces` drives a 1,100-row seed through the loop
  and reads the drafter's calls (512, 512, 76, then the trailing token's 1) and its context (every position once, in order).
  Mutation-checked: one unlimited piece fails it.
- **Widths:** the fold path is prepared for the drafter's widest verify (its block, at most 16 rows), admission checks that
  width, and a one-position block is refused before anything is asked of the resident. The CUDA arena is replaced by a wider
  one when a later drafter asks for more rows (`TestRecurrentSpec_arenaPriced` asks for 4 rows, then 16).
- **G2 and G4 print their verdict** against the registered bands (`VERDICT G2 L=…`, `VERDICT G4 <context>: …`) and fail the
  run on G2's FAIL band and on G4's not-advanced band; a parked G4 context is reported, not failed.

### G2 (2026-10-25, 06:29–06:36 PDT, at `cdbe6459`): PASS, FLAGGED at both lengths → the fold moved into the next pass's scan

`MODE=cost`, the exclusive timing lock, started on the owner's word (Francis, 2026-10-25 6:18 am: "start your G2 and G4 timed runs
now"). Cell starts at load 7.55 and 6.08; no void cell; peak swap 0. Raw: `raw/cost-2026-10-25-0629.jsonl`.

| L | median step | fold of n = 1 / 2 / 3 / 4 rows | at 3.30 rows | of a step | band |
|---|---|---|---|---|---|
| 512 | 15.61 ms | 1.498 / 1.533 / 1.582 / 1.627 ms | 1.595 ms | 10.2% | 5.0–11.0%: passes, flagged |
| 4096 | 16.94 ms | 1.496 / 1.536 / 1.583 / 1.628 ms | 1.597 ms | 9.4% | 5.0–11.0%: passes, flagged |

The fold costs about 1.5 ms whatever its rows: 48 launches and a load and store of the 48 MiB state, the fixed part. The registered
response to a flagged G2 is that the fold moves into the next pass's scan before G4 is read, so G4, which had started, was stopped
in its warm-up (no pair timed; `speed-2026-10-25-0636.log` on nobara) and the fold was moved:

- The verify twins fold the rows the previous verify marked into the window and state, store them once, and then scan their own
  rows without writing them. `FoldVerified` marks rows; `FlushFold` folds them eagerly through prefill's kernels where something
  other than a verify comes next (the guard's plain steps, the trailing forward, the commit).
- The arena keeps two banks, since a verify overwrites the bank it writes while the previous verify's rows fold from the other:
  63,373,312 B allocated on the 9B, priced at 65,011,712 B (62 MiB). This doubles the 30.1 MiB the registration stated, a
  consequence of the registered response.
- Every committing kernel's source is main's, unchanged (`cuda/deltanet.cu` gains only additions), and their PTX matches main's
  apart from label numbers. The 128-wide twin is one loop over the folded rows and its own, and spills 96 B/thread, the committing
  kernel's own figure.
- G1a's new case checks the in-scan fold byte for byte against a prefill, and the next verify's argmax against the prefill's, on
  all three small models; both plants are caught there too.

### G1c on the in-scan fold (06:48–06:57 PDT, at `721cce5e`): PASS

40 of 40 cells token- and state-identical to plain greedy, the guard's fallback 4 of 4, P1 and P2 each caught on 12 of 12 prompts,
G5 passing. Free VRAM with everything attached 1.11 GB; resident context 4,608 kept; peak resident 15.3 GB, swap 0. Raw:
`raw/identity-2026-10-25-0648.jsonl`.

### G2 on the in-scan fold (06:57–06:58 PDT, at `721cce5e`): PASS, FLAGGED

Cell starts at load 1.34 and 1.29; no void cell; peak swap 0. Raw: `raw/cost-2026-10-25-0657.jsonl`. The in-scan fold's cost is a
verify that folds n marked rows less a verify with none (median of 20 interleaved repetitions each).

| L | step | verify, nothing to fold | verify folding 1 / 2 / 3 / 4 rows | fold at 3.30 rows | of a step | band | the separate fold, same run |
|---|---|---|---|---|---|---|---|
| 512 | 15.58 ms | 21.35 ms | 22.27 / 22.29 / 22.35 / 22.32 ms | 0.996 ms | **6.4%** | passes, flagged | 1.596 ms, 10.2% |
| 4096 | 16.93 ms | 27.69 ms | 28.57 / 28.62 / 28.67 / 28.71 ms | 0.994 ms | **5.9%** | passes, flagged | 1.595 ms, 9.4% |

- **Verdict: G2 passes the comparison with R-01 phase 1's 11.0% at both lengths, flagged (5.0–11.0%).** The registered response to
  the flag is done.
- What is left is the one store of the committed state per cycle: a verify that folds costs what today's verify costs (22.2 ms at
  512, which stores the state too). That store is the commit itself, so there is no further copy to remove.

### G4, speed with DFlash against plain decode (06:58–08:30 PDT, at `721cce5e`): short PASS, long PARKED

`MODE=speed`, the exclusive timing lock. 10 A B B A pairs per context, each pair one pass of each arm over SP0's 12 code and math
prompts, 256 new tokens, decode timed from the first token to the end of the turn. Every pair started at load 2.0 or below with
51 GB or more available: no void pair. Peak resident 14.1 GB, swap 0. The tokens were equal in every pair, prompt for prompt (2,917
decode tokens in each arm short, 2,853 long). The exact-attention scope was in force in both arms (the multi-row flash-decode lane
was off for verify on this load). Raw: `raw/speed-2026-10-25-0658.jsonl`.

| context | plain | DFlash, w = 4 | median speedup | pairs above 1 | spread of the pairs | registered verdict |
|---|---|---|---|---|---|---|
| short (SP0's prompts, about 100–250 tokens) | 65.3 tok/s | 95.5 tok/s | **1.462×** | 10 of 10 | 1.453–1.462 | **PASS** (≥ 1.30×, ≥ 9 of 10) |
| long (the same behind a 3,800-token preamble) | 58.7 tok/s | 70.9 tok/s | **1.207×** | 10 of 10 | 1.206–1.208 | **PARKED** (1.20–1.30×, to the owner) |

- **Against SP0's projection:** 1.55× at 512 and 1.29× at 4,096 projected; 1.46× and 1.21× read. The read numbers carry the fold
  (5.9–6.4% of a step) that the projection left out, and the long cell's acceptance falls with context (the observation under G1c).
- **What the parked reading means:** at long context SP2 is still faster than plain decode in every pair (1.21×), and lossless,
  but under SP0's 1.30× bar; the registration sends it to the owner.

**Verdict: G1, G3 and G5 pass; G2 passes, flagged, with its registered response done; G4 passes at short context and is parked at
long context.**

**The owner's decision on the parked cell (Francis, 2026-10-25, "A"):** #232 merges on these readings: lossless, PASS at short
context, and faster than plain decode in every long-context pair. The follow-up is a small PR adding a `serve` option for the verify
width (serve verifies at width 8, where SP0 projects 1.17×), and a served A/B, registered before it runs.

### G4 graded against the owner's merge rule (Francis, 2026-10-25, 8:46–8:48 am)

The rule, set after this gate was registered: a lossless, opt-in feature merges when it is consistently faster, a median of 1.05× or
more with at least 9 of 10 pairs faster, and its PR states the memory cost. SP0's 1.30× is an investment bar and is not a merge bar.
G4 registered SP0's 1.30× for an opt-in (`--drafter`), lossless (token-identical) change; the readings above stay as recorded, and
are graded against the rule here:

| context | median | pairs faster | the merge rule |
|---|---|---|---|
| short | 1.462× | 10 of 10 | **merges** |
| long | 1.207× | 10 of 10 | **merges** (the owner: "1.21x gain there!!!") |

**The memory cost:**
- **Device:** the arena, on a Gated-DeltaNet target with a drafter attached: 63,373,312 B allocated on Qwen3.5-9B, priced in the
  fit guard's companion reserve at 65,011,712 B (62 MiB). Nothing on any other model (its price is 0), and nothing without
  `--drafter`. The drafter's own weights and K/V are `--drafter`'s existing cost, unchanged.
- **Host:** nothing persistent. A long seed's captured rows are joined on the host across passes (rows × taps × hidden × 4 B, about
  511 MB for a 3,900-token prompt on the 9B), the same bytes the single pass downloaded before; the drafter's device scratch for
  them is now bounded by 512-row pieces.
- **The longest context that fits:** with a drafter attached, an auto-sized context shrinks by about 567 positions (65,011,712 B
  over a position's 65,536 B of f32 target K/V plus 49,152 B of drafter K/V, from the fit guard's own prices; computed, not
  measured). The 4,608-position context SP0 and these gates pinned still fits, with 1.11 GB free.

### Fifth review's fix: an early exit leaves nothing to fold (2026-10-25, 09:49–09:59 PDT, at `145913f2`)

- **The defect:** a speculative turn that stopped after a round's `FoldVerified` (a cancelled stream, a device or drafter error) left
  its rows marked, and the next speculative turn's first verify folded them into the new sequence.
- **The fix:** `BlockSpec.generate` drops marked rows on every exit that commits nothing (`DropFold`, deferred), and the resident
  clears them whenever it resets its recurrent state (a prefill from position 0). The verify width also clamps to the fold path's
  16 rows on a recurrent target, so a drafter with a wider block cannot attach and then fail at its first verify.
- **The reproduction:** `TestRecurrentSpec_earlyExitLeavesNothingToFold` stops a turn right after the first round's fold, then
  runs a second speculative turn and checks it against plain greedy, tokens and state bytes. It passes on all three small models
  with the fix. With both halves of the fix removed it fails on all three (divergence at token 3 or 4, the conv window's bytes
  differ), so it sees the defect. `TestBlockSpecGenerate_earlyExitDropsTheMarkedRows` pins the drop at tier 0, mutation-checked.
- **G1c on the fixed path: PASS.** 40 of 40 cells token- and state-identical, the guard's fallback 4 of 4, P1 and P2 each caught
  on 12 of 12 prompts, G5 passing; peak resident 15.5 GB, swap 0. Raw: `raw/identity-2026-10-25-0950.jsonl`.
- **G2 and G4 stand as read.** The fix acts only on an exit that commits nothing and on a reset with rows pending; every G4
  generation ran to the end and flushed, G2 drives the resident directly, and the width clamp starts above 16 rows (G4 ran at
  4). So both readings came from the code path the fix leaves unchanged, and neither is re-run.

## The served A/B of `--drafter` (registered 2026-10-25, before the run)

The follow-up the owner's decision named. `serve -drafter-width` (this branch) sets the width `BlockSpec.GenerateStream` verifies at;
before it, `serve` verified at 8, where SP0 projects 1.17×.

- **The question:** does the served path, a user's `serve --drafter`, deliver what G4 read in process?
- **Instrument:** `scripts/bench_peer.py`, two goinfer arms on one CUDA `serve` binary built at this branch's commit:
  - `goinfer` is plain `serve` with its own defaults (the KV precision a user gets without a drafter, recorded from the log);
  - `goinfer_old` is the same binary with `BENCH_GOINFER_OLD_ARGS="-drafter ~/models/qwen35-9b-dflash -drafter-width 4"` (f32 KV,
    which a drafter forces).
  - `BENCH_MODELS=Q9` (Qwen3.5-9B Q4_K_M; serve loads its `.giw` sidecar), `BENCH_BACKENDS=cuda`, `BENCH_DEPTHS=512,4096`,
    `BENCH_NGEN=256`, `BENCH_CTX=4608` (G1c's context), `BENCH_ABBA=5`: 5 mirrored quads per depth, 10 pairs, each cell a
    restarted server, the harness's warm-up, then 2 runs of 8 completions. `BENCH_PHASES=B` (added to the harness on this branch,
    before the run): the depth cells only, without the backend table's depth-128 cell, which this model has no prompt for.
  - **Prompts:** the harness's essay-v2 prompts, token-calibrated for Q9 at 512 and 4,096 with `scripts/bench_prompts_calibrate.py`
    and committed to `scripts/prompts.json` before the run. Prose, where SP0's prompts were code and math.
  - Greedy (temperature 0), so every request takes the speculative path.
  - **Amended 2026-10-25, 10:24, on the owner's decision (Francis: option A), after seeing data:** both arms run with thinking off,
    `BENCH_GOINFER_ARGS="-thinking off"`; everything else as registered. The amendment came after the first start's 2 cells at depth
    512 (1.02×) and the in-process diagnostic recorded below: Qwen3.5-9B's template renders thinking on by default, and DFlash is
    trained on non-thinking output. G4 rendered its prompts with thinking off, so this is what makes the A/B measure what G4
    measured. What `serve --drafter` gives on this model's default template is recorded by that diagnostic, not by this gate.
- **The reading, per depth:** each pair's drafter tok/s over plain tok/s; the median, the count of pairs above 1, and the pairs'
  spread.
- **Bands, per depth: the owner's merge rule** (Francis, 2026-10-25, 8:46 am; amended here before the run, which had not started;
  first written with SP0's 1.30×, an investment bar that a merge gate must not inherit): `--drafter` is opt-in and lossless, so a
  median of 1.05× or more with at least 9 of 10 pairs faster → the served path delivers the speedup at that depth; anything less →
  it does not, at that depth.
- **What each outcome means:** delivering is the claim a user can rely on, stated with the memory cost (the arena and the drafter,
  and the drafter arm's f32 KV against the plain arm's default). A shortfall where G4 passed is a served-path gap to explain, from
  the KV precision, the HTTP stream or the prose prompts; it does not revisit G4. Nothing here changes `serve`'s default width (8):
  a default change needs the gain judged against its memory trade-off, and the owner's call.
- **Void cells:** the harness's own gates (the 1-minute load cap and its memory and swap voids) and the registered rule (load
  above 8.0 at a cell's start, or available memory under 2 GB with the model loaded) void a cell, re-run in order.
- **Cost:** about 100 s a cell (load and attach 40 s; 16 completions of 256 tokens), 4 cells a quad, 5 quads, 2 depths: about 70
  minutes, plus the 4,096 prefills. Through the timing lock, by day.
- **Tier:** 2 (a number that will be quoted). Tiers 0 and 1 cannot answer it: it is the served end to end.

### The served A/B's first start (2026-10-25, 10:08–10:13 PDT): stopped by me after 2 cells, not graded

- Two cells ran at depth 512: plain 63.6 tok/s, the drafter arm 64.6 tok/s (1.02×). The drafter arm's record showed `"sent": {}`, and
  I stopped the run believing its requests went out without `temperature: 0`. They did not: `goinfer_payload` sends goinfer's
  sampling for both goinfer arms, so the run was valid and only the record was wrong (fixed: a `goinfer_old` cell now records what
  it sent). The two cells and the serve logs are kept on nobara (`~/goinfer-logs/sp2-2026-10/served-void-1008/`).
- **Why 1.02×** (a diagnostic in process, not a gate, on the same essay prompts): Qwen3.5-9B's template renders thinking on by
  default (`serve -h`: "Qwen3.5-9B: on"), and both replies open with `<think>`. DFlash is trained on non-thinking output. At 512 the
  drafter accepted 3.1 tokens a round (4.81 s against plain's 5.47 s, prefill included); at 4,096 it accepted 1, 1, 1, 1, 4, 1, the
  guard switched to plain decode after its 6 rounds, and the turn ended slower than plain (23.0 s against 20.0 s), the long seed's
  capture being the extra.
- So the A/B as registered (the template's default) measures thinking-mode replies, which G4 did not (G4 rendered thinking off).
  How to proceed is the owner's decision; nothing is re-run until it is made.

**Correction (2026-10-25, 10:40), to the diagnostic above and the amendment's stated reason.** The diagnostic rendered its prompts in
the chat package's bare default mode (`asis`: nothing written after `assistant\n`, the model decides), not in the mode `serve` applies,
`-thinking template`. For this GGUF (unsloth's), `template` renders the closed `<think>\n\n</think>\n\n`, thinking off, exactly as
`-thinking off` does (read from the GGUF's own template, and from goinfer's rendering of it in both modes). So:
- the first start's two cells (1.02× at 512) already ran with thinking off, and thinking does not explain them;
- the amendment to `-thinking off` changes nothing for this file: the run measures the same condition it registered, and remains a
  valid reading of `serve --drafter` with thinking off on the essay prompts;
- the load banner's "template default off" is right for this file; `serve -h`'s "Qwen3.5-9B: on" describes the Hugging Face
  checkpoint's template, which opens `<think>\n` by default.

What the 1.02× does come from is open. The likeliest cause is the prompt class (essay prose here, code and math in G4); an
in-process run of the same prompts in `serve`'s rendering will say, after this A/B finishes.

### The served A/B, read (2026-10-25, 10:26–12:48 PDT; binary `a6e9b8c7`, scripts at `300e73c6`): does not deliver on these prompts

`bench_peer.py` as registered and amended (both arms `-thinking off`), the timing lock held by the harness for the whole run. 5
mirrored quads per depth, 10 pairs; every cell started at load 0.98 or below; no void cell, no error. Raw:
`raw/served-ab-2026-10-25.json` (per-cell runs, machine state, serve logs on nobara under `~/goinfer-logs/sp2-2026-10/served/`).

| depth | plain `serve` | `serve --drafter -drafter-width 4` | median of the pairs | pairs faster | spread | the merge rule |
|---|---|---|---|---|---|---|
| 512 | 63.6 tok/s | 64.6 tok/s | **1.016×** | 10 of 10 | 1.011–1.017 | **does not deliver** (median under 1.05×) |
| 4096 | 62.1 tok/s | 61.4 tok/s | **0.989×** | 0 of 10 | 0.987–0.990 | **does not deliver** (slower) |

- **An overlap, recorded:** at 10:34:50–10:35:10 I ran `go test ./cmd/gate/` (about 18 s of CPU) without taking the timing lock,
  inside the plain arm's depth-512 cell of block 1. That cell read 63.5 and 63.6 tok/s, as every plain cell at that depth did; the
  median and the count are the same with that pair dropped (1.016×, 9 of 9).
- **Why, from a diagnostic in process** (not a gate; the same essay prompts rendered as `serve` renders them, thinking off;
  `BlockSpec.GenerateStream` at width 4, the call `serve` makes): 1.021× at 512 and 0.894× at 4096, so the served path behaves as
  the in-process path does. The drafter accepted 2.43 tokens a round over 21 rounds at 512 and 2.27 over 26 at 4096, under the
  acceptance guard's 2.5, so the guard switched each turn to plain decode partway. At 4096 the long seed's capture and the slower
  long-context verify cost more than the few accepted drafts saved.
- **What it means:** the served path delivers what the drafter's acceptance allows. On essay prose this drafter accepts too little to
  pay; on SP0's code and math it accepted 3.3 and more per round, and G4 read 1.46× in process. So the reading here is about the
  workload, not a served-path gap. It does not revisit G4, and `serve`'s default width stays 8.
- **What would answer G4's own condition served:** the same A/B on code and math prompts (SP0's suites, calibrated to depth). That is
  a separate run; it is proposed, not run.

### The essay A/B end to end, and `--drafter`'s host memory (2026-10-25, from #238's second-opinion review)

The headline above is decode tok/s (`decode_timed_s`). The same cells' phases, medians of each arm's 10 cells, 16 timed completions a
cell (`raw/served-ab-2026-10-25.json`, `counts.phases` and `counts.rss_peak_kb`):

| depth | arm | wall of the 16 completions | prefill + request | per request (prefill + TTFT) | decode | peak RSS |
|---|---|---|---|---|---|---|
| 512 | plain | 88.2 s | 24.0 s | 1.50 s | 64.2 s | 4.7 GB |
| 512 | `--drafter` | 92.7 s | 29.6 s | 1.85 s | 63.1 s | 12.8–12.9 GB |
| 4096 | plain | 69.1 s | 3.4 s | 0.21 s | 65.7 s | 5.1–5.2 GB |
| 4096 | `--drafter` | 364.8 s | 298.4 s | 18.65 s | 66.4 s | 13.9–14.1 GB |

- **End to end, `--drafter` lost on these prompts:** 0.95× of plain's wall at 512, and 0.19× at 4096 (5.3× slower).
- **Why 4096 is so far apart: prefix reuse.** The harness sends one identical prompt 17 times a cell. Plain `serve` restores the
  prompt's prefix from a host snapshot after the first request (0.21 s a request). `BlockSpec` restores nothing: its drafter needs
  its own context for any reused prefix, and the snapshots carry the target's rows and recurrent state but not the drafter's, so
  `BlockSpec.generate` sets its reuse to 0 unless the prompt extends its own last turn (`resDrafterSynced`). Every request
  re-prefills all 4,131 tokens on the exact kernels a recurrent model's prefill takes, with the drafter's capture on top: 18.65 s.
  An agent loop, whose turns extend the last, does reuse (G5 measured it: 119 of 119 positions); an identical resend, a retry or a
  second conversation does not.
- **Why the in-process diagnostic read 0.894× and the served headline 0.989×:** the diagnostic timed from the first token to the end
  of the turn, so it held the trailing work and not the prefill; the served headline is decode windows only. Neither held the
  prefill, which is where `--drafter` loses most here.
- **Is the missing reuse a bug?** Not a correctness one: every turn is token-identical to plain greedy, and the cold prefill is the
  defined fallback. It is a missing capability that costs `--drafter` most of its served value on repeated or re-sent prompts:
  snapshots that carry the drafter's context, so `BlockSpec` can take the same prefix plain `serve` takes. It would also bear on the
  code/math A/B now running, whose prompts repeat within a cell in the same way.
- **Host memory: 8–9 GB more at peak with `--drafter`, and no guard prices it.** The drafter's weights are a mapped bf16 file (2.6 GB
  of pages), and attaching widens each matrix to an f32 copy before quantizing it to int8 for upload (`packWeight`), so the attach's
  transient copies sit on top. The peak is `VmHWM`, so how much of it outlives the attach is not yet measured. The load-time host
  guard prices the target model only (`modelload.Load`); `serve` loads the drafter before it and attaches after, so the drafter's
  host memory reaches no fit check. The harness's `fit_need_mb` (5,417, the checkpoint's file size) is the harness's own figure for
  the void rule, not `serve`'s.
- **The void rule's memory term was not applied** in either served A/B: `free_mem_mb` is null in every cell because the harness
  reads it through `go run ./cmd/hostmem` and my run scripts did not put Go on `PATH`. Reconstructed, not recorded: this box's
  `MemAvailable` reads about 60 GB with a served 9B up, against a 14 GB peak for the drafter's process, so no cell came near the 2 GB
  floor. The harness now refuses at preflight when it has no reading (`BENCH_ALLOW_NO_FREE_MEM=1` runs anyway, with a warning, and the
  provenance records it), and later runs set `PATH`.
- **The 3.4 MB of swap-in in a depth-4096 cell** is the system's counter over a 6-minute cell, with zero swap growth; the
  registered rule has no swap term (the harness's swap voids are off unless set), so it voids nothing.
