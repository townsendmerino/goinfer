# CUDA `--kv i8`, S4: fidelity (G4) and speed, registered before any run (nobara-sb01, 2026-11-07 20:05 PST)

S4 of `scope.md`'s staged plan, for the dense families S2 and S3 deliver (llama, mistral, qwen2, qwen3, phi3 without a sliding window). G4 is
the task doc's (`docs/tasks/task-kv-precision-everywhere-2026-10.md` §4): **a published reading, not a refusal**. The contract's item 4 holds:
`--kv i8` asked is delivered whatever this reads. The reading decides what the docs, the banner and the matrix say about int8 quality, and it
moves no default (`auto` stays as it is; int8 is never an `auto` pick). Nothing here is run until this file is committed.

## G4: fidelity against f32, with f16 beside it

**Instrument:** `TestKV8GateVsReference` (written in `backend/cuda/kv16_gate_test.go`, beside and on the shape of `TestKV16GateVsReference`, whose
helpers it reuses unchanged). Three arms in one binary and session, loaded one after another (a 7B's copies do not fit the card together):
`-kv f32` (exact), `-kv f16` and `-kv i8`. Each is teacher-forced on the reference tokens and scored against the **already built** CPU f32/f64
reference rows of held-out prompt set B (`~/goinfer-logs/prefill-ref-b`: `D7-K8000-p0..9`, `S-K3900-p0..9`; ten prompts of 64 positions).
All arms run the production decode lane (`GOINFER_CUDA_FLASH_DECODE=16`), so the gate measures int8 KV as it ships. Set B has been scored for
the flash lane and for f16; this candidate has no tunable parameter, so B is not fitted to it, but it is not unseen for the instrument.

**Cells:** D7 (qwen2.5-7b-instruct-q4_k_m) at K=8000, the decision cell; S (qwen2.5-coder-1.5b-instruct-q4_k_m) at K=3900, the confirmation
cell, which cannot rescue a D7 result.

**Bars per cell, all three, int8 against f32 (the f16 doc's, unchanged):**
- (a) hard flips: i8 <= f32 + 2 * sqrt(f32);
- (b) agreement, paired: i8's mean teacher-forced agreement >= f32's mean - 1.0 pt, AND i8 >= f32 on at least half the prompts;
- (c) KL: i8's mean continuation KL <= 1.10 x f32's.

**Ambiguous is parked:** (b) within its last 0.2 pt, or (c) in 1.05 to 1.10x. A re-run needs a named mechanism, never a re-roll. A (c) pass
whose paired delta sits more than 2 standard errors above zero is reported with the delta and its s.e. The same three numbers are reported for
int8 against f16, as a second reading, with no verdict of its own.

**Preconditions** (a failed one voids the cell; it is not a fail):
1. Vacuity: the int8 arm's K/V allocation is 1 B an element plus 8 B of scales per position, head and layer (read from the build); the
   flash lane's `_kv8` kernel is launched; the int8 decode rows differ from f32's at one position or more of every prompt.
2. A/A: two int8 runs on the first prompt agree bit for bit over all 64 rows.
3. Seed row: the first (prefill-seeded) row is compared and reported, not required equal.
4. `TestKV8_G2exact_readersMatchTheCPUOnOneCache` passes at the binary's commit.
5. Fit: every load the gate makes is fit-checked first by the server's own plan (seconds), and a refused load is not run.

**Prediction (written now):**
- S passes (0.65).
- D7 reads FAIL or PARKED (0.6). f16 itself read PARKED there on (b), and int8 is the coarser cache.
- int8's mean KL against f32 is over 1.10x at D7 (0.5).
- On the 0.5B (not a cell), int8 costs about 20 top-1 points against f32 (from S2's cross-backend G2). That is why the cells are the 1.5B
  and the 7B.

**Cost:** the f16 gate's D7 cell took 1,110 s for two arms, so three arms take about 1,700 s; S about 5 minutes. About 35 minutes in all,
by day, under the exclusive timing lock, one prompt-set read per arm, no early stop.

## Speed: int8 against f16, paired, interleaved

**Instrument:** `scripts/bench_peer.py`, the goinfer-only two-arm form, one binary for both arms. `BENCH_ENGINES=goinfer,goinfer_old`, with
`GOINFER_SERVE_*_OLD` pointed at the same binary. The `goinfer` arm is the default (`--kv auto`, which is f16 on CUDA for these families):
the do-nothing arm, the precision int8 would replace on request. The `goinfer_old` arm adds `-kv i8` (`BENCH_GOINFER_OLD_ARGS`). Served,
greedy, a fresh serve per arm, alternating A B B A, one timed run on the box (the timing lock), the harness's idle gate and the void rule of
CLAUDE.md ("Run budget"): a cell started over load 8.0 or under the fit guard's need plus 2 GB is void and re-run in order.

**Cells:** decode at depths 128, 512, 4,096 and 16,384, on qwen2.5-coder-1.5b-instruct-q4_k_m and qwen2.5-7b-instruct-q4_k_m. Each depth is
fit-checked for both arms first. A depth the f16 arm cannot hold on the 8 GB card is dropped and named, not stretched. The run script, with
its bench keys, rounds and the binary's commit, is committed before it runs.

**Bars**, on the median int8/f16 decode-rate ratio per cell:
- **No regression at short context** (the task doc's speed bar): at 128 and 512 keys, int8/f16 >= 0.95 on both models. Below 0.95 at either
  is a regression, which the record states and the docs carry. In 0.95 to 0.98 it is reported as a small cost.
- **Long context is reported, not gated:** 4,096 and 16,384, where int8 reads half the K/V bytes of f16 and adds a multiply per element.
- **Prefill** (time to first token at each depth) is reported. An int8 cache routes prefill around the tensor-core attention tiles (no int8
  twin), so a TTFT cost at depth is expected and is named, not gated.

**Prediction:**
- 128 and 512 within +-3% of 1.00 (0.7).
- 16,384 on the 7B >= 1.00 (0.5).
- TTFT at 4,096 and over is worse than f16 by more than 10% on the 7B (0.6).

**Cost:** 2 models x 4 depths x 2 arms x 4 runs (A B B A) of a served decode cell. bench_peer's measured cell at these sizes is under a minute,
plus loads and idle-gate waits: about 1.5 to 2 hours, by day, under the timing lock. Over 2 hours goes to the night queue per CLAUDE.md, which
this is not expected to need.

## Order and what each result triggers

1. Write the G4 instrument; run its preconditions on S (the 1.5B) as a smoke test of the instrument only, graded on nothing.
2. G4 on S, then D7. Record and grade against this file.
3. The speed run.
4. The reading goes into `docs/kv-precision-matrix.md`'s notes, the banner's int8 line and `scope.md`; S4 is then whole for the dense
   families, and the flag's remaining refusals (sliding window, other families) are S3c and S5.

## Addendum, 2026-11-07 20:08 PST (logistics only; no bar, cell or prediction changed)

The instrument smoke (step 1) stopped before loading anything (`s4-g4-smoke-S2prompts.log`): the set-B references for S and D7
(`~/goinfer-logs/prefill-ref-b`, built 2026-09-20) have no `.key.json` sidecar. They predate the content-keyed reference cache, so
`PrefillRefCurrentForTest` refuses them, as it should: a gate scores only a reference this tree would compute. The cache holds no
qwen2.5-coder-1.5b or qwen2.5-7b entry. The reference's source hash covers `decoder/`, so the references must be built on the tree the gate
runs on.
- **S (1.5B, K=3900):** rebuilt now on this branch's tree (`TestPrefillGateReference`, `GOINFER_CPU_REF_MODELS=S`, `GOINFER_CPU_REF_KS=3900`,
  set B; CPU, f32 weights and activations, exact attention; the timing lock held shared, `s4-ref-S-K3900.log`). The S cell runs on the same
  tree when it is done.
- **D7 (7B, K=8000):** the September build took about 25 minutes a prompt, so about 4 hours for ten. That is over the 2-hour line, so by
  CLAUDE.md it belongs on the night queue, and nobara is quiet 21:45 to 02:00. Building it is a scheduling decision for the coordinator: a
  night job after 02:00 on a pinned tree, or by day tomorrow. Until then D7, the decision cell, is not run, and S, the confirmation cell,
  cannot stand in for it.

## Results

### S (qwen2.5-coder-1.5b, K=3900, the confirmation cell), 2026-11-07 20:21 PST

References rebuilt on this tree (`s4-ref-S-K3900.log`): 10 of 10, 740 s, keyed to the gate's source. The cell started at a 1-minute load of
6.87 (under nobara's 8.0 void line; this is a fidelity cell, not a timed one) with 60 GB available (`s4-g4-S-K3900.log.start`). It ran 110 s
(`s4-g4-S-K3900.log`).

| reading | base | int8 | criteria |
|---|---|---|---|
| **G4: int8 against f32** | agreement 91.09 %, hard flips 9, mean KL 0.052265 | 89.69 %, 19, 0.086818 | (a) 19 > 9 + 2*sqrt(9) = 15: fails. (b) -1.40 pt, under -1.0, and int8 >= f32 on 4 of 10: fails. (c) KL ratio 1.661: fails. **DOES NOT PASS** |
| int8 against f16 (reported) | 90.94 %, 8, 0.052564 | 89.69 %, 19, 0.086818 | ratio 1.652, the same three fail |
| control: f16 against f32 | 91.09 %, 9, 0.052265 | 90.94 %, 8, 0.052564 | **PASSES**, KL ratio 1.0057: the f16 gate's own S reading of 2026-10-13, reproduced exactly |

Preconditions checked in this run: 1 to 3 (5 and 4 were not; see the addendum of 2026-11-08 below, which makes this reading provisional):
- int8 allocated 60,555,264 B against f32's 234,881,024 B, priced exactly in both arms;
- the flash lane's int8 kernel launched;
- A/A bit-identical over 64 rows;
- int8 rows differ from f32's at 630 positions (every prompt), and seed rows at 10 of 10.

Prediction scored: "S passes (0.65)" was wrong. The confirmation cell reads DOES NOT PASS, so int8 KV's quality cost on the 1.5B is
measurable at this gate's bars: about 1.4 points of agreement, twice the hard flips, and two-thirds more KL. By the contract (item 4) this is
a published reading, not a refusal: `--kv i8` stays delivered on request. It decides what the docs and the banner say, and it is
consistent with what S2's cross-backend readings showed (the CPU's own int8 cache costs the 1.5B the same).

D7, the decision cell, waits on its references (the addendum above).


## Addendum, 2026-11-08 02:29 PST (Gemini's review of #768; no bar, cell or prediction changed)

- **The S reading above is provisional.** Its run checked preconditions 1 to 3 only. Precondition 4 (the exact-input G2 passing at the
  binary's commit) was last run on an earlier tree, and precondition 5 (each load fit-checked first) was not recorded. #768's description
  and this record said "every precondition holds"; that was wrong for 4 and 5. The S cell is re-read with D7 on one pinned tree, with all
  five checked.
- **The gate now enforces 4 and 5.** Precondition 4 runs the exact-input G2 as the gate's first subtest, in the same binary; no cell is
  read if it fails. Precondition 5 prints each arm's fit decision as its build sized it (context, slots, KV pools); `BuildResident`
  refuses a load that does not fit, which fails the cell before any score. Precondition 1 now checks all six precision flags, each arm's
  allocation against its own price (f16's included), and the flash lane's launches on every arm. A missing checkpoint fails the cell
  instead of skipping it.
- **Checked on the new instrument** (`s4-instrument-check-S2.log`): precondition 4 PASS (0.5B, 1.5B, `llama-attnfa-tiny`). The S cell
  then stopped at the reference currency check, as it should: merging main changed `decoder/`, so S's references are stale again. They
  are rebuilt on the tree D7 runs on.
- **Budgets, re-estimated.** On S the int8 arm took about 5x the f32 and f16 arms per prompt (49 s for 11 runs against 9 s for 10): an
  int8 cache prefills through `attn_batched` instead of the tensor-core tiles, and at 8,000 keys the ratio can be larger.
  - D7's three arms: about 70 to 120 minutes, not the 35 minutes written above.
  - The speed check, whose 16,384-key 7B cells carry int8's slow prefill: about 2 to 3 hours.

  Past 2 hours a job belongs on the night queue (CLAUDE.md "Run budget"). D7's reference rebuild (about 4 hours of CPU) and both runs
  are Francis's call; none is started.

## Addendum, 2026-11-08 02:51 PST: D7 and the speed check run now (Francis, 02:50 via the coordinator: "run it now", replacing "by day tomorrow")

- **Pinned tree: `7545a43cf`** (#768's head, with origin/main merged in, up to #769 at ebf07042). Every step below runs from a detached
  worktree at that commit (`~/goinfer-bench/s4-pinned/wt`), with binaries built there once (`~/goinfer-bench/s4-pinned/bin/`: the
  reference generator `decoder.test` and the gate `cuda.test`). Commits after it on #768's branch touch only `docs/`, so the
  reference source (`decoder/`, `internal/runtime/giw`, `constrain`) is the pinned tree's.
- **Order and estimates** (logs durable in `~/goinfer-bench/s4-pinned/logs/`, run detached):
  1. References, set B, on the pinned tree: S at K=3900 (about 13 min), then D7 at K=8000 (about 4 h of CPU), with the timing lock held
     shared (a load, not a timed run).
  2. The gate on the same tree, the exclusive lock held: S re-read (about 3 min), then D7 (about 70 to 120 min). All five preconditions
     are enforced in the binary.
  3. The speed check as registered (about 2 to 3 h), the exclusive lock held, nothing else of mine running on the box.
- **Expected:** references about 03:00 to 07:15; gate about 07:15 to 09:15; speed check about 09:15 to 12:15 PST. Times are re-stated in
  the status file as each step starts.
- No bar, cell or prediction changes. A refusal (fit, memory) stops the run and is reported, never bypassed.

## G4 results on the pinned tree `7545a43cf`, 2026-11-08 (nobara, RTX 2070 SUPER, driver 595.112.07)

All five preconditions were checked in the gate binary, built once from the pinned tree. Logs and the run scripts are in
`s4-pinned-7545a43c/`. Checkpoints came from `~/models` on NVMe.

**References, set B, rebuilt on the pinned tree** (`run-refs.log`):
- S at K=3900: 10 of 10, 714 s.
- D7 at K=8000, pass 1 (`refs-D7-K8000-pass1.log`): 8 of 10 stored, each about 3 h 55 m with 8 running in parallel. The test's own deadline
  guard then declined the last 2, which would have ended past its 6 h timeout, and exited 1. That is the reference test's resume path, not a
  fit or memory refusal.
- The resume (`refs-D7-K8000.log`) found the 8 cached and computed the last 2 in 43 minutes. PASS.
- The 4-hour estimate was low: about 4 h 40 m of CPU in all.

**The gate** (`gate.log`):
- It started at 07:43 at a 1-minute load of 10.6, the tail of the reference build. These are fidelity cells, not timed ones, and 61 GB was
  available.
- It ran 1,014 s: precondition 4 took 11 s, S 108 s and D7 895 s. The "70 to 120 minutes" estimate for D7 was high by about 5x: int8's slow
  prefill on S did not scale with the key count as feared.

### D7 (qwen2.5-7b-instruct-q4_k_m, K=8000, the decision cell): **AMBIGUOUS — PARKED**

| reading | base | int8 | criteria |
|---|---|---|---|
| **G4: int8 against f32** | agreement 93.12 %, hard flips 11, mean KL 0.055900 | 92.97 %, 11, 0.060271 | (a) 11 <= 11 + 2*sqrt(11) = 17.6: passes. (b) -0.15 pt, inside -1.0 and not within its last 0.2 pt, and int8 >= f32 on 7 of 10: passes. (c) KL ratio 1.0782: in the 1.05 to 1.10 band. **AMBIGUOUS, PARKED** |
| int8 against f16 (reported) | 92.34 %, 11, 0.059082 | 92.97 %, 11, 0.060271 | ratio 1.020, +0.63 pt, int8 >= f16 on 9 of 10: inside all three bars (no verdict of its own) |
| control: f16 against f32 | 93.12 %, 11, 0.055900 | 92.34 %, 11, 0.059082 | (a) passes; (b) -0.78 pt, just outside its last 0.2 pt; (c) KL ratio 1.0569, in the band. **PARKED** on (c) |

Preconditions:
1. Vacuity:
   - int8 allocated 242,221,056 B against f32's 939,524,096 B. That is exactly 1 B an element plus 8 B of scales per position, head and
     layer, over the 8,192 pool rows that 8,128 positions round up to in 64-row blocks.
   - The flash lane's `_kv8` kernel launched.
   - int8 rows differ from f32's at 630 positions, every prompt.
2. A/A: bit-identical over 64 rows.
3. Seed row: compared, and it differs on 10 of 10 prompts (reported).
4. The exact-input G2 passed at the binary's commit: `llama-attnfa-tiny`, 0.5B, 1.5B.
5. Fit: each arm's build sized it resident, with 1 slot, a context of 8,128 positions, and KV pools of 939,525,112 B (f32), 469,763,064 B
   (f16) and 242,222,072 B (i8).

### S (qwen2.5-coder-1.5b, K=3900, the confirmation cell): **DOES NOT PASS**, now with every precondition

The re-read reproduces the 2026-11-07 20:21 reading exactly: every agreement, flip count and KL figure in the table above is unchanged.
- int8 against f32: 89.69 % / 19 / 0.086818 against 91.09 % / 9 / 0.052265, ratio 1.661. All three fail.
- The f16 control passes (ratio 1.0057).
- Preconditions 4 and 5 now hold, so this reading is no longer provisional.

### Reading

- **By the rule, D7 is parked, not passed.** int8 KV on the 7B costs nothing measurable on flips or agreement against f32. Its KL ratio of
  1.078 sits in the band where this gate does not decide. A re-run needs a named mechanism, not a re-roll.
- **The f16 control parks in the same band at D7** (1.057). Against f16, the default it would replace, int8 sits inside every bar
  (ratio 1.020).
- **On the 1.5B, int8's cost is clear** (S, all three bars fail).
- **What this decides, by the contract (item 4):** `--kv i8` stays delivered on request, and no default moves. The docs and the banner say
  int8 KV on CUDA is lossy: measurably on the 1.5B, and within the gate's ambiguous band on the 7B.

**Predictions scored:**
- "S passes (0.65)": wrong.
- "D7 reads FAIL or PARKED (0.6)": right, PARKED.
- "int8's mean KL against f32 is over 1.10x at D7 (0.5)": wrong, 1.078.

**Speed check:** running since 08:01 (step 3). Its results go in their own section.
