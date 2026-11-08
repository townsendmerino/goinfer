# CUDA `--kv i8`, S4: fidelity (G4) and speed, registered before any run (nobara-sb01, 2026-11-07 20:05 PST)

S4 of `scope.md`'s staged plan, for the dense families S2 and S3 deliver (llama, mistral, qwen2, qwen3, phi3 without a sliding window). G4 is
the task doc's (`docs/tasks/task-kv-precision-everywhere-2026-10.md` §4): **a published reading, not a refusal**. The contract's item 4 holds:
`--kv i8` asked is delivered whatever this reads. The reading decides what the docs, the banner and the matrix say about int8 quality, and it
moves no default (`auto` stays as it is; int8 is never an `auto` pick). Nothing here is run until this file is committed.

## G4: fidelity against f32, with f16 beside it

**Instrument:** `TestKV8GateVsReference` (to be written in `backend/cuda/kv8_gate_test.go`, on the shape of `TestKV16GateVsReference`, whose
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

## Addendum, 2026-11-07 20:10 PST (logistics only; no bar, cell or prediction changed)

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
