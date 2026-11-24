# CPU KV row pad (option b): registration, 2026-11-23 18:02 PST, before any timed run

**Owner's call** (Francis, 2026-11-23, relayed by the coordinator ~18:05 PST): "Verify b with at least some testing, even if it's a bit of
implementation, then go with b if it's really good." Option (b) of `docs/measurements/cpu-qk-multichain-2026-11-23/README.md`: pad the
CPU KV row by one 64 B line where kvDim × element size is a multiple of 4 KiB, so rows a power-of-two stride apart stop sharing L1 sets.

**Question.** Does the pad make CPU decode faster at depth without changing outputs?

**Prototype** (branch `cpu-kv-rowpad-2026-11-23`, behind the test-only `kvRowPadForTest`): a dense cache (every layer global, one KV width,
f32 or f16, no ring, MLA or hybrid state) stores its rows padded in separate fields that only single-token decode reads, through the
acc64 kernels at the padded stride; every other reader (prefill, `Keys`/`Vals`, `LayerKV`, `TruncateTo`) panics on a padded cache.
Tier 0, all PASS before this registration: `TestKVRowPad_padOnlyAtAnAlignedRow`, `TestKVRowPad_decodeBitIdentical` (f32 and f16,
an aligned and an unaligned row, both QK kernels, the padded store read), `TestKVRowPad_planPricesThePad`, `TestKVRowPad_otherReadersRefuse`.

**Where the pad applies:** Phi-3 mini at f32 (3,072 × 4 B = 12 KiB). Not at f16 (6 KiB), and never on Qwen2.5-Coder-0.5B (512 B and 256 B).
**Memory:** 64 B per row, so 128 B per token per layer (K and V); Phi-3 mini 4,096 B per token, **16 MiB at 4,096 positions** (+0.52% on its
3.22 GB f32 KV).

**Metric.** Decode tok/s, median of A B B A pairs, `BenchmarkDecodeAtDepth/grouped_on` with `GOINFER_BENCH_FILL=1`, 32 timed tokens a
cell, 5 blocks = 10 pairs a comparison, `run_ab.py` (`docs/measurements/cpu-qk-multichain-2026-11-23/`) under the exclusive lock, its
void rule unchanged. Two binaries of this commit: **K16** on aikit `f16-acc64-2026-11-23` (the 16-key f32 QK kernel and the f16 kernels),
**K4** on the same aikit with the f32 16-key dispatch turned off (a named variant branch), so the f32 QK kernel is the 4-key one.

| comparison | B ÷ A | depths | role |
|---|---|---|---|
| P1: Phi-3 f32, K16, padded ÷ unpadded | the pad where the aliasing bites | 3,900, 128 | **decision** |
| P2: Phi-3 f32, K16 padded ÷ K4 unpadded | the pad against the f32 path that ships | 3,900, 128 | **decision** |
| P3: Phi-3 f32, K4, padded ÷ unpadded | the pad with today's kernel | 3,900, 128 | report |
| C1: 0.5B f32, K16, padded ÷ unpadded | pad not applied: same code, A/A | 3,900 | control |
| C2: Phi-3 f16, K16, padded ÷ unpadded | pad not applied: same code, A/A | 3,900 | control |

**Bars** (the coordinator's, tightened before any reading because the kernel microbench put the 4-key kernel flat under a pad, 193 against
191 µs, so a bar read on the 4-key kernel alone would test nothing):
- **ADOPT** ((b) goes on to a full implementation): P1 at 3,900 ≥ 1.15 with ≥ 9/10 pairs faster, **and** P2 at 3,900 ≥ 1.05 with ≥ 9/10,
  **and** no decision cell (P1, P2 at 3,900 and 128) under 0.98.
- **KILL:** P1 at 3,900 under 1.05, or P2 at 3,900 under 1.00.
- **Otherwise PARKED** to Francis with the readings. A control outside 0.98–1.02 voids the session's reading (the instrument moved), not
  the option.
- **Prediction** (from the microbench and the profile's QK share): P1 at 3,900 about 1.15, P2 about 1.08, P3 about 1.00.

**Stopping rule:** fixed N, no early stop. **Cost:** about 140 cells, about 2.4 h, tomorrow by day (2026-11-24), under the exclusive lock.

**What it decides.** ADOPT: (b) goes to a full implementation covering every reader of the CPU KV layout (rings, snapshots, the GPU upload
bridge, prefill), with its own tests, brought to the coordinator before it is armed; the QK dispatch question becomes P2 against P3.
KILL: the pad is dropped and the QK change goes back to options (a) or (c). PARKED: Francis's call.

## Amendment: the trimmed run (Francis, 2026-11-23, relayed by the coordinator at 18:10 PST), written 18:10 PST before any reading

Francis chose the 1.1 h trim. **Run:** P1 and P2 at depth 3,900, and C1 (the 0.5B at 3,900, pad not applied) as the one control: 3
comparisons × 20 cells = 60 cells. **Not run, by Francis's choice:** both decision cells at depth 128, P3 (the 4-key kernel padded), and C2
(Phi-3 f16). **The bars are unchanged**, and their "no decision cell under 0.98" now covers P1 and P2 at 3,900 only. **No short-context check
is made**, so an ADOPT carries "128 not measured" until the full implementation's A/B, which measures it. `run-ab.sh` runs the trimmed set.
