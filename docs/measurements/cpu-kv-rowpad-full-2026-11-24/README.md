# CPU KV row pad, full implementation: registration, 2026-11-24 05:59 PST, before any timed run

**Why.** The trimmed prototype A/B read ADOPT (`docs/measurements/cpu-kv-rowpad-2026-11-23/README.md`, § Result). This measures the full
implementation (every reader of the CPU KV layout takes the padded stride; branch `cpu-kv-rowpad-2026-11-23`), adds depth 128, settles
whether the 16-key QK kernel ships under the pad, and re-measures the f16 read on the kernel that would ship. Its outputs are
bit-identical padded and unpadded (`TestKVRowPad_everyReaderBitIdentical` and the other `TestKVRowPad_*` tests).

**Instrument.** `BenchmarkDecodeAtDepth/grouped_on` with `GOINFER_BENCH_FILL=1`, 32 timed tokens a cell, one process a cell, A B B A ×
5 blocks = 10 pairs a comparison, `run_ab.py` (`docs/measurements/cpu-qk-multichain-2026-11-23/`, `--pad-b` for arm B's pad) under the
exclusive lock, its void rule unchanged (start load over 8.0 or swap growth: void, re-run in place, listed). Binaries of one goinfer
commit: **K16** on aikit `f16-acc64-2026-11-23` (the 16-key f32 QK kernel; the f16 kernels), **K4** on aikit `k4-variant-ab-2026-11-23`
(the same with the f32 16-key dispatch off). The f16 kernels are the same in both. Every padded Phi-3 f32 cell must report `rowpad 16`.

| id | comparison (B ÷ A) | depths | role |
|---|---|---|---|
| R1 | Phi-3 f32, K16, padded ÷ unpadded | 3,900, 128 | the pad (default question) |
| R2 | Phi-3 f32, padded, K16 ÷ K4 | 3,900, 128 | the 16-key kernel under the pad (QK dispatch) |
| R3 | Phi-3 f32, K4, padded ÷ unpadded | 3,900 | report: the pad with today's kernel |
| F1 | Phi-3, K16, f16 ÷ f32 padded | 3,900 | the f16 read against the best f32 |
| C1 | 0.5B f32, K16, pad on ÷ off (not applied: A/A) | 3,900 | control |

**Bars, each fixed now:**
- **R1, the pad as the CPU default** (Francis's call, brought with this reading and the pad's memory, 128 B per token per layer, Phi-3
  16 MiB at 4,096 positions): **CONFIRMED** if R1 at 3,900 is ≥ 1.15 with ≥ 9/10 pairs faster **and** R1 at 128 ≥ 0.98 (no short-context
  cost); **NOT CONFIRMED** if R1 at 3,900 is under 1.05 or R1 at 128 under 0.98; between, PARKED.
- **R2, the 16-key QK kernel under the pad** (the four-condition rule for a bit-identical speed change): **SHIP** with the pad if R2 at
  3,900 is faster on ≥ 9/10 pairs and R2 at 128's median ≥ 0.98; otherwise it does not ship. The 0.5B's 16-key reading is carried from
  `docs/measurements/cpu-qk-multichain-2026-11-23/` (1.099×, 8/8 at 3,900; 1.000× at 128), not re-measured. If the pad does not become
  the default, the 16-key kernel does not ship (it read 0.948× on unpadded Phi-3).
- **F1, f16 KV as a CPU opt-in** (the lossless opt-in bar; fidelity passed on both models): **PASS** at median ≥ 1.05 with ≥ 9/10 pairs.
- **C1** outside 0.98–1.02 voids the session's readings (the instrument moved), not an option.
- **Prediction:** R1 at 3,900 about 1.17, at 128 about 1.00; R2 at 3,900 about 1.12; R3 about 1.00; F1 about 1.40.

**Cost:** R1 and R2 at both depths (four comparisons), R3, F1 and C1 at 3,900: about 105 min, under the exclusive lock, by day, run
fourth in nobara's order of 2026-11-24 once the three sessions ahead have finished. **Stopping rule:** fixed N, no early stop.
