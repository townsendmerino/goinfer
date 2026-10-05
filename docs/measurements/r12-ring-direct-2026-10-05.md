# R-12: CPU decode reads a sliding-window ring's window in place (Gemma 3 1B, Gemma 2 2B), 2026-10-05

Audit entry: `docs/tasks/task-recompute-audit.md` section 5, R-12. Code: `decoder/kvcache.go` (`ring.mirrored`, `ensureMirror`, `window`; `write` and `commitBatch` keep the mirror), `decoder/attention.go` (the `ringDirectDecode` case), commit `16e45a6d`.

## What changed

K=1 decode on an f32 sliding-window ring layer used to copy every resident window row of K and V into scratch each token (`batchReadLocal`), for every local layer. The ring now keeps a mirror of its slots (`[w, 2w)` equal `[0, w)`), allocated the first time that layer's window wraps, so any window is one contiguous slice. Decode writes the new row first (the slot it takes holds position `pos-W`, one row outside the window `[pos-W+1, pos]`) and attends over the slice. The canonical half is untouched, so prefill's copy, snapshot save, `LayerKV` and the scalar path are unchanged. int8 rings keep the copy path.

## Provenance (every row below)

| | |
|---|---|
| machine | `nobara-pc`, amd64, 16 threads, Linux 7.2 (Fedora 44), idle box, the timing lock held for the timed A/B |
| checkpoints | `~/models/gemma3-1b-q4_k_m.gguf` and `~/models/gemma-2-2b-it-Q8_0.gguf` (local NVMe, not `/srv/models`); both loaded as `Backend: cpu, Quant: int4` |
| decode | greedy (temperature 0), 24 tokens per generation for the A/B, 48 for the logits check; prompt = `"Continue this text. "` plus repeated `" the"` tokens, truncated to the stated depth |
| versions | goinfer `16e45a6d`, aikit v1.56.1, Go 1.27.0 |
| date | 2026-10-05 |
| thermal | not recorded (no sensor read); the A/B interleaves the arms so slow drift cannot pose as an effect |

## 1. Size of the copy, by instrument (throwaway timer around the `batchReadLocal` call, since removed)

| model, depth (window) | forward ms/token | ring copy ms/token | share |
|---|---:|---:|---:|
| Gemma 3 1B, 900 (512) | 39.4 | 2.46 | 6.2% |
| Gemma 2 2B, 4500 (4096) | 201.3 | 31.7 | 15.7% |

One run each; the copy is also 33% and 26% of the "rope+KV+scores/softmax/AV core" the decode split reports. These are sizes of the work removed, not of the speedup.

## 2. Bit identity

- `TestRingMirror_invariantAndWindow`: random write, `commitBatch` (K from 1 to 2w), truncate, snapshot-style rebuild and window reads at W = 1, 3, 4, 7; the mirror equals the canonical half after every operation and every window equals what the copy path assembles.
- `TestRingMirror_decodeBitIdenticalToTheCopyPath`: gemma2-tiny, gemma2-hd-tiny, cohere2-tiny, 52 tokens each (past every wrap), every logit of every step equal bit for bit, with a non-vacuity check that the in-place run really allocated a mirror.
- Real Gemma 3 1B at depth 900, 48 decode steps, full logits vector through the sampler's `LogitProcessor`: **48 x 262144 logits compared, 0 differ** (`TestCPURing_directDecodeLogitsBitIdentical`).
- Each shown red by planting a bug: no mirror write in `write`, none in either `commitBatch` run, a window one slot late, a read before the write; and the fit accounting test fails when its change is reverted.

## 3. Speed, Gemma 3 1B at depth 900 (window 512): in-process paired ABBA, `TestCPURing_directDecodeAB/gemma3`

Harness: `cpuDecodeAB` (one load, `ringDirectDecode` flipped between generations, arm order alternating per pair, the forward ms/token from the decode split, a greedy-stream check in every pair), under the timing lock.

| pair | ON (in place) | OFF (copy) | OFF/ON |
|---:|---:|---:|---:|
| 0 | 38.07 | 39.42 | 1.036 |
| 1 | 37.67 | 39.24 | 1.042 |
| 2 | 37.63 | 39.58 | 1.052 |
| 3 | 37.61 | 39.29 | 1.045 |
| 4 | 37.69 | 39.40 | 1.045 |

Means 37.73 ON, 39.39 OFF ms/token; paired median **1.045x**, range 1.036 to 1.052, 5 of 5 pairs faster. That is 1.66 ms saved of the 2.46 ms the copy cost (67%): the rest is the attention re-reading the window from ring memory instead of from cache the copy had just warmed. By TE5(b) this is a resolved direction from a whole-token in-process A/B, so the change is kept by day. It is not a served or peer claim.

## 4. Memory

A layer that has wrapped holds `2*W` rows instead of `W`: +436 MB on Gemma 2 2B at full window (13 local layers, 4096 x 1024 x 4 bytes x 2 for K and V), +0 before the first wrap and for int8 rings. `kvBytesForCtx` now counts it (`TestKvBytesForCtx_countsTheRingMirror`) so the host-RAM fit guard sees what is resident. The saving is the 436 MB per-token copy; the cost is holding it twice.

## 5. Pre-registration: the Gemma 2 2B arm (night queue, committed before it runs)

`run-g2.sh` in this directory, a binary pre-built from `16e45a6d` at `~/goinfer-logs/r12/decoder.test`. Tier: night. Instrument: the same in-process ABBA, 3 pairs at depth 4500 (window 4096), plus the harness's warm-up and reference generation. Stopping rule: fixed N = 3 pairs, no early stop. Cost basis: 8 generations x about 115 s (each re-prefills 4500 tokens; measured 1 m 54 s for one generation in section 1) plus load and idle waits, estimated 20 minutes.

Prediction, from sections 1 and 3: the 1B recovered 67% of its copy cost, and the 2B copy is 31.7 of 201.3 ms, so about 21 ms saved, a ratio of about **1.12**.

Decision rule, written before the result:
- median OFF/ON **at least 1.06** and every pair above 1.00: as predicted or better; record it.
- median in **[1.00, 1.06)**: smaller than predicted; the change stays (bit-identical, and the 1B direction is resolved) and the shortfall is recorded with the mirror's extra memory traffic as the first suspect.
- **any pair below 1.00, or a median below 1.00: ambiguous, parked.** The mirror's 436 MB would then be costing more than the copy it removes at the largest window; reopen before shipping this to a release.
- a greedy-stream mismatch in any pair is a defect (the harness fails the run), not a result.

This arm does not gate the commit: the ship decision rests on bit identity and the 1B result. It sizes the win where the window is largest.
