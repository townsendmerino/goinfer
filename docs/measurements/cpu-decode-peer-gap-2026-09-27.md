# CPU decode vs Ollama, Phase 0 — the gap is a third bytes and two-thirds bandwidth, and the 09-23 record overstated Ollama's bandwidth (2026-09-27)

The Linux half of "CPU decode is the one backend still behind" (the 2026-09-25 peer claim: 0.80× at 0.5B, 1.5B and 7B
void at 0.80× / 0.84× raw). This is exploratory, with no gate and no claim wording. It re-baselines, recounts the bytes
each engine streams per token, and ranks the levers for a pre-registration. Prior art read first:
[`cpu-decode-attribution-2026-09-22-linux.md`](cpu-decode-attribution-2026-09-22-linux.md) and
[`cpu-decode-roofline-2026-09-23.md`](cpu-decode-roofline-2026-09-23.md), whose per-component split this does not
repeat.

**Result:**
1. **Nothing has moved since the claim.** HEAD reads 0.818× / 0.810× / 0.856× Ollama, and is 1.001–1.003× the 09-25
   build.
2. **Correction to the 09-23 record.** Both the 0.5B and 1.5B GGUFs carry a separate `output.weight` (Q8_0 and Q6_K).
   llama.cpp streams that head and only looks rows up in `token_embd`, so Ollama streams 392 / 980 MB per token, not
   the whole file (491 / 1117 MB). Its bandwidth is therefore **22.5 / 23.5 / 26.0 GB/s**, not 28.1 / 26.6 / 26.3.
   The claim that goinfer at Ollama's bandwidth would be "~1.06× Ollama" on the 1.5B is withdrawn: it would be
   ~0.93×.
3. **The gap = bytes ratio × bandwidth ratio.** On the 1.5B and 7B, goinfer streams 7.5% / 5.8% more bytes (f32 group
   scales, int8 head) *and* achieves 13–15% less bandwidth. On the 0.5B it streams 8% fewer bytes and achieves 33%
   less bandwidth.

## Setup

- `nobara`, Ryzen 7 3700X (8c/16t), Nobara 44, kernel 7.2.0.
- goinfer `serve-cpu` at `157196bb` (HEAD) and at `411e7fc4` (the 09-25 claim's build), against Ollama v0.32.5 with
  its defaults (CPU backend).
- `scripts/bench_peer.py`, phase A, CPU only, depth 128, greedy, essay-v2 prompts, 3 runs × 8 completions × 64 tokens
  per cell, a server restart per cell, the harness's load ≤ 1.0 gate per cell.
- qwen2.5-coder-0.5b / coder-1.5b / 7b-instruct q4_k_m from `~/models`.
- Run 2026-09-27 19:15–19:53 PDT. Raw: [`cpu-decode-peer-gap-2026-09-27/`](cpu-decode-peer-gap-2026-09-27/).
- **Process notes.** Twice, a header-reading script of mine ran a whole-file hash while the sweep was up, because
  `scripts/gguf_same_weights.py` executes its comparison at import. Neither overlapped a timed cell: both landed in
  the harness's between-cell idle wait, which held the next cell until load decayed (`baseline.log`). A mistaken
  `bench_peer.py --help` also ran one cell into a file named `--help` with the harness's default v0.15.0 binary; it
  was discarded and is not in the data.

## 1. Re-baseline (`baseline.json`; all cells pass the token gate)

| model | goinfer HEAD tok/s | 09-25 build | Ollama | **HEAD ÷ Ollama** | HEAD ÷ 09-25 build |
|---|---:|---:|---:|---:|---:|
| 0.5B | 46.91 | 46.75 | 57.34 | **0.818×** | 1.003× |
| 1.5B | 19.41 | 19.38 | 23.96 | **0.810×** | 1.002× |
| 7B | 5.09 | 5.09 | 5.95 | **0.856×** | 1.001× |

Medians of 3 runs; every run within 0.3% of its median. The essay-v2 prompts leave no early stops this time, so the
1.5B and 7B, void on 09-25, read cleanly and agree with their void raw ratios (0.80×, 0.84×).

## 2. Bytes per token (`bytes_per_token.py`, header-only; `bytes-per-token.txt`)

- **Q4_K_M, as llama.cpp reads it:** every 2-D tensor at its own ggml type, with the head from `output.weight` where
  it exists.
- **goinfer int4:** 0.625 B/param (a nibble plus one f32 scale per 32-group); the head int8 plus one f32 per row.
- goinfer's column matches the 09-23 record's count from the loaded model (360.4 / 1052.9 / 4623.9 MB).

| model | Q4_K_M MB/token (by type) | goinfer MB/token | bytes ratio |
|---|---|---:|---:|
| 0.5B | 392 (Q5_0 173, Q8_0 head 146, Q6_K 43, Q4_K 29) | 361 | 0.920 |
| 1.5B | 980 (Q4_K 626, Q6_K 354 incl. head) | 1053 | 1.075 |
| 7B | 4371 (Q4_K 3121, Q6_K 1248 incl. head) | 4625 | 1.058 |

## 3. The gap, decomposed

Effective bandwidth = bytes per token × tok/s. The bytes ratio × the bandwidth ratio reproduces the measured time
ratio.

| model | goinfer GB/s | Ollama GB/s | bytes ratio | × bandwidth ratio | = time ratio | 1 ÷ measured |
|---|---:|---:|---:|---:|---:|---:|
| 0.5B | 16.9 | 22.5 | 0.920 | 1.329 | 1.223 | 1.222 |
| 1.5B | 20.4 | 23.5 | 1.075 | 1.151 | 1.237 | 1.234 |
| 7B | 23.5 | 26.0 | 1.058 | 1.105 | 1.169 | 1.169 |

Against the ~30 GB/s read ceiling (`readbw.c`, 09-23), Ollama runs at 75–87%, so it is not at the ceiling either.
goinfer runs at 56–78%.

**What this changes from the 09-23 record.** That record found goinfer's bytes were fewer on the 0.5B and 1.5B, and
ranked bandwidth alone as the gap. On the corrected counts, bytes are a third of the gap on the 1.5B and 7B, the
sizes where the claim was void. They are none of it on the 0.5B, whose gap is efficiency only.

## 4. Levers, ranked for a pre-registration (estimates, none measured)

1. **f16 group scales for CPU int4 (bytes).** 0.625 → 0.5625 B/param saves 22 / 82 / 408 MB per token.
   - That puts the byte ratio at 0.86 / 0.99 / 0.965 of Ollama's.
   - If the matmuls keep their bandwidth, the estimate is 0.5B 1.06×, 1.5B 1.08×, 7B 1.10×. That is ~0.87× / 0.87× /
     0.94× Ollama; it is an upper bound, because the kernel must convert f16 scales.
   - **Quality has precedent.** CUDA (`ws16`), Metal and WebGPU already store int4 group scales as f16, so the CPU is
     the only backend on f32. `GOINFER_INT4_F16_SCALES=1` (`decoder/int4f16scales.go`) already makes a CPU load carry
     exactly those weights, so the quality gate can run before any kernel exists.
   - **Cost:** an aikit kernel for each W4A8 path (AVX2, AVX-512 VNNI, NEON), a `.giw` version, and an aikit release.
     It is not bit-identical to today's CPU output, so the goldens and the parity manifest are re-baselined, and the
     peer same-weights story is unchanged.
2. **Bandwidth on the small projections and `down` (efficiency; the whole 0.5B gap).** The 09-23 split puts q/k/v at
   11.4 GB/s, o at 17.2 and down at 20.1, against gate+up at 24.1.
   - R-06's fused q/k/v (`GOINFER_W4A8_BATCH`, bit-identical) measured 1.053× / 1.039× (1.5B / 0.5B) and is parked by
     its own ≥ 1.15× bar. Flipping it is the owner's call under that bar.
   - Why `down` streams 17% slower per byte than gate+up is still unexplained. The per-worker timestamps inside a real
     token (S-02) are still not taken; they are the next instrument.
   - **ANSWERED, 2026-09-28** ([`cpu-down-vs-gateup-bandwidth-2026-09-28.md`](cpu-down-vs-gateup-bandwidth-2026-09-28.md),
     not the S-02 instrument itself — a cheaper existing one answered the question). Two compounding causes: `down`
     is an unfused, standalone fork/join (pays a ~71 µs/barrier fixed tax alone, where gate+up's own fusion
     amortizes the same tax over ~2× the bytes) and `down`'s serial, scalar, single-threaded activation quantizer
     runs over `intermediate_size` (not `hidden_size`), a length where the two-pass algorithm stops fitting L1/L2
     and its cost turns super-linear (3.5 → 6.8 ns/element, directly microbenched). A 3-parameter global fit across
     all three model sizes (fixed tax + quantizer cost/element + true bandwidth) reproduces the measured gate+up and
     down ms/token within 0.1–8.2%, R² = 0.9998. Not yet built: an AVX2 amd64 quantizer (S-03's own "open" item,
     already shipped for arm64/NEON) is the smaller, lower-risk next lever; fusing `down` into a bigger fork/join is
     the other, structurally harder.
3. **The LM head (bytes).** int8 at 234 MB against Q6_K's ~191 MB on the 1.5B. The ~4% it is worth is the smallest
   of the three and the most quality-sensitive (a 1.5% argmax flip rate for int4 is on record). Not recommended first.
   - **RE-EXAMINED, 2026-09-28, after levers 1 and 2 both shipped.** Two findings, one negative and one positive.
     - **Negative: lever 2's AVX2 activation quantizer (item 5) does NOT touch the head.** Its own activation
       quantize is over `hidden_size` (small, like gate+up's), trivial next to streaming `vocab_size × hidden_size`
       weight bytes — confirmed directly from already-captured decode-timing logs, LM head phase flat at
       5.15→5.13 / 8.72→8.69 / 20.12→20.13 ms (0.5B/1.5B/7B) before vs after the AVX2 quantizer shipped. The
       head's cost is pure weight-streaming bandwidth, not per-token overhead — a different mechanism from `down`.
     - **Positive: the lever is BIGGER now than when ranked, because 1 and 2 shrank everything around it.** The
       head is now 26.5% / 20.3% / 12.0% of a token (0.5B/1.5B/7B) — not the small remainder it was against the
       original, slower baseline. `--embed-int4` (already built, already opt-in, already measured at ~2.3 pts
       top-1 drop, mostly rare tokens — `docs/completed/task-w4a8-neon-bandwidth.md`) is a real, zero-new-code
       lever to re-quantify: measured fresh, its own time drops 40.9% / 42.5% / 43.1% at the head, for a further
       **1.122× / 1.094× / 1.054×** on TOTAL token time — bigger than the original "~4%" estimate, because that
       estimate was computed against a token this campaign has since made ~10-25% smaller. The 2.3-pt quality
       cost is UNCHANGED by any of this (re-quantizing the same weights costs the same accuracy regardless of how
       much time it saves) and was NOT independently re-verified here (needs a real eval harness with reference
       labels, not a decode-timing sample). **This is a quality-vs-speed tradeoff, not an engineering gap — the
       owner's call**, the same way the f16-scale quality gate and the 0.5B's accepted 3% both were.

Stacked, 1 and 2 (R-06 only) would put the 1.5B near ~0.92× by these estimates. Parity on the 1.5B and 7B needs the
unexplained bandwidth in lever 2 as well.

## Not established

- Every speed-up in §4 is an estimate from bytes and a bandwidth assumption; none was measured.
- Ollama's thread count on this box (its default is the physical-core count; goinfer runs 16 threads, and the 09-22
  record found 16 vs 8 inside noise for goinfer).
- Per-shape peer timing. `test-backend-ops perf` re-runs each op on one set of weights, and every 1.5B matrix fits
  the 3700X's 16 MB L3. It would time cache-hot kernels rather than decode's DRAM stream, so it was not used.
- The Mac half (0.5B ambiguous-low, 1.5B void on 09-25).
