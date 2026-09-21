# R2 `attention_fa` — speed band: a real 1.19× at depth, below the registered kill line. KILLED on the band.

**Result: on the pre-registered instrument ([`metal-attn-fa-PREREGISTERED.md`](metal-attn-fa-PREREGISTERED.md),
depth bench, 1.5B, median of three interleaved runs' best-of-batches), `attention_fa` decodes at
44.9 tok/s at depth 4000 against the shipped kernel's 37.8 (1.19×), 54.9 vs 49.4 at 2048 (1.11×),
and is identical below its 1536-key floor (71.9 vs 71.1 at 128, +1.1%; 67.0 vs 67.0 at 512). The
band ships at ≥60 @4000 AND ≥58 @2048, parks at 48–60 @4000, and kills below 48: 44.9 is below the
kill line, and 54.9 is below the 2048 ship bar. Verdict per the registered rule: KILLED. The
served harness agrees independently (goinfer 39.1 → 46.4 tok/s at the 3900-token calibrated
prompt, the same 1.19×; Ollama 76.6 / 76.2 as the drift control, flat). This is the M1's
dispatch/occupancy floor winning a fifth time, as the brief said it would be worth writing down
firmly — with the difference that this attempt is fidelity-clean (gate (3) PASSES,
`r2-attn-fa-rootcause-2026-09-21.md`) and strictly faster than shipped at every depth where it
engages, so whether a fidelity-gated 1.19× that misses the peer-parity band is worth enabling
anyway is a separate owner decision, recorded below as open, not taken here.**

## Method

Box `apple-m1pro` (M1 Pro, 16 GB), Darwin 25.6.0, goinfer `344b9514` (the R2 root-cause commit;
no production code changed since the 2026-09-20 race fix). `qwen2.5-coder-1.5b` GGUF Q4_K_M,
`Quant:"int4"`. Exactly as pre-registered:

- **Depth bench** (`metal/depth_bench_test.go`, `GOINFER_METAL_DEPTH_BENCH=1`): one resident, KV
  warmed incrementally, steady greedy decode via `ForwardArgmax` (synchronous one-command-buffer
  path through the same `encodeAttentionResidualWith` dispatch site), best-of-5 batches at
  {128, 512, 2048, 4000}. **Three runs per arm, interleaved off/on/off/on/off/on, each its own
  process** (one resident at a time on this machine; the arm is fixed per process because
  `GOINFER_METAL_ATTN_FA` is read once at `BuildResident`). Scored as the median of the three
  best-of-batches per depth. 07:10–07:19 PDT, ~85 s per run, swap guard armed, never fired.
- **Served cross-check** (`scripts/bench_peer.py`, `BENCH_MODELS=1.5B BENCH_BACKENDS=metal
  BENCH_DEPTH_BACKEND=metal BENCH_DEPTHS=3900 BENCH_ENGINES=goinfer,ollama`, greedy, 2 runs × 8
  completions × 64 tokens, `BENCH_MAX_LOADAVG=2.0` honored — both invocations waited for the box to
  clear it): the real serving loop (`execLoop`, pipelined command buffers), Phase A at depth 128 and
  Phase B at the 3900-token calibrated prompt, once per arm with Ollama inside each as the drift
  control. 07:18–07:22 PDT. Logs archived beside the other R2 logs in the goinfer-logs/r2
  directory in the home directory.

Not measured, as pre-registered: the brief's `tax_test.go` GPU-busy cross-check (`tax_test.go` is
a nop-kernel binding-tax microbenchmark, not a per-forward GPU timer; the depth bench does not
surface the resident's command-buffer timestamps).

## Data — depth bench (tok/s, best-of-5 batches per run)

| depth | shipped #1 | #2 | #3 | **median** | `attention_fa` #1 | #2 | #3 | **median** | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 71.0 | 71.6 | 71.1 | **71.1** | 71.9 | 71.2 | 71.9 | **71.9** | 1.011× |
| 512 | 65.9 | 67.1 | 67.0 | **67.0** | 67.2 | 66.9 | 67.0 | **67.0** | 1.000× |
| 2048 | 49.0 | 49.4 | 49.5 | **49.4** | 54.9 | 54.4 | 54.9 | **54.9** | 1.111× |
| 4000 | 39.6 | 37.8 | 37.8 | **37.8** | 44.9 | 44.9 | 45.0 | **44.9** | 1.188× |

Spreads: `attention_fa` 0.1 tok/s at 4000 and 0.5 at 2048 across three runs — tight; the shipped
arm's 4000 cell spread 1.8 (39.6 in run 1, 37.8 in runs 2–3; median takes 37.8; the ratio against
run 1's 39.6 would be 1.13×, still a real win and still below the line). Interleaving held: no
monotone drift across the six runs at any depth.

Depth term (ms/token over the 128 cell): shipped 26.5 − 14.1 = **12.4 ms** at 4000; `attention_fa`
22.3 − 13.9 = **8.4 ms** at 4000 — a 32% reduction of the term, still above the ~7 ms the brief names
as "the floor won again". Per position: shipped 3.2 µs/pos (2048→4000), `attention_fa` 2.0.

## Data — served cross-check (tok/s, `bench_peer.py`)

| cell | shipped (`GOINFER_METAL_ATTN_FA` unset) | `attention_fa` (`=1`) | ratio |
|---|---:|---:|---:|
| goinfer, depth 128 | 73.6 (73.7 / 73.5) | 74.0 (73.9 / 74.0) | 1.005× |
| Ollama, depth 128 | 85.9 (85.9 / 85.8) | 86.9 (87.3 / 86.5) | drift control |
| goinfer, depth 3900 | 39.1 (39.1 / 39.0) | **46.4** (46.4 / 46.4) | **1.187×** |
| Ollama, depth 3900 | 76.6 (76.2 / 76.9) | 76.2 (76.3 / 76.2) | drift control |

The pipelined serving path reproduces the synchronous depth bench's ratio to the third digit
(1.187× vs 1.188×), and the two arms' Ollama cells agree within 1.2% — the two invocations are
comparable. Against the peer at depth, goinfer moves from 0.51× (39.1 / 76.6) to 0.61× (46.4 /
76.2) of Ollama; at 128 the arms are the same kernel (0.86× of Ollama either way).

## Reading against the band

Pre-registered: **ships at ≥60 tok/s at 4000 AND ≥58 at 2048 AND no regression beyond 3% at 128;
parked at 48–60 at 4000; killed below 48.**

- 4000: **44.9 < 48 → killed zone.** Not the parked band; 3.1 tok/s below its floor, on a spread of 0.1.
- 2048: 54.9 < 58 → does not ship on that cell either (not independently a kill criterion).
- 128: 71.9 vs 71.1 = +1.1% → the ≤3% regression criterion is met, as predicted by construction
  (both arms dispatch the identical shipped kernel below the 1536-key floor; the only per-token
  difference is `setPos`'s two extra `SetU32` writes from the 2026-09-20 race fix, and they cost
  nothing measurable).

**Verdict: KILLED on the registered band.** No re-baselining: the shipped arm came in at 37.8
against the brief's 40.4 standing at 4000 (within this box's ordinary session drift, and the band
is absolute tok/s as registered), and the kill line stays where it was written.

What the number means mechanically, in the brief's own frame: the brief's premise was that the
75%-of-attention GQA-redundant K/V read was the term, and that a cooperative read (one coalesced
256 B row per key per simdgroup instead of six strided gathers) would recover most of it. The
kernel does exactly that (isolation gate 2026-09-19: 1.26× on the attention term alone at
K=3900), and end to end that is 1.19× on the token — the "isolated attention-term wins shrink once
diluted across a full token's other costs" caution the 2026-09-19 record already stated,
quantified. The depth term that remains, 8.4 ms at 4000, is still ~2.6× the peer's implied ~3.2 ms
(1000/76.2 − 1000/86.9 ≈ 1.6 ms of peer depth term on this box's own Ollama numbers, ×2 for
goinfer's 2.0 µs/pos against the peer's ~0.3 registered) — the remaining gap is not in the K/V read
pattern this kernel fixed. That is the fifth negative on this axis, and this time the reduction
it did achieve is measured rather than inferred.

## What this does not establish

- **Whether to enable it anyway.** The band was written for the peer-parity goal and it decides that
  question: no. It does not decide whether a lane that is fidelity-clean (gate (3) passes with the
  candidate marginally ahead on every pooled measure), strictly faster at every depth ≥1536
  (1.11–1.19×), identical below, and deterministic is worth turning on as an incremental
  improvement outside that goal. That is an owner decision with a real trade — every
  non-bit-identical kernel on the decode path breaks the bit-identity contract `ollama-chase.md` §2
  describes and moves the Metal snapshot golden — and it is recorded here as open, not resolved.
  `GOINFER_METAL_ATTN_FA` stays off by default.
- 0.5B, 7B, phi3-mini, other GQA shapes — the 1.5B decision cell only. Windows, sinks, f32-KV, int8
  KV — out of the kernel's scope by design, unchanged.
- Any split-count (S) tuning beyond `attnFASplitFor`'s registered rule (kvHead × S ≥ 2× core count,
  S=14 here); the 2026-09-19 isolated probe found S=28 marginally better at K=3900 (1.26× vs 1.21×),
  untested end to end. A retune that closed 3.1 tok/s at 4000 is not implausible and would move the
  verdict from killed to parked, not to shipped (48–60 band); noted, not pursued.
