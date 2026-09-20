# Red October — the backend × area gap matrix, and the briefs to close it (2026-09-18)

> **Status: SCOPED 2026-09-18; R4 and R2 step 0 both ran the same day, and the decode fidelity-lane
> decision that R1/R2/R6 needed has been made.** **Owner decision, 2026-09-18:** decode's fast
> kernels may ship as the *default* once they pass the same pooled §3.2 fidelity gate the prefill
> lane already uses (`completed/task-prefill-gap.md` §3) — the bar CUDA decode's own exact path is
> already held to, not a relaxed one; a kernel that does not ship the gate stays opt-in. R1, R2's
> Build phase, and R6 are all unblocked by this. **R4 step 0:** the Metal prefill ladder improved
> post M-03/M-04 (K=512 3.33×→2.54× behind Ollama) but missed the registered ship/park bands, so
> step 2 (a further GEMM-tile change) is killed. **R2 step 0:** the decode-at-depth gap has
> genuinely narrowed against the only true same-session comparison on record — Ollama flat across
> six weeks, goinfer more than doubled at depth (4.19×→1.96× behind at ~4000) — after a same-day
> isolation caught an unrelated KV-cache-quant confound in the first measurement pass. **R1's Build
> attempt (2026-09-19) is PARKED**: the kernel is correctness-proven in isolation (gate 1), but a
> real, reproducible-from-a-clean-input catastrophic divergence was found and localized to layer
> 26 of qwen2.5-1.5B's 28 (gate/up's GEMV specifically) — two natural theories for it were tested
> and falsified, the actual mechanism wasn't found before the investigation was parked. See each
> brief for the full result and record. Everything else below is still a projection band
> registered before anyone measured, held to it as the campaign rule requires.
>
> **Why the name.** The promotion gate is "at least as fast as Ollama on the machines people
> actually have" ([`roadmap.md`](../roadmap.md), owner decision 2026-09-11). This doc is the
> plot of the whole chase on one page — four backends across, the areas that make up a token down
> — with the standing in each cell, the mechanism as far as the record has established it, and the
> delta a brief could earn. Silent running: no headline, no new claim, one page of sonar.
>
> **What it is.** §1 is the matrix. §2 is the per-row analysis with the arithmetic behind each
> band. §3 names three ceilings the record carries that its own later numbers have moved. §4 orders
> the work by machine. §5 is the briefs, R1–R13, written to be handed to a session as they stand.
> §6 is the rules every brief inherits; §7 is what this doc does not claim.
>
> **Siblings.** [`benchmarks.md`](../benchmarks.md) (standings — the only source of a ratio quoted
> here) · [`queue-performance.md`](../queue-performance.md) (P20–P25; P24 is re-scoped by R5) ·
> [`audit-metal-2026-09-12.md`](../audit-metal-2026-09-12.md) (M-/C-/G- numbering) ·
> [`completed/task-prefill-gap.md`](../completed/task-prefill-gap.md) (the §3 contract every
> fidelity-gated lane here inherits) · [`ollama-chase.md`](../ollama-chase.md) (the lever ledger,
> partly superseded) · [`completed/metal-verdict.md`](../completed/metal-verdict.md) (the Metal
> ceilings §3 revisits) · [`task-peer-benchmarks.md`](task-peer-benchmarks.md) (the peer matrix and
> its unbuilt rows) · [`task-l01-hybrid-moe-cpu-gpu.md`](task-l01-hybrid-moe-cpu-gpu.md) (R11's
> CUDA half) · [`task-decode-splitkv-attention.md`](task-decode-splitkv-attention.md) (R6's prior
> art) · aikit's `docs/task-simd-audit.md` (R9's kernel-side record; cross-repo, described not cited).

---

## 0. How to read this doc

**Ratios.** Unless a cell names another peer, a ratio is goinfer ÷ Ollama v0.32.5 on the same box,
same checkpoint, same quant, same session — below 1.0 goinfer is behind. Prefill rows say "N×
behind" in words, matching `benchmarks.md`'s own convention there. llama.cpp (Linux) and MLX (Mac)
are named where they change the picture, because both are ahead of Ollama on every cell they share.

**Tags.** Every cell carries one of: **K** — mechanism measured and named in the record (a profile,
an ablation, a collapse probe); **P** — partly attributed (the term is measured, the cause is a
reading); **?** — unattributed (a number nothing explains yet). A brief on a **?** cell is an
investigation first and a build second, and its band is registered after the investigation, not
before.

**Bands.** Each delta is a range with its arithmetic in §2. The arithmetic is Amdahl on measured
category shares — the method `completed/task-prefill-gap.md` §6.1 showed holds within 1.5%
internally — and, where a peer's number enters, the ~25% optimism its §6.2 measured on the peer
extrapolation is the reason the bands are wide rather than points. Per the campaign rule, a result
is held to its band: below it is a finding, not a rounding.

**Boxes.** `apple-m1pro` (M1 Pro, 6P+2E, 16 GB, ~200 GB/s fabric) and `nobara-pc` (Ryzen 7 3700X,
RTX 2070 SUPER 8 GB, driver `595.91.07`), exactly as `benchmarks.md` Table 2 defines them. Metal and
Mac CPU rows are the M1 Pro; CUDA, WebGPU and Linux CPU rows are the RTX box.

---

## 1. The matrix

| Area | CPU | WebGPU | CUDA | Metal |
|---|---|---|---|---|
| **Decode, ≤512 ctx, dense** | Mac 0.65–0.67× (0.5B/1.5B), 0.81–0.83× (phi3/7B); Linux 0.41× (0.5B), 0.73× (1.5B), 0.82× (7B) → Mac **0.85–1.0×**; Linux **1.4–2.4× on the cell** (P) | 0.37 / 0.60 / 0.63× of goinfer's own CUDA (no peer) → **0.8–0.86× of CUDA** (P) | 1.27 / 1.16 / 0.98×; vs llama.cpp 0.90 / 1.00 / 0.91× → **≤1.1×**, at ceiling (K) | 1.08× (0.5B, n=2) / 0.86× / 0.86×; vs MLX 0.66× (1.5B), 0.58× (7B) → **1.4–1.55×** via a W4F16 GEMV (P) — R1 |
| **Decode, 2k–8k ctx** | no peer row at depth → measure first; the f64 decode path loads and widens each KV head's K/V once per QUERY head (6–7×) → group-major acc64 kernels, bit-identical, **1.5–1.8× on the QK+AV term** by µop count (?) — R13 | −12% from 128→1024 after G36; ≥2k unmeasured → measure (R10) | 0.72–0.79× @3900, 0.63× @8k (7B); vs llama.cpp 0.45–0.70× → **1.0–1.1× Ollama** (1.4–1.6× on the cell) (K) — R6 | 0.64× @2048, **0.52× @4000** (peer flat 85→77) → **~0.8× Ollama** from attention alone (1.3× @2048, 1.5× @4000); ~1.1× stacked with R1 (P) — R2 |
| **Prefill — weight term** | Mac 1.54× behind @512 (weight term ≈ all of it) → **~1.2× behind** (K) — R9; Linux: no peer row | batched path landed 2026-09-13: 317 tok/s @P=1024 (0.5B int8) = **8× behind own CUDA-exact, 45× behind CUDA-fast**; unprofiled → ≥5× plausible (?) — R10 | 0.136 ms/tok flat vs Ollama's *whole* marginal 0.152 → **≤1.2×**, at ceiling (K) | pre-fix GEMM 0.73 TFLOPS vs ≥2.4 (3.3× behind @512); M-03 shipped 2026-09-13, **unmeasured** → **1.5–2× e2e** at K≤1024; parity needs 3.5× on the GEMM (P) — R4 |
| **Prefill — attention term, O(K²)** | Mac 0.91× @3900 (ahead), rate flat 78 tok/s → at parity (K) | per-query-row `attnBatched` (grid nH×M), the pre-L2 shape → inside the ≥5× above (?) — R10 | 3.16× / 1.89× behind on marginal @3900 (1.5B/0.5B); `attn_fused` at 1.72% tensor peak, 12.6% occupancy, **58% of TTFT** → marginal **~1.4–1.7× behind**, TTFT@3900 1.39 s → 0.71–0.84 s = **ahead of Ollama on TTFT** (K) — R5 | M-04 32-key tile shipped; ~18% of TTFT @3900 → ~1.2× e2e (K) — R4. Separately: **prompts of 8–255 tokens run sequential** at ~77 tok/s vs Ollama 790 @256 → floor→64 is **~4× on a 200-token turn** (K) — R3 |
| **Sampled decode, T>0** | host selection fixed (68×); full-V normalize remains → small, measure (R7) | full-logits `MapAsync` every token, greedy included → part of the glue row (R10) | **0.81–0.97× sampled vs 0.99–1.24× greedy** on ≤1.5B; the peer loses 0% → **+30–40%** on small models (K) — R7 | zero-copy logits, but host softmax/select over 152k per token ≈ 1.8 ms of a 13.5 ms token (counted) → ~10–15%, unmeasured (P) — R7 |
| **MoE over capacity — decode** | 26B full-model 5.5 tok/s (62 GB box); Mac 35B paged ~1.4 (SSD-bound) → 1.2–1.8× (cpubrrr Q4_K ceiling) (P) | no paging path; declines → n/a | M35 1.0× Ollama / **0.72× llama.cpp**; M26 1.07× / 0.88×; ~48% of the token is expert DMA → **1.25–1.45×** (L01 async; funding cell unrun) (P) — R11 | 35B ~2 tok/s; pager auto-sized 2026-09-13, unmeasured; 2L+1 sync command buffers at ~14 ms each (M-11 open) → **3–4 tok/s counted**; Zeno's 8.7–13 is structural past that (P) — R11 |
| **MoE prefill, 8k prompt** | P18 expert-major 4.36× landed; no peer row | n/a | wall-clock **M35 17×, M26 4.3× behind** → **2–5×** (expert-major; M35 also needs batched GDN); 2–8× behind after (P) — R11 | FFN half runs M sequential rows; expert-major landed 2026-09-17, unmeasured → measure (R11) |
| **Vision tower, image turn** | 31.3 s/image SigLIP f32 (3700X); exp on `math.Exp`, the S-06 transcendentals have zero callers → 1.5–2× (K) — R9 | 18.8 s (June, stale) → re-measure (R10) | 26.1 s (1.58× over CPU): dp4a batched GEMV + per-row attention = **the pre-L2/L3 shapes**, plus 54 host round-trips → **~7 s counted**; a ~1 s class needs a real int8 tensor-core GEMM at hd=72 (K) — R8 | tower runs on the **CPU** on a Metal box (M-15) → ≥3× once aikit's Metal ViT is wired (P) — R8 |
| **Speculative decode** | Θ 0.5; P25 break-even miscalibrated for MoE targets (Laguna 0.82×) | Θ unmeasured (P22) → measure (R12) | Θ 0.25; 1.2–1.8× shipped → done | **Θ 0.96 — declines to draft**; `ForwardN` is a loop, not one command buffer (P21) → **1.2–1.5×** on agent-style decode (P) — R12 |
| **Concurrency** | one decode worker per model on every backend; the peers run parallel slots; no instrument (W7) → unmeasured everywhere — R12 | | | |

---

## 2. Notes by row — mechanism and the basis for each band

### 2.1 Decode at short context

**Mac CPU.** The tok/s ratios *are* the effective weight-stream ratios. Bytes per token at goinfer's
int4 (0.5625 B per trunk weight, int8 LM head at 1 B, tied embeddings at 0.5B/1.5B, untied at 7B):
0.34 / 0.97 / 4.21 GB. Against the 2026-09-17 Mac row that is 34 / 48 / 72 GB/s for goinfer and,
on Ollama's Q4_K_M files (0.40 / 0.99 / 4.4 GB), 59 / 75 / 91 GB/s. A fixed-cost-plus-rate fit
across the sizes puts goinfer at **~4.5 ms/token fixed + 62–79 GB/s** and Ollama at **~2.4 ms +
92–95 GB/s**. The kernel itself is at 91–97% of its per-core issue ceiling (aikit SIMD audit, S-01
read-back), so the gap is around it: fan-out efficiency ("each core delivers 26–35% of what it does
alone", "fork/join is worth 1.70× and leaves ~2.7× that bandwidth does not explain" — the S-02
per-shard harness that would attribute it was never run) and the fixed cost. Band: fixed→2.5 ms and
stream→90 GB/s is parity on the 1.5B (75 tok/s); 3.5 ms and 80 GB/s is 0.85×.

**Linux CPU.** Same fit on the 2026-08-26 §B8 backend table: goinfer **~13 ms fixed at 22 GB/s**
(Ollama ~5 ms at 27, against a 30.5 GB/s measured read ceiling), and the 0.5B carries a further
**~14 ms/token that neither the fixed cost nor bandwidth explains** — the 1.5B/7B fit predicts 35.5
tok/s where the cell reads 23.5. That 0.41× is the most unattributed number on the page. Band: the
1.5B at Ollama's fixed cost and rate is 24 tok/s (1.4×); the 0.5B is 57 (2.4×).

**WebGPU.** The int4 GEMV is fine: the 1.5B int4 token (7.27 ms at 137.6 tok/s, §B10) minus the
int8 token's measured non-GEMV remainder (4.25 ms) leaves ~3.0 ms for 0.97 GB — ~320 GB/s, 70% of
the card. What CUDA does not pay is the other 4.25 ms: G35's ablation (`docs/QUEUE.md`) bills it to
`quantize` 1.05, `swigluQuant` 0.84, `gemvBias` 0.47, `rmsQuant` 0.46 ms/token plus attention and a
608 KB `MapAsync` per token, against CUDA's ~1.6 ms of glue with K1/K3a fused. G35 also established
that K1 (rmsnorm+quant folded into the QKV GEMV) is expressible in WGSL and K2 is not. Band: glue at
2.0–2.5 ms is 179–196 tok/s, 0.79–0.86× of CUDA's 227.

**CUDA.** At the recorded ceiling — GEMV at 80–85% of DRAM, the token at ~54% effective, K2
fusion measured ~0% and reverted. llama.cpp's +10% on the 0.5B and 7B is launch/glue class. Not a
campaign.

**Metal.** The record ([`completed/metal-verdict.md`](../completed/metal-verdict.md) §2a) closed
dense decode as "at ceiling — nobody demonstrably exceeds ~85 GB/s at batch 1 on this rig" and
said not to fund GEMV work. That ceiling was priced against llama.cpp's 75–83 GB/s band, and it is
the **W4A8 integer path's** ceiling: nibble unpack plus integer MAC at ~10–12 issue slots per
weight, no DP4A, 120–145 GB/s theoretical, 84–105 observed. The 2026-09-04 peer matrix then put
**MLX at 109.8 tok/s on the 1.5B (~95 GB/s) and 37.9 on the 7B (~162 GB/s)** at the same 0.5625
bytes per weight. MLX and llama.cpp both run f16-activation FMA paths. goinfer's weight term runs
~90–98 GB/s (the 1.5B token is 13.5 ms; subtract ~2.8 ms of launch floor, attention and glue). A
W4F16 decode GEMV — the activation precision the prefill lane already uses — at 140–160 GB/s puts
the 1.5B at 103–113 tok/s and the 7B at 30–34. Not bit-identical to W4A8; arguably higher fidelity
(no int8 activation), which is why it is a fidelity-gated lane — the owner decision (2026-09-18,
§4) means this kernel may ship as the default once it passes the §3.2 pooled gate; it stays
opt-in if it does not.

### 2.2 Decode at depth

**CUDA.** At 3900 on the 1.5B the depth term is 3.6 ms/token (8.0 − 4.4) against Ollama's 0.61
and llama.cpp's 0.32. The 2026-09-12/13 campaign (`docs/measurements/splitkv-*`, `vsum-*`) has the
mechanism cold: `splitkv_scores` is LSU-issue-throttled (93.5% `lg_throttle`; q-staging landed),
`splitkv_vsum` is latency-bound at 9% occupancy (75.5% `long_scoreboard`), and the twelve per-head
blocks re-touch each KV head's lines six times through L2. The flash-decode V-sum spike is the
first quarter of the win — +16.2% served at 8k on D7 — and is parked on fidelity alone (KL 1.0585×
exact's, inside the 1.05–1.10 ambiguous band). A full key-parallel, online-softmax, GQA-grouped
kernel taking the term to 0.8–1.2 ms is 178–192 tok/s at 3900 (1.02–1.10× Ollama) and 51–55 on
D7@8k (0.90–0.97×; today 35.7). The gate is the decision to open a fidelity-gated decode lane, not
the kernel (R6).

**Metal.** `benchmarks.md` §B3's depth table (62→18 tok/s) is two kernel generations stale in
goinfer's favour; the 2026-09-13 bench that served as M-10's baseline reads **72.3 / 67.9 / 51.4 /
40.4 at 128 / 512 / 2048 / 4000** (qwen2.5-coder-1.5b W4A8, `TestZZ_metalDepthBench`). The peer's
Metal curve *is* on record — `completed/metal-verdict.md` row 4, Ollama v0.32.5 FA-on, 2026-08-04:
**85.2 / 79.1 / ~80 / 77.5 at 128 / 1024 / 2048 / 4000** — flat. Depth term 5.6 ms at 2048 and 10.9
at 4000 against the peer's 0.8 and 1.2: 7–9×. The four recorded negatives (split-KV 0.95×, grouped
dedup 0.45–0.92×, staged 0.23×, M-09's cooperative K read 0.43×) all kept the per-thread serial
q·k dot and the per-dim serial V fold for bit-identity, and traded either dispatches or occupancy for
it. The untried corner is the peer's own shape: lanes across the head dim with `simd_sum` — which
the record already classes as order-exact — keys across simdgroups, the six query heads of a GQA
group sharing each K/V read, online softmax, and a key-split across threadgroups for occupancy with
one small combine dispatch. Deterministic by construction; a snapshot-golden re-baseline with a
mechanism. Band: depth term → 2–3 ms is 59–63 tok/s at 4000 (0.77–0.82×) and 63–65 at 2048;
stacked with R1's GEMV it lands near 1.1× (R2).

**The depth curves as a line, 2026-09-19 — arithmetic on rows already in `benchmarks.md`, no new
measurement.** Every served depth curve on the page fits `t(K) = F + A·K` (ms/token; F is the
depth-independent cost, A the cost per cached position) at R² ≥ 0.995 on goinfer's side, so each
cell splits into a flat ratio and a slope ratio. Least squares over the published cells; the two
Ollama cells the page itself flags are dropped (Metal K=512, the chunking artifact; CUDA 1.5B@512,
the unstable cell), and the peer's slopes are small enough to be noisy (R² 0.91–0.997), so read its
A as ±20% at best.

| cell (source row) | F goinfer / Ollama (ms) | F ratio | A goinfer / Ollama (µs/position) | A ratio | nH/nKV | A ratio ÷ (nH/nKV) |
|---|---|---|---|---|---|---|
| CUDA 0.5B (§B8, 2026-08-26) | 2.97 / 3.71 | 0.80× | 0.499 / 0.037 | 13.5× | 7 | 1.9 |
| CUDA 1.5B (§B8) | 4.48 / 5.14 | 0.87× | 0.924 / 0.168 | 5.5× | 6 | 0.9 |
| CUDA 7B (§B8) | 13.46 / 13.73 | 0.98× | 1.784 / 0.171 | 10.4× | 7 | 1.5 |
| CUDA phi3-mini (§B5.1, 2026-08-27) | 7.58 / 7.88 | 0.96× | 2.600 / 1.474 | 1.8× | 1 (MHA) | 1.8 |
| Metal 1.5B (§B3, 2026-09-18) | 12.97 / 11.67 | 1.11× | 3.235 / 0.362 | 8.9× | 6 | 1.5 |

(An F ratio below 1.0 means goinfer's flat term is the faster one. Greedy, decode-only, served path,
as each source row states.) Four readings, in the order of how much weight they bear:

1. **The flat term is at parity or ahead on every cell; the slope is the whole depth deficit.** With
   the peer's A and its own F, goinfer would read 194.7 tok/s at 3900 on the CUDA 1.5B (Ollama
   174.2), 70.8 on the 7B (69.5), and 69.5 on Metal (76.7 — 0.91×, the remainder being R1's flat
   term). The windowed control says the same from the other side: gemma3-1b (window 512, §B5.1) has
   no growing-K term, and goinfer leads it 1.08–1.12× at every depth.
2. **The registered bands, restated as slope targets** (served-path F held). R6's ≥170 tok/s at 3900
   is A ≤ 0.36 µs/position — 2.6× under today's 0.924 and about 2× the peer's; its 150 park line is
   A ≤ 0.56. R2's ≥60 at 4000 is A ≤ 0.92 — 3.5× under today's 3.24 and about 2.5× the peer's; its
   48 kill line is A ≤ 1.97. R2's band is registered on the depth bench, whose F differs a little
   from the served path's, so read those two as ±10%.
3. **goinfer's A follows QUERY heads; the peer's follows KV bytes (P — a regularity and a reading,
   not a profile).** On CUDA, A per query-head element (nH·hd·layers) is 23.2 / 21.5 / 17.8 / 26.5 ps
   on the 0.5B / 1.5B / 7B / phi3-mini — flat within ±20% across GQA 7, 6, 7 and MHA. Ollama's A per
   f16 KV byte is 3.0 / 5.9 / 3.0 / 3.75 ps (the 1.5B is its noisy curve): 170–336 GB/s, 38–75% of
   the 448 GB/s roof, i.e. within a small factor of bandwidth-bound on KV read once per KV head.
   That is why the A ratio tracks nH/nKV — the last column is 0.9–1.9 on every cell, the MHA control
   included. **This is not the traffic argument `splitkv-kernel-exploration-2026-09-13.md` retired,
   and that record stands**: the split path reads 1.06× ideal DRAM traffic, and L2 already dedups
   the group. It is a statement about what the kernel's *time* follows. goinfer pays the same
   ~18–26 ps per query-head element whether the bytes come from DRAM (phi3-mini, where
   `splitkv-mechanism-ncu-2026-09-12.md` measured 67.83% DRAM and this fit independently gives
   302 GB/s = 67.5%) or from L2 (the GQA geometries, under 30% DRAM in the same ncu records). A cost
   per *issued element* is what the stall profile's two bounds have in common — load issue in
   `scores`, memory latency per fold step in `vsum`. For R6 step 2 that means the group axis earns
   its place through issue/latency sharing (one K or V load serving all nH/nKV heads), not through
   traffic; and it suggests an acceptance check beside the band — after the kernel, A per KV byte
   should be flat across the four geometries, as the peer's is.
4. **On the MHA control the residual is the KV element width, not the kernel (P).** phi3-mini has
   nothing to group: both engines sit near the bandwidth bound (goinfer 302 GB/s reading f32 KV,
   Ollama 267 GB/s reading f16) and the 1.8× is the 2× in bytes. `completed/plan-still-slow.md` P4's
   "KV-quant is not a speed lever" was measured on the GQA 0.5B/1.5B, where the kernel is
   latency-bound and the verdict holds; it does not cover a geometry that is already byte-bound, and
   once R6 lands the GQA geometries inherit the same 2× floor against an f16 peer. f16 resident KV
   on CUDA (Metal already ships it for the non-Gemma families) is therefore a depth lever with a
   trigger — any geometry whose decode attention reads ≥~50% DRAM — and phi3-mini meets it today:
   halving its A reads 79 tok/s at 3900 against Ollama's 73.3 (arithmetic, not a measurement, and a
   fidelity-gated lane like every other non-exact path). Not funded here; recorded so R6's "KV
   quantisation out of scope" is read as *for the GQA cells*, which is what its evidence covers.

Metal has one cell, so reading 3 cannot be checked across geometries there (goinfer 75.2 ps per
query-head element; Ollama 12.6 ps per f16 KV byte = 79 GB/s, ~40% of the fabric). A phi3-mini or
7B Metal depth row — R12's instrument, one session — would say whether Metal's A follows query heads
too. Against the 2026-08-04 M0 run the Metal slope has come down 9.87 → 3.24 µs/position while F
moved 14.7 → 13.0 ms: nearly all of the six-week gain was slope work, and 8.9× of slope remains.

### 2.3 Prefill — the weight term

**CUDA** is done: `gemm_w4a8_mma` runs 531 ms at K=3900 on the 1.5B = 0.136 ms/token, flat in K,
against Ollama's *entire* marginal of 0.152. Nothing left to fund there.

**Metal.** M-03 (32×32 per-simdgroup block, all-lane dequant, A staged per threadgroup) shipped
2026-09-13 as scoped, but the TTFT ladder was never re-run; the audit's own §6 landscape puts 2× on
the GEMM at ~1.8× behind at K=512 and ~2.2–2.6× at 3900, with parity needing ~3.5× on the kernel
(0.73 → the peer's ≥2.4 TFLOPS). The pre-fix rows (§A, 2026-09-09: 3.3× behind at 512, 8.8× at
3900 on TTFT rate) are the "before"; the same-session peer re-run is owed (R4).

**WebGPU.** The batched path exists as of 2026-09-13 (`docs/measurements/prefill-batched-ttft-2026-09-13.md`,
6–9× over its own sequential loop) but 1024 tokens in 3.23 s on a 0.5B int8 is ~320 GFLOPS on a card
that does several TFLOPS in naive f32 — 8× behind CUDA's exact path at the same depth and 45× behind
its fast path. Never profiled. The GEMM shape and the per-query-row attention are the two suspects
(R10).

**Mac CPU.** 1.54× behind at K=512 where the weight term is nearly all of TTFT; the S-01 tile is at
99% of its issue ceiling for its µop count and S-05 (fold the −8 centering into the SDOT
accumulator, 96→72 µops, 1.33× counted, bit-identical) is the one lever inside it (R9). At 3900
goinfer is ahead (0.91×) because its attention term is flat (A3 head fan-out) and the peer's is not.

### 2.4 Prefill — the attention term

**CUDA — where P24 undersells itself.** P24's "even a perfect attention kernel is capped at ~1.33×
end-to-end" was computed on the L2-only arm (805 of 3255 ms). With L3 in, K=3900 on the 1.5B is
**531 GEMM + 805 attention + 57 other = 1393 ms**: attention is 58% of TTFT, and ~68% of the
marginal at the deepest interval (0.481 − 0.136 − ~0.02). `attn_fused` sits at 1.72% of tensor
peak and 12.6% occupancy — 96 blocks, 178 registers, 1.2 waves (`prefill-l2l3-phase1-2026-09-05.md`
§4). Attention marginal → 0.05–0.10 ms/token gives 0.21–0.26 total, 1.4–1.7× behind on marginal,
and a TTFT of **712–836 ms at 3900 against Ollama's 937** — ahead on TTFT at every depth swept, on
both dense sizes. BM=32 is P24's pre-registered first arm; a proper FA tile is what reaches the band
(R5).

**Metal.** M-04's 32-key tile shipped narrower than scoped (register-O + diagonal-α and GQA
grouping left as a follow-up). Attention was ~18% of TTFT at 3900 before it, ~1.5% at 256, so the
remaining e2e headroom there is ~1.2× (R4). The larger Metal prefill item is below the floor: M-02
moved it to 256 and M-01 stopped the LM head running per prompt token, but **8–255 tokens — the
common chat turn, every adapter prompt, every embedding request — still take the sequential path**
at ~77 tok/s, where Ollama reads ~790 tok/s at K=256. A 200-token turn is ~2.6 s vs ~0.25 s; the
batched path's own 200–270 tok/s makes it 0.75–1.0 s (R3).

### 2.5 Sampled decode

The one row where the peer is the control: on the 0.5B Ollama reads 268.7 greedy and 268.7 / 265.2
at `temperature 1.0` / `0.8 + top_p 0.95`; goinfer 333–342 greedy, 237.7 and 227.2 sampled
(§B5.1). That ~30% is D6's cliff ([`ollama-chase.md`](../ollama-chase.md) §D6): any nonzero
temperature flips `Sampler.ArgmaxEquivalent()` (`decoder/sampler.go:222`), forcing a V-wide logit
readback and a host softmax over V per token. Understood and scoped: a device-side bounded top-K
with a host-verified nucleus and a full-readback fallback. Every Mac row on the page is greedy, so
Metal's version is unmeasured; the counted host cost after the bounded-selection fix (~1.8 ms/token
at 152k vocab) is 13% of a 1.5B token and ~25% of a 0.5B one (R7).

### 2.6 MoE over capacity

**CUDA** is at Ollama parity and 0.72× llama.cpp on M35 because ~48% of the token is host→VRAM
expert DMA at line rate. L01's isolated benchmark says computing all of a layer's misses on the CPU
in parallel beats GPU-compute-plus-DMA at every miss count (1.32× at m=1 to 3.40× at m=8,
`decoder/l01_expert_bench_test.go`), the extraction and merge are built and verified end to end
(`cuda/l01_cpu_offload.go`, `TestL01_e2eDecode_matchesBaseline`), but the prototype is synchronous
and the pre-registered funding cell has not run (R11).

**Metal** is the weaker story: 61–81 synchronous command-buffer boundaries per token at a recorded
~14 ms each, staging at queue depth 1, and the one design that removes the boundary (a single
command buffer with a shared event) rejected on a dense-1.5B measurement where the boundary costs
0.2 ms (audit M-11, G-05). Counted ceiling after M-11/M-12/M-14: 3–4 tok/s from ~2. Anything toward
Zeno's 8.7–13.2 needs a hit-rate story the speculative-prefetch negative (0.1–0.3% exact-set match)
says we do not have (R11).

### 2.7 MoE prefill

The largest absolute gap on the page for the agent-turn workload: M35 at 8000 tokens is **1490 s of
wall-clock against 86 s**. M26's sequential prefill is flat with depth (46.7 → 46.4 ms/token from
M=512 to 2048) because 25 of 30 layers are windowed — P20's "single most fundable fact": essentially
all of it is weight traffic that expert-major batching compresses (the CPU twin, P18, measured 4.36×
bit-identical). M35 additionally needs a batched Gated-DeltaNet (R11).

### 2.8 Vision

`cuda/vision_encoder.go` projects every linear through `gemv_w8a8_batched` — the dp4a GEMV L3
replaced for text — and attends through `attn_img_batched` gridded (nH, M), the per-query-row shape
L2 replaced, at M=4096 patches over 27 layers, plus 54 `addInPlaceHost` round-trips per image. The
text twin measured 4.5× on the gemv category and 3.76× on attention, so 26.1 s → ~7 s is counted,
not guessed; hd=72 means `attn_fused` (hd ∈ {64,128}) will not take it unmodified. The ~1 s class a
GPU peer presumably reaches — unmeasured, no vision peer harness exists, `BENCH_VISION=1` is the new
instrument — needs the tower on a real GEMM at ≥10% of tensor peak (R8). On the Mac the tower runs on
the CPU (M-15); on the CPU it is f32 with `math.Exp` dominating (aikit audit: 2.7–3.1× per element,
exp > attention MACs at so400m) and the S-06 NEON transcendentals unwired (R9).

### 2.9 Speculative decode and concurrency

Metal's Θ=0.96 is an accurate report of an unbatched `ForwardN` (`metal/backend.go:702` — a loop of
`Forward`s, one command buffer each); CUDA's 0.25 with the same drafter is the existence proof that
batching the verify into one command buffer turns speculation from "declines" into 1.2–1.8× on agent
output (R12). Concurrency has no row on any backend; it is the axis a serving deployment buys, and
llama-server's slots are what W7 would measure against (R12).

---

## 3. Three ceilings the record carries that its own later numbers have moved

These are not errors in the measurements; they are conclusions that outlived the numbers under
them. Each is amended by the first brief that touches it, in the page that carries it, per
`CLAUDE.md`'s retraction rule (grep the figure with its unit; fix every instance at once).

1. **P24's end-to-end cap** (`queue-performance.md`): "capped at ~1.33× e2e" was computed on the
   L2-only arm. With L3 shipped the cap is ~2.4×, and the item is worth roughly what L2 and L3 each
   were. R5 rewrites the paragraph and keeps P24's pre-registered category rule as written.
2. **The Metal decode depth curve** (`benchmarks.md` §B3): 62.0 / 47.8 / 27.2 / 18.2 is from
   2026-08-09; the 2026-09-13 bench reads 72.3 / 67.9 / 51.4 / 40.4, and the peer's Metal depth
   row (85.2 / 79.1 / ~80 / 77.5) lives only in the archived `metal-verdict.md`. R2's measurement
   step re-runs both sides in one session and puts the pair in §B3, which also retires the N-03
   labelling caveat by measuring the served path.
3. **`metal-verdict.md` §2a's "no large unclaimed dense-decode pool on an M1 Pro for anyone"** was
   written against llama.cpp's 75–83 GB/s band before the MLX row existed at ~95 (1.5B) and ~162
   GB/s (7B). The archive is a record and is not edited; the correction lands in `benchmarks.md`
   §B3 with R1's first measurement and is noted in [`ollama-chase.md`](../ollama-chase.md) §10
   ("Refuted") next to the GEMV entries it qualifies.

---

## 4. Sequencing — two machines, two tracks

The two boxes run in parallel; nothing on one track blocks the other. Order within a track is
delta-per-hour on the machine that measures it, with measurement-only steps first where a build
would otherwise be scoped against a stale number.

**Mac track (Metal + Mac CPU).** R4 step 0 (re-measure the Metal prefill ladder post M-03/M-04
against the peer — one session, no code) → R1 (W4F16 decode GEMV; lane decision made, build
fundable) → R2 (decode attention in the peer's shape; measurement step first, Build now fundable
too) → R3 (the short-prompt floor) → R12's Mac halves (ForwardN batching; MLX and the Metal peer
depth row into `benchmarks.md`) → R9's Mac half (S-02 attribution, then S-05). R13 (CPU decode
attention, group-major, bit-identical) runs beside this track rather than in it: its step 0 is
measurement-only and fundable now on both boxes, and its build is aikit work that blocks on no lane
decision and on no other brief.

**Linux track (CUDA + WebGPU + Linux CPU).** R5 (prefill attention tile; P24 re-scoped) → R6 (the
flash-decode lane; lane decision made, build fundable; step 1 is a mechanism, not a re-roll) → R8
(the vision tower onto the L2/L3 kernels) → R7 (sampled-decode cliff, CUDA first) → R11 (L01
funding cell; P20 expert-major prefill) → R10 (WebGPU glue fusion; batched-prefill profile) → R9's
Linux half (the
0.5B anomaly).

**Owner decision, 2026-09-18: option 2 — default-on above a fidelity gate, held to the same bar
CUDA decode's own exact path already answers to, not a relaxed one.** R1, R2's Build phase, and R6
may all ship their fast decode kernels as the *default* once they pass the same pooled §3.2 gate
the prefill lane already uses (`completed/task-prefill-gap.md` §3): hard-flip rate pooled ≤
exact + 2√exact, teacher-forced top-1 agreement pooled ≥ exact − 2√d/N, mean KL pooled ≤ exact's
AND lower on ≥ half the decision-set prompts AND no single cell's fast mean KL > 1.1× that cell's
exact mean KL — reported per-cell, never vetoed per-cell (§3.2's own correction: a per-cell veto
fails an arm of equal quality most of the time). A kernel that does not ship the gate stays
opt-in, with the failure recorded beside it, exactly as the prefill contract already specifies for
a backend that fails. Below-floor short prompts stay exact, matching the prefill lane's own Floor
clause. Each brief's own decision-set K's and confirmation cells are as already registered in its
own text (R1 and R2 both name S at depth 3900 as a decision/confirmation cell; R6 registers its
own served-throughput band separately and inherits this fidelity mechanism alongside it, not
instead of it).

---

## 5. The briefs

Status table, kept current as briefs move:

| # | brief | box | size | status |
|---|---|---|---|---|
| R1 | Metal W4F16 decode GEMV — a fidelity-gated decode lane | Mac | M (kernel + gate) | **PARKED 2026-09-19: kernel proven correct (gate 1), catastrophic bug found and localized to layer 26's gate/up GEMV, root cause not found — see the record** |
| R2 | Metal decode attention in the peer's shape | Mac | M–L | step 0 done 2026-09-18 (gap narrowed to 1.96× at 3900). **Build: PARKED 2026-09-19 — attention_fa kernel proven correct in isolation (16/16 adversarial cases), real speed win in isolation (up to 1.26× at K=3900), but diverges starting at the 3rd real decode token past the depth floor; root cause not found — see the record** |
| R3 | Metal short-prompt floor 256 → 64, and what stays sequential | Mac | S | scoped |
| R4 | Metal prefill ladder re-run post M-03/M-04; GEMM step 2 if the band is missed | Mac | S (measure) + M (build) | **step 0 done 2026-09-18: K=512 2.54× behind, step 2 KILLED** |
| R5 | CUDA prefill attention tile — P24 re-scoped with the corrected cap | Linux | M | scoped |
| R6 | CUDA flash-decode lane — mechanism for the parked spike, then the kernel | Linux | M–L | **lane decision made 2026-09-18 (option 2, §3.2 gate) — build fundable** |
| R7 | Sampled-decode cliff — device-side bounded top-K | Linux first, then Mac | M | **CUDA filtered sampling SHIPPED 2026-09-20 (top_p 0.657→0.957 of greedy; top_k 0.961; min_p 0.971). R7b, same day, owner decision: temperature-only by Gumbel-max on every backend — CUDA 0.744→1.008, WebGPU 0.796→1.035 (device draw). Metal device kernel BUILT, GATED AND MEASURED 2026-09-20 (Mac session): 0 mismatches/15,840 draws, 12,000/12,000 real-generation tokens identical to host, device draw 0.96× greedy internally AND peer-verified (goinfer 1.15× Ollama greedy, 1.10× Ollama sampled, both Metal) — see the record.** |
| R8 | CUDA vision tower onto the L2/L3 kernels | Linux | M | scoped |
| R9 | CPU decode attribution (Mac fixed cost; the Linux 0.5B anomaly), then S-05 | both | S (measure) + M (aikit) | scoped |
| R10 | WebGPU glue fusion and on-device argmax; batched-prefill profile | Linux | M | scoped; profile first |
| R11 | MoE: L01 funding cell; P20 expert-major prefill; Metal pager measurement and M-11 | both | L | scoped |
| R12 | Metal `ForwardN` batching (P21); the missing peer rows (MLX, Metal depth, vision, W7) | both | S–M | **MLX row (i) re-confirmed 2026-09-18; Metal depth row (ii) done via R2 step 0; W7 (iv) done 2026-09-19 (simplified — see the record) — goinfer 60.1→36.1→36.4 tok/s at 1/2/4 clients (a real loss that plateaus), llama-server 84.8→95.9→149.7 (scales up), 4.11× gap at n=4; vision (iii) not attempted; P21 build not started** |
| R13 | CPU decode attention, group-major acc64 kernels (bit-identical) | both (aikit + goinfer) | S (measure) + M (two kernels ×2 ISAs, wiring) | **step 0 complete 2026-09-19**: (i) peer depth row done but not benchmarks.md-quality (thermal drift, re-run needed); (ii) softmax caps the QK+AV grouping ceiling to ~1.39-1.54× overall, not 1.8-1.85×; (iii) the cache-dedup gap is depth-dependent — nil below ~K=1024, 2-5× above K=2048, reinforcing (ii)'s decision depth. **SHIPPED 2026-09-20** — aikit kernel A/B real (1.53-2.39×); goinfer wiring found a real bug (softmax accidentally serialized, not the scheduler-contention red herring a first CPU profile suggested — `go tool trace`'s per-goroutine breakdown found the actual cause). Fixed: parity at depth 2048, **1.32× served at depth 8192**. `GOINFER_ATTN_GROUPED` defaults on. Three-arm/both-box/all-model sweep still not done |

Every brief below has the same shape: goal, the standing and the band registered here, what to read
first (prior art and the negatives not to re-propose), what to build, the gates, the measurement
protocol, the decision rule, and where the result is recorded. A brief that reaches its record step
touches `benchmarks.md` in the same commit as the measurement file; a brief that lands a negative
records it at the same value as a win.

---

### R1 · Metal W4F16 decode GEMV — a fidelity-gated decode lane

**Goal.** A Metal decode GEMV whose weight stream runs in the f16-FMA regime MLX and llama.cpp use
(≥140 GB/s at 7B, ≥130 at 1.5B) instead of the W4A8 integer regime (84–105 observed), and a
decode-side fidelity gate that decides whether it can be the default.

**Standing and the registered band.** `benchmarks.md` "Re-run 2026-09-17 — MacBook": 1.5B Metal
73.9 tok/s vs Ollama 85.8 (0.86×), 7B 21.9 vs 25.5 (0.86×); peer matrix 2026-09-04: MLX 109.8 /
37.9 on the same two models. Arithmetic in §2.1. **Band (1.5B, depth 128, served, same protocol):
ships at ≥92 tok/s (1.25×); parked at 81–92; killed below 81 (1.10×).** 7B: ships at ≥27.5, parked
24–27.5, killed below 24. The do-nothing arm is the shipped W4A8 path, interleaved. Registered
prediction, so it can be wrong in public: the W4F16 arm is *closer* to the CPU f32 reference than
W4A8 on ≥7 of 10 prompts (no activation quantisation) — if it is not, that is the first thing to
explain.

**Read first.** `completed/metal-verdict.md` §2a (the ceiling argument and the Stage-A/Stage-B
history — the Stage-B lesson, isolated GEMV wins that vanish end to end, is the reason the band is
served tok/s and not GB/s), audit `M-10` (a dispatch-site swap that was slower at every depth — the
depth bench is the arbiter), `completed/task-metal-cgofree-spike.md` "P1–P5" (the f16 activation
kernels `rmsnorm_f16`, `residual_f16`, `swiglu_f16`, `rope_f16` already exist in `metal/prefill.go`
for M>1 — reuse, do not duplicate), `ollama-chase.md` §2 (reduction width and fast-math are part of
the bit-identity contract; this lane is outside it and must say so), `completed/task-prefill-gap.md`
§3–§3.2 (the pooled gate and why discrete thresholds without resolving power fail an equal arm ~95%
of the time). Do not re-propose: Stage B, ICB, unretained references, the megakernel, dispatch-count
fusions, `sa_qv` — all recorded negatives on the *integer* path and irrelevant to this lane except as
method warnings.

**Build.** One new kernel family, `gemv_w4f16_*` beside the `gemv_w4a8_sa*` set in `metal/kernels.go`:
one simdgroup per output row (or per 2 rows if the register budget allows — measure both), weights
read as `uint4` (32 nibbles) per lane-step, nibble→half via the exponent-bias trick rather than a
convert per element, activation held as `half4` from a `rmsnorm_f16`-style producer, f32
accumulation across groups with the f16 group scale applied per group (the existing scale layout
is untouched — same buffers, same repack, same `.giw`). Fused epilogues mirror the integer set
(`_bias`, `_resid`, `_resid_bias`) so the dispatch count per layer does not grow; the o-proj input
convert replaces the one `quant_vec` dispatch M-audit §5 counted rather than adding one. The LM head
stays W8A8 in step 1 (its 84 GB/s with 19,000 threadgroups is a separate question — N-40 covers the
precision trade). Wire it behind `GOINFER_METAL_DECODE_LANE=w4f16` (name to taste; the flag is the
contract until the decision in §4 is taken), selected in `metal/model.go` at the pipeline build, so
one binary carries both arms.

**Gates.** (1) Kernel unit test vs a scalar f32 reference at the real shapes (K=1536/8960, N up to
151936), max abs and cosine, plus a fixed-input determinism check across two runs. (2) The
snapshot golden (`TestMetalSnapshotGolden`) is *not* the gate for this lane — it pins the W4A8
path; add a second golden entry for the lane so a later change to it is caught the same way, baked
after gate (3) passes. (3) The decode-side fidelity gate: the vsum gate's (a)/(b)/(c) criteria
(`cuda/vsum_split_gate_test.go` is the template; `metal/prefill_gate_ref_test.go` has the Metal
reference plumbing) against the CPU f32-weight, f32-activation reference, teacher-forced, 10
prompts × 64 positions, S (1.5B) as the decision cell and D7 as confirmation if it fits the fit-guard
on the day: hard flips ≤ exact + 2√exact, agreement ≥ exact − 1.0 pt and ≥ half the prompts, KL ≤
1.10× exact with 1.05–1.10 parked. Pre-register the file as
`docs/measurements/w4f16-decode-fidelity-PREREGISTERED.md` before a cell runs. (4) `go test
./metal/...` green, including every resident-parity test, with the lane both off and on.

**Measure.** `scripts/bench_peer.py` on the Mac, `BENCH_BACKENDS=metal`, greedy, depth 128,
`BENCH_MAX_LOADAVG=2.0` with the run count recorded (the macOS qualifier in `benchmarks.md`
Methodology), 0.5B / 1.5B / 7B / phi3-mini, three arms interleaved per cell: W4A8 (shipped), W4F16,
Ollama; add `mlx` to `BENCH_ENGINES` on the same run so the MLX row stops being a one-off (R12).
Then the depth bench (`GOINFER_METAL_DEPTH_BENCH=1 go test ./metal -run TestZZ_metalDepthBench`)
both arms, so the shallow win is not bought with a depth regression. Record the GPU-busy weight-term
bandwidth with the `metal/tax_test.go` instrument for the mechanism line, but the verdict is the
served number.

**Decision rule.** As registered above, on the 1.5B served cell, with the fidelity gate as a hard
precondition (a fast arm that fails (3) is a negative regardless of speed). A ship on speed with (c)
in the parked band is *parked*, not shipped — the same rule that parked the CUDA spike.

**Record.** `docs/measurements/w4f16-decode-2026-MM-DD.md` + the raw JSON under
`docs/measurements/peer-matrix-<date>/`; `benchmarks.md` §B3 gets the three-arm row and the §3
correction (the MLX bandwidth figures next to the verdict's ceiling sentence); the TL;DR Metal
decode row moves; `ollama-chase.md` §10 gets a one-line qualifier on the GEMV entries. If parked or
killed, the same files, same care.

**Build attempt, 2026-09-19 — PARKED, not killed**
([`w4f16-decode-investigation-2026-09-19.md`](../measurements/w4f16-decode-investigation-2026-09-19.md)).
Kernel built and scoped to the plain dense QKV/o-proj/gate-up path (down-proj and every special
case stay W4A8 — the real integration surface turned out to be 13+ dispatch sites, not a single
swap point). Gate (1) passed cleanly: the exponent-bias nibble dequant is bit-exact for all 16
values, and all three kernel variants match an independent scalar reference at cosine 1.000000
(K=1536/8960). The real-checkpoint sanity check then failed badly — cosine 0.48 against the shipped
path — and tracing it down layer by layer found a genuine, reproducible-from-a-clean-input
catastrophic break at layer 26 of qwen2.5-1.5B's 28 (cosine goes negative), localized to gate/up's
GEMV. Two natural explanations (a "massive activation" precision-loss theory; a model-level-chaos
theory, tested directly by perturbing the trusted W4A8 path itself with a matched-magnitude
difference and finding it stays mild) were both directly tested and falsified. The mechanism itself
was not found before the investigation was parked — real, reusable groundwork (a proven-correct
kernel, a precise localization, a clean minimal reproducer that fails loudly by design) for
whoever picks this back up, not a dead end. No speed measurement was taken — there is no point
benchmarking a lane that fails a basic sanity check before reaching the fidelity gate.

**Out of scope.** The LM head (N-40), KV precision, any change to the W4A8 path, prefill (already
f16), the 26B/35B paged path (its GEMVs are launch-floor-bound, not stream-bound — audit §5).

---

### R2 · Metal decode attention in the peer's shape

**Goal.** Take the Metal decode attention depth term from ~2.7 µs/position (10.9 ms/token at 4000
on the 1.5B) toward the peer's ~0.3 µs, by changing the kernel's *shape* rather than its memory
pattern — the axis the four recorded negatives did not move on.

**Standing and the registered band.** goinfer 72.3 / 67.9 / 51.4 / 40.4 tok/s at 128 / 512 / 2048 /
4000 (2026-09-13 depth bench, W4A8 1.5B); peer 85.2 / 79.1 / ~80 / 77.5 (`completed/metal-verdict.md`
row 4). Arithmetic in §2.2. **Band (depth bench, 1.5B, same protocol, min-of-batches): ships at
≥60 tok/s at 4000 AND ≥58 at 2048 AND no regression beyond 3% at 128; parked at 48–60 at 4000;
killed below 48** (a depth term still above ~7 ms means the M1's dispatch/occupancy floor won a
fifth time, and that is worth writing down as firmly as the first four).

**Read first — all of it, before a line of MSL.** `ollama-chase.md` §A1-Metal, §A2-Metal (the
collapse probe: 75% of attention at depth is the 6× GQA-redundant K/V read; the four dedup/split
attempts and *why each lost* — dedup needs co-location, co-location kills occupancy, split needs
dispatches, dispatches are the floor), audit `M-09` (the staged K read, 0.43×, and its plausible
cause: half the threads idle during compute plus 64 barrier pairs), `metal/kernels.go`'s
`attention` kernel (the shipped structure: one threadgroup of 128 per query head, thread-per-key
serial dot, thread-per-dim serial V fold with the 8-wide load batch, the 128-wide pinned reduction
tree), the deep-context tiled path added 2026-09-17 (`26f64807` — same structure, tiled at 4096
keys, so it inherits the same term), and `ollama-chase.md` §2's width rule (`tgReduce*`, and that
`simd_sum` and `max` are order-exact and therefore exempt). Then read llama.cpp's Metal
`flash_attn_ext_vec` for its *shape* — lanes over the head dim, keys over simdgroups, the query
heads of a GQA group in one threadgroup, online softmax per simdgroup, a reduce at the end — not for
its code.

**Step 0 — measurement, fundable now without the lane decision.** Re-run the depth bench and the
peer's depth curve in one session under the M0 protocol (`metal-verdict.md` §5) on the current
tree, both engines, same GGUF, depths {128, 512, 1024, 2048, 4000}, and put the pair into
`benchmarks.md` §B3 replacing the 2026-08-09 table (this is §3 item 2). This is the "before" every
later number in this brief is measured against, and it settles whether the 0.52× at 4000 is current.

**Build.** A new kernel `attention_fa` (name to taste) beside `attention`, not a modification of it:
threadgroups gridded (kvHead × keySplit) rather than (queryHead); 128 threads = 4 simdgroups; each
simdgroup walks its key range with all 32 lanes reading one K row cooperatively (`half4` per lane at
hd=128 — one coalesced 256 B read per key per simdgroup, against the shipped 32-lane 512 B-strided
gather M-09 counted), computes the six group-head dots per key with a per-lane 4-wide product and
`simd_sum`, keeps per-head running (m, l) and a per-lane `acc[6][4]` for V, and emits a partial per
(simdgroup) → per (threadgroup) via threadgroup memory → per (kvHead, split) to a small buffer; a
second dispatch combines splits in a fixed order (deterministic; +1 dispatch per layer, which the
P4/P5 census says is lost in the ~310-dispatch floor — verify, do not assume). Split count S chosen
so kvHead × S ≥ 2× the core count at depth ≥1024 and S=1 below a measured crossover (the split-KV
lesson: gate on the effective attended span `nWin`, per layer, not on position). Sinks (gpt-oss),
windows (Mistral/Gemma) and the f32-KV twin (Gemma's sandwich path) are in scope only after the
dense-GQA kernel clears the band on the 1.5B — a family-by-family port, each with its own golden.

**Gates.** (1) `TestAttention`-class unit gate vs the shipped kernel at tolerance (not byte-exact —
the reduction order differs by design) at nKeys ∈ {1, 127, 128, 129, 255, 256, 257, 999, 2048,
4095, 4096, 4097, 8192}, straddling every width and tile boundary, and vs the CPU f64 reference at
cosine ≥ 0.9999. (2) A new snapshot-golden entry for the kernel (machine-pinned, past both reduction
widths, per `metal/snapshot_golden_test.go`'s own rules) — baked *after* gate (3). (3) The same
teacher-forced fidelity gate R1 uses, S at depth 3900 as the decision cell, W4A8 + shipped attention
as the exact arm; expected to pass comfortably (`reduction-tree-accuracy-2026-09-12.md`: a blocked
fold measured 1.76–4.93× *closer* to f64 than the sequential one on CUDA — a prediction here, not a
result). (4) Paged ≡ non-paged byte-identity holds trivially (same kernel both arms) — keep the test.

**Gate (1), amended 2026-09-19 — the inputs are part of the gate.** A key-parallel kernel with an
online softmax has one piece of new arithmetic — the rescale when the running max moves, and
the fixed-order combine across splits — and a diffuse test cannot see it. `metal/attn_shape_test.go`'s
header says why in this repo's own words: random q/k give every key ~1/nKeys, errors average out,
and a random-weight synthetic passed at 0.997 while the real model did not. So gate (1) runs
`runAttnCase`'s pattern, not random fill — a sharp hot key, an outlier V row, a near-zero-norm sink
at key 0, the reference computed over the f16-rounded K/V — with the hot key placed in the first
split, in the LAST split, and on each side of a split boundary; plus a rising score ramp that moves
the running max in every split; at S ∈ {1, 2, 4}; and on a windowed case with `winStart` > 0. Bars
as that test: cosine ≥ 0.9999 and max|Δ| ≤ 1e-2. A rescale or combine defect is wrong by a fraction
of |V| on these inputs and by almost nothing on diffuse ones. Two debts in the shipped tests, found
while checking this, that the new kernel's tests must not copy. `TestMetalKVI8_DeepContext` drives
`attention_i8`'s tiled path at 8192 keys and asserts only that the output is finite, so that path
has no correctness test (the f16 `attention` kernel's does: `TestAttention_ShippedKernelShapes` puts
its hot key at nKeys/2, which lands in a later tile at 8192 / 16384 / 32768). And
`TestAttentionPrefillFused` uses uniform random fill at ≤ 520 keys against a 0.999 cosine and a
0.05 max|Δ| that is about the size of the output itself on those inputs, where the CUDA twin holds
1e-3 of max|V| and 0.9999 (`cuda/attn_fused_test.go`); what covers that kernel on sharp attention
today is the served §3.2 gate, not a unit test. Each is one `runAttnCase`-style case away from
closed; R4 owns the second.

**Measure.** The depth bench, three runs, both arms, then `scripts/bench_peer.py` at depths 128 and a
4000-token calibrated prompt (the harness's Phase B is hardcoded CUDA-only — extend it or use the
depth bench as the instrument of record and say so). Cross-check the depth term's GPU-busy time with
the `metal/tax_test.go` timestamps so the mechanism line can say what changed.

**Decision rule.** The band above, on the depth bench. A shallow regression beyond 3% at 128 parks
the kernel regardless of the depth win, because the 1.5B at 128 is the cell the promotion gate
reads first.

**Record.** `docs/measurements/metal-attn-fa-PREREGISTERED.md` before the first cell,
`metal-attn-fa-2026-MM-DD.md` after; `benchmarks.md` §B3 depth table and the TL;DR Metal decode row;
`ollama-chase.md` §A2-Metal gets a dated paragraph either way — a fifth negative belongs next to the
four.

**Step 0 result, 2026-09-18** ([`metal-depth-r2-2026-09-18.md`](../measurements/metal-depth-r2-2026-09-18.md)).
Re-ran the depth bench peer-paired for the first time (`scripts/bench_peer.py`, extended with
`BENCH_DEPTH_BACKEND` since Phase B was CUDA-only, per this brief's own suggestion). A first pass
used `OLLAMA_KV_CACHE_TYPE=q8_0` (wrongly inherited from R4's prefill protocol) that was costing
Ollama 14–36% of its decode throughput, growing with depth; isolated and excluded the same day.
**Against the only true same-session comparison that exists** (`metal-verdict.md` M0, 2026-08-04:
63.8/39.8/28.4/18.5 vs peer 85.2/79.1/~80/77.5, ratio 1.34×→4.19×), **Ollama held flat (±4%) across
six weeks while goinfer more than doubled at depth** (18.5→39.1 tok/s at ~4000) — the gap is real
and closing, from 4.19× to 1.96× behind at 3900–4000, on a held-constant peer. `benchmarks.md` §B3
and its TL;DR row updated in the same commit; the pre-fix goinfer-only table moved to
`legacy-benchmarks.md`; audit N-03's caveat marked superseded (this run measures the real serving
path directly rather than arguing the internal one is representative of it).

**Build result — PARKED, 2026-09-19** ([`r2-attn-fa-2026-09-19.md`](../measurements/r2-attn-fa-2026-09-19.md)).
`attention_fa`/`attention_fa_combine` built exactly as scoped (grid by kvHead×split, cooperative
register-direct loads, S sized for real occupancy). Gate (1) fully clear: 16/16 adversarial cases
(hot key at every split boundary, a rising score ramp, S ∈ {1,2,4,8,14}) pass at cosine 1.0000000.
Isolated-kernel speed is real but modest and depth-gated: S=1 is uniformly WORSE than shipped at
every depth (0.38-0.53×, not just "below a crossover" as this brief's own Build text first
framed it — a correction the probe surfaced), crossing to a win only past K≈1536, reaching 1.26×
at K=3900 with S=28. Wired into production (env-gated off by default,
`GOINFER_METAL_ATTN_FA=1`; confirmed a true no-op when disabled — `TestMetalSnapshotGolden` byte-
identical), a real qwen2.5-1.5b decode run diverges starting at the THIRD token past the depth
floor — bit-perfect for two tokens, then a stable, fully deterministic wrong answer (cosine
~0.9995, not a growing drift). Root cause not found; the dispatch parameters themselves were
ruled out via a debug print (`GOINFER_ATTNFA_DEBUG=1`, kept in the source). Same PARKED shape as
R1. Kept, not shipped: `TestAttentionFA_endToEndReproduction` (heavy-gated) is the keeper repro
for whoever picks this up. End-to-end served tok/s against the registered band was never reached.

**Out of scope.** KV quantisation (§A3 — a byte lever on a term this brief says is not byte-bound),
prefill attention (R4), the paged families' attention (their term is the command-buffer boundary,
M-11).

---

### R3 · Metal short-prompt floor, and what stays sequential

**Goal.** Put the common chat turn (8–255 prompt tokens) on the batched prefill path, and account
for what still runs sequentially after that (adapters, declined families, embeddings).

**Standing and the registered band.** Sequential path ~74–77 tok/s TTFT rate at K<256; the batched
path's own record 272 tok/s at P=256 and 4.56× over sequential at P=128 in the 2026-08-16 measurement
(`ollama-chase.md` §13, pre-M-03); Ollama 789.6 tok/s at K=256 (§A, 2026-09-09). **Band: the floor
moves to the smallest K in {64, 128} at which the §3.2 pooled gate ships; at that K the batched arm
must beat sequential by ≥2× on TTFT (ships), 1.3–2× parked, below 1.3× the floor stays.**

**Read first.** Audit `M-01`, `M-02` and their closure notes (`6cc862a0` — the floor is 256 today,
`GOINFER_METAL_FAST_PREFILL_FLOOR` read in `metal/backend.go:526`; `ForwardNoLogits` shipped
synchronous, the `noHead` executor-job version with ~0.9 ms/token of encode-ahead overlap still
open), `G-02`/`G-08` (the pooled gate drops missing cells silently and never exercises
`startPos > 0`, which every prefix-reuse turn uses — fix G-08 as part of this brief, since a
short-prompt floor is exactly the prefix-reuse regime), and the gate record
`docs/measurements/prefill-gate-l1-ref-b-2026-09-09.md` (K=256 PASS twice — the premise the 512
floor was set on was wrong once already; do not set the next floor on a premise either).

**Build.** Nothing new in kernel space. Extend `TestPrefillGateVsReference` (`metal/prefill_gate_ref_test.go`)
to run K ∈ {64, 128} as decision cells with `startPos ∈ {0, 128}`; set the floor from the result;
then the `noHead` executor-job completion of M-01 if the sequential residue (adapters, Gemma-3-declined
shapes, `/v1/embeddings`) measures above 10% of its TTFT in the LM head after the synchronous fix.

**Measure.** `scripts/bench_peer_prefill.py --backend metal --models 1.5B --depths 32,64,128,256`,
three arms interleaved (sequential via `--exact-prefill`, batched, Ollama), n=6 prefixes per cell,
the same session as R4 step 0 if the box allows. Report TTFT in ms for a 100- and a 200-token turn
in the write-up — that is the number a user feels.

**Decision rule.** As registered. A gate failure at 64 with a pass at 128 sets 128; that is a
result, not a partial.

**Record.** `docs/measurements/metal-prefill-floor-2026-MM-DD.md`; `benchmarks.md` §A Metal
prefill subsection gains the short-K rows; the M-02 closure note gets the new floor and its gate
cells; `docs/env-vars.md` if the default changes.

**Out of scope.** The GEMM (R4), anything on CUDA (its floor is 512 on a gate that failed at 256 —
a separate question with its own record).

---

### R4 · Metal prefill ladder re-run post M-03/M-04; GEMM step 2 if the band is missed

**Goal.** Replace the pre-fix Metal prefill peer rows with a same-session measurement of the current
kernels, and fund the GEMM's second step only against that number.

**Standing and the registered band.** Pre-fix (2026-09-09, §A): 3.3× behind at K=512, 8.8× at 3900
on TTFT rate; the L2 record put 3900 at ~4.3× behind before M-03/M-04. The audit's landscape (§6)
projects, after M-03 at 2× on the GEMM: ~1.8× behind at 512, ~2.2–2.6× at 3900, ~2.2× with M-04.
**Step 0 has no band — it is the measurement.** **Step 2's band: K=512 ≤1.8× behind ships (the
audit's own projection for a 2× GEMM), 1.8–2.4× parked, above 2.4× killed** (the tile is not the
lever the audit thought and the parity path needs a different kernel design — say so).

**Read first.** Audit `M-03`, `M-04` and their closure notes (`f6c222ee`, `c660ab78` — what shipped
and what was scoped out: register-O with a diagonal-α MMA, GQA query-head grouping), `N-01` (why no
ratio was invented without the re-run), `metal/prefill.go` (`gemm_w4f16_store`,
`attention_prefill_fused`), the CUDA L3 record (`prefill-l2l3-phase2-2026-09-05.md`: 4.52× on the
category from a worse starting shape — the existence proof that a group-scaled int4 GEMM can be
made fast; the mechanism does not transfer, the ambition does).

**Step 0 — measure.** `scripts/bench_peer_prefill.py --backend metal --models 1.5B --depths
256,512,1024,2048,3900`, three arms (`goinfer_exact`, `goinfer` fast, Ollama), n=6, same session,
`BENCH_MAX_LOADAVG=2.0`, on `~/models` only. Report TTFT rate and the local interval slopes exactly
as the 2026-09-09 row did, so the two are comparable. Add D7 if the fit-guard admits it that day
(M-07's third copy is the reason it may not — note which).

**Step 2 — build, only if step 0's K=512 ratio is above 2.0× behind.** The audit's remaining
levers in order: stage the 32-row A panel for a 32-wide K slab once per threadgroup (M-03 shipped
per-threadgroup staging — verify what is left), 2D register tiling across N, then the M-04
follow-ups. Oracle: the §3.2 pooled gate, not bit-identity (f16 storage rounding already differs
from exact) — `TestPrefillGateVsReference` on S, the K=256/1024 deciding set plus 3900.

**Decision rule.** Step 2 ships/parks/kills on the K=512 ratio as registered, with the pooled
gate as precondition.

**Record.** `docs/measurements/metal-prefill-ladder-2026-MM-DD.md`; `benchmarks.md` §A "Metal
prefill" subsection replaced (the pre-fix table moves to `legacy-benchmarks.md` verbatim, per the
page's rule) and the TL;DR row; audit M-03's closure note gets its "re-measurement not run" line
closed.

**Step 0 result, 2026-09-18** ([`metal-prefill-ladder-2026-09-18.md`](../measurements/metal-prefill-ladder-2026-09-18.md)).
K=512 measured 2.54× behind Ollama on TTFT — improved from the 3.33× pre-fix baseline (M-03/M-04
did help), but past the >2.4× kill line, not in the ≤1.8× ship or 1.8–2.4× park bands. **Step 2 is
KILLED per the pre-registered rule: the tile is not the lever the audit thought.** The whole-curve
marginal fit was invalid on all three arms (TTFT superlinear in K, including Ollama); on local
intervals goinfer fast's own marginal cost grows with depth faster than Ollama's does, so the
ratio widens (3.17×→3.60×) rather than holding flat — a mechanism profile, not a re-roll of the
same tile design, is what a future step 2 attempt would need. `benchmarks.md` §A and its TL;DR row
updated in the same commit; audit M-03's closure note updated.

**Out of scope.** The floor (R3), attention below 18% of TTFT, MoE prefill (R11).

---

### R5 · CUDA prefill attention tile — P24 re-scoped with the corrected cap

**Goal.** Take `attn_fused` off its 1.72%-of-tensor-peak / 12.6%-occupancy floor, in two phases,
and put CUDA ahead of Ollama on TTFT at every depth swept.

**Standing and the registered band.** K=3900, 1.5B, fast path: 531 GEMM + 805 attention + 57 other
= 1393 ms (`prefill-l2l3-phase2-2026-09-05.md`); marginal at the deepest interval 0.481 ms/token vs
Ollama 0.152 (3.16× behind); TTFT 1.436 s vs 0.937 (§B2 table, 2715 vs 4161 tok/s). **Phase 1
(BM=32 grid arm) keeps P24's own pre-registered rule verbatim** — ≥1.25× on the `TestPrefillDecomp`
attention category at K=3900 with no regression at K=512 ships, 1.05–1.25× parked, below 1.05×
parked. **Phase 2 (an FA-style tile: K/V tiles loaded once per 64–128-row query block, both products
on tensor cores, running max/denominator in registers) registers: ≥2.5× on the category (805 → ≤320
ms) ships; 1.5–2.5× parked; below 1.5× killed.** End-to-end consequence, stated so it can be checked:
phase 2 at its ship bar puts TTFT@3900 at ≤0.95 s — TTFT parity with the peer — and the deepest
marginal at ~0.25 ms/token (~1.6× behind on throughput, from 3.16×).

**Read first.** P24 in full (`queue-performance.md`) — the grid arithmetic (96 blocks, 178 regs, 2
blocks/SM, 1.2 waves), the pre-registered rule, and the do-nothing arm; then correct its cap
paragraph (§3 item 1) in the same commit as phase 1's first measurement. `cuda/attn_fused.cu` and
its selection rule (`useAttnFused`, hd ∈ {64,128}, the 512 floor, `cuda/prefill.go`).
`prefill-l2l3-phase1-2026-09-05.md` §4 (the `ncu` capture — re-profile after each phase: the rule
is profile the unit before designing the next fix, and a phase-1 grid change moves the bound).
`completed/task-prefill-gap.md` §2.2 (why the per-query-row shape is O(K²) and what a flat marginal
looks like). CLAUDE.md's PTX note: regeneration is only reproducible at nvrtc 12.6.85 via the
documented venv; the box is on 12.9 — plan the regen step, do not discover it.

**Build.** Phase 1: `BM=32` (two warps per block) as P24 specifies, registers/thread unchanged,
192 blocks. Phase 2: the tile — a query block of 64–128 rows per CTA, K/V streamed in 64-key tiles
through shared memory once per block, QKᵀ and PV on `mma.sync` f16 with f32 accumulation, online
softmax per row in registers, causal masking at the tile level so no wasted tiles beyond the ragged
diagonal; hd ∈ {64,128} first, hd=96 (phi3) and hd=72 (the vision tower, R8) as follow-ups with
padded tiles. `TestKernelFMALint` stays green (it extends to new kernels automatically; a
fidelity-gated kernel is still held to explicit intrinsics so its bits are a function of the source).

**Gates.** `TestAttnFused_vsF16Reference` (exact f64 on the kernel's own f16 operands — isolates a
tiling bug from operand precision) at every tile boundary the new BM/tile sizes introduce; then the
§3 fidelity gate (`TestPrefillGateVsReference`, `cuda/`) at K ∈ {512, 1024, 3900} — BM is not a
numerics knob in principle, and the gate is how that claim is checked; `TestPrefillDivergenceRate`
is irrelevant here (fast path, not exact) and must not be cited as if it applied.

**Measure.** `TestPrefillDecomp` attention category + gemv category as the control (it must not
move), K ∈ {128, 512, 2048, 3900}, best-of-3 per the harness rule, arms ~15 min apart, box idle;
`TestPrefillTTFT` end to end; then a Phase-4-style peer sweep (`scripts/bench_peer_prefill.py`,
interleaved, unique prefixes, `K ∈ {128,512,1024,2048,3900}`) once a phase ships — the peer row is
what moves the TL;DR.

**Decision rule.** Per phase, as registered. Phase 2 is not started on a phase-1 park; a phase-1
park is a finding about the grid, and the `ncu` re-profile says what phase 2 should target.

**Record.** `docs/measurements/attn-fused-bm32-2026-MM-DD.md` / `attn-fused-tile-…`; P24 amended
in place (cap corrected, phases and results); `benchmarks.md` §B2 prefill table and the TL;DR
prefill row; `roadmap.md`'s open-program line for the prefill gap.

**Out of scope.** The GEMM (at ceiling), the exact path (`gemv_w4a8_rn` and `attn_batched` stay
what the parity gates run), MoE prefill (R11), the image-block kernel (R8 borrows the tile).

---

### R6 · CUDA flash-decode lane — a mechanism for the parked spike, then the kernel

**Goal.** Parity with Ollama on dense CUDA decode at depth, by opening a fidelity-gated decode
attention lane and building the kernel the ncu record points at.

**Standing and the registered band.** 1.5B: 227 / 161 / 125 tok/s at 128 / 2048 / 3900 vs Ollama
195 / 180 / 175 (0.72× at 3900; llama.cpp 0.59×); D7 at 8000: 35.7 vs 56.7 (0.63×). Depth term
3.6 ms/token at 3900 vs 0.61 (Ollama) / 0.32 (llama.cpp). The V-sum split spike: +40–43% on the
kernel, **+16.2% served** at 8k on D7 (`vsum-split-spike-2026-09-13.md`), parked at KL 1.0585×
(`vsum-split-fidelity-2026-09-13.md`). **Band (served, `scripts/bench_peer.py`, greedy): 1.5B at 3900 ≥170
tok/s ships (1.36×), 150–170 parked, below 150 killed; D7 at 8000 ≥50 ships, 44–50 parked, below 44
killed (the spike alone reads 44.8 — the kernel must beat its own first step).** No shallow
regression beyond 2% at 128 (split-KV's own 128 column cost 7–14% when forced; the lane gates on
`nWin` like the split path does).

**Read first.** `task-decode-splitkv-attention.md` (the associativity argument, what is and is not
byte-identical, the OPEN item at 8000 now answered), the whole 2026-09-12/13 measurement set in
order — `splitkv-8000-reanchor`, `splitkv-mechanism-ncu` (the gate's stated reason refuted; DRAM
27% vs 68% between the GQA and MHA geometries — suggested, not established), `splitkv-stall-profile`
(opposite bottlenecks in the two kernels; the lesson that a stall profile is per kernel),
`splitkv-q-staging`, `vsum-split-spike`, `vsum-split-fidelity` (and its PREREGISTERED twin —
the rule that a re-run needs a mechanism, never a re-roll), `reduction-tree-accuracy-2026-09-12.md`,
`splitkv-d7-fthreshold` and `splitkv-f-depth-invariance` (the gate is now keyed on nKV·hd,
`cuda/resident.go:249`); `ollama-chase.md` §A2, §D4 (Ollama's `flash_attn_ext`: tiled, parallel
over keys, online softmax, *not* bit-exact — which is what this lane accepts), §7 (the strategic
fork, now partly superseded by L3 on the prefill side — read for the trap paragraph on
tolerance-gated defaults); `cuda/decode_splitkv.cu`, `cuda/attn_block.cu`.

**Step 1 — the mechanism, before any new kernel.** The spike's tree is measured *closer* to f64
than the sequential fold, yet its KL against the reference rose 5.85% on D7 at 8000 while S at 3900
read 1.0026×. That is a contradiction with exactly one of three shapes: a defect in the spike's
combine (the partial-max source, the α = exp(m_partial − m_global) path, f16 anywhere it should be
f32), a prompt-set effect (set A's near-tie share at 8000 — the worst near-tie gap was identical
between arms, 10.899%, which argues against, but check the per-prompt KL split), or a reference
effect (the f32-weight one-worker reference at 8000 took 4 h — was it the same reference build for
both arms; the doc's own rebase note says the scoring binary predates it). Reproduce the KL delta
on 3 prompts with `S ∈ {1, 4}` where S=1 must reproduce the exact arm bit-for-bit (a spike with S=1
that is not byte-identical to the exact path has a defect, and that is the mechanism). Outcome of
step 1 is a written finding either way; only a *found* mechanism unparks the spike.

**Step 2 — the kernel.** `attn_decode_fa` (name to taste) beside `decode_splitkv.cu`: one CTA per
(kvHead, keySplit) handling all nH/nKV query heads of the group (6 on the 1.5B, 7 on D7, 1 on
phi3), warps parallel over keys inside the split, q for the group staged in shared memory once
(the q-staging finding: half the scores kernel's LSU issue was redundant q loads), `float4` K reads,
online softmax per head in registers, V accumulated per head per lane, one deterministic combine
kernel over splits. The split count S from the vsum spike's flat trend (S=4 captured 98% of its
win) as the starting point, gated on `nWin` per layer with the existing per-geometry table
mechanism, `splitkvNever`'s class now keyed on nKV·hd. `TestKernelFMALint` green. The exact path
(`attn_batched` M=1 / split-KV) stays what ships when the lane is off and what the parity gates run.

**Gates.** (1) S=1 arm byte-identical to the exact split path (the built-in defect detector). (2)
The vsum gate's (a)/(b)/(c) on D7 at 8000 (decision) and S at 3900 (confirmation), pre-registered
as `attn-decode-fa-fidelity-PREREGISTERED.md`, with the same D2 reference discipline (f32 weights
and activations, exact f64 attention, one worker). (3) `TestBackendResidentWired`-class token
identity with the lane off; a new golden for the lane on.

**Measure.** `ncu` on the new kernel first (occupancy, DRAM/L1TEX/L2 SoL, the stall reasons —
a per-kernel profile, per the stall-profile lesson), then `scripts/bench_splitkv.py`-style paired
arms (lane on/off, adjacent, alternating order, fresh serve per arm) at {128, 512, 2048, 3900} on
0.5B/1.5B/phi3-mini/gemma3-1b and 8000 on D7, then the peer sweep (`scripts/bench_peer.py`, both boxes'
protocol, `BENCH_DEEP_CTX` scoped to the 8000 cells only — the 2026-09-17 wrapper lesson).

**Decision rule.** As registered, fidelity gate as precondition. A geometry where the single-block
path wins (phi3 class, high KV traffic per key) is expected to decline via the gate — a decline is
not a regression; a forced regression there is.

**Record.** Measurement files as named; `benchmarks.md` §B8 anchor table re-run with the lane on
(a new anchor, dated, the old kept as "exact path"), §B7.1 at 8k/16k/32k, the TL;DR deep-context
row; `task-decode-splitkv-attention.md` status; `ollama-chase.md` §4 closing paragraph amended
(Campaign A's "bit-identity ceiling" line gets its sequel).

**Out of scope.** KV quantisation, prefill (R5), Metal (R2 — different hardware, different
bound; do not carry this diagnosis across, per the record's own §A2-Metal lesson).

**Amendment, 2026-09-19 — from reading the spike's source and from §2.2's fit.** Six things, the
first three about step 1 as written.

1. **The S=1 detector is vacuous as the code stands.** Both the loader (`cuda/backend.go`, where
   `GOINFER_SPLITKV_VSUM_SPLIT` is parsed) and the launch site (`cuda/resident.go`, the V-sum step of
   the split-KV decode) require S > 1, so S=1 silently runs the exact `splitkv_vsum` and the check
   compares the exact path with itself. Step 1 needs a test-only way to run `splitkv_vsum_partial` +
   `splitkv_vsum_combine` at nSplit=1, and a precondition that proves they launched (a launch
   counter — the mirror image of the fidelity gate's "630/630 rows differing").
2. **The spike has no partial max and no α.** Scores and softmax are the exact kernels; only the V
   fold is split — contiguous chunks from (nKeys − winStart, nSplit), an ascending combine, then
   × inv. So the combine-defect surface is the chunk bounds on windowed layers (`winStart` > 0), the
   partials layout and its `nH × maxHd × S` sizing against a per-layer hd, and the z-grid. The
   partial-max / α / f16 candidates named above belong to step 2's kernel, not to this spike.
3. **If the detector comes back clean, the open question is the gate's resolving power.** With
   prompts as the unit (positions inside one teacher-forced prompt share a cache and are not
   independent), D7's per-prompt KL ratios average 1.057 with a standard error of 0.029: t = 1.96,
   two-sided p ≈ 0.08; the "8 of 10 higher" is a sign test at p ≈ 0.11. S reads 0.994, spike lower
   on 6 of 10. A null arm measures what the 1.05 line can resolve: S ∈ {2, 8, 16} beside S=4, all
   blocked trees at least as accurate as the sequential fold
   (`reduction-tree-accuracy-2026-09-12.md`), same reference, same prompts. If equally valid trees
   scatter in KL by about as much as 5.85%, band (c) has no resolving power at D7@8000 and *that* is
   the mechanism — the `completed/task-prefill-gap.md` §3.2 precedent, where two single-cell misses
   turned out to sit under thresholds an equal arm failed most of the time. If every S > 1 sits
   consistently above exact, the excess is a property of blocked-vs-sequential at this cell, and the
   top-k-restricted KL the fidelity record proposed says whether it lives in the head or the tail.
   Either way it is a new pre-registration on prompt set B, never a re-score of set A.
4. **Step 2: do not stop at the V-sum, and justify the group axis by issue, not bytes.** With the
   V-sum split, `splitkv_scores` is ~70% of the kernel (218.9 of 318.1 µs at S=4 on D7@8000), and
   after q-staging it still reads 117.67 cyc of lg_throttle beside 31.89 of long_scoreboard
   (`splitkv-q-staging-2026-09-13.md`) — two stalls, both paid per loaded element. One K load
   serving all nH/nKV dots cuts issued K loads G× and amortises each memory latency over G dots.
   That is a different argument from the traffic one `splitkv-kernel-exploration-2026-09-13.md`
   retired, which stands; §2.2's reading 3 is the evidence for it. A grouped `splitkv_scores` is
   itself bit-identical (each dot keeps its d-order; scores are independent outputs), so it can be
   built and ncu-profiled inside the exact lane, under `TestSplitKV_bitIdentical`, before the
   non-exact kernel exists — with the recanon doc's <5% kill / 5–15% park / >15% keep rule on the
   `scores` kernel, and the cost named in advance: G×hd floats of staged q per block where
   q-staging's hd already cost 12% occupancy.
5. **An acceptance check beside the band, and what "KV quantisation out of scope" covers.** After
   the kernel, A per KV byte should be flat across the 0.5B / 1.5B / 7B (and phi3-mini wherever the
   lane runs), as the peer's is — today goinfer's A is flat per *query-head element* instead
   (§2.2). And the decline this brief expects on the phi3 class leaves that class with no depth
   lever from step 2 at all: it is already byte-bound (67.83% DRAM), its 1.8× against Ollama is f32
   KV against f16, and f16 resident KV is its lever — a step 3 with its own pre-registration, not
   part of this band. From §2.2's fit, halving phi3-mini's A reads 79 tok/s at 3900 (today 56.2,
   Ollama 73.3): **ships at ≥70, parked 62–70, killed below 62**, fidelity-gated under the same
   pooled §3.2 rule. No family is known to need f32 KV: Metal ships f16 KV for every family, and
   the one finding that said otherwise (Gemma, 0.64 against 0.92 cosine) is refuted in
   `metal/model.go`'s own comment — the crater was the position-0 K/V compute, precision was a red
   herring — although the old claim still stands in the comment above `kv_store_f32` in
   `metal/kernels.go` and on the `kvF32` field. *(Corrected the same day: this sentence first read
   "Gemma kept on f32", from that stale comment.)* The family decision is the gate's, behind one
   precondition: record max |K| and max |V| per layer on the gate prompts, and refuse f16 KV for a
   family that comes within 2× of f16's 65504. Every GQA
   geometry inherits the same 2× floor once step 2 makes it byte-bound, so the trigger is general:
   decode attention reading ≥~50% DRAM under ncu.
6. **Kernel-level gate vectors for `attn_decode_fa` and its combine.** R6's gates are S=1 identity,
   the served fidelity gate, and token identity with the lane off. None of them checks the new
   arithmetic at kernel level on inputs built to break it, and the fidelity gate's real prompts put
   many heads' max on the first token, where every later rescale is ×1. Add
   `TestAttnFused_vsF16Reference`'s shape for the decode kernel — exact f64 math over the kernel's
   own rounded inputs, per head and never pooled, bars 1e-3 of max|V| and cosine 0.9999 on
   conditioned rows — with inputs that force the mechanism: a sharp hot key in the first split, in
   the LAST split, and on both sides of a split boundary; a rising score ramp, so the running max
   moves in every tile; a sink that dominates and one that does not; a windowed case with
   `winStart` > 0; nKeys straddling the 128-key tile and every split boundary; S ∈ {1, 2, 4, 8}.

---

### R7 · Sampled-decode cliff — device-side bounded top-K

> **R7b — RESULT 2026-09-20 (owner decision, same day): temperature-only sampling by Gumbel-max on every backend.**
> The R7 result below left the OpenAI-default cell (plain `temperature`) at 0.739 because a top-K cannot reproduce an
> inverse-CDF draw in index order. The owner chose to change the *draw* instead: `argmax(logit/T + Gumbel noise)` with a
> Philox counter RNG, one algorithm on every backend, the host implementation the reference. **A disclosed break: the
> distribution is unchanged, the seeded stream is not.** Record: `docs/measurements/sampled-gumbel-2026-09-20.md`.
> Paired vs greedy, 0.5B: **CUDA 0.744 → 1.008 (n=15); WebGPU 0.796 → 1.035 (n=12)**. Device kernels agree with the host on
> 15,840 (CUDA) and 12,000 (WebGPU) draws; device/host streams are identical on every real-model run. **Open:** Metal
> device kernel (Mac; handoff note written), a Mac/peer re-baseline of any unfiltered-sampling cell, the Ollama peer sweep.

> **R7b Mac — RESULT 2026-09-20: Metal device kernel built and gated (`decoder.ResidentSample`).**
> `metal/gumbel.go` (`gumbel_stage1`/`gumbel_stage2` MSL kernels, ported from `cuda/gumbel.cu`/`gpu/gumbel.go`) +
> `metal/gumbel_sample.go` + `metal/backend.go` (the `*metalResident` delegation every other
> `decoder.ResidentForward` method already needed — easy to miss since `*resident`'s own methods don't
> promote through the wrapper's named field). Kernel-vs-host: **0 mismatches in 15,840 draws**, matching
> CUDA's own figure exactly. End-to-end device-vs-host token streams: **12,000/12,000 tokens identical**
> (qwen2.5-coder 0.5B + 1.5B × T∈{1.0,0.7,1.3}, 1,000 tokens each), device sampler engaged on every token
> (never fell back). Mutation check (a Philox constant flipped) goes red (189/300 mismatches, worst
> host-key gap 11 — confirms the gate can catch a broken kernel, not just pass vacuously). **One real bug
> found and fixed during development**, worth recording since the isolation method is reusable: the
> small-w noise branch was missing a negation (`e = -log1p(-w)`, coded as `e = log1p(-w)`), giving NaN
> keys for roughly half the `uint32` range — invisible on every case where the true winner's LOGIT margin
> dominated the (garbage) noise, but 100% wrong on a flat-logits row where noise is the only
> differentiator; found by isolating Philox (checked clean against the Random123 vectors first) and the
> noise transform (checked against the host's own f64 values next) before ever touching the full
> two-stage reduction. MSL has neither `log1p` (absent from the spec's own accuracy tables — checked, not
> assumed) nor statement-boundary immunity from FMA fusion (the spec documents `fast` contraction as
> fusing *across* statements, not just within one expression) — handled with a ported 12-term polynomial
> and a `#pragma METAL fp contract(off)` bracket respectively, both cited against the spec directly in
> `metal/gumbel.go`'s header. Record: `docs/measurements/r7b-metal-mac-2026-09-20.md`.
>
> **Speed, same day, follow-up pass:** paired vs greedy, interleaved with a rotating arm order, 0.5B,
> same session (`metal/sampled_gumbel_speed_test.go`, no committed CUDA harness to port — built
> to the protocol description). Do-nothing arm (host draw) **0.80-0.84× greedy** (3 runs); device
> draw **0.96×** (3 runs, tight) — a real ~13-21% win over the do-nothing arm, directionally
> consistent with CUDA (0.744→1.008) and WebGPU (0.796→1.035) but not full greedy parity the way
> either of those reach (Metal's UMA has no PCIe/MapAsync readback to eliminate, so this path's win
> is from replacing the host's own full-vocab loop, not from a transfer saving — an inference, not
> measured directly).
>
> **Mac/peer sweep, same day, second follow-up:** `scripts/bench_peer.py`, fixed to give Phase C a
> Metal-backend override (`BENCH_SAMPLED_BACKEND`, mirroring `BENCH_DEPTH_BACKEND` — Phase C was
> hard-coded to `cuda`, which is *why* no Mac sampled peer cell had ever existed, not just that
> nobody had run one). goinfer vs Ollama v0.32.5, both Metal, same weights verified per-tensor:
> **greedy goinfer 1.15× Ollama; temp-only goinfer 1.10× Ollama** — goinfer ahead on both. goinfer's
> own sampled/greedy ratio through this independent instrument (0.974×) confirms the internal
> harness's 0.96× via a different measurement path. `BENCH_MAX_LOADAVG` raised 1.0→3.0, disclosed
> (this box's ambient load with the measuring session itself running never reaches 1.0). Full
> numbers: `docs/benchmarks.md` §B3 addendum, `docs/measurements/r7b-metal-mac-2026-09-20.md` §4.
> **Open:** a profile of the remaining ~2.6% gap between goinfer's own sampled and greedy Metal
> rates.

> **RESULT 2026-09-20 — CUDA `top_k` / `top_p` / `min_p` SHIP; temperature-only does not, and the brief's
> design could not have served it.** Record: `docs/measurements/sampled-topk-2026-09-20.md` (step 0:
> `sampled-topk-baseline-2026-09-20.md`). Paired sampled ÷ greedy on the 0.5B, same session, n=15:
> **top_p 0.95: 0.657 → 0.957 (SHIPS, bar 0.90); top_k 40: 0.961; min_p 0.05: 0.971; T=1.0 temperature-only:
> 0.739, unchanged.** Token streams are identical to the full-row path (3 checkpoints × 4 configs, 12 of 12;
> 102,336 sampler draws, 0 mismatches; kernel exact on 272 crafted rows), with a per-token fallback
> (0–0.5%). **What the brief got wrong, recorded at the same value as the win:** (1) it proposed serving the
> temperature-only path from the returned K ("normalise over the returned K with the tail mass bounded") — the
> tail bound was already refuted by P2b, and that path's draw is an inverse CDF in vocabulary *index* order
> that a top-K cannot reproduce, so the headline `T=1.0` cell (0.74) is untouched; (2) it read P3 as
> banked for good reason, but a post-P2b attribution shows host sampling is still ~85% of the gap
> (0.9–1.3 ms/token) and the readback only ~0.1–0.2 ms, so the banking premise was measured false for
> filtered samplers. **Open:** a temperature-only sampler needs an owner decision on a non-stream-identical
> device sampler; Metal and WebGPU are not done (the Mac matrix's sampled row is still empty); the peer sweep
> is not re-run, so `benchmarks.md` §B5.1's peer ratios are stale for the filtered cells (noted there).

**Goal.** Make `temperature > 0` cost what it costs the peer: nothing measurable.

**Standing and the registered band.** CUDA, depth 128 (§B5.1 / §B8): 0.5B greedy 333–342 tok/s,
`temperature 1.0` 237.7, `0.8 + top_p 0.95` 227.2 (Ollama 268.7 / 268.7 / 265.2); gemma3-1b 170.7
greedy → 143.7 / 124.7; phi3-mini 124.9 → 109.8 / 102.1. **Band (CUDA 0.5B, sampled ÷ greedy on the
same binary, same session): ≥0.90 ships (today 0.70), 0.80–0.90 parked, below 0.80 killed.**
Token-identity is a hard gate, not a band: with a fixed seed, the device-top-K path emits the
identical stream to the full-readback path on every token where K suffices, and the fallback count
per 1,000 tokens is recorded (expected: rare; a fallback rate above 1% at `top_p 0.95` means K is
too small, not that the design is wrong).

**Read first.** `ollama-chase.md` §D6 in full (the two cliffs — the host sort, fixed 68×, and the
readback branch, open; the design sketch with the host-verified nucleus; and "the second, larger
cost", full-vocabulary normalisation on the temperature-only path), `decoder/sampler.go`
(`ArgmaxEquivalent`, `decoder/sampler.go:222`; `sampleChunked`; the bounded selection), `cuda/argmax.cu`
(the on-device argmax the greedy path uses — the top-K reduction is its sibling), G26/G28 in
`docs/QUEUE.md` (why the sampled cells must use a realistic prompt and why the anchor's own spread
was half the reported delta), `benchmarks.md` Methodology (count tokens from `usage`; early EOS at
`temperature 1.0`).

**Build.** CUDA first: a `topk_reduce` kernel returning K (256 default, 1024 for the 262k-vocab
families) logits + ids, launched in place of the logits D2H when `!ArgmaxEquivalent()`; host side,
the existing exact sampler runs over the K entries with the mass check (Σ softmax over the returned
K ≥ top_p, and the K-th logit below the nucleus edge), falling back to the full V readback for that
token otherwise. Temperature-only path: normalise over the returned K with the tail mass bounded
the same way (this is the "second cost"); `top_k=1` already routes to greedy (P1). Metal second:
the readback is zero-copy, so the win is host-side selection over K instead of V — the same host
code, a `topk_reduce` MSL kernel, measured rather than assumed (UMA has already turned one "obvious"
readback win into a wash — the fused-argmax record). WebGPU third, same shape (`MapAsync` K entries
instead of 608 KB), only where the readback crosses PCIe.

**Gates.** Stream identity vs the full path at fixed seed, three sampling configs × three models,
1,000 tokens each, plus the fallback counter; `TestKernelFMALint` (the reduction is a max/compare
tree, FMA-free by construction); the existing sampler unit tests unchanged.

**Measure.** `scripts/bench_peer.py` with the §B5.1 sampled configurations, greedy as the control in
the same run, n=15 per cell (G26's lesson: the anchor's own spread at n=2 was the size of the
effect), a realistic prose prompt named in the record (not `prompts.json`'s four-word filler — G28).

**Decision rule.** As registered, on the CUDA 0.5B cell; Metal and WebGPU carry their own
measured bands after the CUDA result exists.

**Record.** `docs/measurements/sampled-topk-2026-MM-DD.md`; `benchmarks.md` §B5.1 re-anchored
(sampled rows are the headline of this item); `ollama-chase.md` §D6 status; the Mac matrix gains a
sampled row for the first time.

**Out of scope.** Constrained decode's O(V) masker (a separate item in `ollama-chase.md` §13),
logprobs (`Logprobs` keeps the full path by definition).

---

### R8 · CUDA vision tower onto the L2/L3 kernels

**Goal.** Move the resident SigLIP tower from the pre-L2/L3 kernel shapes to the ones text prefill
now ships, and delete the host round-trips.

**Standing and the registered band.** 26.1 s/image resident vs 41.3 s CPU int8 same session
(1.58×), real checkpoint `gemma-3-4b-it`, 896², 4096 patches, 27 layers (`benchmarks.md` §A
"Resident CUDA vision tower"); the CPU f32 tower 31.3 s. **Band: ≤8 s/image ships (≥3.3×, counted
from the text twin's 4.5× gemv / 3.76× attention at a matching M); 8–13 s parked; above 13 s
killed** (below 2× means the tower's cost is not where the text-prefill analogy says it is —
profile before anything else).

**Read first.** `cuda/vision_encoder.go` top to bottom — the header's design note ("no new GEMM
kernel", `bGemv` = `gemv_w8a8_batched`, `attn_img_batched` gridded (nH, M), `addInPlaceHost` with
its own "natural next optimisation" admission), `cuda/attn_img_prefill.cu`, `cuda/gemm_w4a8_mma.cu`
(the L3 kernel and its header on why group scales do not preclude `mma.sync`), `cuda/attn_fused.cu`
and R5 (the tile this brief borrows — hd=72 pads to 80), `docs/multimodal.md` P6 (the design
write-up and the cosine 0.910–0.96 real-checkpoint gate, investigated and explained as accumulated
per-layer rounding, with the row-0 broadcast bug it caught), aikit's `gpu` module ViT implementation (`cuda_vit` and its
`w8a8reg` / `attntiled` tests) (the same tower shapes, on the aikit side — read so the two do not
diverge into two towers with two sets of numbers; goinfer's is the one this brief moves).

**Build.** (1) `gemm_w8a8_mma`: the int8 twin of L3 — `mma.sync m8n8k16.s8` int8×int8→int32 with
per-row weight scale and per-row activation scale folded once per row (simpler than the int4
group-scaled kernel: no per-group float fold), used for every projection in the tower at M=4096 and
available to the text int8int8 batched path afterwards (a second consumer; measure separately, do
not fold it in). (2) A non-causal `attn_fused` variant for the tower (full bidirectional at
imgStart=0/imgEnd=M is the whole image — no mask needed beyond the tile bounds), hd=72 padded to 80
with zeroed lanes, 16 heads. (3) A device-side elementwise add (`cuda/addone.cu` is the shape;
one kernel, three call sites: posEmb, two residuals per layer) — 54 launches replacing 54
download-add-upload trips. (4) `layernorm_f32_batched` already exists; keep it.

**Gates.** `TestGemma3VisionResidentReal_gate` (matched int8 both arms) not below the current run's
range on the same input pattern (0.91–0.96), plus per-layer cosine vs the CPU int8 reference so a
regression is localised to a layer, plus the downstream `GenerateVL` first-token identity check the
P6 record used; the kernel gates from R5 for the attention tile at the padded hd.

**Measure.** The P6 timing driver (interleaved-paired, warm-up discarded, median of 5) on the real
checkpoint — commit it this time under `cuda/` with a `GOINFER_HEAVY_TESTS` gate rather than leaving
it as a throwaway; report tower seconds and the `BENCH_VISION=1` served TTFT for one image against
Ollama's `gemma3:4b` (the first vision peer row on the page — it is a new instrument, not a
re-measurement of the tower figure, and the record must say both).

**Decision rule.** As registered, on tower seconds.

**Record.** `docs/measurements/vision-tower-mma-2026-MM-DD.md`; `benchmarks.md` §A vision rows and
Table 1's multimodal cell; `docs/multimodal.md` P6.

**Out of scope.** Qwen2.5-VL's tower (different attention pattern; its own port after), Metal
(M-15 — the aikit shapes first; R8's kernels are the design the Metal port copies), WebGPU's stale
18.8 s (re-measure in R10).

---

### R9 · CPU decode attribution — the Mac fixed cost, the Linux 0.5B anomaly — then S-05

**Goal.** Attribute the per-token cost the fits in §2.1 say exists, on both boxes, before building
anything; then fund the one counted kernel lever.

**Standing and the registered band.** Mac (2026-09-17): 100.1 / 49.5 / 17.2 tok/s at 0.5B/1.5B/7B
vs 148.5 / 75.9 / 20.6; the fit: ~4.5 ms fixed + 62–79 GB/s vs ~2.4 ms + 92–95. Linux (2026-08-26):
23.5 / 17.6 / 4.9 vs 57.9 / 24.2 / 6.0; the fit: ~13 ms fixed + 22 GB/s vs ~5 ms + 27, and the 0.5B
~14 ms/token below its own fit. **Step 1 (attribution) has no band; its output is a table.** **Step
2 band (Mac 1.5B served, greedy, depth 128): ≥60 tok/s ships (1.21×), 54–60 parked, below 54
killed. Linux 0.5B: ≥40 tok/s ships (1.7×), 30–40 parked, below 30 killed.**

**Read first.** aikit `docs/task-simd-audit.md` (BLUF: kernels at 91–97% of issue ceiling; the four
gaps around them; S-02's per-shard timestamp harness as "step 0 is now one item"; S-05 counted
1.33×; S-09.1's fork/join 1.70× and the 2.7× it leaves unexplained), aikit `docs/audit-2026-09-10.md`
§6's order, `completed/task-w4a8-neon-bandwidth.md` (the STREAM ceilings — 71.9 GB/s one thread,
121 at six on the M1 Pro; the LM-head diagnosis that the old Amdahl model got wrong; "bit-identity
hides dispatch inertness — the performance delta IS the wiring test"), `completed/task-attention-decode-cost.md`,
`completed/task-prefill-gap.md` §4 L4 (the M-block outer loop, S-06 step 1's measured 1.17× prefill
/ flat decode), `benchmarks.md` §A and the 2026-09-17 CPU note (the ratio moved in opposite
directions per size between 08-24 and 09-17 — flagged, not explained; step 1 explains it or says
why not).

**Step 1 — attribute, both boxes.** Build S-02: per-worker timestamps around every fan-out in the
decode token (matmul shards, attention heads, the LM head, the quantiser, sampler) written to a ring
buffer and summarised per token as: critical-path time, max/min worker imbalance, fork/join gap,
serial residue. Run it on 0.5B / 1.5B / 7B on each box (Mac: `maxAttnWorkers = 6`; Linux:
`GOMAXPROCS` 16 on 8 cores — test 8 and 16 explicitly, SMT siblings competing for the same
load ports is the first hypothesis for the Linux fixed cost, and the 0.5B's K=896 LM head is the
second). Output: a per-component ms/token table per (box, size), with the fixed cost and the
stream rate the fit predicted beside what was measured. The Linux 0.5B anomaly is resolved when a
component accounts for ≥10 of the ~14 ms, or the table says it is spread and names the spread.

**Step 2 — build what step 1 names, in this order unless it says otherwise.** S-05 in aikit (the
SDOT centering fold, bit-identical, the raw-bit gate the S-01 tiles used, `perfgate` exception
recorded); then the fan-out shape change step 1 points at (fewer, larger shards; a per-token
worker pool that does not re-fork per matmul; the quantiser and softmax off the calling goroutine —
S-03/S-06 are built and gate-checked in aikit, unmeasured end to end on the Mac); on Linux, `GOAMD64`
pinned to v3 if step 1 shows compiler-generated scalar code on a hot path (the audit noted it
unpinned).

**Gates.** Bit-identity throughout (the CPU decode is the reference everything else is gated
against): `TestParityManifest_fresh`, the full `decoder` suite, aikit's raw-bit gates on any kernel
change; a step-2 change that is not bit-identical is a different brief.

**Measure.** `scripts/bench_peer.py`, `BENCH_BACKENDS=cpu`, `BENCH_ENGINES=goinfer,goinfer_old,ollama`
(the before/after A/B the harness gained 2026-09-12 — record the old binary's commit in its path),
`num_gpu:0` on the peer, both boxes, the Mac under its loadavg qualifier with recorded reps, greedy,
depth 128; the 08-24 vs 09-17 opposite-direction move is re-measured under matched conditions as
part of the same run.

**Decision rule.** Step 2 as registered; step 1 is done when the table exists and is cited.

**Record.** `docs/measurements/cpu-decode-attribution-2026-MM-DD.md` (both boxes, one file);
`benchmarks.md` §A CPU decode row and the §B8 backend table's CPU column; aikit's internal
CPU-acceleration notes for the kernel side (cross-repo and gitignored there too, so described here
rather than path-cited).

**Out of scope.** Prefill (S-05 helps it too — measure, do not claim), the 26B on CPU (D5's
question, R11), vision on CPU (the S-06 transcendentals: wire them under this brief only if step 1
puts the tower's exp above 10% of an image on the CPU path — otherwise R8's GPU path is the answer).

---

### R10 · WebGPU — glue fusion and on-device argmax; the batched-prefill profile

**Goal.** Close the gap between WebGPU's token and CUDA's on the same card where WGSL allows, and
find out where the batched prefill's time goes before scoping a build for it.

**Standing and the registered band.** 1.5B int4 137.6 tok/s (§B10, 2026-09-02) vs CUDA 227; per
G35's ablation the non-GEMV remainder is ~4.25 ms of a 7.27 ms token. Batched prefill: 0.5B int8
P=1024 in 3.23 s (`prefill-batched-ttft-2026-09-13.md`). **Band (decode, 1.5B int4, `gpu/cmd/serve`
server-to-server greedy, the G36 protocol): ≥170 tok/s ships (1.24×), 150–170 parked, below 150
killed. Prefill: no band until the profile exists — register one in the profile's write-up.**

**Read first.** `docs/QUEUE.md` G35 and G36 in full (the ablation harness `TestDecode_dispatchProfile`,
the "13 → 3 was never a WGSL-side possibility" table — K1 yes, K2 no, K3 bandwidth-fatal; the lesson
that a transferred diagnosis transfers its assumptions), `docs/decode-fusion-next.md`,
`docs/completed/gpu-next-levers-assessment.md` (the glue-serialisation wall; the `dot4I8Packed`
story — the binding moved to `oliverbestmann/webgpu` 2026-09-12 and still has no `dot4` symbol,
re-checked 2026-09-13), `gpu/gemv_w4a8.go`, `gpu/device.go` (`quantizeShaderWGSL`), `gpu/prefillrunner.go`
(`attnBatchedShaderWGSL`, `attnKeysBatchedShaderWGSL`, the batched rmsnorm/rope), `gpu/attention.go`
(the G36 key-split kernel with its 2048-key online-softmax tiles), `ollama-chase.md` §13's WebGPU
argmax caveat (on UMA a wash; on PCIe a readback).

**Decode build.** K1: fold `rmsQuant` into the QKV GEMV's prologue (redundant recompute in
`var<workgroup>` per G35's table) and `quantize`/`swigluQuant` into the producing kernels'
epilogues where the consumer is a single GEMV; `gemvBias` into the GEMV epilogue; on-device argmax
for the greedy path with a K-entry `MapAsync` (R7's shape) so the 608 KB readback goes. Bit-identical
by construction for the fusions (same arithmetic, moved), gated by the existing parity suite and
`TestAttnKeys_parity` untouched; ablation profile before and after so each class's delta is
attributable.

**Prefill investigation.** Extend the ablation harness to the batched path at P ∈ {256, 1024}:
per-class time with one class omitted, so the GEMM, the attention, the batched norms/rope and the
KV write each carry a number. Then one arithmetic line: GFLOPS achieved by the GEMM class against
the card's naive-f32 and its shared-memory-tiled expectations. The write-up registers the prefill
band and names the kernel; the build is a separate step.

**Measure.** `gpu/cmd/serve` server-to-server at 128 / 512 / 1024 / 3900 (the last is new for
WebGPU — the depth term past 1024 has never been measured), 1.5B int4 and 0.5B int4 (the 0.5B is
where fixed glue dominates: 127.6 vs CUDA's 342), plus `TestResidentPrefillLast_TTFT` for the
batched path.

**Decision rule.** Decode as registered. Prefill: the profile is done when every class has a
number; a build follows only against a registered band.

**Record.** `docs/measurements/webgpu-glue-2026-MM-DD.md`, `webgpu-prefill-profile-…`;
`benchmarks.md` §B8 backend table (the WebGPU column with a date) and §B10; `docs/decode-fusion-next.md`
status; the stale 18.8 s vision figure in Table 1 re-measured or struck in the same pass.

**Out of scope.** `dot4I8Packed` (blocked upstream; re-check the symbol, do not build around its
absence), MoE paging on WebGPU, a WebGPU peer (there is none — every WebGPU number is
cross-backend and the page must keep saying so).

---

### R11 · MoE — L01's funding cell, P20's expert-major prefill, the Metal pager

**Goal.** Move the three MoE-over-capacity cells that have a named mechanism and no measurement:
CUDA decode (L01), CUDA prefill (P20), Metal decode (M-11/M-13).

**Standing and the registered band.** CUDA @128 (2026-09-17): M35 23.7 vs Ollama 23.8 / llama.cpp
32.8, M26 23.8 vs 22.2 / 27.1; @8000 wall-clock M35 1490 s vs 86 / 62, M26 368 vs 85 / 67. Metal:
35B ~2 tok/s (1.97–2.19 across two harnesses), pager auto-sized 2026-09-13, unmeasured.
**Bands: (a) L01 — the pre-registered decision rule in `task-l01-hybrid-moe-cpu-gpu.md` §0 governs
and is not restated here; (b) P20 on M26 — sequential 46.5 ms/token at M=512 → ≤25 ships (the
"~20" projection the doc refuses to quote as a result), 25–35 parked, above 35 killed; M35 has no
band until batched GDN exists; (c) Metal pager — measurement only: the M35/M26/G20 rows re-run
against the auto-sized pager with the fallback provably not engaged; then M-11's shared-event A/B on
the paged shape with G-05's own rule.**

**Read first.** `task-l01-hybrid-moe-cpu-gpu.md` in full (the corrected header — 272 µs/expert, the
parallel-misses result, the synchronous prototype, §9's remainder), `task-moe-streaming.md` §C′
(async H2D already named), `ollama-chase.md` §D5 (the layer split — the peer's mechanism, and why
the CPU half caps it), P20 in `queue-performance.md` (the M26 guard log, the flat-with-depth finding,
route (a)), `task-gpu-paths-2026-09.md`'s 2026-09-09 note on MoE batched prefill ("M sequential
rows … a layer-major prefill loop, chunked over M … a real dispatch-restructuring project"), audit
`M-05`, `M-11`, `M-12`, `M-13`, `M-14`, `G-05`, `G-09`, `benchmarks.md` "M35/M26 on the Mac — parked"
(the swap/kernel-panic incident: the fallback path is off-limits on this Mac; the pager is not the
fallback, but prove which one ran before a cell counts — `DecodePath()` and the load banner), the
2026-09-15/16 ctx-fit regression (`18a6f73d`: the expert cache is elastic in leftover VRAM; a cell
with a different `-ctx` is a different cell).

**Build, in three independent parts.** (a) L01 async: overlap the CPU misses with the GPU's own
per-layer compute (a per-layer goroutine started at routing readback, joined before `moeMLPPost`'s
merge), then run the funding cell as §0 registers it — paired and interleaved against C′ on
Qwen3.6-35B-A3B at `-ctx 4096`. (b) P20 expert-major on CUDA: route the chunk once, group rows by
expert, stage each distinct expert once per chunk, run its rows with a batched-K GEMM — the shape
P18 shipped on the CPU and the Metal `GOINFER_MOE_EXPERT_MAJOR` path (2026-09-17) implements
(read it; the routing semantics — groupLimit's top-2-per-group, top-k masking — are already matched
there); bit-identical to the sequential FFN by construction (same per-row expert math, reordered
across rows, no accumulation shared across rows — verify with the existing `TestMoEExpertMajor_bitIdentical`
pattern). (c) Metal: nothing to build for the measurement; M-11 afterwards is one command buffer per
token with a shared event between the pread stages and the GPU, measured on the paged shape at
the ~14 ms boundary, G-05's rule.

**Gates.** L01: `TestL01_e2eDecode_matchesBaseline` (cosine 1.0, argmax at every position) with
overlap on; the coherence floor the B4 gate uses (distinct-trigram ≥0.70). P20: bit-identity to
sequential on `testdata/qwen35-tiny`-class fixtures plus one real-checkpoint stream comparison. Metal:
paged ≡ stacked byte-identity (the `idxZeros`/`rIdx` trick) unchanged.

**Measure.** CUDA: `scripts/bench_peer.py` M35/M26 at 128 and 8000 with `BENCH_DEEP_CTX` scoped to the
8000 cells only, `-moe-cache-experts`, unpinned ctx (the fix keeps 4096); the W3 wall-clock column
is the headline for (b), tok/s for (a). Metal: the pager cells with `models-pull` for the
checkpoints, RSS and swap watched live, killed at the first sign of the fallback; `TestQwen35…`-class
in-process rate as the record if the served cell cannot be made safe — say which.

**Decision rule.** (a) per §0; (b) as registered; (c) measurement, then G-05's rule for M-11.

**Record.** L01: its own doc's §9 and `benchmarks.md` peer matrix M35/M26; P20: `queue-performance.md`
P20 closed or redirected, `benchmarks.md` W3 wall-clock column re-run; Metal: `benchmarks.md`
"M35/M26 on the Mac" replaced by a pager row with the fallback-not-engaged evidence in the
provenance; audit M-11/M-13 closure notes.

**Out of scope.** D5's layer split (right shape, capped by the CPU half — revisit after R9), the
26B's KV/slot arithmetic (settled in §B4.2), cpubrrr Q4_K (R9 decides whether the CPU path
matters again).

---

### R12 · Metal `ForwardN` batching, and the peer rows the page is missing

**Goal.** Make speculation pay on the Mac, and fill the four measurement gaps that keep recurring
as "unmeasured" in this doc: the MLX row, the Metal peer depth row, a vision peer, and W7.

**Standing and the registered band.** Metal Θ = 0.96 (2026-09-17 sweep, 20 cells; speculation
declines), CUDA 0.251, CPU 0.5, WebGPU default 0.5 unmeasured (P22). **Band: after batching, the
re-measured Metal Θ ≤ 0.6 AND a `TestSpecDecodeCurve`-shaped Metal curve ≥1.15× at depth 512 on the
D1 prompt set ships; Θ in 0.6–0.8 parked; above 0.8 killed (batching did not move the marginal
verify cost, and the Metal small-M GEMM is the reason — record it beside P10's finding).**

**Read first.** P21 and P22 in `queue-performance.md`, `docs/measurements/theta-per-backend-2026-09-01.md`,
`docs/spec/00-core.md` and `10-optfwd-gate.md` (the lossless contract and the prompt-form caveat),
`completed/task-metal-batched-verify-kernel.md` and `completed/metal-batched-verify.md` (the small-M
verify kernel that measured ~1.13× and was not adopted — P21 is about the command-buffer boundary,
not that kernel), `metal/backend.go:702` (`ForwardN` today), the 2026-09-17 note on `VerifyPathReporter`
(`decoder/residency.go` — the interface that now reports whether the verify is batched; wire it
truthfully), `task-peer-benchmarks.md` (W7's definition; the MLX quant caveat), `scripts/bench_peer.py`
(the `mlx` engine branch; `BENCH_VISION=1`; `scripts/bench_peer_transcript.py` for W4/W7).

**Build (P21).** `ForwardN` on Metal encodes the K verify tokens into one command buffer — the
`PrefillLast` machinery already assembles all layers into one buffer for M>1; the verify is M=K≤16
appended at `startPos`, which is the shape `PrefillLast` with prefix reuse already serves — so the
build is the dispatch (route `ForwardN` through the batched prefill path when K ≥ 2 and the
family's `prefillOK` holds; fall back to the loop otherwise), plus the interface contract in
`decoder/residency.go` finally being true for Metal. Bit-identity: the batched path is the *fast*
lane (f16 activations) — the lossless contract in `00-core.md` requires the verify to match decode
exactly; so `ForwardN` may batch only when the verify oracle is the same numerics as decode. That
is the real question this brief answers first: if the fast lane cannot be the verify oracle (P10's
conclusion), the batching must be of the *exact* per-token kernels into one command buffer — which
removes the K−1 commit/wait boundaries and nothing else, and is the arm to measure.

**Measurement briefs (each one session, no code beyond the harness).** (i) MLX into every Mac
dense cell: `BENCH_ENGINES=goinfer,ollama,mlx`, the quant caveat in every row, 0.5B/1.5B/7B/phi3.
(ii) The Metal peer depth row (R2 step 0 — one run serves both). (iii) A vision peer row:
`BENCH_VISION=1` against `gemma3:4b`, one image, served TTFT both sides, stated as a new instrument
beside the tower figure. (iv) W7: four interleaved conversations from the W4 transcript on both
boxes, goinfer's single worker vs llama-server's slots — the first concurrency number on the page,
expected to be a loss and recorded at full value; it sizes the batched multi-request decode item
that `task-work-queue-2026-09.md` J8 names.

**Gates.** `TestSpeculativeGreedyParity`-class exactness for any batched verify; `TestEncodeAhead`
still green.

**Decision rule.** P21 as registered; the measurement briefs are done when the rows exist.

**Record.** `theta-per-backend-…` successor file; `benchmarks.md` peer matrix (MLX column filled,
a vision row, a W7 row), §B3 (depth row); P21/P22 closed; `docs/spec/` gets a one-line status
line pointing here.

**Measurement briefs, progress 2026-09-18/19.** (ii) is done — R2 step 0 was the same run, per this
brief's own note. (i) is partly done:
[`r12-mlx-row-2026-09-18.md`](../measurements/r12-mlx-row-2026-09-18.md) re-confirms the existing
`benchmarks.md` peer-matrix row (goinfer/Ollama/MLX, 1.5B and 7B, all deltas inside ~3.5% ordinary
session drift — not a revision) but does not extend it to 0.5B or phi3-mini, since no MLX-format
checkpoint for either is cached locally; that needs an explicit download decision, not made here.
(iii) the vision peer row and (iv) W7 were not attempted. P21 (the actual build) has not started.

**(iv) W7 result, 2026-09-19** ([`w7-plain-concurrency-2026-09-19.md`](../measurements/w7-plain-concurrency-2026-09-19.md)).
A simplified variant, not the exact W4 tool-calling transcript — the memory-safe 1.5B model does
not reliably satisfy llama-server's tool-call parser (verified directly, including with the real
Qwen tool template extracted from the 7B checkpoint's own GGUF metadata) and the 7B model that
does carries this session's own demonstrated memory risk. Ran six plain multi-turn text questions
instead (`scripts/bench_w7_plain.py`, new). **goinfer 60.1 → 36.1 → 36.4 tok/s at 1/2/4 clients — a
real loss that plateaus rather than keeps collapsing; llama-server 84.8 → 95.9 → 149.7 — genuinely
scales with concurrency (1.76× from 1→4). 4.11× gap at n=4.** Per-client latency shape is
consistent with goinfer FIFO-serializing one request at a time against llama-server's continuous
batching giving every client near-identical service time. Sizes the brief's own "batched
multi-request decode item" (`task-work-queue-2026-09.md` J8) as real and substantial, not sized to
the tool-calling shape specifically. A clean re-run with the exact W4 transcript remains open if a
memory-safe, genuinely tool-call-capable model becomes available.

**Out of scope.** MTP heads (`spec/09`), DFlash (P15), the drafter zoo — this brief is the
mechanism, not the drafter.

---

### R13 · CPU decode attention, group-major — the acc64 kernels stop paying per query head (bit-identical)

**Goal.** Take the CPU decode attention depth term down by sharing each K/V row's load-and-widen
across the nH/nKV query heads that read it, without moving one output bit — the only depth lever on
any backend that needs no fidelity lane.

**Standing and the registered band.** There is no standing, and that is the first finding: CPU has
no peer row at depth on either box (§1), and the only depth curve on record is the isolated
attention bench in `completed/task-attention-decode-cost.md` (2026-08-23, 1.5B shape, 28 layers,
M1 Pro): 2.81 / 7.80 / 19.65 / 85.23 ms per token at 128 / 512 / 2048 / 8192 — about 10 µs per
position, against §2.2's 3.2 on Metal and 0.92 on CUDA. That table predates S-04's NEON and AVX2
ports, so it is a stale "before" and step 0 replaces it. What the code does today is not in doubt:
`causalAttention` hardcodes `acc64`, `attendBatchedHeads` fans out over QUERY heads, and each head
calls `linalg.MatmulQKAcc64` and `linalg.MatmulAVAcc64` on its own — so each KV head's f32 K and V
rows are loaded and widened to f64 nH/nKV times per layer (6× on the 1.5B, 7× on the 0.5B and 7B):
344 KB issued per position per token on the 1.5B, against 57 KB if the group shared them. **Whether
that is paid in bytes is NOT established.** `completed/task-w4a8-neon-bandwidth.md` measured this
box's STREAM ceiling at 121 GB/s on six threads and the August table works out to ~33 GB/s of as-if
traffic, well under it; and §2.2's reading 3 is the standing warning that a cache can dedup a
group's reads in hardware. So the band is registered on µops, which are paid either way. From the
shipped kernels' own instruction accounting (aikit's `linalg/attn_acc64_arm64.s` header): QK costs
~1.53 SIMD µops per MAC, of which the two loads, two ZIPs and four widens per key pair are K-side
work identical for every head of the group; AV costs ~1.09, of which the loads and the sixteen
widens per 32 dims are V-side. Shared across G heads that is 0.73 + 0.73 at G=6 — **1.80× fewer
µops per MAC (1.85× at G=7; 1.31× for a two-head sub-group).** **Band (QK+AV category time at depth
3900, 1.5B shape, Mac, three arms interleaved, min-of-batches): ships at ≥1.5× AND the flat term F
of `BenchmarkDecodeAtDepth`'s `t(K) = F + A·K` within 3% AND phi3-mini (G=1, the control) within
±2%; parked at 1.2–1.5×; killed below 1.2×.** The slope band and the served band are registered
after step 0, by Amdahl on the category split step 0 measures — they cannot be registered now,
because the split is the thing nobody has (see step 0 (ii) on the softmax).

**Read first.** `completed/task-attention-decode-cost.md`, all of it — the invariant enumerated
(what acc64 buys: decode ≡ verify ≡ exact prefill for speculative decoding, MoE router stability),
moves (a)/(b)/(c) and why each is bit-identical, the depth-aware fan-out argument, and its A2/A3
disposition ("closed for now, revisit if long-context work wants more" — this brief is that
revisit, and it takes the rung *below* A2: still no numerics change).
`task-decode-splitkv-attention.md` for the principle every move here leans on: split the
independent axes, never the reduction. aikit's `linalg/matmul_qk_acc64.go`,
`linalg/matmul_av_acc64.go`, `linalg/attn_acc64_arm64.s` and `linalg/attn_acc64_amd64.s` (the Go
kernels are the definition and the oracle; the identity argument and the register use are in the
`.s` headers) and aikit's `docs/task-simd-audit.md` S-04. `ollama-chase.md` §A2-Metal for what
happened when the same dedup was tried where co-locating a group costs occupancy: those four losses
are about threadgroups and dispatches, neither of which a CPU has, which is why this is not a fifth
attempt at the same thing. §2.2's 2026-09-19 note for the per-query-head regularity and its warning
about reading it as bytes.

**Step 0 — measure; no build, fundable now.** (i) The peer depth row, both boxes:
`scripts/bench_peer.py` with `BENCH_DEPTH_BACKEND=cpu`, depths {128, 512, 2048, 3900}, greedy, the
1.5B and 0.5B int4 plus phi3-mini as the MHA control — the first CPU depth cells `benchmarks.md`
will have. (ii) `BenchmarkDecodeAtDepth` on the current tree at {128, 512, 2048, 3900, 8192}, fitted
to `t(K) = F + A·K`, with the slope split by category — QK, softmax, AV. Gate A0 found the softmax
"noise", but that was against QK/AV kernels roughly ten times slower than today's; the acc64 softmax
is still one f64 `math.Exp` per key per head, and if it is now a third of the slope it caps what
this brief can buy and becomes the next item. (iii) The distinct-bytes probe, on the kernel A/B
bench: the real GQA layout (nKV heads shared by nH query heads) against an MHA-expanded layout with
nH distinct KV heads — identical issued work, nH/nKV× the distinct bytes. If the two read the same,
the caches are not deduping the group, every query head is paying DRAM for its reads, and the top
of the band is live; if GQA is markedly faster, the hardware already dedups and the gain is the µop
sharing alone. It is the CPU's version of the Metal collapse probe and CUDA's ncu traffic ratio, and
it says which end of the band to expect before a line of assembly exists. Output: the
pre-registration, with the slope and served bands filled in.

**Step 0(i) result, 2026-09-19** ([`r13-cpu-depth-row-2026-09-19.md`](../measurements/r13-cpu-depth-row-2026-09-19.md)).
Tooling gap closed (`BENCH_DEPTH_BACKEND=cpu` on the Mac, real Ollama peer). **Not yet a
`benchmarks.md`-quality row**: the ~77-minute run found two real mechanisms rather than a clean
curve — (1) repeated deep-context sampling on a small-context model (phi3-mini, 4096 ctx) exhausts
`-kv-sessions`' per-slot KV memory within a few identical-prompt calls (each cold-misses
`bestExtend`'s strict prefix rule and takes a fresh, expensive slot), correctly refused by the fit
guard, not a bug — `phi3-mini K=3900` is out of scope on this machine under the default session
count, not just at that one depth; (2) unexplained throughput drift of up to ~2× on the *same*
cell measured twice ~18 minutes apart (0.5B/K=128: 99.2 → 48.9 tok/s) with loadavg flat throughout
— plausibly thermal/frequency throttling over sustained CPU saturation, not confirmed (no thermal
telemetry captured). Re-run needed (shorter sub-sweeps, cooldown pauses, `powermetrics` alongside)
before this row is fit to enter `benchmarks.md`. (ii) and (iii) not started.

**Step 0(ii) result, 2026-09-19** ([`r13-attn-category-split-2026-09-19.md`](../measurements/r13-attn-category-split-2026-09-19.md)).
Softmax is NOT a minor category — the single largest share on the 0.5B (~39-40% of QK+softmax+AV,
bigger than QK or AV alone) and a stable ~22-26% on the 1.5B, split stable across depth within each
model but sharply different between them (plausibly the `hd` difference — QK/AV scale with `hd`,
softmax's per-key `math.Exp` does not). Applying Amdahl to the brief's own registered QK+AV grouping
ceilings (1.85× at G=7 for 0.5B, 1.80× at G=6 for 1.5B, softmax untouched) projects the achievable
OVERALL decode-attention speedup at only ~1.39× (0.5B) / ~1.54× (1.5B), not 1.8-1.85× — confirming
this brief's own pre-registered worry ("if it is now a third of the slope it caps what this brief
can buy") and putting a number on it. A temporary diagnostic (time.Now() around the acc64 decode
path, gated off by default, verified bit-identical/cosine-1.0 with every correctness gate) produced
this data and was reverted rather than shipped — see the record for why. (iii) still open.

**Step 0(iii) result, 2026-09-19** ([`r13-distinct-bytes-probe-2026-09-19.md`](../measurements/r13-distinct-bytes-probe-2026-09-19.md)).
Depth-dependent, not a single answer: at K∈{128,512} the real GQA layout and an MHA-expanded
control (same issued QKᵀ/scores·V work, `nH/nKV×` the distinct bytes) run within 1-2% of each
other — the cache already dedups the group's reads there, so an explicit-sharing kernel would buy
nothing extra at shallow K. At K∈{2048,3900} — this brief's own decision depths — GQA is 2.1-3.3×
faster (0.5B) / 3.0-4.9× faster (1.5B): past a crossover somewhere in (512, 2048], the working set
stops fitting in cache and the group's redundant reads become real memory traffic. "The top of the
band is live" exactly where the Build's decision cell (K=3900, 1.5B) is registered, reinforcing
rather than competing with the softmax ceiling from (ii). Standalone, permanent benchmark
(`decoder.BenchmarkAttnDistinctBytes`, synthetic data, touches no production file). **Step 0 is now
complete** (i)/(ii)/(iii).

**Build.** aikit first, goinfer second, two arms so the ladder and the stop rule can disagree (§6
rule 1). *Kernels:* `MatmulQKAcc64Group` and `MatmulAVAcc64Group` — Go definitions first (they
become the oracle), then the NEON and AVX2 ports. QK: per block of keys, load and widen each K quad
once, then one FMLA chain per (head, key) against that head's q — every chain still the d-ascending
fold it is today. AV: per key, load and widen the V dims once, then fold into per-(head, dim)
accumulators key-ascending, each head's scores widened once per token into an f64 row rather than
once per pass. The register budget decides the block shapes and is the design problem: on NEON's 32
registers, QK at 8 keys × G=6 is 24 accumulators + 4 widened K + 2 q = 30, and at G=7 it must drop
to 4 keys (18); AV at 8 dims × G=6 is 24 + 4 V + 1 score = 29, and at G=7 (33) it must drop to
4-dim blocks or split the group 4+3. 8-dim AV blocks mean sixteen passes over the keys where
today's 32-dim kernel makes four per head — cheap while a pass's lines stay cache-resident, and
exactly what the bench decides rather than the brief. *Wiring, arm A (cheap):* keep today's fan-out
— contiguous runs of heads per worker — and group only the same-KV-head run a worker already owns
(two heads on the 1.5B); no new barrier, ~1.3× by the same count. *Arm B (full):* fan out over (KV
head × key range) for QK and (KV head × dim range) for AV, all G heads per task, with the per-head
softmax between them as it is today — two more fork-joins per layer, every load shared by the whole
group. Both arms sit behind one gate on the layer's effective attended span `nWin` (the split-KV
lesson: a windowed layer never sees more than its window), below which the per-head path runs
unchanged; `GOINFER_ATTN_GROUPED=0` restores it everywhere. The same kernels serve M>1 (exact
prefill, spec verify), because no output's fold order depends on M — wire decode first, M>1 only
once decode clears the band.

**Gates.** (1) aikit: raw-bit equality of the grouped kernels against the per-head kernels, G ∈
{1, 2, 4, 6, 7, 8}, nKeys straddling every block boundary {1, 7, 8, 9, 15, 16, 17, 127, 128, 129,
4097}, hd ∈ {64, 96, 128, 256}, nonzero `bOff`/`headOff`, plus NEON-vs-Go and AVX2-vs-Go block
tests in the existing style. (2) goinfer: `TestForwardN_matchesSequential`,
`TestSpeculativeGreedyParity`, the ring-parity suite, `TestAttendBatchedHeads_vsNaive`,
`TestAttendStrided_matchesGatherReference` and the parity manifest — **all green with zero golden
changes; a golden that moves is a defect, not a re-base.** (3) `go test -race` on the new fan-out:
arm B's tasks write disjoint score ranges and disjoint ctx dims, and the race detector is the check
that they do. (4) The wiring proof. Bit-identity hides dispatch inertness
(`completed/task-w4a8-neon-bandwidth.md`'s named lesson): a grouped path that never runs passes
every gate above. A test-hook counter of grouped-kernel calls, asserted through `causalAttention`
to be nonzero above the gate and ZERO below it, and printed once in the served bench's startup line.

**Measure.** aikit's kernel A/B bench first (per-head as the do-nothing arm, A's two-head shape, B's
full group) — it nominates. Then `BenchmarkDecodeAtDepth`, three arms interleaved, at {128, 512,
2048, 3900, 8192} on the 1.5B, 0.5B, 7B and phi3-mini, both boxes; the Linux half matters more than
its size suggests, because the 3700X's measured read ceiling is 30.5 GB/s against the Mac's 121.
Then served: `scripts/bench_peer.py` CPU depth cells, paired on/off adjacent with alternating order,
against the step-0 peer row — it elects.

**Decision rule.** The category band above; the slope and served bands from step 0. Arm A ships
alone if B misses and A clears 1.2× with no shallow regression — a small bit-identical win is still
free. A flat-term regression beyond 3% parks either arm regardless of the depth win.

**Build progress, 2026-09-20** ([`r13-neon-kernel-ab-2026-09-20.md`](../measurements/r13-neon-kernel-ab-2026-09-20.md)).
aikit's kernel A/B bench (the Measure section's first step) is done: G=6 NEON ports of both grouped
kernels are gate-1 green (bit-exact against the per-head kernels) and 1.53-2.39× faster than G
separate calls at depth {130,2048,8192} on the Mac — the win the pure-Go checkpoint couldn't show,
now confirmed once real NEON assembly existed, per the brief's own register-budget argument above.
aikit commit `af926e3`, committed locally, not yet pushed/version-bumped into goinfer.

**SHIPPED, 2026-09-20** ([`r13-served-decode-2026-09-20.md`](../measurements/r13-served-decode-2026-09-20.md)).
Three goinfer-side wiring designs (Arm A full-group, Arm B split, Arm B serial — all bit-identical,
gate-1/`-race` green throughout) all measured SLOWER than the unmodified per-head path against a
real checkpoint (`qwen2.5-coder-1.5b`, depth 2048/8192) before the real cause was found. A first
CPU profile pointed at aikit's MLP matmul parallelism as pre-existing decode-wide scheduler
contention — a plausible-looking but WRONG lead (plain CPU pprof miscounts idle parked workers,
matching a prior refuted false trail in this repo's own history). The correct tool (`go tool
trace`'s per-goroutine time breakdown) found the real cause: `attendGroupedLayer` had accidentally
serialized softmax onto one goroutine, where the ungrouped path already runs it 6-way parallel —
a real, local, fully mechanistic bug, not scheduler contention. Fixed by splitting softmax the
same way QK/AV already are. Result: parity at depth 2048 (~1% overhead, noise-level), a real
**1.32× served speedup at depth 8192** (matching the depth-dependence step 0 predicted).
`GOINFER_ATTN_GROUPED` defaults ON. Not yet done: the three-arm/both-box/all-model sweep and the
served `bench_peer.py` measurement against the peer row — this record used the fast diagnostic
only, at one model, two depths, one machine.

**Record.** `docs/measurements/cpu-attn-group-PREREGISTERED.md` before the first cell,
`cpu-attn-group-2026-MM-DD.md` after; `benchmarks.md` §A gains the CPU depth table (peer-paired)
and the TL;DR a CPU-at-depth row; §1's CPU cell here; aikit's S-04 entry (cross-repo: described,
not path-cited); a negative goes in `ollama-chase.md` §10 beside the GPU ones.

**Out of scope.** Any numerics change on CPU decode. The M-independent f32 kernel (A2), and routing
dense decode through the fused f32 schedule prefill already uses by default (A3's decode half), are
the fidelity-lane follow-ons, with a trigger: the served CPU depth row still more than 1.5× behind
at 3900 after this brief. int8-KV decode's own depth term (`causalAttention` dequantises the
layer's whole history into f32 scratch on every token) — real, separate, unmeasured. f16 KV on CPU.
CPU prefill, which is at parity or ahead at depth (§1, §2.3).

---

## 6. Rules every brief inherits

Stated once, so the briefs can cite them by number.

1. **Pre-register.** Every brief with a band commits a `docs/measurements/<name>-PREREGISTERED.md`
   before its first cell: arms, cells, criteria, the ambiguous → parked band, and the do-nothing
   arm. Two things that can disagree (a ladder and a stop rule; a primary metric and a validity
   check) are better than one. A re-run needs a mechanism, never a re-roll.
2. **The do-nothing arm is a competitor**, interleaved, in every comparison set. A kernel that wins
   in isolation nominates; the served token elects (the Stage-B, split-KV-on-Metal and
   fused-argmax precedents).
3. **Machine state is recorded beside every number**: load average, thermal note, driver, commit,
   peer version, sampling config. The Mac uses `BENCH_MAX_LOADAVG=2.0` with the run count recorded
   (the macOS qualifier); the Linux box is idle-gated at 1.0. Checkpoints from `~/models` only —
   `/Volumes/` and `/srv/models` void a row.
4. **Paired differencing over pooled means**; arms adjacent in time with alternating order; a
   fresh server per cell and a verified-idle device between engines (Ollama keeps a model resident
   for minutes).
5. **Tokens from `usage`, never from frame counts**; realistic prompts for anything content-
   dependent; `BENCH_DEEP_CTX` scoped to the deep cells only.
6. **Bit-identity is a lane, not a default, and each brief says which lane it is in.** R3, R4 step
   0, R5's exact arm, R8's gates, R9, R10's fusions, R11 (b) and R13 keep bit-identity; R1, R2, R5's fast
   arm, R6 and R7's device path are fidelity-gated and say so in their flags, their `--help` text
   and their `benchmarks.md` rows. The snapshot golden is re-baked only with a stated mechanism,
   never because a number moved.
7. **Negatives are recorded at full value**, in the measurement file and in the ledger the lever
   came from (`ollama-chase.md` §10, the audit's finding, the queue entry), so nothing is re-proposed.
8. **A retraction reaches every page quoting the figure** — grep the figure with its unit at the
   retraction site.
9. **Campaign artifacts are cleaned up**: `.giw` bundles generated for a campaign deleted at session
   end; throwaway drivers either committed under a heavy-tests gate or deleted, never left as a
   scrollback; logs archived under `~/goinfer-logs/`, never left in `/tmp`.
10. **Any test or bench that runs longer than a couple of minutes emits progress logging.**
11. **PTX regeneration** is reproducible only at nvrtc 12.6.85 via the documented venv; plan it as a
    step in every CUDA brief.
12. **Prior art before re-deriving**: the reading list in each brief is not optional, and the
    recorded negatives it names are not re-proposed without a new mechanism.

---

## 7. What this doc does not claim

- No band here is a measurement. Every one is Amdahl on the record's own category shares, and the
  peer extrapolation that sits on top has already been ~25% optimistic once (`task-prefill-gap.md`
  §6.2). A brief that lands in its parked band is a result.
- R13's band is µop arithmetic read off the shipped kernels' own headers, on a category whose
  current share of the CPU slope nobody has measured — the softmax's per-key f64 `math.Exp` may no
  longer be the noise Gate A0 found it to be. Step 0 exists to say so before anything is built.
- The Metal kernels in R1 and R2 may lose to the M1 Pro's dispatch/occupancy floor the way four
  designs before them did. The briefs are written so that a fifth negative is worth as much as a
  win — a kernel shape that does not transfer from the peer's silicon is a finding about the
  silicon.
- The peer's Metal depth curve is a 2026-08-04 number on v0.32.5; R2 step 0 re-measures it before
  anything is claimed against it. The MLX figures are from one session (2026-09-04) at n=5 with the
  quant caveat `task-peer-benchmarks.md` §4 states; R12 (i) puts them on the page properly.
- Nothing here changes the exact paths: `gemv_w4a8_rn`, `attn_batched`/split-KV, the W4A8 Metal
  GEMVs and the shipped `attention` kernel remain what the parity gates run, and what a user gets
  with the lanes off.
- MoE prefill on M35 has no band because its blocker (batched Gated-DeltaNet) has no design yet;
  the 17× stays on the page as the size of the hole, not as a promise.
- The Linux CPU 0.5B anomaly and the WebGPU batched-prefill cost are investigations; their bands
  are registered in their write-ups, not here.

---

## 8. Sources

`docs/benchmarks.md` (TL;DR; §A; §B2; §B3; §B4.1; §B5.1; §B8; §B10; peer matrix 2026-09;
re-runs 2026-09-15/16 and 2026-09-17 on both boxes) · `docs/legacy-benchmarks.md` ·
`docs/queue-performance.md` (P20–P25) · `docs/audit-metal-2026-09-12.md` (§0, §1 M-01…M-16, §5,
§6, §7) · `docs/completed/task-prefill-gap.md` (§1, §2, §3–§3.2, §4, §6) ·
`docs/measurements/prefill-l2l3-phase{1,2,4-peer}-2026-09-05.md` ·
`docs/measurements/prefill-l2-metal-fused-attn-2026-09-09.md` ·
`docs/measurements/prefill-gate-l1-ref-b-2026-09-09.md` ·
`docs/measurements/prefill-batched-ttft-2026-09-13.md` ·
`docs/measurements/splitkv-{8000-reanchor,mechanism-ncu,stall-profile,q-staging,d7-fthreshold,f-depth-invariance,sector-efficiency,aa-floor,kernel-exploration}-2026-09-1[23].md` ·
`docs/completed/plan-still-slow.md` (P4, the KV-quant verdict §2.2's 2026-09-19 fit qualifies) ·
`docs/measurements/vsum-split-{spike,fidelity}-2026-09-13.md` ·
`docs/measurements/reduction-tree-accuracy-2026-09-12.md` ·
`docs/measurements/theta-per-backend-2026-09-01.md` · `docs/ollama-chase.md` (§1–§4, §7, §D4–§D6,
§9–§13) · `docs/completed/metal-verdict.md` (§1 rows 3–4, §2a–2b, §4) ·
`docs/completed/task-metal-cgofree-spike.md` (P1–P5) · `docs/completed/cuda-decode-headroom-audit.md` ·
`docs/completed/task-w4a8-neon-bandwidth.md` · `docs/completed/task-attention-decode-cost.md` ·
`docs/measurements/vsum-split-fidelity-2026-09-13.md` (per-prompt tables, for R6's 2026-09-19 amendment) · `docs/QUEUE.md` (G35, G36, G26, G28) ·
`docs/decode-fusion-next.md` · `docs/completed/gpu-next-levers-assessment.md` ·
`docs/tasks/task-gpu-paths-2026-09.md` (G9, G10, G11, the 2026-09-09 notes) ·
`docs/tasks/task-l01-hybrid-moe-cpu-gpu.md` · `docs/tasks/task-decode-splitkv-attention.md` ·
`docs/tasks/task-peer-benchmarks.md` · `docs/hardware-matrix.md` · `docs/roadmap.md` · `CLAUDE.md`
(measurement discipline) · code read for this doc: `cuda/vision_encoder.go`, `cuda/attn_img_prefill.cu`,
`cuda/decode_splitkv.cu`, `cuda/resident.go` (the split-KV gate), `cuda/l01_cpu_offload.go`,
`metal/kernels.go` (`attention`, the `gemv_w4a8_*` set), `metal/prefill.go`, `metal/backend.go`,
`gpu/gemv_w4a8.go`, `gpu/prefillrunner.go`, `decoder/sampler.go`, `decoder/spec_adaptive.go`; for R13 and
R6's amendment (2026-09-19): `decoder/attention.go` (`causalAttention`), `decoder/forwardn.go`
(`attendBatchedHeads`), `decoder/scratch.go` (`headWorkerPool`), `decoder/decode_depth_bench_test.go`,
`cuda/backend.go` and `cuda/resident.go` (the V-sum spike's loader and launch site) ·
aikit: `docs/task-simd-audit.md`, `linalg/attn_acc64_arm64.s` (and, for R13, `linalg/attn_acc64_amd64.s`,
`linalg/matmul_qk_acc64.go`, `linalg/matmul_av_acc64.go`, read at the vendored v1.45.1) and the `dot_w4a8_*` kernels, plus
aikit's internal roofline and CPU-acceleration notes (gitignored there, so described rather than
path-cited).

<!-- doc-reviewed: 2026-09-18 -->
