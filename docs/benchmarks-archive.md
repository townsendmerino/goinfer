# goinfer — benchmarks archive (retired, superseded and historical rows)

> **Nothing on this page is a current claim.** It is the record of what
> [`benchmarks.md`](benchmarks.md) used to say, kept because this repo's measurements are the
> product: a retraction that deletes its own evidence cannot be audited, and a superseded number
> whose chronology is gone reads as though it was never wrong. Every block below was moved here
> **verbatim** on **2026-08-31** — nothing was re-measured, re-derived or edited in the move, and
> the `⚠` boxes each block already carried are part of what is preserved.
>
> **For any current number, go to [`benchmarks.md`](benchmarks.md).** Its §B8 is the anchor for
> goinfer-vs-Ollama greedy CUDA decode; §B4.1, §B5.1, §B6.3 and §B7.1 are the re-anchored
> host↔VRAM, sampled, split-KV and deep-context sections.
>
> **Why each block is here:**
>
> | block | status | superseded by |
> |---|---|---|
> | The 2026-08-25 re-anchor banner | CLOSED 2026-08-31 — no leg still owed a run | the one-line banner on `benchmarks.md` |
> | §B (WebGPU int8 vs native) | **RETIRED** 2026-08-27, withdrawn not corrected | nothing — the rows are gone, not replaced |
> | §B2 (cgo-free CUDA vs Ollama) | historical peer **0.5.7** + the pre-re-anchor **v0.32.5** box | §B8 (decode); the slim §B2 on the live page keeps prefill + the peer-independent results |
> | §B5 (anchored re-measure) + the v0.11.0 qualification | superseded; the qualification was **RETIRED** rather than re-measured | §B5.1 / §B8 |
> | §B6 (split-KV) | superseded | §B6.3 |
> | §B7 (deep context) + its control | superseded | §B7.1 |
> | §B9 (relocated README tables) | **STALE** — peer v0.32.6 on the pre-2026-08-25 stack | §B8 (greedy) and §B5.1 (sampled) |
>
> **One correction made in the move, and only one.** §B9's own box says it is kept because its
> sampling-configuration table "exists nowhere else". That was true when it was written on
> 2026-08-27 and is **no longer true**: §B5.1 re-measured the sampled configurations on the current
> stack the same day. §B9's text below is left exactly as it stood — this note is the correction,
> so that the block stays a faithful record of what it claimed.
>
> **Durable mechanism findings that live in this file** (superseded *numbers*, but the mechanism
> was not withdrawn and is cited from elsewhere in the docs):
>
> - **§B5** — the per-segment coefficients showing the depth step is a **kernel switch, not depth**,
>   and the note on Ollama's depth behaviour (*do not attribute it to flash attention*).
> - **§B7** — the per-segment µs/KV-position coefficients, and **the deep gap is NOT DRAM
>   bandwidth**, where the alternative is falsified rather than assumed.
> - **§B2** — the prefill-campaign chronology, including the **five successive mis-attributions**
>   of the GEMV gap, each refuted by the next measurement, and the `ncu` finding that settled it.
>   That hygiene note is the most reusable thing in this file.

---

## The re-anchor banner (2026-08-25 → CLOSED 2026-08-31)

> **⚠ RE-ANCHOR PENDING (2026-08-26) — every number measured on the `linux` box is STALE until re-measured.**
> The box's whole OS and driver stack was replaced on **2026-08-25 22:28 PDT** (dnf transactions 96-102,
> a Nobara **43 → 44** upgrade, ~7,100 packages altered), and **2026-08-26 04:44 PDT was the first boot
> on it**. What moved:
>
> | | was (anchor of the rows below) | is |
> |---|---|---|
> | NVIDIA driver | `595.58.03-1.fc43` | **`595.91.07-2.fc44`** (dnf txn 99) |
> | CUDA version reported by the driver | *(not recorded)* | **13.2** |
> | running kernel | `7.0.5-200.nobara.fc43` | **`7.2.0-202.nobara.fc44`** (txn 98) |
> | glibc | fc43 | **`2.43-8.fc44`** |
> | distro | Nobara 43 | **Nobara 44** |
>
> Per this repo's rule, a driver change invalidates comparability and requires a **deliberate
> re-anchor, not a silent carry-forward**. The rows below are therefore kept, dated and marked —
> **not** carried forward as current, and **not** edited to match a new number.
>
> **The scope is wider than the driver.** The decision to re-anchor was taken on the driver bump
> alone (2026-08-25); the transaction log shows it arrived inside a distro major upgrade, so kernel,
> libc and the graphics stack moved with it. Anything that touches CPU-side dispatch, scheduling or
> the loader is in scope too, not only the GPU rows.
>
> **It is also not only a throughput question.** The `cuda/` backend is **driver-JIT**: the driver
> compiles the frozen PTX at load, so this upgrade changed the *compiler*, not just the runtime.
> Bit-identical and parity claims across the bump must be re-verified on their own, ahead of any
> tok/s number.
>
> **Scope — 76 data rows in six anchored sections** (was 78 in seven; §B's 2 were RETIRED
> 2026-08-27, not re-measured): §B2 (6), §B4 (8), §B5 (19), the
> v0.11.0 qualification (2), §B6 (16), §B7 (25). Rows already marked RETIRED or HISTORICAL in §B2
> stay that way — this box does not revive them. **§A** (Apple Silicon CPU) and **§B3** (Metal) are
> measured on the MacBook and are **unaffected**.
>
> **§B7 (deep context) DISCHARGED 2026-08-27** — goinfer unchanged (39.1 vs 39.0 at the
> depth-matched 32000 cell); the peer is 40% slower there, so the improved ratio is the peer
> moving, not a goinfer win. See §B7.1.
>
> **§B5 (sampled + phi3-mini/gemma3-1b) DISCHARGED 2026-08-27** — five of six comparable cells
> improved, three by 8–19%; phi3-mini at temperature 1.0 lost 5.8% and is filed. See §B5.1.
> **That loss is RESOLVED (G26, 2026-08-28): about half of it was the anchor cell's own spread, the
> real effect is −2.6%, the cause is optimistic forward, and it is fixed in `c9a79b4`. See the
> resolution box in §B5.1 — the −5.8% figure should not be quoted forward.**
>
> **§B4 (26B host↔VRAM) DISCHARGED 2026-08-27** — the throughput did not move (11.42 vs 11.3 at
> the same 16 slots); the grantable slot cap did, 38 → 30. See §B4.1.
>
> **§B6 (split-KV) DISCHARGED 2026-08-27** — re-measured on this stack with a committed harness
> (`scripts/bench_splitkv.py`); the ratios did not move, most cells agreeing to ±0.005. See §B6.3.
>
> **PARTIALLY DISCHARGED, 2026-08-26.** Parity was re-established first (`gate gpu` PASS at
> `a161bd6`: real-model heavy tier green, CUDA graphs bit-exact, **24/24 PTX byte-identical** — the
> driver's new compiler did not move the numerics), and the greedy peer sweep then re-anchored the
> §B2/§B5-style rows into **§B8**. What that run does **not** cover remains STALE and may not be
> quoted as current: ~~**§B** (WebGPU at int8/q8_0 — §B8 is q4_K_M)~~ **— §B was RETIRED 2026-08-27
> rather than re-measured, so the only leg still owing a run is the v0.11.0 qualification.** Each is a separate leg with its own procedure;
> `bench_peer.py` does not produce any of them.
>
> **CLOSED 2026-08-31 — no leg still owes a run.** The v0.11.0 qualification was RETIRED rather
> than re-measured (see its own box), on the same basis §B was: its numbers were §B6/§B7 by
> code-identity, and both are now re-anchored. Every section this banner listed is therefore either
> re-anchored (§B4.1/§B4.2, §B5.1, §B6.3, §B7.1, §B8) or deliberately withdrawn (§B, v0.11.0).
>
> Re-measure with `scripts/bench_peer.py` (peer Ollama v0.32.5 at `~/ollama-0325`, both sides over
> HTTP, interleaved) — **not** `bench_compare.sh`, which by its own design note never drives the
> peer.


---

## §B — RETIRED 2026-08-27

### B. GPU residency vs native CUDA, at equal quant

> **⚠ RETIRED (2026-08-27) — both rows are withdrawn, not corrected. This box is retained as the
> record of what was claimed.** What stood here: Qwen2.5-Coder-1.5B int8 vs q8_0 at **111.6 tok/s**
> against Ollama-CUDA **147** (**76%**), and Qwen2.5-7B int4 vs q4 at **51.7 tok/s** *(pre-coalescing)*
> against llama.cpp-CUDA **72.8** (**71%**). Neither is reproducible by this page's current rules and
> neither may be quoted.
>
> **Why withdrawn rather than re-measured.** The decision was taken deliberately after costing the
> alternative. The **7B row is unsalvageable**: its peer figure came from `CHANGELOG.md` v0.5.0 —
> a different engine (llama.cpp), a different session, and no interleaving — so nothing short of a
> full new run could rescue it, and a new run against Ollama would be a different row answering a
> different question. The **1.5B row is repairable but not worth it now**: it needs an int8/q8_0
> pair this repo does not hold (the archive carries only q4_K_M for this model), a WebGPU build, and
> a quant axis in `bench_peer.py`, which hardcodes `-quant int4`. It describes **dense-only**
> residency (`decodeRunnerEligible`) on a path untouched since **2026-06-08** and at the WGSL wall,
> so it covers none of the families added since — Granite, Nemotron, DeepSeek, GLM, Kimi, Gemma 4.
> **If a WebGPU claim is wanted later it should be measured fresh against current assets, not
> repaired.**
>
> The original staleness note follows, because the reasoning in it is what led here.
>
> **⚠ STALE — measured before the 2026-08-25 re-anchor** (driver `595.58.03`, Nobara 43, kernel
> `7.0.5-200.fc43`).
>
> **AND THIS ONE IS NOT A RE-ANCHOR JOB — it is a new measurement.** The other stale sections need
> the same run on a new driver. This section's *method* is one this page has since rejected: its
> peer figures were not measured beside goinfer in the same session (the 7B peer is quoted from
> `CHANGELOG.md` v0.5.0), and cross-session drift on this box is ~3.5%, which is a large fraction
> of the gap being reported. The peers have also moved out from under it — Ollama 0.5.7 (2025-01)
> and llama.cpp as of v0.5.0, against today's Ollama v0.32.5.
>
> **Re-running goinfer alone would not fix it, and would look like it had.** goinfer's side is
> expected to land in the same place: dense WebGPU decode is at the WGSL wall (~90 tok/s int8,
> megakernel inexpressible in WGSL — `docs/task-benchmark-refresh.md`), and nothing since
> 2026-06-08 touches that path. What has actually changed is the *peer*, so the ratio can move
> without goinfer moving at all. Redo it as one same-session interleaved pair at int8/q8_0, or
> retire the section — do not carry the ratio forward.
>
> **Scope, which the README used to omit:** these are dense-Qwen2/Llama **residency** decode rows.
> Residency is dense-only (`decodeRunnerEligible`), so every family added since — Granite,
> Nemotron, DeepSeek, GLM, Kimi, Gemma 4 — is ineligible and is NOT described by this figure.

Rig: **RTX 2070 SUPER / Ryzen 7 3700X**, warm (`ollama ps` 100% GPU), greedy. goinfer
WebGPU residency (`-tags gpu`), bit-exact vs CPU decode on the first tokens. Source &
lab notebook: `docs/completed/gpu-assessment.md` §0.0 (goinfer commit `eaf9a6c`, **2026-06-08**);
final corrected numbers only. The 1.5B row (89.7 / 147 / 61%) is from gpu-assessment;
the **7B row's peer figure (72.8 / 71%) is from `CHANGELOG.md` v0.5.0**, not
gpu-assessment — cited there.

*(Both rows retired 2026-08-27 — see the box above, which preserves the figures.)*

> **These WebGPU rows are int8/q8_0 and are NOT comparable to the cgo-free CUDA rows
> below**, which are 4-bit on both sides. Two different backends, two different quants,
> two different peer measurements — read them as separate experiments, never as a
> before/after of the same thing.


---

## §B2 — historical peer rows and the prefill campaign

### B2. cgo-free CUDA (`-tags cuda`) vs Ollama-CUDA — 4-bit both sides

> **⚠ SUPERSEDED for greedy peer rows — measured before the 2026-08-25 re-anchor** (driver
> `595.58.03`, Nobara 43, kernel `7.0.5-200.fc43`). The current anchor is **§B8**, re-measured
> 2026-08-26 on driver 595.91.07 / Nobara 44. What §B8 does **not** cover stays STALE and is not a
> current claim: the sampled (`temp` / `temp+top_p`) rows, and every phi3-mini and gemma3-1b cell —
> that sweep was greedy-only and qwen2.5-coder-only.

> **⚠ RETIRED (2026-08-09) — the 0.5B pair and its 1.78× are withdrawn, not corrected.**
> The two halves were produced by **different methods**: `scripts/bench_compare.sh` measures
> goinfer with **in-process Go benchmarks** and, by its own design note, **never drives the peer**
> ("This script does not run them: their install is yours"), while the peer column came from
> Ollama's HTTP server. The ratio therefore divided a **kernel throughput** by an **end-to-end
> throughput**. Measured server-to-server on 2026-08-09, goinfer's 0.5B at 128 greedy is
> **320.1 tok/s** against Ollama v0.32.5's **269.4**. *The peer half reproduced exactly (268 → 269.4);
> ours did not, because it was never measuring the same thing.* The row previously marked
> `sampling: unrecorded` is **retired** for the same reason — no committed artifact reproduces it.
> Current, fully-provenanced numbers: the README (`### Measured throughput — goinfer` and
> `### Compared with Ollama v0.32.5`). This box is retained as a record of what was claimed.
>
> **⚠ RE-ANCHORED TO CURRENT OLLAMA (2026-08-04). Read this before the historical rows below.**
> The tables further down were measured against **Ollama 0.5.7 (released 2025-01-16)**. Current
> Ollama is **v0.32.5 (2026-07-27)** — ~18 months and a new inference engine + flash attention
> newer. Benchmarking a live claim against an 18-month-old peer is not honest, so §B2 was
> **re-measured against v0.32.5** on the same RTX 2070 SUPER (installed at `~/ollama-0325`, kept
> beside the pinned 0.5.7). The competitive picture changed materially:
>
> | metric (q4_K_M, best of 3, decode-only server-reported · **sampling: unrecorded for the v0.32.5 re-measure**) | goinfer | Ollama 0.5.7 | **Ollama v0.32.5** |
> |---|---|---|---|
> | 0.5B decode | ~476 tok/s | 211 | **268** |
> | 1.5B decode, short ctx | ~221 tok/s | 149 | **186** |
> | 1.5B decode, **2048 ctx** | **157.6 tok/s** (re-measured 2026-08-09; was ~97, then ~133 mid-campaign) | — | **179.2** |
> | 1.5B prefill | ~0.66 ms/tok | ~0.17 | **~0.14** |
>
> **The honest current claims:**
> - **0.5B: goinfer still ~1.7× ahead** (476 vs 268). Tiny-model decode is launch/issue-bound, and
>   the cgo-free path's cheap per-token dispatch is a real, durable edge in that regime.
> - **1.5B short context: PARITY** (~1.19×, not the 1.41× vs 0.5.7). goinfer is marginally ahead; do
>   not call it a win.
> - **1.5B long context (2048): goinfer is BEHIND ~1.41×** (133 vs 188) — its decode still slows with
>   KV depth where current Ollama holds rate, but the deficit narrowed after the fix below. The old
>   §B2 never measured this. *ncu traced the deep-context slowdown to the decode attention's K read
>   (21.96% bytes/sector — uncoalesced, stride-`kvDim`). Wiring the decode M=1 attention to the
>   already-float4-coalesced `attn_batched` kernel (bit-identical to the audited glue `attention` at
>   M=1, `TestAttnBatched_bitIdentical`) recovered 2048-ctx decode **99.5 → 133.5 tok/s (1.34×)** on a
>   same-box A/B (`TestDecodeDepthThroughput`, git-stash A/B), shallow decode unchanged. This shipped
>   after the correction above — a measured improvement, not a rescue: goinfer is still behind the
>   current peer at long context, just by less.*
> - **Prefill: Ollama is ~4–5× faster/token.** The batched-prefill campaign (below) is real
>   *engineering* — bit-identical, crossover 128→320 **against 0.5.7** — but it does **not** make
>   goinfer competitive on prefill against the current peer.
> - **The Ollama crossover collapses ~320 → ~50 tokens** against v0.32.5: with a ~1.19× short-ctx
>   decode edge and ~4–5× slower prefill, goinfer wins total request time only for trivially short
>   prompts. The "wins short/medium prompts" framing was an artifact of the stale peer.
>
> **What is peer-independent and still true** (this is where the release should lead):
> - **cgo-free** (CGO_ENABLED=0, driver-only `ldd` — no libcuda/libnvrtc linked), **portable**,
>   **bit-identical** decode.
> - **The batched-prefill campaign as an absolute engineering result:** real qwen2.5-coder-1.5b
>   **2048-token TTFT 13.1 s → 2.1 s** (6.17× vs the old sequential path) — a measured TTFT win with
>   **no peer involved**, and **bit-identical to sequential prefill** (restored). *History (2026-08-04):
>   the bit-identity claim was briefly FALSE on real models — an **84% token-stream divergence** from a
>   compiler **fma-vs-mul+add contraction** difference between the separately-compiled batched and decode
>   GEMV/RMS kernels (~1 ULP, data-dependent; the old `TestPrefillLast_e2e` missed it because its tiny
>   uniform-magnitude fixture rounds identically). **FIXED:** every float MAC in both paths is now an
>   explicit `__fmaf_rn` intrinsic (no compiler discretion), enforced at build time by
>   `cuda.TestKernelFMALint`. Evidence it's restored: **`TestPrefillDivergenceRate` = 0/50** on the real
>   1.5B (was 42/50), gap byte-identical. Full write-up: `docs/completed/task-batched-prefill-bitidentity.md`.*
> - **§B4's 26B-A4B on an 8 GB card, fully GPU-resident** at 16.98 tok/s. *Re-verified (Task 2): the
>   old "peers fail to load it" claim is FALSE for current Ollama — v0.32.5 loads and runs the same
>   26B via a 42% GPU / 58% CPU-RAM split at ~24.5 tok/s, faster than goinfer here.* The honest,
>   peer-independent claim is narrower: goinfer runs it **fully on the GPU** (experts streamed
>   host→VRAM, not offloaded to CPU) — an architecture distinction, not a capability the peer lacks.
>   *Opt-in: this row needs three environment variables (§B4), none of them the default.*
>
> Everything below this box is a **labeled historical record (vs Ollama 0.5.7, 2025-01-16)**, not a
> current competitive claim.

The `cuda/` backend is a **driver-JIT, CGO_ENABLED=0** path (dlopen `libcuda` via
purego; PTX embedded, no toolkit at build or run time — re-verified by `ldd` at the
commit below showing no `libcuda`/`libnvrtc` linked). It admits an arch only when it
implements every feature that arch needs, else declines to the staged/CPU path
(`decoder/features.go`, the authoritative set). **The numbers below are the dense lane.**
At this commit (`7557723`, 2026-07-16) the backend was dense-only, but MoE (routed +
ungated shared expert) and partial rotary landed the next day (2026-07-17) — `features.go`
now carries `cuda`'s `FeatMoE`/`FeatPartialRotary` plus the full Gemma norm/GeGLU/embed set —
so those lanes are real but **unmeasured here** (still declined: MLA, SSM, the gated shared
expert, YaRN mscale, logit softcap).

Provenance, all rows: **RTX 2070 SUPER**, driver **595.58.03** / Ryzen 7 3700X ·
goinfer commit **`7557723`**, **2026-07-16** · peer **Ollama 0.5.7** · both sides
q4_K_M, warm (`/api/ps` shows `size_vram == size`, 100% GPU), greedy (`temperature:0`),
256-token completions (both hit the cap), **best of 3 warm runs**. **These rows are HISTORICAL (peer Ollama 0.5.7, 2025-01-16); the current-peer numbers are in the re-anchor box above.** (the first run after
load is discarded as a warmup outlier on both sides).

**Method (2026-08-09 re-measurement).** Server-to-server: both engines driven over their own HTTP
server, **client-timed inter-token rate** (decode-only, prefill excluded), **interleaved cell-by-cell
with a server restart between cells**, same WEIGHTS both sides (verified per tensor by
`scripts/gguf_same_weights.py` — a file md5 cannot pass, see §B2's note), sampling sent explicitly to each.
The committed harness is `scripts/bench_peer.py`. *Historical note: the older rows below claimed
"no methodology gap to discount" while goinfer's side came from in-process Go benchmarks — that
claim was false and is withdrawn.*

**Historical method (pre-2026-08-09).** Each engine is driven through
its *own* HTTP server (goinfer `cmd/serve` `/v1/chat/completions`; Ollama
`/api/generate`) and timed by **client wall clock**, so prefill, sampling, detokenize,
JSON, and HTTP are inside *both* numbers. This is the only methodology-symmetric
comparison on this page.

| **HISTORICAL (vs Ollama 0.5.7, 2025-01-16 — see re-anchor box above)** · q4_K_M | goinfer (`-tags cuda`) | Ollama 0.5.7 | goinfer vs 0.5.7 |
|---|---|---|---|
| Qwen2.5-Coder-0.5B | **429.8 tok/s** | **211.1** | **2.04×** |
| Qwen2.5-Coder-1.5B | **210.0 tok/s** | **149.3** | **1.41×** |

- **Conservative cross-check.** Ollama also self-reports a *decode-only* rate
  (`eval_count / eval_duration`, excluding prefill): **216.8** / **153.4** tok/s. Scoring
  goinfer's all-in wall clock against Ollama's prefill-free figure — a bar tilted
  *against* goinfer — still gives **1.98×** / **1.37×**. The ratio is not a
  measurement artifact. (Ollama's wall and decode-only agree within ~3% here, so its
  server overhead is negligible and the two framings barely differ.)
- **⚠ These rows hold only for SHORT prompts — the advantage is prefill-length-bounded.**
  §B2 uses short prompts + 256-token completions, so prefill amortizes away. Ollama **batches**
  prompt processing (measured ~0.17 ms/token, batched cuBLAS), so its total request time is
  nearly flat with prompt length (decode-dominated: ~120–140 tok/s → ~560–600 ms for
  prompt+64). goinfer's resident CUDA **decode** is actually *faster* (~200 tok/s at short
  context), which is the §B2 win — but its prefill cost is what erodes that lead as the prompt
  grows, so there is a crossover length past which goinfer's total time exceeds Ollama's.
- **Batched prefill (`PrefillLast`, `cuda/prefill.go`, 2026-08-03) moved the crossover
  ~128 → ~230 tokens (~1.8×).** The weight-stationary batched W4A8 GEMV replaces the
  sequential **M=1** prefill (one full forward per prompt token) with an M=len pass that reads
  each int4 weight row once per 8-token tile. Measured (real qwen2.5-coder-1.5b, same rig/quant,
  Ollama 0.5.7 `q15`):
  - **TTFT (prefill only), batched vs the old sequential path:** 128 tok **3.46×**, 512 **3.25×**,
    2048 **2.79×** (best at short prompts; the ratio falls as O(M²) attention — paid by both paths
    — grows to dominate).
  - **Total request time (batched prefill + 64 greedy decode):** goinfer 128 tok **450 ms**,
    192 **530 ms**, 256 **623 ms**, 512 **1037 ms**, 2048 **5311 ms**; Ollama ~**560–600 ms**
    flat across 128–628 tok. So goinfer now **wins up to ~230-token prompts** (192: 530 vs ~595;
    256: 623 vs ~600 — the crossover) and loses beyond, vs ~128 before.
  - **Where batched-prefill time goes (`TestPrefillDecomp`, category timers):** at 128 tok GEMV is
    **88%** (attn 8%, glue 4%); at 512, GEMV **73%** / attn 25%; at 2048, attn **56%** / GEMV 43%.
    So GEMV dominates *around the crossover*, and attention (naive O(M²) `attn_batched`) only takes
    over past ~1–2k tokens.
  - **The residual is currently ~1–2.3 ms/prompt-token vs Ollama's ~0.17; attribution (profiled,
    `TestGemvBatchedBandwidth` + `ptxas`, no `ncu` on this box): the GEMV is activation-L2-read-bound,
    NOT compute-bound.** One gate/up GEMV (N=8960, K=1536, M=512) = 4.98 ms = **7.9% of Turing dp4a
    peak**; weight read a trivial **1.4 GB/s**; but activation read runs at **~1.41 TB/s effective**
    (3× the card's 448 GB/s DRAM → saturating L2). Cause: weight-stationary means each of the N
    output-row warps re-reads the whole [M,K] activation — **N×M×K vs the M×K minimum, a factor of
    N (8960×)**. `ptxas` rules out occupancy (43 regs @MT=8 / 61 @MT=32 → 100% on sm_75; MT=64 = 108
    regs → 50%, its regression, **no spill**) and the MT sweep rules out weight-fetch (MT changes
    weight reuse, not the activation re-read → ~6%). **IMMA is refuted: it raises a compute ceiling
    that is 92% unused.** The lever is standard GEMM tiling — stage the [MT,K] activation tile in
    shared memory so it is read once per tile-group, not N times. That is **bit-identical by
    construction** (changes where operands are read, not their accumulation order) — no tolerance
    gate, no IMMA. At long context the separate lever is flash-style attention.
  - **UPDATE — the "activation-L2 traffic" attribution was ALSO wrong; the staging fix refuted it
    (this time the experiment ran before the claim shipped).** Built `gemv_w4a8_staged` (own file,
    bit-identical: `facc` live in registers across all K-chunks, single warp-reduce; staged tile
    padded `[2·KC][MT+1]` to avoid bank conflicts) and measured it. It cut global activation reads
    **8× (7.05 GB → 0.88 GB, gate/up shape)** — exactly as the "N× re-read" model predicted — yet wall
    time moved only **4.98 → 4.2 ms (1.2×)**, 7.9% → 9.3% of dp4a peak. An 8× traffic cut buying ~0
    means the kernel was **not** activation-bandwidth-bound: the 1.41 TB/s figure was the read *rate*
    the compute loop demanded (served fine by L2), not a saturated ceiling. The staged kernel spends
    its 4.2 ms in the compute loop (0.88 GB of global load would be ~0.6 ms alone), so the real bound
    is **the dp4a + one-shared-read-per-MAC compute loop itself — LSU / issue throughput at low
    arithmetic intensity (~4 MACs per activation read)**, not memory. The lever for THAT is arithmetic
    intensity: register-block RN output rows per warp so each activation read feeds RN× the MACs (still
    bit-identical — per-row `facc`, one reduce each). Staging is not wired into the prefill path; the
    kernel + `TestGemvStagedBandwidth` are kept as the reproducible refutation.
  - **RESOLVED by ncu (once the profiler was installed) — L1TEX-latency-bound, and a coalescing fix
    landed.** The profile on the real 4.4 ms kernel: DRAM 1.7%, L2 6%, Compute 46%, but No Eligible 71%,
    IPC 1.07/4, ~11 cyc/instr stalled on an L1TEX scoreboard dependency — latency-bound on L1TEX, not
    compute/DRAM/issue. Root cause named directly: **49.99% bytes-per-sector on global loads** — the
    activation pair `a[2·wi], a[2·wi+1]` was two stride-2 loads, so adjacent lanes read words 2 apart and
    only 16 of 32 sector bytes were used. **Fix (bit-identical, shipped): load the pair as one `int2`** →
    98% bytes/sector, L1TEX 93% → 65%, **4.98 → 4.41 ms (~13%)**. Modest because the bound only *moved*:
    L1TEX latency is now *exposed* rather than throughput-saturated (17.8 cyc/instr, no unit maxed) — too
    few eligible warps to hide the now-efficient loads. The next lever the profile justifies is
    arithmetic intensity: register-block RN output rows per warp (a latency fix, still bit-identical),
    *not* activation-staging (built, refuted) and *not* IMMA (compute ceiling 92% unused).
  - **Note on this report's own hygiene — the gap was mis-attributed FIVE times, each refuted by the
    next measurement, four of them by me before the profiler existed:** (1) *~23× weight-amortization
    ceiling* (MT sweep: ~6%), (2) *needs IMMA* (7.9% of dp4a peak), (3) *activation-L2-bandwidth-bound*
    (staging kernel: 8× traffic cut, 1.2× time), (4) *issue-bound on a fat SASS mix* (ncu: 22.78% issue
    slots — not issue-bound), before (5) the hardware-stated answer: L1TEX latency from uncoalesced
    loads. Every inferred mechanism — including a profiled-looking 1.41 TB/s and an 8-instruction SASS
    count — read as a measurement and was wrong; only `ncu`'s Scheduler/Warp-State sections settled it.
    The specific error in (3) is worth keeping: a *demanded* read rate exceeding DRAM proves the reads
    are cache-served, **not** that the cache is saturated — that inference was mine and it looked like
    data. And the precise form of the (4) failure, different from the earlier three: an instruction-mix
    histogram bounds throughput **from above** — it cannot establish that you are *at* that bound. Only
    stall and eligibility data can. The pattern to watch is the *speed of attribution*: pull the lever
    and move the time, or profile the unit directly; do not record a mechanism as a bound.
  - **RN-blocking landed (the second profile-justified lever).** With the load coalesced, the kernel was
    still L1TEX-latency-bound (17.8 cyc/instr, 100% occupancy — more warps can't help), so the lever is
    fewer loads per MAC. `gemv_w4a8_rn` (own file) computes RN output rows per warp, reusing each `int2`
    load across all RN rows → RN× fewer L1TEX loads; bit-identical (per-row `facc`, one reduce each).
    The unroll-depth proxy first was inconclusive — forcing unroll on the runtime loop wrecks the
    compiler's already-optimal schedule — so RN was tested directly. Sweep knee **RN=2/MT=16, 64 regs,
    100% occupancy, 4.41 → 3.38 ms (1.30×)**; RN·MT>64 spills occupancy and regresses. ncu confirms the
    bound moved: scoreboard stall **17.8 → 7.5 cyc/instr**, Compute 46 → 54%, L1TEX 93 → 67%. Wired into
    `PrefillLast`; gates green (GEMV bit-identical, KV all layers×rows, 64-token decode byte-identical).
  - **Cumulative GEMV lever stack and the moved crossover.** MT (~6%) + coalescing (~13%) + RN (~30%) ≈
    **1.5×** on the gate/up GEMV (4.98 → 3.38 ms). End-to-end on real qwen2.5-coder-1.5b: **TTFT vs
    sequential 128 5.27× / 512 4.43× / 2048 3.33×** (was 3.46/3.25/2.79); total request time (prefill +
    64 decode) 128 **399 ms**, 256 **527**, 320 **600**; **Ollama crossover ~230 → ~320 tokens**
    (~128 before any of this work). goinfer now wins total time up to ~320-token prompts.
  - **Attention lever (float4 coalescing) — the long-context win.** `attn_batched` was profiled
    (ncu, 1.5B, M=2048) as **L1TEX-throughput-saturated (98.86%)**, DRAM idle, Compute 11.5%, with global
    K/V loads at **21.96% bytes/sector** (the QK pass splits threads over keys → stride-`kvDim`
    uncoalesced) — genuinely traffic-bound, *not* the A-staging trap. `float4`-ing the K/Q read (four
    adds kept separate → bit-identical, the GEMV `int2` lesson) → bytes/sector 66.32%, **isolated
    attn_batched 92 → 29.9 ms (3.1×)**, Compute 11.5 → 23%. End-to-end (`TestPrefillDecomp`): attention
    128 11.4 → **4.2 ms**, 512 164 → **53 ms**, 2048 2605 → **820 ms** (~3.2×); attention's share of
    prefill at 2048 dropped **66.9% → 38.9%**. **TTFT vs sequential jumped at long context: 128 5.78× /
    512 5.80× / 2048 6.17×** (was 5.27/4.43/3.33 with the GEMV stack alone). Total prefill at 2048
    3.9 → 2.1 s. The **crossover stays ~320** — it lives at short prompts where attention is ~5% and the
    GEMV dominates, so this lever doesn't move it (it is a long-context lever). Gates green (attn
    bit-identical + e2e byte-identical decode).
    - **Decode reuse (later, Task 3):** the same coalesced `attn_batched` at **M=1** is bit-identical
      to the glue `attention` (`TestAttnBatched_bitIdentical`), so the *decode* attention was wired to
      it too (`resident.go`, guarded by `prefillReady`, glue is the fallback). ncu had traced the
      long-context *decode* slowdown to the same 21.96%-bytes/sector K read; the swap recovered
      2048-ctx decode **99.5 → 133.5 tok/s (1.34×)** same-box A/B (`TestDecodeDepthThroughput`),
      shallow decode unchanged, glue.ptx untouched. See the §B2 long-context bullet.
  - **Attention residual (the next build, not yet done): L1TEX still 99.51% saturated after float4** —
    coalescing fixed the *waste per read*, but the **O(M²) redundant re-reads** remain (each K/V read ~M
    times, L1-served — DRAM idle). Only **query-tiling** (share a staged K/V tile across a query block)
    removes it, the bit-identical query-tiled + 2-pass-recompute design in `task-prefill-attention.md`.
    Design constraint surfaced by the profile: attention()'s exact float-sum order fixes the blockDim=128
    reduction tree and thread→key map, forcing Bk=128 key tiles that strain the 64 KB shared budget at
    hd=128/256 — the 2-pass-recompute (no materialized scores) is how it fits.
  - **The ceiling, stated plainly: kernel tuning will not close the gap at 2048-token prompts.** Two
    reasons, both structural. (a) Attention is still ~39% of prefill at 2048 and the query-tiling lever
    above is bounded by L1TEX/shared, not free. (b) The GEMV residual is the **tensor-core gap**: Compute
    is 54% of the *dp4a* peak, and dp4a is ~1/3 of Turing IMMA — perfect latency hiding buys ~another
    1.85× to the dp4a ceiling, but the ceiling itself is dp4a, not IMMA. Closing that needs the
    tensor-core GEMM in `docs/completed/task-rotation-perrow-imma.md`, which reorders the group-scaled cross-group
    float sum and so cannot be bit-identical — **scoped and unfunded**. An architectural consequence of
    the cgo-free, bit-identical thesis: a **stated trade, not a deficiency**. (Against the current peer
    v0.32.5 the total-time crossover is short — see the re-anchor box at the top of §B2; goinfer stays
    behind Ollama on raw prefill throughput at long context.)
  - Gates: `cuda/prefill_e2e_test.go` (KV bit-identical all layers×rows + logits + 64-token
    byte-identical decode, real windowed Mistral); `TestPrefillTTFT` / `TestPrefillCrossover` /
    `TestPrefillDecomp` (heavy) reproduce the numbers above. The rows in the §B2 table are
    **not edited**.
- **Correctness is gated, not assumed.** CUDA decode is held to the repo's own 3%
  near-tie parity rule against the CPU path on a real q4_k_m checkpoint (9/10 exact
  argmax, 0 hard fails) — the speed is only meaningful because the tokens match.
- **Why the 0.5B ratio is bigger:** small-model decode is launch/issue-bound, and the
  cgo-free path's per-token dispatch is cheap (measured executor tax **15.3 µs**/token).
  The advantage narrows as the model grows and work becomes bandwidth-bound.
- **7B — now MEASURED (`TestB2DenseFlagship`, qwen2.5-7B int4 resident, best-of-3 vs pinned
  Ollama 0.5.7), and the narrowing is confirmed.** Decode tok/s goinfer **70 / 62 / 43** vs
  Ollama **64 / 56 / 38** at 128/512/2048 — still faster, but only **~10%** (both bandwidth-bound
  near peak, so goinfer's dispatch edge no longer dominates). And goinfer **loses all-in at every
  length**: its prefill is ~2.6–3.5 ms/token vs Ollama's ~0.5 (batched cuBLAS), a 5–7× per-token
  penalty the ~10% decode edge cannot cover, so the **crossover for the 7B is below 128 tokens**.
  So the honest headline: goinfer's total-time advantage is a **small-model** phenomenon (real and
  large at 0.5–1.5B, crossover ~320 at 1.5B; decode-only and small by 7B). Publish the 0.5B/1.5B
  rows as the win; publish the 7B as the size-trend, not as a headline.
- **Scope, stated plainly:** these are dense-lane numbers (0.5B/1.5B dense) — a real-speed
  claim about the dense lane, **not** a coverage claim. Coverage is `decoder/features.go`,
  which since 2026-07-17 grants `cuda` MoE (routed + ungated shared) + partial rotary + the
  Gemma set; MoE-resident speed is unmeasured. (Earlier drafts of this bullet said "dense
  architectures only / No MoE" — stale as of 2026-07-17.)


---

## §B5 — superseded, plus the RETIRED v0.11.0 qualification

## B5 — Anchored re-measure, 2026-08-09 (goinfer `686c9f8` vs Ollama v0.32.6)

> **⚠ SUPERSEDED for greedy peer rows — measured before the 2026-08-25 re-anchor** (driver
> `595.58.03`, Nobara 43, kernel `7.0.5-200.fc43`). The current anchor is **§B8**, re-measured
> 2026-08-26 on driver 595.91.07 / Nobara 44. What §B8 does **not** cover stays STALE and is not a
> current claim: the sampled (`temp` / `temp+top_p`) rows, and every phi3-mini and gemma3-1b cell —
> that sweep was greedy-only and qwen2.5-coder-only.
>
> **RE-ANCHORED 2026-08-27 — see §B5.1 below.** Those rows are now re-measured on driver
> `595.91.07` / Nobara 44. The sampled paths moved, mostly upward: **+19.4%** on the 0.5B at
> `temp+top_p`, +8–9% on the 0.5B and gemma3-1b elsewhere, against an Ollama side that did not
> move. One went the other way and is recorded rather than explained: **phi3-mini at
> `temperature 1.0` lost 5.8%**. The rows below stay as the `595.58.03` / v0.32.6 record they are.

One binary for every cell. Supersedes the sampled rows in the README's G11 section; the older rows
elsewhere on this page keep their own binary and peer version and are NOT updated in place.

**Provenance, every row below:** goinfer **`686c9f8`** · peer **Ollama v0.32.6** with
**`OLLAMA_FLASH_ATTENTION:false`** (its default — not overridden), `num_ctx` set explicitly per cell
and **verified via the `ollama ps` CONTEXT column** (all 10 peer cells: observed == requested,
`100% GPU`) · RTX 2070 SUPER, driver **595.58.03** · qwen2.5-coder / phi3-mini / gemma3-1b at
**q4_K_M**, same weights both sides (per-tensor verified; a file md5 cannot pass) · **2026-08-09** · **decode-only, prefill excluded**
(inter-token rate timed client-side from the first streamed token) · servers restarted per cell,
interleaved · ≥8 completions per run, 2 runs per cell, spread shown · sampling sent explicitly.

**Controls.** goinfer greedy@128 measured 318.9 against 320.1 on the pre-P1 binary, and Ollama
greedy@128 measured 269.4 against 269.4 on v0.32.5 — so neither the goinfer sampling work (P1/P2b)
nor the peer version bump moved the greedy path, and the sampled deltas below are attributable.

### Sampled configurations, 128 context

| model (vocab) | config | goinfer | Ollama v0.32.6 | verdict |
|---|---|---|---|---|
| phi3-mini (32k) ᵈ | temp-only | 116.6 ±0.5 | 125.6 ±0.2 | Ollama 1.08× |
| phi3-mini (32k) ᵈ | temp+top_p 0.95 | 99.4 ±0.6 | 121.4 ±0.3 | Ollama 1.22× |
| qwen2.5-coder-0.5b (152k) | temp-only | 219.2 ±1.1 | 269.0 ±0.9 | Ollama 1.23× |
| qwen2.5-coder-0.5b (152k) | temp+top_p 0.95 | 190.3 ±1.6 | 266.2 ±0.3 | Ollama 1.40× |
| gemma3-1b (262k) ᵈ | temp-only | 131.7 ±6.1 ᵉ | 149.1 ±0.5 | Ollama 1.13× |
| gemma3-1b (262k) ᵈ | temp+top_p 0.95 | 115.2 ±2.3 | 149.6 ±0.4 | Ollama 1.30× |

ᵈ **The phi3-mini and gemma3-1b rows were RE-MEASURED on 2026-08-09 and supersede the originals**
(goinfer `ca29d6c`, Ollama v0.32.6, 5 runs × 8 completions, both engines **interleaved in one
session**). The qwen2.5-coder rows are unchanged — they came from the main harness, which already
interleaved.

ᵉ 4.6% spread: under the 5% threshold, but the widest cell on this page. Treat it as indicative.
The `temp+top_p` cell had been **HELD, not published**, at **95.4 ±10.7** — an 11% spread, over the
5% threshold this page uses; publishing a number whose two runs disagree by 11% is how a spread gets
averaged away. It now reads **99.4 ±0.6 (0.6%)** and clears the threshold, so it is published. The
`temp-only` row moved 112.4 → **116.6** and is superseded for the reason below.

**Why the held cell was noisy, and what it exposed — three findings, none of them "we averaged more":**

1. **Not early EOS.** Every completion in the re-measure ran exactly 63 tokens, so the short-completion
   hypothesis this page raises elsewhere does not explain it. Ruled out by recording token counts
   rather than by assumption.
2. **Not a binary change.** The re-measured goinfer figure is ~3.7% above the original. Running the
   *original* binary (`686c9f8`) and the current one back-to-back **today** gives **116.5 vs 116.4** —
   identical. So the shift is session-to-session machine state, not code. Absolute sampled numbers on
   this box carry ~3.5% between-session drift; treat cross-session comparisons of them accordingly.
3. **The original phi3 row paired numbers from two different sessions.** Its goinfer and Ollama cells
   were measured by separate scripts, not interleaved — so its `1.12×` verdict combined a goinfer
   number carrying that drift with a peer number that did not (Ollama moved only 125.8 → 125.6 and
   121.8 → 121.4 across the same gap). The corrected, same-session verdict is **1.08×**. This is
   exactly what the page's own "interleaved cell by cell" rule exists to prevent, and it was skipped
   for these two rows.

   **The same defect applied to the gemma3-1b rows** (the same add-on script measured the peer side
   alone for both models), so those were re-measured too. There the impact was small — 1.12 → 1.13 and
   1.28 → 1.30 — which is the honest result: the construction flaw was real in both rows, and it
   happened to bite one and not the other. That is precisely why it cannot be judged from the output;
   a cross-session pairing is not visibly wrong, it is just not a measurement of the engines.

*What changed since the previous sampled numbers:* goinfer's `top_p` figure on qwen0.5b went
92.8 → 190.3 while the peer moved 266.6 → 266.2, i.e. the deficit went **2.87× → 1.40×** and the
change is entirely ours (P2b's parallel normalization, `686c9f8`).

### Greedy decode by KV depth

| depth | goinfer 0.5B | Ollama 0.5B | goinfer 1.5B | Ollama 1.5B |
|---|---|---|---|---|
| 128 | 318.9 | 269.4 | 217.6 | 195.4 |
| 512 | 243.2 | 269.6 | 181.9 | 176.6 ᵃ |
| 2048 | 244.0 | 266.4 | 157.5 | 179.5 |
| 3900 | 200.1 | 259.8 | 122.3 | 174.3 |

ᵃ High variance (spread 27.0), reproducing the same instability seen in the previous campaign on
v0.32.5 (146–182 over ten runs). Treat as indicative.

**Depth stops at 3900 because of `cudaCtxCap = 4096`** (the constant is declared near the top of
`cuda/resident.go`; the line number is deliberately dropped here — this archive must stay readable
without re-indexing a live source file), the resident KV
capacity — not because 3900 was chosen as an interesting depth. 8k/16k/32k are unmeasurable on the
resident path until that cap is configuration-derived.

### Per-segment coefficients (µs per KV position) — and the step is a KERNEL SWITCH, not depth

| model | segment | goinfer | Ollama |
|---|---|---|---|
| qwen0.5b | 128→512 | +2.542 | −0.007 |
| qwen0.5b | 512→2048 | −0.009 | +0.029 |
| qwen0.5b | 2048→3900 | +0.485 | +0.051 |
| qwen1.5b | 128→512 | +2.349 | +1.419 |
| qwen1.5b | 512→2048 | +0.554 | −0.060 |
| qwen1.5b | 2048→3900 | +0.987 | +0.090 |

KV bytes/position (K+V, **f32** — the resident cache is f32): **0.5B 24.0 KB**, **1.5B 56.0 KB**.

goinfer's 128→512 coefficient is ~2.4–2.5 µs/pos on BOTH models despite different layer counts and
head dims, which a genuine per-position attention cost cannot do. A 6-cell A/B attributes it:

| depth | split-KV ON (default) | split-KV OFF | |
|---|---|---|---|
| 128 (below the 256-key gate) | 314.9 ±3.6 | 319.2 ±1.3 | equal — control |
| 512 | 239.7 ±4.4 | **290.0 ±5.6** | OFF **21% faster** |
| 2048 | 242.5 ±7.9 | **249.3 ±1.8** | OFF faster |

**Split-KV attention costs the 0.5B ~0.72 ms/token at 512 and ~0.11 ms at 2048.** It engages at
`splitkvMinKeys = 256` (the constant as it stood at `686c9f8`; replaced in §B6), inside the 128→512 segment, which is exactly why the
step is fixed-size and layer-count-insensitive. With it off the 0.5B coefficient is an ordinary
+0.820 / +0.366 µs/pos and the anomaly disappears.

The crossover in that constant was characterized on the **1.5B** ("break-even at 256, a clear win
from 384+"); it is **default-on for every model**, and on the 0.5B it is a net loss at both measured
depths. Note the published 0.5B depth curves — including the previous campaign's — were measured with
split-KV on and carry this cost. **→ Followed up and fixed in §B6 below; the constant was wrong on
its own geometry too.**

**Not launch latency.** CUDA graphs measured 1.01× on a fitting model (§10 / `docs/cuda-graphs-investigation.md`),
so this ~0.7 ms is small-kernel **execution** overhead. Do not re-propose graphs for it.

### On Ollama's depth behaviour — do not attribute it to flash attention

Earlier text on this page explains Ollama's flat depth curve as flash attention. **These rows were
measured with `OLLAMA_FLASH_ATTENTION:false` (the v0.32.6 default) and the 0.5B curve is still flat**
(0.029–0.051 µs/pos across 512→3900). So flash attention is not the explanation for the 0.5B, and the
honest statement is narrower: Ollama is **flat at depth, not across the whole range** — on the 1.5B it
shows the same 128→512 step goinfer does (+1.419 µs/pos), so part of that step is likely shared and
not goinfer-specific. Any mechanism claim beyond that is unmeasured. §D4's flash-attention
explanation should be read with this 2026-08-09 caveat rather than trusted as-is.

## v0.11.0 release qualification (2026-08-10) — go/no-go vs these anchors

> **⚠ RETIRED 2026-08-31 — withdrawn, not re-measured, and the decision is costed below.** It was
> measured before the 2026-08-25 re-anchor (driver `595.58.03`, Nobara 43, kernel
> `7.0.5-200.fc43`) and may not be quoted as current.
>
> **Why withdrawn rather than re-anchored.** This was a go/no-go for a tag that shipped on
> 2026-08-10, and its numbers were **never independent**: the table says so itself — "the §B6/§B7
> (CUDA) and §B3 (Metal) anchors below **are** the v0.11.0 numbers", carried by code-identity. Both
> of those anchors have since been re-anchored on the current stack (§B6.3, §B7.1). Re-running this
> leg would produce a fresh no-regression statement about a three-week-old tag, derived from
> anchors that are already current. There is no claim it would make quotable that §B6.3/§B7.1 do
> not already make.
>
> **⛔ AND THE FORWARD-LOOKING CLAUSE BELOW IS VOID — this is the load-bearing half of the
> retirement.** The table said it "doubles as v1.0's sweep **iff** the code delta between the two
> tags stays zero". **It has not stayed zero, by four orders of magnitude:** v0.11.0 → HEAD is
> **802 commits, 467 Go files changed, +48,477 / −2,066 lines**. Anyone reusing this table as v1.0's
> qualification would be qualifying a release against a codebase that no longer exists. v1.0 needs
> its own sweep; this table cannot stand in for one.

The v0.11.0 tag's delta from the last code commit (`6edd1ca`) is **docs-only** (README wording), so no
resident cell's numerics changed and the §B6/§B7 (CUDA) and §B3 (Metal) anchors below **are** the
v0.11.0 numbers — the sweep is a no-regression confirmation, not a re-measure.

| backend | qualification at the tag commit | verdict |
|---|---|---|
| **Metal** (M1 Pro) | re-run on device at the tag: snapshot golden **byte-identical**, LayerB **per-kernel parity vs CPU** (attention / gemvW8A8 / gemvW4A8 / rmsnorm / rope / swiglu), C11 argmax=full-logits, NaN-cosine gate — all green | **GO** — decode numerics unchanged |
| **CUDA** (RTX 2070 SUPER) | carries over from the §B6/§B7 anchors by **code-identity** (the tag path is byte-identical to their measurement commit); the box confirms via the CUDA tier of `go run ./cmd/gate gpu` before the tag is final | **GO (carry-over)** — box re-confirm is the box's step |

No regression vs anchors on either backend (code unchanged; Metal gates green). This table doubles as
v1.0's sweep **iff** the code delta between the two tags stays zero (v1.0 is planned as a data/docs-only
delta — see the v1.0 doc). **Judgment call flagged:** the CUDA half is carry-over-by-code-identity, not
a fresh box run; the strongest no-regression guarantee is that the code did not change, but a box
The GPU gate's CUDA pass is the owner's confirmation before the tag ships.


---

## §B6 — superseded by §B6.3

## B6 — Split-KV decode attention, re-gated (2026-08-09, P6a)

> **RE-ANCHORED 2026-08-27 on the Nobara 44 / driver `595.91.07` stack. The ratios did not move.**
> Re-measured with `scripts/bench_splitkv.py` — a harness that did not exist when the table below
> was made; the original 48 cells were driven ad hoc into a scratchpad that has since been cleared,
> so this section could be read but not reproduced. Raw cells and logs are now committed beside it:
> `b6-splitkv-force-6fa487f.json`, `b6-splitkv-bb42106.json` (+ `_run.log` each) in
> `docs/measurements/`. The re-anchored tables are in **§B6.3** below; the 2026-08-09 table that
> follows is kept as the record it was, on driver `595.58.03` / Nobara 43.

§B5 above flagged the split-KV gate as a follow-up. It turned out to be worse than "characterized on
one model": **the shipped constant `splitkvMinKeys = 256` was wrong on the geometry it was
characterized on, and the rule it encodes cannot be right for all geometries in any form.**

**Method.** 48 cells: 4 resident geometries × 6 KV depths × {split-KV ON, split-KV OFF}. Each cell is
a freshly started `serve` (so the arm is a process-level setting, never a mid-process toggle), int4
q4_K_M, greedy, `max_tokens=64`, one warm request discarded, then 2 blocks of 8 requests; the number
is the **client-timed inter-token rate from the first streamed token** (decode-only — prefill and TTFT
excluded), reported as the mean of the two block means with the block spread. Prompts are
**token-calibrated per depth** (repeated `" the"`, verified against `usage.prompt_tokens`) so a "512"
cell really attends ~512 keys. Anchor: goinfer `686c9f8`, driver 595.58.03, RTX 2070 SUPER (40 SMs).
OFF arm = `GOINFER_SPLITKV_ATTN=0`. Raw cells: `scratchpad/p6a.json`.

**Ratios are ON ÷ OFF; >1 means split-KV wins.** The 128 column is below the old gate on every model,
so it is the control — it should read ~1.000, and it does.

| geometry | nH | nKV | hd | L | 128 | 256 | 512 | 1024 | 2048 | 3900 |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder-0.5b | 14 | 2 | 64 | 24 | 1.023 | **0.839** | **0.819** | **0.869** | **0.955** | 1.197 |
| qwen2.5-coder-1.5b | 12 | 2 | 128 | 28 | 1.007 | **0.941** | **0.939** | 1.078 | 1.191 | 1.280 |
| gemma3-1b (win 512) | 4 | 1 | 256 | 26 | 0.989 | **0.890** | **0.909** | **0.919** | **0.941** | 1.084 |
| phi3-mini (MHA) | 32 | 32 | 96 | 32 | 1.001 | 0.993 | **0.969** | **0.919** | **0.815** | **0.754** |

Bold = the old default was a **live regression** there. The old gate fired from 256 up, so every bold
cell at 256+ was being paid in production. Worst single cells: **0.819** (0.5B at 512, −18%) and
**0.815 / 0.754** (phi3 at 2048 / 3900, −19% / −25%).

**Two separate defects.**

1. **The characterization overstated itself ~3–4× on its own geometry.** `TestSplitKVCrossover`
   measured the **1.5B** and reported "break-even at 256, a clear win from 384+". The 1.5B *loses* at
   256 and 512; its real crossover is in (512, 1024]. The cause is methodological, not arithmetic:
   that test times a tight in-process `ForwardArgmax` loop and takes **best-of-3 minimum**. Both
   choices flatter split-KV against real serving — the loop hides the per-token CPU dispatch a real
   request exposes, and best-of-min favours the higher-variance arm, which is consistently ON (spread
   3.6–6.4 tok/s vs OFF's 0.1–0.6). The test now carries this caveat in its docstring.
2. **A one-geometry constant was applied to every model.** qwen0.5b does not break even until ~2560
   (localized: 2560 → 1.019, 3072 → 1.061); phi3-mini **never** crosses over.

**Why the gate is a lookup and not a formula.** Split-KV buys occupancy and pays for it in DRAM:
`attn_batched` launches nH blocks and keeps the whole score row in **shared** memory; split-KV
materializes an nH×nWin f32 score array in **global** memory and touches it three times (scores
writes, softmax reads+writes, vsum reads) to fill the SMs that nH blocks leave idle. So

    net(nWin) ≈ (A − B)·nWin − 2·nLayers·T_launch

with A growing with the occupancy deficit (it needs nH ≪ SM count) and B growing with nH. A > B gives
a crossover; **A < B means split-KV never wins and the deficit widens with depth** — which is exactly
phi3-mini's monotone 0.993 → 0.754 (nH=32 on a 40-SM part: almost no deficit to recover, and the
largest score array of the four). This is why no formula fits: a formula gate has the shape
**"ON iff nWin ≥ f(geometry)"**, which always predicts ON wins at sufficient depth. **phi3 falsifies
that shape, not merely its constants.** Two candidate laws (`nLayers/(nH·hd)` and `nLayers/(nKV·hd)`)
additionally underpredicted the 0.5B crossover by ~2×. Shipped as a measured per-geometry table with a
"never" class and a conservative default, cited beside the constants in `cuda/resident.go` and pinned
by `TestSplitKVGate_measuredGeometries`.

**Asymmetric loss.** Firing early costs up to 18–25%; firing late costs a few percent, because OFF's
slope near the crossover is mild. Every threshold is therefore rounded **up**, and an unmeasured
geometry gets the conservative default rather than an extrapolation.

**A structural bug fixed at the same time: the gate tested the wrong quantity.** It compared the raw
position `nKeys`, but the kernel's work is set by the **effective attended span** `nWin` (window-clamped).
A sliding-window layer never attends more than `window` keys, so gemma3-1b's window-512 layers were
taking the split path at a 512-key span — its loss regime — at *every* depth past the window, which is
why its curve above is a mixture and stays below 1.0 out to 2048. The gate is now evaluated **per
layer on `nWin`**. Both arms are byte-identical, so a layer changing arms mid-request as `nWin` grows
is safe by construction.

### B6.1 — What the new default recovers (measured on the fixed binary)

Same harness and anchor; `DEF` is the shipped default, `OFF`/`ON` are the forced arms. `old DEF` is
the previous default (= the forced-ON number, since the old gate fired at every depth here).

| model | depth | old DEF | new DEF | forced OFF | forced ON | recovery |
|---|---|---|---|---|---|---|
| qwen2.5-coder-0.5b | 512 | 240.0 | **286.6** | 282.0 | 240.0 | **1.19×** |
| qwen2.5-coder-0.5b | 2048 | 240.0 | **250.2** | 250.8 | 240.0 | **1.04×** |
| qwen2.5-coder-1.5b | 512 | 182.2 | **195.0** | 194.1 | 182.2 | **1.07×** |
| qwen2.5-coder-1.5b | 1024 | 182.3 | **182.6** | 169.1 | 182.3 | 1.00× (see below) |
| gemma3-1b | 512 | 152.7 | **167.7** | 168.0 | 152.7 | **1.10×** |
| gemma3-1b | 1024 | 147.4 | **161.5** | 163.5 | 147.4 | **1.10×** |
| gemma3-1b | 2048 | 147.3 | **155.6** | 155.9 | 147.3 | **1.06×** |
| gemma3-1b | 3900 | 154.5 | **163.0** | 142.4 | 154.5 | **1.055×** |

Three things to read off this table:

- **The gate still engages where split-KV genuinely wins.** The qwen1.5b 1024 row is unchanged by
  design: `new DEF 182.6 ≈ forced ON 182.3`, and both beat `forced OFF 169.1` by 1.08×. This is not a
  blanket disable — 1024 is exactly that geometry's measured threshold, and the gate fires there.
- **Below threshold, `new DEF ≈ forced OFF`**, which is the expected no-op result.
- **The gemma3 3900 row beats *both* uniform arms** (163.0 vs 154.5 all-split and 142.4 all-single).
  That is the per-layer `nWin` gate doing the thing only it can do — the global-attention layers
  (nWin = 3900) take the split path while the window-512 layers stay on `attn_batched`. **No per-model
  gate can reach this point, whatever constant it uses**, because the right answer differs *between
  layers of the same model at the same position*.

### B6.2 — Anchored depth rows, re-measured on the fixed binary

The §B5 "Greedy decode by KV depth" rows above **stay as published** — they are the honest record of
what `686c9f8` did. These are the same cells on the P6a binary, default settings, same harness and
anchor. Ollama columns are unchanged (v0.32.6, `OLLAMA_FLASH_ATTENTION:false`) and are repeated only
so the comparison is readable.

| depth | goinfer 0.5B `686c9f8` | goinfer 0.5B **P6a** | Ollama 0.5B | goinfer 1.5B `686c9f8` | goinfer 1.5B **P6a** | Ollama 1.5B |
|---|---|---|---|---|---|---|
| 128 | 318.9 | 320.9 | 269.4 | 217.6 | 218.5 | 195.4 |
| 512 | 243.2 | **286.6** | 269.6 | 181.9 | **195.0** | 176.6 ᵃ |
| 2048 | 244.0 | **250.2** | 266.4 | 157.5 | 157.0 | 179.5 |
| 3900 | 200.1 | 200.8 | 259.8 | 122.3 | 122.1 | 174.3 |

ᵃ Same high-variance caveat as §B5.

**Depths where the gate decision did not change re-measure unchanged** — 128 (below threshold either
way), 2048 on the 1.5B and 3900 on both (the gate fires in both the old and new schemes). That is the
no-collateral-perturbation check: this was a selection change, and only the cells whose selection
flipped moved.

**One peer comparison changes sign, in goinfer's favour.** At 512 on the 0.5B, `686c9f8` read 243.2
vs Ollama's 269.6 — Ollama ahead by 1.11×. The corrected number is 286.6, i.e. **goinfer ahead by
1.06×**. The 1.5B at 512 goes from 1.03× to **1.10×** ahead. Both were self-inflicted by the gate, not
engine differences.

**The §B5 sampled/G11 rows are unaffected and were not re-run.** Those cells use a 129-token prompt,
so their deepest `nKeys` is 193 — below even the old 256 gate. They never paid this cost, so no
sampled figure and no README row moves because of P6a.

**NOT DEVICE-PORTABLE.** The occupancy term scales with SM count and every cell here is one 40-SM
Turing part. On a much wider GPU, nH=32 would be starved and phi3's "never" would not hold. Re-measure
per device class; do not rescale these on paper. `GOINFER_SPLITKV_MIN_KEYS=<n>` now overrides the
threshold on a stock binary (0 ⇒ always split) so re-characterization no longer needs a rebuild —
needing one is part of why a refuted number survived.


---

## §B7 — superseded by §B7.1

## B7 — Deep context: 8k/16k/32k decode (2026-08-09, cap-raise leg)

> **RE-ANCHORED 2026-08-27 — see §B7.1 below, and read the direction before quoting the ratio.**
> goinfer's deep-context decode did **not** move: at the depth-matched 32000 cell it reads **39.1**
> against the **39.0** recorded here. The peer did — Ollama is **40% slower** at that same depth
> (128.4 vs 215.9). So the headline ratio improving from **Ollama 5.54× to 3.28×** is the peer
> moving, not goinfer closing a gap, and quoting it as a goinfer win would be wrong. Note the peer
> is a DIFFERENT VERSION on the two anchors (v0.32.6 then, v0.32.5 now). The rows below stay as the
> `595.58.03` / v0.32.6 record they are.

§B5 recorded that every depth curve this project had published stopped just under 4096 because
`cudaCtxCap` was a compile-time constant — infrastructure, not a chosen depth. The cap is now
configuration-derived (`-ctx`, `ca29d6c`), so these are the first cells past it.

> **SCOPE — cells deliberately NOT measured, and why.** The **1.5B at 32000 was skipped**, as was any
> cell past 32000. This was a decision, not an omission or a silent truncation. The 1.5B/32000 goinfer
> cell alone costs 45–70 minutes (six 32k prefills on the larger model) and no decision turns on it:
> the per-position coefficient had already plateaued, and the 0.5B 32000 pair — kept precisely as the
> bend-detection probe — confirmed no second regime change past 16k. A louder version of the same
> answer was not worth the hour. **`-ctx 32768` is separately and functionally verified** by the
> cap-raise gates (resident build at 32768, VRAM +1570 MiB against a predicted +1568, clean fail-fast,
> `checkCap` at the new cap) — the skipped cells are throughput numbers, not a capability question. A
> future campaign can fill them against this same anchor.

**Method.** Same anchor and discipline as §B5/§B6: fresh server per cell, interleaved engine by
engine, same GGUF both sides, token-calibrated prompts verified against `usage.prompt_tokens`,
client-timed inter-token rate from the first streamed token (**decode-only** — prefill and TTFT
excluded). goinfer `ca29d6c` at `-ctx 32768`; Ollama **v0.32.6** with `num_ctx 32768`,
`OLLAMA_FLASH_ATTENTION:false`, and the `ollama ps` CONTEXT column captured per cell (it read `32768`
on every one). **One protocol difference, deliberate:** a 32k prefill costs orders of magnitude more
than the decode being measured, so deep cells use fewer requests with more decode tokens each
(`num_predict` 128, 1 warm + 2×2) instead of the shallow harness's 17 completions. Run-to-run spreads
came out at 0.0–0.7 tok/s, so the smaller sample did not cost precision. Deepest cell is **32000, not
32768**: prompt + chat template + the 128 generated tokens must fit inside the 32768 cap.

**Decode tok/s by KV depth** (the shallow rows are §B6.2's, same anchor, repeated for continuity):

| depth | goinfer 0.5B | Ollama 0.5B | | goinfer 1.5B | Ollama 1.5B | |
|---|---|---|---|---|---|---|
| 128 | 320.9 | 269.4 | goinfer 1.19× | 218.5 | 195.4 | goinfer 1.12× |
| 512 | 286.6 | 269.6 | goinfer 1.06× | 195.0 | 176.6 | goinfer 1.10× |
| 2048 | 250.2 | 266.4 | Ollama 1.06× | 157.0 | 179.5 | Ollama 1.14× |
| 3900 | 200.8 | 259.8 | Ollama 1.29× | 122.1 | 174.3 | Ollama 1.43× |
| 8192 | 124.4 | 259.1 | **Ollama 2.08×** | 80.7 | 164.0 | **Ollama 2.03×** |
| 16384 | 70.6 | 239.7 | **Ollama 3.39×** | 48.3 | 154.4 | **Ollama 3.20×** |
| 32000 | 39.0 | 215.9 | **Ollama 5.54×** | *skipped* | *skipped* | — |

**These deep cells measure the CORRECTED kernel selection** (`2693dce`): the split-KV decode attention
is now gated per geometry and per layer, so the 0.5B takes the split path only at ≥3072 effective keys
and the 1.5B at ≥1024. Every cell here at 8192+ therefore *is* on the split-KV path for both models.
**Do not diff these against any pre-`2693dce` depth curve** without accounting for that — the old
default engaged split-KV from 256 keys up and was a net loss on most geometries below ~2560.

### Per-segment coefficients — µs per KV position

| segment | goinfer 0.5B | Ollama 0.5B | goinfer 1.5B | Ollama 1.5B |
|---|---|---|---|---|
| 128→512 | +0.971 | −0.007 | +1.436 | +1.419 |
| 512→2048 | +0.330 | +0.029 | +0.808 | −0.060 |
| 2048→3900 | +0.531 | +0.051 | +0.983 | +0.090 |
| 3900→8192 | +0.713 | +0.002 | +0.979 | +0.084 |
| 8192→16384 | +0.748 | +0.038 | +1.015 | +0.046 |
| 16384→32000 | +0.735 | +0.029 | *skipped* | *skipped* |

**There IS a regime change, and it completes — it does not compound.** goinfer's per-position cost
rises from **+0.330 µs/pos** mid-range to a **plateau of ~+0.74** (0.5B) and **~+1.0** (1.5B), reached
by roughly 8k. The 16384→32000 probe is what establishes the plateau rather than assuming it:
**+0.748 → +0.735 is flat.** So deep decode is **linear in depth with a large fixed coefficient**, not
superlinear. That distinction matters — a superlinear curve would mean deep context is unreachable in
principle; a linear one with a bad constant means it is an optimization target with a predictable cost.

Against the peer that constant is a flat **~25× penalty** (0.735 vs 0.029 on the 0.5B; 1.015 vs 0.046
on the 1.5B). The 5.54× throughput gap at 32k is that constant integrated over depth, not an
accelerating divergence.

### B7 control — the cap change did not move the shallow numbers

> *Renumbered in the 2026-08-31 move: this subsection was headed `B7.1`, colliding with the
> unrelated **§B7.1 re-anchor** that remains on the live page. Only the heading changed.*

Two controls, because the obvious one is not sufficient. The deep table's coefficients are computed
against the shallow rows, so it is not enough that the **default** cap still reproduces them — the
deep cells were measured at `-ctx 32768`, and if merely *allocating* a 32k KV slowed shallow decode
(larger buffers, worse locality) the anchor would be wrong and every coefficient with it. So the
shallow depths were re-measured on the cap binary (`ca29d6c`) in **both** arms:

| model | depth | anchor (`2693dce`) | `ca29d6c` default | `ca29d6c` `-ctx 32768` |
|---|---|---|---|---|
| 0.5B | 128 | 320.9 | 312.7 ᶜ | 309.3 |
| 0.5B | 512 | 286.6 | 286.6 ᶜ | 284.2 |
| 0.5B | 2048 | 250.2 | 249.9 | 248.7 |
| 0.5B | 3900 | 200.8 | 199.5 | 199.0 |
| 1.5B | 128 | 218.5 | 216.0 | 216.7 |
| 1.5B | 512 | 195.0 | 192.1 | 193.6 |
| 1.5B | 2048 | 157.0 | 156.8 | 156.9 |
| 1.5B | 3900 | 122.1 | 121.8 | 122.0 |

Every cell reproduces within **2.6%**, most within 1%, in both arms. So the cap change is
allocation-only as claimed, **and** running with a 32k cap costs nothing at shallow depth — the deep
table's anchor is sound.

ᶜ **These two cells were re-run, and the reason is worth recording.** In the first pass they read
**151.0** and **258.7** — a −52.9% and −9.7% "regression" that appeared *only* in the default arm
while the same cells in the `-ctx 32768` arm were clean. A real regression cannot be arm-specific in
that direction. They were cells 1 and 2 of the run, launched immediately after the deep sweep's Ollama
server was killed, and Ollama holds a model resident for minutes after its last request: the first
cells were contending for the GPU with a process that was still unloading. Re-run on a verified-idle
device they read 312.7 and 286.6. The lesson is the harness's, not the engine's — **a fresh-server
protocol does not protect against the _previous_ engine still holding the device**, and interleaved
peer benchmarking makes that a standing hazard rather than a one-off.

### The deep gap is NOT DRAM bandwidth — and the alternative is falsified, not assumed

Each decode token attends the whole KV cache, so the bytes it must read are known exactly from the
cache geometry (§B5: **24.0 KB/position** on the 0.5B, **56.0 KB/position** on the 1.5B, f32, K+V):

| model | depth | engine | KV MB read/token | GB/s | % of 448 GB/s peak |
|---|---|---|---|---|---|
| 0.5B | 16384 | goinfer | 393.2 | 27.8 | **6.2%** |
| 0.5B | 16384 | Ollama | 393.2 | 94.3 | 21.0% |
| 1.5B | 16384 | goinfer | 917.5 | 44.3 | **9.9%** |
| 1.5B | 16384 | Ollama | 917.5 | 141.7 | 31.6% |

**goinfer moves KV at 6–10% of peak bandwidth at depth**, and **Ollama is ~3.2× faster reading
identical bytes.** Whatever bounds deep decode here, it is not DRAM traffic — which is why a
byte-count reduction (KV quantization) cannot be the CUDA speed lever; see P4 in
`docs/plan-still-slow.md`.

*The competing read model is ruled out, not waved away.* If each GQA query head re-read its own KV
head's bytes (×7 on the 0.5B, ×6 on the 1.5B), Ollama would sit at **147–190% of peak — physically
impossible**. So the bytes are read ~once. Even under that falsified upper bound goinfer never exceeds
59% of peak, so the conclusion survives either model. This is arithmetic from measured throughput plus
known geometry, **not an ncu profile** — a profile could refine the number, not flip the sign.


---

## §B9 — STALE relocated README tables

## B9 — tables relocated from the README front page (2026-08-27)

> **⚠ STALE — measured against Ollama v0.32.6 on the pre-2026-08-25 stack.** Not a current claim.
> §B8 above supersedes the greedy decode rows with the same models on the re-anchored stack, and
> its tables carry the peer columns these ones do. Kept because two things here exist nowhere else:
> the **sampling-configuration table** (goinfer's non-greedy paths against the peer's), which §B8
> does not cover, and the v0.32.6-era decode figures the sampling rows were taken beside.
>
> Relocated verbatim while shortening the README; nothing was re-measured, re-derived or edited.

### Compared with Ollama v0.32.6

> **⚠ The current peer anchor is [docs/benchmarks.md](benchmarks.md) §B8** — re-measured
> 2026-08-26 on driver 595.91.07 against **Ollama v0.32.5**, greedy, three models, four depths,
> 33/33 cells. The rows in this section are against **v0.32.6** on the old stack and are kept
> because they carry the sampled configurations and the per-cell annotations §B8 does not have.
> **Do not mix the two**: different peer version, different OS. §B8 records that Ollama itself
> reproduced its 2026-08-09 greedy numbers to within 0.7% across the upgrade, which is why the
> goinfer-side deltas there are attributable.

Secondary, and annotated — read the absolute numbers above and the cgo-free property first. Same
measurements as the tables above, with the peer measured **identically**: both engines driven over
their own HTTP server, client-timed inter-token rate, **interleaved cell-by-cell with a server
restart between cells**, the **same weights** on both sides — verified **per tensor** by
`scripts/gguf_same_weights.py`, *not* by file md5 (see below) — sampling sent
explicitly to each. Peer: **Ollama v0.32.6** (`OLLAMA_FLASH_ATTENTION:false`, its default), except
the one row footnoted ᵇ, which is carried from the earlier v0.32.5 campaign and labelled as such.

**Greedy, by KV depth** (re-measured 2026-08-09, goinfer `2693dce` vs Ollama **v0.32.6**,
`OLLAMA_FLASH_ATTENTION:false`, `num_ctx` verified per cell, interleaved with a server restart per
cell):

| context | goinfer 0.5B | Ollama | | goinfer 1.5B | Ollama | |
|---|---|---|---|---|---|---|
| 128 | 320.9 ±0.0 | 269.4 ±0.4 | goinfer 1.19× | 218.5 ±0.9 | 195.4 ±0.0 | goinfer 1.12× |
| 512 | 286.6 ±2.4 | 269.6 ±0.3 | **goinfer 1.06×** | 195.0 ±0.8 | 176.6 ±27.0 ᵃ | goinfer 1.10× |
| 2048 | 250.2 ±1.0 | 266.4 ±1.4 | Ollama 1.06× | 157.0 ±0.6 | 179.5 ±0.3 | Ollama 1.14× |
| 3900 | 200.8 ±0.0 | 259.8 ±0.2 | Ollama 1.29× | 122.1 ±0.2 | 174.3 ±0.1 | Ollama 1.43× |

ᵃ The peer's rate in this one cell varied widely (spread 27.0, and 146–182 across ten runs in the
previous campaign) — wider than the gap between the engines. Treat that cell as indicative only.

**What changed from the previously published rows, in both directions.** The 0.5B 512 cell **changes
sign** — it read `Ollama 1.06×` and is now `goinfer 1.06×`; the 0.5B 2048 gap narrows from
`Ollama 1.11×` to `Ollama 1.06×`; the 1.5B 512 lead is essentially unchanged (1.11× → 1.10×). Going
the other way, the **1.5B 3900 deficit widens slightly, 1.42× → 1.43×**, and 128/2048 on the 1.5B and
128/3900 on the 0.5B are unchanged. The improvements are goinfer's own regression being removed
(`2693dce`), not the peer moving; the peer columns are a fresh v0.32.6 measurement, which is why they
differ slightly from the v0.32.5 numbers these rows previously carried.

Still ahead at short context, behind at long, the gap widening with depth: Ollama's flash attention
holds nearly flat (269 → 260 on 0.5B) while goinfer decays (321 → 201).

> **Why not md5, and why these rows are pending a refresh (2026-08-25, v0.15.0).**
>
> **The md5 guarantee these tables used to claim was unsatisfiable and never held.** `ollama create`
> repacks a GGUF in a different **tensor order**: metadata keys and values identical, tensor names,
> shapes and types identical, every **offset** different. So the file hash — and any tail slice of
> it — differs even when the weights are bit-identical, for every model, always. It is replaced by
> a per-tensor comparison at each file's own offsets (`scripts/gguf_same_weights.py`), which is what
> fairness actually rests on and which **does** pass: **339/339**, **339/339** and **291/291**
> tensors identical for the three models now used.
>
> **The numbers above predate this release's CPU and attention work and are NOT re-measured for
> v0.15.0.** A 2026-08-22 run at `5b040f2` against the same Ollama v0.32.6 reproduced the 1.5B row
> and the depth-3900 rows closely, but read **0.5B @128 = 284.3 tok/s (goinfer 1.08×)** against the
> **320.9 (1.19×)** published here. One cell moving while its neighbours hold is more often a
> measurement artifact than a regression, and it was **not** confirmed by a second run — so neither
> figure is being promoted over the other here. Then the A1 attention restructure (2.4–2.8×), the
> W4A8 row4 repack and the int4-mode LM head all landed, which moves the goinfer column again.
> **Treat this section as v0.14.0-era until a fresh interleaved run is done at the v0.15.0 tag.**
> Method and raw data: `docs/measurements/bench-peer-v0.15.0-2026-08-22.md`,
> `docs/measurements/mac-cpu-decode-vs-ollama-2026-08-22.md`.

**By sampling configuration, 128 context** (re-measured 2026-08-09, goinfer `686c9f8` vs Ollama
**v0.32.6**, `OLLAMA_FLASH_ATTENTION:false`, `num_ctx` verified per cell):

| configuration | goinfer 0.5B | Ollama | goinfer gemma3-1b | Ollama |
|---|---|---|---|---|
| greedy (`temperature 0`) | 318.9 | 269.4 | — | — |
| `temperature 0.8` + `top_k 40` | 268.8 ᵇ | 284.7 ᵇ | — | — |
| `temperature 1.0`, no truncation (goinfer's default) | 219.2 | 269.0 | 131.7 ᵈ | 149.1 ᵈ |
| `temperature 0.8` + `top_p 0.95` | 190.3 | 266.2 | 115.2 ᵈ | 149.6 ᵈ |

ᵇ `top_k` row carried from the previous campaign (Ollama v0.32.5) — not re-measured in this pass.

ᵈ gemma3-1b cells **re-measured 2026-08-09** with both engines interleaved in one session (the
original pair was measured by separate scripts, which is not a valid engine comparison — see
`docs/benchmarks.md` §B5). The verdicts barely moved (1.12× → 1.13×, 1.28× → 1.30×).

**Where this leaves sampled decoding.** goinfer is **1.08–1.40× behind** the peer under sampled
configurations, down from 2.1–2.9× before the parallel-normalization work (`686c9f8`): the qwen0.5b
`top_p` figure went 92.8 → 190.3 tok/s while the peer was unchanged (266.6 → 266.2). Greedy and
`top_k` remain the fastest paths — see the sampling note above. phi3-mini's previously held `top_p`
cell has been **re-measured and published** (99.4 ±0.6, a 0.6% spread against the 5% threshold —
Ollama 1.22×), together with its `temp-only` row, as a fresh same-session interleaved pair;
`docs/benchmarks.md` §B5 records the re-measure and what the original row got wrong.

Absolute tok/s are **not** comparable across the CUDA and Metal sections — that would compare two
graphics cards, not two engines. Method, hardware and history:
[docs/benchmarks.md](benchmarks.md).
