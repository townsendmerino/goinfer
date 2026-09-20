# goinfer task: R6 — the CUDA flash-decode lane: the spike's mechanism, then the kernel, then f16 KV

> Written 2026-09-19 against goinfer `894c9a9a` plus the same day's edits to
> `docs/tasks/red-october.md`. **Box: `nobara-pc`.** Three phases with a written record between
> each; phase 1 builds no kernel.
> **The brief is R6 in [`../tasks/red-october.md`](../tasks/red-october.md), and it is the
> authority** — goal, bands, gates, measurement protocol, record. This prompt is the order of work
> plus the six points of R6's 2026-09-19 amendment, which exist because the brief as first written
> got three things about the spike wrong.
> **Before anything else:** `git pull`, then `grep -n "The depth curves as a line" docs/tasks/red-october.md`
> and `grep -n "Amendment, 2026-09-19" docs/tasks/red-october.md`. If either is missing, the Mac has
> not pushed the edits this prompt depends on — stop and say so.
> **Read first, all of it, in R6's order** (§6 rule 12 — the reading list is not optional), plus
> `docs/measurements/splitkv-q-staging-2026-09-13.md`, §2.2's 2026-09-19 note, and `CLAUDE.md`.

## What this is for

Fitted to `t(K) = F + A·K`, every CUDA depth curve in `benchmarks.md` says the same thing: the flat
term is at parity or ahead of Ollama (0.80× / 0.87× / 0.98× on the 0.5B / 1.5B / 7B) and the cost
per cached position is 13.5× / 5.5× / 10.4× worse. With the peer's slope and its own flat term the
1.5B reads 194.7 tok/s at 3900 against Ollama's 174.2. The whole deep-context loss is that one
coefficient, and R6 is the brief that owns it.

| model | A today (µs/position) | what R6's band means |
|---|---|---|
| 0.5B | 0.499 | — |
| 1.5B | 0.924 | ≥170 tok/s at 3900 (ships) is **A ≤ 0.36**; the 150 park line is A ≤ 0.56 |
| 7B (D7) | 1.784 | ≥50 tok/s at 8000 (ships) — register its A from your own phase-2 "before" |
| phi3-mini | 2.600 | the lane is expected to decline here; see phase 3 |

goinfer's A is flat per *query-head element* (17.8–26.5 ps across all four geometries, MHA
included); Ollama's is flat per *KV byte* (3.0–3.75 ps, 5.9 on its one noisy curve, the 1.5B — within
a small factor of the card's bandwidth roof). **That is not the traffic argument `splitkv-kernel-exploration-2026-09-13.md`
retired — that record stands: the split path already reads 1.06× ideal DRAM traffic.** It is a
statement about what the kernels' time follows, and it is why the group axis below is argued from
issued loads and latency, never from bytes.

## Phase 1 — the mechanism for the parked spike. No new kernel.

The V-sum split reads +40% on its kernel and +16.2% served on D7@8000, and is parked on one number:
KL 1.0585× exact's, inside the 1.05–1.10 band. A re-run needs a mechanism, never a re-roll.

1. **Make the S=1 detector real first.** R6 step 1 asks for S=1 to reproduce the exact arm
   byte-for-byte. As the code stands that check is vacuous: the loader in `cuda/backend.go` and the
   V-sum launch in `cuda/resident.go` both require S > 1, and the gate's test hook sets the same
   field — so S=1 runs the exact `splitkv_vsum` and compares it with itself. Add a test-only way to
   launch `splitkv_vsum_partial` + `splitkv_vsum_combine` at nSplit=1, and a precondition that
   proves they launched (a launch counter; the mirror image of the gate's "630/630 rows
   differing"). Then require raw-bit identity with the exact split path, on a full-causal GQA
   geometry and on gemma3 (windowed, `winStart` > 0), at depths straddling the 128-key tile and the
   chunk boundaries.
2. **Know what the spike is before hunting defects in it.** It has no partial max and no α: scores
   and softmax are the exact kernels, and only the V fold is split — contiguous chunks from
   (nKeys − winStart, nSplit), an ascending combine, then × inv. The defect surface is the chunk
   bounds on windowed layers, the partials layout and its `nH × maxHd × S` sizing against a
   per-layer hd, and the z-grid. The partial-max / α / f16 candidates in R6's original step 1
   belong to phase 2's kernel.
3. **If S=1 is clean, measure the gate's resolving power — that is the mechanism question left.**
   With prompts as the unit (positions inside one teacher-forced prompt share a cache), D7's
   per-prompt KL ratios average 1.057 with a standard error of 0.029 — t = 1.96, two-sided p ≈ 0.08;
   "higher on 8 of 10" is a sign test at p ≈ 0.11. S reads 0.994. So run the null: S ∈ {2, 8, 16}
   beside S=4 and exact, same reference, same prompts — every one of them a blocked tree at least
   as accurate as the sequential fold. If equally valid trees scatter in KL by about as much as
   5.85%, band (c)'s 1.05 line cannot resolve anything at this cell, and *that* is the finding
   (the `task-prefill-gap.md` §3.2 precedent). If every S > 1 sits consistently above exact, the
   excess is real for blocked-vs-sequential here, and KL restricted to the reference's top-k says
   whether it lives in the head of the distribution or its 152k-token tail. Report per-prompt
   ratios with their spread; never a pooled mean alone.
4. **Pre-register before the first cell** (`docs/measurements/vsum-split-mechanism-PREREGISTERED.md`):
   arms, cells, the rule that separates "no resolving power" from "systematic", and its own
   ambiguous → parked band. **Prompt set B only** — check first that no CUDA decode gate has
   consumed it since 2026-09-13; if one has, draw a fresh set and say so. Re-scoring set A is a
   re-roll.
5. **Budget the reference.** D7 at K=8000 cost 3h58m for ten prompts at f32 weights, one worker
   (`GOINFER_CPU_REF_WORKERS=1`; eight workers were on course for ~25 h). The null arms reuse one
   reference, so the cost is one Phase A plus minutes per arm. Detach it (`setsid nohup`, PPID 1),
   progress-log it, archive the log under `~/goinfer-logs/`.

**Output of phase 1:** `docs/measurements/vsum-split-mechanism-2026-MM-DD.md`, a finding either way.
**Owner decision point:** if the finding is "no resolving power", whether band (c) changes is
Francis's call, not yours — write the recommendation, do not amend the rule. Only a found
mechanism unparks the spike.

## Phase 2 — the kernel, split before it is built

**2a. A grouped `splitkv_scores`, inside the exact lane.** With the V-sum split, `scores` is ~70% of
the kernel (218.9 of 318.1 µs at S=4 on D7@8000), and after q-staging it still reads 117.67 cyc of
lg_throttle beside 31.89 of long_scoreboard — two stalls, both paid per loaded element. One thread
per (kvHead, key) that loads the K row once and computes all nH/nKV dots from it cuts issued K
loads G× and amortises each memory latency over G dots. It is **bit-identical** — each dot keeps
its d-order and its `__fmaf_rn` sequence, and scores are independent outputs — so it ships or dies
under `TestSplitKV_bitIdentical` and its gemma3 twin with no fidelity question at all. ncu `scores`
alone, before and after, on D7@8000 and the 1.5B@3900: duration, lg_throttle, long_scoreboard,
occupancy. Pre-register the recanon doc's rule on the `scores` kernel time: **<5% kill, 5–15% park,
>15% keep.** State the cost in advance: G×hd floats of staged q per block, where q-staging's hd
alone cost 12% occupancy; and the block count falls from nH to nKV per key tile, which matters at
shallow depth and is why the existing `nWin` gate stays in front of it. q-staging halved the issued
loads and returned 5.5% because the latency stall rose underneath — a result in the park band here
would be the same story, and is worth recording as firmly as a win.

**2b. `attn_decode_fa`, as R6 step 2 specifies** — one block per (kvHead, keySplit) handling the
whole group, the group's q staged once, `float4` K reads, online softmax per head in registers, V
accumulated per head per lane, one fixed-order combine over splits, S=4 as the starting point,
gated on `nWin` per layer through the existing per-geometry table. Its own `.cu` and `.ptx` (the
isolation pattern `attn_block.cu`'s header explains): no shipped PTX changes. The exact path stays
what ships with the lane off and what every parity gate and spec-decode verify runs.

**2c. Gates, measurement and decision exactly as R6 registers them**, plus one acceptance check
beside the band: fit `t(K) = F + A·K` on {128, 512, 2048, 3900} per model from the
`scripts/bench_peer.py` rows and report A per KV byte for the 0.5B / 1.5B / 7B (and phi3-mini
wherever the lane runs). It should come out flat across geometries, as the peer's does; if it is
still flat per query-head element instead, the kernel is still paying per query head and the band
result needs that sentence beside it. ncu the new kernel before any served number (a stall profile
is per kernel — the 2026-09-13 lesson); the kernel nominates, the served token elects (§6 rule 2).

**2d. Kernel-level gate vectors (amendment item 6).** The served fidelity gate cannot stand in for
a kernel test here: real prompts put many heads' max on the first token, where every later rescale
is ×1. Build `TestAttnFused_vsF16Reference`'s shape for the decode kernel — exact f64 math over the
kernel's own rounded inputs, per head, 1e-3 of max|V| and cosine 0.9999 — on inputs that force the
mechanism: a sharp hot key in the first split, the LAST split and on both sides of a split
boundary; a rising score ramp; a dominating sink and a non-dominating one; a windowed case with
`winStart` > 0; nKeys straddling the 128-key tile and every split boundary; S ∈ {1, 2, 4, 8}.

## Phase 3 — f16 resident KV. Its own pre-registration; not before phase 2's record is written.

phi3-mini is the geometry R6 expects the lane to decline on, which leaves it with no depth lever
from phase 2 at all. It does not need one: it is already byte-bound (`splitkv-mechanism-ncu-2026-09-12.md`
measured 67.83% DRAM on `attn_batched`; the fit independently gives 302 GB/s = 67.5%), and its 1.8×
against Ollama is goinfer's f32 KV against the peer's f16. `plan-still-slow.md` P4's "KV-quant is
not a speed lever" was measured on the latency-bound GQA 0.5B/1.5B, where it holds; it does not
cover a byte-bound geometry, and every GQA geometry becomes one the moment phase 2 works. **Band,
phi3-mini at 3900, served, greedy: ≥70 tok/s ships (today 56.2, Ollama 73.3; halving A reads 79),
62–70 parked, below 62 killed.** Fidelity-gated under the same pooled §3.2 rule. No family is
known to need f32 KV: Metal ships f16 KV for every family, and the one finding that said otherwise
(Gemma, 0.64 against 0.92 cosine) is refuted in `metal/model.go`'s own comment — the crater was the
position-0 K/V compute — although the old claim still stands in the comment above `kv_store_f32`
in `metal/kernels.go`. The family decision is the gate's, behind one precondition: record max |K|
and max |V| per layer on the gate prompts and refuse f16 KV for a family that comes within 2× of
f16's 65504. `attn_batched`, the
split kernels and the FA kernel each need an f16-KV twin. The general trigger for any other
geometry: decode attention reading ≥~50% DRAM under ncu. Halved KV VRAM is a capacity win on the
8 GB card either way — record it, do not lead with it.

## Do not

- Do not re-score prompt set A, and do not amend a pre-registered rule after seeing a result. If a
  rule looks wrong, say so in the record and leave the decision to Francis.
- Do not carry any of this to Metal. R2 is a different bound on different hardware, and the record
  has already paid for that lesson once (§A2-Metal).
- Do not touch the exact paths or re-bake a golden. `attn_batched`, split-KV and the snapshot
  goldens move only with a stated mechanism, never because a number moved.
- Do not regenerate a PTX before proving a no-op regen of that module is byte-identical, and check
  which nvrtc the module was built with first — `splitkv-q-staging-2026-09-13.md`'s provenance
  paragraph records that `decode_splitkv.ptx` was built at 12.9.86 while the 12.6.85 freeze covers
  `moe.ptx`, `glue.ptx` and the gemv family. §6 rule 11 is the default; that paragraph is the
  exception to check.
- Do not quote a kernel-level gain as a served one, or a served one without its peer row from the
  same session.
- Do not benchmark from `/srv/models`. `~/models` only, idle-gated, machine state beside every number.

## Deliverable

Per phase, in this order: the `-PREREGISTERED.md` file committed before its first cell; the dated
measurement record; the `benchmarks.md`, `ollama-chase.md` and `task-decode-splitkv-attention.md`
updates R6's **Record** paragraph lists, in the same commit as the measurement they quote; R6's
status row in `red-october.md` §5. Commit in increments that survive interruption. A negative is
recorded at full value, in the measurement file and in `ollama-chase.md` §10.

<!-- doc-reviewed: 2026-09-19 -->
