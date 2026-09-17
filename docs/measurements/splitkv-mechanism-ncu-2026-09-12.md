# ncu on both geometries: the GATE's stated reason is refuted; MY mechanism is not established

Profiled 2026-09-12, nobara-pc. Pre-registration: `splitkv-mechanism-ncu-PREREGISTERED.md`, written
before profiling. `attn_batched` at **M=1 decode**, split-KV OFF, depth 3900, both geometries, binary
`serve-cuda-1f224682`, driver `595.91.07`. 16 launches each, grid `(32,1,1)` verified on both sides.

| metric | mistral-7b (split-KV WINS) | phi3-mini (LOSES) | phi3/mistral |
|---|--:|--:|--:|
| Achieved occupancy | 12.68 % | 11.34 % | 0.89x |
| Theoretical occupancy | 50.00 % | 50.00 % | 1.00x |
| Active warps / SM | 4.05 | 3.63 | 0.90x |
| **DRAM throughput** | **26.88 %** | **67.83 %** | **2.52x** |
| Memory throughput | 28.16 % | 67.83 % | 2.41x |
| L1/TEX throughput | 56.19 % | 35.36 % | 0.63x |
| L2 throughput | 26.59 % | 17.07 % | 0.64x |
| Compute (SM) throughput | 12.91 % | 8.14 % | 0.63x |
| Duration | 275.8 µs | 327.6 µs | 1.19x |

## Verdict 1 — the gate's own justification is REFUTED

`cuda/resident.go:222-221` says: *"at/above this many query heads the single-block kernel **already
fills the device**, so split-KV is pure cost."* That is the entire stated basis for `splitkvMaxHeads`.

Measured at nH=32 on both models: **achieved occupancy 12.68% and 11.34%**, 32 blocks on a 40-SM
part, ~4 active warps per SM against a theoretical 50%. **The device is not close to filled by
either.** This is the same starvation the split-KV design was built to fix, and it is present in
exactly the geometries the gate excludes on the grounds that it is not.

This was the pre-registered SECOND check and it is independent of any mechanism: the two models have
identical nH, identical block count and near-identical occupancy, yet opposite measured signs. So
whatever decides the sign, it is not device fill — and the comment that says it is should be rewritten
whether or not the rest of this file holds up.

## Verdict 2 — my proposed mechanism FAILS its own pre-registered bar

The mechanism was "phi3-mini is nearer memory-pipe saturation, so extra blocks cannot help it." The
pre-registration operationalised that as **max(L1TEX, L2, DRAM) >= 1.5x** — deliberately max-of-three,
because `task-prefill-attention.md:60` had measured this engine's attention as L1TEX-saturated with
DRAM idle, so the saturated pipe could not be assumed in advance.

    mistral max = 56.19% (L1TEX)   phi3 max = 67.83% (DRAM)   ratio 1.21x  <  1.5x

**AMBIGUOUS → PARKED**, by the letter of the rule. The two models are not saturated in the same place:
mistral leans L1TEX (56%) with DRAM idle at 27%; phi3 leans DRAM (68%) with L1TEX at 35%. Taking the
max of each, they are much closer than the bar required.

**The DRAM-specific number is 2.52x and points exactly where predicted** — and I am not going to
promote it to the verdict. The pre-registration says in as many words: *"do not reach for a third
metric post hoc to rescue the story."* Choosing DRAM-only after seeing that DRAM is where the
difference landed is precisely that move. The mechanism is **suggested more strongly than before and
established less than the bar demanded**, and those are different things.

## What the numbers do support, stated conservatively

A coherent picture, not a confirmed mechanism: GQA (4:1 here) has four query heads sharing each KV
read, so the same attention work moves ~1/3 the KV bytes per key (1024 floats vs 3072). mistral shows
that as **high L1 reuse (56% L1TEX) with DRAM idle (27%)**; phi3, reading a distinct KV pair per
query head, shows **DRAM at 68%**. A kernel at 27% DRAM has headroom for more concurrent blocks; one
at 68% has much less. That is consistent with every measured ratio, and it is also the kind of
after-the-fact story this campaign has already banked three of — which is why it is written here as
consistency and not as a finding.

## What would settle it

A DRAM-specific prediction, pre-registered **before** the run, on a third nH >= 24 geometry: if the
mechanism is real, a GQA model should land near mistral's DRAM figure and an MHA model near phi3's,
and the split-KV sign should follow the DRAM number rather than nH. That is a clean test because the
metric is now named in advance instead of chosen from a menu.

## Method notes — one trap cost a full profile

The first phi3-mini profile was **invalid and nearly believed**: it captured grid `(32, 512, 1)` —
PREFILL launches, not decode. Cause: `useAttnFused` requires hd in {64,128} and phi3-mini is **hd=96**,
so its prefill *declines* the fused kernel and falls back to `attn_batched` — the same kernel name
decode uses. mistral (hd=128) takes `attn_fused` for prefill, so its first `attn_batched` launches
were already decode. Profiling "the first 16 launches of `attn_batched`" therefore meant different
things on the two models. Fixed with `--launch-skip 320` (chunk 512 x 32 layers ~ 256 prefill
launches) and verified by asserting the captured grid is `(32,1,1)` on both sides before comparing.
Those discarded numbers read DRAM 70.67% / occupancy 77.16% — a plausible-looking table that would
have compared phi3's prefill against mistral's decode.

Temporary driver `cuda/zz_ncu_mechanism_driver_test.go` was used to give ncu a deterministic
decode-at-depth and has been deleted. Prompt was synthetic (repeated token ids): valid for kernel
grid/access-pattern/volume, which do not depend on token values, and not quotable as end-to-end tok/s.
