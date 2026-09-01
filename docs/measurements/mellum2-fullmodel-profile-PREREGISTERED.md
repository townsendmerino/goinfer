# PRE-REGISTERED — full-model Mellum2 prefill profile at K=8192 (2026-09-01)

**Written before the arms ran. Nothing below is a result.**

## The question, and why it is not the one originally filed

The open item read: *"Confirm why the 4-layer slice doubled the win. One full-model profile at
K=8192 settles the cache-residency hypothesis."* The 97.1% attention share it refers to was
measured 2026-08-28 on a 4-layer slice, int8int8 — and necessarily under **acc64** attention,
because the `--cpu-fast-attention` MoE exclusion was still in force that day (removed
2026-08-29, `66d0a05`).

Since then the default changed twice: f32 prefill attention became the default above 512 tokens,
and today it fans out over heads (1.92× end-to-end at K=4096). So "attention is 97.1% of MoE
prefill" describes a configuration that no longer runs by default. Profiling the old
configuration would settle a historical question about a code path most users no longer take.

**Two arms, one run pair**, which answers both:

| arm | attention | what it answers |
|---|---|---|
| **A** — today's default | f32 + head fan-out | What IS the MoE prefill profile now? |
| **B** — `GOINFER_CPU_FAST_ATTENTION=0` | acc64 | Does attention's share fall from the slice's 97.1% on the FULL model *at the same kernel*? |

Arm B is the cache-residency test: it holds the attention kernel fixed at what the slice used and
varies only model size, which is the comparison the hypothesis is about. Arm A is what the repo
should quote going forward.

## Predictions, recorded so they can be wrong

1. **Arm B's attention share comes in materially below 97.1%** — I predict **70–90%**. If it lands
   at ~97% the cache-residency hypothesis is *not* supported and the slice was representative
   after all, which would be the more surprising and more interesting result.
2. **Arm A's attention share is lower than arm B's**, since A makes attention cheaper without
   touching the weight matmuls. I predict **arm A below 70%**.
3. **Attention remains the single largest bucket in arm A**, even after both speedups.
4. **Arm A is faster in wall clock than arm B** by more than the 1.59× recorded for f32-vs-acc64
   before the fan-out existed.

Prediction 3 is the one that decides whether attention is still the lever. If it fails — if weight
matmul or something else overtakes attention in arm A — the prefill story has changed and the next
lever is elsewhere.

## Decision rule

- Arm B attention share **≤ 90%** → cache residency (or model size generally) is supported as the
  reason the slice overstated; the slice figure is retired to "slice-only, explained".
- Arm B **> 95%** → hypothesis NOT supported; the slice was representative on this axis and the
  2× gap between 3.11× and 1.52× needs a different explanation, which is then the open item.
- **90–95% → AMBIGUOUS, parked**, and explicitly not written up as a confirmation. This band is
  deliberately wide because a profile share is a ratio of noisy buckets, not a timing.

## Method

`decoder/mellum2_prefill_profile_test.go`, the same harness that produced the slice figure, so
the instrument does not change between the two numbers being compared. It calls
`forwardLayersN(..., cpuFastAttention())`, so the env var selects the arm and nothing else differs.
`go tool pprof -top` on the resulting profiles; park samples (`pthread_cond`) excluded, as before —
they are the known idle-M artifact a CPU profiler miscounts.

Full 28-layer checkpoint, `~/models/mellum2-unq`, K=8192, `varied` ids (real routing).

## Stated limitations, before the fact

- **QUANT IS NOT MATCHED TO THE SLICE.** The slice ran int8int8; these run **int4**. The checkpoint
  is 23 GB bf16 (~11.5B params), so int8int8 is ~11.5 GB against 16 GB of RAM with 6.2 GB of swap
  already in use — it would page, and a paging run profiles I/O rather than compute. Arm A and arm
  B are matched to each other, which is what makes their comparison sound; the comparison to the
  slice's 97.1% carries this caveat and must be quoted with it. A quant-matched slice re-run would
  remove it and is not done here (no slice checkpoint is on this machine).
- **Paging voids the profile.** The harness records swap and load before and after. If swap grows
  materially during a run, that arm is reported as void rather than quoted — the M1 Pro run of this
  model in the 2026-08-29 campaign grew swap 8→13 GB and moved +2.65M pages.
- One machine, one model, one K. This is a profile, not a sweep.
- `forwardLayersN` only: the embedding and LM head are outside it by construction, so these shares
  are of the layer stack, not of a full request.
