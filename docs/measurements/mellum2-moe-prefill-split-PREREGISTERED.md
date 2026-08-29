# Is expert-major MoE prefill batching a COMPUTE lever on CPU? — pre-registered

**Written before the arms were run.** Pilot numbers exist (a uniform-id ladder run
19:45–19:53 on 2026-08-28) and are treated as a pilot, not as the record: they were
taken in a different session-slice from the arm they would be compared against, and
session drift on this box is ~3.5%.

## The question, and why the dense profile does not answer it

Two levers are live for long-context prefill and they attack different terms:

| lever | term | status |
|---|---|---|
| f32 attention (`--cpu-fast-attention`) | O(K²) attention | shipped 2026-08-26, **refuses MoE** |
| expert-major batching (`task-moe-streaming.md` Lever 4) | per-row MoE FFN | not started |

The K=8192 profile that says *~70% of long-context prefill is attention* is a **dense
1.5B**. It cannot arbitrate between these two, because the MoE FFN it would be
arbitrating against does not exist in it.

## Design — paired arms, matched on everything but routing

Mellum2, 4-layer slice (`decoder/testdata/mellum-mellum2-slice`), int8int8, M1 Pro,
`forwardLayersN` only (model load and LM head excluded by construction).

The slice is representative **of this model specifically, and the reasoning does not
transfer**: Mellum2's `layer_types` has period 4 (s,s,s,f), so 21 sliding / 7 full —
exactly 25% full_attention — is reproduced exactly by layers [0,4); every layer is
`sparse`; 64 experts, top-8, `moe_intermediate` 896, window 1024. It does **not**
preserve the embedding's share of the total.

**A 4-layer slice is exactly the kind of shortcut that gets reused where it does not
hold.** It is valid here only because the interleave period DIVIDES the slice length.
Take a family whose full-attention interval is 4 but which is sliced at 6, or one
whose first layers are a dense prefix (Laguna, Qwen3-Next), and the slice silently
re-weights the very term under study — an attention-share measurement on it would be
wrong by the ratio of the two mixes, and would look fine. Before reusing this method:
check the period against the slice length, and check that no layer KIND is dropped.
`pin_slice_oracle.py` chooses its own N for exactly this reason and documents it
per family.

Two arms at each K, **interleaved in one session**, alternating:

- **varied** — `ids[i] = (i*131+7) % vocab`. Real routing diversity.
- **uniform** — one repeated id. Degenerate routing: near-identical rows select
  near-identical experts, so the top-8 weights stay in cache.

`uniform` is the control AND the **ceiling** for Lever 4: a chunk whose rows all
select the same experts is precisely what expert-major batching is trying to
manufacture, and it cannot do better than already having it. Attention cost is
content-independent, so it is a matched constant across the pair and the difference
isolates the routing term rather than pooling it.

**Ladder:** K ∈ {1024, 2048, 4096, 8192}. The full ladder is run even if an early
step looks decisive — the 2026-08-28 Metal slot sweep is the standing reason
(a marginal stopping rule encodes an unstated assumption about curve shape, and
there it stopped two doublings short of the only resolvable win).

**Statistic:** `bound(K) = (t_varied − t_uniform) / t_varied`, the upper bound on
Lever 4's compute-side headroom at that K.

## Decision rule — fixed before the numbers

| bound at K ≥ 4096 | verdict |
|---|---|
| **≥ 20%** | Lever 4 is a compute lever; fundable on compute grounds alone |
| **10–20%** | **ambiguous → PARKED**, revisit only with a streaming arm |
| **< 10%** | not a compute lever; justify only on streaming I/O, and measure it there |

A second, independent pre-registration that can disagree with the first: **the
direction of `bound(K)` across the ladder.** If the bound RISES with K, the rule
above is reading the wrong regime and the long-context case needs its own ladder
past 8192 before anything is concluded.

## Scope limits, stated up front

1. **Resident weights, no streaming.** This measures only Lever 4's compute half.
   Its actual motivation in `task-moe-streaming.md` is the streamed case, where the
   same expert can be re-fetched once per row and the win is I/O, not FLOPs. A
   negative here does NOT retire the lever; it relocates the argument.
2. **Slice, not the full 28-layer model.** Absolute tok/s is not a model-level number.
3. **CPU only, M1 Pro.** Says nothing about Metal or CUDA.
4. **The full checkpoint would not fit.** 12 GB at int8int8 against a box already
   holding 8.1 GB of swap at the start of the session; a paging run returns a
   plausible wrong profile, which is the failure class that made the withdrawn G15
   cliff expensive.
