# f is depth-invariant by the pre-registered metric — but my roof model is not, and that is the finding

Ran 2026-09-13. Pre-registration: `splitkv-f-depth-invariance-PREREGISTERED.md`, written before
profiling. ncu `attn_batched` at M=1, split-KV OFF, depth 8000, grids asserted `(28,1,1)` / `(32,1,1)`.

## Result

| model | f = roof/duration @3900 | @8000 | r | ncu DRAM @3900 | @8000 | r |
|---|--:|--:|--:|--:|--:|--:|
| D7 Qwen2.5-7B | 13.5 % | 13.2 % | **0.978** | 14.04 % | **17.84 %** | **1.271** |
| mistral-7b | 25.9 % | 26.4 % | **1.019** | 26.88 % | 27.33 % | 1.017 |

**Pre-registered verdict: INVARIANT.** Both models sit inside [0.85, 1.15] on the metric the
pre-registration defined — f = t_roof / t_actual. The advance prediction held too: D7's kernel at
8000 was predicted "near 540 µs, roughly 2x its 263.9 µs" and measured **554.4 µs**, 2.7% out.

## The divergence, which matters more than the verdict

The pre-registration treated two things as interchangeable because they agreed to 0.5–1.0 pp at
3900: the **computed** roof fraction and the **measured** ncu DRAM throughput. At 8000 they part
company on D7 — 13.2% computed against 17.84% measured, and by the ncu figure D7's r is **1.271**,
outside the band. Read that way the verdict would be SPLIT, not INVARIANT.

Mistral shows no such gap (26.4 vs 27.33). So it is not a general drift: it is specific to the
geometry with the **smallest** KV footprint.

**Why, most likely:** my roof counts only the ideal K and V bytes. ncu measures DRAM traffic
actually moved, which includes sectors fetched and discarded. `docs/task-prefill-attention.md:68`
measured exactly this on a sibling kernel — **21.96% bytes/sector, 4.5x wasted** — because threads
split over keys read at stride kvDim and never coalesce. Waste is a fixed overhead per access
pattern, so it is a larger *fraction* of D7's small KV footprint than of mistral's, and it grows
with the number of keys walked. That predicts the sign and the ordering of the gap, and it is not
tested here.

## Consequence for the "one formula" gate — weaker than it looked

The question this run existed to answer was whether the gate could be a roofline formula computed
from config, replacing the hand-maintained per-geometry table. The honest answer:

- **The sign is stable with depth.** Both models hold their f across a 2x depth change, so a
  geometry's classification does not flip as context grows. That half survives, and it is the half
  that says the benefit growing with depth is attention's rising *share* of the step, not a change
  in the kernel's regime.
- **But the config-computable roof underestimates real pressure, geometry-dependently.** A formula
  gate would need a waste factor that is a property of the access pattern, not of the config — so
  it cannot be computed from `nKV`, `hd` and `nKeys` alone. "One formula plus a depth threshold" is
  not yet supported; "one formula, a depth threshold, and a measured per-geometry correction" is
  most of the table it was meant to replace.

The clean version of the idea is still reachable, but it runs through **fixing the coalescing**, not
through modelling the waste: if the kernel's KV reads were coalesced, actual and ideal traffic would
converge and the formula would predict the measurement at any depth. That is a kernel change with
its own justification (`task-prefill-attention.md` already made it for the prefill twin) and it
would make the gate simple as a side effect.

## What is unchanged

Nothing here touches the three-point ordering result. `splitkvNever` keyed on `nH` is still refuted
at identical nH, D7 still forfeits **+9.94%** at depth 8000 under the shipped gate, and the
occupancy claim in `cuda/resident.go:222` is still false on three geometries. This run was about the
shape of the replacement, not whether a replacement is needed.
