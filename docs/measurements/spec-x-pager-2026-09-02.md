# Does speculative verify break the expert pager? — results

**Pre-registration:** `spec-x-pager-prereg-2026-09-02.md`, written before any arm ran; its R1–R4
thresholds are unedited. Read it first. Box: nobara-pc, RTX 2070 SUPER 8 GB, driver 595.91.07,
taken idle 2026-09-02 16:12 (16 MiB GPU, load 0.02). Logs: `~/bench-logs/spec-x-pager/`.

## The answer

**The hypothesis is refuted, and the reason is not that the effect is small.** On CUDA, the set of
models that both NEED the expert pager and are PERMITTED to speculate is **empty**. Speculation and
expert paging do not fight over the slot budget here; they never meet.

Four independent gates produce that, and no two are the same mechanism:

| gate | where | refuses |
|---|---|---|
| batched verify declines for MoE | `cuda/prefill.go:169` (`r.moe \|\| r.gemma4Moe`) | **every** MoE, block-drafted spec |
| rollback unsafe: recurrent | `decoder/forwardn.go:146` (`Recurrent`) | Gated-DeltaNet / Mamba-2, n-gram spec |
| rollback unsafe: windowed | same (`SlidingWindow > 0`) | Gemma-3/4, Mistral, Phi-3, n-gram spec |
| MLA not resident on CUDA | `cuda/backend.go:96` | DeepSeek-V2/V3, Moonlight, Kimi — never reach the pager at all |

Measured, per venue:

| model | needs pager | block spec | n-gram spec |
|---|---|---|---|
| qwen3.6-35B-A3B (int4 .giw) | yes | DECLINED — `arch needs the sequential path (moe/gemma4moe)` | DECLINED — recurrent (Gated DeltaNet) |
| gemma-4-26B-A4B (int4 .giw) | yes | see note below — the drafter cannot even be ATTACHED | DECLINED — `SlidingWindow > 0` |
| DeepSeek-V2-Lite Q4_K_M | would | n/a — no paired drafter exists | **resident DECLINED**: `arch needs unimplemented feature(s) [mla]` |
| GLM-4.5-Air Q2_K | yes | — | **not testable here**: its int4 bundle is 67 GB against a 62 GB box, so it cannot be host-pinned |

DeepSeek-V2-Lite is the case worth naming, because it is the one that would have answered the
throughput question: MoE, MLA, neither recurrent nor windowed, 10.3 GB against 8 GB of VRAM. It
would pass `specRollbackSafe` — INFERRED from its config (no `sliding_window`, not recurrent), not
observed, because the guard is never reached. It never gets the chance: CUDA has no MLA resident path, so it falls to
the staged/CPU path and has no pager to stress.

### The gemma-4-26B confirmation could not be completed, and the reason is a fourth way this combination fails

Attaching the block drafter to the 26B **OOMs**:

```
NewBlockSpec: cuda: executor job panicked: cuda: device allocation failed
(typed-len, 15859712 bytes): cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY
```

The expert cache auto-sizes to measured free VRAM at load (`capSlots`, here building 31 slots/layer
against a request of 32 — honoured, then capped), leaving `slotMarginBytes` = 384 MiB of headroom.
It has no knowledge that a drafter is about to be attached, so the cache claims the VRAM first and
the drafter cannot fit in what remains — a 15.8 MB allocation fails.

So on this venue `--drafter` with `--moe-cache-experts` does not reach the MoE verify decline at
all; it dies at attach. Note that 384 MiB is flagged in `cuda/resident.go` as the only UNMEASURED
constant in that path, and this is a case where it is provably too small for a real combination of
flags. **Recorded as an observation, not chased** — the block path declines for every MoE anyway, so
the drafter would have been useless had it attached. It is a third independent route to the same
conclusion, and it means the gemma4 row confirms the block decline by inspection of the same
`r.moe || r.gemma4Moe` branch rather than by its own measured run.

## R1–R4 against the pre-registered rules

**R2 — is it H-paging? REFUTED.** The instrument's validity condition held exactly: the `off` arm
read **8.0000** distinct experts per staging event at every rung, on the nose of topK=8, as the
router picking top-8 distinct requires. No speculative arm ever raised it, because no speculative
arm ever ran. The field report's mechanism — a width-K verify presenting K positions' routing at
once and overflowing a decode-tuned slot budget — **cannot occur on this path**: MoE verify is
refused outright on the block path, and where it would fall back to `ForwardN` it walks position by
position, so the pager sees decode-shaped traffic at any K.

**R1 — is the regression real? NOT EVALUABLE, and that is a result rather than a gap.** No arm
produced a paired ratio because no speculative arm produced a token. A "no regression" verdict here
would have been false in the way that matters: nothing regressed because nothing ran.

**R3 — alpha vs paging? NOT EVALUABLE.** No rounds, so no realized alpha.

**R4 — config fix or structural? STRUCTURAL, measured.** The block decline is identical at 16, 32
and 64 slots/layer. It cannot be a slot-budget effect: `prefillStaticDecline` tests the ARCH and
returns before any slot arithmetic. Tuning N per speculation width would fix nothing.

## The off-arm baseline (35B, paged, the only arm that ran)

Wall-clock normalized to 64 emitted tokens; spread is WITHIN-prompt, averaged over prompts.

| slots/layer | mean s | median s | within-prompt spread | dist/stage | pager hit% | misses/stage |
|---|---|---|---|---|---|---|
| 16 | 4.871 | 4.386 | 27.6% | **8.0000** | 46.8% | 4.255 |
| 32 | 4.065 | 3.721 | 29.5% | **8.0000** | 62.8% | 2.979 |
| 64 | 3.309 | 3.251 | 30.5% | **8.0000** | 78.9% | 1.688 |

More slots buy reuse as expected — hit rate 46.8% → 62.8% → 78.9%, misses/stage 4.26 → 2.98 → 1.69,
and 32% of wall clock from 16 to 64. Nothing here is a speculation result; it is the denominator the
speculative arms would have been measured against.

**dist/stage is 8.0000 at every rung**, which is the pre-registered validity condition (R2) and the
refutation in one column: the pager's per-stage demand is pinned at topK and does not move with slot
depth, prompt, or anything else. The within-prompt spread of 27–31% is NOT run-to-run noise — it is
prefix reuse making repeat 1 and 2 of a prompt cheaper than repeat 0 (which pays full prefill), the
same feature the section above shows to be incorrect on this family. Treat these wall-clock figures
as an order-of-magnitude denominator, not as a tuned benchmark row.

## FOUND IN PASSING, AND MORE URGENT THAN THE ITEM: resident prefix reuse is wrong on recurrent families

`3358e6ba` (2026-09-02 15:53, "prefix reuse on the resident KV") produces **silently wrong output
on Gated-DeltaNet models**. Reproduced on qwen3.6-35B-A3B, greedy, temperature 0, single process,
idle box.

`cuda/pager_determinism_test.go` runs the same prompt five times and A/Bs the feature:

| arm | positions per run | result |
|---|---|---|
| reuse ON (default) | 71, then 49, 49, 49, 2 | **run 1 diverges from run 0 at token 0** — and every repeat differs from the last (248068 → 262 → 309 → 285), degrading to a 1-token reply |
| reuse OFF (`GOINFER_NO_RESIDENT_REUSE=1`) | 71, 71, 71, 71, 71 | **5/5 identical**, byte-for-byte, hit rate identical at 64.9% |

Both prompts show it; the second diverges at token 0 on all four repeats. `gen.Err()` is nil
throughout — there is no error anywhere, which is exactly the failure mode the feature's own doc
comment warns about: *"A wrong prefix match produces confidently wrong output with no error
anywhere."*

**Mechanism.** `residentReuseLen` (`decoder/resident_reuse.go`) gates only on
`GOINFER_NO_RESIDENT_REUSE` and the token-id match. It has **no recurrent-state exclusion**, and the
rest of the tree already knows this family cannot do this: `decoder/deltanet.go` says a Gated
DeltaNet's conv ring and matrix state are "NOT position-truncatable (why qwen3_5_moe falls back from
prefix reuse / speculative)", and `decoder/forwardn.go`'s `specRollbackSafe` refuses the same family
for the same reason. The resident path re-zeroes that state only at `pos == 0`
(`cudaResident.Forward` / `ForwardNoLogits`) — which a reused prefix never reaches. So generation N
decodes from generation N−1's tail state.

The feature's three stated safety rules are about matching the right PREFIX; none of them covers a
family whose state cannot be rewound to that prefix at all. The commit's premise — "the resident
cache is POSITIONAL, so truncate costs nothing" — is true of KV and false of recurrent state, and
the family selector never asks which one the model has.

**FIX APPLIED, and the feature is kept.** `residentReuseLen` now returns 0 when
`Model.hasRecurrentState()` — so a recurrent model reuses only from position 0, while every
attention-only family keeps the 21.7x agent-turn win. That is most of what an agent loop runs, so a
revert would have cost far more than the guard does. Reuse for DeltaNet needs a state checkpoint at
the reuse point (L-05), deliberately not patched in here. `TestPagerDeterminism/reuse-on` is the
red-to-green gate.

The predicate is deliberately not a new list: `specRollbackSafe`'s inline copy and the new guard both
read one named `Model.hasRecurrentState()`, which reads the dispatch table's `Recurrent` bit. This
was the EIGHTH consumer of that family list to miss a state kind in a week — `hasRecurrentState`'s
own comment already predicted it ("the fourth kind will be added here, once, or it will be missed at
four sites again") — so the durable answer is L-16, deriving it once and reading it everywhere,
which the tree has now argued for on its own.

**Blast radius.** Any repeat or prefix-extending request on a resident Gated-DeltaNet / Mamba-2
model — which is precisely the agent-loop case the commit was written to speed up, and where turn
N+1 is a prefix extension of turn N by construction. It is invisible to a single-generation test,
which is what the existing 26B/35B cache tests are.

## Two findings that are not about throughput

**1. `--drafter` on a MoE passes startup and then silently serves at 1x.** `BlockSpecCapable()`
checks only that the resident implements `ResidentDrafterHost`; it does not check that a batched
verify exists for the arch. So `NewBlockSpec` succeeds, the banner prints `block drafter attached`,
and every request then declines and falls back to plain `Generate` in `internal/serveapp/openai.go`.
`internal/serveapp/blockdrafter.go:13` states the opposite intent in as many words — "IT FAILS
STARTUP RATHER THAN DEGRADING SILENTLY … not a fleet quietly serving at 1x" — with a sampler
mismatch as the ONLY intended per-request exception. An arch-level property is not that exception.
The operator pays the drafter's VRAM and gets no drafting, with nothing said.

**2. `thetaFor` keys the adaptive controller's cost constant on BACKEND NAME, not on whether the
verify is actually batched.** `decoder/spec_adaptive.go:177` returns 0.251 for "cuda", measured by
`cuda/theta_probe_test.go` on qwen2.5-coder 0.5B and 1.5B — both DENSE, where `ForwardN` runs
`prefillCore` and one weight stream covers the block. A MoE gets the same 0.251 while its `ForwardN`
is a per-token loop, which is precisely the condition under which the Metal constant was measured
at ≥1 and set to disable speculation, with the comment "that is the finding, not a workaround for
it." `verifyTheta()` already makes the resident-vs-staged distinction; it does not make the
batched-vs-sequential one, which is the same distinction a level down. Today this is latent — every
CUDA MoE is refused before the controller matters — but it is a live mis-tuning the moment any of
the four gates above is lifted.

**UNMEASURED, and deliberately left that way.** A probe for this was written — the dense
`TestThetaProbe_CUDA`'s method unchanged, so a MoE number would sit directly beside its 0.251 — and
then deleted rather than committed, because it had never been run. A measurement harness that has
produced no measurement is a claim of coverage with nothing behind it, which is worse than the gap
it appears to fill. Whoever picks this up should re-derive it from `cuda/theta_probe_test.go`: seed
a context, time `ForwardN` over a ladder of n truncating back between calls, and read
Theta = slope/T(1). The prediction to test is that T(n)/T(1) is LINEAR in n for a MoE, as it is on
Metal (16.07–16.81 at n=16), which would put Theta near 1.0 against the 0.251 the controller
currently applies.

## What this does NOT say

- **Nothing about Metal.** `ssh franciss-macbook-pro` is refused on port 22, so that venue was not
  measured. Two code-level observations, explicitly unmeasured: Metal's `ForwardN` is a sequential
  loop for every arch, and `ResidentDrafterHost` is implemented only by `cudaResident` and a CPU
  host — so Metal has no block-drafting path at all, and the brief's Metal × block-drafting arm
  does not exist even with access.
- **Nothing about whether speculation WOULD thrash a pager** if a venue existed. The mechanism is
  refuted for the paths that exist here (verify never reaches the pager as a batch); it is untested
  for a hypothetical batched-verify MoE path, which is exactly what Lead 5 would build.
- **No number from the originating field report is used, quoted or compared against.** It is the
  origin of the question and nothing else.

## Reproduce

```
GOINFER_HEAVY_TESTS=1 GOINFER_MOE_CACHE_SLOTS=32 \
  go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestSpecPagerInteraction -v -timeout 90m
```
`GOINFER_SPECPAGER_MODEL` / `GOINFER_SPECPAGER_DRAFTER` select the venue. The demand instrument is
`PagerStageStatsForTest` / `ResetPagerStatsForTest` / `CacheSlotsForTest` in `cuda/testhooks_gen.go`.
