# R11/P20's gemma4 extension, built and measured on the real target: 2.66x at M=512, still 2.26x at M=8012, DEFAULT ON

Follows `p20-expert-major-2026-09-21.md` (the generic-path build, opt-in, small measured win on Mellum2) and `docs/queue-performance.md`'s own "OPEN, ORPHANED" item this closes. Build: `cuda/moe_expert_major_gemma4.go`, wired into `cuda/prefill.go`'s
per-row MoE FFN loop as the `Ly.g4moe` branch, mirroring `moe_expert_major.go`'s mechanism (route every row first, bucket by expert, admit/DMA each distinct expert once, fold in rank order via sequential `residual_batched` launches — see that file's header
for the full bit-identity argument, reused unchanged here) against Gemma-4's own parallel dense‖MoE FFN shape (`gemma4MoeMLPPre/Post`): the expert loop now accumulates into a per-row scratch buffer instead of the shared decode-native `r.g4x2`, and the
remaining join (`normF32(g4x2)` → `x1+=x2` → `normF32` → `x=h+comb` → `*layerScalar`) runs per row afterward — cheap, not the DMA-heavy part, so a plain loop over the batched buffer's row-views is enough.

## Correctness

`TestMoEExpertMajorGemma4CUDA_bitIdentical`, a new tiny fixture `testdata/gemma4-moe-tiny-k3` (`PIN_TOPK=3 PIN_OUT=testdata/gemma4-moe-tiny-k3 scripts/pin_gemma4_moe_forward.py` — topK=2 in the existing `gemma4-moe-tiny`/`gemma4-moe-kv-tiny` fixtures cannot
catch an accumulation-order regression, same reason as the generic path's own k=3 fixture): **0/256 logits differ**, non-vacuity confirmed.

**Mutation results, stated precisely because they differ from the generic path's:** a wrong-rank-weight mutation (value-level, not order) is caught decisively (255/256 logits differ) at M=24. A pure fold-order reversal is **NOT caught on this fixture**, at
M=24 or M=96 (0/256 differ either way) — genuinely checked twice, not assumed; my first attempt at this mutation silently hit the WRONG loop (a routing-bucket loop earlier in the file that happens to share the same `for j := 0; j < r.topK; j++` text) and
produced a false pass I caught by grep-confirming which line actually changed before trusting the result. With the correct loop mutated, the fold-order-reversal mutation is genuinely invisible on this fixture's specific magnitudes and downstream norms —
the same class of result the generic path's own k=3 test showed only partially (482/512, not 512/512): reordering sensitivity is real but not uniform across every numeric pathway. The mechanism is architecturally identical to the generic path's (already
proven order-sensitive there), and the value-level mutation here confirms the (row, rank) bookkeeping itself is exercised and correct — recorded honestly as a null result on this specific check, not hidden.

**One real slip during construction, corrected before it reached the tree:** `PIN_OUT` redirects the checkpoint directory but the pin script's own `OUT` (the golden JSON path) still resolves to the SHARED, tracked `testdata/gemma4_moe_forward_golden.json`
when only `PIN_OUT` is set (its own comment says this is intentional for scratch/uncommitted use) — running it overwrote that real fixture's golden. Caught immediately via `git status`, reverted with `git checkout --`, confirmed clean before continuing. The
new checkpoint directory itself (`gemma4-moe-tiny-k3/`) was never at risk (a distinct path).

## Real hardware: `TestPrefillLongPrompt`, `gemma4-26b-int4.giw` (M26), `-moe-cache-experts -ctx 8192`, RTX 2070 SUPER, driver 595.91.07

| M | off (ms/token) | on (ms/token) | ratio | sequential off | sequential on |
|---:|---:|---:|---:|---:|---:|
| 512 | 50.301 | **18.897** | **2.66x** | 54.343 | 54.342 |
| 2048 | 49.588 | **19.870** | **2.50x** | 54.005 | 53.891 |
| 4096 | 49.494 | **20.705** | **2.39x** | 53.907 | 53.759 |
| 8012 | 50.280 | **22.261** | **2.26x** | 53.557 | 53.429 |

**The sequential control is unmoved (within 0.2-0.3%, noise) at every depth** — expert-major only touches the batched-prefill MoE loop, exactly as scoped; nothing else changed. **Every cell clears R11(b)'s registered ships band (`docs/tasks/red-october.md`:
sequential 46.5-46.7 ms/token baseline, >=1.86x / <=25 ms/token ships) by a wide margin, including at M=8012 where the win is smallest.** This matches the `p20-expert-locality-2026-09-21.md` projection (~2.32x, ~20 ms/token) closely — the projection was
built from real distinct-expert-count data on this exact model, not shape arithmetic, and it held.

## Decision: SHIPPED, DEFAULT ON

`prefillExpertMajorEnabled()` flipped from `!= ""` (opt-in) to `!= "0"` (default on, `=0` opts out) — mirroring the CPU precedent this build mirrors throughout, `decoder/mlp.go`'s P18 (`GOINFER_MOE_EXPERT_MAJOR`, same convention, "an escape hatch and an
A/B handle, not a user setting"). This affects BOTH the generic path (Mellum2, 3.5-4.3%, real but small) and the gemma4 path (M26, 2.26-2.66x) — one flag, both bit-identical, neither measured a regression anywhere, so there is no reason to default one on
and not the other.

## Not established

Byte-level DMA counts (only call counts were the basis for the projection). M35 (Gated-DeltaNet) is out of scope — it never reaches the batched-prefill path at all (`prefillStaticDecline` refuses `r.dnet != nil` for an unrelated, correct reason). A served
peer comparison (Ollama) for M26 prefill was not re-run after this change.
