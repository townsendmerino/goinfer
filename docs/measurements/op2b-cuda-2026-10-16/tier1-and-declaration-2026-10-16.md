# OP2b on CUDA: gate F, tier 1 and the declaration (2026-10-16)

Registered in `docs/tasks/task-forward-architecture-2026-10.md` §4.7, "OP2b on CUDA". nobara-pc, RTX 2070 SUPER (8,192 MiB; the desktop holds the rest, 7,235 MiB free before each load), driver 595.91.07, GPU runs under the timing lock, models read from `~/models` and `~/goinfer-bench` (never `/srv/models`). Branch `op2b-cuda-tier1-2026-10`, on step 2d (PR #81). **For the owner before merging.**

## Gate F, the resident's own plan

`TestNemotronResidentCUDA_fitPlan` loads each bundle once and alone and reads the build's plan. Logs: `gateF-nemotron-h-4b.log`, `gateF-nemotron-9b.log` (and the 9B bundle build, `prequant-9b.log`) under `~/goinfer-logs/op2b/`.

| model, load | result |
|---|---|
| **4B**, plan's default | resident, ctx 16,384, 1 KV slot of 512 MiB, Mamba state 98.5 MiB, **3,243 MiB free after** (the load took 3,992 MiB) |
| 4B, `ResidentContext` 16,384 | resident, 3,375 MiB free after |
| 4B, 65,536 | resident, KV 2,048 MiB, 1,833 MiB free after |
| 4B, 131,072 (the model's maximum) | refused by the build: 4.29 GB of KV against 4.06 GB free beside the weights, 384 MB reserved |
| **9B**, plan's default | resident, planned ctx 4,529, **final 4,314**, KV 134.8 MiB, Mamba state 165.8 MiB, **373 MiB free after** (the load took 6,862 MiB) |
| 9B, 16,384 | refused: 0.54 GB of KV against 0.54 GB free (plus the reserve) |
| 9B, 65,536 and 131,072 | refused |

**Against the predictions committed before a resident existed (§4.7):**
- **The 4B: held.** "Admitted whole at 16,384 or more with 3 GiB or more of headroom": admitted at 16,384 and up to 65,536, 3.17 GiB left at the default. **A detail was wrong:** it priced the KV at f16, 16 KiB a token, 256 MiB at 16,384; the resident's KV is f32 (the family is not in the f16 policy), 32 KB a position over 4 attention layers, 512 MiB. The Mamba state predicted at about 88 MiB plus the windows is 98.5 MiB.
- **The 9B bundle's size: held.** Predicted 6,935 MiB for Metal's, "the CUDA counterpart within a few percent": the CUDA bundle is 7,273,734,558 bytes, 6,937 MiB.
- **The 9B's admission: wrong.** Predicted "refused, or admitted only at a context under 2,048; falsified if admitted at 4,096 or more." It is admitted at 4,314. The prediction priced the context, the state and the scratch against the 810 MiB left after the weights as if the KV were large; at 32 KB a position 4,314 positions is 135 MiB, and the card holds a resident whole with 373 MiB to spare. Per the registered rule the whole 9B therefore joins tier 1 as a further cell.

## Tier 1

`TestNemotronResidentCUDA_tier1`: the resident against the CPU model it was built from at int4, 32 steps of greedy lockstep (a fixed prompt, then the CPU's argmax), positions from 2 graded, G2-R's bars (min logit cosine at least 0.88, argmax equal on at least half the steps, no NaN row, no all-zero row). The dense Qwen2.5-Coder 1.5B runs first through the same harness in the same session: under 0.95 the session is void.

| cell | min cosine | argmax | verdict | CUDA step (recorded, not graded) |
|---|---|---|---|---|
| control, dense Qwen2.5-Coder 1.5B | 0.997580 | 29/32 | PASS (each of the 4 sessions) | 4.2 ms |
| **Nemotron-H 4B, whole** | **0.993996** | 30/32 | **PASS** | 11.5 ms (87.3 tok/s) |
| Nemotron-Nano-9B-v2, layers 0, 1 and 14 | 0.990851 | 25/32 | PASS | 2.6 ms |
| **Nemotron-Nano-9B-v2, whole** (token table at int4, context 4,096) | **0.953892** | 28/32 | **PASS** | 21.1 ms (47.4 tok/s), first step 0.04 s |

Notes on the 9B whole cell: its first attempts were VOID, not FAIL, and no guard was bypassed. Loaded from the GGUF with the token table at int8 it was refused (the table did not fit beside the rest); with the table at int4 on both sides (as serve loads by default, and as the bundle with `e4h` is baked) the plan chose 12,353 positions and the build refused it by 8 MB, because a guard-pinned context is not trimmed; with an explicit 4,096 positions, an ordinary `-ctx`, it ran. The CPU side used the same options. The 4B and the slice ran at their defaults. For comparison, Metal's whole 9B read 0.929161, 27/32.

## The declaration

CUDA declares `ssm`, `non-gated-mlp` and `non-gated-relu2` (`decoder/features.go`), int4 only. It rests on the tier 0 gates (K, D, I from step 1; M, S, G, P, the int8 decline and the block kinds from steps 2b to 2d, each with its planted defect), gate F above and tier 1 above.

- **Matrices:** `docs/hardware-matrix.md` regenerated: Nemotron-H reads resident on CUDA; Granite-4.0-H stays on the CPU on its runner-shape gate (only "missing: `ssm`" goes); GPT-2's CUDA row loses `non-gated-mlp` and keeps `learned-pos` and `non-gated-gelu-tanh`. The admission golden lists CUDA for `nemotron_h` and `granitemoehybrid`; the over-claim test's CUDA list grows by the three.
- **`gate identity -backend cuda` against main:** **28 IDENTICAL, 0 within tolerance, 1 DIFFERENT, 14 not run.** The one is `nemotron_h` at int4: it runs resident now and ran on the CPU in the base, max |diff| 0.00954 in the first logit row, argmax equal on all 16 steps. The gate sends it "to its reference gates", which are the ones above. The 1,864 attention digests are equal.
- **Heavy-group table:** `cmd/gate/heavy_groups.go` classifies the Nemotron checkpoints (a `nemotron-h` group, large).
- **Banner:** the real `serve` binary on the 4B, no test hook: `decode path: cuda-resident (int4)`, `fit: 16384-token cap needs 0.5 GB KV (weights 3.1 GB ...)`, `KV plan: 1 conversation x 16384 positions`, and a greedy reply that opens correctly ("Paris is the capital of France."). That checkpoint has no chat template in this loader (`chat: raw (no template)`), so the reply runs on past its sentence; a second identical request reuses nothing (`reused=0`, a recurrent family continues only by exact extension). The prefill path line was misleading ("needs int4 or int8 projections"): it now says the Mamba-2 mixer has no batched kernel, and a test pins it.
- **Not in this build:** Granite-4.0-H, Nemotron-3-Nano's MoE blocks, int8int8, a batched prefill for recurrent families (a prompt runs one forward per token), a speed gate. The sequential prompt's time to first token was 171 ms for 14 tokens on the 4B.
