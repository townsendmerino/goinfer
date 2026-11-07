# Task: multimodal support, what is left after S1–S20 (2026-10)

> **Status 2026-10-31: open, no session owns it.** Split on 2026-10-31 (coordinator) from the multimodal-support task doc,
> which is archived as the record at `docs/completed/task-multimodal-support-2026-10.md`: every phase's design, gates and
> readings, S1 to S20, are there, and each row below names the section it comes from. This doc holds only the open work.
> A row closes here, in the same commit as the work.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| S12-defaults | done | 2026-11-07 (Francis decided 2026-11-06 21:32): each WebGPU device vision tower is the default under `--backend webgpu` (Gemma 3, Gemma 4, Qwen2.5-VL, Qwen3.5+, Qwen3-VL, GLM-OCR); `multimodal.DefaultTowerOnDevice` is now the declaration, `-vision-device cpu` keeps the CPU towers, `-vision-device gpu` is kept as the same as `auto`, `-vision-quant int8` still picks the int8 Gemma 3 tower. The support table and its test, README, `docs/server.md`, the flag help, a changelog fragment. The v0.25.0 plan's name for this, "the `--vision-int8` default", was wrong | record, S12, "The registered consequence is not applied yet" |
| S10q-paged | open | nobara: Qwen3-VL MoE's paged CUDA arm read FAIL on the two-image request twice (G-S10q-d, G-S10q-d3). The instrument the owner chose, G-S10q-e (the paged path's teacher-forced logits against HF float32), is built; its first run made no reading (c75a283a), and no later reading is recorded | record, S10, "G-S10q-e" |
| S13-full | open | the full peer comparison; S13-lite (one model per box) is done. Not started | record, S13 |
| S15 | blocked | parked (owner, 2026-10-11: "leave video parked"); video is in scope but has no phase work | record, S15 |
| S16-hybrid | blocked | parked: Metal has no hybrid m-RoPE prefill, so a Qwen3.5+ image turn on Metal prefills and decodes on the CPU (text turns are resident) | record, "Where it stands" |
| Metal-embed-int4 | open | `--embed-int4` is on by default everywhere but Metal (owner, 2026-10-10); on Metal it needs an int4 table in the resident, which is not built or scheduled | record, status block, "With the owner" |

## Conventions

The record's "Rules for every phase" still apply to any row taken up here: gates registered and committed before the first
measurement, and the tiers and timing rules in `CLAUDE.md`.
