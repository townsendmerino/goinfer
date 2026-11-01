# Task: multimodal support, what is left after S1–S20 (2026-10)

> **Status 2026-10-31: open, no session owns it.** Split on 2026-10-31 (coordinator) from the multimodal-support task doc,
> which is archived as the record at `docs/completed/task-multimodal-support-2026-10.md`: every phase's design, gates and
> readings, S1 to S20, are there, and each row below names the section it comes from. This doc holds only the open work.
> A row closes here, in the same commit as the work.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| S12-defaults | decision | Francis: the five WebGPU device-tower speed gates read SHIP on 2026-10-12 (median 0.355 to 0.488 of the CPU towers), and their registered consequence, each device tower the default under `--backend webgpu`, is not applied: it edits `decoder/support_table_test.go`, so its timing is the owner's. The towers stay opt-in (`-vision-device gpu`) | record, S12, "The registered consequence is not applied yet" |
| S10q-paged | open | nobara: Qwen3-VL MoE's paged CUDA arm read FAIL on the two-image request twice (G-S10q-d, G-S10q-d3). The instrument the owner chose, G-S10q-e (the paged path's teacher-forced logits against HF float32), is built; its first run made no reading (c75a283a), and no later reading is recorded | record, S10, "G-S10q-e" |
| S13-full | open | the full peer comparison; S13-lite (one model per box) is done. Not started | record, S13 |
| S15 | blocked | parked (owner, 2026-10-11: "leave video parked"); video is in scope but has no phase work | record, S15 |
| S16-hybrid | blocked | parked: Metal has no hybrid m-RoPE prefill, so a Qwen3.5+ image turn on Metal prefills and decodes on the CPU (text turns are resident) | record, "Where it stands" |
| Metal-embed-int4 | open | `--embed-int4` is on by default everywhere but Metal (owner, 2026-10-10); on Metal it needs an int4 table in the resident, which is not built or scheduled | record, status block, "With the owner" |

## Conventions

The record's "Rules for every phase" still apply to any row taken up here: gates registered and committed before the first
measurement, and the tiers and timing rules in `CLAUDE.md`.
