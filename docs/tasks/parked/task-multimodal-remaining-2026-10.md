# Task: multimodal support, what is left after S1–S20 (2026-10)

> **Status 2026-11-08: S12-defaults and S10q-paged done (G-S10q-e PASS); every other row is parked.** Split on 2026-10-31 (coordinator) from the
> multimodal-support task doc, which is archived as the record at `docs/completed/task-multimodal-support-2026-10.md`: every phase's design, gates and
> readings, S1 to S20, are there, and each row below names the section it comes from. This doc holds only the open work.
> A row closes here, in the same commit as the work. S10q-paged read on 2026-11-08, so the rest are parked rows and this doc moved to
> `docs/tasks/parked/` that day.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| S12-defaults | done | each WebGPU device vision tower is the default under `--backend webgpu` on a unified-memory Mac; a discrete GPU keeps the CPU tower unless `-vision-device gpu` asks; `-vision-device cpu` keeps the CPU tower | #736, #750 (eb8726c2, 2026-11-07); `multimodal/tower_support.go:DefaultTowerOnDevice`; `internal/app/serveapp/main.go:config.routeVisionTower`; `docs/completed/task-multimodal-support-2026-10.md` "S12" |
| S10q-paged | done | G-S10q-e read 2026-11-08 by day on nobara at main 604f7ddf (the script's `cd` fixed), under the registered rule unchanged: **PASS**, CUDA minus CPU int4 -1.00 points [-3.4, +1.2], every logit finite. Serve is unchanged; both G-S10q-d FAILs stay recorded; re-registering G-S10q-d against HF is the owner's call | `docs/completed/task-multimodal-support-2026-10.md` "S10, Qwen3-VL MoE" (G-S10q-e, "Read 2026-11-08"); `docs/measurements/multimodal-support-2026-10/s10q-qwen3vlmoe/gs10q-e-2026-11-08/` |
| S13-full | blocked | parked: the full peer comparison comes after S10's remaining families, S11 and S15; no bar registered. Trigger: S15 unparked, or Francis wants a published multimodal peer number. S13-lite is done | `docs/completed/task-multimodal-support-2026-10.md` "S13" |
| S15 | blocked | parked (owner, 2026-10-11: "leave video parked"); Gate 0's desk map is done. Trigger: Francis unparks video | `docs/completed/task-multimodal-support-2026-10.md` "S15"; `docs/measurements/multimodal-support-2026-10/s15-video-deskmap-2026-10-07.md` |
| S16-hybrid | blocked | parked: only CUDA declares a hybrid m-RoPE prefill, and only for the dense hybrids, so a Qwen3.5+ image turn prefills and decodes on the CPU on Metal and WebGPU (dense and MoE) and on CUDA for the MoE. Trigger: Francis asks for resident Qwen3.5+ image turns on the Mac | `backend/cuda/prefill.go:cudaResident.HybridMRoPEPrefill`; `docs/multimodal.md` |
| EG2-rest | blocked | parked (Francis, 2026-11-09): EmbeddingGemma 2's video as frames, text before an image, and several images per input. Trigger: a user asks for one of them | `docs/completed/task-embeddinggemma2.md` status block |
| Metal-embed-int4 | moved | to `docs/tasks/parked/task-metal-audit-followups-2026-10.md` owner decision O3 (B-N01, "An int4 head on Metal"), the same item | `internal/app/loadflags/loadflags.go:Flags.embedInt4` |

## Conventions

The record's "Rules for every phase" still apply to any row taken up here: gates registered and committed before the first
measurement, and the tiers and timing rules in `CLAUDE.md`.

<!-- doc-reviewed: 2026-11-08 -->
