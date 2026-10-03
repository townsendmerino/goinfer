# GLM-OCR vision-tower cost on the Mac CPU: pre-registration

Written and committed 2026-10-02, before any run. The record will be `docs/measurements/glm-ocr-tower-cost-2026-10.md`,
written from the first night's logs.

**Question.** How does the GLM-OCR vision tower's wall time grow with pixel count on the MacBook's CPU? O4 in
`docs/tasks/task-glm-ocr-2026-10.md` reads the answer to choose a default pixel cap for laptops; this is O2's "cost,
measured and recorded, not gated".

**A record, not a gate.** There is no pass or fail and no decision rule.

## What runs

`run-cost.sh`, on the Mac's night queue as `glm-ocr-tower-cost`. It runs aikit's `TestGlmOcrVisionEncoder_costSweep`
at v1.52.0: the wall time of one `GlmOcrVisionEncoder.Forward` (the 24-block, 434 M-parameter GLM-OCR ViT, f32
reference path, `quant=false`, CPU) on random `pixel_values` (seeded, and the cost does not depend on the values) at
three points:

| Point | grid_thw | Patches |
|---|---|---|
| 1 MP | [1, 70, 72] | 5,040 |
| 2 MP | [1, 100, 102] | 10,200 |
| 4.8 MP | [1, 128, 192] | 24,576, the processor's ceiling (6,144 image tokens) |

The whole sweep runs 3 times back to back, one fresh process per point: 9 processes, 9 times.

- **Idle gate:** before each process, `bench_peer.py`'s instant gate: the CPU at most 10% busy over 3 s and no foreign
  timed workload, waiting up to 10 min. Otherwise the job stops with `NOT IDLE`.
- **Amendment, 2026-10-03, before any point ran:** the gate was a 1-min load average of at most 1.0. On the first night
  this Mac sat at 1.7–2.4 with VS Code open and the CPU near idle, and the job stopped `NOT IDLE` before its first point.
  The instant gate is the harness's darwin default since TE1 attempt 5. Nothing else changes.
- **Load record:** the test's own load readout reads `/proc/loadavg` and prints `n/a` on macOS. The script's
  `sysctl vm.loadavg` lines before and after each process are the load record.
- **The stop:** if the first repeat's 4.8 MP point takes over 30 min, the job stops after it and the record holds one
  repeat.

## Reported

- Per point: the median of the 3 times, the min–max range, and all three times. Nothing is averaged away.
- ms per patch at each point. If ms/patch at 4.8 MP is more than 2× ms/patch at 1 MP, the record says the
  attention-quadratic term dominates.

## Provenance the record carries

- Machine: the MacBook (M1 Pro, 16 GB), CPU model, macOS version and Go version, all from the log.
- Checkpoint: `zai-org/GLM-OCR` at revision `2e85a62840ccac27daa451df36c736c4636b8628`, read from `~/models/glm-ocr`
  on the internal SSD (the bench set, never the archive). The record also carries the sha256 of `model.safetensors`
  (2,650,579,464 bytes).
- Code: aikit `v1.52.0` (01928d7a), built once from a worktree at the tag into
  `~/goinfer-bench/glm-ocr-tower-cost-2026-10/`.
- Path: the f32 reference, on the CPU, with random pixels.
- Date, and a thermal note (`pmset -g therm`) before and after each point.
- Logs: `~/goinfer-logs/glm-ocr-tower-cost-2026-10/<date>/`.

## Expected cost and memory

The estimate is a **guess**:
- Nobara's 1 MP point took about 30 s, but that box is not comparable.
- The 4.8 MP point was guessed at 10–20 min, because attention grows with the square of the patch count.
- Queued at 150 min: 3 × (1 + 3 + 15) min, plus gates. The first night's log replaces it.

Peak memory at 4.8 MP should be about 4–5 GB, an estimate from the code, not a measurement:
- About 1.7 GB of f32 weights.
- Activation buffers of patches × hidden and patches × intermediate.
- Attention is fused and tiled (`AttendTileFusedContractExp`), so no patches × patches score matrix is built.

## Not citable

nobara's 30 s at 1 MP ran at a load average of about 14. It is contaminated and must not be cited. This run is the
first quotable cost point.
