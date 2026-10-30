# SB2 memory reads, step 1: the memory-plan seam in cuda's build (2026-10-29, nobara-sb01)

<!-- citations-at: f875b2102 -->

Gates registered in `docs/tasks/task-subsystems-2026-10.md`, "SB2 memory reads, step 1", at `322ea6e9b`, before the code.
G-MP3 was amended twice, each time before the pass it changed: at `6f95a981f` before any after-run, and at `5875b2c8f` after
the 190 MiB pass. Box: nobara, RTX 2070 SUPER, driver 595.91.07. Every device run held the timing lock shared and started with
only the desktop compositor on the card (`*-card.txt`).

## What changed

`buildResident` gains a memory-plan mode, `cudaBackend.planMemory`. It is the build's own planning, run past plan-only's stop:
- the resident's fields;
- the planned layer geometry;
- the context margin;
- the context plan;
- the weights checks (`openForPlan`'s for a MoE build, and the setup job's);
- the KV fit, over the planned layers windowed as the setup job windows them.

It takes the free-VRAM readings supplied in a `memReadings` and returns a `memPlan`. It allocates nothing, opens no device and
starts no executor. The decisions are functions the real build calls too: `weightsFitDecision`, `kvFitDecision`, and
`resolveCtxCapFitFree` (free VRAM through the function it is given).

## G-MP1 (no device)

- `TestWeightsFitDecision`, `TestKVFitDecision` and `TestPlanMemory` pass. `TestPlanMemory` plans `testdata/llama-tiny` with no
  device, and refuses at the weights site or the KV site when that reading is short.
- Layer B-cuda's census matches its golden, and its plant still turns its 42 cells red (`gmp1-census.log`).
- `decoder/` is untouched, so MM1's decision census cannot move.
- vet and staticcheck pass under `cuda goinfer_testhooks` and untagged.
- Under `-tags cuda` alone, staticcheck reports only findings that predate this change. Two exceptions were mine and are fixed
  here: `itoa` and the `mustalloc_test.go` helpers, which #431 left unused under that tag set.

## G-MP2: the real build, before and after

`sb2_memprobe_before_test.go.txt` runs the real `BuildResident` on every layer B-cuda cell that reaches the build. That is
412 cells: 279 build and 133 decline. Each answer records the build's verdict and its memory decisions:
- the planned context;
- the context after the KV fit;
- the KV slots;
- the margin, slack and companion reserve;
- the expert-cache slots.

| run | tree | result |
|---|---|---|
| before A1 (`before-A1.txt`) | `322ea6e9b` | 412 cells |
| before A2 (`before-A2.txt`) | `322ea6e9b` | every decision field equal to A1; the readings move run to run (excluded: they are inputs, not decisions) |
| after (`after.txt`) | this branch | every decision field byte-identical to A1 |

## G-MP3: the plan agrees with the build

On the after tree, the probe (`sb2_memprobe_after_test.go.txt`) feeds the memory plan the readings the real build took on each
cell. It then compares the plan's weights verdict, planned context, context after the KV fit, slots and margin with the
build's.

- **Real readings (`after.txt`):** agreement on all 412 cells. 279 match on every field; on the other 133, the build declines
  statically before any memory decision, and the plan gives the same static decline.
- **G-MP3b at 190 MiB (`after-kvforce.txt`, `after-kvforce-plant.txt`):** agreement on all 412 cells, with and without the
  plant. The plant could not show here. The only cell that can trim, qwen3next-tiny at 16384, has a 238.7 MiB margin, so both
  sides refuse it at 190 MiB. The second amendment moved the figure for this reason.
- **G-MP3b at 245 MiB (`after-kvforce245.txt`):** agreement on all 412 cells. The real build trims qwen3next-tiny from 16384 to
  16320 on three cells (int4, int4/ssm, int4/kv-i8), and the plan trims the same. At kv-f16 the halved KV fits at 16384.
- **Plant at 245 MiB (`after-kvforce245-plant.txt`):** the trim turned off on the plan's side only. Exactly those three cells
  go red (the plan refuses the context the build trimmed), and nothing else changes.

## G-MP4: the MoE pool on the card

A MoE build whose context is planned checks the weights against a reading taken before `moe_route`'s local-memory pool is
paid. The plan then reads after the pool. In this census only qwen3next-tiny plans its context (see below). On its four built
cells, the two readings differ by exactly 138,412,032 bytes (132 MiB) every time. That matches the "about 138 MB retained" in
the KS-H6 notes. A census that feeds a profile's `cuda_free_bytes` to a MoE build's plan reading must subtract this figure.

## For step 2

The tiny fixtures' context windows are mostly under the 4096 floor: `testdata/llama-tiny` has 128 positions and
`testdata/mixtral-tiny` 64. For those, the context plan has nothing to choose and never reads free VRAM, so every dense cell
here plans at the floor. Only qwen3next-tiny (32768) exercises the plan and trim arms. A census meant to exercise them needs
cells with wider windows, or a `-ctx` cell.

## After Gemini's review of #468 (2026-10-29, 17:08–17:16 PDT)

Fixed red-first (`review-red.log`):
- A MoE plan that refuses at `weightsOpen` now records the margin, companion bytes and slot request that check priced.
- The context plan's reading hooks fire for the live reading only (`liveCtxFree`), so a supplied figure no longer echoes into
  the real build's record. Earlier `after*.txt` lines end in a second `plan=` reading for this reason. That was in the
  excluded readings section: G-MP2 compares decisions only, and G-MP3 takes the first reading per site, so neither was affected.
- `TestPlanMemory` fails on a fixture error. `TestPlanMemory_moePlannedContext` plans qwen3next-tiny, `TestKVFitDecision` covers
  the multi-slot trim, and `TestKVFitDecision_trimPlant` gives the plant a committed use.

The probe (`sb2_memprobe_after_test.go.txt`, now the reviewed version) also checks:
- the static decline's text against the real build's;
- on built cells, the companion bytes and the slot request;
- each of the plan's log lines against what the real build printed to stderr.

Re-run on the fixed tree, card clear (`after-r2*-card.txt`):

| run | result |
|---|---|
| `after-r2.txt` | G-MP2: every decision byte-identical to `before-A1.txt`. G-MP3: agreement on all 412 cells. 279 built cells match on every field and log line; the 133 static declines match text for text. No duplicate plan reading. |
| `after-r2-kvforce245.txt` | agreement on all 412 cells; qwen3next-tiny trims 16384 → 16320 on int4, int4/ssm and int4/kv-i8, and the plan prints the build's trim line |
| `after-r2-kvforce245-plant.txt` | exactly those 3 cells red |
