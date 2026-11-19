# E5: accounting coverage at serve's shape, and the 7B's minimum free VRAM with four generations (memory audit F9, F4; 2026-11-18)

One box: nobara, RTX 2070 SUPER 8 GB, driver 595.112.07, desktop session up (`kwin_wayland` holds 19 MiB on the card). Tree:
`origin/main` at `cb6ad797f` (E2's weight arena merged) plus this branch. Checkpoints from `~/models` on NVMe. Every card run held the
exclusive timing lock; load average 0.66 at the first start. These are memory reads; nothing here is timed.

## What was added

- **Qwen3-8B in `accountingModels`** (`~/models/qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf`, on local disk), in the Qwen text models' band.
- **`TestBuildScratchAccounting_serveShape`:** G-M1's cell at the options serve loads with (`serveOptions`: the int4 head, windowed KV,
  four KV slots), on the 7B. The row's KV is now read from the device's ledger: a multi-slot pool holds fewer blocks than slots × the
  per-slot cap, which `kvChargedBytes` would price (the first run read KV 11,536 MiB and a post-KV figure of −8,620 MiB, a check that
  could not fail).
- **`TestConcurrentMinFree_7BServe`:** a read, not a gate (`E5_MINFREE=1`).
- **`TestBuildScratchAccounting_plantedDefect` rewritten.** It failed on `main` since E2 (#1088): its plant priced no slack, and with
  the arena the slack is 0 to 2 MiB, so the plant could no longer be seen (7B residual 5 MiB without the slack, against a bar of the
  290 MiB margin; Gemma-3-4B trimmed 0 positions either way). The plant is now `weightArenaOffForTest`: the build uploads every weight as
  its own allocation while the plan prices the arena, the disagreement G-M1 must catch. `allocSlackOffForTest` is removed.

## Readings

**G-M1 (`TestBuildScratchAccounting`), seven models:** residual +4 MiB on every one (`run1-accounting-and-minfree.txt`). Qwen3-8B:
device before KV 4,326 MiB, Plan 4,321 + slack 1, context 8,192 at one slot, KV 2,448 MiB, post-KV scratch 0 against its 261 MiB margin.

**Serve shape, 7B:** device before KV 3,800 MiB (Plan 3,794 + slack 2: the int4 head, against the gates' 4,022 with the int8 head),
residual +4 MiB; four slots at a 26,112-position cap, 2,916 MiB of KV charged on the ledger; post-KV scratch 0 and first use 0 against
the 290 MiB margin (`run2-plant-and-serve-shape.txt`).

**The plant:** with the weights outside their arena the residual reads +126 (1.5B), +422 (7B), +324 (Gemma-3-4B), +170 (VL-3B), +198
(E2B) and +112 MiB (Qwen3-8B), against +4 on each without it. Every planted row is past its G-M1 band; Gemma-3-4B's build trims the
planned 15,936 positions to 14,784, and Qwen3-8B's 8,192 to 8,128.

**Minimum free VRAM, four generations at once (F4).** The 7B loaded as serve loads it, `EnableResidentConcurrency(4)` = 4: three
generations decoding (64-token prompts, 512 new tokens each), then a fourth arriving with a 2,048-token prompt (512-row prefill
passes). A sampler read `cuMemGetInfo` every 2 ms (5,797 samples over 11.6 s).

| | MiB |
|---|---|
| free before the load | 7,064.9 |
| free after the load (4 slots, 26,112-position cap) | 348.9 |
| **lowest free during the four generations** | **228.9** |
| drop below the loaded figure | 120.0 (the 512-row pass is priced at 117) |
| headroom above the driver's 155 MiB allocation floor | **73.9** |
| the context margin | 290.3 |

No allocation failed. The audit's one-request figure on `main` before E2 was 205 MiB lowest free (50 MiB above the floor); this read is
after E2, whose freed weight bytes went to the KV pool. Not measured: a sampler slower than 2 ms would miss transients shorter than
that, and the scenario is one arrival pattern.

## Bands: a proposal, not changed here

G-M1's live bands are [−16, +64] MiB on the Qwen text models and [−8, +128] on the others. Every residual read today, at both options
and on all seven models, is +4 MiB (+2 to +4 before E2). The smallest planted residual is +112 MiB. Proposal for Francis: **the upper
bound to +16 MiB on every model, the lower bounds unchanged** ([−16, +16] on the Qwen text models, [−8, +16] on the others). +16 leaves
12 MiB above today's reading and catches the smallest plant by 96 MiB. This PR changes no band.

**Adopted 2026-11-18 (Francis, about 15:03 PST, relayed by the coordinator):** the upper bounds are +16 MiB on every model, the lower
bounds unchanged; the plant's 7B clause reads against the same +16.
