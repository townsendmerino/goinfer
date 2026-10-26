# SP2b: the drafter's context in prefix snapshots

Registered 2026-10-26 by nobara-sp2, before any code, on the owner's decisions (Francis, 2026-10-25 13:38, option A: SP2b after
the 35B; 2026-10-26 09:30: "move on to SP2b"). Task: `docs/tasks/task-forward-architecture-2026-10.md`, "SP2b".

## The problem, measured

`serve --drafter` re-prefills a repeated or re-sent prompt cold, with the drafter's tap capture on top, where plain `serve`
restores the prefix from a host snapshot. On the essay A/B at depth 4096 that was 18.65 s a request against plain's 0.21 s, wall
0.19× (`docs/measurements/sp2-2026-10/README.md`, "The essay A/B end to end").

Why, read from the tree:
- `BlockSpec.generate` acquires its slot with snapshots switched off (`sc.Off`), and `serve` hands `BlockSpec.GenerateStream`
  the request's context without its KV scope (`loadedModel.drive`), so a speculative turn neither restores nor takes a
  snapshot.
- Its seed is one batched pass (`PrefillSeedArgmax`) with no mark, and a recurrent snapshot can only be taken at a pass
  boundary (`kvBoundaryPlan`; `residentSnapshotPrompt` returns at once on a recurrent model).
- The drafter's context (`residentDrafter`'s K/V, `cuda/drafter.go`) is one per model, not per KV slot. `BlockSpec` keeps it
  across turns only while `Model.resDrafterSynced` says no other writer touched the slot.
- `GenerateStream` reports neither `Generation.PrefillReused` nor `KVSnapRestored`.

## The change

1. **`BlockSpec` takes the request's KV scope** (Keep, Boundary), and `serve` passes it as for plain decode.
2. **A marked seed:** the seed's prefill splits at the scope's mark (the plain path's boundary plan), with the tap capture armed
   over both passes, so the snapshot at the mark holds the target's rows and the DeltaNet state exactly there.
3. **The snapshot carries the drafter's fused context rows for its prefix** (after `fc` and `hiddenNorm`), not the drafter's
   K/V: 16 KiB a position on Qwen3.5-9B's drafter against 48 KiB of K/V (6 layers × 2 × 1,024 × 4 B), and a row's K/V depends
   only on its fused row and position, so a restore replays them through `ExtendContext`.
4. **A restore** in `BlockSpec`'s acquire puts the target's rows and state back (`RestoreKV`), rebuilds the drafter's context
   from the snapshot's fused rows, and marks the drafter synced to the slot. Only snapshots that carry fused rows serve a
   speculative turn: one taken by plain decode (a sampled request on the same model) is declined, and the decline is counted
   (`KVSnapshotStats`, "no drafter context").
5. **`GenerateStream` reports its reuse** (`PrefillReused`, `KVSnapRestored`), as plain decode does.

Not changed: the snapshot budget, its eviction, its 2,048-id threshold, and plain decode's own snapshots.

## Gates

- **G1 — identity (tier 0, then tier 1).**
  - **Tier 0:** on the three small recurrent fixtures with the scripted drafter (`qwen35-tiny`, `qwen3next-tiny`,
    `qwen3.5-0.8b`), a speculative turn restored from a snapshot against the same turn cold: tokens, the Gated-DeltaNet state,
    the target's K/V rows and the drafter's context rows all equal; the restored turn reports `KVSnapRestored` and the reused
    rows. Plants that must go red: (P1) the restore skips the drafter's rows (the drafter drafts from an empty context: tokens
    still equal, so the gate reads the context rows and must catch it there); (P2) the mark is off by one pass (state differs).
  - **Tier 1:** Qwen3.5-9B with z-lab's DFlash (`5fc3b3d4`), w = 4, SP0's 12 code and math prompts behind a 3,800-token
    preamble: each prompt sent twice, the second restoring; tokens equal plain greedy decode and the cold speculative turn,
    and the second turn reports the restore.
  - **Bar:** all equal and both plants red. Any divergence: the first token and the row are named, and nothing merges.
- **G2 — the snapshot's added host bytes (tier 0).**
  - The snapshot's price (`KVSnapshotBytes`) grows by exactly the fused rows: positions × hidden × 4 B. A tier-0 test pins
    the formula and that the store's budget accounting charges it.
  - Stated for the 9B at a 3,840-token mark: 3,840 × 4,096 × 4 B = 60 MiB per snapshot beside the target's rows and state.
  - **Bar:** the stated, priced and charged bytes agree.
- **G3 — served, end to end (tier 2, by day, the timing lock exclusive).**
  - **Instrument:** `bench_peer.py`, `BENCH_ENGINES=goinfer,goinfer_old` (TE5(a): the decision is new ÷ old), both arms
    `serve --drafter -drafter-width 4 -thinking off` on Qwen3.5-9B: `goinfer` a binary at SP2b's head, `goinfer_old` one at
    the main it branches from. Phase B only, depth 3,840, the code and math set (`scripts/prompts-sp2-codemath.json`,
    `Q9:3840`), the harness's 16 completions a cell, which resend each prompt so the snapshot path is exercised; 5 mirrored
    quads, so 10 pairs.
  - **The reading:** per pair, the arm's mean wall time per request, old over new. Recorded beside it: TTFT, decode tok/s, the
    snapshots restored (`/admin/status`), and peak RSS.
  - **Bar: the owner's merge rule** on the wall time per request: median ≥ 1.05× and at least 9 of 10 pairs faster → SP2b
    merges; anything else → it does not, and the record says which way it missed.
  - **Void cells** as `bench_peer.py` and CLAUDE.md define them (load above 8.0; available memory under the fit need plus 2 GB),
    re-run in order and listed.
  - Not measured here: plain `serve` (unchanged arm; the ratio to it is a separate same-session run, if the owner wants it).
- **G4 — agent-style reuse (tier 1, in process).** Two conversations interleaved on one resident, each turn extending its own
  last one (so the other's turn breaks the drafter's sync between them): the speculative turns' tokens equal plain decode, and
  with SP2b each turn restores its conversation's snapshot instead of re-prefilling. **Recorded:** restores and prefill rows
  per turn, before and after. No speed bar: G3 is the speed gate.
- **Stopping rule:** fixed; all cells, no early stop.

## Cost, stated before any run

- **G1 tier 0:** seconds. **Tier 1:** 12 prompts × (one plain and two speculative turns) × 256 tokens at the 9B's ~65 tok/s,
  plus 24 prefills of ~3,900 tokens at the 9B's prefill speed: about 10 min. A fit check first (the 9B with its drafter at
  4,608, as SP2 ran it).
- **G3:** a cell about 100 s plus the old arm's cold 3,840-token re-prefills (#249 measured them); 20 cells: about 45 min.
- **G4:** about 5 min.
- **Total about 65 min,** by day.

## Readings

(none: no code was written)

## STOP POINT (2026-10-26, 10:07 PDT): parked by the owner's wind-down

The gates above are registered (`ad12a015`) and nothing else exists: no code, no run. What is next, in order:
1. **The fused-rows choice first:** confirm a restore's `ExtendContext` replay of 3,840 fused rows costs little beside the
   re-prefill it replaces (the drafter's own projection, 6 layers), or store the drafter's K/V instead (48 KiB a position, 3× the
   bytes). Either way G2 prices what is stored.
2. The change in the order listed above (scope into `BlockSpec`, the marked seed, the rows in the snapshot, the restore, the
   reporting), with G1's tier-0 tests and plants written beside each piece.
3. G1 tier 1, G4, then G3 (about 65 min of runs, by day).

**A dependency to read first:** the DFlash drafters' sliding, causal layers (`docs/measurements/dflash-window-2026-10/README.md`,
parked at its own STOP POINT) change what a fused row's K/V means for a windowed layer at a restore past 4,096 positions. Land that
fix before SP2b's G3, so G3 measures the drafter as it was trained.
