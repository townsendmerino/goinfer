# B35 follow-up: a restored request keeps its own point (registered 2026-10-24, before any code)

**Francis's decision, 2026-10-24 4:35 pm (relayed by the coordinator):** "B35 follow-up option A. Register and build it: a request that
restores a kept state also marks its own Keep (decoder/kvsnap.go, Model.kvPrefillMark). Register the gate first: B35-TL re-run shows Q3
reusing more than Q2, restored still equal to cold, plus B35's own gate re-run. Then stop for his go before merging."

## Why

- **What B35-TL found** (`docs/measurements/b35-toolloop-2026-10/README.md` on branch `kapitsa-steps-2026-10`, verdict Q1-ONLY):
  - The second question after a tool loop restores Q1's kept state, and its reply equals cold.
  - The third question restores the same point: 4,096 of 8,870 tokens, at 0.64 of cold against the second question's 0.50.
- **The cause:** `Model.kvPrefillMark` returns no mark when the request restored a snapshot (`acq.snapRows > 0`). So a
  restoring request never keeps its own `KVScope.Keep`, and the kept point stays at the first question for the rest of the
  conversation.

## The change

- **What changes:** for a model with recurrent state, `kvPrefillMark` no longer declines because the request restored a
  snapshot. It still declines when the planned point is at or before where the prefill starts (`q <= from`), when the passes
  do not end on it, and when a held snapshot covers it.
- **Admission is unchanged** (`Model.residentSnapshotRows`): a point more than a rollback past the restored snapshot's end
  replaces that snapshot.
- **Unchanged for attention models:** the decline on a restore stays, so their snapshot behaviour is bit-identical.
- **Unchanged for every cold prefill:** the change acts only on a request that restored a snapshot.

## Gates (registered before the code)

| gate | tier | what | bar |
|---|---|---|---|
| G0 | 0, the Mac | `decoder/b35_snapshot_test.go:TestB35_aRestoredTurnKeepsItsOwnPoint` on the hash fake: turn 2 restores turn 1's point and takes its own; turn 3 restores turn 2's, deeper; turn 3's reply equals a cold run | passes on the change, and **fails on `main`** (run before the change is applied: the plant is `main` itself) |
| G0b | 0, the Mac | every other test in `decoder/b35_snapshot_test.go` and `decoder/kvsnap*_test.go`, through `scripts/touched.py` | all pass |
| G1 | 0, nobara (CUDA device) | `cuda/b35_snapshot_test.go` at the change | all pass |
| G2 | 1, nobara | B35-TL re-run at the change (`run-b35-toolloop.sh`, the same conversation and rules) | verdict **PASS**: R2 met (Q3 reuses more than Q2, on both servers), I1 (restored equals cold), R1, R3 and T1 green |
| G3 | 1, nobara | B35's own end-to-end gate (`docs/measurements/b35-e2e-2026-10-20/run-b35-e2e.sh <change> <main>`) | every row green, as registered there (I1, I2, I3, R1, T1) |

- **Every gate must pass, and none is waived.** A G2 or G3 PARKED goes to Francis. Any FAIL stops the merge.
- **Also required before the push** (CLAUDE.md, "Day loop"):
  - the parity hash refresh, because a non-test `decoder/*.go` file changes;
  - the citation lint;
  - `gofmt`, `go vet` and the pinned `staticcheck` on `decoder`.
- **Order:** G2 and G3 run on nobara after SP0's measurement job finishes (Francis's instruction).
- **Merge:** a PR, a Gemini review by the coordinator (product code in a KV path), then a stop for Francis's go.

## Pre-registration (`docs/measurements/noise-registry.md` §8)

- **Tiers:**
  - G0, G0b and G1 are tier 0: counters and bit-identity, seconds.
  - G2 and G3 are tier 1: one small real model, by day, about 6 and 12 minutes (their own records' walls).
- **Interleaving:** none needed. The bars are identity and B35's ratio rule against a cold control on the same server.
- **Cell validity:** each nobara job records the load average and `MemAvailable` at its start. Above 8.0 load it is void.
- **Reused readings:** none. The change is new, and both end-to-end gates re-run at it.
- **Fit check:** both gates load the same model with the same flags as their own passing runs (`-ctx 16384`, Qwen3.5-9B,
  nobara). The first server start's banner is the check.
- **Stopping rule:** fixed. Every gate runs once.
- **Ambiguous band:** T1's 1.5–2.5×, in both end-to-end gates, parked for Francis.
- **Cost:** G0, G0b and G1 take seconds. G2 is about 6 min and G3 about 12 min, plus builds. Queued on nobara after
  `sp0-measure`.
