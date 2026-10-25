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
- **Amended while building, before G1–G3; G0 is what exposed it.** Two more places held a restored request's point as already
  served, and for a recurrent model both are wrong, because a recurrent snapshot serves only its own end:
  - **`kvPrefillMark`'s covering check** counted the restored, shorter snapshot as holding the new point. For a recurrent model
    it now declines only when the held snapshot reaches the point.
  - **`Model.residentSnapshotRows`'s admission** required a restored request's point to be more than `kvmgr.RollbackFloor`
    (448 rows) past the snapshot it restored from. That margin exists for windowed attention rings. For a recurrent model the
    margin is 0, and the new point replaces the restored snapshot.
  - Before this change, no recurrent request reached either branch, so neither alters anything that runs today.
- **Unchanged for attention models:** all three conditions keep their old form, so their snapshot behaviour is bit-identical.
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

## Results: every gate PASS (2026-10-24)

| gate | where | result |
|---|---|---|
| G0 | the Mac | `TestB35_aRestoredTurnKeepsItsOwnPoint` failed on `main` (turn 2 restored 1,152 rows and took no snapshot) and passes at the change |
| G0b | the Mac | `scripts/touched.py` green; the 17 other snapshot and B35 tests pass, none skipped; `go vet -tags realckpt`; the pinned `staticcheck` (its canary red); the parity refresh found 0 lines to refresh |
| G1 | nobara, 16:46 PDT | `cuda/b35_snapshot_test.go`: 3 of 3 pass |
| G2 | nobara, 16:47–16:52 | B35-TL at the change: **PASS** (R1, R2, R3, I1, T1 all green) |
| G3 | nobara, 16:52–17:03 | B35's end-to-end gate, the change against `main` `4814c869`: **PASS**, every row at 4k and 8k |

The `b35ka-gates` job ran under nobara's timing lock at `e6673f9d`, driver 595.91.07, starting at load 0.08 with 59,163 MiB
available. Records on nobara: `~/goinfer-logs/b35-keep-advance-2026-10/gates-2026-10-24-1646/` (G1 to G3) and
`~/goinfer-bench/b35-e2e/results-2026-10-24-1652/` (G3's own).

**G2, the tool loop, against the same conversation on `main`** (`docs/measurements/b35-toolloop-2026-10/README.md`, Q1-ONLY):

| | `main` | the change |
|---|---|---|
| Q2 (r4) reused | 4,096 of 7,329 | 4,096 of 7,329 |
| Q2 time to first token, of cold at its length | 0.502 | 0.514 |
| **Q3 (r6) reused** | **4,096** of 8,870 | **7,168** of 8,870 |
| **Q3 time to first token** | **28.12 s, 0.639 of cold** | **11.45 s, 0.260 of cold** |

Q2 restores Q1's point, as before, and now keeps its own. Q3 restores Q2's point. Restored replies equal cold (I1: 48 and 39
completion tokens, identical).

**G3, plain chat turns:** every row green, as on B35's own pass. One reading moved:
- **What moved:** at 4k, turn 3 now restores the state turn 2 kept, 4,096 rows instead of 3,584.
- **What it bought:** its time to first token is 0.28 s, 0.018 of cold. B35's own pass had 2.48 s, 0.157.

Cold output is unchanged against `main` (I2, every turn, both sizes).
