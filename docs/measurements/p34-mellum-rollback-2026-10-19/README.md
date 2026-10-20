# P34 cause 2: Mellum2.1's rollback grant under an expert cache, before and after the fix (pre-registered 2026-10-19)

Registered before any run. The owner approved the fix on branch `mellum-rollback-2026-10` (7b0c9a59): an expert-cache load of a
windowed model takes the rollback grant just before its expert cache is sized, never below the cache's minimum
(`cuda/kvwindow.go:cudaResident.kvGrantRollback`, and a new method on the branch, expertCacheFloorBytes, that prices the cache's minimum). Before it, such a load kept the
floor rollback (448 positions, up to 512 by the compaction phase), which refuses tack's E0 resend (diverging about 518 back) at every
size (`docs/queue-performance.md`, P34). Merge needs this probe to show the predicted pattern and the CUDA gate below to pass, then
the owner's yes.

- **Tier 0 (CUDA, tiny fixtures):** `TestWindowedKV_*` at the fix. `TestWindowedKV_expertCacheGetsTheGrant` (an expert-cache load is
  granted like the control; with only the reserve and the cache's minimum free it is not, and still loads) and
  `TestWindowedKV_e0ResendReplay` (kvWindowNextBase over E0's shape: 406, 518 and 600 back at the floor and the granted rollback) are new.
  Plant: the pre-fix skip put back in `kvGrantRollback`; `TestWindowedKV_expertCacheGetsTheGrant` must then fail.
- **The probe (by day on nobara, under the timing lock):** Mellum2.1 (`~/models/mellum2.1-thinking`), `-backend cuda -moe-cache-experts
  -ctx 8192 -log-requests`, on the old build (main 6ce495c4) and the new one (7b0c9a59), one load each. Four `/v1/completions` requests,
  `max_tokens` 1, temperature 0: A (about 8k tokens of `docs/benchmarks.md`, cold), A with its last ~400 tokens replaced, A again, A with
  its last ~600 tokens replaced. Read from each request line: `reused=`, `div=`, `ttft=`.
- **Prediction, per build:**
  - old: the 400-token edit reused (reused > 0), the 600-token edit cold (reused = 0);
  - new: both reused.
  - "A again" (about 400 back from the held edit) reused on both.
- **Bands:**
  - **PASS:** the prediction on both builds, and the tier 0 tests pass with the plant failing.
  - **AMBIGUOUS:** the old build's 400-token edit is also cold, with its logged divergence above 448 back (the compaction phase or
    block rounding moved it past the floor). Re-probe at a 300-token edit before reading it.
  - **FAIL:** the new build's 600-token edit is cold, or the old build's 600-token edit is reused (the mechanism is not the one
    claimed), or any tier 0 test fails, or the plant passes.
  - **VOID:** a load refused, a load not `cuda-resident`, or a request error.
- **No stop rule** beyond the one run of each; no timing claim (ttft is reported, not graded).
- **Cost:** about 15 minutes of nobara (two builds, about 2 minutes of tests, two loads, two or three cold 8k prefills at about
  93 tok/s), after tack's comparison releases the lock (about 20:20 PDT).
- **Job:** `~/goinfer-bench/p34/run-p34.sh` on nobara, from a bundle of the branch (not on GitHub); logs in `~/goinfer-bench/p34/results/`.
