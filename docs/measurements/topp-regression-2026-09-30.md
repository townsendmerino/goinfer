# Top-p sampling on CUDA ran 26% slower from `7a44a58e`: found, bisected and fixed (nobara-pc, 2026-09-30)

**Status: FIXED** in the commit that adds this record. Found by the peer re-run
([`peer-sweep-2026-09-29.md`](peer-sweep-2026-09-29.md), finding 4); the owner asked for it to be fixed by day.

**Symptom.** goinfer's 0.5B at temperature 0.8 / top-p 0.95 on CUDA decoded 239.3–242.9 tok/s in the 2026-09-29 sweep, against
318.4–318.5 at 09-25's `411e7fc4`, while greedy and untruncated temperature 1.0 got faster.

## Reproduced goinfer-vs-goinfer, and the `--embed-int4` flip ruled out

`scripts/bench_peer.py`, `BENCH_ENGINES=goinfer,goinfer_old`, CUDA, 0.5B q4_k_m, depth 128, 3 runs, one session, nobara-pc (RTX 2070 SUPER,
driver 595.91.07), 2026-09-30 05:31–05:40 PDT. Binaries: `serve-cuda-754f12d3` (the sweep's) and `serve-cuda-411e7fc4` (09-25's).
[`run-ab.sh`](topp-regression-2026-09-30/run-ab.sh), [`run-ab-embedint8.sh`](topp-regression-2026-09-30/run-ab-embedint8.sh), raw JSON beside them.

| arms | greedy, 754f12d3 / 411e7fc4 | top-p 0.95, 754f12d3 / 411e7fc4 |
|---|---|---|
| each binary at its defaults | 352.5 / 330.8 tok/s | **238.3 / 321.6 (0.741×)** |
| both with `-embed-int4=false` | 342.6 / 331.6 | **227.8 / 322.6 (0.706×)** |

The regression is there with the int4 LM head off on both sides, so the 2026-09-28 `--embed-int4` default flip is not the cause.

## Bisected to `7a44a58e`

`git bisect run` over `411e7fc4..754f12d3` (328 commits) with [`bisect-step.sh`](topp-regression-2026-09-30/bisect-step.sh): build the CUDA
server at the commit (its own pinned aikit), time one 0.5B top-p cell, good above 290 tok/s, bad below 260, skip between. Eight steps,
every reading unambiguous (good 317.7–330.8, bad 229.2–239.1; [`bisect-steps.log`](topp-regression-2026-09-30/bisect-steps.log)). First bad
commit: **`7a44a58e` "MC3 on CUDA: batched multi-request decode — the resident implements ResidentBatchStepper"** (2026-09-27).

## Cause

`decoder/model.go` chooses the device top-K fast path for a filtered sampler (top-k / top-p / min-p) only with `mc3 == nil`: "MC3: off
— a TopKRow's Full() reads the resident's logits a token later, after another generation's step may have overwritten them." MC3
existed on Metal first; `7a44a58e` gave CUDA a `ResidentBatchStepper`, so from then on every CUDA model that can batch ran its
generations under MC3, and every top-p request, **alone or not**, took the full-row path: a 151,936-logit readback and a host sort for
top-p on every token. Greedy and temperature-only sampling were untouched, because they pick on the device (`ForwardArgmax`,
`ForwardSample`). The MC3 grading on 2026-09-27 covered greedy and temperature 0.8 only, which is why it did not see this.

## Fix

Keep the top-K path under MC3 and remove the hazard instead: under MC3 the draw is made **inside the resident call**, while the
generation still holds the resident. If the K candidates hold the whole retained set, the token is drawn there (`topKPre`); if not,
the full row is read there and copied (`topKPreFull`), before any other generation's step can overwrite the buffer. Without MC3
nothing changes. The sampler is the generation's own and nothing else draws from it between the forward and the point the old code
drew, so the RNG is consumed in the same order and the output is unchanged. A token that a batched step serves (several
generations at once) still returns the full row, as before.

**Tests.** `TestMC3_topPConcurrentMatchesAlone` (new; `decoder/mc3_batch_test.go`): the MC3 fake gains a top-K path whose `Full()`
fails if any later write reached the shared logits buffer. Alone under MC3 the top-K path must serve draws (top-p 0.5) and fall back
(top-p 0.999, with K capped at 4); four concurrent generations must equal the same conversations served alone. Putting the old
`mc3 == nil` condition back turns it red ("no top-K row was requested"). Its limit, stated: the fake's timing does not make a solo
top-K call interleave with another generation's step deterministically, so the in-call resolution is covered by the identity
checks below rather than by this test failing without it. `TestMC3_*` and `TestSampleFromTopK_*` pass.

## Measured after the fix (served, one session, 2026-09-30 05:50–05:55 PDT)

[`run-fix.sh`](topp-regression-2026-09-30/run-fix.sh): the fix built from `0a3f0a7f` plus the patch, on nobara-pc; `go vet -tags
'cuda goinfer_testhooks' ./cuda/` clean there.

| arms | greedy (fix / other) | top-p 0.95 (fix / other) |
|---|---|---|
| fix vs `411e7fc4` (before the regression) | 359.6 / 334.6 tok/s | **351.2 / 319.7 (1.10×)** |
| fix vs `754f12d3` (the regressed build) | 366.1 / 361.2 | **347.0 / 246.0 (1.41×)** |

**Output identity.** Five seeded top-p requests (128 tokens) to each binary: the fix returns the same text as `754f12d3` on **5/5**
(the full-row path it replaces), and matches `411e7fc4` on 0/5, which is expected: that build predates the
`--embed-int4` default and other numeric changes, so its logits differ. [`identity.json`](topp-regression-2026-09-30/identity.json).

**Not covered, and why it does not matter elsewhere.** Only CUDA implements `ResidentTopK` (`cuda/topk.go`), so Metal and WebGPU never had
the top-K path and top-p there always read the full row; this regression, and this fix, are CUDA's. CPU batching (MC3c) is a
different path and was not affected.
