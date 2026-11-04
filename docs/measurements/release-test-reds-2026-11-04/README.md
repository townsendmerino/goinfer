# Two test reds at the v0.24.0 release commit, found with fixtures placed (2026-11-04)

Both fail at `f5261f4bf` (main) once the gitignored fixtures are placed, which is how the release gates run and how CI does not. CI never saw either: one needs `gemma4-moe-tiny`, the other real GGUFs and the card. `red-at-f5261f4bf.log` is the red run (13:48 PST); `green-with-the-fix.log` is the same tests on this branch (13:56).

## 1. `decoder`: `TestPlan_tableDriven` (two MoE subtests) and `TestPlan_extraBytesReservedAheadOfExperts`
They pass at v0.23.0 and fail at the release commit, so this is inside the release's range. `git bisect` (`bisect-decoder-plan.log`, 12 steps) names `61315ebab`, **MM1's "MoE reserve (Francis's option A)" of 2026-10-27**, which changed the planner (`internal/runtime/mem/placement.go:SolvePlacement`) and its own tests and census golden, and not these tests.

The change is intended: placement tries resident first, with every expert reserved, then the expert cache, and each picks its own context. The three tests ask `Plan` for an **unpinned** context at a tight budget and expect the expert cache or a decline. On this fixture the experts total less than one KV block, so shrinking the context by a block or four (`KV@8128`, `KV@7936` in the failure text) makes everything fit resident, and resident wins. The tests mean a tight budget at a *fixed* context.

**Fix:** pin the context in four calls (`CtxPinned: true`), as an explicit `-ctx` does. All the assertions then hold as written, and all of `TestPlan*` passes.

This does not decide whether MM1's resident-first answer for an unpinned tiny MoE is what Francis wants; it keeps the tests asserting what they were written for.

## 2. `backend/cuda`: `TestKV16_everyAttentionPathStaysCloseToF32` (all 10 subtests)
`decoder.PrefillGatePromptSet` names its prose seeds `../testdata/prefill-gate-prose-<label>/<file>`, which holds from `decoder/`. Layout PR 1 moved the backend tests to `backend/cuda/`, where that path names nothing: `read prose seed ../testdata/prefill-gate-prose-a/audit-2026-09-02.md: no such file`.

**Fix:** the one reader every caller goes through, `PrefillGateProseIDsForTest`, now tries the levels above when the path is not found (`proseSeedAbove`). The path strings are unchanged, so nothing keyed on them moves. `TestProseSeedAbove` covers it with no device; disabling the walk makes it fail (the plant in `green-with-the-fix.log`).

Five more tests in `backend/cuda` read their seeds through the same function (`flash_decode_gate_test.go`, `kv16_gate_test.go`, `kv4b_gate_test.go`, `vsum_mechanism_test.go`, and `kv16_test.go`'s second call). They skipped in the 2026-11-04 full-suite run, but the GPU gate sets `GOINFER_HEAVY_TESTS=1` itself, so they may run in it; this fixes their path too, and **I did not run them**.

## Release consequence
Both changes are in `decoder/`, a gate-qualifying tree (`cmd/gate/grade_records.go:gpuQualifyingTrees`), so they change what a gate record at `f5261f4bf` qualifies. They are meant to go in with the golden of the real-size plan-slack test, so the commit is re-picked once.
