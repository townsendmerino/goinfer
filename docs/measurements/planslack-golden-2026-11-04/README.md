# TestPlanSlack_realSizeDiff: a stale golden, explained group by group (2026-11-04)

**Question (the coordinator, 13:42 PST):** `backend/cuda/kv_slots_test.go:TestPlanSlack_realSizeDiff` fails on clean main, all 63 groups. Is the golden stale after this week's intended planner changes (a), or did the planner's decisions change (b)?

**Answer: (a).** The only change is that the plan now prices dense vectors, and it moved every decision boundary by exactly that amount. No decision changed except one, by one context block, which that amount predicts.

## What changed
`3a899d303` (MM8, #645: "the plan prices dense vectors, gpt-oss's expert biases, the CPU's structs and the recurrent checkpoint") made the plan's dense bytes sum every tensor the load holds, not only the matrices: norms, biases and small f32 tables (`decoder/weightbytes.go:f32SliceBytes`). The test passed at `937cfe8fb` (the commit before it) and fails from `3a899d303` on, at every commit sampled.

Per model, `DenseBytes` grew by a constant:

| model | dense bytes added |
|---|---|
| qwen2.5-coder-0.5b | +286,208 |
| qwen2.5-coder-1.5b | +579,584 |
| qwen2.5-7b | +1,333,248 |

The 7B figure checks by hand against Qwen2's layers: per layer, the q bias (3,584 floats), the k and v biases (512 each) and two norms (3,584 each) are 47,104 bytes; 28 layers are 1,318,912; the final norm adds 14,336.

## Evidence
`explain.py`, in this directory, reads the priced side's plan lines the test dumps (`-planslack.out`) at both commits and exits 0 only if all of this holds:

- **Every other plan term is unchanged:** `slack`, `expertUsed` and `extra` did not move in any of the 63 groups.
- **Every decision boundary moved by exactly the dense delta.** The old run has 231 true change points (a reading whose neighbour one byte below decides differently). All 231 are in the new run at exactly +delta, with the same decision on both sides, and no group has a change point that does not correspond one to one (`groups.tsv`, per group).
- **At nobara's two free readings, 125 of 126 cells decide identically.** The one that differs is the 7B, requested ctx 32,768, with a 256 MiB companion, at 7,648,378,880 bytes free: resident context 23,808 becomes 23,744, one 64-position block less, as dearer dense bytes predict.
- **Every boundary moved up,** so a model needs slightly more free memory than before to get the same plan: the safe direction.
- **Why all 63 hashes change:** each cell's line prints `dense` and `need`, so any growth in dense bytes changes every group's hash, whether or not a decision moved.
- **The dumps are the golden's content:** the old golden equals the hashes of the old dump on all 63 groups, and the regenerated one the new dump's on all 63.

## What this does not show
- **That the new dense figure is right against a real build of these three models.** The memory gate that found the gap (`backend/cuda/sb2_census_test.go:TestMemGate_cuda`) reads tiny fixtures. The 7B figure matches the model's own geometry by hand, which is the evidence here.
- **Anything about the other backends' plans,** or the planner at other free readings than the test's. The golden holds the priced side only.

## Release consequence
The golden is `backend/cuda/testdata/mm2b_realsize.json.gz`, inside the `backend/cuda` tree. A gate record qualifies by the git tree id of `backend/cuda`, `backend/metal`, `backend/webgpu` and `decoder` (`cmd/gate/grade_records.go:gpuQualifyingTrees` and `parityQualifyingTrees`), and a `quick -full` record by any change outside docs, `apps/site`, `.github`, `changelog.d` and top-level markdown. So this change voids a gate record made at the commit before it. The GPU gate's run at `f5261f4b` was stopped five minutes in for that reason, before any verdict.

## Reproduce
At each commit, in `backend/cuda` (in `cuda/` at `937cfe8fb`), under the timing lock on the card, with the three GGUFs in `~/models`:

    CGO_ENABLED=0 go test -count=1 -v -tags 'cuda goinfer_testhooks' -run '^TestPlanSlack_realSizeDiff$' . -planslack.out DIR

`DIR/after.txt` is what `after-*.txt.gz` here hold. Then `python3 explain.py`. The golden is regenerated with `-planslack.update` instead; `regen.log` is that run and the run that held it, at `f5261f4bf`, on the RTX 2070 SUPER, driver 595.112.07, 2026-11-04 13:44 PST.
