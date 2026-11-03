# Review notes: writeup 15, "An upgrade that can't change your answers" (drafted 2026-09-29)

Draft, `reviewed:` empty. Body about 760 words. Site build with `-drafts` passes. Nothing was run; the report example is built from the code's format strings and labelled as placeholders, not a recorded run.

| Claim | Source |
|---|---|
| Command, flags (`-backend`, `-families`, `-assets`, `-quant`, `-only`, `-steps`, `-logdir`, `-keep`, `-record`, `-timeout`), defaults, positionals anywhere, exit codes 0/1/2 | `cmd/gate/identity.go` (`runIdentity`, `reportIdentity` tail); `defaultIdentityQuants` in `identity_assets.go` |
| Verdicts IDENTICAL / WITHIN TOLERANCE / DIFFERENT / NOT RUN; tolerance = 2x own run-to-run max diff, argmax and tokens exact; GPU cell that fell back to CPU is NOT RUN | `cmd/gate/identity_compare.go` (`judgeCell`, `identityTolFactor`) |
| Worktrees, dumper, run order new1 -> old -> new2, `GOINFER_*` stripped, decode path printed | `cmd/gate/identity.go` header comment; task doc TE6(b) "BUILT" |
| 120/120 CPU, 38/38 Metal (19 families), 80/80 WebGPU (30 families), kill criterion did not fire, tolerance mode unused | `docs/completed/task-test-efficiency-2026-09.md`, TE6(b) "Determinism holds"; commit `ce97d5ef` message |
| identity HEAD~1 HEAD, tiny, CPU, MacBook: 43-46 s, 30 IDENTICAL, 7 NOT RUN | same, "Wall, tiny assets"; `ce97d5ef` |
| Real 0.5B demo, L1 change 3cd62e6d -> 5c85f7c0 DIFFERENT from prefill logit 0; three equivalence tests | same, "Real demo" and "Equivalence" |
| Kill criterion written before build (2026-09-28 10:31, `8a472437`; tool `ce97d5ef` 14:42) | `git log`; task doc TE6 "Kill (b)" |
| 0.897 logit cosine with argmax held | `CLAUDE.md` ("PREFER DIFFERENCING PER LAYER") |
| arm64 fuses `x*y+z`, amd64 does not; 93% of logits differ; contract is argmax+cosine across arches | `docs/parity-coverage-policy.md`, "Mechanism" (~line 1301) |
| Limits: no CUDA, LogitProcessor dumper skips device-side argmax, 7 families no tiny fixture, not yet used on a real row, only CPU same-arch eligible | task doc TE6(b) "Limits" / "OWNER DECISION"; policy doc "Identity-inherited validation" (~line 72) |

## Conflicts and resolutions
- The brief and `task-site-2026-09.md` cite `docs/measurements/test-efficiency-2026-09.md` for 120/38/80. That file (the TE0 census) does not contain them; grep finds them only in `docs/completed/task-test-efficiency-2026-09.md` (and the commit message). I cite the task doc as the source. The site-task row should be corrected.
- The figures are run-to-run determinism counts, not old-vs-new results. The page says so in the table heading.

## Left out as unverified
- Which machine produced the CPU and WebGPU determinism counts (only the Metal-adjacent timing run names the MacBook), and how many families the 120 CPU cells cover.
- "114,000x" decision headroom for arm64 vs amd64 exists in a memory note, not in a repo record I opened; the policy doc says 10^5x, which I paraphrased as "far below".
- Any real recorded console output: none was found, so the example uses placeholders.

## Questions for the owner
1. The 120/38/80 counts are from one day's build. Is that the evidence you want on the page, or should a fresh night run be queued to name machines and date the figures?
2. The page names L1 with two commit hashes. Fine for a public page, or drop it?
3. Should the "not yet used on a real row" limit stay, given it will go stale when the first row is recorded?
