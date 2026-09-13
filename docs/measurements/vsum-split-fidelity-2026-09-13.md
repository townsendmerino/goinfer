# V-sum split spike — §3.2 fidelity gate (IN PROGRESS)

Pre-registration: `vsum-split-fidelity-PREREGISTERED.md`, committed `f9d50cab` before Phase A was
launched. Gate: `cuda/vsum_split_gate_test.go` (`2ef265d7`). *(The pre-registration still names
`docs/task-prefill-gap.md`, which was archived to `docs/completed/task-prefill-gap.md` while this
gate ran — left unedited on purpose, since a pre-registration that changes after results exist
stops being one.)*

**Which build was measured.** This work was rebased onto origin/main mid-run, rewriting every hash
cited here; the ones above are the rebased hashes. The scoring binary for both cells was built
BEFORE the rebase, from the commit now published as `cc90c18f` but then at `13c75842` (preserved
locally as `backup/pre-rebase-20260913`). Between that build and main: the CUDA kernel sources and
PTX are byte-identical, and the split-KV / V-sum Go code differs only in one comment path — but the
rebase brought 169 Go-file changes elsewhere, including `cuda/resident.go`, `cuda/backend.go` and
`cuda/prefill.go`. So the exact-vs-spike **comparison** is a property of the V-sum reduction tree and
carries to main; the **absolute** agreement and KL figures are for the pre-rebase build.

**Status: S (confirmation) scored — DOES NOT PASS under the registered rule, on criterion (a) by one
hard flip; PASSES under criterion (a) as AMENDED by owner decision (below), re-scored in a logged run
before any D7 result. D7 (decision) reference still generating. The gate's verdict is D7's, and is not
in yet.**

## S — confirmation cell — DOES NOT PASS

qwen2.5-coder-1.5b-instruct-q4_k_m, nH=12 nKV=2 hd=128, K=8000, prompt set A, S=4. Reference: CPU,
f32 weights and activations, exact f64 attention, 1h39m. Phase B log
`goinfer-logs/vsum-fidelity-phaseB-S-20260913-121657.log`, 5m39s.

| S, 10 prompts x 64 positions | exact | spike | pre-registered criterion |
|---|---:|---:|---|
| hard flips vs reference | 7/640 | 8/640 | **(a) spike <= exact — FAIL** |
| mean teacher-forced agreement | 94.84% | 94.06% | (b) >= exact − 1.0 pt AND >= half: −0.78 pt, 6/10 — pass (outside the 0.2 pt parked band by 0.02) |
| mean KL(reference ‖ arm) | 0.042577 | 0.042689 | (c) <= 1.10x: **1.0026x** — pass |
| worst near-tie gap | 5.458% | 5.725% | — |

**Preconditions, all held:** A/A bit-identical over 64 rows; the prefill seed row identical between
arms on every prompt; **630/630 decode rows differing** — the spike was active on every position of
every prompt, so this is a measurement and not the exact path scored against itself.

Per prompt (agreement exact / spike, hard flips exact / spike, KL exact / spike):

| p | agree | HF | KL |
|---:|---|---|---|
| 1 | 96.9 / 96.9 | 1 / 1 | 0.06323 / 0.06229 |
| 2 | 93.8 / 95.3 | 2 / 1 | 0.03020 / 0.02961 |
| 3 | 95.3 / 93.8 | 0 / 1 | 0.02241 / 0.02375 |
| 4 | 100.0 / 100.0 | 0 / 0 | 0.00776 / 0.00778 |
| 5 | 96.9 / 96.9 | 0 / 1 | 0.06743 / 0.06947 |
| 6 | 85.9 / 84.4 | 1 / 1 | 0.08363 / 0.08829 |
| 7 | 89.1 / 84.4 | 1 / 1 | 0.05702 / 0.05598 |
| 8 | 95.3 / 93.8 | 1 / 1 | 0.04535 / 0.04423 |
| 9 | 100.0 / 100.0 | 0 / 0 | 0.02387 / 0.02357 |
| 10 | 95.3 / 95.3 | 1 / 1 | 0.02487 / 0.02191 |

The verdict stands as registered. S is the confirmation cell: it cannot decide the gate, and it
cannot be rescued.

### Two statements recorded BEFORE the D7 result, so neither can read as a rescue of it

**1. A labelling error in the pre-registration — and the rule is NOT being switched.** The
pre-registration is titled a "§3.2" gate, but criterion (a) was written from **§3**'s strict form,
`spike HF <= exact HF`. The Metal §3.2 pooled gate (`prefill-gate-l1-ref-b-2026-09-09.md`) uses a
noise-aware ceiling, `exact + 2√exact`; for S that is 7 + 5.3 = 12.3, which 8 would clear. **It is
not adopted, for S or for D7.** The committed rule text is unambiguous, and moving to the friendlier
bar after watching a cell fail the strict one is precisely what pre-registration exists to prevent.
The mislabel is mine and is recorded as an error, not as a loophole. *(Superseded the same day:
Francis overruled this and adopted the §3.2 ceiling for (a) — see "OWNER DECISION" below. This
paragraph is kept as the record of what was recommended at the time, not as the rule in force.)*

**2. The registered criterion (a) has almost no power at these counts.** 7 against 8 hard flips is
well inside Poisson noise (σ ≈ √7 ≈ 2.6). D7 will likely sit in the teens per arm — its K=1024
prefill cell was 16 v 16 — so D7's (a) outcome will be largely set by noise whether the spike is
harmless or mildly harmful. The metric that *can* resolve a difference here is KL, and on S the arms
are indistinguishable at 1.0026x. **This is a statement about the gate's resolving power, written
while D7's outcome is unknown.** It does not change D7's rule, and if D7 fails (a) this paragraph is
not a reason to overlook that; it is the reason a follow-up with a noise-aware rule would have to be
pre-registered on NEW data (held-out prompt set B), not applied to this run.

**An observation, not a finding:** three prompts have spike trailing exact on agreement with hard
flips tied — 6, 7 and 8. On 7 and 8 its KL is *lower*; on 6 it is higher (0.08829 vs 0.08363). So
this is 2 of 3, not a pattern. *(Corrected the same day: the first version of this paragraph named
only 7 and 8 and read as if every such prompt went that way, which prompt 6 contradicts.)* Where it
does hold, the argmax disagreements sit at positions the reference itself scores as near-ties, the
arms landing on opposite sides while their distributions stay about equally close. Three prompts is
not evidence of a mechanism.

## OWNER DECISION — criterion (a) amended (2026-09-13, after S was scored, before any D7 result)

**Francis overruled the strict criterion (a)** on reading the two statements above. Recorded here in
full because it is a rule change made *after seeing a result*, and the only thing that makes such a
change legitimate is that everything about it is visible.

| | as pre-registered | as amended |
|---|---|---|
| (a) hard flips | `spike HF <= exact HF` (§3 strict) | `spike HF <= exact HF + 2·√exact HF` (§3.2) |
| (b) agreement | >= exact − 1.0 pt AND >= half the prompts | **unchanged** |
| (c) KL | <= 1.10x, parked above 1.05x | **unchanged** |

**What triggered it, stated plainly:** the S confirmation cell failing strict (a) by one flip. It
was not noticed in the abstract first. The amendment would not have been examined today without that
failure, and a reader should weigh it knowing so.

**The mechanism, which is what separates this from moving a bar because a number moved** (CLAUDE.md:
"move a bar only with a mechanism"): a hard-flip count in single digits to the teens is a Poisson
count, and a strict `<=` between two such counts is decided by noise — 7 v 8 sits well inside
σ ≈ 2.6. The replacement is not invented for this gate. It is the ceiling
`docs/completed/task-prefill-gap.md` §3.2 specifies ("with the Poisson noise of a count of ~40 written into
it instead of ignored") and `metal/prefill_gate_ref_test.go` implements — the bar the Metal fast
prefill shipped under on 2026-09-09. The pre-registration's own title named §3.2; its rule text did
not match its title, and the amendment makes them agree.

**What it does NOT do:** it does not touch (b) or (c). §3.2's noise-aware (b), `exact − 2·√d/N`,
would be *looser* than the registered 1.0 pt at these counts; it is deliberately not adopted, so the
agreement bar and its parked band stay as registered. It does not touch the preconditions.

**Auditability:** the gate computes and prints the strict (a) on every cell alongside the amended one
(`critA(...exact+2*sqrt(exact))=… [strict spike<=exact, as pre-registered: …]`). Every cell's result
is therefore reported under both rules, and a reader who rejects the amendment can read the
registered verdict directly off the same log line.

**Timing:** decided and committed while D7's reference was still generating on the CPU, with no D7
reference file on disk, so D7 is judged under a rule fixed before its data existed. S is re-scored
under the amended rule in a separate, logged run; its registered verdict (above) is not withdrawn.

## S re-scored under the amended rule — PASSES (confirmation cell)

Separate process, binary built from the tree at the amendment commit (`5677f44e` after rebase), log
`goinfer-logs/vsum-fidelity-phaseB-S-amended-20260913-124204.log`, 6m17s.

**Every per-prompt figure reproduced exactly** against the first run — agreement, hard flips and KL to
five decimals on all ten prompts, and the same cell totals. Scoring is deterministic across
processes, not only within one (the A/A precondition covers the latter; this covers the former).

| criterion | S result | verdict |
|---|---|---|
| (a) amended: `spike HF <= exact + 2·√exact` | 8 <= 7 + 5.29 = 12.29 | **pass** |
| (a) strict, as pre-registered — printed alongside | 8 <= 7 | fail |
| (b) >= exact − 1.0 pt AND >= half | −0.78 pt, 6/10 | pass (0.02 pt outside the parked band) |
| (c) KL <= 1.10x, parked above 1.05x | 1.0026x | pass |

**S: PASSES under the amended rule; DOES NOT PASS under the registered one.** Both stand, from one
log line. (b)'s margin to its parked band is 0.02 pt — one position in 640 would have moved it — so
the S pass is not comfortable on agreement, only on KL. It remains a confirmation cell either way.

## Deviations from the pre-registration

Every one of these was decided **before any reference file for K=8000 existed**, so none can have
been steered by a result. Each is stated with what it was, why, and what it can and cannot affect.

### D1 — D7's reference weights: f32 → int8 (decided 2026-09-13 11:24, Phase A at 48 min, S cell unfinished)

The pre-registration named f32 reference weights for D7 (`GOINFER_CPU_REF_QUANT_D7=""`), to match
this box's existing D7 cells. It was changed to int8 weight-only (`GOINFER_CPU_REF_QUANT_D7=int8`)
after the run itself showed the pricing was wrong:

- **The run was slower than projected**, which is what exposed it. S at K=8000 had not finished one
  prompt after 45 minutes. Scaled from this box's own Sep-5 cells — S K=1024 3.5 min to first batch,
  K=3900 18.5 min, i.e. 5.3x the cost for 3.8x the tokens, attention's K² showing — D7 at K=8000
  projects to 4-5.5 h, not the ~3 h quoted when the run was launched.
- **The memory was never priced before launch, and the fit guard was bypassed.** S at K=8000 sat at
  16.5 GB RSS, ~1.3 GB per worker over its weights. Scaled to D7's wider layers plus 28 GB of f32
  weights, that lands near ~50 GB against 58 GB available — close enough that an OOM hours into the
  D7 cell was a live risk, and an OOM kill of a detached process writes nothing to its log.

**Why int8 is a valid reference for this gate**, by the harness's own reasoning
(`decoder/prefill_ref_gen_test.go`, `d7RefQuant`): the reference must have f32 **activations**,
because activations are the axis the arms could differ from it on. Reference weight precision is
**common-mode**: both GPU arms load identical int4 weights and differ only in the V-sum reduction
tree. int8 moves where the reference sits, equally for both arms; it cannot favour either.

**What it can affect:** absolute agreement / KL figures for D7 are not comparable to the Sep-5 f32
D7 cells. **What it cannot affect:** the pre-registered criteria, which are all exact-vs-spike
comparisons against the same reference.

**Mechanics.** The S cell was left to finish at f32 (as registered — S's f32 weights are 6 GB and
were never at risk). A detached watcher (`goinfer-logs/vsum-fidelity-handoff.sh`, log alongside)
waited for the D7 subtest to start, verified all ten `S-K8000-p*.bin` were on disk, stopped the f32
run, and relaunched `-run TestPrefillGateReference/D7` at int8.

## Separately recorded, not a deviation

The served A/B that the spike record had called unestablished was re-run and archived before Phase
A launched — `vsum-split-spike-2026-09-13.md` §"What this does NOT establish" item 2. It is a speed
measurement and bears on nothing in this gate.
