# Task: an autonomous kernel-optimization loop (autoresearch) over goinfer's gates

> **Status: PLAN / setup guide (not started).** Tracked as queue-engineering **E9**. This is an
> *execution method* for kernel campaigns, not a new campaign: an agent runs
> edit → benchmark → keep/revert unattended, gated by goinfer's existing correctness harness.
> Drafted 2026-08-13 after sankalp's "232× kernel via a Codex autoresearch loop" (GPU Mode qr_v2) and
> the `autokernel` project it spawned. Same freeze as the rest of E — code after the v0.13.0 tag; this
> doc lands now.

## 1. What it is, and why goinfer is unusually suited to it

The loop, from the source method: an agent modifies **one file**, runs a **fixed benchmark harness**,
and **keeps or reverts** the change on the measured result — indefinitely, ~40 experiments/hour. The
harness is **correctness-first**: every candidate is checked against a reference *before* any
performance number is recorded, so "a fast but wrong kernel is immediately reverted." Reward-hacking
(a kernel that's fast because it computes garbage) is structurally prevented, not policed.

Most projects have to *build* that correctness harness (autokernel's is a 5-stage smoke / shape-sweep /
stability / determinism / edge-case suite). **goinfer already has a stronger one** — the bit-identity
tests, the parity gates (argmax/cosine vs the CPU reference, itself parity-gated to HF), the Metal
snapshot golden, the determinism checks. The bit-identity contract that *costs* goinfer speed (the
whole flash-attention fork it can't take on the exact path) is the **same contract that makes an
autonomous keep/revert loop safe**: "fast but wrong" is auto-detectable and auto-revertable here in a
way it is not in a tolerance-only project. goinfer is a near-ideal host for this method for exactly the
reason it is slower.

## 2. The synthesis with the relay — what the loop is *not* for

The relay pattern (human briefs a leg → box executes → review) earned its keep by catching **premise
errors**: Lazy Z killed on cost before a kernel shipped, KV-quant refuted as a speed lever on both
backends before one was built. An autonomous loop **cannot do that** — it optimizes *within* a premise;
it does not question whether the premise is worth optimizing. So the two compose, they don't compete:

> **The relay decides *what* to search and defines the gates. The autoresearch loop does the mechanical
> searching inside that frame.**

Do not point a loop at an unvalidated premise. A leg's premise is validated first (by the relay); then
the loop grinds the search space that leg opened.

## 3. Targets — ranked, with the honest caveat on each

1. **The bit-identical flash-attention-style decode lever** (`docs/ollama-chase.md` — "the remaining
   long-context lever"; the §A2 split-KV is its start). This is the single largest measured deficit
   (5.54× behind at 32k; ~25× per-position coefficient), the search space is large (tiling, combine
   order, occupancy), and the arithmetic says parity is *reachable*. **Caveat that sets the mode:**
   the *exact-path* bit-identical levers are largely **exhausted** — Campaign A closed at ~1.17×, the
   V-sum unroll was refuted. So a loop gated **byte-identical** on the exact path will mostly
   re-confirm that ceiling. Its real value is either (a) a genuinely new bit-identical structure the
   hand search missed, or (b) the **fast-mode lane** (a `--mode fast` path gated by determinism +
   accuracy-floor rather than byte-equality), where the search space is far bigger and the win is the
   parity-at-depth prize.
2. **GEMV / tiling micro-tuning within bit-identity** — the MT/RN/int2-coalescing class that was done
   by hand. Bounded search, byte-identical gate, safe to automate; smaller wins.
3. **Cheap refutation.** Pointing the loop at a wall and having it fail to beat the floor across
   hundreds of candidates is itself a *written-refutation artifact* — the same product M3 (Metal FA)
   reached by hand. Lower value, but real: it converts "we think there's no lever" into "N candidates,
   none beat the floor," logged.

**Do NOT point it at Metal expecting wins.** Five independent refutations say the walls there are
structural (dispatch-/occupancy-bound; dedup-vs-occupancy opposition; megakernel closed). A loop would
spend a night re-deriving M3's refutation — acceptable as (3), wasteful as (1).

## 4. Preconditions — the guards that keep the loop honest

These are non-negotiable; each maps to a lesson already paid for in this repo.

- **The gate must carry the adversarial cases, or the loop reward-hacks the accuracy floor.** From
  P2/P3: a kernel can pass an accuracy floor on easy inputs and fail on near-ties. The correctness
  harness the loop runs per candidate must include the tie-heavy / near-boundary / flat-distribution
  cases, or the loop will "find" a kernel that is wrong exactly where it is hard to see. On a
  fast-lane target, add the determinism golden and the token-divergence ceiling.
- **Tiered gate, or 40 exp/hour is unreachable.** A fast inner check per candidate (a bit-identity
  microbench, seconds) with the full parity/heavy suite only as a **final confirm** on a survivor.
  This is the tiered structure goinfer already uses; **E8's `go test -json` gate-runner
  (`docs/task-gate-runner.md`) is the natural substrate the loop drives.**
- **Determinism of the bench.** The measured number must be stable enough that a real win is
  distinguishable from clock-ramp/thermal noise. Order-alternated best-of-N (the P6a clock-ramp lesson
  — an "impossible" 13% win was run-order), not best-of-min.
- **Never let the loop touch the gate.** It edits the kernel file only. The correctness harness is
  read-only to the agent — otherwise it can "pass" by weakening the check (the vacuous-gate trap,
  automated).

## 5. How to set it up (the concrete recipe)

The pieces are small; goinfer already has three of the four.

**(a) A single-number bench with a verdict.** A script/target that, for the current kernel file, prints
one line: `CORRECT|WRONG <metric>` — e.g. `CORRECT 165.2` (tok/s) or `WRONG` (any gate red). Structure
it as: run the fast bit-identity/accuracy gate → if red, print `WRONG` and stop (no perf number, so a
wrong kernel can never post a score); if green, run the throughput microbench (order-alternated
best-of-N) and print `CORRECT <tok/s>`. This is the autokernel `bench.py` role, in Go/shell over
`go test`. Reuse the E8 gate-runner once it exists; until then a thin shell wrapper around the existing
bit-identity test + a decode microbench.

**(b) The keep/revert harness (git-based, ~30 lines).** Loop: snapshot the kernel file (or `git
stash`/commit), let the agent make one edit, run (a); if `CORRECT` and metric beats the incumbent,
**commit as the new incumbent and record the row**; else **revert** to the incumbent. Append every
attempt to a `results.tsv` (candidate id, verdict, metric, %-of-peak, note) — human-readable, the
audit trail. Keep the loop's git history *separate* from main (a scratch branch); only the final
survivor is proposed for review through the normal relay/gate path.

**(c) The agent.** Codex CLI or Claude Code, pointed at the kernel file with a fixed instruction:
"propose one change to `<file>` that may improve `<metric>` while keeping the bench `CORRECT`; you will
be told the measured result; iterate." Feed it the last `results.tsv` tail so it grounds the next
mutation in measured outcomes, not guesses (the measured-not-guessed rule, applied to the agent). Give
it the roofline context (compute- vs memory-bound from the profile) so it proposes shape-appropriate
changes.

**(d) Review the survivors, don't trust them.** The loop's output is a *ranked list of candidates that
passed the fast gate*, not a merge. Each survivor goes through the full heavy parity confirm and normal
review before it lands on main — the loop replaces the *grind*, not the *judgment*. A survivor that
passed the fast gate but fails the full parity confirm is a gate-coverage finding, not a regression.

## 6. Sequencing and scope

- **After v0.13.0**, behind §C1 + the CUDA gate — same freeze as E7/E8. Build (a) on top of E8's
  gate-runner rather than duplicating its `go test -json` plumbing; the loop is a thin driver over it.
- **First target: GEMV micro-tuning (target 2)** — bounded, byte-identical, a safe proof that the
  harness and keep/revert discipline work before pointing it at the FA lane. Then the FA fast-lane
  (target 1b) once `--mode fast` and its quality lane exist.
- **Not in scope:** the loop making merge decisions (survivors go through review); pointing it at Metal
  for wins (target 3 only); any change to the gates themselves (§4, the loop is gate-read-only);
  automating the relay's premise judgment (§2).
