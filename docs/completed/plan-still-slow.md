# Plan — the three "still slow" items (post-v0.10.3)

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


**COMPLETED RECORD.** Drafted and executed in one relay span, 2026-08-09 (`31829de` P0 → `b3ac838`
close). Filed to `docs/completed/`. Commit list in the phase ledger below and the effort table at the
foot. Scope: the three residual slow paths named at the close of the v0.10.3
sampling work: (1) the nonzero-temperature ~3× cliff (readback + full-vocab softmax, D6
remainder), (2) `top_k=1` running 13–18% behind greedy, (3) long-context decode and prefill,
untouched since v0.9.0. Sources: `docs/ollama-chase.md` §3/§8, `docs/completed/audit-2026-08-05.md`,
`decoder/sampler.go`, `cuda/argmax.cu`, `metal/kernels.go`. Per the ground rules, nothing here
re-proposes anything in §10 (refuted) and every phase names its gate.

## Outcome — the three items, closed (relay complete, 2026-08-09)

This plan is executed; the sections below are the working record and the phase details. What the
three "still slow" items became, with measured outcomes:

- **The sampling cliff — largely closed.** Temperature-only decode is **+126–137%** and temp+top_p
  **+98–108%** (P2b, deterministic parallel host normalization); `top_k=1` is **+18–22%** (P1,
  greedy-fast routing). The published sampled deficit vs the peer went from **2.1–2.9× to 1.08–1.40×**.
- **`top_k=1` — closed outright.** Its "hard prerequisite," the CUDA argmax index tie-break, was
  already fixed (audit **C-14**, `c6600fc`); the three texts calling it open were stale — P0 corrected
  them and confirmed the fix live (`TestArgmaxTieBreak`).
- **Long-context decode & prefill — measured and bounded, not closed.** Split-KV was re-gated (**+19%
  at 0.5B@512**) and a live regression fixed (P6a); depth was measured to 32k and is **linear with a
  plateau coefficient** (~0.74 / 1.0 µs/pos CUDA, **~9–12 µs/pos Metal**) vs the peer's ~0.03–0.09 —
  a **5.54× gap at 32k**. KV-quant was **refuted as a speed lever on both backends** (P4); prefill
  (P5) stays banked. The remaining decode lever is the occupancy/latency (FA-class) rewrite, which is
  non-bit-identical and unfunded.

**Method findings promoted to standing notes** (ollama-chase §11 / benchmarks §maintenance):
cross-session pairing on a ~3.5%-drift box; previous-engine device contention; microbench
best-of-min flattery; the vacuous-gate trap; no-silent-caps for benchmark tables.

**Phase ledger.** P0 ✓ `31829de` · P1 ✓ `ed81e13` · P2 **NO-GO** `bc59c56` (refuted on cost;
finding kept) · P2b ✓ `686c9f8` · P3 **BANKED** (P2b residual too small; reopen price below) · P4
**DECIDED** `d4472b9` (reachability-only on both backends; build deferred) · P5 **BANKED** · P6 ✓
`9d08165` + `b3ac838` (CUDA + Metal depth) · P6a ✓ `2693dce` · cap-raise ✓ `ca29d6c` · P7
**DEFERRED** (precondition unmet).

---

## P0 — Truth maintenance (hours; do first, on the CUDA box)

1. **Confirm C-14 is live**, not just present in the tree: `GOINFER_HEAVY_TESTS=1 go test -run
   TestArgmaxTieBreak ./cuda/` (needs the 0.5B fixture). The audit says fixed; the D6 text written
   *after* the audit says open — one of them is wrong, and P1 is gated on the test being green, not
   on either doc.
   **CONFIRMED (CUDA box, 2026-08-09).** Green, and it genuinely ran (3.91 s — loads the 0.5B and
   builds the resident; a skip would be instant):
   ```
   === RUN   TestArgmaxTieBreak
   --- PASS: TestArgmaxTieBreak (3.91s)
   PASS
   ok  	github.com/townsendmerino/goinfer/cuda	4.031s
   ```
   So the **audit is right and the D6 text was stale**. With P0.3 (Metal) already confirmed, both
   device thirds of P1's tie-agreement now hold; the host third is the amendment-1 contract in
   `topFilterLogits`, gated by `TestTopFilterLogits_MatchesReference`.
2. **Fix the stale texts** (`sampler.go` comment, ollama-chase §8 D6 + `top_k=1` section). Decide
   the CHANGELOG note by commit order: if `c6600fc` predates the v0.10.3 tag, the entry was wrong
   at press time and gets a correction line; if not, it stands.
   **DONE (2026-08-09).** Commit order decided: `c6600fc` is dated **2026-08-05**, the v0.10.3 tag
   commit `03b1832` **2026-08-08**, and `git merge-base --is-ancestor c6600fc v0.10.3` confirms it —
   so `c6600fc` **predates the tag** and the CHANGELOG entry was **wrong at press time**, not merely
   overtaken. It carries a dated correction line rather than a silent edit. Also flipped:
   `decoder/sampler.go`'s amendment-1 comment and ollama-chase §8 D6 (both now cite C-14 / `c6600fc`
   and the `cuda.TestArgmaxTieBreak` gate). The `top_k=1` prerequisite in D6 is therefore
   **satisfied on both backends** — P1 is unblocked.
3. **Confirm which Metal amax kernel is actually dispatched.** The `w4a8` fused amax carries an
   N-09 note ("NOT currently dispatched — no pipeline is created for it"); the int8 twin is the
   logit-critical one. P1's Metal claim depends on the *live* greedy path tie-breaking correctly —
   check `model.go`, don't assume from the kernel source.
   **CONFIRMED (Mac, 2026-08-09).** `ForwardArgmax` dispatches `gemv_w8a8_amax` + `argmax_finish`
   (`pGemvW8Amax`/`pArgFinish`, the int8 head; model.go:1011). `gemv_w4a8_sa_amax` is the N-09
   variant — unwired, pipeline dropped (model.go:410–412) — so the trap does not touch the live path.
   Both dispatched kernels merge on `(v desc, i asc)` → **lowest tied index** (kernels.go:264 and
   272–273), matching the ascending-id contract. So the **Metal third of P1's tie-agreement holds**;
   only P0.1 (CUDA `TestArgmaxTieBreak`, box-only) remains to confirm the CUDA third before P1.

## P1 — Route `top_k=1` to the greedy fast path — **DONE (2026-08-09)**

> **SHIPPED.** `Sampler.GreedyEquivalent()` (`decoder/sampler.go`) is a NEW predicate beside
> `ArgmaxEquivalent` — `TopK == 1 && !HistoryDependent() && !Logprobs`, unconstrained in temperature —
> and `generateInto` routes on `ArgmaxEquivalent() || GreedyEquivalent()`. Measured on this box
> (decode-only, prefill excluded; "before" = same binary with `GOINFER_NO_GREEDY_FASTPATH=1`):
>
> | model | `top_k=1` before | after | greedy | recovered |
> |---|---|---|---|---|
> | qwen2.5-coder-0.5b | 271.4 ±1.8 | **319.2** ±5.5 | 315.2 ±3.3 | +17.6% |
> | gemma3-1b | 144.3 ±0.8 | **175.9** ±1.0 | 175.9 ±3.0 | +21.9% |
>
> The before-figures reproduce the recorded gap (272 / 148) independently, and after routing
> `top_k=1` collapses onto greedy. Gates: `decoder.TestTopK1_MatchesGreedy` (byte-identical streams
> at temps 0.01/0.7/1.5, both vocab widths, tie-heavy logits), `TestGreedyEquivalent_predicate`,
> `TestGreedyEquivalent_stableAcrossHistory`.
>
> **Scoped deliberately: speculative eligibility is UNCHANGED.** The speculative paths gate on
> `sp.Temperature <= 0` directly (speculative.go:69, spec_grammar.go:179, spec_eagle.go:194,
> spec_ngram.go:176) and never consult either predicate, so `top_k=1`-with-temperature remains
> speculative-ineligible. Making it eligible would be correct but is a separate change; it does not
> ride inside P1.
>
> **Metal: pending.** The predicate is sampler-level so every backend inherits the routing, but no
> e2e A/B has been run on the Mac. Not claimed.

### Original plan (retained)


Measured gap: qwen2.5-coder-0.5b 272 vs 312 tok/s, gemma3-1b 148 vs 180. `top_k=1` at any
temperature is mathematically greedy: temperature scaling is monotone (order preserved) and a
one-token distribution is deterministic.

**Honest framing of the payoff.** The 13–18% applies ONLY to `top_k=1`-with-nonzero-temperature
requests. That shape is narrow but NOT purely pathological: some client libraries and eval harnesses
use `top_k=1` as their idiom for "deterministic output" instead of `temperature=0`, so real traffic
does hit it and those users get the 13–18% as a side effect. Still, it is not the traffic D6's rank
note points at (`temperature>0` broadly, which P2/P3 serve). So P1's real value is not the headline
number: it is (a) a correctness/consistency fix — `top_k=1` *should* be greedy-fast on every
backend — and (b) the cheapest place to **prove the C-14 tie-break agreement end-to-end** (host
`topKByLogit` vs device argmax vs Metal amax all resolving the same tied index) that P3 later relies
on. Treat it as the warm-up that validates P3's hardest assumption, with a real-if-narrow throughput
side effect — not a throughput win in its own right.

**Predicate — extend the sampler, not the backends,** so every backend inherits it. Add a
`GreedyEquivalent()` (or widen `ArgmaxEquivalent`, renaming honestly) that returns true for:

    TopK == 1 && len(LogitBias) == 0 && !penaltiesConfigured() && !Logprobs

- Use **`penaltiesConfigured`**, not `penaltiesActive`: Active is history-dependent (false before
  the prompt is Observed, true after), and the backend consults the predicate once per request.
  `HistoryDependent()` already exists with exactly the right shape — reuse it plus the Logprobs check.
- `top_p` / `min_p` at any value are safe alongside `TopK==1`: both cuts clamp at ≥1 retained token
  and the top token is always kept (`cut < 1 → cut = 1` in `topFilterLogits`).
- `Logprobs` excluded (needs the full distribution → full readback regardless).
- `LogitProcessor` continues to be checked at the call site, as for the existing branch.
- **RNG stream:** routing skips the per-token `rng.Float64()` draw. Unobservable — under
  `top_k=1` every step is deterministic, so no later draw can depend on the skipped ones.
- **Tie agreement is the whole reason this waited:** host `topKByLogit` keeps the smaller id on a
  tie; device argmax now returns the lowest tied index (C-14); Metal amax merges on `(v, lower i)`.
  All three agree with the ascending-id contract. P0 step 1 is the gate.

**Gates:** (a) sampler-level sweep — `top_k=1` at temperatures {0.01, 0.7, 1.5} byte-identical to
greedy across seeds and both vocab widths (extend `sampler_sweep_*_test.go`); (b) heavy e2e on the
CUDA box — expect ≈272→~312 and ≈148→~180, i.e. the `top_k=1` column collapses onto greedy;
(c) same sweep on Metal. CPU backend is unaffected (host argmax already, no readback to skip).

## P2 — Lazy Z — **BUILT, MEASURED, REFUTED (2026-08-09). Do not re-propose from intuition.**

> **NO-GO.** Lazy Z was implemented, proven CORRECT against a slow exact reference (432 matched
> draws across peaked / flat / tie-heavy distributions, plus boundary-straddling draws that force the
> grow path), and then measured **3.3× SLOWER at 152k and 4.4× slower at 262k** than the exact
> full-vocab softmax it was meant to replace. Reverted; the temperature-only path is unchanged.
>
> **Why, quantitatively.** The remainder bound `R = (V−K)·exp((x_K−m)/T)` is far too loose at a large
> vocabulary. For it to be useful you need `R < S_K`, i.e. a gap of `ln((V−K)/S_K) ≈ ln(150000/2) ≈
> **11.2 nats**`. Measured on REAL decode logits (qwen2.5-coder-0.5b, V=151936):
>
> | K | max − x_K | S_K | R | R/S_K |
> |---|---|---|---|---|
> | 32 | 5.29 nats | 2.085 | 763 | **366×** |
> | 256 | 8.97 nats | 2.227 | 19.27 | **8.65×** |
> | 2048 | 11.61 nats | 2.277 | 1.366 | **0.60×** |
>
> So K must reach ~2048 before the bound is even comparable to the retained mass — and there the
> interval still spans 60%, so most draws straddle a token boundary and grow again. Every growth step
> costs a full O(V) `topKByLogit` pass, so the algorithm does several full passes where the exact path
> does one. **The tail is not skippable this way**: real decode distributions are peaked in the top
> few tokens but not peaked enough to dominate 150,000 unseen ones.
>
> **What step 1's measurement established (still valid, and it is the useful output).** The
> temperature-only host cost is **~30–34 ns per vocabulary entry**, flat across widths: 1.00 ms/tok at
> 32k, 5.25 ms/tok at 152k, 8.00 ms/tok at 262k. Against the readback (0.12 ms on qwen, 0.43 ms on
> gemma3-1b, from P1's fast-path A/B), the host term is **~78% of the temperature-only penalty and the
> readback ~2%** — so this cost is real and worth attacking, just not by bounding Z. **P3 (device-side
> sampling) now owns essentially all of it**, which is the opposite of the earlier framing that treated
> the readback as the main term.

### Original design (retained, with its error corrected)

**DESIGN CORRECTION.** The resolution condition below originally read "resolves inside K when it
lands inside the top-K prefix for every Z in [S_K, S_K+R] (conservatively: `r < S_K/(S_K+R)`)". That
is **insufficient**: it pins which PREFIX the target lands in, not which TOKEN. The target is
`t = r·Z`, so as Z moves across the interval, t SLIDES along the prefix and can cross a token
boundary while staying inside the top-K — returning a different token than the exact path for
near-boundary draws. The correct condition is index equality at both endpoints:

    resolve iff index(r·S_K) == index(r·(S_K+R))

where `index(t)` is the token the CDF walk selects at unnormalized target t (descending-prob,
ascending-id ties, comparator `t < cum` matching `drawFull`). On disagreement, grow K and retry; at
K = V, R = 0, the interval collapses and the answer is exact by construction, so there is no separate
fallback path. The implementation used this corrected rule and was verified against it — the refutation
above is about COST, not correctness.



The temperature-only path (`drawFull(softmaxStable(...))`) normalizes all V every token: measured
**~44 ns per vocab entry, flat 32k→262k** (+1.19 / +6.66 / +11.57 ms at phi3/qwen/gemma widths), a
**3.1× throughput factor** at 262k — plain `temperature` is currently the slowest sampled config.
This is D6's "approach 1", host-side, and it also helps the pure-CPU backend (the readback half,
P3, is GPU-only).

**Measure first (half a day):** the temp-only-vs-greedy delta bundles readback + softmax + draw.
Instrument once to split copy-time from normalize-time per backend, so P2 and P3 each get an
attributable before/after instead of sharing one number. (The five-attribution lesson: profile the
unit before designing the fix.)

**Design** (recorded in D6 so it isn't re-derived): take top-K by logit with **no exp** —
`topKByLogit` already exists; sum the retained exp-mass `S_K` exactly (descending order); bound the
unseen remainder `R ≤ (V−K)·exp(x_K − m)` where `x_K` is the smallest retained logit. The draw
resolves inside K when it lands inside the top-K prefix **for every Z in [S_K, S_K+R]**
(conservatively: `r < S_K/(S_K+R)` resolves; near the boundary, grow K ×8 — `topPCandidates`'
adaptive shape — or fall back to the exact full pass, deterministically). Logprobs requested →
full path always, unchanged.

**Contract cost — name it, don't slip it:** `drawFull` walks the CDF in ascending token-id order;
Lazy Z resolves over a descending-prob prefix. Same distribution, different r→token mapping, so
**given-seed temp-only output changes once**. This is exactly the v0.10.3 amendment class
(tie/boundary re-specification, distribution unchanged). Amend the contract: temperature-only
draws resolve in descending-prob / ascending-id order, matching the filtered path. Gate
bit-for-bit against a slow reference that shares the new order (the `refTopFilter` pattern), plus
the existing throughput-ratio gate extended to temp-only vs greedy.

**The load-bearing gate case is near-boundary `r`, not the happy path.** The bug surface is where the
draw's CDF position falls inside the interval `[S_K, S_K+R]` — the code must then *deterministically*
grow K (or fall back to the exact full pass), and the SAME `(logits, seed)` must always take the same
branch or the output is nondeterministic. The reference gate must therefore include adversarially
constructed near-boundary draws (an `r` engineered to land just inside/outside `S_K/(S_K+R)`, and
flat distributions where R never shrinks), the same way the v0.10.3 `topFilterLogits` gate needed
tie-heavy and boundary inputs. A gate that only tests peaked distributions passes while the boundary
logic is wrong.

**Seed-churn discipline.** This is the SECOND given-seed output change in two releases (v0.10.3
re-specified the filtered path; this re-specifies temp-only). Do not dribble a third: bundle any
other pending seed-affecting sampler change into P2's release, and state once, plainly, in that
release's notes — "sampled output for a given seed changed in vX; the distribution is unchanged" —
rather than surprising seed-pinning users across successive versions.

**Expected:** most of the ~44 ns/entry disappears (the exp+sum over V); the remaining temp-only
gap becomes the readback, which is P3's job. Optional follow-on, not scoped: the same
interval-resolution trick can sometimes decide the exact top-p cut without the full Z pass (if the
cut index agrees at both interval endpoints); bank it unless profiling shows the Z pass matters
after P3.

## P2b — Deterministic parallel host normalization — **DONE (2026-08-09)**

Replaces the refuted P2. Lazy Z tried to SKIP the full-vocabulary exp+sum; this does the same work
in parallel instead, and deletes the normalize-divide pass by drawing against unnormalized weights.

**Motivating attribution (bc59c56)** — the temperature-only penalty splits as:

| | phi3-mini 32k | qwen2.5-0.5b 152k | gemma3-1b 262k |
|---|---|---|---|
| host normalize+draw | 1.00 ms/tok | 5.25 ms/tok | 8.00 ms/tok |
| per vocabulary entry | 31.3 ns | 34.5 ns | 30.5 ns |
| logit readback | — | 0.12 ms/tok | 0.43 ms/tok |

i.e. **host ~78%, readback ~2%** — so the host term was the whole game, and P3's premise was wrong.

**The load-bearing design decision: `numChunks` is a COMPILE-TIME CONSTANT (64), never
`runtime.NumCPU()`.** Workers schedule over chunks freely; the REDUCTION is folded in ascending
chunk index. Float addition is not associative, so the grouping decides Z's last ULPs and therefore
which side of a boundary a near-boundary draw lands on — if chunk shape followed core count, the
same seed on the same build would emit different tokens on a 4-core and a 16-core machine.
`TestChunkedSoftmax_MachineIndependent` runs the same draws at GOMAXPROCS 1/2/8 and requires
identical output *and* identical Z, so a completion-order reduction or a NumCPU-sized split is
unwriteable rather than merely discouraged.

**Measured, host-only** (16 cores): 1.62× (32k), **3.06×** (152k), **4.72×** (262k) — per-entry
31.3 → 17.3, 34.5 → 10.0, 30.5 → 7.0 ns. The gain grows with vocabulary, the opposite of what a
core-count-sized split would do.

**Measured, end-to-end** (decode-only, prefill excluded; RTX 2070 SUPER / driver 595.58.03; q4_K_M
int4; 128-token prompt; 8 completions × 2 runs; "before" = the P1 binary ed81e13):

| model | config | before | after | |
|---|---|---|---|---|
| phi3-mini | temp-only t=1.0 | 88.8 ±6.4 | **113.7** ±5.4 | +28% |
| qwen2.5-0.5b | temp-only t=1.0 | 97.6 ±9.8 | **220.2** ±3.8 | **+126%** |
| gemma3-1b | temp-only t=1.0 | 56.6 ±7.3 | **134.2** ±0.2 | **+137%** |
| phi3-mini | temp0.8+top_p0.95 | 81.5 ±1.5 | **94.7** ±2.3 | +16% |
| qwen2.5-0.5b | temp0.8+top_p0.95 | 93.4 ±0.4 | **184.7** ±10.6 | +98% |
| gemma3-1b | temp0.8+top_p0.95 | 56.3 ±0.1 | **117.0** ±0.4 | +108% |

Step 3 (bundling top-p's Z pass) was specified as a timing requirement — land the seed shift once —
and turned out to be a performance win too, since that denominator paid the same full-vocab exp-sum.

**WHAT P3 COULD STILL BUY — the residual.** On qwen2.5-0.5b, temperature-only is now **220.2** against
greedy's **320.1**. The remaining gap is the readback (0.12 ms/tok measured) plus the parallel host
term (~1.53 ms/tok), i.e. **P3's device-side reduction addresses only the readback portion**, which
this box measures at ~2% of the original penalty and a small fraction of what remains.

**Seed churn spent.** Both Z passes are regrouped, so given-seed sampled output changes once, for
temperature-only and top-p together. Distribution unchanged.

## P3 — Device-side top-K logit reduction — **BANKED (relay close). NOT "the campaign that finishes D6".**

> **FINAL (relay close, 2026-08-09): BANKED.** After P2b, the only thing P3 could still buy is the
> per-token logit **readback** — measured at **0.12–0.43 ms/token**, ~**2% of the original sampled
> gap**. Not worth its reopen price: **device-exact-Z top-K kernels on both backends** (CUDA + Metal,
> under the bit-identity lint) plus a **per-backend given-seed divergence** to re-specify and gate
> (each backend's Z-reduction order differs from the host's). A 2% ceiling behind two device kernels
> and a third seed-contract change is not a fundable trade. Reopen only if the readback ever becomes a
> material fraction of the gap again (it will not without a much faster decode path underneath it).
>
> **RE-SCOPED (2026-08-09). Read this before building kernels.** This section previously read as the
> campaign that finishes D6. Two measurements since have removed that footing:
>
> 1. **The interval-verifier design is refuted for temperature-only.** P3's host-side half was to
>    verify the returned K carry mass ≥ top_p and fall back on a short read — the SAME interval
>    argument Lazy Z used, over the SAME distributions. bc59c56 measured that argument dead at these
>    vocabularies: the remainder bound needs an ~11.2-nat gap and real decode logits give 5.29 nats
>    at K=32 (R/S_K = 366). A device top-K does not change the arithmetic; it only moves where the
>    K is chosen. Any revival needs **device-exact Z**, not a bound.
> 2. **The readback it removes is ~2% of the penalty, not the bulk.** Measured attribution: host
>    normalize+draw ~78%, readback ~2% (0.12 ms/tok on qwen 0.5B, 0.43 on gemma3-1b). P2b then took
>    the host term down 3–4.7×, so what P3 addresses is a small fraction of a now-smaller gap.
>
> **Cost side, unchanged and non-trivial:** new kernels on TWO backends, both device boxes, plus a
> per-backend given-seed divergence — device `exp` is not bit-identical to host `math.Exp`, so a
> device-exact Z makes sampled output differ *between backends* for the same seed, which the
> host-only work so far has carefully avoided.
>
> **Decision: BANKED.** Re-open only if the P2b residual is measured to matter on a real workload —
> the number to beat is qwen2.5-0.5b temperature-only at 220.2 vs greedy 320.1, of which the readback
> is ~0.12 ms/tok. Do not start from the text below without re-reading this box.

### Original design (retained, premise now falsified — see above)


D6's "approach 2", subsuming the readback branch entirely: CUDA + Metal kernels return the K
highest (logit, id) pairs (K fixed, 256–1024) instead of the V-wide row — the ~608 KB/token copy
at 152k vocab (2× that at gemma3's 262k) disappears. The host runs the *existing exact sampler*
over the returned K:

- **Sufficiency check, host-side:** for top-p, verify retained mass ≥ `topP·Z`-equivalent via the
  P2 interval bound; for temp-only, the P2 resolution logic. On a short read, **fall back to a full
  V readback for that token** — bit-identical to today whenever K suffices, identical-by-fallback
  otherwise. P2 is therefore a prerequisite (its interval machinery is the verifier).
- **Kernel shape:** FMA-free compare/merge tree (same tie-break contract: value desc, id asc), so
  it sits under the bit-identity lint trivially. CUDA: new `.cu` → own PTX (the `argmax.cu`
  precedent — do not regenerate glue.ptx at 12.9). Metal: extend the amax partial-reduction
  pattern from single-max to K; mind the N-09 unwired-variant trap and Metal's dispatch-count
  sensitivity (one extra dispatch per token is fine; per-layer would not be).
- **Gates:** crafted-logits equivalence vs host selection (both backends, both vocab widths,
  tie-heavy AND near-boundary sufficiency-check cases — the same boundary surface as P2, since P2's
  interval bound IS the short-read verifier); e2e sweep — temperature-only and temp+top_p converge
  to within the selection cost of greedy; fallback-path exercised (adversarial flat distribution)
  and byte-identical to the full path; `TestMetalSnapshotGolden` untouched (selection reads logits,
  never writes).
- **Bound the fallback COST, not only its correctness.** A flat/adversarial distribution can make
  EVERY token short-read. Note what the fallback actually costs: the V-wide logit row already lives
  in VRAM after the forward, so the fallback readback is a copy of an EXISTING buffer, not a
  recompute. The worst case is therefore today's copy + the top-K reduction + one extra host-side
  decision round-trip — i.e. **two syncs per token instead of one**; the regression risk is added
  sync latency, not duplicated compute. So the perf gate should measure **per-token latency overhead
  vs today** on the adversarial-flat case (not just "byte-identical"), and the "stage alongside"
  remedy reduces to "don't free/overwrite the logit row until the host has verified sufficiency,"
  which is nearly free. Decide from the measured number, but expect the overhead to be small.
- **Expected:** the reporter's ~2.9× nonzero-temperature penalty collapses to ≈ the P2 residue;
  gemma3-1b (widest vocab, largest readback) is the headline number.

## P4 — Long-context decode: KV-cache quantization, opt-in — **DECIDED (2026-08-09, `d4472b9`): reachability-only on BOTH backends; build DEFERRED**

> **FINAL (relay close): reachability-only on BOTH backends, a speed lever on NEITHER.** CUDA: the
> §B7 bandwidth falsification (deep decode moves KV at 6–10% of peak → bandwidth 90% idle). Metal:
> the half-width KV probe (`d4472b9`, box below) — int8 KV cost 88% of f16, so halving the bytes moved
> attention only 12%; latency-bound, not bandwidth. So q8 KV cuts VRAM (reachability) and buys no
> decode speed on either backend. **Build DEFERRED until a real capacity demand exists** — there is no
> current 8 GB-card workload that needs deep context it cannot fit, so shipping a lossy opt-in cache
> now is a quality surface with no consumer. **If built, ladder f16 KV first** (2× cut from the
> current f32 resident cache, minimal quality surface — dot/accum already run in f32) and only then q8
> (4×), each behind the argmax-faithful + cosine-floor quality lane as scoped below. The decode SPEED
> lever both depth verdicts point at is the occupancy/latency (FA-class) rewrite — non-bit-identical,
> unfunded, and P7's would-be first constituent.
>
> **DECIDABLE — the deep coefficients landed (P6, §B7). Split verdict: NO-GO as a CUDA speed lever,
> GO as a reachability lever.** The deferral condition ("until the cap-raise leg delivers 8k–32k
> coefficients") is discharged.
>
> **The premise that deferred this is now refuted in KV-quant's favour, and it still does not save the
> speed case.** The per-position term is *not* near-zero at depth — on qwen0.5b it rises from
> +0.330 (512→2048) through +0.531 (2048→3900) to **a plateau of +0.713 / +0.748 / +0.735 µs/pos**
> across 3900→8192→16384→32000. So there is a large depth term to attack, which is what P4 was waiting
> to see, and it is a *constant* one: decode is linear in depth past ~8k, ~25× the peer's per-position
> cost (0.735 vs 0.029). But *what bounds it* is the question KV-quant's value actually turns on, and
> the arithmetic answers it:
>
> | model | depth | engine | KV MB read/token | GB/s | % of 448 GB/s peak |
> |---|---|---|---|---|---|
> | 0.5B | 16384 | goinfer | 393.2 | 27.8 | **6.2%** |
> | 0.5B | 16384 | Ollama | 393.2 | 94.3 | 21.0% |
> | 1.5B | 16384 | goinfer | 917.5 | 44.3 | **9.9%** |
> | 1.5B | 16384 | Ollama | 917.5 | 141.7 | 31.6% |
>
> **goinfer's deep decode moves KV at 6–10% of peak DRAM bandwidth.** A byte-count reduction cannot
> be the lever when bandwidth is 90% idle. The comparison seals it: **Ollama is 3.2× faster while
> reading identical bytes** — the difference is latency hiding (occupancy / flash attention), not
> traffic.
>
> *The pessimistic read model is falsified, not assumed away.* If each GQA query head re-read its KV
> head's bytes (×7 on the 0.5B, ×6 on the 1.5B), Ollama would sit at **147–190% of peak — physically
> impossible**. So the bytes really are read ~once. Even under that falsified upper bound goinfer
> never exceeds 59% of peak, so the conclusion holds under either model. This is arithmetic from
> measured throughput plus known cache geometry, not an ncu profile — but the bracket is wide enough
> that a profile could only refine the number, not flip the sign.
>
> **What survives: reachability.** q8 KV is a **4× byte cut from f32** (not 2× — the resident cache is
> f32, corrected below), so it is what makes deep context *fit*, and that is a VRAM claim, not a tok/s
> claim. Concretely, the 1.5B's KV is 1.88 GB at 32k f32 → ~0.47 GB at q8 on an 8 GB card. Ship it
> for capacity and say so; do not promise CUDA decode speed from it.
>
> **The CUDA speed lever the coefficients actually point at** is the same one Campaign A closed at:
> occupancy/latency in the deep-context attention path. That is the §7 non-bit-identical fork or a
> flash-attention-shaped rewrite, not KV-quant.
>
> **Metal is NOT a speed exception — the deciding half-width probe (Mac, 2026-08-09) closes it.** The
> "75% of attention is distinct per-key DRAM reads" finding was the *pin-every-read-to-key-0* collapse
> (21.5→5.3 ms), which drops BOTH bytes and latency, so it could not tell bandwidth from latency. The
> third arm — int8 KV, half the bytes per key at the SAME element count / ALU (q8's exact profile) —
> answers it (all-28-layer attention @2048, M1 Pro, min GPU-busy; `TestZZ_attnKVWidthProbe`):
>
> | arm | time | vs full |
> |---|---|---|
> | full (f16 KV) | 16.62 ms | — |
> | q8 (int8 KV, half the bytes) | **14.66 ms** | **88%** |
> | pin0 (key-0, zero distinct DRAM) | 5.22 ms | 31% (3.2× collapse — control) |
>
> pin0 reproduces the 2026-08-04 floor (5.3 ms) → harness sound. **Halving the KV bytes moved
> attention only 12%, nowhere near proportional** — so Metal decode attention is latency/occupancy-
> bound too, NOT bandwidth-bound; the distinct per-key reads are serial-latency-exposed, and reading
> fewer bytes per key barely helps while bandwidth sits idle. **q8 is therefore NOT a Metal speed
> lever.** The earlier "Metal-first for the quantization *speed* work" rationale is refuted; P4 is
> **reachability-only on BOTH backends** (q8 for VRAM/capacity, speed on neither). The Metal decode
> speed lever, like CUDA's, is the occupancy/latency rewrite, not KV-quant. ("No diagnosis transfers"
> held — but so did the *conclusion*, once the byte term was isolated on each backend.)
>
> **Corrected KV geometry (measured, not assumed).** The resident cache is **f32**, K+V:
> **24.0 KB/position** (qwen0.5b: 24 layers × 2 KV heads × 64 head-dim × 2 × 4 B) and
> **56.0 KB/position** (qwen1.5b: 28 × 2 × 128 × 2 × 4). The plan previously reasoned from an f16
> baseline; from f32, **q8 KV is a 4× byte cut on CUDA, not 2×**, which strengthens the reachability
> arithmetic proportionally. At 32k positions the 1.5B needs ~1.88 GB f32 / ~0.47 GB q8 beside the
> weights on an 8 GB card — feasible, but it is a real budget, not headroom.

### Original P4 plan (retained; CUDA case now deferred — see the box above)

Where things stand, honestly: CUDA Campaign A **closed at the bit-identity ceiling** — split-KV
landed (99.5→160 tok/s @2048 cumulative, **1.17× behind** Ollama), the V-sum ILP unroll was tried
and refuted, and pushing past needs a non-bit-identical reduction. Metal closed harder: four
independent confirmations that decode attention is dispatch-/occupancy-bound, dedup layouts dead,
A1-Metal's 1.37–1.40× the one capturable win. The notes' own conclusion: "the lever is elsewhere
(KV-quant §A3, or accept the floor)." This plan takes A3 — **opt-in, never default**, per its
scoping.

- **P6's extended depth sweep feeds this gate directly** (per-position coefficients and KV
  bytes/position at 8k–32k) — run it first or alongside the profiles below. At agentic depths KV
  size, not tok/s, is the first wall (~28 KB/position ⇒ ~3.6 GB at 128k), so q8/q4 KV is what
  makes deep context *reachable* on an 8 GB card before it is a speed win.
- **Profile first, per backend (the A2-Metal lesson: no diagnosis transfers).** Metal: the collapse
  probe already showed **75% of attention time is distinct per-key K/V DRAM reads** (~16 ms/tok
  @2048) — re-run the probe with half-width reads to bound q8's win before building. CUDA: ncu the
  `splitkv_scores`/`vsum` pair for DRAM vs occupancy share at 2048–4000 ctx; if the depth term is
  occupancy-bound (as the residual was), q8 buys little on CUDA and the build is Metal-first.
- **Build:** q8 KV (per-head or per-group scales), quantize at cache-write so prefill and decode
  read the same bytes; q4 later only if q8's quality gate holds with margin. Opt-in flag
  (`--kv-quant q8` shape), decode and prefill paths both covered or the flag declines.
- **Gate — this is NOT bit-identical, so it gets its own lane, not a tolerance-relaxed existing
  one:** the repo precedent is the 26B pager ("lossy but argmax-faithful, ≥0.99 cosine vs the f32
  cache"). Same shape: per-family argmax-faithfulness + cosine floor vs the f32-KV run at depth
  (past 2048), on the parity-matrix families that opt in; plus long-decode stability (no drift
  cliff at 4k). Windowed + hd=256 (gemma3) exercised explicitly.
- **Expected:** Metal long-context decode is the target (its floor is the product complaint);
  CUDA number only if the profile says bandwidth. Also halves KV VRAM at depth — worth reporting
  even where tok/s moves little.

**Not reopened by this plan:** §7 rotation+IMMA stays closed on its recorded conditions (M=1 GEMV
DRAM SpeedOfLight showing the ~11% largely realized AND a funded rotation campaign); B2 dp4a
headroom stays retired; Metal split-KV/dedup stay dead; the tolerance-gated-flash middle path
stays banned (§7's trap argument).

## P5 — Prefill: B1 path 2 (reduction-order re-baseline, ~1.3×) — optional, fund explicitly

Prefill is past its usability threshold (2.1 s @2048) and cannot reach parity under the current
format (§3b/§7) — so this runs **only if prefill speed is wanted for its own sake**, and the plan's
default is to leave it banked. If funded, take **path 2** (deterministic per-query denominator,
Bk-free) over path 1 (~1.15×, intricate, multi-session): one coordinated goldens refresh, done
once — it cascades to decode + `attn_batched` + `splitkv_*`, so the re-baseline is a single
release-boundary event with the Metal snapshot-golden re-bake following its own two-branch refresh
protocol (env-same → investigate; this is the "verified change" arm). This is a
*re-specification*, still exact and deterministic — not the tolerance trap. Gate: divergence-rate
tests re-pinned at the new baseline, full parity manifest, both backends, before the tag.

## P6 — Extend the depth sweep — **COMPLETE (2026-08-09). Cap raised, deep cells measured.**

> **DONE. The 8k+ blocker is gone and the cells are measured.** `cudaCtxCap` is configuration-derived
> (`-ctx`, `ca29d6c`), and the deep sweep ran at `-ctx 32768`. Full numbers, method and the scope
> decision: `docs/benchmarks.md` **§B7**.
>
> **Headline — the regime change is real, and it COMPLETES rather than compounding.** goinfer's
> per-position decode cost rises from **+0.330 µs/pos** mid-range to a **plateau of ~+0.74** (0.5B) /
> **~+1.0** (1.5B) reached by ~8k, and the 16384→32000 probe pins it flat (**+0.748 → +0.735**). Deep
> decode is therefore **linear in depth with a large fixed constant, not superlinear** — deep context
> is an optimization target with a predictable cost, not unreachable in principle. Against the peer
> that constant is a flat **~25×** (0.735 vs 0.029); the 5.54× throughput gap at 32k is that constant
> integrated over depth, not an accelerating divergence.
>
> | depth | goinfer 0.5B | Ollama 0.5B | goinfer 1.5B | Ollama 1.5B |
> |---|---|---|---|---|
> | 3900 | 200.8 | 259.8 | 122.1 | 174.3 |
> | 8192 | 124.4 | 259.1 | 80.7 | 164.0 |
> | 16384 | 70.6 | 239.7 | 48.3 | 154.4 |
> | 32000 | 39.0 | 215.9 | *skipped* | *skipped* |
>
> **And the bound is NOT bandwidth**: goinfer moves KV at **6–10% of peak** at depth while Ollama is
> 3.2× faster reading *identical* bytes; the competing "each GQA head re-reads K" model is falsified
> because it would put Ollama at 147–190% of peak. That is what makes **P4 decidable** (above): no-go
> as a CUDA speed lever, go as a reachability lever.
>
> **Scope, recorded not silent:** the 1.5B/32000 pair was **deliberately skipped** (45–70 min for a
> number that changes no decision, once the plateau was established by the 0.5B probe). `-ctx 32768`
> remains functionally verified by the cap-raise gates. §B7's header states this; a future campaign
> can fill the cells against the same anchor.
>
> **Deep cells measure the corrected kernel selection** (`2693dce`, split-KV re-gated per geometry):
> at 8192+ both models are on the split-KV path. Pre-`2693dce` curves are not comparable without that
> caveat.
>
> **Control (§B7.1): the anchor holds in BOTH arms.** The shallow depths reproduce on the cap binary
> at the default cap *and* at `-ctx 32768`, every cell within 2.6% (most within 1%) — so the cap
> change is allocation-only and a 32k cap costs nothing at shallow depth, which is what makes the
> deep coefficients trustworthy. Two cells needed a re-run after being contaminated by GPU contention
> with the just-killed peer server; recorded in §B7.1 rather than quietly replaced.
>
> ---
> *Historical (the blocker this leg removed):* **8k/16k/32k were BLOCKED on `cudaCtxCap = 4096`** —
> the resident KV
> capacity is a compile-time constant, so `checkCap` refuses those depths and the request falls to
> the staged path. A staged number is a different engine under the same label, so those cells were
> NOT measured (option 3 rejected). Unblocked by the cap-raise leg: cap becomes
> `min(model context, serve setting)`, VRAM-checked at load with a fail-fast error, default
> behaviour unchanged.
>
> **The published 3900 ceiling was the cap showing through.** Every depth curve this project has
> published tops out just under 4096 — that was infrastructure, not a chosen depth, and it was never
> stated as such.
>
> **DONE in this leg:** the truncated greedy sweep {128, 512, 2048, 3900} on qwen 0.5B/1.5B, and the
> full G11 sampled refresh (three models, both engines, anchored to one binary). Numbers and full
> provenance: `docs/benchmarks.md` §B5. Metal depth leg (M1) **pending, not claimed**.
>
> **Headline coefficients (µs per KV position):** 0.5B goinfer +2.542 / −0.009 / +0.485 across the
> three segments against Ollama's −0.007 / +0.029 / +0.051; 1.5B goinfer +2.349 / +0.554 / +0.987
> against +1.419 / −0.060 / +0.090.
>
> **THE 128→512 STEP IS A KERNEL SWITCH, NOT DEPTH.** It measured ~2.4–2.5 µs/pos on both models
> despite different layer counts and head dims — impossible for a true per-position cost. A 6-cell
> A/B (`GOINFER_SPLITKV_ATTN=0`, with d=128 as the below-gate control) attributes it to split-KV
> attention engaging at `splitkvMinKeys = 256`:
>
> | depth | split-KV ON | split-KV OFF | |
> |---|---|---|---|
> | 128 | 314.9 ±3.6 | 319.2 ±1.3 | equal (gate inactive) — control |
> | 512 | 239.7 ±4.4 | **290.0 ±5.6** | OFF 21% faster |
> | 2048 | 242.5 ±7.9 | **249.3 ±1.8** | OFF faster |
>
> Split-KV costs the 0.5B **~0.72 ms/token at 512** and ~0.11 ms at 2048. With it off the coefficient
> is an ordinary +0.820 / +0.366 µs/pos. Its crossover was characterized on the **1.5B** yet it is
> **default-on for every model** — on the 0.5B it is a net loss at both measured depths, in the range
> where goinfer otherwise leads. Follow-up: per-geometry gating or a re-characterized crossover.
> **Not launch latency** — CUDA graphs measured 1.01×, so this is small-kernel execution overhead;
> do not re-propose graphs (§10).

## P6a — Re-gate split-KV decode attention — **DONE (2026-08-09)**

P6 found the 128→512 "depth" step was a kernel switch and named the split-KV gate as the suspect.
P6a characterized it properly and replaced it. **The finding is worse than P6 assumed.**

**What was measured.** 48 cells — 4 resident geometries × 6 KV depths × {ON, OFF} — each a freshly
started `serve`, int4, greedy, decode-only inter-token rate, token-calibrated prompts. Then a
refinement pass on the fixed binary. Full table, method and derivation: `docs/benchmarks.md` §B6.

**Two separate defects in the shipped `splitkvMinKeys = 256`:**

1. **It was wrong on its own geometry.** The constant came from `TestSplitKVCrossover` on the
   **1.5B** ("break-even 256, clear win from 384+"). The 1.5B *loses* at 256 (0.941) and 512 (0.939);
   its real crossover is (512, 1024]. The cause is methodological — that test times a tight
   in-process `ForwardArgmax` loop at **best-of-3 minimum**, which hides the per-token CPU dispatch a
   real request exposes and favours the higher-variance arm (ON). Recorded in its docstring so the
   number cannot be re-derived the same way.
2. **One geometry's constant was applied to all.** qwen0.5b breaks even at ~2560, not 256.
   **phi3-mini never crosses over at any depth** — 0.993 → 0.969 → 0.919 → 0.815 → **0.754**,
   monotonically worse with context.

**Formula vs lookup — decided by phi3.** A formula gate has the shape "ON iff nWin ≥ f(geometry)",
which *always* predicts ON wins at sufficient depth. phi3 falsifies that **shape**, not just its
constants, so no threshold-in-nWin law can be right. (Two candidate laws, `nLayers/(nH·hd)` and
`nLayers/(nKV·hd)`, also missed the 0.5B crossover by ~2×.) Shipped as a measured per-geometry table
with a "never" class and a conservative default. The mechanism that explains all four curves —
split-KV trades **shared**-memory scores for a 3×-touched **global**-memory nH×nWin array in exchange
for occupancy, so `net(nWin) ≈ (A−B)·nWin − 2·L·T_launch` and A<B means never — is documented beside
the constants rather than compressed into a fitted number.

**A structural bug fixed at the same time: the gate tested the wrong quantity.** It compared raw
position `nKeys`; the kernel's cost is set by the **effective attended span** `nWin`. gemma3-1b's
window-512 layers were therefore taking the split path at a 512-key span — its loss regime — at every
depth past the window. Now gated **per layer on `nWin`**; both arms are byte-identical, so a layer
switching arms mid-request is safe by construction.

**Asymmetric loss is load-bearing.** Firing early costs up to 18–25%; firing late costs a few percent
(OFF's slope near the crossover is mild). Thresholds are rounded **up**, and unmeasured geometries get
the conservative default instead of an extrapolation. This deliberately forfeits ~2% at the 0.5B's
2560 break-even.

**Scope: kernel selection only.** No kernel, no numerics, and no weight path changed; output is
byte-identical on both arms. `GOINFER_SPLITKV_ATTN=0` still force-disables, and a new
`GOINFER_SPLITKV_MIN_KEYS=<n>` overrides the threshold on a stock binary (0 ⇒ always split) — needing
a rebuild to re-characterize is part of why a refuted number survived a release.

**Not device-portable.** The occupancy term scales with SM count; every cell is one 40-SM Turing part.
On a wider GPU phi3's "never" would not hold. Re-measure per device class; do not rescale on paper.

**What it recovered** (decode-only tok/s, new default vs old default; full table `docs/benchmarks.md`
§B6.1): 0.5B@512 **240.0 → 286.6 (1.19×)**, 0.5B@2048 240.0 → 250.2, 1.5B@512 182.2 → 195.0,
gemma3-1b@512 152.7 → 167.7 (1.10×), @1024 147.4 → 161.5, @2048 147.3 → 155.6.

Two results worth keeping:

- **The gate still fires where split-KV wins.** 1.5B@1024 is unchanged by design — new default 182.6
  ≈ all-split 182.3, both beating all-single 169.1 by 1.08×. This is a re-gating, not a retreat.
- **gemma3-1b@3900 beats *both* uniform arms** — 163.0 vs 154.5 all-split and 142.4 all-single —
  because its global layers take the split path while its window-512 layers do not. **No per-model
  gate can reach that point**, whatever constant it uses: the right answer differs between layers of
  the same model at the same position. That is the argument for gating per layer on `nWin`, and it is
  measured, not asserted.

**Anchored depth rows re-measured** on the fixed binary (`docs/benchmarks.md` §B6.2); the `686c9f8`
rows stay as published. Cells whose gate decision did not change re-measure unchanged (0.5B 128
318.9→320.9, 3900 200.1→200.8; 1.5B 2048 157.5→157.0, 3900 122.3→122.1) — a selection change moved
only the cells whose selection flipped. **One peer comparison flips in goinfer's favour:** 0.5B at 512
was 243.2 vs Ollama 269.6 (Ollama ahead 1.11×) and is now 286.6 (**goinfer ahead 1.06×**). **The G11
sampled cells were NOT affected and not re-run** — a 129-token prompt tops out at `nKeys` 193, below
even the old 256 gate, so no sampled figure and no README row moves because of P6a.

**Gates:** `TestSplitKVGate_measuredGeometries` pins the selection decision for all four geometries at
representative depths, so a future "simplify back to one constant" goes red. Both bit-identity tests
pass on GPU (`TestSplitKV_bitIdentical` at depth 2048, `..._gemma3` hd=256/windowed at 1536) and now
force the split path explicitly — with thresholds raised they would otherwise have compared
`attn_batched` against itself and passed **vacuously**. Depths where the gate decision is unchanged
re-measure unchanged (0.5B@128 318.9 → 320.9), confirming no collateral perturbation.

### Original P6 plan (retained; 8k+ blocked — see the box above)

The current depth axis (128–3900) covers the chat regime. Measured agent workloads run far
deeper: a 30k-request sample of the author's Claude Code transcripts puts per-request context at
**p10 128k / p50 439k / p90 871k / max ~1M**. Those are hosted-model depths, but the local-serving
analog — "the full window of whatever fits" — is still 8–30× past the deepest current cell. Two
things the shallow sweep cannot see: (a) the depth term dominates out there — extrapolating the
measured 0.5B coefficients (goinfer ~0.55 µs/position, Ollama ~0.035) turns the 1.11× gap at 2048
into **~4× at 32k**; (b) KV memory binds before throughput (~28 KB/position on the 1.5B ⇒ ~0.9 GB
at 32k, ~3.6 GB at 128k), which re-ranks **P4 from a throughput lever to a reachability lever**.

- Add 8192 / 16384 / 32768 cells to the calibrated depth A/B for 0.5B and 1.5B (32k is those
  models' native cap), same token-calibrated-prompt discipline that caught the attempt-1 depth
  inflation. CUDA box first; Metal after (M1 unified memory holds 32k KV easily).
- **Harness checks before trusting a cell:** the serve-side context/`max_tokens` clamps (C-18/C-20
  class) must admit a ~32k prompt; build depth via batched + KV-only prefill (sequential prefill
  at 32k would dominate the run); confirm KV VRAM at 32k fits beside the weights on the 2070S.
- **Peer discipline:** set Ollama's `num_ctx` explicitly per cell — its default silently truncates
  context, which would fake a flat curve (same error class as attempt 1's inflated depths);
  confirm flash attention is active; re-pin the version before publishing.
- **Report the coefficient, not just tok/s:** µs per KV position per backend per depth range, plus
  KV bytes/position. Those are the numbers P4's profile step consumes directly, and the honest
  form of the crossover claim — the 0.5B data already shows crossover depth is per-model (128–512
  there vs ~1000 on 1.5B), so publish coefficients and let the crossover fall out.

## P7 — `--mode exact|fast`: one word for "give me Ollama's deal" — **DEFERRED (relay close): precondition unmet**

> **DEFERRED (2026-08-09).** P7 aggregates *landed, output-changing* levers into one flag. There are
> none: P4 is build-deferred (reachability-only), P5 is banked, and the only Metal lever that would
> qualify (`GOINFER_METAL_BATCHED_PREFILL`) is a single, quality-ungated path — not a stack worth a
> mode switch. Shipping `--mode fast` now would be a switch wired to almost nothing. **Revisit only
> if P4 builds or an FA-class attention path is funded.** Note the occupancy/latency rewrite both
> depth verdicts point at is non-bit-identical — it would be P7's *first real constituent* if it ever
> exists. The design below stands as the spec for that day.

Today the escape hatch from the bit-identity contract is scattered across env vars and flags
(`GOINFER_METAL_BATCHED_PREFILL=1`, P4's future `--kv-quant`, whatever P5 adds), each discovered by
reading this backlog. A user who wants what the peer offers — speed with no exactness contract,
which is Ollama's *only* mode — has no one-word way to say so. Add `--mode exact|fast`, `exact`
the default. It also sharpens the published story: an exact-mode cell and a fast-mode cell against
Ollama is the fair fight plus a measured price tag on the contract, instead of only competing with
one hand tied.

**Design constraints — each one is load-bearing:**

- **Explicit printed expansion, never a behavior.** `fast` resolves at startup to a concrete flag
  set and prints it on the startup line (the derived-body-caps precedent): e.g.
  `mode=fast ⇒ kv-quant=q8, metal-batched-prefill=on`. The expansion is a documented,
  changelog-tracked list — its meaning changing across versions is fine *only* because every run
  states what it resolved to. A bare `--fast` whose meaning drifts silently is the §7
  middle-path trap in mutated form: the danger was never that lossy modes exist, it was ambiguity
  about what is being promised.
- **Explicit flags override the preset**: `--mode fast --kv-quant off` composes, explicit wins.
- **Aggregates only output-changing levers.** Bit-identical wins (split-KV, coalescing, batched
  prefill where it is byte-exact) are just defaults and stay out of the preset — otherwise the
  flag muddies what fast actually costs.
- **Its own gate lane, before it ships.** Nothing currently tests the non-bit-identical levers
  *in combination*. Each constituent needs its quality gate first (the argmax-faithful /
  cosine-floor lane — the 26B pager precedent; note the Metal batched prefill it would enable is
  54% token-divergent with quality per se currently ungated), plus one e2e fast-stack smoke per
  backend so the combination can't rot unobserved.
- **Sequencing honesty: this is a capstone, not a quick win.** On CUDA today the preset would
  enable almost nothing — the levers it aggregates are exactly P4 and P5, unbuilt. Shipping it
  early is a switch wired to one Metal path. Land P4 (and P5 if funded) first; P7 is then a
  small flag-plumbing change plus docs.
- Two different sacrifices may eventually shelter under it — deterministic-but-rebaselined (P5)
  vs lossy (P4). `--mode` (vs a bare boolean) leaves room for a middle tier if one ever earns
  its way in; until then `fast` bundles both and the startup line says so.

## Order, cost, and what each buys

| phase | final status | outcome |
|---|---|---|
| P0 truth maintenance | ✓ `31829de` | C-14 confirmed live; stale "open" texts fixed; unblocked P1 |
| P1 `top_k=1` routing | ✓ `ed81e13` | +18–22% on `top_k=1`; C-14 tie-break agreement proven e2e |
| P2 Lazy Z | **NO-GO** `bc59c56` | refuted on cost (finding kept); superseded by P2b |
| P2b parallel host normalize | ✓ `686c9f8` | temp-only +126–137%; temp+top_p +98–108% |
| P3 device top-K | **BANKED** | residual (readback) 0.12–0.43 ms ≈ 2% of the gap; reopen price too high |
| P4 KV-quant opt-in | **DECIDED** `d4472b9` | reachability-only both backends; speed on neither; build deferred (f16 first if ever) |
| P5 B1 path 2 | **BANKED** | prefill past its usability threshold; fund explicitly if ever |
| P6 depth sweep | ✓ `9d08165` + `b3ac838` | 8k–32k CUDA + Metal depth; plateau coefficients; decided P4 |
| P6a split-KV re-gate | ✓ `2693dce` | +19% @0.5B@512; live-regression fix |
| cap-raise (`-ctx`) | ✓ `ca29d6c` | resident cap configuration-derived; unblocked deep cells |
| P7 `--mode exact\|fast` | **DEFERRED** | no landed output-changing lever to aggregate |

**How it played out.** P0→P1 (and P2→P2b, once Lazy Z was refuted on cost) closed most of the
sampled-decode complaint — the deficit vs the peer is now 1.08–1.40×, and most real chat traffic
(temperature > 0) is served by P2b. P3 was re-scoped from "the campaign that finishes D6" to banked:
P2b left it only the readback (~2% of the gap), below its two-backend-kernel reopen price. The
long-context leg (P6/P6a/cap-raise) measured depth to 32k and *decided* P4 rather than building it —
KV-quant is reachability, not speed, on both backends, so it is build-deferred until a capacity
demand exists. P5 stays banked (prefill is already past usability). P7 has no constituents to
aggregate and is deferred. The one decode-speed lever both depth verdicts point at — an FA-class
occupancy/latency rewrite — is non-bit-identical and unfunded; it is the next real decision, not part
of this plan.

**Measurement hygiene for anything that ships a number:** re-pin and re-measure the Ollama peer
first (peer versions expire — §11), and record the sampling config on every row per the v0.10.3
benchmarks policy.
