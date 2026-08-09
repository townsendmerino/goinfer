# Plan — the three "still slow" items (post-v0.10.3)

Drafted 2026-08-09. Scope: the three residual slow paths named at the close of the v0.10.3
sampling work: (1) the nonzero-temperature ~3× cliff (readback + full-vocab softmax, D6
remainder), (2) `top_k=1` running 13–18% behind greedy, (3) long-context decode and prefill,
untouched since v0.9.0. Sources: `docs/ollama-chase.md` §3/§8, `docs/completed/audit-2026-08-05.md`,
`decoder/sampler.go`, `cuda/argmax.cu`, `metal/kernels.go`. Per the ground rules, nothing here
re-proposes anything in §10 (refuted) and every phase names its gate.

**Status headline that changes the plan: the `top_k=1` blocker is already gone.** D6's "hard
prerequisite" — the unspecified CUDA argmax-reduce index tie-break — was fixed on the Linux box as
audit **C-14** (`c6600fc`, 2026-08-06): `cuda/argmax.cu` tie-breaks to the LOWEST index, split into
its own `argmax.ptx` so the audited 12.6 `glue.ptx` was not regenerated, gated by
`TestArgmaxTieBreak`. The Metal fused-amax kernels (`gemv_w*a8_*amax` + `argmax_finish`,
`metal/kernels.go`) already carry the same lowest-index merge key. Three texts still call the
tie-break open and are stale: `decoder/sampler.go` (amendment-1 comment in `topFilterLogits`),
`docs/ollama-chase.md` §8 ("an open audit critical"), and the v0.10.3 CHANGELOG entry (historical —
leave it, it was true at press time... verify commit order first; see P0).

---

## P0 — Truth maintenance (hours; do first, on the CUDA box)

1. **Confirm C-14 is live**, not just present in the tree: `GOINFER_HEAVY_TESTS=1 go test -run
   TestArgmaxTieBreak ./cuda/` (needs the 0.5B fixture). The audit says fixed; the D6 text written
   *after* the audit says open — one of them is wrong, and P1 is gated on the test being green, not
   on either doc.
2. **Fix the stale texts** (`sampler.go` comment, ollama-chase §8 D6 + `top_k=1` section). Decide
   the CHANGELOG note by commit order: if `c6600fc` predates the v0.10.3 tag, the entry was wrong
   at press time and gets a correction line; if not, it stands.
3. **Confirm which Metal amax kernel is actually dispatched.** The `w4a8` fused amax carries an
   N-09 note ("NOT currently dispatched — no pipeline is created for it"); the int8 twin is the
   logit-critical one. P1's Metal claim depends on the *live* greedy path tie-breaking correctly —
   check `model.go`, don't assume from the kernel source.

## P1 — Route `top_k=1` to the greedy fast path (days; cheap consistency fix, and de-risks P3)

Measured gap: qwen2.5-coder-0.5b 272 vs 312 tok/s, gemma3-1b 148 vs 180. `top_k=1` at any
temperature is mathematically greedy: temperature scaling is monotone (order preserved) and a
one-token distribution is deterministic.

**Honest framing of the payoff.** The 13–18% applies ONLY to `top_k=1`-with-nonzero-temperature
requests, which is a rare, near-pathological shape (sampling from a one-token set) — and D6's own
rank note says the traffic that matters is `temperature>0` broadly, which P2/P3 serve, not this. So
P1's real value is not the headline number: it is (a) a correctness/consistency fix — `top_k=1`
*should* be greedy-fast on every backend — and (b) the cheapest place to **prove the C-14 tie-break
agreement end-to-end** (host `topKByLogit` vs device argmax vs Metal amax all resolving the same tied
index) that P3 later relies on. Treat it as the warm-up that validates P3's hardest assumption, not
as a throughput win in its own right.

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

## P2 — Lazy Z: kill the full-vocab softmax on the temperature-only path (days; host-only, no device box)

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

## P3 — Device-side top-K logit reduction (the campaign; weeks, needs both device boxes)

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
  EVERY token short-read → a top-K reduction PLUS a full-V readback per token, i.e. strictly more
  work than today. Correctness is preserved (identical-by-fallback) but throughput could regress
  below the current full-readback path. Add a perf gate: the adversarial-flat case must stay within a
  stated factor of today's full-readback path (not just "byte-identical"). If it can't, the kernel
  should stage the full row alongside the top-K so a short read costs one copy, not a second
  round-trip — decide this from the measured all-fallback number, not up front.
- **Expected:** the reporter's ~2.9× nonzero-temperature penalty collapses to ≈ the P2 residue;
  gemma3-1b (widest vocab, largest readback) is the headline number.

## P4 — Long-context decode: KV-cache quantization, opt-in (1–2 weeks, profile-gated)

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

## P6 — Extend the depth sweep into the agentic regime (8k/16k/32k) (days; measurement only)

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

## P7 — `--mode exact|fast`: one word for "give me Ollama's deal" (capstone; after P4/P5 land)

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

| phase | effort | needs | buys |
|---|---|---|---|
| P0 truth maintenance | hours | CUDA box | unblocks P1; docs stop lying |
| P1 `top_k=1` routing | days | CUDA box (gate) | 13–18% on `top_k=1` requests, all backends |
| P2 Lazy Z | days | none (host) | kills the ~44 ns/entry softmax; temp-only stops being the slowest config |
| P3 device top-K | weeks | CUDA box + Mac | kills the readback; nonzero-temp ≈ greedy |
| P4 KV-quant opt-in | 1–2 wks | both boxes, profile-gated | Metal long-context floor; KV VRAM halved |
| P5 B1 path 2 | campaign | funding decision | ~1.3× prefill attention; one re-baseline |
| P6 depth-sweep extension | days | CUDA box (Metal after) | the axis agents actually run at; feeds P4's gate |
| P7 `--mode exact\|fast` | days (plumbing) | P4 landed (P5 optional) | one-word contract opt-out; fair-fight benchmark cell |

P0→P1→P2 are sequential and cheap — together they close most of the sampled-decode complaint
(most real chat traffic runs temperature > 0, per D6's rank note). P3 is the campaign that
finishes D6; start it only with both device boxes available. P4 is independent of P1–P3 and
profile-gated; P6 is its cheapest input and can run any time the CUDA box is free. P5 stays
banked pending an explicit decision. P7 caps the sequence: it ships only once it has real levers
to aggregate (P4 at minimum), and its gate lane is part of its cost, not an afterthought.

**Measurement hygiene for anything that ships a number:** re-pin and re-measure the Ollama peer
first (peer versions expire — §11), and record the sampling config on every row per the v0.10.3
benchmarks policy.
