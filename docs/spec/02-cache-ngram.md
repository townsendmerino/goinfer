# 02 — Cache / n-gram drafting

> Status: **proposal**. Depends on [00-core](./00-core.md). Low effort; high ROI on
> code-edit, RAG, summarization, and agent loops. Build alongside [01](./01-grammar-fused.md).

## Idea

A large fraction of real output is **copied verbatim from the input or from earlier
output**: code edits echo the surrounding file, RAG answers quote retrieved
passages, summaries reuse phrasing, and agent loops repeat a fixed system prompt,
tool schemas, and prior tool results. At those positions the target distribution
puts most of its mass exactly where a copied n-gram would — low `TV(p, q)`, high
acceptance ([00-core](./00-core.md) §1) — for essentially zero compute.

This is the cheapest drafter that works on free text. Recent results
("Cacheback", REST, prompt-lookup) show cache-only drafting is competitive on these
workloads with no draft model at all.

## Mechanism

Maintain a **suffix automaton** (or a rolling n-gram index) over the concatenation
of: the current prompt + everything generated so far in this session. On each step,
take the last `m` generated tokens as a key, look up where that suffix occurred
before, and propose the token(s) that followed — as a draft chain, or as several
branches when the suffix had multiple continuations (a natural tree for
[03](./03-router-tree.md)).

- **`QProb` in the rejection step must be `q(x)=1` (a point mass), not the empirical
  frequency.** An n-gram proposal is *deterministic* — we commit to one token, we do
  not sample from a distribution — so the lossless coupling treats it as the point
  mass `δ_x`: accept with probability `min(1, p(x)/q(x)) = min(1, p(x)) = p(x)`, and on
  rejection sample the correction from `p` with `x` removed and renormalized
  (`(p − δ_x)+`). Using a fractional `q` like `0.93` here would make the accept
  probability `min(1, p(x)/0.93) ≥ p(x)` — it **over-accepts** and silently biases the
  output away from `p`. The empirical continuation frequency is still valuable, but as
  a **feature for `α̂`** ([00-core](./00-core.md) §4), never as the `q` in the
  rejection arithmetic. (If we ever proposed *stochastically* — sampling a branch from
  the index's frequencies — then `q` would legitimately equal those frequencies; we do
  not, and shouldn't, since the deterministic top continuation is both cheaper and
  higher-acceptance.)
- Match length is the key acceptance signal: longer / more specific suffix matches
  predict higher acceptance ([00-core](./00-core.md) §4). Feed match length into
  `α̂` and let [04](./04-adaptive-depth.md) extend the copy run while `α̂` stays high.

## Why it suits goinfer

- The server **already keeps warm KV sessions** and the token history they imply, so
  the corpus to index is already resident — the automaton is a cheap side-structure,
  not new state to persist.
- It is purely a token-stream structure: no model weights, no GPU, pure Go, trivially
  within the no-cgo default build.
- It shines on `cmd/serve`'s agent traffic (fixed system prompt + tool specs repeated
  every turn) and on the RAG coding agent in `demo/agent`.

## Expected payoff

Large committed-per-verify on `codeedit` / `rag` / `agent` suites; little to nothing
on novel `reasoning` text (where 05's head earns its keep). Because cost is ~zero,
it is almost always worth running as one source in the [03 router](./03-router-tree.md)
tree even when its hit rate is low — a miss costs nothing, a hit is free tokens.

## Variants to evaluate

- **Suffix automaton** (online, `O(1)` amortized append, exact longest-suffix match)
  vs a simpler fixed-order **n-gram hash** (cheaper, approximate). Start with the
  hash for a baseline, measure the gap.
- **Scope of the corpus:** prompt-only vs prompt+session vs prompt+session+a static
  domain corpus (e.g. the user's codebase). More corpus = more hits but more index
  cost and staleness risk; measure the curve.
- **Session-local online learning:** the index improves within a session as the
  model establishes repeated structure — quantify the within-session acceptance lift.

## Risks / open questions

- Index memory growth on very long sessions — bound it (sliding window / cap) and
  measure the acceptance cost of the bound.
- Tokenizer alignment: index and propose at the token level to avoid byte/҂token
  mismatches with the verifier.
- Staleness when a static corpus drifts from the live distribution — a miss is cheap,
  but a confidently-wrong long branch wastes verify width; cap branch length by `α̂`.

## Validation plan

- Correctness: lossless by construction (verifier owns `p`); assert output ≡ baseline.
- Speed/acceptance: `codeedit`, `rag`, `agent` suites in the [00-core](./00-core.md)
  harness; report committed/verify and the acceptance-vs-match-length curve.

---

# Next step — suffix-tree / suffix-automaton drafting, measured against the shipped baseline

> Status: **pre-registration + step 1–2 measurement**. Opened 2026-08-31. The doc comment on
> `NgramDrafter` names this as the deliberate next step ("not yet the suffix automaton — start
> simple, measure the gap"); this section measures the gap before anything is built.

## Why now — external evidence, and its regime

Two results arrived that make the question live. **Both are directional only. They are server-GPU,
batched, vLLM-shaped, and single-source; no number from either belongs in any table on this page.**

- **SuffixDecoding** (arXiv [2411.04975](https://arxiv.org/abs/2411.04975), shipped in Snowflake's
  Arctic Inference for vLLM) reports up to **2.8×** over EAGLE-2/3 and **1.9×** over Token
  Recycling, and **1.8–4.5×** on end-to-end SWE-Bench task completion including prefill and
  external actions.
- **An empirical study of SD on software-engineering tasks** (arXiv
  [2604.26469](https://arxiv.org/abs/2604.26469), replication package
  `github.com/AltriaSetsuna/speculative-decoding-empirical`) reports a **negative correlation
  between model scale and speedup**: ~1.62× average on an 8B, ~27% more benefit than a 32B, most
  pronounced for **model-free** methods — PLD 1.71× on 8B, near-nothing on 70B. It also finds code
  tasks more predictable than natural language.

**The second matters more than the first, and it is the reason to look now.** [README](./README.md)'s
scorecard says a drafter translates only when the target step is expensive enough to dwarf the
draft. That is a statement about **model** drafters, where the draft costs a forward pass. For a
model-free drafter the draft is nearly free, so the ratio inverts and **small models benefit most**
— which is this project's lane, not an aside from it.

## What is genuinely different — and what is already shipped

**Already shipped. Do not rebuild:**

- adaptive draft depth from a running acceptance estimate — `spec_adaptive.go`, `AdaptiveDepth`,
  `GenerateNgramSpeculativeAdaptive` ([04](./04-adaptive-depth.md)).
- the `Drafter` plugin seam (`spec_ngram.go`). **A suffix drafter is a new source behind the
  existing interface, not an architecture change.**
- warm-KV prefix reuse in the n-gram path.

**The whole scope of this task is three things, and they are separable:**

1. **Data structure.** Current matching is exact-suffix, greedy-longest, **linear backward scan**
   (`NgramDrafter.Draft`: `for L := hi; L >= minM; L--` over `for s := n-L-1; s >= 0; s--`). A
   suffix tree ([2411.04975](https://arxiv.org/abs/2411.04975)) or suffix automaton
   (SAM-Decoding, arXiv [2411.10666](https://arxiv.org/abs/2411.10666)) gives sublinear lookup that
   does not degrade as context grows. Pick one and say why.
2. **Candidate scoring.** We take the **most recent** earlier hit and `Confidence` is the
   suffix-match length. SuffixDecoding scores by empirical **frequency** — how often that
   continuation followed that pattern. Different signal, possibly better, and **testable
   independently of the data structure**. Likely the cheaper half of the win.
3. **Scope of history.** Ours searches the running context of **one generation**. SuffixDecoding
   keeps a tree spanning previous outputs **across requests**, which is where an agent loop's
   repetition actually lives. This interacts with the stateless OpenAI surface — see *Scope and the
   stateless surface* below.

## Pre-registration (written before step 1 was run)

Fixed in advance so the ambiguous zone cannot be argued into a win afterwards. Every band below has
an explicit **parked** middle.

**Definitions.** Per generation: `H` = fraction of rounds in which the drafter proposed ≥1 token
(**hit rate**); `α` = realized per-position acceptance on rounds that proposed
(`AdaptiveDepth.Alpha()` / `SpecStats.AcceptanceRate()`); `A` = mean **accepted** draft tokens per
round. `A` is the throughput-relevant quantity: a round commits `A+1` tokens for one verify pass.

**Gate 1 — is there a ceiling worth chasing?** Measured on realistic traffic, on the shipped
drafter:

| condition | verdict |
|---|---|
| `H ≥ 0.60` **and** `α ≥ 0.85` | Baseline is already near its regime ceiling. Structure cannot add acceptance it is already getting; only **scope** (item 3) can add hits. → chase scope, **park structure**. |
| `H ≤ 0.25` | The drafter rarely fires at all on realistic input. A better structure cannot find matches that are not there. → **park**, and correct this page's copy-heavy claim. |
| `0.25 < H < 0.60` | Headroom exists in **hit rate**. → proceed to step 2. |
| `α ≤ 0.50` on rounds that do fire | It proposes badly when it fires; **scoring** (item 2) is the lever and is cheap. → proceed to step 2. |

**Gate 2 — does frequency scoring justify building anything?** Step-2 offline replay, paired on the
**same rounds**, most-recent vs frequency-scored candidate:

| Δ in `A` (accepted tokens/round) | verdict |
|---|---|
| **≥ +15%** on ≥2 realistic workloads | **build** — start with scoring (item 2), which needs no new data structure |
| **+5% … +15%** | **ambiguous → parked.** Record it, do not build on it. |
| **< +5%** | scoring is not the lever; structure alone must carry item 1, which raises its bar |

+15% is set where it is because at a realistic `A ≈ 2` it is ≈ +0.3 committed tokens per verify —
about a 10% end-to-end move — which is the smallest effect worth new code on the hot path.

**Gate 3 — what makes the whole thing a park.**

- `off` (no drafter) beats both speculative modes in wall-clock on realistic traffic → **park and
  file a regression**; the spoke is not paying for itself where it claims to.
- `H ≤ 0.25` **and** `α ≤ 0.60` → **park**; the honest deliverable is then the correction to this
  page, not a new drafter.

**Exclusion rule, fixed in advance.** [2604.26469](https://arxiv.org/abs/2604.26469) warns that
models in agentic scenarios fall into **infinite loops**, and a drafter that feeds on repetition
looks spectacular on a looping run. Any trace whose *generated* tokens score **distinct-trigram
< 0.70** is **excluded before analysis**, not explained afterwards — the same bar §B4 uses as its
coherence floor in `docs/benchmarks.md`. This is the routing-confound lesson in a new place: the
thing that makes the number big is the thing that makes it meaningless.

## The input is the experiment — and the existing corpus does not qualify

**`scripts/prompts.json` is out**, per [10](./10-optfwd-gate.md)'s finding: every prompt has four
unique words, which is maximally favourable to a repetition drafter. Using it would manufacture the
result.

**So is this page's own `specWorkloads` corpus, and that is the more uncomfortable half.** The three
prompts in `decoder/spec_harness_test.go` are hand-written and *deliberately* copy-heavy — its own
comment says so ("completion-style prompts with heavy internal repetition — the regime n-gram
drafting is built for"). `codeedit` is `GetX/GetY/GetZ/SetX/SetY/Set`; `rag-copy` literally
instructs "Reproduce the source list exactly"; `agent-json` is three near-identical JSON lines.
**These are not samples of copy-heavy traffic, they are constructions of it.** [10](./10-optfwd-gate.md)
listed "02's n-gram drafter wins on copy-heavy traffic" as an unaudited claim contaminated by
`prompts.json`; the contamination is in fact one level deeper, in this spoke's own fixtures.

**Consequence that must be stated plainly: the shipped `ngramAlphaAnchors` table
(`spec_ngram.go`, 0.70 → 0.97, AUC 0.818) was fit on those constructed prompts.** It is a
calibration of how the drafter behaves on inputs built to make it succeed. That does not make it
wrong, and §06's cross-workload validation (held-out ECE 0.14, and an *inverted* sql AUC of 0.32)
already shows it does not generalize uniformly — but it does mean the prior is anchored to a
favourable regime, and it is another reason the online correction (§06 §9) is load-bearing rather
than optional.

**Therefore the realism of the input is measured, not asserted.** Every input in step 1 is scored
for **copy density** — the fraction of positions whose preceding 4-token suffix already occurred
earlier in the context — computed on the token stream, independent of any drafter. `prompts.json`
and `specWorkloads` are scored on the **same** metric, so the distance between the constructed
corpus and real traffic is a number on this page rather than a caveat.

## Scope and the stateless surface (item 3), before proposing it

Cross-request history is where an agent loop's repetition lives, and goinfer's OpenAI surface is
**stateless**: `/v1/chat/completions` re-sends the whole conversation every turn and the server
keeps no per-conversation identity. Three consequences, which must be settled before item 3 is
scoped, not after:

- **The repetition is already in the prompt.** Because the client resends the transcript, a
  *within-request* index over prompt + generation already sees the prior turns. Cross-request
  history buys what the resend does **not** carry — and on this surface that is much less than the
  paper's setting implies.
- **A cross-request index is per-model shared mutable state** on a server that today keeps none, so
  it needs a thread-safe owner, a bound, and an eviction policy. §06 §9 already flags exactly this
  for the online α̂ correction ("true cross-session workload-drift adaptation needs shared,
  thread-safe per-model state").
- **It leaks across requests by construction.** One user's outputs become another's draft
  candidates. Lossless decode means it cannot change *what* is emitted — the target owns every
  token — but it is a timing side channel, and that is a security review, not a perf ticket.

The warm-KV `Session` path is the honest place for cross-request scope, because identity there is
explicit and already scoped to one conversation.

## What this does NOT unblock

**Model-free does not mean seam-free.** This drafter still verifies `k` tokens and discards the
tail, so `specRollbackSafe` still refuses the Gated DeltaNet / Mamba-2 families and the staged
sliding-window ring cache — `validateNgramSpec` rejects them outright. **Nothing here unblocks Qwen
(qwen3_5 / qwen3_5_moe / qwen3_next).** A suffix tree changes where proposals come from; it does not
change that rollback must restore recurrent state, which is the actual blocker.

Also unchanged: `LogitProcessor` (constrained / tool decoding) is unsupported in either mode. That
bites precisely here — the real agent in `demo/agent` runs its DECIDE phase under a JSON-schema
grammar mask, so **the most repetitive phase of a real agent loop is the one speculation cannot
serve today.**

## Step 1 — what the shipped drafter actually does on realistic traffic

**Provenance.** `nobara-pc` (Ryzen 7 3700X), **CPU backend**, goinfer `574229b` (tree dirty — these
are working-tree harnesses), qwen2.5-coder **0.5B** and **1.5B** at q4_K_M from `~/models` (local
NVMe), **greedy** (`Temperature 0`), 160 max tokens, `K = 8`, single run per cell, 2026-08-31.
Harnesses: `decoder/spec_suffix_probe_test.go` (step 1), `decoder/spec_suffix_replay_test.go`
(step 2).

> **These numbers are NOT a `docs/benchmarks.md` row and must not be promoted into one.** Wall-clock
> here is a single un-interleaved run on one box; this page's standard is provenance for
> *acceptance*, not for throughput. The acceptance quantities (`H`, `α`, `A`, match length) are
> machine-independent; the `×` columns are **indicative only**. They are also **CPU**, where
> [04](./04-adaptive-depth.md)'s `Theta ≈ 0.5`; on a memory-bound GPU verify `Theta → 0` and the
> depth calculus changes, so the ratios below do not transfer to the resident path.

### The corpus confound, measured rather than asserted

Copy density (fraction of positions whose preceding 4-token suffix already occurred), computed on
the token stream with **no drafter and no model**:

| corpus | copy density (4-gram) |
|---|---|
| `specWorkloads` **agent-json** | **0.429** |
| `specWorkloads` **rag-copy** | 0.263 |
| `specWorkloads` **codeedit** | 0.257 |
| real Go source (`spec_adaptive.go`) | **0.060** |
| real Go source (`spec_sample.go`) | **0.067** |

**The constructed corpus is 4–7× more copy-dense than the real code it stands for.** That is the
number [10](./10-optfwd-gate.md) predicted without being able to quantify, and it is the reason the
step-1 inputs are read from real repo files at run time rather than authored.

### Results

`H` = hit rate (rounds proposing ≥1 token), `α` = `SpecStats.AcceptanceRate()` (accepted/proposed),
`A` = accepted draft tokens per round, `tok/v` = committed tokens per verify pass.

| model | input | copy4 | tri | H | α | tok/v | `off` | fixed-K | adaptive |
|---|---|---|---|---|---|---|---|---|---|
| 0.5B | code-continue | 0.060 | 0.715 | 0.403 | 0.348 | 2.08 | 21.3 s | 1.06× | **1.19×** |
| 0.5B | code-continue-2 | 0.067 | 0.854 | 0.769 | 0.850 | 6.15 | 16.7 s | **1.44×** | 1.36× |
| 0.5B | agent-loop-turn2 | — | 0.297 | — | — | — | — | **EXCLUDED** | |
| 0.5B | prose-doc | — | 0.570 | — | — | — | — | **EXCLUDED** | |
| 1.5B | code-continue | 0.060 | 0.759 | 0.387 | 0.240 | 1.72 | 28.5 s | **0.83×** | **0.98×** |
| 1.5B | code-continue-2 | 0.067 | 0.880 | 0.647 | 0.756 | 4.71 | 26.5 s | 1.15× | 1.17× |
| 1.5B | agent-loop-turn2 | — | 0.671 | — | — | — | — | **EXCLUDED** | |
| 1.5B | prose-doc | — | 0.665 | — | — | — | — | **EXCLUDED** | |

**`off` wins on the 1.5B's harder code trace — 0.83× fixed, 0.98× adaptive.** This is Gate 3's park
trigger, and it is only visible because `off` was a required arm. Two things follow. First, the
[04](./04-adaptive-depth.md) controller is doing its job: it turns a 17% *loss* into a 2% loss by
collapsing depth when acceptance falls, which is exactly the over-draft it was built to kill.
Second, **the spread between two real Go files (0.83× and 1.15× on the same model, same day) is
larger than any effect this task set out to chase.** Two files is not a sample; it is two points.

**The direction of the scale effect reproduces, in our own lane.** arXiv
[2604.26469](https://arxiv.org/abs/2604.26469) reports model-free speculation benefiting *smaller*
models more (PLD 1.71× on 8B, near-nothing on 70B). Between 0.5B and 1.5B on the same prompts, the
same ordering holds: 1.19×/1.44× → 0.98×/1.17×. **This is a directional agreement on two models and
two files, not a measurement of that paper's claim**, and it is confounded — the two models generate
*different* text, so the traces are not matched observations. It is recorded because it points the
same way, not because it establishes anything.

### Both non-code workloads were excluded, and the loop guards disagree

The pre-registered rule (distinct-trigram < 0.70) excluded `agent-loop-turn2` and `prose-doc` on
**both** models. The second, independent detector — is the tail *periodic*? — disagrees:

| trace | distinct-trigram | tail cycle | guards agree? |
|---|---|---|---|
| 0.5B agent-loop-turn2 | 0.297 (excludes) | **period 25, 4 repeats** | ✅ genuinely looping |
| 0.5B prose-doc | 0.570 (excludes) | none | ❌ **disagree** |
| 1.5B agent-loop-turn2 | 0.671 (excludes) | none | ❌ **disagree** |
| 1.5B prose-doc | 0.665 (excludes) | none | ❌ **disagree** |

**The first agent exclusion was a defect in the INPUT, and the guard caught it.** The first attempt
re-sent the *same* retrieved snippet for both turns; the model copied it back and produced a real
25-token cycle. A real second query retrieves different code, so the input was fixed and the guard
left alone — the artifact arXiv 2604.26469 warns about, caught by a rule written in advance.

**The remaining three exclusions are the rule misfiring, and that cost the workload this task most
needed.** distinct-trigram measures *repetitiveness*; an agent answer grounded in re-sent context is
legitimately repetitive without looping, and at 1.5B the trace sits at 0.671 against a 0.70 bar
written for prose coherence, with no cycle anywhere in it. **The pre-registered rule binds anyway —
the bar is not moved after seeing the data — so item 3 is UNANSWERED rather than answered
negatively.** What the disagreement buys is knowing *why*: the next run must pre-register a loop
rule keyed on the actual failure mode (tail periodicity) rather than on repetitiveness, and that
rule must be fixed before the run, not chosen from these two after it.

This is [CLAUDE.md](../../CLAUDE.md)'s own corollary working as intended: pre-register two things
that can disagree, because after a rule fails the instinct is to write a better rule, and the more
reliable fix is a second independent pre-registration. The right one caught the wrong one.

## Step 2 — candidate scoring, replayed offline

**No model is loaded.** In greedy mode a model-free drafter's acceptance is exactly *does the
proposed token equal the token the target emitted*, and losslessness guarantees the recorded stream
**is** what the target emitted. So every policy replays exactly, on the **same rounds** — the paired
comparison the pre-registration asked for. Structure is held fixed: every arm selects the same
longest matching suffix, so differences are scoring alone.

**Replay validity, cross-checked against the live runs:** replay reproduces live accepted-token
counts to **97–99%** (86→83, 136→134, 69→67, 133→126) at *identical* round counts. The residual is
an end effect, confirmed not assumed: every run emitted exactly 160 tokens, so the final round
accepted tokens that the cap truncated before emission, which the replay cannot see. Each gap is
below one round's maximum of 8.

| model | input | most-recent (shipped) | frequency | deep-cap 64 | **oracle (ceiling)** |
|---|---|---|---|---|---|
| 0.5B | code-continue | 1.078 | 1.051 (**−2.5%**) | 1.078 (+0%) | 1.133 (**+5.1%**) |
| 0.5B | code-continue-2 | 5.154 | 5.154 (+0%) | 5.154 (+0%) | 5.154 (**+0%**) |
| 1.5B | code-continue | 0.720 | 0.739 (**+2.6%**) | 0.720 (+0%) | 0.798 (**+10.7%**) |
| 1.5B | code-continue-2 | 3.706 | 3.706 (+0%) | 3.706 (+0%) | 3.848 (**+3.8%**) |

(accepted tokens per round, `A`.)

**The oracle is the finding.** It cheats — it reads the future and picks whichever earlier
occurrence *would* have been accepted longest — so it is the ceiling for **any** candidate-scoring
scheme at this structure, frequency or otherwise. It reaches **+0% to +10.7%**, never the
pre-registered **+15%** build bar. Frequency scoring, the actual implementable thing, lands
**−2.5% to +2.6%** — inside the noise and below the **+5%** park line on every trace.

**This closes item 2 rather than merely failing to support it.** A negative result on frequency
alone would leave "maybe a better score exists"; the oracle forecloses that whole class.

**`deep-cap 64` is +0% on 4 of 4**, which answers a question the histogram raised. On
`code-continue-2`, 12–14 of ~20 hits sit at **exactly** the `MaxMatch = 16` cap — matches visibly
truncated. Raising the probe to 64 changes **no** accepted tokens, because those long matches have a
single earlier occurrence: every policy picks the same one. The cap therefore **mis-reports
confidence** — `ngramAlphaAnchors` tops out at 16 → 0.97, so a 40-token verbatim repeat and a
16-token one are scored identically, which matters to the [03](./03-router-tree.md) router's source
ranking — **but it costs no accepted tokens.** A reporting defect, not a throughput one.

## What the drafter costs — item 1's premise, measured

| model | input | context | `Draft` cost | decode step (§B8, this box) | drafter share |
|---|---|---|---|---|---|
| 0.5B | code-continue | 902 tok | **12.8 µs** | 42.6 ms (23.5 tok/s) | **0.030%** |
| 0.5B | code-continue-2 | 816 tok | **2.6 µs** | 42.6 ms | **0.006%** |
| 1.5B | code-continue | 902 tok | **16.9 µs** | 56.8 ms (17.6 tok/s) | **0.030%** |
| 1.5B | code-continue-2 | 816 tok | **3.2 µs** | 56.8 ms | **0.006%** |

*(Denominator is `benchmarks.md` §B8's measured CPU decode rate. The harness's own `off_ms ÷ tokens`
gives an even smaller share — 0.002–0.010% — because it includes prefill in the step; the §B8
denominator is the conservative one and is used here for that reason.)*

**Item 1's premise does not hold at these context lengths.** A suffix automaton buys sublinear
lookup "that does not degrade as context grows" — but the linear scan it replaces is **three
hundredths of one percent** of a round. Making lookup infinitely fast buys ~0.03%. The asymptotic
argument is correct and irrelevant: at 800–900 tokens the constant is a rounding error, and the
scan is bounded by `MaxMatch = 16` anyway, so it grows linearly in context, not quadratically. **A
structure change has to be justified by *what it finds*, not by *how fast it finds it* — and
`deep-cap 64` shows it finds nothing more.**

## Verdict against the pre-registered gates

| gate | measured | verdict |
|---|---|---|
| **1** — ceiling worth chasing | `H` 0.387–0.769, `α` 0.240–0.850; the two code files straddle every band | **split** — one trace says "near ceiling, chase scope", the other says "scoring is the lever". Proceeded to step 2, which settles it. |
| **2** — frequency scoring ≥ +15% | frequency **−2.5% … +2.6%**; **oracle ceiling +0% … +10.7%** | **< +5% → PARK.** Not just unsupported — bounded. |
| **3** — park conditions | `off` beats both arms on 1.5B code-continue (0.83× / 0.98×) | **park trigger fired** |

**PARK all three items.** Structure is chasing 0.03% of a round and finds nothing extra when
un-capped; scoring is bounded below the build bar by an oracle; scope is already being served by the
prompt on a surface that re-sends it.

> **And on Metal the park is stronger than "not worth it".** Its Theta ≈ 1.0 (below) means a verify
> node costs a full decode step, so *no* drafter — n-gram, suffix tree or otherwise — can pay there
> until `ForwardN` is genuinely batched. On that backend this is not a question about drafters at all.

**PARK item 3 as well.** An earlier draft of this section called it "unanswered" because the agent
trace was excluded. That was over-hedged: the exclusion costs the *magnitude*, not the answer, and
the answer does not depend on it.

**Where the drafter's matches actually come from** (`TestSuffixProbe_matchProvenance`), 1.5B:

| trace | match in PROMPT (= re-sent history) | match in own generation | miss |
|---|---|---|---|
| code-continue | **41.5%** | 22.6% | 35.8% |
| code-continue-2 | **88.7%** | 3.8% | 7.5% |
| agent-loop-turn2 *(excluded; lower bound)* | **22.6%** | 37.7% | 39.6% |
| prose-doc *(excluded; lower bound)* | **28.3%** | 37.7% | 34.0% |

**The shipped drafter is already harvesting exactly the history item 3 proposes to add**, because on
a stateless surface it is already in the prompt: `/v1/chat/completions` re-sends the whole transcript
every turn and `NgramDrafter` indexes the whole prompt. 41–89% of matches on code come from there.

So a cross-request index can only add supply on the **miss** rounds, and only for a pattern that
appeared in an earlier request *without being re-sent* — which on this surface means **a different
conversation**, not an earlier turn of this one. That residual is precisely the case with the
timing side channel and no evidence of demand behind it.

**The bias runs the safe way on the excluded traces.** A looping generation manufactures long recent
SELF-matches, and longest-suffix-wins makes those outrank prompt matches — so looping *deflates* the
prompt share. The agent trace's 22.6% is a floor, not a ceiling, which means the excluded measurement
could only strengthen this conclusion, never overturn it.

What the exclusion does still cost: the agent workload's **acceptance** numbers, and so the size of
the prize if someone ever does show cross-conversation reuse exists. Reviving item 3 requires that
demonstration first — not a suffix tree.

**The `off` regression is filed as its own item, not fixed here.** `GenerateNgramSpeculative` at
fixed `K = 8` is a **17% loss** against no drafter on 1.5B real code. Adaptive nearly rescues it
(0.98×) and should be the default anywhere fixed-K is currently chosen; that is a
[04](./04-adaptive-depth.md) question, and it needs more than two files before anything is changed.

## What this corrects on this page

- **"Large committed-per-verify on `codeedit`/`rag`/`agent` suites"** — measured on *constructed*
  suites 4–7× more copy-dense than real code. On real code the honest range is `tok/v` **1.72–6.15**
  across two files and two models, with `off` winning one of the four cells.
- **"Because cost is ~zero, it is almost always worth running as one source"** — the *drafter's*
  cost is indeed ~zero (0.03% of a round, now measured). But the **verify** cost is not: fixed-K
  over-draft is what makes the 1.5B cell 0.83×. The claim is right about the drafter and wrong about
  the round; "a miss costs nothing" holds, "a low hit rate costs nothing" does not.
- **The `ngramAlphaAnchors` table** (`spec_ngram.go`) is fit on the constructed corpus. §06's own
  cross-workload validation already showed it does not generalize uniformly (held-out ECE 0.14, sql
  AUC **0.32** — inverted). This adds *why*: the prior is anchored in a favourable regime, which
  makes the §06 §9 online correction load-bearing rather than optional.
- **`AdaptiveDepth.Observe` does NOT have the G27 latch** (checked as asked, and measured rather
  than read). `genNgramInto` calls `Observe` on **every** round including `Depth()==0` rounds
  (passing `0,0`), and `Depth` tests `idle >= ProbeEvery` *before* the `alpha <= Theta` bail, so the
  probe fires and the estimate unfreezes. Verified: alpha driven to 0.0001 → probe at idle round 16
  → recovers to 0.36. **Recovery is slow though** — with `Lambda 0.8` each probe moves alpha
  `0.8α+0.2`, so climbing back past `Theta = 0.5` takes ~4 probes ≈ **64 rounds**. Not a defect, not
  filed; recorded because a controller that takes 64 rounds to re-enable is worth knowing about on
  short generations.

## What a next attempt needs

Not a suffix tree. In order:

1. **A loop rule pre-registered on tail periodicity**, fixed before the run — the current
   trigram bar cannot tell a repetitive agent answer from a looping one, and that is what blocked
   item 3.
2. **More than two files.** The between-file spread (0.83× vs 1.15×, same model, same day) exceeds
   every effect measured here. Any future claim needs enough real files to put an interval on it.
3. **The GPU-resident path — but as its own question, not this one.** Two of the three verdicts
   above are **backend-invariant** and do not need re-running: item 2's oracle ceiling is computed on
   token streams (same model, same greedy output, same acceptance on any hardware), and item 3's
   match provenance is a token-stream property plus a fact about the HTTP surface. What IS
   backend-dependent is whether **the spoke pays at all**, and item 1's cost share — which rises
   ~10× at CUDA decode rates (12.8 µs against a 3.0 ms step at §B8's 332.7 tok/s ⇒ **~0.4%**), still
   under half a percent.

   **Two defects found while scoping that, both worth fixing regardless of suffix trees:**

   - **`Theta = 0.5` is a CPU-measured constant that the adaptive controller uses on the GPU-resident
     path.** `spec_adaptive.go` says in its own words that Theta "is the relative cost of one extra
     verify node on *this backend* — measure it", and it never has been on any GPU backend: the
     default is the batched-CPU value and nothing overrides it. The error is in the **conservative**
     direction, so it costs speed rather than correctness — at α = 0.756, `Theta = 0.5` gives
     `D = ln(0.5)/ln(0.756) ≈ 2` while `Theta = 0.1` gives `D ≈ 8` (the `MaxDraft` cap). On a verify
     that streams the weights once for the whole block, the controller is plausibly drafting a
     quarter as deep as it should. **Measuring Theta on the resident path is the single highest-value
     next measurement on this spoke**, and it is a prerequisite the design already wrote down.
   - **`gpu/spec_ngram_resident_test.go` uses the IDENTICAL constructed corpus** — its
     `ngramWorkloads` are `specWorkloads` copied verbatim, comment included. So the GPU-side evidence
     for this spoke inherits the same 4–7× copy-density defect measured above. Whatever those numbers
     currently say, they say it about engineered input, and re-running them on realistic traffic is
     needed before any GPU claim is quotable.
4. **Only then**, if item 3 survives, the structure question — and by then the justification has to
   be what it finds, since this page has now measured that speed is not the constraint.

---

# Theta MEASURED — the constant the controller runs on was never measured on a GPU

> Follow-on to the "what a next attempt needs" item above, run 2026-08-31. This is a
> [04](./04-adaptive-depth.md) result recorded here because this page's step-1 regression is what
> exposed it; it should move to 04 if that page is revised.

`spec_adaptive.go` ships `Theta = 0.5`, describes it as the batched-CPU value, and says in its own
words that Theta "is the relative cost of one extra verify node on **this backend** — measure it."
Until now it never had been, on any backend. **The GPU-resident path has been running the adaptive
depth controller on a CPU constant.**

**Definition** (identical in both harnesses, so the numbers are comparable): `T(n)` = wall time of
one verify pass over `n` tokens at a fixed context depth; `Theta = (least-squares slope of T(n)) /
T(1)`. That is the marginal node cost in units of one single-token target step, which is exactly
what `D = floor(ln Theta / ln α)` trades against.

**The CPU control ran first, and reproducing the believed value is what licenses the GPU number.**
`decoder/theta_probe_test.go` on the staged CPU path returned **0.456** at depth 128 — the shipped
0.5, confirmed. An instrument that could not reproduce it would not be trusted on CUDA.

**Provenance.** `nobara-pc`, RTX 2070 SUPER, driver 595.91.07, Nobara 44 · goinfer `574229b`
(dirty) · qwen2.5-coder 0.5B / 1.5B q4_K_M from `~/models`, `Backend: "cuda"`, `Quant: "int4"` ·
median of 9 reps after 2 warm-ups, `TruncateTo` between every call · widths 1,2,3,4,6,8,12,16 ·
2026-08-31 · harnesses `decoder/theta_probe_test.go`, `cuda/theta_probe_test.go` · logs
`docs/measurements/spec04-theta-{cpu,cuda}-2026-08-31.log`.

| backend | model | depth | T(1) | slope (µs/node) | **Theta** |
|---|---|---|---|---|---|
| CPU (staged) | 0.5B | 128 | 32 429 µs | 14 789 | **0.456** ← reproduces the shipped 0.5 |
| CPU (staged) | 0.5B | 512 | 56 884 µs | 17 301 | **0.304** |
| **CUDA resident** | 0.5B | 128 | 4 585 µs | 712.5 | **0.155** |
| **CUDA resident** | 0.5B | 512 | 4 533 µs | 801.3 | **0.177** |
| **CUDA resident** | 1.5B | 128 | 5 357 µs | 1 259.8 | **0.235** |
| **CUDA resident** | 1.5B | 512 | 5 709 µs | 1 431.9 | **0.251** |

**Theta on CUDA is 0.155–0.251 — the shipped constant is 2–3× too high.** The direction was
predicted by [04](./04-adaptive-depth.md) ("→0 on a fully memory-bound GPU verify"); the magnitude
was not, and "→0" overstates it: a node costs 15–25% of a step, not ~nothing, because the verify
still reads back logits per position.

**Sensitivity, because the denominator is arguable.** `T(1)` here is a `ForwardN` of width 1, which
returns full logits; production greedy decode uses the `ResidentGreedy` on-device argmax and skips
that 594 KB readback, so a real single-token step is faster and Theta correspondingly larger.
Against §B8's measured decode rates (332.7 / 220.8 tok/s) the same slopes give **0.237 / 0.267 /
0.278 / 0.316**. **Both framings land well under 0.5, so the conclusion is robust to the choice**;
the in-probe number is the self-consistent one (numerator and denominator measured identically in
one process) and is what the table reports.

**What it costs, in drafted depth** (`D = floor(ln Theta / ln α)`, `MaxDraft = 8`), using the α
values measured in step 1:

| measured α | `Theta = 0.5` (shipped) | `Theta = 0.235` | `Theta = 0.155` |
|---|---|---|---|
| 0.850 (0.5B code-continue-2) | 4 | **8** | **8** |
| 0.756 (1.5B code-continue-2) | 2 | **5** | **6** |
| 0.348 (0.5B code-continue) | 0 | **1** | **1** |
| 0.240 (1.5B code-continue) | **0** | **1** | **1** |

**The controller is drafting 2–2.5× shallower than the hardware justifies on its best cells, and
turning speculation OFF entirely (D=0) on cells where the GPU would still profit from D=1.** The
last two rows are the interesting ones: those are exactly the traces where `off` beat the drafter on
CPU, and on CUDA's cost structure the correct depth is not zero.

### Metal: Theta ≈ 1.0 — and that is not a mistuned constant, it is a missing premise

Measured on the MacBook (`metal/theta_probe_test.go`, `8ef4d69`), completing the table:

| backend | Theta | vs the shipped 0.5 |
|---|---|---|
| CPU (staged) | 0.456 (d128) → 0.304 (d512) | ≈ right at shallow depth |
| CUDA resident | 0.155 – 0.251 | **2–3× too high** → under-drafts |
| **Metal resident** | **0.995 – 1.046** | **2× too LOW** → over-drafts |

Metal's value is **flat** across depth 128/512 and across 0.5B/1.5B, with the ladder linear out to
n=16 (16.57). The mechanism was predicted from the dispatch and then measured: **`metal/backend.go`'s
`ForwardN` is a plain loop of single-token `Forward` calls, not a batched kernel.** There is no block
to amortise the weight stream over, so the *n*-th verify node costs a full step.

**The consequence is larger than a constant.** At Theta ≈ 1, verifying K drafted tokens costs exactly
what decoding those K tokens would have cost — so the trade speculative decoding is built on does not
exist on this backend. It is not that the depth rule is mistuned; it is that **no depth is
profitable** until `ForwardN` is genuinely batched. Two independent measurements agree: post-fix at an
839-token prompt, Metal speculation is **1836 ms against `off`'s 1709 ms** — still behind, with the
prefill bug gone.

**It also retires an open puzzle on this page.** `gpu/spec_ngram_resident_test.go` has been printing
~0.3× speedups and asserting nothing about them. That was read here as a symptom of the corpus and of
the missing gate; on a backend whose Theta is 1.0 it is the *expected* result, and the harness was
faithfully reporting a real absence of a win nobody read.

**Not filed as a queue item** — batching Metal's `ForwardN` is a roadmap call, not a defect ticket,
and the number is the useful part. Recorded here so the next person to ask "why is speculation slow
on Metal" finds the answer rather than re-deriving it.

**Theta is depth-dependent on CPU and CUDA, and the two move in OPPOSITE directions.** CPU falls with depth
(0.456 → 0.304: `T(1)` grows with attention faster than the marginal node does), CUDA rises
(0.155 → 0.177, 0.235 → 0.251: the resident single-token step is nearly flat in depth — 4585 → 4533 µs
on the 0.5B — while each extra verify node attends over a longer context). A single scalar cannot be
right at both ends of either curve, but the spread within a backend (≤0.02 on CUDA) is far smaller
than the gap between backends (0.46 vs 0.16), so **per-backend is the split that matters; per-depth
is not worth machinery.**

**Not changed here.** This measures the constant; it does not re-tune the controller. Changing
`Theta` changes emitted-token *timing* only — the scheme stays lossless whatever depth it picks,
since the target verifies every position — but it is a live perf default on the resident serve path
and wants its own A/B with `off` in the arm set, on the realistic corpus rather than the constructed
one. **That is now the highest-value open item on this spoke**, and it is a [04](./04-adaptive-depth.md)
change, not a drafter change.

## Theta A/B on the resident path — pre-registration (written before the run)

**Question.** Does re-tuning `Theta` from the shipped CPU constant (0.5) to the value measured on
CUDA change resident decode throughput on **realistic** traffic?

**Arms, all on the same box, same session, interleaved arm-by-arm per input** (drift between
sessions is ~3.5% here and silently corrupts ratios):

| arm | what |
|---|---|
| `off` | plain resident `Generate` — the do-nothing arm, **required** |
| `fixed-8` | `GenerateNgramSpeculative`, K = 8 |
| `ada@0.5` | adaptive, **shipped** Theta |
| `ada@measured` | adaptive, this backend's measured Theta (0.155 on 0.5B, 0.235 on 1.5B) |
| `ada@0.30` | adaptive at the conservative sensitivity value (the §B8-denominator framing) |

**Inputs.** The realistic corpus (real repo files read at run time), not `specWorkloads`. The
pre-registered loop exclusion (distinct-trigram < 0.70 on the generated tokens) is applied **before**
any timing is read, exactly as in step 1.

**Protocol.** 3 repetitions per (input, arm), median reported; greedy; 160 max tokens; the lossless
gate is absolute — **any arm whose token stream differs from `off` fails the run outright**, it is
not a slower-but-acceptable result.

**Decision rule, fixed now:**

| outcome | verdict |
|---|---|
| median speedup of `ada@measured` vs `ada@0.5` **≥ +5%** across included inputs, **and** no single input regresses vs `ada@0.5` by more than 2%, **and** no arm loses to `off` | **adopt** a per-backend Theta |
| **+2% … +5%** | **ambiguous → parked.** Record; do not re-tune on it. |
| **< +2%** | Theta is not the lever on this path; the measurement stands as a corrected constant, unshipped |
| any arm not token-identical to `off` | **run fails**, nothing is read from it |

The "no single input regresses by more than 2%" clause is there on purpose: deeper drafting must
help the high-α cells *without* paying for it on the low-α ones, and a mean alone would hide that
trade.

## Theta A/B — result, and the much larger thing it found

**Provenance.** `nobara-pc`, RTX 2070 SUPER, driver 595.91.07, Nobara 44 · goinfer `574229b`
(dirty) · qwen2.5-coder 0.5B / 1.5B q4_K_M, `Backend: "cuda"`, `Quant: "int4"` · greedy, 160 max
tokens, 3 reps per (input, arm) interleaved arm-by-arm, median · realistic corpus · 2026-08-31 ·
harness `cuda/theta_ab_test.go` · log `docs/measurements/spec02-theta-ab-cuda-2026-08-31.log`.
**Every arm was token-identical to `off` in every cell — the lossless invariant held throughout.**

| model | input | `off` | fixed-8 | `ada@0.5` | `ada@measured` | `ada@0.30` | measured vs 0.5 |
|---|---|---|---|---|---|---|---|
| 0.5B | code-continue | **786 ms** | 2610 | 2681 | 2584 | 2591 | **+3.7%** |
| 0.5B | code-continue-2 | **743 ms** | 2213 | 2231 | 2205 | 2234 | **+1.2%** |
| 0.5B | agent-loop-turn2 | **631 ms** | 2867 | 2898 | 2865 | 2868 | **+1.1%** |
| 0.5B | prose-doc | — | — | — | — | — | EXCLUDED (tri 0.335) |
| 1.5B | code-continue | **1445 ms** | 4278 | 4224 | 4151 | 4152 | **+1.8%** |
| 1.5B | code-continue-2 | **1345 ms** | 3448 | 3448 | 3457 | 3466 | **−0.3%** |
| 1.5B | agent-loop-turn2 | **1593 ms** | 4712 | 4704 | 4679 | 4652 | **+0.5%** |
| 1.5B | prose-doc | — | — | — | — | — | EXCLUDED (tri 0.373) |

### Verdict on the pre-registered question: Theta is NOT the lever here

Mean effect of the measured Theta against the shipped 0.5 is **+1.3%** (range −0.3% to +3.7%) —
inside the pre-registered **"< +2% → not the lever"** band and nowhere near the +5% adopt bar.
**Do not re-tune Theta.** The measurement stands as a corrected constant, unshipped.

The reason it barely moves anything is the finding below: depth is not what is costing time on this
path, so making the depth rule smarter cannot help. **This is the pre-registration earning its keep
— a +1.3% result on a rule written afterwards is exactly the size of effect that gets narrated into
a win.**

### `off` beats every speculative arm by 2.5–4.5×, on every included cell

Not a regression *in* the drafter — a regression in the resident **speculative prefill**.

`decoder/model.go`'s `generateInto` prefills a resident model with **one batched on-device pass**
(the optional `Prefiller` seam, `pf.PrefillLast(embs, 0)`, taken whenever `len(prompt) >= 8`).
`decoder/spec_ngram.go`'s `genNgramInto` instead loops **`target.resident.Forward(...)` once per
prompt token**. `cudaResident` implements `PrefillLast` (`cuda/prefill.go`) — **the fast path exists
and is simply not called on the speculative path.**

**It is an oversight with a date, not a design choice.** `0fd54e8` (2026-06-20) landed n-gram
speculative decode with the sequential loop. `c36698a` (2026-07-16) wired batched prefill into
`generateInto` and did not update its speculative twin. No comment anywhere justifies the
difference, and the speculative prefill has never been touched since it was written.

### The control: the penalty is linear in prompt length

If the cause is per-token prefill, the *absolute* gap must grow with prompt length and the ratio
must collapse on short prompts. `cuda/spec_prefill_regression_test.go`, 0.5B, 64 tokens generated,
median of 3, lossless-checked at every length:

| prompt | tokens | `off` | speculative | ratio | absolute gap |
|---|---|---|---|---|---|
| 150 ch | 33 | 199 ms | 376 ms | 0.53× | +177 ms |
| 400 ch | 96 | 210 ms | 476 ms | 0.44× | +266 ms |
| 1000 ch | 253 | 285 ms | 933 ms | 0.31× | +647 ms |
| 2000 ch | 562 | 409 ms | 1941 ms | 0.21× | +1532 ms |
| 3000 ch | 839 | 536 ms | 2816 ms | 0.19× | +2280 ms |

**Least-squares: overhead = 30 ms + 2.66 ms per prompt token, R² = 0.9977.** The `off` arm's own
prompt cost fits at **0.42 ms/token**. **The missing batched call costs 6.3× per prompt token**, and
the intercept is ~0 — there is no fixed speculative penalty, only the per-token one. That is as
direct an attribution as this kind of measurement gets.

### Why no test caught it — two independent reasons, both instructive

1. **The only GPU speculative harness has no long prompts.** `gpu/spec_ngram_resident_test.go`'s
   corpus is `specWorkloads` copied verbatim: **36–74 tokens**. At 33 tokens the control above
   measures 0.53× — a *visible* loss, but a small absolute one that reads as noise. The defect scales
   with prompt length and the corpus has no length to scale with. **The constructed corpus did not
   merely overstate the drafter's win; it hid a 3× production regression.**
2. **That harness measures the right quantity and asserts nothing about it.** Its own header says
   *"Parity is hard-gated; speedup is logged per workload for fixed K=8 vs adaptive vs plain resident
   greedy."* A `plain` arm exists and a speedup column is printed — so a 0.5× would have been
   **printed and gated by nothing**. Correctness was hard-gated; performance was narrated. This is
   the `off`-arm rule one level up: including the do-nothing arm is necessary but not sufficient if
   nothing fails when it wins.

### Scope, and what is NOT done here

- **Correctness is unaffected.** Every arm in every cell was token-identical to `off`. This is
  latency only.
- **Measured on CUDA.** The defective code path is backend-agnostic (`genNgramInto`'s resident
  branch), so WebGPU is affected by construction — but its **magnitude there is unmeasured** and is
  not claimed.
- **The fix is not made here.** It is a small change — take the `Prefiller` seam in `genNgramInto`
  as `generateInto` already does — but it is a shipped-path change and belongs in its own commit,
  with the bit-identity gate that `c36698a`'s own history says this seam needs
  (`TestPrefillDivergenceRate`, `docs/task-batched-prefill-bitidentity.md`), plus a regression test
  that **fails** rather than logs when `off` wins.
- **This also re-frames the CPU step-1 result.** The CPU path already uses batched prefill
  (`prefillLogits`), so the 0.83× seen there is a genuinely different phenomenon — fixed-K
  over-draft — and is not this bug. The two should not be conflated.

## THE FIX — and what it does to every number above

**Shipped in this change** (`decoder/model.go`, `decoder/spec_ngram.go`): the resident prefill is now
**one shared function**, `Model.residentPrefillSeed`, called by both `generateInto` (plain) and
`genNgramInto` (speculative). The duplication was the root cause, so the fix removes the duplication
rather than copying the missing block a second time.

The speculative path was missing **three** things, not one — all of them things `generateInto` had:

1. **batched prefill** via the optional `Prefiller` seam (the 6.3×-per-prompt-token item above);
2. **KV-only prefill** (`ResidentPrefillKV.ForwardNoLogits`), which skips the LM head — a big-vocab
   matmul plus a ~1 MB readback — on every prompt token but the last, even in the sequential
   fallback;
3. **G18's `ctx.Err()` check** inside the prefill loop. Without it an abandoned client leaves the
   whole prompt streaming through the device — the exact failure G18 fixed in `generateInto` and
   left in place here.

**Backend reach — and the first version of this paragraph was wrong.** It read: "CUDA and Metal
implement `PrefillLast`; WebGPU does not… Metal is affected by construction and is fixed by this
change." **Implementing the seam is not the same as taking it**, and the second check was not done.
Corrected, by tracing `residentPrefillSeed` per backend:

| backend | `PrefillLast` | `ForwardNoLogits` (KV-only) | effect of this fix |
|---|---|---|---|
| **CUDA** | taken | **implemented** (`cuda/resident.go`) | **the measured 3–4.5× → 1.0–1.6×** |
| **Metal** | **DECLINES by default** | **not implemented** | **NO-OP** |
| **WebGPU** | not implemented | not implemented | no-op (never had the asymmetry) |

Metal's `PrefillLast` returns an error unless `GOINFER_METAL_BATCHED_PREFILL=1` /
`--metal-fast-prefill`, because its batched prefill is **not bit-identical to decode** — f16-MMA vs
int8 activations, **54% greedy-stream divergence** (`TestMetalPrefillDivergenceRate`, §A2-Metal). That
decline is deliberate and correct. With the batched path declined *and* no KV-only fallback, the
helper degenerates to the plain per-token full-logits loop — **byte-for-byte what the old
`genNgramInto` did.** So on default Metal this change alters nothing.

**RETRACTED — I wrote "that cannot be this bug" and it was this bug.** On the Metal measurement
relayed the same day (5.313 ms/prompt-token, R² 0.9999, 3.6× slower at 839 tokens) I argued: the fix
is inert on default Metal, both arms prefill identically there, therefore the gap must be something
else. **The premise was an assumption I did not check — that the measurement was taken at Metal's
default.** It was taken under `--metal-fast-prefill`, where `PrefillLast` is *not* declined, the
asymmetry is real, and it is exactly this bug. Struck rather than edited, because the reasoning
failed in an instructive way: I had just corrected one unchecked assumption about Metal (implements ≠
takes) and immediately made a second one in the same paragraph, about which configuration a number
came from. **A measurement's configuration is part of the measurement.**

**Metal, measured** (MacBook, `metal/spec_prefill_regression_test.go`, commit `8ef4d69`):

| configuration | slope (ms/prompt-token) | R² | gate |
|---|---|---|---|
| unfixed, `--metal-fast-prefill` | **5.313** | 0.9999 | FAIL |
| unfixed, Metal **default** (control) | 0.110 | 0.4214 | pass |
| **fixed**, `--metal-fast-prefill` | **0.075** | 0.4211 | pass |

**The fix works on Metal**: 839-token prompt **6180 → 1836 ms (3.37×)**, and the gate was proven red
*and* green, so it is not stuck-red. The control row is what proves the mechanism instead of arguing
it: with batched prefill off, the speculative arm barely moves (6180 → 6064) while `off` collapses
(1714 → 5936) — plain generation coming down to meet the per-token loop speculation was always taking.

**Scope, correctly stated at last:** Metal is affected **only under `--metal-fast-prefill`**; its
`PrefillLast` declines by default and it does not implement `ResidentPrefillKV`. So **CUDA had two
asymmetries and Metal has one, behind an opt-in flag.**

**The lesson, twice over on this page:** an interface check told me *which backends could* be
affected and I read it as *which backends were* — the declining branch was two lines away in the same
file. Then, correcting that, I read a number without asking which configuration produced it. Both
errors are the same shape: treating a property of the system as settled when only part of it had been
looked at.

### After the fix: speculation wins 6 of 6 where it lost 6 of 6

Same harness, same box, same day. Log `docs/measurements/spec02-theta-ab-cuda-FIXED-2026-08-31.log`.

| model | input | `off` | best spec arm | **before** | **after** |
|---|---|---|---|---|---|
| 0.5B | code-continue | 1039 ms | 913 ms | 0.30× | **1.14×** |
| 0.5B | code-continue-2 | 797 ms | 539 ms | 0.34× | **1.48×** |
| 0.5B | agent-loop-turn2 | 631 ms | 624 ms | 0.22× | **1.01×** |
| 1.5B | code-continue | 1507 ms | 1281 ms | 0.35× | **1.18×** |
| 1.5B | code-continue-2 | 1447 ms | 918 ms | 0.39× | **1.58×** |
| 1.5B | agent-loop-turn2 | 1688 ms | 1232 ms | 0.34× | **1.37×** |

The prompt-length control confirms the mechanism is gone: **2.66 → 0.12 ms per prompt token**, with
R² collapsing **0.9977 → 0.33** — no longer a relationship at all, just noise. At the longest prompt
(839 tokens) the speculative path went **2816 ms → 680 ms, a 4.1× speedup**.
Log `docs/measurements/spec02-resident-prefill-FIXED-2026-08-31.log`.

### The gate that was missing

`cuda/spec_prefill_regression_test.go` now **fails** instead of logging. It gates the **defect
signature** — the gap growing linearly in prompt length — not "does speculation pay", which is a
noisy question that would make the test flap. Bar 0.50 ms/prompt-token, between the defect's 2.66 and
the fixed 0.12.

**Proven live, not assumed green:** with the fix stashed the gate reports
`slope = 2.805 ms/prompt-token` and **FAILS** with a message naming `residentPrefillSeed`; with the
fix restored it passes at 0.12. An empty gate is indistinguishable from one that never ran, so it was
made to go red first.

### The Theta question re-opens — and this data cannot close it

With the prefill bug gone, the measured Theta looks better: **+6.8% mean** vs the shipped 0.5, which
would clear the +5% adopt bar. **It is not adopted, because the `off` arm says the noise floor is
higher than the effect.**

`off` is a **control**: this change cannot affect it. Between two otherwise-identical runs on the
same box, the same day, it moved:

| cell | before | after | Δ |
|---|---|---|---|
| 0.5B code-continue | 786 ms | 1039 ms | **+32.2%** ← first cell after model load |
| 0.5B code-continue-2 | 743 ms | 797 ms | +7.3% |
| 0.5B agent-loop-turn2 | 631 ms | 631 ms | 0.0% |
| 1.5B code-continue | 1445 ms | 1507 ms | +4.3% ← first cell after model load |
| 1.5B code-continue-2 | 1345 ms | 1447 ms | +7.6% |
| 1.5B agent-loop-turn2 | 1593 ms | 1688 ms | +6.0% |

A control that swings **32%** on a cell nothing touched puts the noise floor around or above the
effect being adopted. Excluding that first cell the Theta effect is **+3.7%** — the pre-registered
**ambiguous → parked** band. **Verdict: still parked, now for a different reason** — not "the effect
is absent" but "this instrument cannot resolve it".

**The specific defect in my own harness:** it ran no warm-up before the first cell after a model
load, and both 32%-class outliers were exactly those cells.

### Re-run with the instrument fixed — and the verdict stands

The harness now discards a warm-up generation after each model load and takes **5** reps instead of
3. Log `docs/measurements/spec02-theta-ab-cuda-R2-warmup-2026-08-31.log`.

**The instrument fix validated itself against the control:** the 0.5B `code-continue` `off` cell came
back to **777 ms**, against 786 ms in the pre-fix run and the 1039 ms outlier — so that 32% swing was
the missing warm-up, exactly as diagnosed, and is now gone.

`ada@measured` vs `ada@0.5`, six included cells: **−1.2, +1.6, +3.8, +4.0, +6.5, +13.0 %**.

| statistic | value | bar |
|---|---|---|
| **median** — the pre-registered statistic | **+3.9%** | +5% to adopt |
| mean — *not* the pre-registered one | +4.6% | — |
| worst single-cell regression | −1.2% | clause allows −2% ✅ |
| `ada@measured` vs `off`, worst cell | 1.01× | never loses ✅ |

**Verdict: +3.9% → the pre-registered "+2%…+5% ambiguous → parked" band. Theta is NOT re-tuned.**
Two clauses pass and the primary statistic does not, which is what the ambiguous band is for.

Worth stating plainly because it is where motivated reasoning lives: the **mean is +4.6%**, so even
switching to the more favourable statistic after the fact would not have cleared the bar — but the
median is what was pre-registered and the median is what was applied. The candidate is genuinely
better on 5 of 6 cells and rescues the shipped default's worst cell (0.93× → 1.14× in the first run,
1.01× → 1.15× here); that is a real signal, and it is still not a +5% one. **A third run with more
models and more files is what would settle it, not a re-reading of these six cells.**

### What this changes about the earlier sections

- **The `off` regression is fixed on resident CUDA**, so the "PARK all three items" verdict is
  unaffected but its context changes: items 1–3 were parked on **backend-invariant** evidence
  (drafter cost, the oracle ceiling, match provenance), none of which this fix touches.
- **The CPU step-1 result stands unchanged.** The CPU path already batched its prefill, so the 0.83×
  seen there is fixed-K over-draft — a different phenomenon, and still real.
- **`gpu/spec_ngram_resident_test.go` still needs the realistic corpus and a failing gate.** This
  change fixes the code it exercises; it does not fix the harness that missed it.
