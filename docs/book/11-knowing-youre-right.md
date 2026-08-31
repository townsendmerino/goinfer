# Chapter 11 — Knowing you're right

*You know how to profile Go. This chapter is about why that isn't enough here, and what
this codebase does instead.*

---

## The problem this chapter exists for

In most Go work, correctness and performance are separate concerns. A function returns the
right answer or the function does not; separately, the function is fast or the function is
not. You test correctness and you benchmark performance, and testing and benchmarking barely
touch.

Inference breaks that separation. Almost every optimization in this repo changes the
arithmetic. Running attention in f32 instead of f64 is faster because it does less work,
and it produces different numbers. Quantizing weights to 4 bits is faster because it moves
fewer bytes, and it produces different numbers. Speculative decoding is faster because it
guesses ahead, and if the rollback is wrong it produces different numbers.

So "is the code correct?" and "is the code fast?" stop being independent questions. Every
speedup is a proposed trade, and you cannot evaluate the trade without measuring both halves
of the trade. Measuring both halves is the whole subject of this chapter.

---

## Half one: what "correct" means when the arithmetic moved

The reference is HuggingFace `transformers` running the same checkpoint in Python. If
goinfer's forward pass disagrees with HuggingFace `transformers`, goinfer is wrong. That
comparison is the parity gate, and the parity gate is why the repo has a `parity` runner in
`cmd/gate` alongside the normal tests.

But "disagrees" needs a definition, and there are two useful definitions.

**Bit-identical** means the output tokens are exactly the same, every time, for the same
input and the same seed. Bit-identical is the strongest claim, and the repo makes the
bit-identical claim for ordinary decode. Bit-identical is also the claim that makes
speculative decoding tractable: a drafting scheme is only allowed to ship if greedy decoding
with the drafter produces byte-for-byte the same tokens as greedy decoding without the
drafter. That requirement turns "is the speedup safe?" into a test rather than a judgement
call. Either the tokens match or the tokens do not.

**Cosine similarity** is the weaker measure, used when the arithmetic deliberately changed.
Run the same layer in f64 and in f32, take the two output vectors, and measure the angle
between the two output vectors:

```
              a · b                 sum of a[i]*b[i]
  cos(a,b) = ---------  =  ------------------------------------
             |a| |b|        sqrt(sum a[i]²) * sqrt(sum b[i]²)

  cos = 1.0      the two vectors point the same way  (identical)
  cos = 0.9976   the shipped divergence for --cpu-fast-attention, dense path
```

Cosine similarity ignores length and measures only direction, which is what you want when
the question is "did the answer move?" rather than "did the answer change size?". A cosine
of 0.9976 is not bit-identical and is not claimed to be; a cosine of 0.9976 is a stated,
measured divergence that the user opts into with a flag.

The Go instinct here is roughly right: bit-identical is a `reflect.DeepEqual` on the output,
and cosine similarity is a fuzzy comparison you reach for when exactness is not the contract.
Where the analogy breaks is that the fuzzy comparison is not a convenience — the fuzzy
comparison is the actual product specification for that flag, and the fuzzy comparison has to
be measured on the workload that can expose the worst case rather than on whatever workload
is easiest to run.

---

## Half two: how a measurement lies

This half of the chapter is the part with no Go analogue, and this half is where most of the
repo's hard-won material lives. `CLAUDE.md`'s measurement-discipline section is the
accumulated list. Here are the failure modes worth internalizing, each failure mode recorded
because it actually happened in this tree.

### The instrument returns a plausible number for the wrong reason

The most dangerous measurements are the ones that look fine.

A benchmark of a device-to-device GPU copy came back at 8145 GB/s. The card has roughly
448 GB/s of memory bandwidth, so the number was not merely surprising — it was impossible.
The cause was that the copy call is asynchronous despite the copy call's name and despite
the copy call dispatching the synchronous primitive, so timing the copy call alone measured
the dispatch and not the copy. Timing copy-plus-synchronize gave the real figure.

What saved that measurement was having a physical bound to violate. Most wrong numbers
violate nothing. A slot sweep run while another job held the box gave 0.544% where the
clean run gave 0.531–0.537% — close enough to be believed, and wrong. Contamination that
produces obviously-broken output is harmless; contamination that produces plausible output is
the contamination that ships.

The defence is structural rather than attentional: run on a quiet box, gate the run on load,
and report spread rather than a single draw. A number recorded from one run has no error bar,
and a later reader will treat it as central tendency when it might have been a 90th
percentile draw. That exact thing happened here — a recorded anchor of 116.6 turned out to
be near the top of its own build's distribution, which manufactured a regression that never
existed.

### The component is right and the composition is wrong

Two instances, in unrelated subsystems, both worth knowing.

A gate with a two-way hysteresis band had a test that drove the gate's `Observe` method in an
unconditional loop and confirmed the hysteresis band worked. Production calls `Observe` only
inside the branch that the gate's own `Should()` guards:

```
  what the test did                 what production does

  for each sample:                  for each sample:
      gate.Observe(x)                   if gate.Should():        ← the guard
      gate.Should()                         gate.Observe(x)      ← only reached when ON

  every sample observed             once the gate turns OFF, Observe is never
  → the band works                  called again → the estimate freezes → the
                                    re-enable half of the band is UNREACHABLE
```

The test passed. The test was correct. The test vouched for behaviour the system could not
produce.

The rule that came out of the hysteresis-band failure: when a component's contract depends on
*how often* or *under what condition* the component gets called, test the component through
its caller, or the test is only asserting your assumption back at you.

The same shape appears in measurement. The real workload — snapshotting recurrent state on the
resident CUDA path — issues 36 separate device copies, and that composition costs 2× the
primitive:

```
  one contiguous copy of the same bytes      347 GB/s
  36 separate copies (the real workload)     174 GB/s     2x worse

  why: 18 of the 36 copies are 73 KB each, against roughly 9 µs of
  dispatch apiece — those are dispatch-dominated, not bandwidth-bound.
  The 1 MiB copies are not. Batching the small ones recovers most of
  the penalty: roughly 446 µs → 250 µs.
```

Isolation proves the primitive and never the composition.

### The comment claims coverage the body doesn't have

A variant of the composition failure, and worse, because a false doc comment defeats the
person auditing rather than the person running.

`TestA3FastAttentionDivergence` opened with a doc comment stating that it pinned an
exclusion: `--cpu-fast-attention` could not be enabled for MoE architectures at all. The
body loaded a dense checkpoint and asserted nothing about MoE anywhere. Anyone checking
whether that exclusion was justified would read the promise, find a test name that matched,
and stop.

The exclusion had never been measured on a MoE architecture. When the exclusion finally was
measured, the exclusion did not hold, and the flag was extended to cover MoE architectures.

The check: for any doc comment asserting coverage, the test body must contain an assertion
naming the thing the doc comment claims. Note that neither the false doc comment nor the
hysteresis-band failure is caught by running the suite — both defeat the same defence, which
is reading the test name instead of the test body.

### The input is calibrated for a different question

The subtlest one, because the instrument is fine.

A benchmark prompt file used entries with four unique words each. Four-unique-word filler is
correct for measuring throughput on a dense model: decode cost is content-independent, and
repeated filler gives exact token counts. Feed the same prompt file to a Mixture-of-Experts
model and the prompt file silently becomes a best case, because on a Mixture-of-Experts model
*content is routing*. Identical rows select identical experts, the expert weights stay hot in
cache, and the feed-forward network reports a cost no real prompt will produce.

The same prompt file had already distorted a different measurement, where the value being
measured depended on how predictable the generated text was, and four-unique-word filler is
maximally predictable. A crossover measured at T≈0.95 on that filler moved to T≈0.37 on
real prose.

Two lessons. First, a prompt set calibrated for one measurement class is not usable for
another measurement class, and the failure is silent. Second, when you find a confound like
the four-unique-word prompt file, check whether the confound can be turned into a control arm
rather than discarded — the uniform-routing case turned out to be exactly the ceiling that
expert-major batching could aspire to, so the confound went from bug to bound.

### The stopping rule encodes an assumption you didn't state

"Stop when a doubling buys less than 5%" sounds like disciplined engineering. The rule is
valid only where the curve is known to be monotone-diminishing.

Measured here on a slot sweep: 8→16 bought +4.1%, under the bar, so the rule as written
stops at 16. The only resolvable win was **+14.8% at N=64**, two doublings later.
Plateau-then-step is the normal shape wherever a resource threshold gets crossed, and
crossing one was exactly what happened at 64.

What caught the stopping rule was that two things had been pre-registered — the stopping rule
*and* the full ladder — and the two pre-registrations disagreed. That disagreement is the
durable lesson. After a rule fails, the instinct is to write a better rule. The more reliable
fix is a second, independent pre-registration, so that the wrong rule can be caught by the
right one.

### The guard inverts under the condition it guards

Same sweep. A memory ceiling keyed on RSS, written specifically to catch the slot-pressure
cliff, reported **263–426 MB at N=128 against 1154 MB at N=8** — less memory at the failure
point than at the baseline. Darwin's unified buffer cache reclaims under pressure, so RSS
reports what survived, not what was asked for.

A guard that inverts under the guard's own condition is worse than no guard, because such a
guard actively reassures. Key a budget check on a quantity you compute yourself — allocated
bytes are known at build time — and never on the operating system's account of what remains.

### The retraction doesn't reach every page

Striking a number where you found it is not finishing the job.

A figure of `~1.2–1.4 tok/s` was withdrawn in one document and went on disqualifying a model
from a qualifying run in *another* document for months. Direct measurement later put the same
quantity at 1.52–1.73 on CPU and 1.97–2.02 on Metal. The withdrawal was recorded correctly in
the document where the figure was found, and the withdrawn figure propagated anyway.

The convention: when you strike a number, grep for **the figure with its unit**, right then,
at the retraction site, and fix every instance while you still have the context. The unit
matters — the bare digits matched six unrelated quantities in this tree; the digits plus
`tok/s` matched only the real ones.

The grep-the-figure-with-its-unit rule is deliberately a convention rather than a lint. The
citation lint validates that a citation points somewhere stable and explicitly scopes out what
is *said* about the target; scoping that out was a considered decision, and one incident does
not overturn a considered decision. And note the reach limit
either way: figures propagate into published artifacts outside the repo, where no repo-side
tooling can ever follow them.

---

## The habits, condensed

Most of the failure modes above reduce to a handful of practices that this repo applies as
defaults.

**Difference matched observations; do not pool them.** Where two variants can be interleaved
on the same input, pair the two variants rather than comparing two piles of numbers:

```
  POOLED — every A run against every B run
     A runs:  ▇ ▇  ▇   ▇ ▇▇   ▇        standard deviation 10–35 tok/s
     B runs:   ▇▇ ▇  ▇▇   ▇  ▇
     the spread CONTAINS the input-to-input variation, which is larger
     than the ~8% effect → the effect is invisible

  PAIRED — A and B on the SAME input, then average the differences
     input 1:  A₁ vs B₁  →  d₁          standard deviation 5.5–8.7
     input 2:  A₂ vs B₂  →  d₂          11 of 12 differences agree
     input 3:  A₃ vs B₃  →  d₃
     the input cancels out of every difference → the effect is visible
```

The same data, read the two ways, disagreed about whether the effect existed at all. Pairing
does not make the measurement more precise by taking more samples; pairing removes a source of
variance that pooling leaves in.

**Include the do-nothing arm.** "Beats every configuration" means nothing if *off* wins. A
speculation suite here was found where no verify width beat running no drafter at all, and
that result was visible only because `off` was in the comparison.

**Pre-register the decision rule**, including an explicit *ambiguous → parked* band. The zone
just below a threshold is where motivated reasoning lives.

**Commit negative results with the same care as wins.** And never re-baseline a floor because
a number moved — move a bar only with a mechanism, or you have blessed a regression.

**Read the prior art before re-deriving.** A document may say something narrower than the
document's headline. One verdict here about a specific drafter, if read as a verdict on the
technique in general, would have closed a live question with the wrong evidence.

---

## What it costs, and why it's worth it

A worked example of the whole chapter in one finding.

A profile of a 4-layer model slice showed that attention accounted for **97.1%** of an
8k-token Mixture-of-Experts prefill. A flag existed that made attention 2.28× faster on dense
models, and that flag refused Mixture-of-Experts models — behind a doc comment claiming the
refusal was pinned by a test that asserted nothing about Mixture-of-Experts models.

Measuring the exclusion showed the exclusion's mechanism was real in kind but did not hold in
magnitude. The flag was extended to Mixture-of-Experts models. And the win, measured on the
*full* model rather than on the 4-layer slice, was **1.52×** — not the 3.11× the slice
implied.

Every step there is a lesson from this chapter. The slice figure was regime-specific and would
have been quoted as a whole-model figure. The exclusion was documented as tested and was not
tested. The real win was smaller than the profile promised, and the real win is still a large
win worth having.

The cost of this discipline is that things take longer and some sessions end with a number
and no feature. The return is that the numbers in `docs/benchmarks.md` can be traced to a
commit, a date, and a machine — and that when the machine's driver stack changed underneath
them, the page could say so and mark the affected rows stale rather than quietly carrying
them forward.

That traceability is the actual product. An inference engine that is fast and cannot prove
that it is fast is indistinguishable from an inference engine that is wrong.

---

*Sources: `CLAUDE.md` (measurement discipline, tests), `docs/benchmarks.md` (provenance rule,
re-anchor banner), `docs/parity-coverage-policy.md`, `docs/QUEUE.md` (retraction record).*
