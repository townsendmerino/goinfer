# Chapter 9 — Guessing ahead

*Decode is slow because it is sequential and because it wastes the hardware. Speculative
decoding attacks both at once, and it is the closest thing in this book to optimistic
concurrency control.*

---

## The observation

Chapter 8 established that decode is memory-bound: reading the weights dominates, and the
arithmetic units are largely idle waiting for them.

Now notice something. Running the model on *one* position and running the model on *five*
positions cost nearly the same, because both read the same weights once. The extra positions
ride along in the arithmetic capacity that was going unused.

So verifying five candidate tokens is roughly as cheap as generating one token. If you had five
candidates from somewhere, you could check all five in a single pass.

---

## The scheme

1. Some cheap **drafter** proposes the next k tokens.
2. The real model runs once over all k, producing what it would have said at each position.
3. Compare. Accept the longest prefix where the drafter agreed with the model. Discard the
   rest.
4. Repeat.

```
  drafter proposes 5:   the   cat   sat   on   the
  model would say:      the   cat   sat   in   a
  compare:               ✓     ✓     ✓     ✗    —

  accept the leading run of agreements: "the cat sat"  → 3 tokens
  then take the model's own token at the first disagreement: "in" → 1 more
  discard everything after the disagreement, and start the next round

  4 tokens produced for roughly the price of 1 forward pass
```

If the drafter guessed 4 of 5 correctly, you produced 4 tokens for roughly the price of 1. If
the drafter guessed none correctly, you paid slightly more than a normal step and got one
token — the same token you would have gotten anyway.

The critical property: **the output is identical to what the model would have produced alone.**
The drafter proposes; the model decides. A bad drafter costs speed and never costs correctness.

If you've written optimistic concurrency, this is that. Guess, do the work, validate, roll
back on conflict. Chapter 3's optimistic forward was a one-token version of the same bet.

In this repo it's the `docs/spec/` series, with `decoder/speculative.go` and
`decoder/blockspec.go` as the implementation.

---

## Where drafts come from

`docs/spec/00-core.md` treats the drafter as a plugin behind one interface — grammar, n-gram,
model, head are all just different sources of a proposal.

**A smaller model.** Run a 0.5B model to draft for a 7B model. Simple, and a smaller model pays
only when the target step is expensive enough to dwarf the draft — which means big models and
GPUs, not small models on CPU.

**An n-gram / cache drafter.** No model at all: look at the recent context and propose a
continuation that already appeared in that context. Almost free, and startlingly effective on
copy-heavy traffic — code editing, tool output, anything that quotes itself. The n-gram drafter
is shipped, wired into `cmd/serve --spec ngram`, and the n-gram drafter reuses the warm KV
prefix from Chapter 4.

**A grammar.** When output must match a JSON schema, the grammar often forces the next few
tokens regardless of what the model thinks. Those are free tokens.

**A trained head.** A small extra network bolted onto the target model, trained to predict
its next few tokens. EAGLE-3 is one design; **MTP** — multi-token prediction — is another,
where the head ships inside the checkpoint because the model was trained with it.

---

## The number that decides everything

One quantity governs whether any of this pays: **α**, the average number of tokens accepted
per verification round.

The arithmetic is unforgiving:

```
  without speculation:   α tokens cost   α × (one decode step)

  with speculation:      α tokens cost   1 × (one target forward pass)
                                       + the drafting work for k tokens
                                       + bookkeeping

  so speculation pays only when

           cost of draft + verify + bookkeeping
     α  >  ------------------------------------
                  cost of one decode step
```

If α is 1.0 you have added overhead for nothing. Break-even is usually somewhere between 2 and
4 accepted tokens per round, and where break-even sits depends on how expensive the draft is
relative to the target step. Note what the denominator does: **the cheaper the target step, the
higher α has to be** — which is why speculation suits big models on fast hardware and not small
models on CPU.

Measured here, an EAGLE-3 head reached about 1.6 tokens per verify — genuinely better than
nothing, and a net wall-clock *loss* on CPU because the draft itself was not free enough. The
conclusion recorded was that the binding constraint is α, not the verification machinery.

An MTP head — the kind that ships inside the checkpoint, because the model was trained with the
head — was measured against that 1.6 on a 0.8B target, drafting K=6:

```
  suite    tokens per verify        vs the 1.6 reference
  code           2.024                     above
  math           2.905                     above
  chat           2.476                     above

  the gate asked for two suites of three; all three cleared
```

**Read that pass as narrowly as it was pre-registered**, because a good result is exactly when
people stop reading the gate. Three limits, all recorded before the measurement:

- **It is a pass on MECHANISM, not on economics.** What it establishes is that a head trained
  jointly with its own target predicts that target better than an imported general head does. The
  break-even gate was *not evaluated*, because a 0.8B target is precisely the regime where the
  denominator in the inequality above is too small for the economics to mean anything.
- **The third digit is noise.** One prompt per suite, 42 draft positions, no repeats. Changing the
  prompt's *formatting* — wrapping the identical text in a chat template — moved the code suite
  from 2.238 to 2.024 on its own. The spread across suites is larger than any gap being read.
- **A pass authorises nothing.** The models carrying MTP heads are the Qwen families, which are
  exactly the families the rollback seam refuses for the reason the next section gives. Shipping
  this would require the state-checkpoint work first, which is a materially larger build than
  wiring up a drafter.

This is also why `docs/spec/README.md` states the scheme as a scorecard rather than a
recommendation: a model drafter translates only when the target step is expensive enough.
Chapter 3's rule shows up again — the fraction is set by the denominator.

---

## Rollback, and the models that can't

Step 3 said "discard the rest." That is where it gets architectural.

Verifying k tokens advances the KV cache by k positions. Accepting only 3 of 5 means undoing
2. For standard attention that's trivial: the cache is a list, so drop the last two entries.

Chapter 4's Gated DeltaNet layers keep a running total instead. Each token folds into a
fixed-size matrix updated in place. There is no "position 4" to remove, and the update cannot be
inverted because the gating deliberately discards information. `decoder/speculative.go`
therefore refuses Gated DeltaNet architectures outright, with an error saying rollback cannot
losslessly restore the state.

That refusal is not a missing feature. With what is stored, the earlier state does not exist.

The fix would be checkpointing: snapshot the recurrent state before the verify, restore it on
rejection. The state is fixed-size and doesn't grow with context, which makes it appealing.
Measured on CPU, the snapshot costs 2–4% of a decode step — cheap. Measured on a resident GPU
path, where the decode step is 16× faster, the same fixed-size copy costs about 11% of a decode
step, which is at the edge of not paying.

The pattern is worth extracting: a fixed cost measured against a shrinking denominator gets
worse with every improvement elsewhere. Cheap on the slowest backend is the least interesting
place for a fixed cost to be cheap.

---

## What it costs

The n-gram drafter is shipped and wins on copy-heavy traffic. The head-based schemes are
measured and parked, because α has not cleared break-even in the regimes reachable on this
repo's hardware.

The fair summary is that speculative decoding is the chapter where the most careful work has
produced the fewest shipped wins — and that the careful work is why the wins are few. A speculation suite here was found in
which no verification width beat running no drafter at all, visible only because *off* was
included as a competitor. That is Chapter 11's do-nothing-arm rule earning its place.

The durable lesson: speculative decoding is not a speedup, it is a bet on α. Everything else
— drafters, widths, rollback machinery — is infrastructure for making and settling that bet.

Chapter 10 goes down a level, to the kernels all of this runs on and the pure-Go constraint
they operate under.

---

*Sources: `docs/spec/00-core.md`, `docs/spec/02-cache-ngram.md`, `docs/spec/05-eagle3-head.md`,
`docs/spec/09-mtp-heads.md` §"Gate 1 result" (the MTP numbers, their pre-registered reading, and
the precision caveats), `decoder/mtp_head_test.go` (`TestMTP_acceptedLength`),
`decoder/speculative.go:89-93`, `decoder/deltanet.go:145-153`, `CLAUDE.md` (do-nothing arm).*
