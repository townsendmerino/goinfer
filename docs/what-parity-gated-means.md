# What "parity-gated" means

goinfer's README says it supports 35 model families, HuggingFace-parity-gated. This page explains
what stands behind that phrase, and — as importantly — what it does not cover.

---

## The short version

For each family, goinfer's forward pass is diffed against the same checkpoint running in
HuggingFace `transformers`. The reference is Python; goinfer is the thing under test. If the two
disagree beyond a stated tolerance, the family does not ship a validated claim.

That is the whole idea. Everything below is detail about how strong the comparison is, which
varies by family and by what hardware can run.

---

## What gets compared

Where a family is validated against released weights, the check is:

- **argmax agreement** — does goinfer pick the same token as the reference, at every position
- **logit cosine** — how close the raw scores are, not just the winner
- **ΔNLL** over a real prompt set
- **a norm-checked spot trace**, so a passing cosine can't hide a scaled-but-wrong tensor

Argmax alone is not enough and the repo has a case proving it. Promoting `olmo3` to a released-weights
oracle produced exact argmax agreement and an exact 8-token continuation, while cosine came in at
0.992789. The gap was real: goinfer was applying YaRN RoPE scaling to the full-attention layers but
not the sliding-attention ones, where the reference applies one shared table to all of them. Only
the cosine bar caught it.

For large models where running a full Python reference is impractical on available hardware, the
fallback is **coherent free-generation plus ΔNLL** over a real prompt set. Which bar was used is
recorded per family rather than blurred.

---

## The tiers, in plain terms

`docs/capability-matrix.md` labels each family with its strongest validation. In rough order of
strength:

| tier | what it means |
|---|---|
| `full-oracle` / `real-oracle` | diffed against a **released** checkpoint — the strongest claim |
| `weight-diff` | diffed against released weights by another route |
| `tiny-oracle` | diffed against a small purpose-built fixture, used when **no released model is small enough to run on available hardware** |
| `coherent-gen` | a real model ran and produced coherent output; no numeric oracle |
| `shared-path: X` | an alias family riding X's oracle on the same forward file — X's numbers are the measurement |
| `pending` | not yet recorded |

A `+coherent` suffix means a real model also ran qualitatively.

**`tiny-oracle` is a weaker claim and we say so.** It is what you get when the smallest released
member of a family will not fit on the machines available. It exercises the same code path, but a
fixture built from the same wiring cannot catch a defect *in* that wiring. Two of the families
promoted from `tiny-oracle` to released-weights validation turned out to have real defects sitting
behind a passing tiny fixture.

---

## Scope: a green run names what actually ran

**A missing fixture skips rather than fails.** A run reporting `28 ran / 20 skipped / 0 failed` is
a pass — of 28 things. What a parity run proves is scoped to the checkpoints that machine has on
disk.

This matters when reading any claim of a green suite. On a MacBook on 2026-08-31, all eleven
GGUF-quant gates skipped for want of a local checkpoint, while int4 and one of three int8×int8
goldens ran. Quote a run's counts, not the word "green."

---

## Before a release

A qualification sweep runs one representative **real** checkpoint per family per backend — not
synthetic weights, not fixtures. Per cell:

- **correctness**, preferably argmax plus logit-cosine against the CPU path on the same real
  weights (the backend-versus-CPU twin comparison), with the coherent-generation fallback where a
  full CPU reference is impractical
- **a timing**, measured server-to-server through the HTTP server with sampling, detokenization and
  JSON included, reported alongside a reference peer on the same machine

So the serve path is exercised end to end, not just the decoder.

---

## What this does not tell you

**Parity gating is not a quality benchmark.** It establishes that goinfer computes what the
reference implementation computes on the same weights. It says nothing about whether the model is
good at anything.

Concretely, these are outside what the gates cover:

- task accuracy, reasoning benchmarks, or any measure of answer quality
- whether a quantization you chose degrades output for your use case — quantization is a stated,
  measured numerical trade, not a quality claim
- defects that preserve the numerics but break the surrounding behaviour: a wrong chat template, a
  bad stop token, a tokenizer edge case. The coherent-generation checks would likely catch the
  worst of these, and only on the families where that bar is used

If the model is worse than you expected, parity gating being green is consistent with that. It
means the engine is not the reason.

---

## One thing the gates caught that was not ours

Promoting `internlm2` to released-weights validation, the diff failed at cosine 0.87. The obvious
reading was that goinfer was wrong.

It was not. The model repo's remote-code rotary embedding registers `inv_freq` as a
non-persistent buffer, and the `transformers` fast-init path in use never re-ran that
initialization for buffers absent from the checkpoint — so the frequency table was uninitialized
memory rather than the computed values. goinfer had been correct throughout.

A strict numeric bar does not tell you *who* is wrong. It tells you that something is, and the work
is finding out which.

---

## Going deeper

- [docs/capability-matrix.md](capability-matrix.md) — the generated per-family table
- [docs/parity-coverage-policy.md](parity-coverage-policy.md) — how coverage is scoped and claimed
- [docs/parity-hunt-playbook.md](parity-hunt-playbook.md) — how a parity failure is chased
- [docs/benchmarks.md](benchmarks.md) — current performance claims, provenance-gated
