# Task — ternary weights: Bonsai 2 27B and the sub-4-bit question

**Status:** CLOSED 2026-09-21 — Gate 0 FAILS for the format this doc describes. See Step 0 Results.
Filed 2026-09-17.
**Venue:** `mac` and `linux`.
**Deliverable of step 0 is a decision, not a kernel.**

---

## Step 0 Results (2026-09-21) — Gate 0 fails; not ambiguous, closing rather than parking

Research only, per this doc's own Constraints — no code written.

| # | Question | Finding |
|---|---|---|
| 1 | Retention independently supported? | **No.** Every quality figure (83.9, 98.2%, every category average) traces to PrismML's own 20-benchmark suite and own harness. Zero third-party reproductions found, 4 days after release (2026-09-17). One independent commentary piece (orcarouter.ai) explicitly flags this ("nobody outside the company has reproduced it") and notes PrismML's own numbers already show a real weak spot: Terminal-Bench 2.1 at 52.8 vs the FP16 baseline's 69.7, which the vendor itself calls "partial." |
| 2 | What's distributed, what format? | GGUF and MLX, at all four sizes (1.7B/4B/8B/27B) — the task doc's assumption that only the smaller sizes ship a cheap format does not hold; all sizes ship both. GGUF uses two **custom** quant types (`PTQ1_0`, `PQ2_0`) that do not exist in upstream llama.cpp. |
| 3 | Ternary format documented well enough to transcribe? | **Split, and this is the load-bearing finding.** The GGUF format — the one actually carrying the 1.76-bit/98.2%-retention claim this task is about — applies a "blockwise Hadamard rotation (block 1024, fixed ±1 signs)... folded into the stored weights, with the matching transform applied to activations at runtime." That is named but never specified: no formula for how the block's sign pattern is derived, no algorithm for the runtime activation-side transform, no public reference code — only PrismML's own closed-source llama.cpp fork (`github.com/PrismML-Eng/llama.cpp`), explicitly stated to be required ("not in stock llama.cpp"). This is real new arithmetic on the activation side, not a loader transcription — Step 0 Q3's own dividing line ("if it is only defined by their kernels, transcribing it is a reverse-engineering job, not a loader job") is exactly what this is. Separately, the MLX 2-bit builds describe a DIFFERENT, simpler scheme with no rotation at all (`w_i = scale_g · t_i`, group 128, plain group-wise ternary affine quant) — genuinely easier to transcribe, but not confirmed to be the same quantization the 98.2% claim was measured on. |
| 4 | Reference to diff against? | **No, for the format this task describes.** No PyTorch/HuggingFace `trust_remote_code` reference exists anywhere for the Hadamard-rotated ternary scheme — the only way to run it is PrismML's own fork, which is not an independent oracle (same vendor, same unverified-elsewhere position as the benchmark claim). The MLX 2-bit variant runs on stock, unmodified `mlx-lm` (`pip install mlx-lm`, no fork) and COULD serve as a real oracle on this Mac — but only for that separate, unverified-retention scheme, not for "Bonsai's ternary format" as advertised. |
| 5 | Vision tower coupling? | Confirmed optional — the vision tower ships as a separate Q8_0 `mmproj` file, loaded only for image input. Text-only use is real and does not pull in a second front. |

**Gate 0 (loadable and diffable) FAILS for the format this task is actually about**, on two of
its three sub-conditions independently: the Hadamard rotation is named but not specified precisely
enough to reproduce, and no standard-framework reference forward exists to diff against — only the
vendor's own closed fork. Per this doc's own rule ("No oracle, no ship — that is not negotiable"),
that alone closes the item; **Gate 1 also fails independently** (no third-party retention evidence
at all, one critical commentary piece actively pointing at a real weak spot in the vendor's own
numbers). Two failures, not one ambiguous one — this is a close, not a park.

**What does NOT close, and is worth naming so it isn't silently lost:** the MLX 2-bit variant is a
genuinely different, smaller, more tractable question — a plain non-rotated group-wise ternary
scheme, runnable via stock `mlx-lm` on this exact machine, that could plausibly clear a Gate 0 of
its own. But it is not confirmed to carry the 98.2%-retention claim (that number was measured on
the Hadamard-rotated GGUF build specifically, per every source found), so pursuing it would need
its OWN retention check, not inherited credit from this task's numbers, and its own task doc rather
than being folded into this one's now-closed verdict. Not started here — filing a new item is a
separate decision, not a natural continuation of this one.

---

## Why this one and not sub-4-bit generally

Going narrower than int4 has been argued against here before, for good reasons: sub-4-bit is a
quality cliff rather than a slope, it attacks decode (where we are already competitive) rather than
prefill (where we are not), and MoE routing is unusually sensitive to precision loss because a
top-k near a tie flips which expert runs.

Bonsai 2 27B (PrismML, released 2026-09-17, Apache 2.0) changes two of those premises:

- **It is quantization-aware training, not post-hoc compression.** The claimed retention is 98.2%
  of aggregate benchmark performance against full-precision Qwen3.8 27B, with the gap over the
  previous Bonsai generation closed from ~95%. A post-hoc 2-bit checkpoint and a QAT ternary one
  are not the same object and the cliff argument does not transfer between them.
- **The base model is one we already support.** It is Qwen3.8 27B underneath — our `qwen3_5`
  family (`docs/capability-matrix.md:28`, currently `experimental: tiny-oracle 100.0%/1.00000
  +coherent`). Layer topology, the Gated DeltaNet 3:1 hybrid, RoPE, attention, SwiGLU: all code we
  have and have run. **The only new thing is the weight representation.**

And the footprint is the part that matters to us:

| | size | fits 8 GB CUDA card | fits 16 GB Mac |
|---|---|---|---|
| Qwen3.8-27B int4 | 15.3 GB | no | no |
| Ternary Bonsai 2 27B | 5.9 GB | **yes** | **yes, with room** |

A 27B-class model has never run resident on either of our boxes. That is the whole case.

---

## What we would be building

Ternary {−1, 0, +1} weights with FP16 group-wise scaling, 1.76 effective bits per weight.

Confirmed greenfield: a grep for `ternary`/`bitnet` across the tree returns **zero** hits. Our
internal compute formats are int4, mxfp4, w4a8, int8int8 and q8_0 — nothing sub-4-bit, and ternary
is not a narrower int4. It is different arithmetic.

The group-wise FP16 scaling is familiar block structure (same shape as every other quantization we
read). The {−1, 0, +1} weights are not.

**One thing makes this a better fit for us than most formats:** a ternary multiply is add,
subtract, or skip. There is no multiply in the inner loop. That suits hand-written Go assembly
better than formats whose performance depends on tensor-core paths we cannot reach from pure Go —
this is a case where owning the forward pass is an advantage rather than a tax.

---

## Which repo

**Start in goinfer. The aikit question does not arise unless the gates clear.**

Everything through gate 2 is reader work — transcribe the ternary layout,
dequantize to an existing internal format, run the normal path, diff against a
reference. No new arithmetic, and the parity oracle it has to satisfy lives here.
The precedent is `decoder/gptoss_safetensors.go`: that layout was transcribed and verified
bit-for-bit in goinfer, not in aikit. Step 0 is even more clearly goinfer's —
whether the model is loadable and diffable at all is a question about our loaders
and our gate runner.

A **native ternary kernel is step 2, and that belongs in aikit**, under the
Experimental tier, the same way Q8_K, W4A8, the acc64 attention kernels and
MXFP4's layout support (v1.36.0) landed there with goinfer calling them. Add,
subtract or skip in the inner loop, hand-written NEON and AVX2.

So the order is: goinfer establishes whether there is anything to build, then
aikit builds the fast path. That is also the cheaper sequence — if gate 0 fails
because the 27B ships only in PrismML's custom kernel format with no reference to
diff against, nothing has been spent in either repo.

## Step 0 — verify the premises before anything else

Every number above is vendor-published on release day. Establish, and report as a short table:

1. **Is the retention claim independently supported?** The whitepaper is the only backing today.
   Search for third-party evaluation. If none exists yet, say so and state how long we would wait.
   A vendor selling compression reporting 98.2% retention on its own benchmark suite is a claim,
   not a measurement.
2. **What is actually distributed, and in what format?** Their platform coverage is CUDA and MLX
   via custom low-bit kernels. Check the HF collection for what the 27B ships as. **If there is no
   GGUF or safetensors for the 27B, we may have nothing to load and nothing to diff against** —
   that alone could close this item. Note that the smaller Bonsai models (1.7B, 4B, 8B) do appear
   to ship GGUF, which would be a much cheaper entry point.
3. **Is the ternary format documented?** Group size, scale layout, packing, element ordering. If it
   is only defined by their kernels, transcribing it is a reverse-engineering job, not a loader
   job. Compare against what `decoder/gptoss_safetensors.go` needed — that layout was transcribed
   from a reference Python library and verified bit-for-bit, which is the standard to meet.
4. **Is there a reference we can diff against?** Our whole validation model is a numerical diff
   against a HuggingFace forward. If the model only runs through their custom kernels, establish
   what the oracle would be. **No oracle, no ship** — that is not negotiable and it may be the
   binding constraint here rather than the kernel work.
5. **Does the vision tower come along?** It is described as multimodal text-and-image. We are
   mid-way through the multimodal seam work; a ternary vision tower is a second front. Establish
   whether text-only use is possible.

---

## Step 1 — only if step 0 clears

Start with the **smallest Bonsai that ships in a format we read** (1.7B or 4B), not the 27B. Same
reasoning as the smollm3 promotion: prove the mechanism on the cheapest case and get a timing
before committing to the expensive one. A ternary 1.7B that loads and diffs correctly tells us
almost everything; a 27B that does not tells us nothing about why.

Scope for step 1: a **reader** producing correct dequantized weights, verified against a reference,
with no new kernel at all — dequantize to an existing internal format and run the normal path. That
separates "can we read it" from "can we compute on it", and the first question is much cheaper.

A native ternary kernel is step 2 and needs its own authorisation.

---

## Pre-registered gates

**Gate 0 — loadable and diffable.** The model is distributed in a format we can read, its ternary
layout is documented well enough to transcribe, and a reference forward exists to diff against.
Any one of these failing closes the item. Record which.

**Gate 1 — retention holds outside the vendor's own suite.** At least one independent evaluation,
or our own coherence check on the small model, consistent with the claim. State the threshold
before looking.

**Gate 2 — dequantize-and-run works at parity.** The small model loads and matches the reference
within our normal tolerance, through the existing int4 or f32 path. Fails → the layout
transcription is wrong and that is the finding.

Anything past gate 2 — a native ternary kernel, the 27B, the vision tower — is separate work.
Ambiguous → parked, as always.

---

## What this is worth if it lands

A 27B-class model, with vision, running resident on an 8 GB card and a 16 GB laptop, from a single
static Go binary with no native dependency. That is a distinctive capability rather than an
incremental one, and it is the first thing in months that would change what hardware goinfer makes
useful rather than how fast it runs on hardware it already serves.

It is also a lane nobody else in the Go space occupies: yzma inherits llama.cpp's formats and
llama.cpp has no ternary path for this model.

Set against that: this is a single vendor's format, one day old, with self-published numbers, and
the work is real kernel engineering. Step 0 exists so we find out which of those dominates before
spending anything.

---

## Constraints

- Step 0 is reading and searching. No code.
- No cgo, on any path.
- Parity gating applies unchanged. A new weight format does not get a weaker bar.
- Do not start with the 27B.
- Label the regime at the point of recording.
- Do not use the words "honest" or "honesty".

---

## Related

`docs/capability-matrix.md:28` (`qwen3_5`, the base family), `decoder/gptoss_safetensors.go` and
`docs/completed/task-mxfp4-gptoss.md` (the precedent for transcribing a low-bit layout and verifying it
bit-for-bit), `docs/tasks/task-fp4-formats.md` if landed (the FP4 direction, which this is an
alternative to rather than a continuation of), `docs/post-v1.0-models.md` (the standing judgement
that families stopped being the high-leverage work — this item is a format, not a family, which is
why it is filed separately).
