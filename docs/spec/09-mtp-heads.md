# 09 — MTP / NextN self-draft heads

> **Defensive publication, 2026-08-27.** Scope: measure acceptance. Nothing here is built, and
> nothing in the serving path, router, guard or P15 is touched. This page exists so the decision to
> build or not build rests on a number rather than on the availability assumption that
> `docs/ollama-chase.md` D2 was written from.

## Idea

A multi-token-prediction head is a self-draft: one extra transformer block, trained jointly with
the target, that predicts token *t+1* from the target's own hidden state at *t*. It drafts without
a second model and without an imported head, so the draft distribution is the target's own — which
is the term [05](./05-eagle3-head.md) found binding.

The round shape is EAGLE-3's: a head forward per drafted token, plus the capture/correction forward.
The draft is not free. **MTP changes exactly one term — α.** Everything downstream (batched verify,
rollback, the router) is already built for [05](./05-eagle3-head.md) and
[08](./08-dspark-dflash.md) and is deliberately out of scope here.

## Why the D2 dismissal is superseded

D2 reads: *"Requires model support; most checkpoints do not have the heads."* That was true when
written. Our own loader now contradicts it — we detect these heads, name them, and skip them:

| site | what it does |
|---|---|
| `decoder/gguf_qwen35.go:33` | `numLayers := blocks - u("nextn_predict_layers")` — drops the NextN block |
| `decoder/gguf.go:644` | same subtraction, with the comment "block_count includes the trailing NextN/MTP block(s) goinfer drops" |
| `decoder/weights.go:541` | "MTP heads (`mtp.*`) are simply never requested" |
| `decoder/registry.go:1143` | `num_nextn_predict_layers` MTP head is dropped |

## Gate 0 — inventory (RUN 2026-08-27, PASSED on availability)

Checkpoints already on disk, scanned by reading GGUF metadata and safetensors indices. No model was
loaded.

| checkpoint | format | arch | head | head size | own embed / LM head? |
|---|---|---|---|---|---|
| `qwen3.6-35b-a3b-Q8_0.gguf` | GGUF | qwen35 MoE | `blk.40` + `nextn.*`, 20 tensors | **844.6M** | no — shares target's |
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | GGUF | qwen35 dense | `blk.64` + `nextn.*`, 15 tensors | **424.7M** | no — shares target's |
| `GLM-4.5-Air-Q2_K.gguf` | GGUF | glm4moe | `blk.46` + `nextn.*`, 23 tensors | **3616.6M** | **yes** — own `embed_tokens` + `shared_head_head`, 151552 vocab |
| `qwen3.5-0.8b` | safetensors | qwen35 dense | `mtp.*`, 15 tensors | — | no |
| `qwen3.6-35b-a3b` | safetensors | qwen35 MoE | `mtp.*`, 19 tensors | — | no |
| `qwen3.8-27b` | safetensors | qwen35 dense | `mtp.*`, 15 tensors | — | no |
| `qwen3next-80b-partial` | safetensors | qwen3_next | `mtp.*`, 1553 tensors (per-expert) | — | no |
| `moonlight-16b` | safetensors | deepseek_v3 | `num_nextn_predict_layers: 0` | — | none present |

**Three families carry heads (qwen35 line, qwen3_next, glm4moe), so Gate 0 passes on availability.**

Four facts from the scan worth carrying forward:

1. **The head is ONE block everywhere here.** `nextn_predict_layers: 1` on all three GGUFs, and a
   single `mtp.layers.0.*` in every safetensors checkpoint. No multi-block heads on disk.
2. **GGUF and safetensors store it differently.** GGUF appends it as a trailing *block*
   (`blk.<N>.*` for the transformer weights, `blk.<N>.nextn.*` for the projection and norms) and
   declares the count in metadata. safetensors puts it in its own `mtp.*` namespace and — measured
   here — **does not declare `num_nextn_predict_layers` in `config.json` at all** for the Qwen
   checkpoints. Detection therefore differs by format: a metadata key in one, tensor presence in
   the other.
3. **GLM's head carries its own embedding and LM head; Qwen's does not.** That is the whole 3.6 GB
   against 424M–845M, and it changes the cost model per family rather than per size.
4. **The Qwen 3.6 head is MoE** (256 experts in the head block), so its 844.6M parameters are not
   844.6M of work per drafted token. Any cost estimate must use active, not total.

## The binding constraint is the seam, not the checkpoints

`decoder/forwardn.go:123`:

```go
func (m *Model) specRollbackSafe() bool {
	a := m.w.arch
	if a.granite != nil || a.nemotron != nil || a.qwen35 != nil { return false }
	if a.SlidingWindow > 0 { return false }
	return true
}
```

`qwen35Architecture`, `qwen35DenseArchitecture` and `qwen3NextArchitecture` all set `qwen35`.
`glm4moeArchitecture` sets none of the three and `SlidingWindow: 0`.

| family with an MTP head | admitted by `specRollbackSafe`? |
|---|---|
| qwen3.5 / 3.6 / 3.8 (qwen35 line) | **no** |
| qwen3_next | **no** |
| GLM-4.5-Air (glm4moe) | **yes** |

**Four of the five MTP-bearing checkpoints cannot speculate today**, for reasons that predate MTP
and have nothing to do with it — exactly the shape [08](./08-dspark-dflash.md) records, where the
seam and not the checkpoint was the constraint.

### The refusal is NOT 08's capture seam — it is rollback safety, and that is worse

Worth settling before Step 2, because it sets the size of whatever follows. The two gates are
different lists:

| gate | refuses |
|---|---|
| `ForwardCapture` (`decoder/model.go:707`) — 08's capture seam | granite, nemotron, mla, llama4 — **not qwen35** |
| `specRollbackSafe` (`decoder/forwardn.go:123`) | granite, nemotron, **qwen35**, `SlidingWindow > 0` |

**qwen35 passes the capture seam and is refused by rollback safety.** The cause is in the arch
itself: `qwen35Architecture` sets `layerIsLinear: cfg.IsLinearLayer // Gated DeltaNet layers`, and
`specRollbackSafe`'s own comment names recurrent state as the reason — a verify advances the
target's KV by K and rolls back the rejected tail, which a recurrent state cannot losslessly
restore.

This is the same root cause already recorded for this family in `docs/qwen3_5_moe.md:114`: the
recurrent state "is not position-truncatable", cross-call prefix KV reuse "falls back to full
recompute", and *"optimizing those for hybrid models (state checkpoints) is a later track."*

**So if Gate 1 passes on a Qwen family, the follow-on is not a capture-seam project — it is that
state-checkpoint track**, and that is a materially bigger build than wiring a seam. The likely
shape of a passing result is therefore "α is good on Qwen, and shipping it requires hybrid state
checkpoints first". That is still worth knowing, and it is worth knowing *before* the probe rather
than after.

**This does not block Gate 1.** Acceptance is measured offline against recorded target outputs, the
way [05](./05-eagle3-head.md)'s `TestEagleAcceptedLength` does it: draft K, count the leading run
that matches what the base actually decodes. No KV rollback happens, so `specRollbackSafe` is not
consulted. **The seam blocks deployment, not measurement** — and if Gate 1 fails, the seam never
matters.

## What it requires

**The head cannot currently be loaded at all.** That is the one piece of work Gate 1 needs, and it
is a load-time read rather than a serving change:

- a loader path that *requests* the tensors the four sites above deliberately skip — `blk.<N>.*` +
  `blk.<N>.nextn.*` for GGUF, `mtp.*` for safetensors — and returns them as a structure separate
  from the target's layers, leaving `numLayers` and every existing forward untouched;
- **two detection paths, because the formats disagree.** GGUF declares the count in metadata
  (`nextn_predict_layers`, arch-prefixed: `qwen35moe.`, `qwen35.`, `glm4moe.`); the safetensors Qwen
  checkpoints **do not declare `num_nextn_predict_layers` in `config.json` at all** — measured, see
  Gate 0 — and the head is discoverable only by tensor presence in the index. A single
  metadata-keyed detector finds nothing on safetensors and would report "no head" on a checkpoint
  that has one;
- a single-block forward for the head, reusing the existing block forward rather than a new one;
- for Qwen, the target's embedding and LM head are reused (the head has none); for GLM, the head's
  own `nextn.embed_tokens` / `nextn.shared_head_head` are used instead.

`TestEagleAcceptedLength` is then pointed at it. **Answering the brief's question directly: the
existing harness cannot be pointed at an MTP head as things stand, because nothing loads the head —
and the smallest adapter is the loader above, not a new measurement stack.**

## Licensing / IP note

The heads ship inside checkpoints already on disk under their existing licences (Qwen3.5/3.6/3.8,
GLM-4.5-Air). Nothing is imported from a third-party speculative-decoding project: unlike
[05](./05-eagle3-head.md), which imports a separately-trained EAGLE-3 head, an MTP head is part of
the checkpoint its target came from. No new licence surface.

## Pre-registered kill-gates

Stated before measuring, in order.

**Gate 0 — at least two families on disk carry usable MTP heads.**
**PASSED 2026-08-27** on availability: three families (qwen35 line, qwen3_next, glm4moe). Recorded
with the caveat above — only glm4moe is admitted by the speculative seam today.

**Gate 1 — MTP α exceeds EAGLE-3's measured ~1.6 tok/verify on at least two of the three suites**
(code / math / chat).

> **"Same target" is not available, and the gate is a SCREEN rather than a paired comparison.**
> 05's head targets Qwen3-1.7B, which carries no MTP head (verified: `nextn` metadata absent, zero
> `mtp.*` tensors); no MTP-bearing checkpoint has an EAGLE-3 head. So the 1.6 is a reference point
> measured on a different target, and any Gate 1 result is cross-target by construction. That
> weakens it as a comparison and it is recorded as such — but it survives as a screen, because the
> claim being tested is directional: a head trained jointly with its own target, evaluated on that
> target, should beat an imported general head if joint training is the lever. If it does not, the
> lever is not joint training.
05 records **1.60 tok/verify**, token-identical to plain greedy (`TestEagleSpecParity`), and
accepted length 0.64 → ~1.64 at K=6. **Fails → stop.** The lever is α; if α does not move, nothing
downstream matters and no build is justified.

**Gate 2 — α clears break-even given the round cost**, using `breakEvenTokensPerRound`
(`decoder/blockspec.go:521`) rather than a new cost model. That comment records **~3.5 tok/round**
on the measured 4B / 2070S pairing (~39 ms per round against an 11.1 ms decode), and records that
the shipped guard sits at 2.5 — deliberately *below* break-even, because acceptance measured over
the first few rounds is not acceptance over the generation.
**The 3.5 cannot simply be moved to new silicon: the cost SHAPE differs, not just its constants.**
`breakEvenTokensPerRound` was derived for a *block* drafter, whose round is one draft forward plus
one batched verify plus the capture seam. An MTP round is a **per-token head forward — K of them,
each depending on the last — plus the capture/correction forward and the verify.** The draft term
scales with K where the block drafter's does not, so re-deriving means re-deriving the model:

    block drafter:  break-even accept  >  (c_draft + c_verify + c_capture) / c_decode
    MTP head:       break-even accept  >  (K·c_head + c_capture + c_verify) / c_decode

**Threshold, stated before measuring: α must clear the MTP form above, evaluated on the pairing the
probe actually uses, with `c_head`, `c_capture`, `c_verify` and `c_decode` measured on that pairing
rather than carried over. 3.5 is recorded here as the block-drafter figure it is — a reference
point, not the threshold.** Where K is small and the head is cheap relative to the target (a 424M
head against a 27B dense trunk), the MTP form can land well below 3.5; where the head is a 3.6 GB
GLM block, it can land above.

**What a passing Gate 1 on a REFUSED family obliges — decided now, before measuring.**

Nothing automatic. A pass on Qwen produces a recorded number and nothing else; it does not authorise
the state-checkpoint track, and no work begins on that track by momentum. This is written down
because a good result is exactly when a pre-registered gate stops being consulted.

What it *does* do is add a second reason to a decision that already exists on its own merits. The
hybrid state-checkpoint work is already a deferred item with an independent payoff:
`docs/qwen3_5_moe.md:114` records that cross-call prefix KV reuse falls back to **full recompute**
for these models today. MTP acceptance would be evidence joining that case, not creating it — and
the case gets decided on the combined value of prefix reuse plus speculation, by a person, not by
this page.

**The tension worth stating plainly, because it will decide whether this becomes a project.** The
economically most favourable candidate is on the blocked side of the seam. Gate 2's K-scaling term
cuts in favour of a small head against a large trunk — a 424M head on a 27B dense target has a far
lower per-token draft cost relative to decode than the 4B/2070S pairing that produced 3.5. That
pairing is `qwen3.8-27b`, which `specRollbackSafe` refuses. The one family that ships today, GLM,
carries the most expensive head on the page (3.6 GB, its own embedding and LM head), so it is the
least favourable on Gate 2 even if it is strong on Gate 1.

Anything past Gate 2 is a separate task and needs sign-off.

## Validation plan

**Ordered cheapest-first with an early exit, because three of the four outcomes do not justify the
expensive runs.**

1. **Gate 0 — done.** Inventory above.
2. **Loader adapter.** Read the head; assert tensor names, shapes and dtypes against the table
   above so a silently-empty head cannot pass as a loaded one.
3. **Gate 1 — offline acceptance, on `qwen3.5-0.8b` FIRST.** It is the cheapest checkpoint carrying
   a head (0.8B, 15 tensors) and it is the strongest form of the question: a head trained jointly
   with *that* model, evaluated on *that* model. If joint training does not beat an imported general
   head's 1.6 there, it is unlikely to elsewhere, and D2 closes without paying for a 27B or a 106B
   run. Point `TestEagleAcceptedLength`'s method at it on the suites 05 and
   [06](./06-acceptance-analysis.md) used, greedy, and report tok/verify per suite beside 05's 1.60
   — labelled cross-target, per the note on Gate 1.
   **Only if that clears the screen:** `qwen3.8-27b` (the favourable-economics candidate, blocked)
   and `GLM-4.5-Air` (the shippable one). Those are separate authorisations, not a continuation.

   > **HOW A 0.8B RESULT IS READ — fixed before it exists, because the modal outcome is ambiguous
   > by construction.**
   >
   > A 0.8B target is the cheapest step on this page, and this directory's own scorecard says a
   > model drafter pays only when the target step is expensive enough to dwarf the draft. So the
   > economics that would make MTP profitable on a 27B trunk are precisely what this target lacks.
   >
   > **A pass at 0.8B is a pass on MECHANISM — whether a jointly-trained head predicts its own
   > target better than an imported general head does. It is explicitly NOT a pass on economics,
   > and Gate 2 is NOT EVALUATED at this scale at all.** Any break-even computed here would be
   > dominated by a `c_decode` that does not resemble the regime the question is about.
   >
   > | 0.8B result | reading, fixed in advance |
   > |---|---|
   > | α ≤ 1.6 | Mechanism fails. **D2 closes with a number.** No further runs, no further authorisations. |
   > | α > 1.6 | Mechanism passes. Economics **unanswered, not favourable-by-implication.** Returns for a separate decision with the number in hand. |
   >
   > **No value of α at this scale authorises a larger run by itself.** A middling result — say 1.9,
   > over the screen and under any plausible break-even — is a mechanism pass and an economics
   > non-answer, and must be reported as both. Writing this down now is the guard against a number
   > that gets argued into whatever the reader already wanted.
   >
   > `c_head / c_decode` **is** worth recording at 0.8B as a diagnostic, and is **not transferable**:
   > it is a one-block head against a 24-block trunk with a tied embedding, and it will not resemble
   > a 424M head against a 27B dense trunk. Labelled as such wherever it appears, so it is never
   > later mistaken for a break-even.
4. **Gate 2 — break-even.** Compare against the threshold above.
5. **Losslessness, before any build.** Any scheme must be greedy bit-exact against non-speculative
   decode, as every scheme in this directory is. For the qwen35 line and qwen3_next that additionally
   requires the seam to admit them, which today it does not.

## Gate 1 result — PASSED on all three suites (2026-08-27)

Measured on `qwen3.5-0.8b` (safetensors, f32), greedy, RTX 2070S box, `TestMTP_acceptedLength`
in `decoder/mtp_head_test.go`. Method is `TestEagleAcceptedLength`'s, unchanged: prefill the head
over the realized prefix, draft K=6, count the leading run matching the target's own greedy
continuation, over M=48 continuation tokens → 42 draft positions per suite.

| suite | mean accepted | tok/verify | histogram (accepted 0..6) |
|---|---|---|---|
| code | 1.024 | **2.024** | `[23 3 9 6 1 0 0]` |
| math | 1.905 | **2.905** | `[11 6 10 9 4 1 1]` |
| chat | 1.476 | **2.476** | `[17 6 8 5 5 1 0]` |
| *05 EAGLE-3 reference* | *0.64* | *1.60* | *— (**CROSS-TARGET**, Qwen3-1.7B)* |

Gate 1 asked for two of three; all three clear. **Gate 1 PASSES.**

**Read exactly as the table above pre-registered it: α > 1.6 is a pass on MECHANISM — a
jointly-trained head predicts its own target better than an imported general head does. Economics
are UNANSWERED, not favourable-by-implication. Gate 2 was not evaluated and is not evaluable at
this scale. No larger run is authorised by this result; it returns for a separate decision with the
number in hand.**

**Precision of these numbers, stated so they are not over-read.** Each is one prompt, 42 positions,
no repeats and no variance estimate — the same shape as the 1.60 it is screened against, which is
also a single prompt. Two independent reasons to treat the third digit as noise:

- **Prompt form alone moved the code suite ~10%.** The first run used a bare completion prompt and
  got **2.238**; wrapping the identical text in the `<|im_start|>` chat template (matching 05's
  form, which is why it was changed) gave **2.024**. A formatting choice, not a model change.
- **n=1 prompt per suite.** The spread across suites (2.02 → 2.91) is larger than any of the gaps
  being read, so "math drafts better than code" is a plausible reading of this data and not a
  finding it establishes.

What the result does support is the direction, which is what the screen was for: every suite clears
the reference, the worst by 26% and the best by 82%, and the histograms show a real tail (drafts of
4–6 accepted tokens occur in all three) rather than a mean propped up by a few outliers.

**Cost diagnostic — NOT a break-even.** `c_head` 3.51 ms vs `c_decode` 137.2 ms, ratio **0.025**,
stable to ±0.001 across the three runs. Per the pre-registration this is recorded and labelled
**non-transferable**: it is a one-block head against a 24-block trunk with a tied 248320-row
embedding, and it will not resemble a 424M head against a 27B dense trunk. It is not an input to
Gate 2, which needs `c_capture` and `c_verify` measured on the pairing an actual build would use.

**What was built to get here**, kept to the minimum the probe needed: `decoder/mtp.go` — a
load-time read plus a single-block forward (`LoadMTPHead`, `MTPPrefill`, `MTPStep`,
`MTPDraftFrom`), reusing the target's own `qwen35Attention` and `gatedMLP`. `numLayers` and every
existing forward path are untouched, no serving/router/guard/P15 surface is touched, and nothing is
wired into the speculative seam — consistent with the seam analysis above, which holds that the
seam blocks deployment and not measurement.

**Two correctness checks the number depends on, both verified rather than assumed:**

- **The head is fed an UNNORMED hidden state, which is what it expects.** The head carries its own
  `pre_fc_norm_hidden`, so feeding it a post-`model.norm` tensor would double-norm and understate α.
  `captureResidual`'s stated contract is "the residual stream AFTER layer l is complete", so
  capturing at `NumLayers-1` yields the pre-final-norm residual. Correct as fed.
- **Drafting does not mutate the target's KV.** `mtpProject` borrows the trunk's output projection
  and touches only `cache.scr`, never keys or values — which is what lets the probe draft 42 times
  against a cache the target already filled. It does, however, return the target's **shared logits
  scratch**: an integration that interleaves drafting with target forwards would have its target
  logits silently overwritten. The probe is safe because it captures every feature up front and
  only then drafts. Noted in the code at the function.

**Two loader notes worth keeping**, both silent-wrong hazards rather than errors:

- **Head detection has two paths and this checkpoint needs the second.** `config.json` declares
  nothing (no `num_nextn_predict_layers`); the head is found only by the presence of `mtp.*`
  tensors in the safetensors index. A detector keyed on config alone reports "no MTP head" on a
  checkpoint that has one.
- **`q_proj` is `[4096, 1024]`, not `[2048, 1024]`.** Qwen3.5's output-gated attention emits
  `[query ‖ gate]` interleaved per head, so the head's projection is `2*qDim`. Asserting the
  un-gated shape rejects a valid head. The norms ship BF16 and need `TensorF32`, not
  `Tensor().Float32s()`.

## Risks / open questions

- **One checkpoint is a bad basis for a family-general claim, in either direction.** That is the
  whole reason the offline probe is specified across families rather than on the seam-admitted one.
  An earlier draft of this page argued that running Gate 1 only on GLM would bias the result
  *against* MTP because its head is the most expensive — **that reasoning was wrong, and it
  conflated the two gates.** Head size is a COST term: it enters Gate 2's break-even arithmetic and
  has no bearing on Gate 1, which measures acceptance quality. On that axis GLM is plausibly the
  *strongest* candidate, not the weakest — a head carrying its own output projection has more
  capacity to match the target than one borrowing the trunk's. The conclusion stands; the argument
  for it is simply that n=1 generalises to no family.
- **Break-even was derived on a 4B/2070S pairing.** A 27B dense target on a GPU is the regime the
  field report describes, and it is not the regime `breakEvenTokensPerRound` was measured in. The
  threshold must be re-derived if the probe pairing differs, which Gate 2 says explicitly.
- **Field data is directional only.** The six-day agentic report (Qwen3.8-27B, M2 Max, 8-bit,
  llama.cpp 10–11 → 17–19 tok/s with MTP drafting) is n=2 per condition, one machine, subjective
  composite scoring, no variance. It is a reason to measure and not a number to cite, and it is not
  carried into any table on this page.

## Pricing the narrow state snapshot (MEASURED 2026-08-28, `linux`)

**Measurement only. Nothing was built.** `specRollbackSafe` is untouched, no snapshot or restore is
wired into any decode path, and the harness that produced these numbers
(`decoder/deltanet_snapshot_cost_test.go`) is called from nothing else.

### What is being priced, and why it is not the deferred track

The refusal above traces to `decoder/deltanet.go:150` — `deltaState` holds a running recurrent
matrix plus a conv window, and the comment at `decoder/deltanet.go:147` records that it is fixed
size, independent of sequence length, and **not position-truncatable**. A verify advances that
state by K tokens; a partial rejection needs it as of an earlier token, which no truncation or
inversion recovers. `decoder/speculative.go:92` is where that refusal is applied.

`docs/qwen3_5_moe.md:117` defers a remedy — *"optimizing those for hybrid models (state
checkpoints) is a later track"* — but that entry was scoped for **cross-call prefix reuse**:
restore to an arbitrary earlier position, later, possibly across requests
(`docs/qwen3_5_moe.md:114`). Speculation needs something much weaker: snapshot immediately before
the verify, restore on rejection, discard. **One buffer, one round deep, lifetime of
milliseconds.** The two were bundled because they share a root cause, not because they are the
same size of problem.

**The narrow version is a strict subset, and a cheap result here does not authorise the broader
track.** Arbitrary-position restore needs state retained across calls and indexed by position;
none of that is priced below.

### Snapshot size, computed from config and checked against the live cache

`bytes = NumValueHeads·KeyHeadDim·ValueHeadDim` f32 for `s`, plus `(ConvKernel−1)` vectors of
`2·KeyHeadDim·NumKeyHeads + ValueHeadDim·NumValueHeads` f32 for `convWin`, per linear layer. The
harness asserts each of those against what the loaded model actually allocates, so the table is not
a paper calculation.

| model | layers | linear | full | layer split from | bytes/layer | total snapshot |
|---|---|---|---|---|---|---|
| qwen3.5-0.8b | 24 | 18 | 6 | `layer_types` | 1,122,304 | **19.3 MiB** |
| qwen3.6-35b-a3b | 40 | 30 | 10 | `layer_types` | 2,195,456 | **62.8 MiB** |
| qwen3next-80b | 48 | 36 | 12 | **computed, `full_attention_interval=4`** | 2,195,456 | **75.4 MiB** |
| qwen3.8-27b | 64 | 48 | 16 | `layer_types` | 3,268,608 | **149.6 MiB** |

**The 3:1 interleave was read per model, not assumed, and one model has no `layer_types` at all** —
`qwen3next-80b` declares only `full_attention_interval`, so its 36/12 split is synthesized. It does
land on 3:1, by computation.

**The snapshot does not scale with model size, and the ordering is not the one anybody would
guess.** Per-layer size varies **2.9×** across these four because `linear_num_value_heads` runs
16 / 32 / 32 / 48, and the totals invert against parameter count: the **27B needs 149.6 MiB, the
80B needs 75.4 MiB** — the smaller model carries twice the state. Snapshot cost is set by
`num_value_heads × linear layer count`, which is a config choice, not a size proxy. This is the
second reason the projections above carry no percentages: neither numerator nor denominator can be
extrapolated from a model's parameter count.

### Copy cost and the denominator — MEASURED, qwen3.5-0.8b only

**REGIME: qwen3.5-0.8b, CPU backend, single sequence, 30 paired rounds per run, 3 settled runs per
quant. These figures are NOT transferable to a 27B/35B/80B trunk.** Snapshot and decode are timed
in the same loop on the same cache, so machine drift moves both together instead of landing on one.

| regime | decode step (median of 3 runs) | snapshot+restore | % of one decode step | % of K=4 round | % of K=7 round |
|---|---|---|---|---|---|
| f32 weights | 135.58 – 135.76 ms | 2.881 – 2.911 ms | 2.122 – 2.146% | **0.531 – 0.537%** | **0.303 – 0.307%** |
| int8 weights, f32 activations | 74.92 – 75.24 ms | 2.946 – 2.990 ms | 3.933 – 3.974% | **0.983 – 0.994%** | **0.562 – 0.568%** |

Spread is across three runs, each started on a box settled below loadavg 0.30; ranges are given
rather than a point value.

**The copy is bandwidth-sensible, which is what stops it reading as an artifact.** Snapshot+restore
moves 40.4 MB and runs at **13.5 – 14.0 GB/s** in situ. The decode step is itself
bandwidth-bound at a comparable rate, so the time ratio tracks the byte-count ratio rather than
reflecting anything structural about the copy.

**Tight-loop control agrees with the in-situ figure.** A back-to-back copy loop with no decode
between rounds gives 14.3 – 14.6 GB/s, an in-situ/tight ratio of **1.04 – 1.07×** — the in-situ
number is marginally slower, the expected direction given the decode's cache pressure. There is no
disagreement to record here; had there been one, the in-situ figure would govern.

**Reuse the buffer.** Allocating a fresh snapshot each round costs 3.24 – 3.30 ms at f32 and
**5.21 – 6.23 ms at int8** — allocator behaviour is regime-sensitive and can more than double the
cost. The measured figures above are for one reused buffer, which is what the narrow scheme
implies.

### Projected copy cost for the models not timed

**Projected, not measured** — from the measured 13.9 GB/s, scaled by snapshot bytes.

| model | snapshot bytes | copy cost (PROJECTED) |
|---|---|---|
| qwen3.6-35b-a3b | 65,863,680 | 9.5 ms |
| qwen3next-80b | 79,036,416 | 11.4 ms |
| qwen3.8-27b | 156,893,184 | 22.6 ms |

**No percentage is given for these, and none should be inferred.** A ratio needs a decode step;
decode time scales with weights, not with snapshot bytes, so dividing one projection by another
would produce a figure with no measurement under it. Their bands are **unknown**.

### Bands

| regime (qwen3.5-0.8b, CPU) | K=4 | K=7 | band |
|---|---|---|---|
| f32 | 0.531 – 0.537% | 0.303 – 0.307% | **< 1% — cheap** |
| int8 | 0.983 – 0.994% | 0.562 – 0.568% | **< 1% — cheap, with almost no margin at K=4** |
| qwen3.6-35b-a3b, qwen3next-80b, qwen3.8-27b | — | — | **unknown, denominator not measured** |

**The int8 K=4 cell sits about 1% of the boundary value below the 1% line, and the direction is
systematic rather than noise.** The snapshot is fixed at 20.2 MB; only the denominator moved. Going
f32 → int8 shrank the decode step 1.80× and pushed K=4 from 0.53% to 0.99% — two regimes on one
model span nearly the whole cheap band. **A faster decode step is what makes this expensive**, and
this is the slowest backend available. Where a third regime lands is not measured and is not
projected here.

**A device-side copy has different economics and is unmeasured.** No CUDA or Metal work was done.
The CPU figures must not be read as a bound on a GPU-resident path in either direction.

### The snapshot must cover `convWin`, not just `s` — and at K≥4 the whole window is stale

This one is answerable from the code rather than by measurement, and the answer is not "probably".
`gatedDeltaNetStep` reads `convWin` as the depthwise conv's left context every step
(`decoder/deltanet.go:184`, taps `j = 0..K-2`) and mutates it every step, appending the current
mixed vector and sliding to the last `K-1`. A verify of width K advances that window by K tokens.

**With `ConvKernel = 4` the window is 3 vectors, so any verify of width K ≥ 4 replaces it
entirely.** A restore that recovers `s` and misses `convWin` does not leave a slightly stale
window — it leaves one in which *every* tap is a rejected token's mixed vector, feeding the conv of
the next accepted token. The logits would be wrong and nothing would report it.

`convWin` is **73,728 B of 1,122,304 B per layer — 6.6%**, which is exactly what makes it
skippable-looking. Every figure in this section already includes it; an implementation that drops
it to save 6.6% would be measuring something cheaper than what it needs.

### What this does not settle

**Cost is the easier half.** A restore has to reproduce the state **bit-exactly**, or the lossless
invariant every scheme in `docs/spec` is gated on breaks. A clone and write-back of plain
`[]float32` should be exact by construction — but that is an assumption, and it is untested here.
Nothing in this section measured correctness.

Also unpriced: what a rejection costs *beyond* the restore, whether one buffer suffices under
batching, and every model in the table above except the 0.8B.

### Measurement note

The first attempt at these numbers was discarded. Its settle loop timed out after 10 minutes and
measured anyway, at loadavg 1.75, while an unrelated benchmark held the box — a loop that gives up
and proceeds launders a contaminated run into a plausible-looking one. The loop now refuses. The
same conclusion was reached independently for `scripts/bench_peer.py` in `ebb1e3e`. The discarded
run gave 0.544% at K=4 against the clean 0.531 – 0.537%: close enough to have been believed, which
is the point.

## Step 1 — the resident CUDA cost, and why the CPU number was answering a different question

**MEASURED 2026-08-28, `linux`. Measurement only** — `specRollbackSafe` untouched, nothing wired
into a decode path. Harness: `cuda/deltanet_snapshot_cuda_test.go`.

### The three paths, all measured, all labelled

Same logical operation — snapshot 20.2 MiB of DeltaNet state and put it back — across two orders of
magnitude, decided entirely by which copy primitive is available.

| path | cost | vs one decode step | what kind of measurement |
|---|---|---|---|
| host memcpy, CPU decode | 2.9 ms | 2.1 – 4.0% | in situ, real state, real decode |
| **PCIe round trip, resident CUDA (today's only option)** | **8.1 ms** | **~100% ± 13 pp** | in situ, real state, real decode |
| DtoD, real 36-copy shape (needs a passthrough) | ~446 µs | ~5.5% | **through the PRIMITIVE, not an implemented snapshot** |

**The third row is not the same kind of number as the first two** and must not be quoted as an
observed integration cost. It was measured outside goinfer entirely — a standalone `gocudrv` program
copying synthetic buffers of the same sizes — because the wrapper offers no device-to-device copy
to measure through. It is the right number for the decision and the wrong number to cite as "the
snapshot costs 5.5%".

### The conclusion is about the primitive, not the approach

**"The narrow snapshot does not pay on resident CUDA" would be the wrong sentence to carry
forward.** What is measured is that it does not pay *through the copy primitive currently
available*. Those read identically today and diverge completely once a passthrough exists.

`aikit/gpu` exposes `Upload` and `Download` and no device-to-device copy, so a snapshot of state
that is *already on the device* (`cuda/resident.go:248` — `dnWin`, `dnState`) has to cross PCIe to
the host and come back: measured **5.0 GB/s**, about a third of the host memcpy rate. The primitive
exists one layer down — `gocudrv`'s `memcpyDtoD` / `memcpyDtoDAsync`. The gap is plumbing, and the
plumbing is worth ~18×.

### The paired ratio, and why the ratio of medians could not be corrected for

Six settled runs, ratio formed **per round** rather than as median(cost)/median(decode):

| quant | min | p50 | max | mean | sd |
|---|---|---|---|---|---|
| int4 | 76.8% | 100.4% | 134.9% | 102.3% | 13.5 pp |
| int4 | 74.9% | 100.5% | 166.3% | 106.5% | 24.5 pp |
| int4 | 83.9% | 107.9% | 132.8% | 107.1% | 12.1 pp |
| int8 | 82.0% | 98.6% | 158.2% | 100.7% | 14.5 pp |
| int8 | 71.1% | 99.4% | 124.5% | 98.9% | 13.8 pp |
| int8 | 73.7% | 100.7% | 147.5% | 103.4% | 16.1 pp |

**≈100% ± 13 pp.** The components are noisier than the ratio — the decode step alone spans 2.0×
within a run while the ratio spans ~1.6× — so pairing recovers real precision rather than
relabelling it.

**The argument for pairing is not that the ratio of medians is biased.** A fixed bias could be
corrected. Here it disagreed with the paired form by **7.6 pp in one run and under 1 pp in others** —
unpredictably, run to run. That is what makes it uncorrectable, and it is the reason to form the
ratio per round rather than a preference about statistics.

### The 0.8B resident denominator is not a proxy for a 27B resident denominator

**int4 and int8 land on top of each other** — medians interleave across the six runs with no
separation by quant, and the decode step is 8.06 ms against 8.01 ms. So resident decode at 0.8B on
this card **is not weight-bandwidth-bound**: the lever that spanned most of the cheap band on CPU
(f32 → int8, 1.80×) does nothing here.

That has a consequence beyond this table. The denominator measured here is set by something other
than weight traffic — dispatch, launch geometry, the DeltaNet chain's serial structure — and it will
not scale the way the CPU figures would suggest. A 27B resident trunk is plausibly weight-bound
where this is not, so **neither the ratio nor its direction transfers**. Do not extrapolate either
way; measure it there.

### An impossible number, caught by a physical bound

**This is the most transferable thing in this section.** The first device-to-device reading was
4.96 µs — **8145 GB/s** on a card whose VRAM peak is 448 GB/s. Not a suspicious result: an
impossible one.

The cause was a call named `CopyToDevice`, dispatching the **synchronous** `memcpyDtoD` rather than
the Async form, that nonetheless returns before the transfer completes. Timed without a synchronize
it measures dispatch (~9 µs) instead of transfer (~116 µs). Every part of that combination points
the wrong way, and the resulting number is entirely plausible if you have no bound to check it
against.

**Where a physical ceiling exists, checking against it should be routine rather than incidental.**
Almost every other instrument failure this week produced numbers that were wrong but plausible —
a settle loop measuring at loadavg 1.75 and giving 0.544% against a clean 0.531%; a sampler
microbenchmark inverting a sign; a harness counting chunks instead of tokens. None of those had a
hardware bound available to violate. This one did, and impossibility is a far sharper instrument
than suspicion. The probe now **derives** the ceiling from the device
(`MemoryClockRate × BusWidth × 2` → 448 GB/s, matching spec) and asserts against it, so the check
travels to other cards instead of carrying a stale constant. Its dispatch-only figure stays in the
output beside the real one, so the trap is self-documenting for whoever runs it next.

### The composition costs 2× the primitive

One contiguous 20.2 MiB DtoD runs at **347 GB/s (78% of peak)**. The real shape — 18 layers ×
(`dnWin` + `dnState`) = 36 separate copies — runs at **174 GB/s (39% of peak)**, taking 223 µs
against 116 µs. The 18 `dnWin` copies are 73 KB each against ~9 µs of dispatch and are
overhead-dominated; the 18 `dnState` copies at 1 MiB are not.

A single contiguous copy would have reported 347 GB/s and been believed. **Isolation proves the
primitive, never the composition** — the same lesson the hysteresis test produced the same day, in
a different subsystem, from the opposite direction.

**Design note, not a chase:** if the conv windows were contiguous, or issued as one copy, most of
the 2× penalty disappears. That moves ~446 µs toward ~250 µs. Recorded in `docs/qwen3_5_moe.md`
beside the buffer-reuse and `convWin` notes.

### Where this leaves the decision

Nothing to build yet, and the blocking item is not in goinfer. A snapshot implemented today costs
about a token per round, and no acceptance rate rescues that. After a passthrough it is ~5.5% of a
decode step on this trunk — the **1–5% band** under the K-step framing, which returns alongside α
rather than being settled here. And since a batched verify on a bandwidth-bound decode costs nearer
one step than K, the K-step framing is the favourable reading, not the neutral one.

## Step 1, concluded — measured through the real primitive (aikit/gpu v0.31.0, 2026-08-28)

The passthrough exists. `aikit/gpu` v0.31.0 adds `CopyDevice`, `CopyDeviceBatch` (one synchronize
for many copies) and a coalescing pass; goinfer's `cuda` module is bumped to it. **This replaces the
synthetic-buffer projection above with an in-situ measurement, and the projection was optimistic.**

### The projection was wrong by 2.6× on bandwidth, and that is the point of measuring in situ

| | synthetic probe | in situ |
|---|---|---|
| snapshot+restore | ~446 µs | **623 µs** |
| effective traffic, identical 36-copy shape | 174 GB/s | **65 GB/s** |

Same call, same byte counts, same number of copies. The difference is **buffer layout**: the probe
allocated its 36 buffers consecutively, so aikit's coalescing had adjacent pairs to merge. The real
`dnWin`/`dnState` live at bind offsets inside the resident arena (`cuda/resident.go:241` calls them
COMPOUND) interleaved with everything else the model allocated, so there is far less to coalesce.
A figure measured through a primitive on synthetic buffers is not an integration cost — here the
gap was 2.6×, not a rounding difference.

**Read that as a caution about synthetic buffer probes generally, not a fact about this snapshot.**
Anything measured on purpose-allocated buffers inherits a layout the real workload will not have:
consecutive allocation is the best case for coalescing, adjacency and locality alike, and a real
allocator's output is not. The probe was still the right thing to run — it converted "plausibly
tenths of a millisecond" into a number and decided whether the passthrough was worth requesting —
but its output is a bound on what the primitive can do, never a prediction of what a caller gets.

### Results, three paired runs per path, alternated in one session

| path | snapshot+restore | traffic | paired ratio p50 (spread) |
|---|---|---|---|
| device-to-device (`CopyDeviceBatch`) | **623 µs** | 65 GB/s | **11.0%** (10.8 / 11.3 / 10.8; sd 2.0–2.9 pp) |
| PCIe round trip (pre-v0.31.0 control) | **8080 µs** | 5.0 GB/s | 103.2% (97.6 / 108.2 / 103.7; sd 13.5–21.6 pp) |

**Quote the cost ratio, 13.0×, not the ratio-of-ratios.** The two paths do not share a denominator:
the resident decode step measures **5.69 ms in the D2D runs and 7.92 ms in the PCIe runs**, with no
overlap across three runs each — **the snapshot path perturbs the decode step it is being measured
against**, by 1.39×. Plausibly L2 eviction (40 MB of traffic against a 4 MB L2) plus the
synchronization the PCIe path forces per buffer, but the mechanism is not established here and the
number does not depend on it.

Two consequences. The PCIe ratio is **flattered** by its own contamination: against the clean
denominator it is **142% of a decode step**, not 103%. And the honest statement of what the
passthrough bought is the **cost** ratio — 8080 µs → 623 µs, **13.0×** — which is denominator-free.
The earlier "~18×" from the probe overstated it, for the same buffer-layout reason as above.

**This is the same class as the async-copy trap, one level up.** There the instrument perturbed its
own reading; here the *treatment* perturbs the *control variable*. A ratio is only safe when its
denominator is independent of the arm.

### Verdict: the narrow snapshot does NOT pay on its own, on the evidence available here

**11.0% of a resident decode step.** The pre-registered rule reads **">5% — the narrow version does
not pay on its own. Report and stop."** This clears that bar only under the K-step framing (2.75% at
K=4, 1.57% at K=7), and that framing was already recorded above as the **favourable** reading rather
than the neutral one: a batched verify on a bandwidth-bound decode costs nearer *one* decode step
than K, which puts the figure at **5.5–11%**. Applying the rule as written, to the more realistic of
the two readings: **it does not pay.**

Returning it alongside α is procedurally right and is not the same as it being open. **The α needed
to overcome an 11% standing per-round cost is high, and nothing measured points that way** —
Gate 1's acceptance is 2.02–2.91 tokens per verify on one prompt per suite at n=1, with the third
digit already established as noise.

**The one thing that could rescue it is the one thing this box cannot measure.** int4 and int8 land
on top of each other here, so a 0.8B resident decode is *not* weight-bandwidth-bound. A 27B resident
trunk plausibly is, which would make its decode step slower, the denominator larger, and the
fraction smaller. That is the rescue — and it requires a resident 27B or 80B, neither of which fits
an 8 GB card. **So the remaining case for this track rests entirely on a regime for which no
hardware here can produce a number**, and that is a cleaner place to stop than "pending α".

**What is not conditional:** the aikit passthrough was worth writing regardless. 13.0× on a
device-to-device copy is real, it is in `aikit/gpu` v0.31.0, and it is available to anything that
needs to move bytes between device buffers — independent of whether this track ever resumes.

**If it resumes**, the entry condition is a resident measurement on a 27B-class trunk, not a better
α on a small one. Everything else — the sizes, the `convWin` requirement, the buffer-reuse and
coalescing notes — is recorded and does not need re-deriving.
