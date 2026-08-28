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
| `decoder/gguf.go:631` | same subtraction, with the comment "block_count includes the trailing NextN/MTP block(s) goinfer drops" |
| `decoder/weights.go:489` | "MTP heads (`mtp.*`) are simply never requested" |
| `decoder/registry.go:1027` | `num_nextn_predict_layers` MTP head is dropped |

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

`decoder/forwardn.go:60`:

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
| `ForwardCapture` (`decoder/model.go:709`) — 08's capture seam | granite, nemotron, mla, llama4 — **not qwen35** |
| `specRollbackSafe` (`decoder/forwardn.go:60`) | granite, nemotron, **qwen35**, `SlidingWindow > 0` |

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
(`decoder/blockspec.go:399`) rather than a new cost model. That comment records **~3.5 tok/round**
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
