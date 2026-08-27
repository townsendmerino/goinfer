# Scoping: Qwen3.8-Flash-Next (Qwen4 architecture preview)

> **Status:** scoping only — nothing here is committed work. Drafted 2026-08-27 following a
> Francis ↔ fable conversation about [the Qwen blog post](https://qwen.ai/blog?id=qwen3.8-flash-next);
> fable's read is the starting hypothesis, and this pass re-verified the load-bearing claims
> directly against the [HF model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), the
> [GitHub repo](https://github.com/QwenLM/Qwen3.8-Flash-Next), and the LICENSE file rather than
> carrying them forward — per the claim-discipline rule in `docs/parity-coverage-policy.md`
> ("a claim that arrives with its own corroborating detail has not been corroborated"). Two
> things fable said did **not** reconcile on re-check; flagged in place below, not smoothed over.

**Verdict:** the specific checkpoint is out of reach on both rigs and isn't worth chasing. The
architecture is a different question — three of its five new-or-scaled pieces ride substrate
goinfer already has parity-gated on real checkpoints, and Alibaba is explicit that this release
previews Qwen4, the generation that will actually ship at goinfer's target sizes. The move is a
synthetic-tiny bring-up of the genuinely new pieces now, so day one of a real Qwen4 small/mid
checkpoint is a config-mapping exercise instead of a from-scratch bring-up — **not** a build
today, and not gated on this doc alone (see "What's actually needed before this starts").

## What shipped, confirmed 2026-08-27

| | value | source |
|---|---|---|
| License | `qwen-community-1.0` | HF model card + [LICENSE file](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) |
| Total / active | 180B on disk = 125B MoE backbone (6B active) + 51B n-gram embedding + 4B MTP | HF model card |
| Hidden dim / token embed | 2560 / 248,320 (padded) | HF model card |
| Layers | 48 = 12 × (3 × (GDN→MoE) → 1 × (QSA→MoE)) | HF model card |
| Gated DeltaNet (GDN) | 48 V heads / 16 QK heads, head_dim 128 | HF model card |
| Qwen Sparse Attention (QSA) | 24 Q heads / 2 KV heads, head_dim 256, budget 512 blocks / 2048 tokens | HF model card |
| MoE | 512 experts, 10 routed + 1 shared active, expert intermediate 640 | HF model card |
| N-gram embedding | 20,000,000 entries (bigram/trigram), injected at layer 2 | HF model card |
| MTP head | 1 layer, "trained with multi-steps" | HF model card |
| Context | 262,144 native, extensible to 1,000,000 | HF model card |
| Gated Residual (GR) | "widens the residual stream into 4 branches and controls reads and writes with a dynamic gate" | [GitHub README](https://github.com/QwenLM/Qwen3.8-Flash-Next) |

**Two things that do not reconcile, left open rather than picked:**

- **Checkpoint dtype/size.** fable cited an FP8 checkpoint at 172.8 GiB. The HF model card's own
  `Tensor type` field — normally auto-derived from the safetensors headers, not a hand-set label —
  reads **BF16**. 180B params at BF16 is ~335–360 GiB; at FP8 it's ~172–180 GiB, so this is a ~2×
  discrepancy, not rounding. Neither number is trusted here; pin it from the actual file listing
  (sizes + `config.json` `torch_dtype`) on the HF repo before it feeds any download or hardware
  estimate.
- **Gated Residual's rank.** fable described a "rank-320 bottleneck." The GitHub README confirms
  the 4-branch structure and the dynamic read/write gates verbatim but states no rank or bottleneck
  dimension. The likely primary source is `tech_report.pdf` in the same repo (2.26 MB) — WebFetch
  can't extract a binary PDF, so this is unread. Don't build against "rank-320" until someone has
  actually read the tech report.

## Why not this checkpoint

The 62 GB Linux box already has a direct precedent at almost this scale: **Qwen3-Next-80B-A3B**
(163 GB bf16) can't get a full reference forward there at all — G5 (`docs/queue-correctness.md:188-193`)
only cleared a 4-layer slice oracle (cosine 1.00000000, argmax + greedy exact) for exactly that
reason, and stays `experimental` because "no full reference forward of a 163GB bf16 model fits
62GB." Qwen3.8-Flash-Next is 180B — bigger than the model that already couldn't be fully
referenced on that box. At int4 (goinfer's usual deployment quant) it lands around 90–110 GB,
which doesn't fit the 16 GB Mac in any paged regime and is at-or-past what made the *smaller*
model's *reference* forward infeasible on the 62 GB box — here it's the model itself, quantized,
not just the HF comparison run. Bringing up this specific checkpoint would be a bring-up with no
runnable payoff on either machine.

## What it actually previews

Alibaba's own framing (both the blog and the third-party coverage) is that this release previews
the Qwen4 architecture ahead of the Qwen4 line itself — GDN/QSA/GR/n-gram-embed/MTP is stated as
the shape future Qwen4 checkpoints will share, including whatever sizes land in goinfer's actual
niche (single-user, consumer hardware, safetensors/GGUF). That reframes the question from "can we
run this 180B" (no) to "when Qwen4 ships at a size that fits, how much of this is already done."

## What goinfer already has

**Gated DeltaNet — shipped and parity-gated, real checkpoints, bit-exact.** `qwen35Architecture` /
`qwen35DenseArchitecture` / `qwen3NextArchitecture` (`decoder/registry.go:44-47,728,787,821,1883`)
already implement the same 3-linear-attention : 1-full-attention hybrid, with the MoE and dense
variants both live. Real-checkpoint slice parity is bit-exact on **two** real checkpoints:
`Qwen3.6-35B-A3B` (recorded in the completed qwen3.6 real-checkpoint task record — internal and
untracked, so named in prose rather than cited as a path — cosine 1.0000000000 across
embed + 4 layers, DeltaNet recurrence + 256-expert fused-stacked MoE + softmax layer all inside
the compare) and `Qwen3-Next-80B-A3B` (G5 above, cosine 1.00000000). Qwen3.8-Flash-Next's GDN
block (48V/16QK heads, head_dim 128) is a config-mapping check against this primitive, not a new
one — 36 of its 48 layers ride it.

**MoE + expert demand-paging — shipped, validated on real checkpoints under memory pressure.**
`decoder/moepaging.go` (`expertPager`) + `decoder/layerpaging.go` (`layerPager`) are bit-exact
against fully-resident (`TestExpertPaging_bitExact`) and validated on `Qwen3.6-35B-A3B` (512 MB
cache against ~16 GB of experts: hits=4706, misses=5534, evictions=5190, byte-identical decode
over 24 tokens) and on Gemma-4 26B-A4B (32/128 experts resident, 77.5% hit rate) —
`docs/task-moe-streaming.md`. Current families top out at 256 experts (`qwen3_5_moe`); scaling to
512 experts at a 10-routed+1-shared split is an extension of `moeMLP`'s existing routed+shared+gate
machinery, not a new mechanism — though the 10:1 ratio needs the same kind of config-flag gotcha
check that `qwen3_5_moe`'s bring-up hit with `NormTopKProb` (`docs/qwen3_5_moe.md`: silently wrong
without it, cosine 0.9985 vs 1.0).

**The native MTP head lands on a question goinfer's own spec-decode campaign already answered
twice, negatively.** `docs/spec/05-eagle3-head.md` (EAGLE-3, a feature-level draft head trained
*separately* over the target's hidden states) is built end-to-end and lossless but a CPU
wall-clock **loss** — low acceptance means the extra draft-forward-plus-correction cost isn't
recovered (`docs/spec/README.md`). `docs/spec/07-stageb-gemm-verify.md` (GPU-batched verify) is
"NO-GO small models; parked conditional-GO for ~70B + short drafts (needs >8 GB GPU)" — the
amortization only shows up on hardware goinfer's niche doesn't target. Both results point the same
direction: a drafter that has to be trained and hosted separately from the target either doesn't
pay for itself on CPU or needs GPU scale goinfer isn't built for. `docs/spec/08-dspark-dflash.md`
(queued `P10`/`P15`) is goinfer's existing bet on the fix — pretrained *block* drafters trained
specifically against a target family. A native, ship-with-the-checkpoint MTP head is the same bet
at zero integration/hosting cost, *if* a model using one is ever brought up: the `Drafter`/
`Verifier` interfaces (`docs/spec/00-core.md`) are already architecture-agnostic, so wiring one in
is an adapter, not new verify/rollback machinery. Worth naming as a side benefit — not a reason to
prioritize this model on its own.

## What's genuinely new

1. **QSA (Qwen Sparse Attention) — a real bring-up.** 24Q/2KV heads, head_dim 256, a learned
   indexer selecting 512 blocks (2048 tokens) of context per step. Nothing in goinfer does learned
   block-sparse selection over the target model's own attention today — the two things that sound
   adjacent, the grammar DFA mask (`constrain`) and the n-gram speculative-decoding drafter
   (`decoder/spec_ngram.go`), both operate on *draft* tokens, not on which context the target
   attends to. The indexer's own architecture and training (jointly trained? auxiliary loss?) are
   unread — in `tech_report.pdf`, not the README.

2. **Gated Residual — plumbing, not a layer.** Confirmed real (see table above): 4 parallel
   residual branches, one dynamic read gate, per-branch write gates. `decoder/forwardn.go` is in
   the explicit **core** hashed-file set (`docs/task-parity-coverage.md:33`, alongside
   `model.go`/`attention.go`/`mlp.go`/`kvcache.go`/`rope.go`/`rmsnorm.go`/`registry.go`/`arch.go`/
   `config.go`) — changing it re-stales every family's parity record, the same "maximum blast
   radius" `docs/scoping-lfm2.md` §G flagged for touching `registry.go`/`kvcache.go`. Concretely,
   `forwardn.go` today assumes one residual stream through the attention+MLP fold (its own comments
   describe "one residual add" points in the batched path); widening that to 4 gated branches
   touches the shared spine every family's forward pass runs through, not a new `forward_X.go`.

3. **N-gram embedding table — new, and namespace-collides with something unrelated.** 20M
   bigram/trigram entries, looked up and injected at layer 2, accounting for 51B of the 180B total.
   This is a parametric memory component with no existing goinfer analog. **It has nothing to do
   with `decoder/spec_ngram.go`** (the suffix-automaton speculative-decoding drafter,
   `docs/spec/02-cache-ngram.md`, goinfer's shipped CPU speculative-decoding win) beyond sharing the
   word "n-gram" — flagging this now so a future pickup doesn't conflate the two.

4. **512-expert, 10-routed+1-shared MoE — the cheap one.** An extension of the substrate in
   "What goinfer already has" above, not a new mechanism.

## License — gate zero, per the P15 precedent

The house pattern for a model with a licensing question is `docs/prompts/dspark-license-issue.md`'s:
step (0) is a licence read, done before anything else, re-run at each pickup rather than trusted
from the first pass (P15 re-ran it 2026-08-21 and got the same answer twice). This model's case is
easier in one way — there's a named, readable license rather than an absent one — but the same
discipline applies: read it firsthand at pickup, don't inherit this summary.

**What this pass found, reading the LICENSE file directly:** `qwen-community-1.0` grants broad use
— "use, copy, modify, merge, publish, distribute, sublicense, sell, deploy, host, fine-tune, and
create derivative works." Two restrictions, both keyed to scale goinfer (a single-user local
inference engine) doesn't operate at: products/services over 100M MAU or $20M/month revenue must
prominently display the model name in their UI, and "Model as a Service" / "AI Work Assistant"
businesses need a separate commercial license from Qwen. No clause found restricting use of outputs
to train other models. Redistributing an inference engine that loads these weights (rather than
redistributing the weights themselves) isn't explicitly addressed but isn't what either restriction
targets. **Read plainly, this is the same shape as other MAU-threshold community licenses goinfer
already navigates for other families — not a blocker for the inference-engine use case** — but this
is a summary of one read, not the audit itself; re-read the LICENSE text directly before writing
any code that ships against this family, per the P15 pattern.

## What's actually needed before this starts

Beyond the license re-read, two things this pass could **not** confirm, and one of them may matter
more than the license:

- **Is there a runnable HF reference yet?** Every hybrid bring-up in this repo so far — `qwen3_5_moe`
  pinned against `transformers==5.10.2`'s `modeling_qwen3_5_moe.py` (`docs/qwen3_5_moe.md`) — has
  had a working `transformers` forward pass (native or `trust_remote_code`) to pin a golden against.
  No `modeling_*.py` was found linked from the GitHub repo (README + `tech_report.pdf` only) at
  this check. If Alibaba's reference code hasn't landed anywhere loadable, there's no oracle to
  validate a synthetic-tiny fixture against yet — check this before scoping the bring-up further,
  it may be the actual gate, license aside.
- **Read `tech_report.pdf` firsthand** for the Gated Residual's bottleneck dimension and the QSA
  indexer's architecture/training — both unread in this pass (binary PDF, not text-fetchable here).
- **Confirm the checkpoint dtype** (BF16 vs FP8) from the HF repo's actual file listing — it changes
  download size ~2× and, if it's genuinely FP8, raises the same blocker already on record for
  DeepSeek V4-Flash: **"there is no fp8 support anywhere in the tree today"**
  (`docs/post-v1.0-models.md`). Worth checking early since it could gate the whole effort
  independently of QSA/GR/n-gram-embed.

## Effort, if picked up (synthetic-tiny bring-up only — not the 180B)

| Component | Size | Notes |
|---|---|---|
| QSA (new mixer + indexer) | **L** | genuine new bring-up, no existing primitive; indexer training/inference unread |
| Gated Residual (4-branch) | **M–L** | the risk is blast radius (core `forwardn.go`, re-stales all families), not line count |
| N-gram embedding table | **M** | new but isolated; no interaction with existing forward paths beyond the lookup-and-add |
| 512-expert / 10+1 MoE | **S** | extension of shipped `moeMLP` + `expertPager` substrate |
| Tiny-synthetic T1 golden + pin script | **S**, *contingent* | mirrors `scripts/pin_qwen3next_tiny.py`; blocked on an HF reference existing |

## Sequencing

**Not started; nothing is queued.** This is prep, filed the way `docs/scoping-lfm2.md` files a
post-freeze family: visible, reasoned, not claimed. Pick up when either holds:

1. A real Qwen4 checkpoint ships at a size that actually fits goinfer's niche (the stated target of
   this preview) — at that point this doc plus whatever the synthetic-tiny bring-up already
   validated should make it a config-mapping exercise on the GDN/MoE-paging side, with QSA/GR/
   n-gram-embed as the remaining real work either way.
2. Someone decides the architecture bet is worth making ahead of a real Qwen4 release, accepting
   synthetic-tiny-only validation (no real-checkpoint T3) until one exists.

Either way, gate zero is unchanged: re-read the LICENSE, confirm an HF reference exists, read
`tech_report.pdf`, resolve the dtype discrepancy — in that order, before any code.

**Cross-reference:** add a line under `docs/post-v1.0-models.md`'s "Watching" section pointing here,
so this doesn't sit as an orphaned scoping doc nobody's roadmap points to.

## Sources checked this pass

- [Qwen blog post](https://qwen.ai/blog?id=qwen3.8-flash-next) (metadata only — no architecture detail fetchable)
- [Hugging Face model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Qwen3.8-Flash-Next LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
- [GitHub repo README](https://github.com/QwenLM/Qwen3.8-Flash-Next) (`tech_report.pdf` in the same repo is NOT read — binary, not text-fetchable here)
- [MarkTechPost coverage](https://www.marktechpost.com/2026/08/26/alibabas-qwen-team-releases-qwen3-8-flash-next-a-125b-multimodal-moe-with-6b-active-parameters-previewing-the-qwen4-architecture/)
- goinfer tree, read directly 2026-08-27: `decoder/registry.go`, `decoder/config.go`,
  `decoder/forwardn.go`, `decoder/moepaging.go`, `docs/qwen3_5_moe.md`,
  the completed qwen3.6 real-checkpoint task record (internal, untracked), `docs/queue-correctness.md`,
  `docs/task-moe-streaming.md`, `docs/spec/README.md`, `docs/task-parity-coverage.md`,
  `docs/parity-coverage-policy.md`, `docs/prompts/dspark-license-issue.md`,
  `docs/scoping-lfm2.md`, `docs/post-v1.0-models.md`
