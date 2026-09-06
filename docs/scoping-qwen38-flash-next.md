# Scoping: Qwen3.8-Flash-Next (Qwen4 architecture preview)

> **Status:** scoping only — nothing here is committed work. Drafted 2026-08-27 following a
> Francis ↔ fable conversation about [the Qwen blog post](https://qwen.ai/blog?id=qwen3.8-flash-next);
> fable's read was the starting hypothesis, re-verified directly against the
> [HF model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), the
> [GitHub repo](https://github.com/QwenLM/Qwen3.8-Flash-Next), the LICENSE file, and — as of the
> same day, once Francis pulled it into the chat — the architecture's own `tech_report.pdf`
> ("On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability",
> Qwen Team, 2026-08-26) — rather than carried forward, per the claim-discipline rule in
> `docs/parity-coverage-policy.md` ("a claim that arrives with its own corroborating detail has
> not been corroborated"). One thing fable said did **not** reconcile even after the primary
> source; flagged in place below, not smoothed over.

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
| Total / active | 180B on disk = 125B MoE backbone (6B active) + 51B n-gram embedding + 4B MTP | HF model card; params table confirmed in `tech_report.pdf` Tab. 11 (125B / 6B active / 51B n-gram) |
| Hidden dim / token embed | 2560 / 248,320 (padded) | HF model card |
| Layers | 48 = 12 × (3 × (GDN→MoE) → 1 × (QSA→MoE)) | HF model card; `tech_report.pdf` Fig. 1 |
| Gated DeltaNet (GDN) | 48 V heads / 16 QK heads, head_dim 128; gated delta rule, zero-centered-RMSNorm + sigmoid output gate | HF model card; `tech_report.pdf` §2.1.1, eq. 1–11 |
| Qwen Sparse Attention (QSA) | 24 Q heads / 2 KV heads, head_dim 256; **indexer** is separate — MQA, 4 query heads + 1 shared key head, compression ratio r=4, partial RoPE (64/128 dims), block budget K=2048 tokens ⇒ 512 blocks | HF model card (mixer heads); `tech_report.pdf` §2.1.2, eq. 12–19 (indexer, now fully spec'd) |
| MoE | 512 experts, 10 routed + 1 shared active, expert intermediate dim 640 | HF model card |
| N-gram embedding | 20,000,000 entries (bigram/trigram), placed at layer 2 only, "augments the corresponding token representation" | HF model card; `tech_report.pdf` §2.3, §2.3.1 |
| Gated Residual (GR) | 4 branches, per-branch RMSNorm read + **rank-320 (=d/8) low-rank elementwise read gate** + per-branch scalar write gate; no branch-mixing operator; one GR module for the attention sublayer and a separate one for the MLP sublayer, every layer | `tech_report.pdf` §2.2, eq. 21–34 — **now fully confirmed**, including the rank fable cited |
| MTP head | 1 layer, trained with multi-steps; reuses QSA's top-k block indices across speculative steps | HF model card; `tech_report.pdf` §2.1.2 ("Implementation Details") |
| Context | 262,144 native, extensible to 1,000,000 | HF model card |

**The one thing that still doesn't reconcile: checkpoint dtype/size.** fable cited an FP8
checkpoint at 172.8 GiB. The HF model card's own `Tensor type` field reads **BF16** (180B params
≈ 335–360 GiB at BF16 vs ≈ 172–180 GiB at FP8 — a ~2× gap, not rounding). `tech_report.pdf` doesn't
settle this either: its only FP8 mentions are an *internal* optimization — storing the widened
**residual activations** in FP8 to halve memory traffic on the GR read/write (§2.2, "Inference
Efficiency") — which is unrelated to what dtype the *released weights* ship in. Don't infer one
from the other. Pin the actual dtype from the HF repo's file listing / `config.json` `torch_dtype`
before it feeds any download or hardware estimate.

## Why not this checkpoint

The 62 GB Linux box already has a direct precedent at almost this scale: **Qwen3-Next-80B-A3B**
(163 GB bf16) can't get a full reference forward there at all — G5 (`docs/completed/queue-correctness.md`, archived 2026-08-31)
only cleared a 4-layer slice oracle (cosine 1.00000000, argmax + greedy exact) for exactly that
reason, and stays `experimental` because "no full reference forward of a 163GB bf16 model fits
62GB." Qwen3.8-Flash-Next is 180B — bigger than the model that already couldn't be fully
referenced on that box. At int4 (goinfer's usual deployment quant) it lands around 90–110 GB,
which doesn't fit the 16 GB Mac in any paged regime and is at-or-past what made the *smaller*
model's *reference* forward infeasible on the 62 GB box — here it's the model itself, quantized,
not just the HF comparison run. Bringing up this specific checkpoint would be a bring-up with no
runnable payoff on either machine.

## What it actually previews

Alibaba's own framing is explicit, and it's a training-efficiency argument, not just an
architecture-shape one: `tech_report.pdf` Tab. 11 shows Qwen3.8-Flash-Next-Base beating its own
27B dense sibling on all 14 evaluated benchmarks and beating (or matching) the much larger
397B-A17B Qwen3.7-Plus-Base on 8 of 14 — at roughly a third of the activated parameters, a third
of the training tokens, and a ninth of the training FLOPs. That efficiency claim is *why* this
shape is the one worth being ready for: it's not a one-off flagship experiment, it's presented as
the cheaper way to train the next generation, including whatever sizes land in goinfer's actual
niche (single-user, consumer hardware, safetensors/GGUF).

**Possibly-relevant, not confirmed:** Tab. 11's comparison baseline is named `Qwen3.8-27B-Base` —
a plausible match for the dense hybrid goinfer's `qwen35DenseArchitecture` already implements and
calls "Qwen3.8" in its own code comment (`decoder/registry.go:2491`). If they're the same family,
goinfer already runs a sibling from this exact lineage today, not merely something
architecturally adjacent. Worth a five-minute config check at pickup — not asserted here.

## What goinfer already has

**Gated DeltaNet — shipped and parity-gated, real checkpoints, bit-exact.** `qwen35Architecture` /
`qwen35DenseArchitecture` / `qwen3NextArchitecture` (`decoder/registry.go:49-47,728,787,821,1883`)
already implement the same 3-linear-attention : 1-full-attention hybrid, with the MoE and dense
variants both live. Real-checkpoint slice parity is bit-exact on **two** real checkpoints:
`Qwen3.6-35B-A3B` (recorded in the completed qwen3.6 real-checkpoint task record — internal and
untracked, so named in prose rather than cited as a path — cosine 1.0000000000 across
embed + 4 layers, DeltaNet recurrence + 256-expert fused-stacked MoE + softmax layer all inside
the compare) and `Qwen3-Next-80B-A3B` (G5, cosine 1.00000000). Qwen3.8-Flash-Next's GDN block
(48V/16QK heads, head_dim 128, zero-centered RMSNorm + sigmoid output gate per `tech_report.pdf`
eq. 11 — a bounded-sigmoid gate rather than the plain SiLU gate of the original GDN paper, "we
observe consistent improvements" per the report) is a config-mapping check against this primitive,
not a new one — 36 of its 48 layers ride it.

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
twice, negatively — and Alibaba's own numbers now quantify the gap.** `docs/spec/05-eagle3-head.md`
(EAGLE-3, a feature-level draft head trained *separately* over the target's hidden states) is
built end-to-end and lossless but a CPU wall-clock **loss** — acceptance around 1.6 tokens/verify,
too low for the extra draft-forward-plus-correction cost to be recovered (`docs/spec/README.md`).
`docs/spec/07-stageb-gemm-verify.md` (GPU-batched verify) is "NO-GO small models; parked
conditional-GO for ~70B + short drafts (needs >8 GB GPU)." `tech_report.pdf` Tab. 4 reports
Qwen3.8-Flash-Next's *native* MTP head, under 4-step speculative decoding with QSA index reuse, at
a **mean accepted length of ~4.06–4.07 across MT-Bench/GSM8K/MATH/HumanEval/MBPP** — roughly 2.5×
goinfer's own EAGLE-3 acceptance, from a head trained jointly with the target rather than bolted on
after. That's the mechanism, quantified: a drafter trained separately from the target either
doesn't pay for itself on CPU or needs GPU scale goinfer isn't built for; one trained *with* the
target and shipped inside the checkpoint sidesteps both problems. `docs/spec/08-dspark-dflash.md`
(queued `P10`/`P15`) is goinfer's existing bet on the fix via *third-party* block drafters; a
native in-checkpoint MTP head is the same bet at zero integration/hosting cost, *if* a model using
one is ever brought up — the `Drafter`/`Verifier` interfaces (`docs/spec/00-core.md`) are already
architecture-agnostic, so wiring one in is an adapter, not new verify/rollback machinery. Worth
naming as a side benefit — not a reason to prioritize this model on its own.

## What's genuinely new

1. **QSA (Qwen Sparse Attention) — a real bring-up, now fully spec'd.** Core mixer: 24Q/2KV heads,
   head_dim 256. Its indexer (`tech_report.pdf` §2.1.2, eq. 12–19) is a separate small MQA
   structure — 4 query heads + 1 shared key head, keys compressed to blocks of r=4 tokens by
   average-pooling, partial RoPE on 64 of 128 indexer-head dims, block-causal ReLU-summed
   importance scores, top-K_B block selection (K=2048 tokens ⇒ K_B=512 blocks, matching the HF
   card exactly). Nothing in goinfer does learned block-sparse selection over the target model's
   *own* attention today — the two things that sound adjacent, the grammar DFA mask (`constrain`)
   and the n-gram speculative-decoding drafter (`decoder/spec_ngram.go`), both operate on *draft*
   tokens, not on which context the target attends to. One implementation nuance worth flagging
   for whoever builds the tiny fixture: the indexer is trained via a two-stage CPT distillation
   (dense distillation from the backbone's own full-attention scores, then joint sparse training —
   `tech_report.pdf` §2.1.2 "Training Details"), so a random-weight synthetic-tiny golden will
   exercise the forward math (shapes, block-causal masking, top-k expansion, RoPE) correctly but
   won't produce meaningful block selections — expected, and consistent with how every other T1
   tiny-golden in this repo already works (it gates forward-pass correctness, not learned
   behavior).

2. **Gated Residual — plumbing, not a layer, and now fully specified.** Confirmed exactly, formula
   and all: 4 branches, each read via its own RMSNorm (`γ_i`) and an elementwise gate computed
   through a rank-320 (`r = d/8`, `d`=2560) low-rank bottleneck (down-project → SiLU → up-project →
   sigmoid), averaged into the block input; written back through a per-branch *scalar* sigmoid gate
   (no elementwise write, no branch-mixing operator — `H_res` is deliberately dropped: "the
   dominant inference cost of a widened stream" is the read, so the design spends complexity there
   and keeps the write cheap). GR **replaces** each block's pre-normalization rather than sitting
   in front of it, and there are **two** GR modules per layer — one for the attention/mixer
   sublayer, one for the MLP/MoE sublayer (`tech_report.pdf` §2.2, eq. 21–34). None of that changes
   the blast-radius finding: `decoder/forwardn.go` is in the explicit **core** hashed-file set
   (`docs/task-parity-coverage.md:33`, alongside `model.go`/`attention.go`/`mlp.go`/`kvcache.go`/
   `rope.go`/`rmsnorm.go`/`registry.go`/`arch.go`/`config.go`) — changing it re-stales every
   family's parity record, the same "maximum blast radius" `docs/scoping-lfm2.md` §G flagged for
   touching `registry.go`/`kvcache.go`. `forwardn.go` today assumes one residual stream through the
   attention+MLP fold (its own comments mark "one residual add" points in the batched path);
   widening that to 4 gated branches touches the shared spine every family's forward pass runs
   through, not a new `forward_X.go` — the risk is still blast radius, not arithmetic complexity
   (the actual math per read/write is compact and cheap, by design).

3. **N-gram embedding table — new, and namespace-collides with something unrelated.** 20M
   bigram/trigram entries, looked up per token and, per `tech_report.pdf` §2.3, "augment[ing] the
   corresponding token representation" — placed at layer 2 only in the shipped config (an ablation
   over layers 1/2/3/4/10/15/25 and combinations found no depth regime consistently better, so
   Alibaba picked layer 2 specifically to overlap host-memory prefetch with layer-1 compute, not
   because it's numerically special). The exact combination point relative to the 4 GR branches at
   that layer isn't spelled out in the sections read — worth confirming from the eventual reference
   implementation rather than assumed. Designed for host-memory residency with async prefetch
   (`tech_report.pdf` §2.3: "deterministic addressing enables host-memory offloading and
   asynchronous prefetching") — which is architecturally closer to what `decoder/moepaging.go`
   already solves (demand-loading a sparse subset of a large table into a memory budget) than to a
   green-field problem, even though the lookup key (an n-gram hash) and storage shape (a flat
   table, not per-expert weight tensors) are both genuinely new. **Flagging the name collision
   explicitly:** this has nothing to do with `decoder/spec_ngram.go` (goinfer's existing
   suffix-automaton speculative-decoding drafter, `docs/spec/02-cache-ngram.md`) beyond sharing the
   word "n-gram" — a future pickup should not conflate the two.

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

Beyond the license re-read, one practical gate this pass could **not** confirm — and it may bind
harder than the license:

- **Is there a runnable HF reference yet?** Every hybrid bring-up in this repo so far — `qwen3_5_moe`
  pinned against `transformers==5.10.2`'s `modeling_qwen3_5_moe.py` (`docs/qwen3_5_moe.md`) — has
  had a working `transformers` forward pass (native or `trust_remote_code`) to pin a golden
  against. No `modeling_*.py` was found linked from the GitHub repo (README + `tech_report.pdf`
  only); the repo does link a kernel library, `github.com/QwenLM/FlashQLA` (the fused GDN kernel
  used for training, per `tech_report.pdf` §"Kernel Efficiency"), which is not a reference
  forward pass either. If Alibaba's reference code hasn't landed anywhere loadable, there's no
  oracle to validate a synthetic-tiny fixture against yet — check this before scoping the
  bring-up further, it may be the actual gate, license aside.
- **Confirm the checkpoint dtype** (BF16 vs FP8) from the HF repo's actual file listing — it
  changes download size ~2× and, if it's genuinely FP8, raises the same blocker already on record
  for DeepSeek V4-Flash: **"there is no fp8 support anywhere in the tree today"**
  (`docs/post-v1.0-models.md`). Worth checking early since it could gate the whole effort
  independently of QSA/GR/n-gram-embed.
- **Five-minute check:** is `Qwen3.8-27B-Base` (the tech report's own comparison baseline) the same
  family goinfer already runs as `qwen35DenseArchitecture`? If so, GDN's real-checkpoint coverage
  extends one model further than stated above.

## Effort, if picked up (synthetic-tiny bring-up only — not the 180B)

| Component | Size | Notes |
|---|---|---|
| QSA (new mixer + indexer) | **L** | spec is now fully known (`tech_report.pdf` §2.1.2, eq. 12–19) — the size is real new code (indexer + block-sparse core attention), not unknown scope |
| Gated Residual (4-branch) | **M–L** | formula now fully known and compact (low-rank read, scalar write, no branch mixing) — the risk is blast radius (core `forwardn.go`, re-stales all families), not the math |
| N-gram embedding table | **M** | new but isolated; access pattern resembles `moepaging.go`'s demand-loading more than a green-field problem; exact GR-branch injection point still to confirm |
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

Either way, gate zero is unchanged: re-read the LICENSE, confirm an HF reference exists, resolve
the dtype discrepancy — in that order, before any code.

**Cross-reference:** added under `docs/post-v1.0-models.md`'s "Watching" section pointing here, so
this doesn't sit as an orphaned scoping doc nobody's roadmap points to.

## Sources checked this pass

- [Qwen blog post](https://qwen.ai/blog?id=qwen3.8-flash-next) (metadata only — no architecture detail fetchable)
- [Hugging Face model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Qwen3.8-Flash-Next LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
- [GitHub repo README](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- `tech_report.pdf` ("On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and
  Training Stability", Qwen Team, dated 2026-08-26 on the document itself) — supplied by Francis
  and read directly, 28 pages, in full
- [MarkTechPost coverage](https://www.marktechpost.com/2026/08/26/alibabas-qwen-team-releases-qwen3-8-flash-next-a-125b-multimodal-moe-with-6b-active-parameters-previewing-the-qwen4-architecture/)
- goinfer tree, read directly 2026-08-27: `decoder/registry.go`, `decoder/config.go`,
  `decoder/forwardn.go`, `decoder/moepaging.go`, `docs/qwen3_5_moe.md`,
  the completed qwen3.6 real-checkpoint task record (internal, untracked), `docs/queue-correctness.md`,
  `docs/task-moe-streaming.md`, `docs/spec/README.md`, `docs/task-parity-coverage.md`,
  `docs/parity-coverage-policy.md`, `docs/prompts/dspark-license-issue.md`,
  `docs/scoping-lfm2.md`, `docs/post-v1.0-models.md`
