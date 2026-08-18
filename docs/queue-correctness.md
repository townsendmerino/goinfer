# Correctness queue

Parity, numerics, goldens, quantization, model families. Anything whose success criterion is **agreement with a reference** — a cosine, an argmax match, a golden. If the question is *does it compute the right thing*, it belongs here.

> **One of four queues.** The work list is split by *success criterion*, not by component:
> [performance](queue-performance.md) · [correctness](queue-correctness.md) ·
> [engineering](queue-engineering.md) · [release](queue-release.md).
> [`QUEUE.md`](QUEUE.md) is the index over all four and holds the cross-cutting sweeps.
>
> **Task docs are NOT queues.** `docs/task-*.md` are *design records* — why a thing is built as it
> is — and they are cited from 88 code comments. A queue entry cannot carry that, so the task docs
> stay put and the queues hold only the open work.
>
> Entries keep the section they were filed under (`In flight`, `Queued`, …) and their original IDs,
> so a citation to an ID still finds it.


## Queued

**G4 · Nemotron 3 Nano (30B-A3B) as a new family — Phase 0/1 + T1 + **T3 DONE**
(`mac` then `linux-62gb`, 2026-08-17)** — `linux`

**T3 PASSED on the real 30B-A3B checkpoint (`linux-62gb`, 2026-08-17): argmax exact, cosine
0.997668, and all 6 continuation tokens exact** against an HF bf16 oracle of the same weights
(`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, 63 GB). Recorded on the existing `nemotron_h`
manifest row rather than a sibling — the manifest is keyed by REGISTRY model_type and the
registry dispatches both the dense and MoE variants on `nemotron_h`, unlike deepseek_v2/v3 which
are genuinely separate keys. Test: `decoder/nemotron3nano_real_test.go`; oracle pin:
`scripts/pin_nemotron3nano_real.py`; asset `GOINFER_NEMOTRON3NANO_HF`. The adapter needed NO
changes — config parsing, the 52-block MEMEM* pattern (23 mamba / 23 moe / 6 attn), the
noaux_tc router and the non-gated relu² experts were all correct on released weights.

**The one real finding: this variant must NOT be run with int8 ACTIVATIONS.** The first T3 run
failed at cosine 0.978086 with a degenerate continuation, and the cause was not a bug — it is
router sensitivity. Measured on the same forward:

| quant | cosine |
|---|---|
| `int8int8` (int8 weights + int8 activations) | 0.978086 |
| `int8` (int8 weights, f32 activations) | **0.997668** |

The model routes **6 of 128 experts (4.7%)** — far sparser than deepseek_v3 (0.99951),
qwen3_5_moe (0.99333) or granitemoehybrid (0.99566), all measured at int8int8 — and its own
DENSE parent scores 0.99574 at int8int8. Quantizing activations perturbs the router enough to
flip which experts run, a discrete change no averaging smooths (the expert-flip cliff already
recorded for granite's MoE stack). Comparing against the repo's other MoE rows is what
distinguished "sensitive" from "broken" cheaply, before any hunt for a forward bug.

**T2 DONE too — G4 IS COMPLETE.** Both tiny gates are now in `scripts/parity_sweep.sh`'s required
list: `nemotron3nano-tiny|TestNemotron3NanoMoE_textParity` and, closing a pre-existing hole found
in the same pass, `nemotron-tiny|TestNemotron_textParity` — **the DENSE variant had never been in
the sweep either**, despite being T3-validated since 2026-08-15. Its fixture is pin-generated and
gitignored exactly like deepseek-tiny/kimi-tiny/phi3-tiny/llama4-tiny, so it follows the
established pattern; a release box must have run the pin script, which is what "run it on the box
with the full asset set" already requires.

**Cross-machine reproducibility confirmed while doing it:** the tiny MoE fixture regenerated on
`linux-62gb` matches the Mac's committed golden at **cosine 1.000000** with identical argmax and
continuation (max |Δ| 3.06e-07 on the raw logits — float32 rounding). The Mac's golden was kept
rather than churned.

**The GGUF path is now proven end-to-end too** (`decoder/nemotron_moe_real_test.go`, Q4_K_M
24.7 GB): 23 mamba / 23 moe / 6 attn survive the metadata round-trip, 128 experts / top-6 and the
shared width 3712 all resolve, and the model generates coherently through its real chat template
— distinct-trigram 0.843, *"Eiffel Tower, Louvre Museum, Notre-Dame Cathedral"*. That was a
genuine hole: T3 loads SAFETENSORS, and GGUF is a different loader with a different expert layout
(one tensor per expert vs ALL experts fused into one 3-D tensor per projection). It had been
verified by reading a real header and hand-dequantizing one layer — correct dims and sane values,
which is not the same as a model that runs. A fused expert stack read with the wrong stride gives
finite values, correct shapes and confident nonsense.

The new gate uses the chat template + distinct-trigram bar rather than the raw-completion +
distinct-token floor the DENSE gate uses — following the audit note on
`TestNemotronReal_gate` itself, which records that its own pattern measures "did the forward
avoid TOTAL collapse", not coherence, and that on gemma-4-26b that manufactured a false "int4 is
broken" signal surviving a week.

GPU residency is still correctly DECLINED for this family on both cuda and metal, so it is
CPU-only for now — the one remaining limitation, and not a correctness one.

*Original handoff below, kept for the record.*

**T3 handoff prompt: `docs/prompts/nemotron3nano-t3.md`.** Real-checkpoint parity — needs the
actual weights (20GB+), which doesn't fit this Mac's disk/RAM headroom alongside a reference
oracle load. Self-contained; has every gotcha found so far (no fp8 support — don't grab the
`-FP8` release variant, use `-BF16` or a GGUF quant instead) and exactly what's already verified
vs still open.

Phase 0 (config-verified, not assumed): pulled the real `config.json` + NVIDIA's
`modeling_nemotron_h.py` (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-{BF16,FP8}`). `model_type` is
literally `"nemotron_h"` — the SAME string as plain Nemotron-H, distinguished only by a new `E`
character in `hybrid_override_pattern` (MoE-FFN, alongside `M`/`*`/`-`), which the old parser
correctly hard-rejected rather than silently mis-loading. Router is byte-for-byte DeepSeek-V3's
`noaux_tc` (sigmoid + `e_score_correction_bias` + group-limited top-k + `routed_scaling_factor`,
shared expert added ungated) — fully reusable via `routeExperts`. But the experts are NON-GATED
relu² (`up_proj`/`down_proj` only, no `gate_proj` — confirmed against the real safetensors index),
which `moeMLP` hard-rejects (SwiGLU only); Nemotron-H has its own dedicated forward loop anyway
(`runLayersNemotron`), never touches the shared `moeMLP` other MoE families use. One field gotcha
caught by verifying rather than assuming: `moe_shared_expert_intermediate_size` (3712) is NOT
`n_shared_experts * moe_intermediate_size` (1*1856=1856) — needed its own Config field.

Phase 1 (implemented, `go test ./decoder/...` green, `TestNemotron_textParity` still cosine
1.000000 across two separate shared-file edits — the safetensors loader change and the later
`gguf.go` change — each independently proven inert for every existing family via
`scripts/refresh_parity_hashes.sh`'s goldens gate, not asserted): new `nemoMoE` block-kind threaded
through the pattern parser, `arch.go`, `nemotronhArchitecture`, `runLayersNemotron` (new
`nemotronMoE`/`nemotronExpertFFN`, non-gated relu² experts), and the safetensors loader
(`backbone.layers.N.mixer.{gate,experts.I,shared_experts}.*`, matching the real tensor index).

**Real T1 golden landed, not just hand-computed units.** Installed `torch`+`transformers` into a
fresh venv (`~/.venv-nemotron3`, ~845 MB — no `trust_remote_code` needed, mainline `transformers`
5.15.0 already ships `NemotronHConfig`/`NemotronHForCausalLM` with the MoE fields natively).
`scripts/pin_nemotron3nano_tiny.py` builds a tiny random-weight checkpoint (mamba+attention+moe,
no plain dense-mlp — matching the real family's own pattern shape) and runs the real HF forward.
`TestNemotron3NanoMoE_textParity`: **cosine 1.000000, exact argmax match.** Caught one real bug
along the way: current `transformers` canonicalizes `layers_block_type` entries to
`"linear_attention"`/`"full_attention"` regardless of input spelling (confirmed empirically, not
assumed) — goinfer's switch had a bare `default: // "mamba"` that would have silently
misclassified those as Mamba layers. Fixed to accept both spellings and error on anything truly
unrecognized instead of assuming mamba. The real NVIDIA-released checkpoint is unaffected (it
ships `hybrid_override_pattern`, parsed independently), but this is now real, not
theoretical, protection for anything re-saved through a current transformers version.

**GGUF loader also landed**, verified against a real file's header (fetched via HTTP Range — first
30 MB of `bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF`'s Q4_K_M, parsed with goinfer's own
`embed.GGUFFile` reader — ground truth, not a paraphrase of the llama.cpp PR that added support).
Real findings from that: llama.cpp's GGUF arch string is `nemotron_h_moe` (different from HF's
`model_type`, which stays `nemotron_h` — normalized on load), the MoE metadata keys are
`expert_count`/`expert_used_count`/`expert_feed_forward_length`/
`expert_shared_feed_forward_length`/`expert_shared_count`/`expert_weights_norm`/
`expert_weights_scale`/`expert_group_count`/`expert_group_used_count` (all confirmed present,
including the `topk_group` equivalent an early secondhand PR summary claimed was absent — it
wasn't, which is exactly why this got verified against a real file instead of trusted), and
experts are FUSED per-projection (`blk.N.ffn_up_exps.weight`/`ffn_down_exps.weight`, one 3-D
tensor each) unlike the safetensors path's one-tensor-per-expert layout — reuses the existing
`stackedExperts` helper other GGUF MoE families already use. `TestNemotron3NanoMoE_ggufConfig`
validates the config/layer-classification parsing against the real captured metadata (not a
hand-crafted fixture).

**Tensor-loading side also spot-checked against real data (2026-08-17), not left as "provably
additive" alone.** Fetched the first 3 GB of the same real Q4_K_M GGUF file (HTTP Range, well
under the 24 GB free budget — the full file is 20GB+) and directly exercised the actual
dequantization path (`g.Tensor`/`g.RowDequantizer`, the same calls `buildWeightsFromGGUF`'s new
`nemoMoE` case makes) against one full real MoE layer's six tensor kinds: router, correction bias,
shared-expert up/down, and BOTH fused fully-quantized 3-D expert tensors — checked expert 0,
expert 1, AND expert 127 (the last index, not just the easy-to-get-right first one) for both
up_exps and down_exps. Every dimension matched what the loader code expects exactly; every
dequantized value was sane (small-magnitude, symmetric around zero for the weights; the
correction bias came back a real, non-trivial ~56.7-56.8 uniformly — notably NOT near-zero the way
an untrained/synthetic bias would be, which is itself a small confirmation this is real trained
data, not a placeholder) with zero NaN/Inf across every tensor. This is real structural/dequant
verification, not a full model forward — that still needs the whole checkpoint (T3, below).
Partial-download files deleted after verification; disk back to the ~24 GB baseline.

**Also fixed, not just implemented:** `decoder/residency.go`'s `decodeRunnerEligible` had an
unconditional `if a.nemotron != nil { return true }` — admitting ANY Nemotron-H model to every GPU
resident backend. `gpu/residency.go`'s WebGPU builder switches on block-kind with cases 0/1/2 and NO
default; an unhandled `nemoMoE` (kind 3) would silently leave that layer's op buffers nil rather
than erroring. Changed to `return a.MoE == nil` — declines GPU residency for MoE Nemotron-H until a
backend actually implements it (verified against its dispatch, not assumed).

**Follow-up (2026-08-17): the "may already be GPU-accelerated via the staged path" question above
is now resolved — no.** Checked `metal/backend.go` directly: `metalBackend.MatmulBT` is literally
`linalg.MatmulBT`, the SAME shared CPU SIMD kernel `cpuBackend` uses, and `*metalBackend` doesn't
implement `QuantBackend` (the interface a backend needs to get a specialized int8 dispatch in
`decoder/weightmat.go`'s `matmul()`) at all. Int4 weights bypass any backend hook entirely
(`matmul()`'s int4 branch calls `linalg.MatmulBTW4A8Into` unconditionally, no `Backend`-specific
dispatch exists in the generic path for any family). So `runLayersNemotron`'s `matmul(m.be, ...)`
calls, when `m.be` is `*metalBackend` but residency is declined (true for MoE Nemotron-H, and
would also be true for plain Nemotron-H if `GOINFER_SSM_RESIDENT` weren't set), run on CPU
regardless — **`--backend metal` currently gives this family zero speed benefit over `--backend
cpu`.** Real Metal acceleration would need either a Metal `QuantBackend` implementation (moderate,
reusable scope) or a genuine resident-path Metal kernel port of the single-op-per-block dispatch
(substantial — a different shape of work than any Metal kernel work landed so far this session,
which targeted dense mixer+FFN families, not Nemotron-H's structure). Neither is in scope for `G4`
as currently filed; noting it here so a future GPU-residency pass starts from a confirmed
baseline, not the open question this used to be.

**Remaining: T3 real-checkpoint parity, handed off — see `docs/prompts/nemotron3nano-t3.md`.** A
full forward pass through the actual model, safetensors or GGUF, compared to a real oracle. Needs
downloading the full checkpoint (20GB+ GGUF or a comparable safetensors quant) — this Mac's ~24 GB
free (as of 2026-08-17) can't absorb that alongside a reference-oracle load; the CUDA box has more
headroom for exactly this kind of validation, which is why it's filed there. (The
partial-download-plus-cleanup technique used above for the GGUF tensor spot-check remains
available on the Mac if more of the loader ever needs checking without a full download — noted in
the handoff prompt too.) Full scoping: `docs/post-v1.0-models.md` "Next up" §1.

**G5 · Qwen3-Next / Qwen3-Coder-Next (80B-A3B) as a new family** — Phase 0/1 + real T1 golden DONE
(`mac`, 2026-08-17); **T3 real-checkpoint parity remains, tagged `linux`**

Full scoping and reasoning: `docs/post-v1.0-models.md` "Next up" §2. Confirmed against the real
`Qwen/Qwen3-Next-80B-A3B-Instruct` config and `modular_qwen3_next.py` (not assumed) to be a
config-mapping delta on `qwen35Architecture`, as expected — but with three real deltas, one of them
a checkpoint-layout difference rather than a config field:

1. No `layer_types`; a `full_attention_interval` stride instead (`"linear_attention" if
   (i+1) % interval else "full_attention"`, 0-indexed) — `normalizeQwen3NextLayerTypes`
   (`decoder/config.go`) synthesizes `LayerTypes` from it.
2. `partial_rotary_factor` is a top-level field with no `rope_parameters` object at all (unlike
   `qwen3_5_moe`, which always carries one) — `parseRopeSpec`/`validateQwen35` both hard-required
   `rope_parameters`, so `qwen3NextArchitecture` (`decoder/registry.go`) resolves RoPE via a
   dual-path (nested-if-present, else flat) mirroring `deepseekArchitecture`'s pattern, and a new
   `validateQwen3Next` (`decoder/config.go`) accepts either.
3. **The real checkpoint fuses the Gated DeltaNet input projections**: `linear_attn.in_proj_qkvz`
   and `linear_attn.in_proj_ba`, not the four separate tensors (`in_proj_qkv`/`z`/`b`/`a`)
   `loadQwen35Attn` (`decoder/weights.go`) already reads for `qwen3_5_moe`. Verified the exact split
   against `modular_qwen3_next.py`'s `Qwen3NextGatedDeltaNet.torch_forward` (per-key-head-group
   `torch.split`, order q/k/v/z and b/a) and against the tiny fixture's actual tensor shapes. Fixed
   with a **splitter, not a joiner**: `decoder/qwen3next.go`'s `splitQwen3NextQKVZ`/`splitQwen3NextBA`
   un-fuse the two tensors into the same four `deltaNetWeights` fields at load time (one new
   `qwen35Params.FusedDeltaNetProj` flag branches `loadQwen35Attn`) — the shared forward/gguf/serialize
   pipeline is untouched. (A joiner — refactoring `qwen3_5_moe` and the forward path onto the fused
   layout instead — would have touched validated/frozen-adjacent code for zero benefit; this is a
   load-time reshape, off the decode hot path, so there's no performance difference between the two,
   only blast-radius.)

Real T1 golden (`scripts/pin_qwen3next_tiny.py`, transformers 5.15.0 `Qwen3NextForCausalLM`,
deliberately rewritten post-save to the release's flat config shape so the fixture actually
exercises deltas 1–2, not just what `qwen3_5_moe`'s fixture already covers) passes at **cosine
1.000000**, exact argmax, exact greedy continuation, both `forward` and `Generate` paths.
`parity_manifest.json` row added (`status: experimental`, `method: tiny-golden`, matching the `G6`
precedent for a tiny-golden-backed but not-yet-real-checkpoint family). Full onboarding-gate suite
green (`representativeConfig`, `familyDoc`, `archFeatureProfile`, `admissionGolden`,
`TestParityManifest_fresh`) except the pre-existing, unrelated `TestDispatchCensus` failure on a
Laguna GGUF metadata switch (confirmed via `git stash -u` against clean origin/main — not this
family's issue).

**Remaining: T3 real-checkpoint parity, GGUF loader.** Qwen3-Coder-Next is the recommended agentic
coding model for 64GB-class systems; closes a real gap in goinfer's strongest family. **80B total
does not fit the M1 Pro 16 GB rig even at int4** — this is a WebGPU/CUDA-streaming showcase, tagged
`linux` for that reason; scope the residency story before estimating total effort.

**G6 · Laguna (poolside) — DONE (`linux-62gb`, 2026-08-17): XS-2.1 + XS.2 + M.1 on one adapter** — `any`

**SHIPPED.** One `laguna` adapter serving all three released generations. T1 tiny goldens ×3
(one per generation, each vs ITS OWN `modeling_laguna.py`) at **cosine 1.000000** on both the
sequential and batched-prefill paths, plus a **real `poolside/Laguna-XS.2` (33B-A3B @int4)**
loader/coherence gate that generates *"1. Eiffel Tower / 2. Notre-Dame Cathedral / 3. Louvre
Museum"*. Manifest row is `experimental` **on purpose** — the real gate is coherence+structure,
not a T3 logit oracle, and `TestParityManifest_methodTier` correctly rejected the overclaim.

New primitives: softplus attention output gating (both granularities) and per-layer QUERY head
counts (`headsAt`/`maxHeads`), plus `RotaryDimLocal` completing the local/global RoPE triple.
Declares `FeatAttnOutputGate`, so every resident backend declines (`laguna → admitted by []`)
rather than silently skipping the gate.

**Four assumptions the released artifacts overturned**, all caught before they shipped:
QK-norm is unconditional and appears in no config; XS.2's `g_proj` is per-HEAD despite
`gating: true` (so granularity is read from the tensor shape); experts ship per-expert, not
fused; and XS.2 carries `mlp_layer_types` but not `mlp_only_layers`. The real gate then found
two more — batched prefill sized `q`/`ctx` from `NumHeads`, and never applied the gate at all
(cosine 0.957, identical to the gate-disabled mutant).

**Open, all nice-to-have:** a true T3 layer-slice oracle (a bf16 33B will not co-reside with
the int4 model in 62GB); an XS-2.1 real gate (another 63GB download; its per-head path is
covered by a tiny golden); **GGUF** (CORRECTION: I wrote "none exist yet" — wrong. llama.cpp has FIRST-CLASS `laguna`
support and official `poolside/Laguna-XS-2.1-GGUF` / `-S-2.1-GGUF` exist, plus many community
ones. The GGUF carries per-layer head counts as an ARRAY (`laguna.attention.head_count`),
separate `rope.dimension_count` 64 / `dimension_count_swa` 128, `expert_gating_func` 2
(sigmoid), `expert_weights_scale` 2.5, `leading_dense_block_count` 1, an `attn_gate` tensor,
and the chat template the safetensors dir lacks. This is a real, tractable loader task on top
of the existing stacked-expert/`exp_probs_b` machinery); and the **DFlash drafter pairing**
— `poolside/Laguna-XS.2-speculator.dflash` is downloaded and P10's block drafting shipped, so
this is the first vendor-blessed pairing available. **It is NOT a drop-in**: it ships its own
`embed_tokens`, a REDUCED-vocab `lm_head [32000, 2048]` and `d2t`/`t2d` translation tables,
whereas goinfer's DFlash path assumes the drafter borrows the target's embedding and head.
goinfer's EAGLE path already implements `d2t`, so the work is joining the two — a feature, not
a leftover. (5 taps at `aux_hidden_state_layer_ids` [1,9,17,36,39], block_size 8, mask token 12.) M.1 gets no real gate on this box (~220B).

Full record: `docs/task-laguna.md`.

*Original scoping entry:*

**Verified against the real released configs before estimating**, per the discipline G4 earned.
`model_type: "laguna"`, `LagunaForCausalLM`, custom `auto_map` code. XS: hidden 2048, 40 layers,
48/8 heads at head_dim 128, **256 experts top-8**, `moe_intermediate_size` 512, shared expert 512,
`moe_routed_scaling_factor` 2.5, vocab 100352, sliding_window 512.

**THE SCOPING WAS RIGHT ABOUT THE SHAPE AND MISSED TWO THINGS.** Confirmed: the 3:1 interleave is
real (30 `sliding_attention` / 10 `full_attention`, pattern `full,slide,slide,slide,…`), and
softplus gating + per-layer RoPE are the new primitives. Not in the scoping:

1. **Per-layer QUERY head count.** `num_attention_heads_per_layer` = `[48,64,64,64,48,…]` — full
   layers use 48 heads, sliding layers 64. goinfer has `headDimAt`/`kvHeadsAt`/`ffnAt` per layer
   (gemma4) but **no per-layer query-head count**; that is a new accessor plus every place that
   derives `qDim` from a scalar.
2. **Generation schema drift, same `model_type`.** XS-2.1 spells `gating: "per-head"` with a
   `gating_types[]` array; **XS.2 spells `gating: true`, which the vendor code maps to
   PER-ELEMENT** (`num_heads*head_dim` gates, not `num_heads`). Same field, different granularity,
   different tensor shape. XS.2 also adds `partial_rotary_factor: 0.5` and drops `norm_topk_prob`,
   `mlp_only_layers`, `decoder_sparse_step`. This is G4's lesson repeating — a family's own
   releases disagree on schema — so the loader must accept both spellings from the start.

**The one genuinely new primitive, exactly** (from the vendor's `modeling_laguna.py`, which says
Laguna attention *"is identical to Qwen2MoE attention except"*): an extra per-layer projection
`g_proj: [hidden → num_heads]` (per-head) or `[hidden → num_heads*head_dim]` (per-element), with

    attn_output *= softplus(g_proj(hidden_states))     # BEFORE o_proj

**Everything else reuses shipped primitives:** SWA/global interleave (gemma3/gemma4/mellum2),
MoE + shared expert + `norm_topk_prob` + routed scaling (deepseek/glm4_moe), YaRN (deepseek),
partial rotary (glm4/phi3), per-layer RoPE base (gemma4's local/global). `rope_parameters` is
keyed BY LAYER TYPE — full: YaRN theta 500k factor 32 partial 0.5; sliding: default theta 10k
partial 1.0 — which is gemma4's local/global split with YaRN on one side, i.e. config plumbing
rather than new math. **Attention sinks are NOT enabled** in either release
(`swa_attention_sink_enabled` absent), so that branch of the vendor code is dead weight here.

**Estimate stands as "otherwise cheap"** — two new primitives (softplus gating at two
granularities, per-layer query heads) on an otherwise-familiar softmax-GQA MoE.

**The strategic tie-in is now real, not prospective:** `poolside/Laguna-S-2.1-DFlash` and
`poolside/Laguna-XS.2-speculator.dflash` are published, and P10's block-drafting path SHIPPED
today (`serve --drafter`, gate 3 measured). Laguna would be the first pairing with a
VENDOR-BLESSED drafter rather than a third-party one.

**Note a newer generation exists:** `Laguna-XS.2` (320 likes) and `Laguna-M.1` are out, ahead of
the XS-2.1/S-2.1 this entry was filed against. Target XS.2 or support both — the config delta is
small but real, and XS.2's per-element gating is the more demanding of the two.

**SCOPE SET (2026-08-17): three generations under ONE adapter — XS-2.1, XS.2, M.1.** Full Phase 0
config-verify of all three, the vendor modeling code, the new primitive, the traps, and the
real-gate feasibility call now live in `docs/task-laguna.md`. Headlines: the vendor modeling code
is **byte-identical across generations** (differences are entirely config), so one adapter serves
all three; **M.1 is structurally SIMPLER** (no sliding window, no per-layer head counts, routed
scaling 1.0) but ~220B and so **not real-gateable on this box** (tiny-golden only, Kimi-K2 call);
`gating` has **three spellings across three releases** (`"per-head"` / `true` / `"per-element"`,
where the latter two are the same path); and the layer-type-keyed RoPE that looked new is mostly
existing machinery (`ropeScalingLocal` already exists). T3 real gate = **XS.2** (~63GB).

*Original entry:*

Full scoping and reasoning: `docs/post-v1.0-models.md` "Next up" §3. Softmax-GQA territory (mixed
SWA/global attention 3:1, the interleave pattern already handled for Gemma/Mellum2/gpt-oss) —
softplus attention gating and per-layer RoPE scales are the genuinely new primitives, otherwise
cheap. Strategic tie-in: ships with **official DFlash draft models**, which makes it the natural
flagship demo for P10 (`P10`, `docs/queue-performance.md:1135`) once that lands — vendor-blessed
drafters instead of self-trained ones. XS 2.1 is the consumer-hardware entry point (33B/3B
active); S 2.1 targets the 96-128 GB Mac crowd.

**G7 · gpt-oss residency upgrade (safetensors + MXFP4 loader, GPU residency)** — `any`

Full scoping and reasoning: `docs/post-v1.0-models.md` "Next up" §4. Not a new family — `gpt_oss`
already has real-oracle parity (`docs/capability-matrix.md:89`) — but it is **GGUF-only,
CPU-only** today. The MXFP4 *reader* already exists (`decoder/mxfp4.go`, verified bit-for-bit
against the reference `gguf` Python library) but was built for the GGUF path only. Two pieces: (a)
a safetensors loader for gpt-oss's native MXFP4-packed weights, (b) Metal/CUDA GPU residency.
gpt-oss-20b is one of the most-run local models in 2026 guides — an upgrade to an already-popular
family plausibly moves more real users than a new family would; weigh against `G4`-`G6` on that
basis, not just novelty.

**G8 · DeepSeek V4-Flash as a new family — blocked on fp8 support, post-1.0** — `any`

Scoping already done: `docs/completed/task-model-family-deepseek-v4-kimi-k3.md`'s Phase 0 verdict.
**Not** a `deepseekArchitecture` alias — eight new primitives (DSA sparse attention over a learned
Indexer, strided KV compression, sliding-window + attention sink, grouped low-rank output
projection, hash routing, `sqrtsoftplus` router scoring, hyper-connections, clamped SwiGLU).
**Hard prerequisite, not a subtask:** V4-Flash ships fp8 e4m3 blockwise-quantized weights and
**there is no fp8 support anywhere in the tree today** — file/estimate the fp8 reader as its own
piece of work before scoping the primitive additions. MIT license, DeepSeek's brand pulls the
whole local community, and native sparse attention is where the field (V3.2, GLM-5.1, V4) is
converging — building the DSA/compressor path once plausibly buys the next several Chinese
frontier releases, which is the strategic case for filing this now even though it's not a
near-term ship. Lowest priority of the five items filed alongside this one (`G4`-`G7`).

**G1 · LFM2.5-2.6B as an experimental family** — `linux`

Scoping prompt written. A fifth sequence-mixing family: interleaved gated short-convolution blocks
and GQA, `layer_types` controlling the pattern, `conv_L_cache` 3, LayerNorm QK-norm (not RMSNorm),
FFN dim computed rather than stated. The conv layers carry a rolling conv state instead of a KV
cache.

The estimate turns on two questions: whether Mamba-2's causal depthwise `conv1d` is factored out or
inlined, and whether the cache abstraction already carries mixed per-layer state types
(Granite-4.0-H and Nemotron-H suggest it may). Also unestablished: **whether LFM2.5 is
architecturally the same as LFM2** — the transformers docs cover only LFM2.

Blast radius matters: anything touching shared `decoder/` core re-stales all 19 enforced families.
Answer that before estimating.

**Q2 · The GGUF-quant cross-gate gap — CLOSED, and it was unplumbed too** — `linux`, `bd08936`→

The cross-gate check showed `scripts/parity_sweep.sh` covering the GGUF quant formats while the goldens
refresh did not. **(a) Exposure: a LAG, not a hole.** `scripts/parity_sweep.sh` is not in CI — it is
release-only, run by hand on the box (`RELEASING.md` §C1). So the formats are covered at release and
**not between releases**, which is exactly when a frozen-core edit gets only the goldens refresh.

**(b) Both routes priced before choosing, and route B turned out unnecessary:**

| route | cost |
|---|---|
| extend the goldens selector to the existing GGUF gates | **26.8 s**, 11 gates, no new fixtures |
| author GGUF-quant goldens for those 11 rows | unnecessary — the gates already exist and already pass |

Same shape as Q1(b): **unplumbed, not missing.** The gates were simply outside `GOLDEN_RE`. Adding
`^TestGGUF_.*_parity$` took the refresh from **19 passed / 0 quantized** at the start of this campaign
to **33 passed / 14 quantized**, and the cross-gate check now reports *"the two gates span the same
quantizations."*

One bug fixed in the cross-gate check itself: it compared a composite label (`int4/int8`, from a file
driving two quantizations) against atomic ones and reported a difference that was purely notational —
a permanent false positive in the check built to make real differences visible. Both sides are
atomised now.

**Q1 · The forward goldens prove f32 ONLY — no quantized path has a golden that runs** — `linux`,
**NEW. G-01 at the largest scale it has appeared.**

> **The "14 quantized" composition figure, resolved by enumeration rather than by authority
> (2026-08-12).** Two classifiers disagreed — an ad-hoc name grep said **7**, the refresh script's
> said **14** — and 14 had already propagated into commit bodies and into the proof requirement.
> Adopting it because it was the script's would have been a tiebreak by authority, so both were
> tested instead.
>
> **7 was structurally incapable of being right**, for two independent reasons. Five of the fourteen
> carry no quantization token in their NAME at all: `TestGemma4_logitParity` and
> `TestMellum2_logitParity` set it in the test body (`Options{Quant: "int8int8"}`), and
> `TestGGUF_gemma3/qwen2/qwen3_parity` set it in the **fixture filename** the test loads. No
> name-based match can see either. (The other two misses, `Q2_K` and `Q3_K_M`, were a plain gap in
> the ad-hoc pattern, which listed `q4|q5|q6|q8` — a bug rather than a structural limit, but it lands
> in the same place.)
>
> **The script's classifier cannot double-count.** `grep -c` counts matching LINES; every top-level
> result is one line; subtest lines are indented and excluded by its `^--- PASS:` anchor. Measured on
> the captured run: 33 top-level PASS lines, **0** indented ones, no duplicate names among the 14.
>
> And it does not misclassify — all fourteen drive a genuinely quantized path:
>
> | gate | quantization | set where |
> |---|---|---|
> | `TestGemma4_logitParity` | int8×int8 | test body |
> | `TestMellum2_logitParity` | int8×int8 | helper body |
> | `TestInt4_forwardParity` | int4 group-wise | test body |
> | `TestGGUF_Q2_K_parity` | Q2_K (+Q3_K/Q4_K/Q6_K mix-ins) | fixture |
> | `TestGGUF_Q3_K_M_parity` | Q3_K (+Q4_K/Q6_K) | fixture |
> | `TestGGUF_Q4_0_parity` | Q4_0 | fixture |
> | `TestGGUF_Q4_K_M_parity` | Q4_K (+Q6_K) | fixture |
> | `TestGGUF_Q4_K_S_parity` | Q4_K_S | fixture |
> | `TestGGUF_Q5_K_M_parity` | Q5_K (+Q6_K) | fixture |
> | `TestGGUF_Q6_K_parity` | Q6_K | fixture |
> | `TestGGUF_Q8_0_parity` | Q8_0 (tinyllama) | fixture |
> | `TestGGUF_gemma3_parity` | Q8_0 (gemma-3-270m) | fixture |
> | `TestGGUF_qwen2_parity` | Q8_0 (Qwen2.5-0.5B) | fixture |
> | `TestGGUF_qwen3_parity` | Q8_0 (Qwen3-1.7B) | fixture |
>
> **So 14 stands, and every commit body citing it is correct.** The reason is now recorded, which is
> the point: the figure is load-bearing in the proof requirement, and "the script said so" is not a
> reason. Note what the table also shows — **11 of the 14 take their quantization from a fixture**,
> so any future classifier that reads test names will undercount for the same structural reason.

int4 is the documented default quantization. **Zero goldens drive it.** And the hole is wider than
that: of the 19 goldens that actually RAN in the 2026-08-12 refresh, **every one is f32**.

| quantization | golden files | did any RUN? |
|---|---|---|
| f32 (explicit or default) | 24 | **19 ran** |
| `int8int8` (W8A8) | 3 — `gemma4_parity`, `gemma4_12b_parity`, `mellum2_parity` | **all 3 SKIPPED** |
| `int8` (weight-only Q8) | 1 — `gptoss_real` | not matched by the goldens regexp at all |
| **`int4` / W4A8** | **0** | — |

So `scripts/refresh_parity_hashes.sh` — the sanctioned freeze-exception path, and the thing that makes a
core edit auditable — **proves f32 numerics and nothing else**. A change that is bit-identical in f32
and wrong in int4 passes it in 6 seconds.

**Retroactive scope, and this is the part to act on.** Any claim of the form *"the parity suite
covers X"* is scoped to **the quantizations the goldens drive**, which today is f32. Every place such
a claim is written down needs that scope added — `docs/parity-coverage-policy.md`'s tier table,
`RELEASING.md`'s §C1, the README's support matrix, and the P6 commit body (which states it already).

**And the freeze protects what the goldens check.** The `6edd1ca` numerics freeze over `decoder/` is
enforced by `deps_hash` staleness, whose release valve is this goldens run. Where the goldens are
silent — every quantized path — the freeze is a *procedural* barrier with no numeric proof behind it.
That is not an argument for lifting it; it is an argument for knowing what it is.

**WHY THIS OUTRANKS THE REST OF THE QUEUE — sequencing, not enthusiasm.**

**P1 is the v1.0 headline and lives in the frozen core.** The numeric proof available when that core
unfreezes was **f32-only**. So lifting the freeze did not buy the ability to verify the work the
freeze defers — and the shortfall **would not have announced itself**, because the goldens would pass.
An f32-green refresh over an int4 regression is a passing gate, not a silent one; nothing in the
output distinguishes them.

That makes Q1(c) a **prerequisite for the v1.0 core work**, not a parallel item, and it belongs ahead
of the E-group release gate for that reason rather than because it is interesting. **Done
2026-08-12 (`1d0d1ed`)**: 23 fixtures across 16 architectures, so the prerequisite is now met for
int4 specifically.

**RUN WHAT EXISTS FIRST — and most of it was UNPLUMBED, not missing.** Done 2026-08-12, `a6c5b57`:

- **(b) the three `int8int8` goldens** skipped for one liftable reason, the same for all three:
  `GOINFER_HEAVY_TESTS` unset. **Two of the three pass here in ~70 s** (gemma4, mellum2). The refresh
  now enables heavy by default. The third (gemma4-12B) skips on a genuinely absent GGUF — an asset
  question, not a plumbing one.
- **(a) the `int8` golden did NOT turn out to be a selector bug.** `TestGptOssReal_logitParity` **does**
  match the regexp. It is invisible because `decoder/gptoss_real_test.go` is behind `//go:build realckpt`,
  which the refresh does not pass — and with the tag it still skips for a missing GGUF. **Two gates,
  either sufficient.** A one-line regexp change would have bought nothing.

**Non-f32 rows after (a) and (b): 2** (21 passed, 2 quantized). The distinction the ordering was meant
to test comes out clearly: **int8 was unplumbed** (one env var), **int4 is genuinely missing**, and the
gpt-oss int8 row is **asset-blocked behind a build tag**.

The refresh now also prints the **quantization breakdown**, because "19 passed" and "21 passed" read
identically to a human and that is precisely how this stayed invisible through nine prior refreshes.

**(c) int4 goldens — DONE `1d0d1ed`.** Scope measured *before* authoring and stated as a target: int4
has no divisibility constraint (`nGroups` is a ceiling divide), so eligibility was never the limit —
fixture availability was. **Target: 23 fixtures / 16 architectures. Delivered: 23 / 16.**

The goldens compare **int4 output against recorded int4 output**, not int4 against f32 within a
tolerance. A tolerance band against f32 measures quantizer loss — a real question with its own gate
on the policy's quant axis — and would read as "int4 is covered" while proving nothing about whether
the W4A8 path still computes what it computed yesterday. Only the self-comparison catches a
regression in the path the freeze protects and P7 will change.

Fixtures are **enumerated** from `testdata/` rather than listed by name, so a new family is picked up
without editing the gate, and a run comparing **zero** fixtures **fails** rather than passing.
Mutation-checked by perturbing the quantizer itself (`int4GroupSize` 32 → 64 → red).

Recorded **absences**, not gaps: `gpt_oss` (MXFP4-prequant, rejects a conflicting `--quant` by
design), `siglip_vision_model` (an encoder), `gpt2` / `mellum` / `qwen2` / `qwen3` (no tiny
safetensors fixture), `qwen2_moe` and `gemma4-dense-scaled-{24,48,64}` (incomplete fixture dirs).

**Refresh now reports 22 passed / 3 quantized**, against 19 passed / 0 quantized when this began.

**Also record with P6's 6.09 s price: cheap and thorough are different properties.** 6.09 s buys 19
passes and 11 skips. The skips are not free — they are the coverage this item is about.

**`TestDecodeParityInt4` diverges from its recorded golden — REAL checkpoint, NOT the synthetic
goldens above, found 2026-08-15, unclaimed.** `decoder/parity_int4_test.go`, real
qwen2.5-coder-0.5b int4 (W4A8, safetensors-loaded gguf), greedily continuing a fixed prompt: got vs
want diverge at token index 5 (`got 1438 want 11047`) and every token after — not a subtle drift,
a different continuation entirely. **Confirmed pre-existing and unrelated to two same-day changes**
via an isolated `git worktree` bisect: fails identically on the P1 pre-change tree AND at aikit
`v1.17.1` (before the day's aikit v1.19.0 bump) — same got/want arrays, byte for byte. So this
predates both P1 (`97f824a`) and the bump (`fb8e26b`); it was sitting on `main` before either.

One live lead, not yet chased: the test's own comment records a **recent asset-resolution fix**
("this site previously skipped whenever `GOINFER_PREQUANT_GGUF` was unset... under a bare `go test
./decoder`, this gate now RUNS where it used to skip") — meaning this real-checkpoint gate may have
been silently skipping for a long stretch, during which `parityWantInt4`'s golden could have gone
stale against real drift nobody was watching for. That is a hypothesis, not a finding — needs a real
bisect (not the two-point check done here) to find which commit actually broke it, or whether the
golden itself was simply never right. ~~**Unclaimed — pick up either box.**~~

**RESOLVED 2026-08-15 (`8f63a7d`, linux). The lead above was right, and it was the whole answer.**
The real bisect ran (474 revisions, 9 steps, isolated worktree): first bad commit **`7deb368`**
(2026-06-14) — *"integrate aikit v1.8.1 Qwen2.5-VL vision encoder"*. Its only code delta is
`go.mod`; aikit 1.7.3→1.8.1 carries two `linalg` commits that are not vision at all (`36ce824`
"fold W4A8 weight scales in-register", `52890f5` wiring it on NEON, both **aikit-repo** SHAs).
Folding the scales changes W4A8 accumulation, which moves a greedy continuation. **So: red for two
months, not two days** — consistent with the two-point check finding it already red at v1.17.1,
since v1.17.1 is far downstream of the actual cause.

**Answering the entry's own either/or: the golden was right when captured, and went stale — it was
NOT "never right".** And it is stale in the direction that matters. Scored by leading ids matching
an **f32 forward** of the same 0.5B on the same prompt: the new int4 path matches **11** of 24,
the pinned golden **5**, int8int8 (unchanged) **19**. The kernel made int4 *twice as faithful to
f32*; the gate was holding the *worse* path. Re-captured on that measurement rather than on the
gate being red — the distinction `parity-coverage-policy.md` draws between promoting a first-run
result and silencing a gate — and the identical mistake that file's own 2026-06-12 note records
for its predecessor. Second time on this gate.

**The finding worth keeping is not the fix.** The dark gate hid not just the failure but **when it
started**, and therefore what caused it — a dependency bump moved a numerics path with the one gate
watching it skipping. That is filed against the selector-coverage campaign in
`queue-engineering.md`, where "red for at least two days" is now corrected as the visible floor,
off by a factor of thirty.
