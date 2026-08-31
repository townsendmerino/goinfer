# Correctness queue — CLOSED ENTRIES, archived 2026-08-31

> **This is the closed half. The live queue is [`docs/queue-correctness.md`](../queue-correctness.md)**,
> which still holds the open work (G7, G8, G1, Q1) — this queue is **not** empty the way the
> performance queue is.
>
> Seven entries moved here because they are finished: **G4** (Nemotron 3 Nano, T3 done), **G5**
> (Qwen3-Next, slice oracle cosine 1.00000000), **G6** (Laguna), **G9** (GPT-2 Metal residency,
> shipped), **G10** (gpt-oss Metal bridge, declared), **G11** (Mellum-on-Metal, both halves green)
> and **Q2** (the GGUF-quant cross-gate gap).
>
> Archived under `docs/README.md`'s rule that finished work is kept **for the reasoning rather than
> the outcome**. `docs/completed/` is excluded from `scripts/queue_citation_lint.py`, so these
> entries' `path:line` citations are now retired from the live gate and will drift as code moves.
> That is the documented trade for archiving: a citation below is evidence of what was true when
> written, not a live pointer.
>
> **G5 kept a residue and it is a nice-to-have, not a blocker:** GGUF, and a full-model T3 on a
> bigger box. Its capability row stays `experimental` because a slice is not a full-model T3.
>
> **G10's header was stale when it was archived and is corrected below.** It read "declaration held
> pending Mellum's own gate"; its own body ends "**Net state — DECLARED**". Mellum's gate (G11)
> closed the same day. Left visible rather than rewritten, because the pattern — a resolution
> appended under a conclusion nobody deleted — is the single most common defect these queues have.

## Closed entries

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

**T2 DONE too — G4 IS COMPLETE.** Both tiny gates are now in the parity sweep's required
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

**G5 · Qwen3-Next / Qwen3-Coder-Next (80B-A3B)** — Phase 0/1 + T1 golden DONE (`mac`, 2026-08-17);
**real-weight SLICE oracle DONE (`linux-62gb`, 2026-08-17): cosine 1.00000000**, argmax and greedy
continuation exact on 3 Gated DeltaNet layers + 1 full-attention layer of the real
`Qwen3-Next-80B-A3B-Instruct`. Row stays `experimental`: a slice is not full-model T3, and no full
reference forward of a 163GB bf16 model fits 62GB. Four layers is the MINIMUM that spans the hybrid
(`full_attention_interval: 4` ⇒ layers 0-2 linear, layer 3 full), and the slice needs only 6 of the
41 shards (~24GB). Remaining nice-to-have: GGUF, and a full-model T3 on a bigger box.

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

**T3 METHOD DETERMINED 2026-08-26 (`linux`), and the recorded blocker is SOFTER than it reads.**
Work order: `docs/prompts/qwen3next-t3-validation.md` (`b6335bf`).

**1. The `shared-path (via qwen3_5_moe)` proxy does NOT apply — checked first, before any download,
and it fails structurally rather than on judgment.** `docs/parity-coverage-policy.md` requires an
alias family share the source's forward file(s) **and the same `deps_hash`**, and names the
disqualifier outright: *"If an alias family ever gains its own forward file or a distinct
`deps_hash`, it loses proxy status and needs its own T3."* `qwen3_next` trips both. The hashes are
`sha256:0e752ed6…` vs `sha256:1b8ec4f7…`, and they can never agree: `deps_hash` is a content hash
over the family's dep files (`decoder/parity_manifest_test.go:197`) and the two `own:` sets differ by
six files. The difference is live, not a stale record — `TestParityManifest_fresh` passes with 27/27
families enforced. Substantively it is right that it fails: `decoder/qwen3next.go` is where deltas
1–3 above actually live, so a proxy row would assert that `qwen3_5_moe`'s oracle covered code
`qwen3_5_moe` never executes.

**2. "No full reference forward of a 163GB bf16 model fits 62GB" is true of a RESIDENT reference and
only of that.** The constraint is co-residency, and co-residency is not required: pin the reference
logits to disk in one process, then run goinfer against the pinned file in another. That is already
how every tiny golden is made (policy: *"its golden logits pinned once offline via the HF
reference"*) — the slice route generalised the *slice*, when what needed generalising was the
*pinning*. With `accelerate` disk-offload the reference's resident footprint is roughly one layer,
not the model, and a single prefill pass over the eval prompt streams the 163 GB from NVMe **once**
rather than per token. Confirmed present on this box: `~/.venv-vl` has `Qwen3NextForCausalLM`
(transformers 5.12.0), `accelerate` 1.14.0, torch 2.12.0+cpu, and 502 GB free on the NVMe.
**So the target method is `full-forward-oracle`, not `weightDiff`** — and `weightDiff` was never
available anyway, since it is GGUF-vs-safetensors and the GGUF loader is the other open item below.

**3. PRE-REGISTERED, before the checkpoint finished downloading and before any number exists.**
Two facts from the released `config.json` (read, not assumed) make a below-floor result a live
possibility rather than a remote one, and the decision rule is fixed now so it cannot be argued
after the fact.

- **goinfer's side is forced to `int4` by capacity, not chosen.** 80B at int8 is ~80 GB against
  62 GB of RAM; int4 is ~40 GB and fits. There is no int8 run to fall back to on this box.
- **Routing is 10 of 512 experts — 1.95%, the sparsest in the table by a factor of two.**
  `nemotron3nano` at 6/128 (4.7%) already measured **0.978 at int8int8 vs 0.9977 with f32
  activations**, and its recorded reason is the expert-flip cliff: perturbing the router flips
  *which* experts run, a discrete change no averaging smooths. `qwen3_next` is sparser still, and
  is being quantized harder.

**Decision rule, fixed in advance:**

| measured full-vocab cosine | verdict |
|---|---|
| ≥ 0.98, divergences near-ties under the 3% rule | `validated`, `method: real-model-oracle`, quant recorded as `int4` |
| 0.975 – 0.98 | **PARKED, not argued into a pass.** The band below a threshold is where motivated reasoning lives |
| < 0.975 | not a pass — and the cause is *measured*, not asserted, per below |

**Below the floor, the two causes are distinguishable and must be distinguished.** Smooth,
non-monotonic per-layer agreement = quant noise at extreme sparsity: the row stays `experimental`
with the measured number and a deployment note ("do not run this family at int4"), which is a real
deliverable. A cliff, or a curve that cannot recover, = a loader or forward defect: **STOP and
file** — the work order puts a fix and its own validation outside this task, and E2's precedent is
that four "demotion judgments" were really two loader bugs. Floor the **mean**, not min-over-N.

**RESULT 2026-08-26 — the method works; the number misses a bar it was never calibrated against.**
Full write-up: `docs/measurements/qwen3next-t3-int4-2026-08-26.md`.

    argmax        got=12095 want=12095            EXACT
    continuation  [12095 13 576 6722 315 9856]    EXACT, all 6 tokens ("  Paris. The capital of Germany")
    logit cosine  0.989876  against the gate's 0.99   FAIL by 0.000124

**The co-residency blocker is retired, and not only for this family.** A full bf16 reference forward
of the 163 GB model completed on this 62 GB box in **26 minutes** (11 layers CPU / 41 disk, 1179 s
to load and place, 70 s for the prompt forward). Any family whose reference will not co-reside can
now have a real T3 instead of a slice.

**The gap is quantization, established by evidence rather than argued.** The family's slice oracle
loads the SAME real checkpoint at **f32** and gets **cosine 1.00000000** — same loader, same tensor
layout, same fused `in_proj_qkvz` split — so the loader and forward are exact on these weights. The
full run differs by depth (48 vs 4 layers) and int4, and every discrete check stayed exact. A defect
does not survive six argmax decisions intact.

**Row stays `experimental`, and the reason is a bar-calibration question, not a numerics one.** The
threshold is `if cos < 0.99 { // int8 W8A8 vs bf16 }` — an **int8** bar, and this is the repo's first
**int4** T3, forced by capacity (80B at int8 ≈ 80 GB against 62 GB RAM). The pre-registered rule
above said ≥0.98; the gate says 0.99; **quoting the looser one after seeing the number is what
pre-registration exists to prevent**, so the stricter committed standard governs. Filed as **G23**.

**The manifest was not hand-edited.** `emitParityRow` no-ops on failure, so `EMIT_MANIFEST` will not
write this run, and the work order's rule is rows-by-measurement-never-by-hand. The row still reads
`tiny-golden`, which is conservative and true. `decoder/qwen3next_oracle_test.go` is likewise still
absent from `own:`: adding it moves `deps_hash`, and refreshing that hash while the row carries the
old metrics would re-bless an unmeasured row. Both move together, once the bar exists.

**STATUS: the full checkpoint downloaded** to `~/models` (NOT `/srv/models`), 41 shards /
162.6 GB, resumable, detached; 6 shards were already present from the 2026-08-17 slice pull.
**Feasibility of the offloaded reference is REASONED, not yet measured** — it is verified when a
reference forward actually completes, and if it does not, that is the finding and the row stays
`experimental` with the measured reason.

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
flagship demo for P10 (now owned by `docs/spec/08-dspark-dflash.md` — the performance queue closed 2026-08-31) once that lands — vendor-blessed
drafters instead of self-trained ones. XS 2.1 is the consumer-hardware entry point (33B/3B
active); S 2.1 targets the 96-128 GB Mac crowd.

**G9 · GPT-2 GPU residency on Metal — SHIPPED (2026-08-18): admitted end-to-end, min cosine
0.999094** — `mac`

Follow-on from `G7`'s two shared kernels (`FeatOutBias`, `FeatRopeMscale`): GPT-2 needed all four
of `FeatLayerNorm`/`FeatNonGatedMLP`/`FeatLearnedPos`/`FeatOutBias`, since it's the one admitted
family that departs from RMSNorm/gated-SwiGLU/RoPE entirely — checked directly against
`decoder/registry.go`'s `gpt2Architecture` and `decoder/mlp.go`'s `nonGatedMLP`/`decoder/rmsnorm.go`'s
`layerNorm`, not assumed. Real work, not just re-flagging: `layernorm_quant` (mean-subtract, generalized
with a `hasBias` flag since Cohere's LayerNorm has none, GPT-2's does) and `act_quant` (the
non-gated activation-only counterpart to `swiglu_quant`) are two more genuinely new kernels, plus
`gemv_w4a8_resid_bias` (the coal-family counterpart to `G7`'s `gemv_w4a8_sa_bias_resid`, needed
because GPT-2's FFN down-proj K=3072 exceeds the SA family's 1536 cap) — all three gated
standalone the same way (`metal/gpt2_kernels_test.go`, `TestCoalGemvBiasResid` in
`gemv_w4a8_sa_bias_resid_test.go`).

**A genuine cross-backend finding, not Metal-specific:** `decoder/residency.go`'s
`decodeRunnerEligible()` had a HARD, backend-agnostic decline —
`if a.NonGatedMLP || a.LearnedPosEmbed || a.OutBias { return false }` — sitting BEFORE the
per-backend feature-flag check every other capability (sandwich norms, softcap) already routes
through. This blocked GPT-2 on EVERY backend regardless of what kernels existed, and explains why
WebGPU's own `FeatNonGatedMLP: true` (declared for Nemotron-H, which bypasses this check via its
own early-return) was reachable for Nemotron-H but not GPT-2. Relaxed to match the sandwich-norm
precedent (verified affects ONLY GPT-2 in practice — Nemotron-H and gpt-oss both short-circuit
before reaching this line; full `go test ./decoder/...` clean, `TestParityManifest_fresh` clean —
`residency.go` isn't in the hashed dependency set). **Side finding, material to `G7`:** gpt-oss
hits an EARLIER, deeper decline in the same function — `case a.qwen35 != nil || a.llama4 != nil ||
a.gptoss != nil: return false // own forward, not yet bridged` — meaning gpt-oss's own non-uniform
forward path (`runLayersGptOss`) has NO bridge to the generic resident dispatch on ANY backend yet.
Building gpt-oss's three kernels (this session, both backends) is necessary but not sufficient:
even a fully-declared `FeatAttnSink` would still hit this earlier block. That bridging work — a
dedicated resident forward function mirroring Gemma-4's own-forward bridge — is a separate,
larger piece nothing built so far addresses. GPT-2 does NOT hit this (it rides the generic
uniform-layer dispatch), which is why it was reachable and gpt-oss, so far, is not.

**Full wiring landed and builds/runs correctly** — `resident`/`residLayer` struct fields, per-layer
weight+bias loading in `BuildResident` (including the position-embedding table via
`PosEmbedResident`), the `encodeNorm` helper routing LayerNorm vs RMSNorm at every norm site, the
RoPE-skip in `encodeAttention`, the non-gated MLP branch in `encodeLayer`, and the host-side
`addLearnedPos` add at every production embedding entry point (`Forward`, `ForwardEmb`,
`ForwardEmbPipe`'s pipelined executor). Two real bugs, both caught before landing on main:

1. `L.invf = d.NewBufferFloats(invf)` panicked on GPT-2's genuinely-empty inv-freq table (no RoPE
   at all) — `aikit/gpu.Device.NewBufferFloats` doesn't handle a zero-length slice. Guarded on
   `len(invf) > 0`; `L.invf`/`L.mscale` simply stay unused for a learned-pos family, matching the
   RoPE dispatch already being skipped there.
2. **C-10, resolved for the vocab case specifically (not the general SA-family fix).**
   `BuildResident` declined with `metal: vocab width 50257 is not a multiple of 8 — SA-GEMV
   tail-write hazard (audit C-10); use the CPU path` — a real, deliberate, pre-existing safety
   gate (the `gemv_w4a8_sa`-family decode kernels have no `row >= N` bounds guard, so a non-%8
   output width risks a silent tail-write into already-written or uninitialized rows). Traced the
   exact mechanism rather than assuming: Metal's `dispatchThreads:` makes the LAST threadgroup
   non-uniform (fewer than 256 threads) whenever the dispatch total isn't a multiple of the
   threadgroup size, and `row = tgid*(tgs>>5)+sgid` silently miscomputes for that non-uniform
   group (`tgs` shrinks, so the formula recomputes an EARLIER row instead of the true tail row).
   **The LM head only ever touches vocab width through two kernels, and only one has the hazard:**
   `gemv_w8a8_coal` (`forwardLogits`, the full-logits path) addresses its row directly via
   `threadgroup_position_in_grid`, which Metal guarantees correct regardless of a threadgroup's
   uniformity — no hazard, %8 or not. `gemv_w8a8_amax` (`ForwardArgmax`'s fused fast-decode path)
   uses the same hazardous `tgs`-derived formula as the SA family. Considered and rejected padding
   the weight/output to a safe multiple of 8 (the low-risk fix that worked for `G7`'s bias+resid
   kernels): a padding row's score depends on the real, unknown-at-load-time activation, so it
   cannot be guaranteed to lose the argmax against real logits — it could just as easily win it,
   silently returning the wrong token. Routed around the kernel instead: `ForwardArgmax` now
   detects `V%8 != 0` and falls back to the full-logits path + host `argmaxF32`, which is already
   safe. Slower for a non-%8-vocab family (materializes the whole vocab instead of the fused
   reduction) but correct always beats fast-but-wrong, and GPT-2's vocab is cheap to materialize.
   `bad8("vocab", V)` dropped from `BuildResident`'s width checks accordingly — `hidden`/
   `intermediate` stay checked, since those still feed the SA-family kernels this doesn't touch.
   Gated directly (`TestGPT2ForwardArgmax_matchesFullLogits`): `ForwardArgmax` must equal
   `argmax(Forward)` at every position, since `residentParity`/`TestGPT2ResidentParity` only
   exercises the full-logits path and would have stayed green even with a broken fallback.
   **The general fix (an `N` param + `row >= N` guard across the SA family/MoE variants) is still
   not done** — out of scope here since nothing GPT-2 needs touches those kernels with a non-%8
   width; remains open for any future family whose hidden/intermediate dims aren't %8.

**Scoped the general fix on request (2026-08-19) — DEFERRED, deliberately, not forgotten.**
Checked whether the vocab fix's trick (pad the buffer, dispatch the padded count, never read the
padding) generalizes to `hidden`/`intermediate`: it does not, and the reason is real, not
theoretical. Vocab-width data (logits) is TERMINAL — computed once, read once, discarded — so
padding it is inert. Hidden/intermediate-width data is the RESIDUAL STREAM, flowing through every
subsequent layer, and multiple call sites read it unboundedly rather than to an explicit `H`
(`addLearnedPos`'s `for i := range dst`, `loadEmbedRow`'s `dst := r.x.Floats()`), while
`aikit/linalg.WeightMat.Row` only fills the true-width elements of `dst` and leaves any padding as
untouched garbage — checked directly in aikit's source, not assumed. Padding `H`/`I` the way the
vocab fix did would let that garbage silently enter every norm/GEMV downstream. The only fully
general fix is what the kernel comment already said: real `N` param + `row >= N` guards across
every SA-family/MoE dispatch site — genuinely invasive, touches every currently-working dense/MoE
family's hot path, and has no current beneficiary (every shipped and realistically-near-term
family's hidden/intermediate dims are %8). Decision: leave `bad8()`'s decline as the permanent,
correct behavior — it is sound, just untested (no test constructs a synthetic non-%8-hidden
config to confirm the decline fires cleanly rather than corrupting; a real but low-priority gap,
not fixed here). Revisit only if a real family with non-%8 hidden/intermediate dims actually shows
up.

**Third bug, caught the same way as the first:** `r.invf = d.NewBufferFloats(m.RopeInvFreq())` (the
model-level RoPE table used only by the prefill path) ALSO panicked on GPT-2's empty inv-freq
table — a second call site with the identical empty-slice hazard as `L.invf`, found only because
getting past the vocab decline exposed it. Guarded the same way; `prefillOK` is already `false` for
a learned-pos family (`prefillFeatures` doesn't include `FeatLearnedPos`), so `r.invf` is
confirmed genuinely unused.

**Declared for real this time, evidence in hand:** `FeatLayerNorm`/`FeatNonGatedMLP`/
`FeatLearnedPos`/`FeatOutBias` are now permanently declared for Metal — unlike the earlier
temporary declaration (tried, reverted when C-10 first surfaced, matching `G7`'s `FeatAttnSink`
precedent), this one has `TestGPT2ResidentParity` (min cosine 0.999094, 17/24 argmax-exact) and
`TestGPT2ForwardArgmax_matchesFullLogits` actually green, not just building. Verified with a REAL
checkpoint, not synthetic: `testdata/gpt2` (GPT-2 small, 124M, downloaded via
`huggingface_hub.snapshot_download('gpt2', ...)` — `HF_HUB_DISABLE_XET=1` needed, the default xet
transport 404'd), `scripts/pin_gpt2_real.py` regenerated `testdata/gpt2_forward_golden.json`
(benign float32 noise in the last few digits vs the previously-committed golden — same real
checkpoint, different torch/hardware build), and `decoder`'s own `TestGPT2_forwardParity` confirms
cosine 0.9999999999998812 / exact argmax against it. `TestResidentBackendFeatures_noOverclaim`
(11→15 features) and `admissionGolden` (`gpt2` → `[metal]`) updated deliberately, not silently;
`docs/capability-matrix.md`/`docs/hardware-matrix.md` regenerated. Full `go test ./decoder/...` and
`go test -tags goinfer_testhooks ./metal/...` clean.

**G10 · gpt-oss Metal residency bridge — DONE 2026-08-18: DECLARED.** *(Header corrected on
archive 2026-08-31. It previously read "declaration held pending Mellum's own gate" — a hold that
its own "Net state — DECLARED" paragraph below had already lifted, on the same day G11 closed
Mellum's gate.)* — `mac`

Continuing `G7`. Extended the GENERIC MoE machinery (`metal/moe.go`'s `moeLayer`/`moeResident`/
`buildMoE`/`buildMoELayer`/`encodeMoEFFN`) with an `isGptOss` split rather than duplicating a whole
bridge file (Gemma-4's precedent) — gpt-oss's MoE (routed-only top-4/32, no shared expert) is much
closer to the existing Mixtral/Qwen-MoE shape than Gemma-4's parallel dense‖MoE. Added per-expert
stacked bias buffers (`expGuBias`/`expDBias`, addressed `idx[slot]*2*inter`/`idx[slot]*H` matching
the kernels' own convention) and two new decoder-side accessors, `GptOssExpertDownBiasResident`
(mirroring the existing `GptOssExpertBiasResident` for gate‖up) — the down-bias accessor didn't
exist yet, everything else (`GptOssActResident`/`GptOssSinksResident`) was already there from
CUDA's own in-progress bring-up (`cuda/backend.go` loads `gptOssSw`/`gptOssSinks`/`gptOssExpBias`
but never dispatches them — dead code today, first real consumer). Relaxed
`decodeRunnerEligible()`'s hard `case a.gptoss != nil: return false` to fall through to the common
checks (mirroring gemma4's precedent) — safe on its own: `metal.BuildResident`/`cuda.BuildResident`
both check `MissingResidentFeatures` BEFORE building anything, so an arch-shape admit with an
undeclared feature still declines cleanly, never half-builds.

**Consolidated a real duplication before it shipped:** initially added `AttnSinkResident`/
`GptOssSwigluResident` accessors to `decoder/residency.go`, not realizing `GptOssActResident`
(alpha/limit + ok) already existed and was already CUDA's convention. Removed the duplicates,
switched `metal/model.go` to the existing accessor — one source of truth for "is this gpt-oss"
across both backends, not two.

**Got a genuine end-to-end pass, then hit the pre-1.0 core freeze.** With two real bugs fixed —
(1) `decoder/gguf.go`'s gpt-oss router load used the AMBIENT quant mode (`mat(...)`) instead of
staying f32 like every other family's router (`gptoss_safetensors.go`'s own comment: "top-k
selection is discrete and quantizing it flips experts"); latent since nothing had ever loaded
gpt-oss at a non-f32 quant before Metal residency's `f32Mat(router)` panic caught it; fixed with
`streamMat(..., quantNone, ...)`, matching qwen35's own precedent three lines below it in the same
file — (2) `decoder/forward_gptoss.go`'s `runLayersGptOss` never set `cache.scr`, unlike every
other own-forward family (`forward_llama4.go`/`forward_deepseek.go`/`forward_qwen35.go`/
`forward_granite.go`/`forward_nemotron.go` all guard `if cache.scr == nil` at entry); invisible
because every existing gpt-oss golden goes through `Model.NewCache` (which sets it), so only a
raw-`NewKVCache` caller (Metal's own `residentParity` harness) ever hit the nil-deref — the tiny
fixture ran resident on Metal and matched CPU: 8/8 argmax-exact, min cosine 0.998901.

Both fixes touch `decoder/gguf.go`/`forward_gptoss.go`, explicitly named in the standing pre-1.0
core-numerics freeze (`TestParityManifest_fresh` confirmed it, going stale for 25 families).
Reverted both, verified the freeze tripwire green again, and surfaced the conflict rather than
push through it. Got an explicit scoped exception for exactly these two diffs — both provably
non-numeric for every currently-pinned golden (the router fix is a no-op under the default f32
quant every gpt-oss golden uses; the `cache.scr` guard is dead code for every `Model.NewCache`
caller, which is every existing golden) — confirmed via `scripts/refresh_parity_hashes.sh`'s
auditable goldens-gated refresh (26 passed / 0 failed / 19 skipped for want of fixtures) plus a
direct run of `TestGptOssGGUF_parity` (cosine 1.000000 exact, since that specific test isn't
matched by the refresh script's `GOLDEN_RE` pattern — `TestGptOssGGUF_parity` doesn't end in
`_forwardParity`/`_logitParity`/`_textParity` nor match `TestGGUF_.*_parity`, a real gap in that
script's coverage worth fixing separately).

**Declaring `FeatAttnSink` turned out to need declaring `FeatRopeMscale` too (YaRN), and THAT flag
is shared with Mellum** — `mellumArchitecture` requires exactly `{FeatMoE, FeatPerLayerRoPE,
FeatQKNorm, FeatRopeMscale, FeatSlidingWindow}`, all of which Metal already has except
`FeatRopeMscale`. Declaring it for gpt-oss would have silently admitted Mellum too, with ZERO
Metal validation (no real ~24GB Mellum checkpoint on this box). Tried building a synthetic
structural test (`TestMoE_assemblyVsDense`'s pattern: a tiny safetensors fixture through the real
loader) to at least prove the QK-norm+RopeMscale+MoE+sliding-window COMPOSITION wires correctly
without needing real weights — it failed, but the failure was a methodology artifact, not a real
bug: bisecting down (identical experts → real vs identity QK-norm → a plain dense qwen2 control
with NO QK-norm at all) showed even the QK-norm-free control fails the same 0.95 cosine bar
against fully-random untrained weights at realistic dims (0.898). Random synthetic weights lack
the structure real checkpoints have (dominant outlier dimensions that make quantization error
small relative to signal), so int4/int8 quantization noise dominates regardless of which features
are in play — `TestMoE_assemblyVsDense` only avoids this by forcing EXACT mathematical equivalence
(identical experts, same underlying int4 GEMV kernels cancel), not by achieving a high cosine on
independent random weights. Abandoned the synthetic-test approach; debug files removed.

**Net state — DECLARED (explicit user call, overriding the initial "hold" decision above): no
Mellum checkpoint is reachable on this box, so waiting was not actually available as an option
right now.** `FeatAttnSink`+`FeatRopeMscale` are both `true` for Metal; `TestGptOssResidentParity`
(new: `metal/gptoss_real_test.go`, uses the existing tiny fixture
`decoder/testdata/gptoss_tiny.gguf`) is live and green (8/8 argmax-exact, min cosine 0.9989).
`docs/capability-matrix.md`'s `GPUResident` column for `gpt_oss` reads "yes". Full
`go test ./decoder/...` and `go test -tags goinfer_testhooks ./metal/...` clean.

**Known consequence, tracked as `G11` below: Mellum is now `ResidentEligible` on Metal with ZERO
validation there.** Not a theoretical risk — it is the literal reason `FeatRopeMscale` was held in
the first place, and the hold was lifted by explicit choice, not because the risk resolved. Mellum
already has a trusted GPU path (WebGPU), so this is a second, unvalidated path on one backend, not
a new failure mode for the family overall — but it needs closing out before the next tag, with a
REAL trained-weight gate (`G11`), not another synthetic-random-weight attempt (proven inconclusive
above).

**G11 · Mellum-on-Metal real-weight validation — CLOSED 2026-08-18: both halves green** — `mac`

**Status 2026-08-18: steps 1-3 are done and green on the Linux box.** The real-weight slice
exists and goinfer's own CPU forward matches the HF f32 reference on it EXACTLY:

```
mellum2 slice (REAL weights, 4 layers): argmax got=417 want=417 | logit cosine=1.00000000
mellum2 slice batched prefill:          argmax=417              | cosine=1.00000000
```

`decoder/mellum_slice_test.go` (tag `realckpt`) also pins, on real weights, the things G10
actually changed: the 3-sliding/1-full interleave survived slicing, `SlidingWindow` 1024, and
**YaRN mscale 1.2772588722239782 on layer 3 with 1.0 on the sliding layers** — the exact scalar
`FeatRopeMscale` is about. The doc's step-2 worry was checked rather than assumed: the generic
by-length truncation caught BOTH per-layer lists this family carries (`layer_types` and
`mlp_layer_types`), and `full_attention` really is at index 3 in the released config, so a
4-layer slice does cover YaRN.

**The step-4 handoff needs one correction: the slice CANNOT be committed.** It is 4.0 GB (four
layers of a 64-expert MoE), so it is gitignored like every other slice fixture
(`decoder/testdata/mellum-*-slice/`); only the golden
(`decoder/testdata/mellum_mellum2_slice_golden.json`, tracked) ships. The mac regenerates it
bit-identically instead, and the tracked golden is what pins both boxes to the SAME reference:

1. Fetch **two of the five shards** — `model-00001-of-00005.safetensors` (5.0 GB) and
   `model-00005-of-00005.safetensors` (4.3 GB) — plus `config.json`,
   `model.safetensors.index.json` and the tokenizer files from
   `JetBrains/Mellum2-12B-A2.5B-Instruct`. Those two shards hold layers 0-3 and the
   embed/norm/head; the slice script is partial-download-friendly and reads nothing else.
   ~9.3 GB instead of 23 GB.
2. `SLICE_SRC=<dir> SLICE_TAG=mellum2 SLICE_LAYERS=4 SLICE_PREFIX=mellum python scripts/pin_slice_oracle.py`
   (transformers 5.10.2 / torch 2.12 was used here). It writes the same 4.0 GB
   `decoder/testdata/mellum-mellum2-slice/` and would overwrite the golden — **check `git diff`
   on the golden afterwards: it must be unchanged.** A changed golden means the two boxes'
   references disagree, and that is the finding, not a nuisance.
3. Run the decoder gate first as a cross-box control:
   `GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestMellumSlice -v`.
   It must reproduce argmax 417 / cosine 1.0 on the mac before Metal is asked anything.
4. Then `metal/mellum_real_test.go`: `residentParity` against
   `decoder/testdata/mellum-mellum2-slice/`, the same harness `TestGPT2ResidentParity` /
   `TestGptOssResidentParity` use. Nothing in `decoder/features.go` changes — the flags are
   already declared; this only adds the missing PROOF.
5. If it finds a REAL bug (a real slice does not have the synthetic noise floor): fix it, or if
   it cannot be fixed quickly, revert `FeatRopeMscale`/`FeatAttnSink` for Metal and reopen the
   `G10`/`G11` split.

**Steps 4-5, done on the mac (2026-08-18).** Regenerated the slice locally per the recipe above
(two shards, ~8.7 GB download; the venv had none of transformers/torch/safetensors/huggingface_hub
preinstalled, so those were installed fresh first). The golden check passed: `git diff` on
`decoder/testdata/mellum_mellum2_slice_golden.json` showed only the machine-local `source` path and
benign float32 build noise (max abs diff 4e-05, cosine 0.9999999999957 — the same class of harmless
noise GPT-2's own golden regeneration hit) — the two boxes' references AGREE, so the regenerated
file was discarded in favor of the tracked one rather than committing a no-op diff. The cross-box
control (`GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestMellumSlice -v`)
reproduced argmax 417 / cosine 1.00000000 exactly, as required before touching Metal.

`metal/mellum_real_test.go` (new): `residentParity` against the regenerated slice, gated on
`requireHeavyModel` + `FeatRopeMscale` being declared (same double-guard convention as the other
heavy Metal tests). Result: **9/12 argmax-exact, min cosine 0.972006** — comfortably clears the
0.95 int4-noise floor every other resident gate (GPT-2, gpt-oss, the dense control) uses, and is
NOT the tighter 0.9999 bar `TestMellumSlice_realWeightOracle` uses, because that one compares
f32-vs-f32 (goinfer CPU vs HF reference) while this one compares int4/int8-quantized Metal against
int8 CPU — the same reason `TestGPT2ResidentParity`/`TestGptOssResidentParity` also gate at 0.95,
not 0.9999. No real bug found — `FeatRopeMscale`/`FeatAttnSink` stay declared for Metal, now with
a genuine trained-weight proof behind Mellum's admission instead of an accepted-but-unverified
side effect. Full `go test ./decoder/...` and `go test -tags goinfer_testhooks ./metal/...` clean
throughout. The downloaded shards (`~/models/mellum2-unq`, ~8.7 GB) and the regenerated slice
checkpoint (`decoder/testdata/mellum-mellum2-slice/`, 4.0 GB, gitignored) were left on disk at the
user's direction rather than cleaned up — available for reuse if this needs re-running.

**Original entry, for the reasoning behind all of the above:**

`G10` declared `FeatRopeMscale` for Metal to unblock gpt-oss's YaRN, which — as a documented,
accepted side effect — also admits Mellum (`decoder/registry.go`'s `mellumArchitecture` needs
exactly `{FeatMoE, FeatPerLayerRoPE, FeatQKNorm, FeatRopeMscale, FeatSlidingWindow}`, all of which
Metal already had except this one) onto the Metal resident path with **zero end-to-end
validation there**. This is the one open correctness gap from that work and should close before
the next tag ships a Metal build that will actually run Mellum checkpoints.

**Do NOT repeat the synthetic-random-weight approach — it is proven inconclusive, not just
unlucky.** `G10` tried building a tiny hand-rolled safetensors fixture (random independent
weights) through `metal.residentParity` and got a false-positive "QK-norm is broken" signal that
bisected down to: even a plain dense qwen2 control with **no QK-norm at all**, at realistic dims
(hidden=256), fails the SAME 0.95 cosine bar against fully-random untrained weights (0.898). Random
weights lack the structure a real checkpoint has (dominant outlier dimensions that keep
quantization error small relative to signal), so int4/int8 noise swamps everything regardless of
which features are in play. A synthetic fixture cannot discriminate a real Metal bug from this
noise floor — don't spend time on it again.

**The right tool already exists and is generic:** `scripts/pin_slice_oracle.py` — "Tiny goldens use
random weights, where the router is near-uniform... A real slice exercises the same code on the
distributions the model actually produces" (its own docstring, written for exactly this class of
problem on Laguna/qwen3_next). It slices the first N REAL layers out of a real safetensors
checkpoint (partial-download-friendly: only the shards holding those layers + embed/norm/head),
truncates every per-layer config list generically (by length match against the original layer
count, not by field name), and writes a small, git-trackable `<prefix>_<tag>_slice_golden.json` +
sliced checkpoint dir. `scripts/pin_mellum2.py` already names the real checkpoint path
(`~/models/mellum2-unq`, ~24GB, "fits the 64GB box" — this box).

**Concrete steps for the Linux box:**
1. `SLICE_SRC=~/models/mellum2-unq SLICE_TAG=mellum2 SLICE_LAYERS=4 SLICE_PREFIX=mellum
   ~/g4venv/bin/python scripts/pin_slice_oracle.py` (or whichever venv has `transformers`/`torch`/
   `safetensors` — see `scripts/pin_mellum2.py`'s shebang comment). `SLICE_LAYERS=4` mirrors
   Laguna's own reasoning: Mellum's 3:1 sliding/full interleave needs at least 4 layers to cover
   BOTH attention types (3 sliding + 1 full/YaRN) — the same coverage floor, not copied blindly;
   confirm layer 3 (0-indexed) is genuinely `full_attention` in Mellum2's real `layer_types` before
   trusting a 4-layer slice covers YaRN at all.
2. Verify the script's generic per-layer-list truncation actually catches everything Mellum's
   config carries per-layer (`layer_types` is the one that matters here) — it truncates by length
   match against `num_hidden_layers`, which should catch it automatically, but confirm rather than
   assume (this script's own header names Laguna's `layer_types`/`num_attention_heads_per_layer`/
   `mlp_layer_types` explicitly as the reason it went generic — a new family is exactly the case
   that pattern exists to protect).
3. Write `decoder/mellum_slice_test.go` mirroring `decoder/laguna_slice_test.go` /
   `decoder/qwen3next_slice_test.go` (both ~130-160 lines, same shape) — the decoder-CPU-only gate
   proving goinfer's OWN forward matches the real HF reference on the slice. Commit the golden +
   sliced checkpoint (small — a handful of MoE layers' worth of real weights, not the full 24GB).
4. Hand the committed slice back to `mac`: build `metal/mellum_real_test.go` (the REAL-weight
   successor to the abandoned `metal/mellum_wiring_test.go`) using `residentParity` against the
   sliced checkpoint directory directly — same harness `TestGPT2ResidentParity`/
   `TestGptOssResidentParity` use, just pointed at `decoder/testdata/mellum-mellum2-slice/`. This
   is the test that actually closes the gap `G10` opened; nothing else needs to change in
   `decoder/features.go` (the flags are already declared) — this only adds the missing PROOF.
5. If the slice test finds a REAL bug (not noise — a real slice won't have the noise-floor problem
   the synthetic attempt did), that's real information either way: fix it, or if it can't be fixed
   quickly, revert `FeatRopeMscale`/`FeatAttnSink` for Metal and reopen the `G10`/`G11` split.

**Q2 · The GGUF-quant cross-gate gap — CLOSED, and it was unplumbed too** — `linux`, `bd08936`→

The cross-gate check showed `scripts/parity_sweep.sh` covering the GGUF quant formats while the goldens
refresh did not. **(a) Exposure: a LAG, not a hole.** The parity sweep is not in CI — it is
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

