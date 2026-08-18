# task: MXFP4 quantization + gpt-oss model family, and the CPU decode scheduler

> Status: **NOT STARTED.** Drafted 2026-07-26 from a review of
> [`arizqi/cpubrrr`](https://github.com/arizqi/cpubrrr).
>
> Depends on aikit's Q8_K integer-accumulation task note (uncommitted in that repo, so not reachable from here) for §4 only. §1–§3 and §5
> are independent and can start immediately.

---

## 1. The gap

```
$ git grep -il -E 'mxfp4|gpt.?oss|gptoss'
(no output)
```

goinfer supports f32/bf16/f16 + int8/int4 from safetensors/GGUF/GPTQ/AWQ across ~20
architectures. It does **not** support MXFP4, and therefore cannot run **gpt-oss:20b** or
**gpt-oss-120b** at all.

This is a missing model family, not a performance knob. It is also the family where
cpubrrr measured its largest margin — **~77 tok/s vs llama.cpp's ~14 tok/s on an M4 Max
CPU, ~5×** — which says llama.cpp's MXFP4 MoE CPU path is leaving most of the available
memory bandwidth unused. *(Those are cpubrrr's own published figures on an M4 Max — an
external CPU-vs-CPU citation, not a goinfer measurement; used only to size the opportunity.)* That is an unusually large, unusually specific opening.

The capability matters more than the speed: "runs gpt-oss:20b and 120b, in pure Go, one
binary" is a headline. "Runs it 5× faster than llama.cpp on CPU" is the second sentence.

## 2. MXFP4 format

Microscaling FP4 (OCP MX spec): blocks of 32 elements, each element an `e2m1` 4-bit float
(1 sign, 2 exponent, 1 mantissa — 16 representable values), plus one shared 8-bit `e8m0`
power-of-two block scale. 4.25 bits/weight.

**Do not infer the layout.** cpubrrr's README is explicit that they recovered the
architecture math from llama.cpp source and verified their 4-bit unpacking **bit-for-bit
against the official `gguf` Python library before writing the forward pass** — which is
why their new architecture produced correct output on the first run. Replicate that order:

1. Write the unpacker.
2. Verify bit-exact against the reference `gguf` library on a real gpt-oss checkpoint.
3. Only then write the forward pass.

`scripts/gguf_index.py` and `scripts/extract_experts.py` in cpubrrr show the shape of the
extraction; the file layout is GGUF, which `embed/gguf.go` in aikit already parses.

### Kernel strategy
`e2m1` has 16 values, so the dequant table fits a NEON `tbl` lookup — cpubrrr's approach
is exact 4-bit arithmetic on **integer** hardware via `sdot`/`tbl`, with **no float
dequant in the inner loop**. Same principle as the Q8_K work: stay integer, convert once.

This is a natural sibling to the existing W4A8 path (`aikit/linalg/dot_w4a8_arm64.s`,
`quant_w4a8*`). Decide early whether MXFP4 becomes a new `WeightMat` kind in aikit or a
goinfer-local kernel; **prefer aikit** for consistency with int4, unless the block-scale
semantics force otherwise.

## 3. gpt-oss architecture

Sparse MoE. `gpt-oss:20b` ≈ 21B total / ~3.6B active per token. `gpt-oss-120b` ≈ 117B /
~5.1B active.

Register it through the existing architecture-descriptor path so the feature taxonomy
handles backend admission correctly. **Critically: it must be declined at load by the
CUDA and Metal backends until they implement MXFP4, and fall back to the CPU path — never
mis-run.** That guarantee is the load-bearing claim in the README; do not weaken it for a
new family.

**Memory.** 120b does not fit in RAM. cpubrrr runs it via `mmap`'d weights so it pages in
under pressure instead of OOM-killing. goinfer already has `aikit/mmap` (`MapReadOnly`,
`Advise`, `SpanCache` — an LRU of page-aligned spans under a byte budget), which is
exactly this substrate. Wire 120b through `SpanCache`; do not write a new mmap path.

## 4. Interleaved expert layout *(after aikit A1 §4)*

cpubrrr's largest MXFP4 win was quad-interleaving the weight bytes so each worker core
reads one sequential stream. For MoE decode — where per-token you touch a *subset* of
experts — the interleave must be **per-expert**, so that activating experts {3, 17, 22}
still yields one sequential stream per core within each.

Do this at prequant time and cache it in the `.giw` bundle. `cmd/prequant` already owns
that pipeline, and `.giw` already maps zero-copy, so the reordering cost is paid once.

Measure the kernel win and the layout win **separately**.

## 5. CPU decode scheduler — the yielding spin-barrier

Independent of everything above. Ship it on its own.

cpubrrr's most transferable debugging result: a **52 tok/s regression** traced to
thread-pool dispatch, where condvar wakeups silently cost **7.5 ms/token**. Their fix was
12 persistent workers running the whole forward pass behind a *yielding* spin-barrier —
spin briefly, then let the OS in.

The two failure modes they mapped:

- **Condvar/channel per layer:** wakeup latency dominates. At ~30 layers × a few barriers
  each, microseconds of scheduler latency become milliseconds per token.
- **Pure spin barrier:** collapses under jitter — one descheduled worker burns every other
  core spinning, and the failure is nonlinear.

The middle path is spin-then-yield. In Go: bounded spin on an atomic, then
`runtime.Gosched()`, then park. Persistent workers for the whole forward pass, not a pool
re-entered per layer.

**Audit `decoder`'s CPU path for both patterns before changing anything.** Instrument
per-token barrier wait time first — if goinfer's current path is already
worker-persistent, this is a smaller task than it looks, and the measurement tells you so
in an hour.

## 6. Acceptance criteria

1. MXFP4 unpacking bit-exact vs the reference `gguf` library, on a real checkpoint,
   **committed as a test with a golden fixture** — matching the existing parity discipline.
2. gpt-oss:20b runs end to end on the pure-Go CPU path and is parity-gated against the
   HuggingFace reference (argmax-exact + logit cosine), same bar as every other family.
3. gpt-oss-120b loads and decodes through `SpanCache` on a machine with less RAM than the
   model. Document the tok/s honestly, including the paging cost.
4. CUDA and Metal **decline** gpt-oss at load and fall back to CPU. Assert this in a test.
5. `docs/capability-matrix.md` and `docs/hardware-matrix.md` regenerate cleanly and pass
   the CI freshness gate.
6. Decode tok/s for gpt-oss:20b in `docs/benchmarks.md` with full provenance, measured
   server-to-server against llama.cpp/Ollama CPU-only. **Verify GPU placement from the
   server's own logs** — cpubrrr's README documents that Ollama's `num_gpu:0` is a
   *request, not a fact*, and that this contaminated their first published numbers. Check
   the logs; do not trust the flag.
7. Scheduler change (§5) measured independently: per-token barrier wait before and after.

## 7. Non-goals

- MXFP4 on CUDA or Metal. CPU first; GPU residency is a follow-up once the CPU path is
  parity-gated.
- Training or fine-tuning in MX formats.
- The other MX types (MXFP6, MXFP8, MXINT8).
- gpt-oss multimodal or tool-calling specifics beyond what the existing `chat` package
  already provides per-family.

## 8. First move

`./scripts/setup_gptoss.sh` in cpubrrr shows how they extract runtime data from a local
Ollama copy without copying weights. Pull `gpt-oss:20b`, write the unpacker, and get the
bit-exactness test against `gguf` green. That single test de-risks the whole task — it is
the step that let cpubrrr's forward pass work on the first run.

---

## Phase 0 for the gpt-oss UPGRADE (safetensors/MXFP4 + residency) — `linux-62gb`, 2026-08-18

§7 above listed "MXFP4 on CUDA or Metal … GPU residency is a follow-up once the CPU path is
parity-gated" as a non-goal. That CPU path is now T3 (`real-model-oracle`, argmax-exact vs HF
bf16), so this is that follow-up. Verified against the real `openai/gpt-oss-20b` before
estimating.

### What is already free

**Config routing costs nothing.** `model_type` is `gpt_oss`, which is ALREADY in the registry
mapping to `gptOssArchitecture` — the same adapter the GGUF path resolves through. The real
config carries `sliding_window: 128`, `swiglu_limit: 7.0`, YaRN (`factor 32`,
`original_max_position_embeddings 4096`, `truncate: false`), `rope_theta 150000`, 24 layers,
2880 hidden, 64/8 heads at head_dim 64, 32 experts top-4 — all fields the adapter reads today.
So this is a **weight-loader** task, not a family task.

**The dequant math is already written and bit-exact.** `decoder/mxfp4.go` carries goinfer's own
MXFP4 unpacker, verified bit-for-bit against the reference `gguf` library on a real checkpoint
(`scripts/extract_mxfp4_golden.py`).

### The one real difference: how the bytes are SOURCED

| | GGUF (shipped) | safetensors (new) |
|---|---|---|
| layout | 17-byte blocks, scale ‖ 16 data, contiguous | **two tensors**, split |
| data | — | `*_blocks` U8 `[32, 5760, 90, 16]` |
| scales | — | `*_scales` U8 `[32, 5760, 90]` |

Block size is 32 either way (16 bytes = 32 nibbles; 90 × 32 = 2880 = hidden), and the scale is
the same E8M0 exponent byte. So `mxfp4DequantBlock` already does the arithmetic; what it cannot
do is read a scale that lives in a different tensor. The change is a split-source variant
sharing the same table and `e8m0ToF32Half` — **no new numerics**, which means the existing
golden keeps covering the math.

Also present and NOT in the GGUF path's shape: **per-expert bias tensors**
(`gate_up_proj_bias` / `down_proj_bias`, BF16 `[32, 5760]` / `[32, 2880]`). gguf.go already has
`stackedExpertBias`, so the concept is loaded today — the safetensors names differ, that is all.

`self_attn.sinks` is BF16 `[64]` per layer — one per attention head, exactly what
`gptOssParams` expects.

### The gate this gets for free — and it is unusually strong

gpt-oss's GGUF path is already **T3-validated**. So the safetensors loader can be gated against
*the same model through an already-validated reader* — weightDiff plus a logit compare — rather
than needing a fresh HF oracle. Almost nothing else in the queue has a validated reference
sitting in-tree to diff against. **Estimate: 2–3h including that gate.**

### Residency is a separate go/no-go, and it is NOT small

Three constraints, none of which the loader work touches:

1. **It is a kernel change.** `FeatAttnSink` exists precisely because no resident backend
   implements the per-head sink in the softmax denominator; it declines CUDA, Metal AND WebGPU
   together. The clamped interleaved-SwiGLU expert with per-expert biases is a second kernel.
2. **This box cannot hold it plainly.** 20b MXFP4 is ~13.8GB against the 2070's 8GB. It is
   testable here only via the host↔VRAM MoE streaming path (which has done a resident 26B-A4B
   decode), which couples two hard things at once.
3. **Metal is unverifiable here** — that is the macbook's side, so it needs coordinating rather
   than assuming.

**Recommendation: do the loader now, decide residency on its own evidence afterwards.** The
loader is bounded, strongly gated, and delivers format coverage users hit first; residency is a
multi-part kernel effort whose own Phase 0 should be written separately.


### CORRECTION to the Phase 0 verdict above (same day, before any loader shipped)

Phase 0 said the safetensors difference was "**no new numerics**, only the addressing differs",
and that the existing bit-exactness golden would therefore transitively cover it. **Both halves
of that were wrong**, and diffing against the already-validated GGUF reader is what caught it.

Two layout facts, each measured at cosine **1.000000** against the same weight read through the
GGUF path (and ~0.08 — noise — for the assumed alternative):

1. **Intra-block nibble order is SEQUENTIAL, not GGML's.** GGML packs elements *j* and *j+16*
   into byte *j*; safetensors packs *2j* and *2j+1*. So `mxfp4DequantSplit` cannot reuse the
   GGML core, and the GGUF-captured golden does NOT cover the safetensors path.
2. **`gate_up_proj` is INTERLEAVED, not concatenated.** Row 0 is gate row 0 and **row 1 is UP
   row 0** — not rows 0..2879 = gate, 2880..5759 = up. (This is what "clamped *interleaved*
   SwiGLU" was naming all along; the GGUF path never sees it because llama.cpp's converter
   already separates the two into `ffn_gate_exps` / `ffn_up_exps`.)

Either mistake alone produces finite values, correct shapes and plausible magnitudes — and
completely wrong weights. Neither is visible in a shape check.

**How it was caught, and the near-miss worth noting.** The first cross-check compared a
dequantized safetensors expert against the GGUF and got cosine 0.081, which is equally
consistent with "wrong unpacking" and "different model". Resolving that ambiguity needed a
tensor unquantized in BOTH files: `input_layernorm` vs `attn_norm` matched at max|diff| = 0,
proving same weights and forcing the conclusion onto the unpacking. Without that step the
honest reading would have been "these files differ, gate unavailable" — and the real bug would
have survived.

The tests now pin the DIFFERENCE (`TestMXFP4_splitIsSequentialNotGGML`) rather than an
equivalence. The earlier version of that test asserted the two orders AGREE and passed, because
the implementation shared its mistake — a test and an implementation derived from the same wrong
assumption cannot check each other.

## Phase 2 scoping — GPU residency: NOT recommended as the next step

Phase 1 made the CPU path format-complete (GGUF **and** safetensors, the latter diffed against
the former at cosine 0.999121 with identical argmax). Residency was always the other half of
the "gpt-oss upgrade", so here is what it actually costs, read from the kernels rather than
estimated.

### It is TWO kernel changes, not one

1. **The attention sink is the easy one.** `FeatAttnSink` names a learned per-head logit that
   joins the softmax DENOMINATOR without being a key. In `cuda/decode_splitkv.cu` that is one
   extra term in `splitkv_softmax`'s `max + exp + denom` fold, plus getting the per-head sink
   array to the kernel. The math is small; the same term has to appear in every attention
   kernel that path can take (`decode_splitkv`, `attn_block`, `prefill_batched`), or the
   sink applies at decode and vanishes at prefill — a discrepancy that would read as drift.

2. **The expert kernel is the hard one, and it is the AUDITED one.** gpt-oss's expert is not
   the SwiGLU `gemv_w4a8_moe` implements: it is *clamped interleaved* SwiGLU,
   `gate·sigmoid(α·gate) · (up+1)`, clamped, **with per-expert biases**. `gemv_w4a8_moe` has
   no bias epilogue, no α and no clamp. That is a real kernel, and it lands in `moe.cu` —
   whose `moe.ptx` is the frozen, audited artifact. Changing it means the PTX-regeneration
   ritual (reproducible at nvrtc 12.6.85 via a pinned pip venv, per the CUDA notes) and
   re-auditing.

### And two environment constraints

3. **It does not fit this box.** gpt-oss-20b is ~13.8GB against the 2070's 8GB. Testing
   residency here means coupling it to the host↔VRAM expert-streaming path
   (`GOINFER_MOE_CACHE_EXPERTS`) — two hard, independent things at once, which is exactly how
   a failure becomes unattributable.
4. **Metal cannot be validated here at all.** That is the macbook's side, so the work needs
   coordinating rather than assuming.

### Recommendation

**Do not start residency as the next step.** The user-visible half — "gpt-oss is GGUF-only" —
is now closed, and it closed cheaply. What remains is a multi-part kernel effort against the
audited PTX, on a box that cannot hold the model without a second subsystem, for a backend
whose other target cannot be tested here.

Better next candidates, in the same spirit of "what do users hit first":
- **gpt-oss-120b** through the new safetensors path (36 layers, 128 experts) — the loader is
  written; this is a download and a run, and the layer-slice trick makes even a 120b oracle
  affordable (6 of 41 shards sufficed for Qwen3-Next-80B).
- **A dense CPU-only target for the CPU BlockSpec**, where batched verify should actually
  amortize — the Laguna measurement (0.82×) was specifically a sparse-MoE result.

If residency is wanted anyway, do it as its own task with the attention sink FIRST (small,
non-audited PTX, independently gateable on a dense gpt-oss-shaped fixture) and the expert
kernel second, rather than both at once.


### CORRECTION: residency needs THREE kernels, not two — and a bridge decision

Phase 2 scoping said "two kernel changes". Wiring the third revealed a third, found by reading
`moe_route`'s contract against `gptOssMoE` rather than by assuming the router was generic.

**The router semantics differ, and not cosmetically:**

| | gpt-oss (`gptOssMoE`) | `moe_route` (audited `moe.ptx`) |
|---|---|---|
| bias | added to the logits — affects **selection AND the emitted weight** | `sel = score + bias`, but the weight is `score[best]` — **selection only** |
| weights | **softmax over the top-k logits** | the selected score, then optional renormalize |

`moe_route`'s own comment is explicit that the bias "steers SELECTION only, never the weight",
which is right for DeepSeek/GLM and wrong for gpt-oss. Running gpt-oss through it would produce
plausible mixing weights that are simply not this model's — a silent quality loss, not a crash.

**This does NOT force an audited-PTX change.** The same escape used for the activation applies:
a `route_gptoss` kernel alongside `glu_quant_gptoss` in `gptoss_act.cu`, leaving `moe.ptx`
untouched. Three family-specific kernels in one family-specific module is a coherent boundary.

**The other open question is the bridge.** `decodeRunnerEligible` lists gpt-oss with
qwen35/llama4 as "own forward, not yet bridged". But `forward_gptoss.go` says its layer
skeleton is "otherwise plain pre-norm (NormPre2): everything else — embedding, norms,
residuals, final head — is the shared path", so the ONLY departures are the sink, the clamped
activation and the router. With all three as kernels, gpt-oss looks bridgeable in a way
qwen35 (Gated DeltaNet) and llama4 (iRoPE) are not — but "looks bridgeable" is a hypothesis,
and the flip that admits it must not land before a gate that runs it end to end.

**Sequencing that follows from this:** the router kernel, then a tiny-fixture parity gate with
admission behind an opt-in, then the admission decision on that gate's evidence. The repo's own
warning applies and is worth quoting: *"Do not re-gate a family on an env var without a gate
behind it; the flag reads as caution and functions as a coverage hole."*
