# Route C: Cloudflare's Clef and Clef-flash (D10–D14) — 2026-10-01 to 2026-10-04

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.
>
> **Carved out of [`docs/tasks/task-constrained-confidence.md`](../tasks/task-constrained-confidence.md) on 2026-10-04**, verbatim, when D10–D14 were all graded. The parent keeps
> the open items (below it, "Route C — what is still open"). Struck at archival: the line "a site decision-model tag goes on Clef-flash only after D13's fidelity gate
> passes", which the owner replaced on 2026-10-03 (the tag may go on at `int8int8` as his decision, not as a gate pass; it waits for a release and a registry entry).
> The int4 findings below are superseded by the owner's 2026-10-03 decision that a Clef model is not served at int4 (`quant=int4` is refused at load).

### Route C: Cloudflare's Clef (D10–D14, added 2026-10-01)

**Why it is here.** Cloudflare released Clef and Clef-flash on 2026-10-01: Apache-2.0 decision models that speak
TypeSafe's `/v1/systemone` API, which D5 already serves. Both are on Hugging Face, and both sit on backbones goinfer
already loads:
- **Clef:** Qwen3.8-27B, `model_type: qwen3_5`, 64 layers, hidden 5120.
- **Clef-flash:** Qwen3.5-9B.

What they add is a learned **joint head** that scores every option of every question together.

It is a different, newer design from autotrust's JEV students (D2–D4). It is also the first one a well-known
publisher maintains, which makes it the more likely thing a goinfer user asks for by name.

**What the release is (read 2026-10-01 from the repo files and Cloudflare's blog; D10 re-verifies):**
- **Backbone.** Cloudflare trained a frozen Qwen backbone with rank-256 adapters, but both repos ship **full
  safetensors shards and no adapter files**. Clef's shards total about 55 GB in bf16. The adapters appear to be
  merged, so **the checkpoint loads as a plain `qwen3_5` model** and needs nothing from D3. D10 confirms this.
- **Head** (`joint_head.safetensors`, 256 MB, and `joint_head_config.json`: width 1024, 2 routing layers, 4 decoder
  layers, 16 heads, feed-forward 4096), implemented in the custom module `joint_schema_model.py`:
  - It reads `last_hidden_state` from the text model at **every position**, not only the last.
  - It LayerNorms and projects that to the head width.
  - It mean-pools the token spans of each question and each option.
  - It routes evidence through `EvidenceRoutingLayer`s (multi-head attention), then `TransformerDecoderLayer`s.
  - It scores each option as a prior (cosine of lexical anchors) plus `sigmoid(residual_gate)` times a joint term
    (cosine plus a residual MLP over `[field, option, field·option, |field−option|]`), with learned logit scales
    clamped at `log(100)`.
  - It applies a softmax per question. Question kinds are `noul`, `choice` and `score`.
- **One backbone pass for all questions.** The record encoder serializes the state and every schema field into one
  chat prompt, ending with an empty `<think></think>` block and `JOINT SCHEMA DECISIONS:`. So a request with five
  questions costs one prefill, not five. **D8's trigger does not apply to this route.**
- **Images and video** go in through the backbone's own vision tower. goinfer's `qwen3_5` is text-only until
  `multimodal.md` §P8a lands.
- **An independent check exists, and it is modest:** one engineer's 42 labelled agent decisions. Clef-flash scored
  66.7% against hosted Jev's 71.4%, with 83% agreement between them (a DEV Community write-up, local bf16 on an
  RTX 3090). It is a small sample, and it is not ours.

**Should goinfer support it?** Yes, scoped to text and JSON states first, with Clef-flash as the product target:
- The endpoint, the fidelity method (D6b) and the backbone family all exist.
- The new code is a seam, a small head and an encoder.
- Clef 27B runs on the Linux box's CPU (about 28 GB at int8int8; int4 is refused for Clef, D13), not resident on the 16 GB Mac or the 8 GB card.
  It is a "runs, slowly" target, stated as such.
- The risk is format churn: a custom Python module at version one. Pin the repo revision, and refuse any
  `joint_head_config.json` key or tensor shape the loader does not know.

#### D10 — prior art, reference fixture, and the encoder contract (no goinfer code)

**Status 2026-10-02: read and swept; the fixture is queued.** Record: [`decisions-d10-clef-2026-10-02.md`](../measurements/decisions-d10-clef-2026-10-02.md) (the encoder and head contract in full, the merge check, the prior-art sweep, the pins). Done: the module read at the pinned revision; the merge confirmed (every linear projection adapted including the GDN ones, `embed_tokens`/`lm_head` and every norm untouched, MTP layer dropped; **no D3**); the sweep (llama.cpp PR #29831 open and text-only, MLX and Ollama conversions exist, vLLM none); the 150 items re-expressed as Clef records and the official encoder's output committed. Queued: the f32 run (with `last_hidden_state` for 3 items) and the bf16 run (this CPU has no bf16 hardware, so PyTorch's native bf16 GEMM is 6x slower than f32; the script does bf16 GEMMs as f32 GEMMs rounded to bf16, validated to within 0.031 logit of the native path, about 50 minutes for all 150). Corrections to this section's guesses: the head is **121.8M parameters** (not about 60M), and the reference defaults to a 16,384-token context.

- **Read `joint_schema_model.py` in full, at a pinned revision.** Record, in this doc, every fact the port depends
  on:
  - **Encoder:** the exact prompt template; Python's JSON rendering of the state (`sort_keys=True` and the
    separators actually passed); how question, option and lexical-anchor spans are computed from tokenizer
    offsets.
  - **Head:**
    - which embedding table the lexical anchors average;
    - the exact layer semantics of `EvidenceRoutingLayer`;
    - `TransformerDecoderLayer`'s `norm_first` (PyTorch's default is post-norm), activation and `batch_first`;
    - that dropout is inactive at inference;
    - whether the head's input is before or after the backbone's final norm (HF's `last_hidden_state` is after).
- **Confirm the merge.** Diff a handful of Clef-flash tensors against Qwen3.5-9B. Deltas should be nonzero on the
  adapted projections. Note which projections were adapted, including the GDN ones; this is for the record only,
  since nothing needs to merge.
- **Prior-art sweep (mandatory):** Cloudflare's Workers AI page for Clef, any vLLM or SGLang integration they
  published, and whether llama.cpp, Ollama or MLX run the head. As of the blog post none was named.
- **Reference fixture.** Run Cloudflare's own code on Clef-flash on nobara (bf16, about 18 GB; CPU is fine offline).
  Use the D0 fixture's 150 items, re-expressed as Clef records, and record:
  - **the probabilities:** every per-option probability, as goldens under `testdata/decisions/clef/`;
  - **the encoder:** the full prompt string, token ids and spans for every item;
  - **the backbone's output:** `last_hidden_state` for 3 items, as intermediate goldens for D11 and D12.

#### D11 — the all-positions hidden seam

**DONE 2026-10-02.** `decoder.Model.PromptHiddenAll(ctx, prompt) ([][]float32, error)` (`decoder/prompt_hidden_all.go`): the post-final-norm hidden state at every position, K rows of HiddenDim. CPU only and it says so (it never asks a resident backend). It takes
the same paths `PromptHidden` takes, in the same order: the generic batched path (`runLayersFromEmbedN`), Qwen3.5's batched forward (`runLayersQwen35N`), and a per-token fallback with the final norm applied to a copy of each row; both batched paths already
returned post-final-norm rows, so most of the work was the gate. **Gate passed** (`TestPromptHiddenAll_matchesHF`, golden `decoder/testdata/prompt_hidden_all_golden.json` from `scripts/pin_prompt_hidden_all.py`, transformers 5.16.1, f32): per position
cosine 1.00000000 and relative L2 at most 4e-7 (the bar is 1e-5) over 117 positions per fixture on dense, MoE and `qwen3_5-tiny-normw`, on BOTH the public path and the sequential path. The same five prompts as D2's golden (asserted equal by the pin script) plus one
length-1 prompt per fixture that checks shape only (attention over one key is the identity, so it is not a bar). Also: the last row equals `PromptHidden`'s exactly; the batched path is within 1.4e-7 of the per-token one at every position (bar 1e-6), f32 and int4,
dense and MoE; bad input and cancellation are refused on both paths; the rows do not alias. **Mutation proof, four ways:** dropping or doubling the final norm in the sequential path fails the HF subtests for all three fixtures on `sequential`; dropping or
doubling it in the shared Qwen3.5 batched path fails all three on `public`. **Not done, by design:** the resident follow-up (all-position hidden states from the CUDA and Metal batched prefill), which the spec names as this route's speed lever and its own measurement.
**A defect found on the way, in D2** (below): the key fixture's weights were never committed, so D2's strongest subtest skipped on a fresh checkout.

- `Model.PromptHiddenAll(ctx, prompt) ([][]float32, error)`: D2's loop already computes every position's
  hidden state; apply the final norm to each and keep them. It is CPU-only like D2. Memory is K × H f32 (a
  4K-token prompt on the 27B is about 84 MB).
- **Gate:** per position, cosine ≥ 0.9999 and relative L2 ≤ 1e-5 against HF `last_hidden_state` at f32, on the
  tiny fixtures including `qwen3_5-tiny-normw`. The same mutation discipline applies: dropping or doubling the
  final norm must fail.
- **Follow-up, its own measurement:** exposing all-position hidden states from the CUDA and Metal residents' batched
  prefill. That is this route's speed lever, and it is not needed for correctness.

#### D12 — the encoder and the joint head, in Go

- **Encoder** (`internal/serveapp`, beside D1's decider): a port of Clef's record encoding.
  - It needs a Python-compatible JSON renderer for the state. Go's `encoding/json` differs in key order,
    separators and escaping.
  - **Gate:** prompt string, token ids and every span identical to the D10 dump on all 150 items. This is the
    same input-identity bar D6a held (150/150).
- **Head:** load `joint_head.safetensors` and its config, refusing unknown keys or shapes. Implement LayerNorm,
  linear, 16-head attention, the routing and decoder layers exactly as D10 records them, the residual scorer, the
  prior, the clamped scales and the gate. At about 60M parameters in f32 it is a negligible cost on CPU.
  - **Gate:** given the reference's own `last_hidden_state` as input (isolating the head), per-option probabilities
    within 1e-5 absolute of the reference at f32.

#### D13 — wiring and fidelity

- **Loading:** a model directory carrying `joint_head.safetensors` loads as a decision model. `/v1/systemone` (D5)
  gains `route: "clef"`, and `/v1/models` advertises it.
- **Image parts on this route** are refused with a clear error until P8a.
- **Fidelity, by D6b's method** (Clef-flash against the D10 goldens):
  - arms: f32, int8int8, int4 and q4k;
  - pre-registered: mean KL ≤ 0.01 at f32 and ≤ 0.03 at int4, top-1 agreement ≥ 98%;
  - ECE reported per arm.
- **Expect what D6b found for JEV:** quantized arms keep calibration and miss the top-1 bar. The decision-model
  default quant is the owner's call that D6b already left open. Make it once, for both heads.
- **Compare the trained routes on one fixture.** Clef-flash and JEV-9B are both 9B on `qwen3_5`, so run them on
  the same items: top-1 against gold where it exists, ECE, and agreement. **Owner decision 2026-10-03: both routes are
  kept and extended equally, so this comparison is informational and no rule hangs on it** (it was "if Clef-flash is at
  least level on both, JEV's route is kept but not extended").
- **Graded 2026-10-03** ([`decisions-d13-clef-fidelity-2026-10-03.md`](../measurements/decisions-d13-clef-fidelity-2026-10-03.md) §7), 150 records, six arms:
  - **f32 PASS** (KL below 1e-5, top-1 1.000): the port is correct.
  - **int8int8 FAIL on top-1** (KL 0.0165 inside the band, top-1 0.907); **int4 FAIL** (KL 0.051, top-1 0.840), on the CPU and on CUDA (0.044, 0.840); **q4k FAIL on top-1** (KL 0.021, top-1 0.887, bartowski's imatrix GGUF).
  - **Calibration unresolved in every arm** (each interval reaches 0); none failed it. The bf16 reference against f32 reads KL 0.00005, top-1 0.993.
  - **The default stays int8int8** under the record's rule (a top-1 miss alone does not reopen it), with the disagreement stated: 14 of 150 argmaxes change.
  - **Clef against JEV** (84 gold rows, informational): accuracy 0.690 against 0.774 and ECE 0.107 against 0.122, neither difference resolved.
  - Predictions that failed, recorded as failed: int4 inside the KL band (CPU and CUDA).
  - **Owner decision 2026-10-03: no int4 for Clef.** `quant=int4` on a Clef model is refused at load (`clefQuantRefusal`, tested through `loadDecoder`); `int8int8` stays the default and `f32` is available.
  - **Owner decision 2026-10-03: Clef-flash at `int8int8` may carry the decision-model tag, as the owner's decision and NOT as a pass of D13's gate** (int8int8 reads top-1 0.907 against the registered 0.98; the gate as written is failed on that metric and the record says so). The tag's page must show the measured figures (KL 0.0165, 90.7% agreement with the f32 reference, int4 not offered) and the single-question limit below.
  - **Not built yet:** the tag is a site change, which lands only when a release is cut (`docs/tasks/task-site-2026-09.md`, owner decision 2026-09-29), and Clef-flash is not a registry checkpoint (it is served from a local directory with `--model`). The 150 graded records are single-question; the multi-question path was then checked on real weights (5 records of 5 questions, 2026-10-03, [`decisions-d13-clef-multiquestion-2026-10-03.md`](../measurements/decisions-d13-clef-multiquestion-2026-10-03.md)): goinfer f32 matches the reference to 1.2e-06 and int8int8 reads mean KL 0.0055, both PASS on the pre-registered rule. Joint versus single was then characterized on the reference (no bar): the same five questions asked alone differ from the joint answers by mean KL 0.0135 (max |ΔP| 0.138; top-1 23 of 25, both flips near-ties), about the size of int8int8's own error, so the two modes are not interchangeable and the tag's page must not say they are.
- **Then Clef 27B on the Linux box's CPU,** at f32 against a smaller reference set. Report its speed as measured,
  with the machine named.
  **Done 2026-10-03 night** ([`decisions-d13-clef27b-2026-10-03.md`](../measurements/decisions-d13-clef27b-2026-10-03.md) §7), at int8int8 against a bf16 reference (a 27B does not fit at f32 on 62 GB), on 30 records: KL 0.0121, top-1 0.867, **FAIL on top-1** as predicted, calibration unresolved; 160 ms per input token on nobara's CPU (2.96 times Clef-flash's), 46 s per record. No tag decision covers the 27B.

#### D14 — speed, and the multi-question shape

- **Projection first,** from D7's model: one prefill of the whole record, plus the head, against JEV's one prefill
  per question.
- **Measure:** latency for 1 and 5 questions about one state, Clef-flash against JEV-9B, on the Mac CPU, quiet box,
  paired. This is the number that says whether Clef makes D8 unnecessary for TypeSafe-shaped requests.
  **Measured 2026-10-03 on nobara's CPU** (amendment 2b moved it there;
  [`decisions-d14-clef-speed-2026-10-03.md`](../measurements/decisions-d14-clef-speed-2026-10-03.md) §7): every cell as
  projected; five questions, Clef is 1.14x faster at K = 256 and 1.75x at K = 1,024. The D8 rule read 2.01 at K = 256,
  ambiguous by the rule, and **the owner decided: D8 is unnecessary for Clef's five-question shape** (one backbone pass
  already scores every question).
- **Resident:** D11's follow-up now exists for CUDA ([`decisions-d11-resident-hidden-2026-10-03.md`](../measurements/decisions-d11-resident-hidden-2026-10-03.md): the all-positions residual comes back from the device; the CUDA int4 arm measured 3.2 ms per token and was graded for fidelity in D13: KL 0.044, top-1 0.840), so the CUDA speed cells can be registered; Metal and WebGPU do not implement the seam yet. Each cell gets its own pre-registered band.
- **Docs (D9's list):** the route, the per-arm fidelity figures, the measured latency with its machine, and the
  §2 caveat. A site decision-model tag goes on Clef-flash only after D13's fidelity gate passes.
