# Task: one forward architecture — models as operation lists, one ragged forward pass (OP0–OP6, RF0–RF6, HL) — 2026-10

> **Status: SCOPED 2026-10-13. Not started.** Written on the owner's ask after a Cowork conversation the same day ("we have
> spent a bunch of time on decode, prefill and now kv cache. in terms of time spent by goinfer on a query, where is there any
> room to re-architect to a cleaner internal structure that could simplify and improve either model coverage or (most
> importantly) performance?", then "yes one consolidated doc, put it into the docs/tasks dir, and also update kv doc").
>
> **Its companion is `docs/tasks/task-kv-architecture-2026-10.md`**, the storage half. This doc is the compute half: how a
> model is described to the GPU backends (Track OP), and how a forward pass is called (Track RF). The third item from the
> same conversation, one residency manager for KV blocks and MoE experts, went into the KV doc as KV11.
>
> **The facts about the code were read at main `9593bd70` (2026-10-13).** This doc took two readings: the tokenizer's cost,
> and every family's decline reason on each GPU backend. Both are exploratory, not registered, and their programs and output
> are in `docs/measurements/forward-arch-baseline-2026-10-13.md`. Every gate below is a sketch: whoever starts a phase
> registers its gates first (rule, void conditions, stopping rule, cost basis), as `CLAUDE.md` requires.

## 0. Summary

**Where a query's time goes.** Almost all of an agent turn is prefill and decode.
- **Handling the request is not where the time is.** Tokenizing 25,327 tokens took 24.1 ms on the M1 Pro, 0.95 µs a token.
  The same Mac prefills a 3,900-token prompt on the 1.5B at about 1.1 ms a token.
- **The host's gap between decode tokens is small.** It is 0.58–0.66 ms on Metal, and the run-ahead chain already hides it
  where it applies.
- **What costs most is falling off the fast path.**
  - A model a GPU backend cannot build resident decodes on the CPU, measured about 15× slower.
  - A request shape a backend has not built for a family runs a slower path. A Claude Code first request prefilled token by
    token for about 4.5 minutes on the 8 GB card before KV3 (the KV doc, §8 Q2).

**Two structural causes.**
1. **The GPU runners are uniform-layer runners** (`docs/ARCHITECTURE.md`'s phrase). A family runs resident only if the backend
   implements every feature it needs and its layers all have one shape.
   - The hardware matrix has 43 families. In the model-free default configuration it reads, 10 decode on the CPU under CUDA,
     12 under Metal and 16 under WebGPU.
   - The reasons are a short list (§3.2): missing operations, four families whose layers differ in shape, and two gates.
   - Gemma 2 is the 8th most-pulled model in Ollama's library (34.4M pulls). It lacks exactly one operation on every GPU
     backend: attention softcapping.
2. **The resident seam has one entry point per request shape.** CUDA's resident has 18 methods that run the model, Metal's 18
   (counting its run-ahead chains) and WebGPU's 7. The decoder type-asserts a resident against 34 optional interfaces.
   - Every family × shape × backend combination nobody has built falls back to something slower.
   - Different shapes run different kernel classes. That caused MC4's identity failure on a reused prefix, and KV0's 16-row
     rule.

**What this doc proposes.**
- **Track OP:** a model becomes a per-layer list of operations, and backends implement operations, not families.
- **Track RF:** one ragged forward call carries rows from any number of conversations, each row with its own input and
  requested output. Decode, verify, a prefill chunk, an image turn and a batched step all become shapes of it.
- **HL:** a small host-loop change, so Metal's run-ahead survives tool and thinking-budget turns, and CUDA gets one.

| phase | what | backends | size | needs |
|---|---|---|---|---|
| OP0 | decline reasons in the hardware matrix; features mapped to operations; prior art | — | S | — |
| OP1 | attention softcapping: Gemma 2 on every GPU | CUDA, Metal, WebGPU | M | — |
| OP2 | the next missing operations by demand: MLA on Metal, SSM on CUDA and Metal, KDA | per operation | M–L each | OP0 |
| OP3 | the layer program, built from the `Architecture`; the CPU executes it bit for bit | CPU | L | OP0 |
| OP4 | residents walk the program; admission by operation; fusion tables; one memory planner | CUDA, Metal, WebGPU | L | OP3 |
| OP5 | the families whose layers differ: Llama 4, LFM2.5, Laguna, Spark-X2.5, Granite-4.0-H | per family | M each | OP4 |
| OP6 | Phi-3/4 on the fast paths: Metal's default load, a q4k prefill, WebGPU | Metal, CUDA, WebGPU | M | — |
| RF0 | measure first: per-turn phase timing (telemetry X1); a shape × family × backend census | — | S | — |
| RF1 | one call shape in the decoder, over today's entry points; no kernel change | all resident | M | RF0 |
| RF2 | ragged attention over paged KV: decode, verify and prefill rows in one pass | CUDA | L | KV4, RF1 |
| RF3 | mixed steps: decode rows and prefill-chunk rows share a pass | CUDA | M | RF2 |
| RF4 | the same on Metal | Metal | L | KV5, RF3 |
| RF5 | retire the old entry points and interfaces; images and hidden outputs as row attributes | all resident | M | RF4 |
| RF6 | speculation inside a batch | CUDA, Metal | M | RF3 |
| HL | run-ahead under a gated logit processor (Metal); a run-ahead chain on CUDA | Metal, CUDA | S–M | — |

## 1. Where a query's time goes

| phase | what it costs | evidence | re-architect? |
|---|---|---|---|
| parse, template, tokenize | tokenizing: 24.1 ms for 25,327 tokens (best of 5, 0.95 µs a token); the tokenizer loads once, in 89 ms | `docs/measurements/forward-arch-baseline-2026-10-13.md` | **no** |
| prefill | the largest share of an agent turn. Claude Code's first request is 19,692–20,199 tokens, and its thread switches re-prefilled 8,000 to 17,000 tokens each, 78% of its request time | `docs/measurements/kv0-traffic-2026-10-13.md` | the KV doc, RF |
| decode | one GPU forward per token | `docs/benchmarks.md` | OP, RF |
| host gap per token | 0.58–0.66 ms on Metal; the run-ahead chain hides it where it applies (1.010–1.160× decode) | `CHANGELOG.md`, "the next token's forward is queued on the GPU" | HL (small) |
| **fallbacks** | a model on the CPU decodes about 15× slower; a declined shape runs a slower path (above) | `internal/loadflags/loadflags.go` (`-ctx`'s help); the KV doc §8 Q2 | **OP, RF** |

- **There is no per-phase timing on the wire today.** `docs/tasks/task-turn-telemetry-2026-09.md` (X1–X8) is scoped and
  unstarted, so the split of a real turn is inferred, not read. RF0 builds its X1.
- **Fallbacks are being removed one at a time.**
  - KV3 lifts CUDA's 12,288-key bound on batched prefill for the layers its fused attention serves. In a one-run smoke on its
    branch, a request of Claude Code's size answered in 25.1 s. Its graded run is queued for the night of 2026-10-13.
  - A tiled exact attention kernel for Gemma's head dims of 256 and 512 is built on the `kv3-tiled-2026-10` branch. Its night
    gates are queued.
  - Per KV3's report, these still decline past the bound: MLA, the all-logits, embedding and residual tails, and image blocks.
  - Phi-3 at `--quant q4k` decodes resident on CUDA, but its batched prefill declines to the sequential path
    (`docs/benchmarks.md`).
  - Each of these was found by someone hitting it. RF removes the class: with one call path, no shape is left without a fast
    path.

## 2. What must not change

1. **Bit-identity.** Each migration step is bit-identical to the code it replaces. That holds per backend, on every family
   fixture and real checkpoint the backend runs: the parity manifest plus each resident's identity suites.
2. **Hand-written kernels stay.** This is not a graph compiler. Fusions stay explicit and per backend, each gated against the
   unfused operations it replaces.
3. **A backend that cannot run something declines by name, and never runs it wrong.** Admission is
   `decoder/residency.go:Model.residentAdmission`, in three steps:
   1. the runner shape (`decoder/residency.go:Model.decodeRunnerDecline`);
   2. `decoder/features.go:residentGateReason`;
   3. the backend's own build, which declines through `decoder/residency.go:DeclineResident`.
4. **The CPU stays the reference.** The descriptor-driven loop and the own-forward families are the oracle. The layer program
   reproduces them exactly before anything else moves.
5. **The Hard API tier and the two-step tag.** The backends are separate modules, so every seam change lands with
   `RELEASING.md`'s two-step tag.
6. **The run budget.** Heavy gates go on the night queue under the timing lock.

## 3. Today

### 3.1 How a model is described, and how a GPU backend runs it

- **The CPU is descriptor-driven** (`docs/ARCHITECTURE.md` §1, "One decoder, read from a descriptor").
  - `decoder/registry.go:resolveArchitecture` turns a checkpoint's config into a `decoder/arch.go:Architecture`. Every
    `Architecture` passes `decoder/registry.go:Architecture.validateResolved`.
  - A family's specifics are fields: norms, attention options, position scheme, FFN kind, MoE configuration.
  - Nine families run their own layer loop (`decoder/arch.go:ownForwardFamily`, the `ownForwards` table, and twelve files named
    `forward_*.go` in `decoder/`). They are:
    - the recurrent hybrids: Gated DeltaNet, short conv, two Mamba-2 families, and KDA with MLA;
    - MLA itself (DeepSeek);
    - three softmax families whose layers differ: Gemma 4, Llama 4 and gpt-oss.
  - Six of the nine pass the runner-shape check through a per-family bridge; Ling 3.0 then declines on KDA. Llama 4 and LFM2
    have no bridge, and Granite-4.0-H waits on a bring-up gate.
- **The GPU backends are uniform-layer runners.**
  - Each builds a per-layer struct from the `Architecture` and branches on its flags (`cuda/backend.go:cudaBackend.BuildResident`,
    `metal/backend.go:metalBackend.BuildResident`, `gpu/residency.go:webgpuBackend.BuildResident`).
  - CUDA's `cuda/resident.go:cudaLayer` has 47 lines of fields. Metal has `metal/model.go:residLayer` beside its own DeltaNet
    and MoE layer types. WebGPU has `gpu/decoderunner.go:runLayer`.
- **Admission** is the chain in §2, item 3.
  - An own-forward family needs a per-family bridge to pass the runner-shape check.
  - Every `ResidentFeature` the architecture needs must then be in the backend's declared set
    (`decoder/features.go:ResidentBackendFeatures`).
  - The backend's own build may still decline, for memory for example.
- **Size**, in non-test Go:
  - about 45K lines in `decoder/`;
  - 13.5K in `cuda/`, plus 35 `.cu` files;
  - 18K in `metal/`, whose shaders live in its Go files;
  - 19K in `gpu/`.

### 3.2 Why families decode on the CPU (read 2026-10-13)

This is the census behind `docs/hardware-matrix.md`'s CPU cells. For every registry family on each GPU backend, it applies the
runner-shape check and then `decoder/features.go:residentGateReason`, in the model-free default configuration the matrix
reads. "—" means the family runs resident there. The program and its raw output are in the baseline record.

| family | CUDA | Metal | WebGPU |
|---|---|---|---|
| Gemma 2 | attn-logit-softcap | attn-logit-softcap | attn-logit-softcap |
| DeepSeek-V2, DeepSeek-V3, Kimi K2 | — | mla | — |
| Ling 3.0 | kda | kda, mla | kda |
| Nemotron-H | non-gated-mlp, ssm | ssm | — |
| GPT-2 | learned-pos, non-gated-mlp | — | layer-norm, learned-pos |
| Command-R, Command-R7B | — | — | layer-norm, pairwise-rope, parallel-block (Command-R7B also nope) |
| GLM-OCR | — | — | pairwise-mrope, pairwise-rope |
| Ministral 3 | — | — | attn-temp |
| Olmo 3, Olmo Hybrid | — | — | post-only-norm, qk-norm-whole (Olmo Hybrid also nope) |
| SmolLM3 | — | — | nope |
| Llama 4, LFM2.5 | runner shape: own forward, not bridged | the same | the same |
| Laguna, Spark-X2.5 | runner shape: local and global RoPE tables of different lengths | the same | the same |
| Granite-4.0-H | a bring-up gate (`GOINFER_SSM_RESIDENT`), not a missing kernel | the same | the same |
| Phi-3, Phi-4 | the activation hazard: int8 activations round its outliers away | the same | the same |

**Four kinds of decline:**
- **A missing operation** on one or more backends: softcapping, MLA, SSM, KDA, and a set of small operations WebGPU lacks.
- **Layers of different shapes.**
  - Llama 4 and LFM2.5 run their own forwards, which no GPU runner bridges.
  - Laguna and Spark-X2.5 use the generic loop, but give local and global layers RoPE tables of different lengths.
    `decoder/residency.go:Architecture.ropeResidentCompatible` refuses that.
  - No uniform-layer runner can express these, however many operations it has.
- **A gate.** Granite-4.0-H waits on its parity bring-up. WebGPU already declares its SSM feature (`docs/hardware-matrix.md`'s
  note).
- **A numerics guard, read in one configuration.** Phi-3/4's activation outliers are rounded away by per-row int8 activation
  scales (`decoder/actquant_hazard.go:ActivationQuantHazard`). The matrix shows CPU everywhere, but the loader picks a safe
  configuration (`internal/modelload/modelload.go:activationSafeQuant`):
  - **On CUDA,** a Phi-3/4 GGUF loads at `--quant q4k` and runs resident: 112.6 tok/s at depth 128 on the 8 GB card
    (`docs/measurements/peer-sweep-2026-09-29.md`). Its batched prefill declines to the sequential path.
  - **On Metal,** it runs resident at an explicit `--quant q4k`, a lane built 2026-10-08 (`docs/tasks/task-metal-q4k-2026-10.md`).
    The default load still picks int8int8 with per-32 scales, which declines to the CPU.
  - **WebGPU** has neither.

**How much it matters.** Pulls in Ollama's library, in millions (`docs/measurements/ollama-coverage-2026-10-09.md`):
- **gemma2, 34.4:** Gemma 2 decodes on the CPU on every GPU backend.
- **phi3, 18.2, and phi4, 7.7:** Phi-3/4 decode on the CPU under WebGPU, and under Metal without `--quant q4k`.
- **deepseek-v3, 3.8, and deepseek-coder-v2, 3.2** (the V2 architecture): DeepSeek-V2/V3 and Kimi K2 decode on the CPU under
  Metal.

### 3.3 How a forward pass is called

- **The seam** is `decoder/residency.go:ResidentForward`, plus 34 optional interfaces the decoder finds by type assertion on a
  resident. 28 of them are declared in `decoder/residency.go`. They include:
  - **the prefills:** `Prefiller`, `ResidentPrefillKV`, and the image, m-RoPE and deepstack prefills;
  - **the outputs:** `ResidentHiddenLast`, `ResidentResidualAll`, `ResidentGreedy`, `ResidentSample`, `ResidentTopK`, and the
    greedy and sample chains;
  - **batching:** `ResidentBatchStepper`;
  - **the reporters:** `PrefillPathReporter`, `VerifyPathReporter`, `ResidentFastPrefill`, `ResidentPrefillKernelFloor`.
- **The methods that run the model:**
  - **CUDA, 18:** `Forward`, `ForwardArgmax`, `ForwardMRoPE`, `ForwardN`, `ForwardNoLogits`, `ForwardSample`, `ForwardTopK`,
    `HiddenLast`, `PrefillImageBlocksLast`, `PrefillImageLast`, `PrefillLast`, `PrefillLastN`, `PrefillLastNArgmax`,
    `PrefillMRoPEDeepstackLast`, `PrefillMRoPELast`, `PrefillSeedArgmax`, `ResidualAll`, `StepBatch`.
  - **Metal, 18:** the same set less `ForwardArgmax`, `ForwardTopK`, `PrefillLastN` and `PrefillSeedArgmax`, plus the greedy
    and sample chains (a start and a next for each).
  - **WebGPU, 7:** `Forward`, `ForwardMRoPE`, `ForwardN`, `ForwardNoLogits`, `ForwardSample`, `PrefillLast`,
    `PrefillMRoPELast`.
- **Kernel classes differ by shape.**
  - On CUDA, a pass runs the fast kernels when it has at least 16 rows and its prompt is at least 512 tokens (KV0's class rule;
    `ResidentFastPrefill`). Anything else runs the exact ones. Metal has its own floor (`ResidentPrefillKernelFloor`).
  - The consequences so far:
    - MC4's first rerun failed identity on exactly this. A reused prefix had been computed by a different kernel class than a
      cold prefill would use (`docs/measurements/mc4-candidate-cuda-2026-10-01.md`, "ROOT CAUSE (2026-10-02 evening)").
    - KV0 had to state the class rule and put its 16-row condition into the prefix key.
    - KV3's chunker had to be shaped around it.
- **What one entry point per shape blocks:**
  - **Speculation inside a batch.** It is an owner decision in `docs/tasks/parked/task-concurrency-2026-09.md`: "CUDA copy L =
    1.54×; needs a ragged-batch kernel, no design".
  - **Removing the pause a newcomer's prefill causes in other conversations' decode.** Chunked prefill cut the decoders'
    longest stall from 5.38 s to 1.23 s (`docs/measurements/chunked-prefill-2026-09-27.md`). It cannot remove the stall,
    because a chunk and a decode step are different calls.

## 4. Track OP — a model is a list of operations

### 4.1 Design

- **The layer program.** At load, the `Architecture` produces a per-layer list of operation instances.
  - A dense layer, for example: `norm(rms, +1)`, `proj(qkv, int4)`, `qk-norm(per-head)`, `rope(partial, base, yarn)`,
    `attention(window W, softcap c, sinks, temperature)`, `out-gate`, `proj(o)`, `residual`, `norm`,
    `ffn(gated swiglu | moe{router, experts, shared})`, `residual`.
  - The recurrent mixers are operations too: `deltanet`, `mamba2`, `kda`, `short-conv`. So are per-layer embeddings and KV
    sharing.
  - A layer can differ from its neighbours, which is the point.
- **Operation kinds are a closed, versioned set.** Most `ResidentFeature` values already name an operation kind or one of its
  parameters: softcap, sinks, temperature, NoPE, partial or pairwise RoPE, layer norm, parallel block, post-only norm,
  whole-vector QK-norm, MLA, SSM, DeltaNet, KDA, short conv, MoE.
- **A backend declares the operation kinds it implements,** and runs a program by walking it layer by layer.
- **Fusions are explicit.**
  - A backend's fusion table claims sequences of operations. For example, CUDA's `fused_rms_qkv` would claim
    `[norm, proj(qkv)]`. It folds the RMSNorm and the int8 activation quantization into the Q/K/V GEMV (`cuda/fused_qkv.cu`).
  - Each fusion has an identity gate against the operations it replaces.
  - There is no automatic fusion and no general compiler. goinfer's speed is in hand-written kernels, and its trust is in
    gating them.
- **Admission becomes:** every operation kind in the program is implemented, and the memory fits. This replaces the family
  switch in `decodeRunnerEligible` and the feature sets.
- **The CPU executes the same program.** Its own-forward code becomes operation implementations. `deltaNetCore`, `mamba2Step`
  and `kdaRecurrentStep` mostly already are.
- **Memory planning.** A program lets each backend size its scratch once, per operation and per chunk.
  - The recompute audit's same-work-twice class (`docs/tasks/task-recompute-audit.md` §5, R-11–R-25) was found by hand. A
    planner makes avoiding it a property.
  - The planner prices through the KV doc's one sizing function (its §5.10), so admission has one source.

### 4.2 What it buys

- **Coverage per operation, not per family.** An operation built once per backend lights up every family that uses it.
  - Softcapping puts Gemma 2 on all three GPUs.
  - MLA on Metal brings DeepSeek-V2, DeepSeek-V3 and Kimi K2 there.
- **Performance where it matters most.** A family leaving the CPU fallback gets roughly the 15× decode difference. That is the
  largest single change available to a user of that family.
- **New families get cheaper.** A family made of existing operations needs an adapter (descriptor → program) and a parity gate,
  not code in three runners.
- **The families whose layers differ stop being special.** Each becomes a program of mostly existing kinds:
  - **Llama 4:** chunked attention on its RoPE layers, NoPE layers, L2 QK-norm, attention temperature;
  - **LFM2.5:** a gated short convolution on 22 of its 30 layers;
  - **Laguna:** per-layer query heads and RoPE geometry;
  - **Spark-X2.5:** per-layer RoPE geometry;
  - **Granite-4.0-H:** Mamba-2 and attention layers.

### 4.3 How close the code is

- **Close on knowledge.** The `Architecture` holds every fact the program needs.
  - The per-layer predicates exist: `isLinearLayer`, `isMambaLayer`, `isConvLayer`, `LayerIsLocalResident`, and Gemma 4's
    per-layer geometry.
  - So does the feature list.
  - The residents already branch per layer on flags.
- **Missing:**
  - an explicit program object;
  - runners that walk layers of different kinds;
  - backend operation tables in place of feature sets;
  - fusion tables in place of fusions hard-coded into each runner's build.

### 4.4 Phases

- **OP0 — the decline table, the operation census, prior art (S).**
  - `docs/hardware-matrix.md` prints each declining cell's reason: the operations missing, or the gate. It is generated from
    `residentGateReason`, as the matrix already is.
  - A row whose default load runs resident in another configuration says so: Phi-3/4 at `--quant q4k` on CUDA.
  - Map every `ResidentFeature` to an operation kind or parameter.
  - Read the prior art (§10).
- **OP1 — attention softcapping on every GPU (M).** Worth doing before the program exists.
  - Softcapping is a parameter of every attention kernel variant per backend: decode, split-KV and flash-decode, fused and
    exact prefill, and the MC3 rows.
  - **Gates:**
    - identity on every family without softcap;
    - Gemma 2 parity against the CPU (the tiny fixture plus the real 2B and 9B);
    - a decode peer cell against Ollama on each GPU box.
- **OP2 — the next operations, in order of demand (M–L each).** Each has its own gates, as OP1 does.
  - MLA on Metal: DeepSeek-V2/V3, Kimi K2.
  - SSM on CUDA and Metal: Nemotron-H, and Granite-4.0-H once its gate lifts.
  - Then KDA.
- **OP3 — the layer program, on the CPU (L).**
  - Build the program for every registry family. The CPU executes it, and its logits are bit-identical to today's generic loop
    and own forwards on the whole parity manifest.
  - **Can the gate go red?** Plant a dropped operation, two swapped operations, and a parameter off by one.
- **OP4 — the residents walk the program (L).** CUDA first, then Metal, then WebGPU.
  - It brings admission by operation, per-backend fusion tables with each fusion gated, and the memory planner.
  - **Gates:**
    - every family's identity suite unchanged on each backend;
    - speed within 0.99 at depth 128 and 3,900 (night queue).
- **OP5 — the families whose layers differ, onto the GPU (M each).** Llama 4, LFM2.5, Laguna, Spark-X2.5, and Granite-4.0-H
  after its gate. Each brings only its missing operations.
- **OP6 — Phi-3/4 on the fast paths (M).** This may become its own task (E5).
  - Metal's default load becomes `--quant q4k`, where the loader now picks int8int8, once the lane's peer cell is read.
  - A batched prefill for the q4k kinds, on CUDA and Metal.
  - Per-32 activations or a q4k lane on WebGPU.

## 5. Track RF — one ragged forward pass

### 5.1 Design

- **One call per backend: `Step(batch)`.** A batch is a list of rows, and each row has:
  - **a sequence:** its KV block tables (KV4/KV5), or a slot before then, plus its position;
  - **an input:** a token id, an embedding row, or an image-embedding row with its m-RoPE triple and image-span mark;
  - **an output request:** nothing, logits, an argmax, a sample with its parameters, top-k, or a hidden state at a layer.
- **Every shape is a batch:**

  | today | as a batch |
  |---|---|
  | `Forward`, `ForwardArgmax`, `ForwardSample`, `ForwardTopK` | 1 row, with an output kind |
  | `ForwardN` (verify) | k rows of one sequence |
  | `PrefillLast` and its variants | a chunk of rows; the last asks for logits or an argmax |
  | image and m-RoPE prefills | rows with embeddings and positions |
  | `HiddenLast`, `ResidualAll` | output kinds |
  | `StepBatch` (MC3) | B rows, one per sequence |
  | a mixed step (new) | a newcomer's chunk plus the other conversations' decode rows |

- **Ragged attention reads paged KV through block tables.** This is the core new kernel, in the shape of FlashAttention's
  variable-length interface and FlashInfer.
  - It arrives with KV4/KV5. KV4 puts every KV kernel behind a block table and gives MC3's step a table per row (the KV doc,
    KV4a).
  - RF2 adds rows of different kinds and counts per sequence in one pass.
- **One kernel-class rule, set by what is invariant: the pass, not the whole prompt.**
  - KV0 measured chunk and split invariance on CUDA and WebGPU. Chunked prefill measured Metal's chunk invariance.
  - The KV manager keys every block by the kernel class that wrote it (the KV doc, §5.5–5.6).
  - So a row's class cannot depend on how it was batched.
- **A scheduler in the decoder** gives each step a token budget: decode rows first, then prefill rows to fill it. This is
  Sarathi-Serve's stall-free batching. MC3's batcher becomes this scheduler.

### 5.2 What it buys

- **No more cliffs.** Every family a backend runs resident runs every shape through one path: long prompts, images,
  all-logits requests, hidden-state routes.
- **No more stalls.** A newcomer's prefill shares passes with the others' decode instead of pausing them. Decode tokens ride
  along in prefill passes nearly free.
- **Speculation inside a batch** becomes possible.
- **One identity story:** a handful of interfaces instead of 34, and far fewer kernel variants to keep equal.

### 5.3 How close the code is

- **Exists:**
  - MC3's `StepBatch`: B sequences, one row each (`cuda/batchstep.go:cudaResident.StepBatch`,
    `metal/batch.go:metalResident.StepBatch`);
  - `ForwardN`: one sequence, k rows;
  - chunked prefill;
  - Metal's row kernels (`metal/batch_rows.go`). CUDA's step runs attention row by row
    (`cuda/batchstep.go:cudaResident.stepAttnRows`).
- **Missing:**
  - several rows per sequence in a step that also holds other sequences;
  - ragged attention;
  - per-row output kinds;
  - images as row attributes;
  - the token-budget scheduler.

### 5.4 Phases

- **RF0 — measure first (S).**
  - Per-turn phase timing on the wire: `docs/tasks/task-turn-telemetry-2026-09.md`'s X1, built here.
  - A census of shape × family × backend: which combinations decline today, to what, and at what cost.
  - Prior art (§10).
- **RF1 — the call shape, over today's entry points (M).**
  - Every caller goes through `Step(batch)`, and an adapter turns each batch into today's calls.
  - No kernel changes, so identity is unchanged by construction.
  - It can start any time.
- **RF2 — ragged attention on CUDA (L), after KV4.** Decode, verify and prefill rows in one pass.
  - **Gates:**
    - identity per shape wherever the kernel class is unchanged, and a registered fidelity gate where it changes;
    - speed per shape:
      - decode at depth 128 to 8,000: 0.98 or better;
      - time to first token at K = 512 and 3,900: 0.97 or better;
      - the MC3 aggregate: 0.98 or better.
    - **Kill:** if decode at depth stays under 0.98 after the registered tuning budget, keep the old decode call beside the
      ragged one.
- **RF3 — mixed steps on CUDA (M).** The token-budget scheduler.
  - **Gates**, on chunked prefill's harness (`scripts/bench_prefill_stall.py`), with bands registered against chunked
    prefill's:
    - the longest decode stall during a newcomer's prefill;
    - the newcomer's time to first token;
    - the solo guard.
- **RF4 — Metal (L),** after KV5 and RF3.
- **RF5 — retire the old entry points and interfaces (M).** Images, m-RoPE and hidden outputs become row attributes. WebGPU
  keeps the RF1 adapter, since D3 keeps its slots.
- **RF6 — speculation inside a batch (M).** The parked "spec inside a batch", on the ragged path, with MC4's identity gate.

## 6. HL — the host token loop (small)

- **Today:** Metal's run-ahead chains run only when no logit processor is attached
  (`decoder/residency.go:ResidentGreedyChain`, `decoder/residency.go:ResidentSampleChain`,
  `decoder/model.go:Model.generateInto`).
  - A gated processor already keeps the single-token fast path armed. Only the chains refuse it, because "a gated one would need
    the full row mid-chain".
  - In auto mode, a turn that declares tools attaches the lazy tool union, a gated processor that is on by default
    (`internal/serveapp/tools.go:constrainToolUnion`). A turn with a thinking budget attaches another
    (`internal/serveapp/budget.go`). So agent turns lose the chain.
  - CUDA has no chain at all.
- **The change on Metal:** run ahead while the gate is closed.
  - When the gate opens at token t, the host reads step t's full row and applies the mask.
  - If the masked pick differs from the device's, the host discards the forward already queued for t + 1, truncates the one
    position it wrote, and redoes it.
  - That costs at most one wasted forward per opening, and openings come about once per tool call.
- **The change on CUDA:** add a chain. `ForwardArgmax` already picks on the device, and decode already replays CUDA graph
  segments.
- **Gates:**
  - tokens identical to today's;
  - speed on the prose part of tool turns (Metal 1.5B and 7B; CUDA 1.5B and 7B), registered against the 1.010–1.160× the
    chain gave plain turns.
- **Size:** S–M.

## 7. Order, and what each step is worth alone

- **Now, independent of everything:** OP0, OP1 (Gemma 2 on every GPU), RF0, HL, and OP6's Metal default (E5).
- **With the KV track:** RF1 any time; RF2 after KV4; RF4 after KV5; RF3 and RF6 after RF2.
- **The structural work** comes after OP1 and OP2 have shown the operation boundary holds: OP3, then OP4, then OP5.
- **Every phase is bit-identical where it says so, and ships alone.** A track that stops early leaves the tree consistent,
  because the adapters keep the old paths working.

## 8. Owner decisions

| id | question | recommendation |
|---|---|---|
| **E1** | Hand-written kernels with explicit fusion tables, and no general graph compiler? | Yes. |
| **E2** | OP1 (Gemma 2's softcapping) before the program exists? | Yes: it is the most-pulled gap, and it tests the operation boundary. |
| **E3** | The kernel-class rule for RF: set per pass, with invariance measured? | Yes, after RF0 and KV0's data. |
| **E4** | WebGPU in RF? | The RF1 adapter only, consistent with D3. |
| **E5** | Phi-3/4 (OP6): here, or its own task? And `--quant q4k` as Metal's default load once its peer cell is read? | Its own small task; yes to the default once the cell is read. |
| **E6** | RF3 mixes one user's conversations in a step, the same step continuous batching takes for many users. Keep it framed as stall removal for one user's agents, with continuous batching for many users still parked? | Yes. |
| **E7** | Queue position? | OP1, RF0 and HL now; RF1 alongside KV2; the rest after KV4. |

## 9. Risks

| risk | containment |
|---|---|
| **Rewriting around hand-tuned kernels** | no compiler; kernels keep their code; per-family identity gates; one backend at a time; adapters keep the old paths until they are retired |
| **Ragged attention slower than today's specialised kernels** | speed bands per shape; RF2's kill rule keeps the old decode call |
| **Identity across shapes** | invariance measured (KV0); one class rule; the class recorded per block |
| **Seam churn across modules** | the two-step tag |
| **Scope creep** | every phase stands alone; OP1, OP2 and HL pay off without the structural phases |
| **Test budget** | the night queue under the timing lock, per `CLAUDE.md` |

## 10. Prior art, to read in OP0 and RF0

- **llama.cpp and ggml:** a fixed operation set implemented per backend, with a graph builder per architecture over it.
  `llama_batch` already is the ragged call: tokens with positions, sequence ids and a per-token output flag.
- **Ollama's Go engine:** model definitions in Go over an `ml` backend. Confirm the shape when reading.
- **vLLM V1's scheduler:** a step is a token budget across requests, and prefill and decode are not separate kinds of step.
  Confirm the wording.
- **Sarathi-Serve** (OSDI 2024): chunked prefills with stall-free batching.
- **FlashAttention's variable-length interface and FlashInfer:** ragged and paged attention kernels.
- **MLX and MLC:** lazy graphs and compiled IR. That is the path not taken (E1), so read them for what they cost in kernel
  control.

## 11. What this absorbs and changes elsewhere

- **`docs/tasks/task-kv-architecture-2026-10.md`:**
  - KV4/KV5 supply the tables RF2/RF4 read;
  - its KV11 takes the expert half of memory;
  - OP4's planner prices through its §5.10.
- **`docs/tasks/task-turn-telemetry-2026-09.md`:** RF0 builds its X1.
- **`docs/tasks/parked/task-concurrency-2026-09.md`:** "spec inside a batch" becomes RF6. MC5's continuous batching stays
  parked (E6).
- **`docs/tasks/task-recompute-audit.md` §5:** its class becomes a planner property (OP4).
- **`docs/hardware-matrix.md`:** decline reasons and the Phi-3/4 configuration note (OP0).
- **`docs/ARCHITECTURE.md`:** §1 ("One decoder, read from a descriptor") and §3 ("Registration and the runner seam",
  "Admission: which models run resident") are rewritten as phases land. §3 says "About a dozen optional interfaces" where the
  decoder asserts 34; fix that in OP0's doc pass.

## 12. When this closes

`docs/ARCHITECTURE.md` carries the design as it lands, and this doc moves to `docs/completed/` with its records. There is no
separate architecture doc while this is a proposal.

## 13. Sources

Code, read at `9593bd70`:
- **The descriptor and admission:**
  - `decoder/registry.go:resolveArchitecture`, `decoder/registry.go:Architecture.validateResolved`;
  - `decoder/arch.go:Architecture`, `decoder/arch.go:ownForwardFamily`;
  - `decoder/residency.go:Model.residentAdmission`, `decoder/residency.go:Model.decodeRunnerDecline`,
    `decoder/residency.go:Architecture.decodeRunnerEligible`, `decoder/residency.go:Architecture.ropeResidentCompatible`,
    `decoder/residency.go:DeclineResident`;
  - `decoder/features.go:ResidentEligible`, `decoder/features.go:residentGateReason`,
    `decoder/features.go:ResidentBackendFeatures`;
  - `decoder/actquant_hazard.go:ActivationQuantHazard`, `internal/modelload/modelload.go:activationSafeQuant`.
- **The seam and the decode loop:** `decoder/residency.go:ResidentForward`, `decoder/residency.go:ResidentGreedyChain`,
  `decoder/residency.go:ResidentSampleChain`, `decoder/model.go:Model.generateInto`.
- **The runners:**
  - `cuda/resident.go:cudaLayer`, `cuda/backend.go:cudaBackend.BuildResident`, `cuda/batchstep.go:cudaResident.StepBatch`,
    `cuda/batchstep.go:cudaResident.stepAttnRows`, `cuda/fused_qkv.cu`;
  - `metal/model.go:residLayer`, `metal/backend.go:metalBackend.BuildResident`, `metal/batch.go:metalResident.StepBatch`,
    `metal/batch_rows.go`;
  - `gpu/decoderunner.go:runLayer`, `gpu/residency.go:webgpuBackend.BuildResident`.
- **Serve:** `internal/serveapp/tools.go:constrainToolUnion`, `internal/serveapp/budget.go`, `internal/loadflags/loadflags.go`.

Docs and records:
- `docs/ARCHITECTURE.md`, `docs/hardware-matrix.md`, `docs/benchmarks.md`, `CHANGELOG.md`;
- `docs/measurements/forward-arch-baseline-2026-10-13.md`, `docs/measurements/kv0-traffic-2026-10-13.md`,
  `docs/measurements/ollama-coverage-2026-10-09.md`, `docs/measurements/peer-sweep-2026-09-29.md`,
  `docs/measurements/mc4-candidate-cuda-2026-10-01.md`, `docs/measurements/chunked-prefill-2026-09-27.md`;
- `docs/tasks/task-kv-architecture-2026-10.md`, `docs/tasks/task-turn-telemetry-2026-09.md`,
  `docs/tasks/task-metal-q4k-2026-10.md`, `docs/tasks/parked/task-concurrency-2026-09.md`,
  `docs/tasks/task-recompute-audit.md`.

<!-- doc-reviewed: 2026-10-13 -->
