# Task: one forward architecture — models as operation lists, one ragged forward pass (OP0–OP6, RF0–RF6, HL) — 2026-10

> **Status: OP0 DONE 2026-10-13. OP1's gates are registered (§4.6). OP1 on Metal: step 1 and the softcap's mechanism are on main
> and pass their day gates; the softcap is NOT declared, because its tiny parity gate failed as registered (§4.6, the last part);
> the real Gemma 2 2B gate decides it. The three night gates passed on 2026-10-14 (the record's last section). OP1 on CUDA and WebGPU, and every other phase, are not started.** OP0's summary is under §4.4, its feature census is §4.5 and its prior-art reading is §10a. Written on the owner's ask after a Cowork conversation the same day ("we have
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
| OP0 | decline reasons in the hardware matrix; features mapped to operations; prior art. **Done 2026-10-13** | — | S | — |
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
  - Nine families run their own layer loop: the nine entries of the `ownForwards` table (`decoder/arch.go:ownForwardFamily`),
    one `runLayers*` function each, across twelve files named `forward_*.go` in `decoder/`. They are:
    - the recurrent hybrids: Gated DeltaNet, short conv, two Mamba-2 families, and KDA with MLA;
    - MLA itself (DeepSeek);
    - three softmax families whose layers differ: Gemma 4, Llama 4 and gpt-oss.
  - Six of the nine pass the runner-shape check through a per-family bridge.
    - Ling 3.0 then declines on KDA everywhere.
    - Nemotron-H declines on SSM on CUDA and Metal and runs on WebGPU. A Nemotron 3 Nano checkpoint (MoE) also fails the
      runner shape on CUDA and Metal.
  - Llama 4 and LFM2 have no bridge. Granite-4.0-H waits on a bring-up gate, and on CUDA and Metal also lacks SSM.
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
reads. "—" means the family runs resident there. Where an earlier check refuses first, what the later ones would also name is
listed after it ("plus"). The programs and their raw output are in the baseline record.

**Since OP0 the matrix generates this table itself** (`docs/hardware-matrix.md`, "Why a cell reads CPU"), from every gate and
not only the first. The table below is kept as the reading the design started from, with two rows corrected by what the
generator showed: Spark-X2.5 (the activation, below) and Kimi K2 on Metal (a router capacity the missing `mla` hid).

| family | CUDA | Metal | WebGPU |
|---|---|---|---|
| Gemma 2 | attn-logit-softcap | attn-logit-softcap | attn-logit-softcap |
| DeepSeek-V2, DeepSeek-V3 | — | mla | — |
| Kimi K2 | — | mla, plus the router's capacity (384 experts; Metal's kernel routes 256) | — |
| Ling 3.0 | kda | kda, mla | kda |
| Nemotron-H | non-gated-mlp, ssm (plus runner shape if MoE) | ssm (plus runner shape if MoE) | — |
| GPT-2 | learned-pos, non-gated-mlp | — | layer-norm, learned-pos |
| Command-R, Command-R7B | — | — | layer-norm, pairwise-rope, parallel-block (Command-R7B also nope) |
| GLM-OCR | — | — | pairwise-mrope, pairwise-rope |
| Ministral 3 | — | — | attn-temp |
| Olmo 3, Olmo Hybrid | — | — | post-only-norm, qk-norm-whole (Olmo Hybrid also nope) |
| SmolLM3 | — | — | nope |
| Llama 4 | runner shape: own forward, not bridged | the same | the same |
| LFM2.5 | runner shape: own forward, not bridged, plus short-conv | the same | the same |
| Laguna | runner shape: local/global RoPE tables differ in length, plus attn-output-gate | the same | the same |
| Spark-X2.5 | runner shape: local/global RoPE tables differ in length, plus attn-output-gate, gated-gelu-erf | the same | the same |
| Granite-4.0-H | bring-up gate (`GOINFER_SSM_RESIDENT`), plus ssm | the same | bring-up gate (`GOINFER_SSM_RESIDENT`) |
| Phi-3, Phi-4 | the activation hazard: int8 activations round its outliers away | the same | the same |

**Four kinds of decline:**
- **A missing operation** on one or more backends: softcapping, MLA, SSM, KDA, short conv, the attention output gate
  (Laguna's softplus, Spark-X2.5's sigmoid), and a set of small operations WebGPU lacks.
- **Layers of different shapes.**
  - Llama 4 and LFM2.5 run their own forwards, which no GPU runner bridges.
  - Laguna and Spark-X2.5 use the generic loop, but give local and global layers RoPE tables of different lengths.
    `decoder/residency.go:Architecture.ropeResidentCompatible` refuses that.
  - No uniform-layer runner can express these, however many operations it has.
- **A gate.** Granite-4.0-H waits on its parity bring-up on every backend. On WebGPU that is the only reason, since WebGPU
  already declares its SSM feature (`docs/hardware-matrix.md`'s note). CUDA and Metal also lack SSM.
- **A numerics guard, read in one configuration.** Phi-3/4's activation outliers are rounded away by per-row int8 activation
  scales (`decoder/actquant_hazard.go:ActivationQuantHazard`). The matrix shows CPU everywhere, but the loader picks a safe
  configuration (`internal/modelload/modelload.go:activationSafeQuant`):
  - **On CUDA,** a Phi-3/4 GGUF loads at `--quant q4k` and runs resident: 112.6 tok/s at depth 128 on the 8 GB card
    (`docs/measurements/peer-sweep-2026-09-29.md`). Its batched prefill declines to the sequential path.
  - **On Metal,** it runs resident at an explicit `--quant q4k`, a lane built 2026-10-08 (`docs/tasks/task-metal-q4k-2026-10.md`).
    The default load still picks int8int8 with per-32 scales, which declines to the CPU.
  - **WebGPU** has neither.

**The activation hazard, closed by OP0.** Every resident MLP kernel picks its activation from the ordinal
`decoder/residency.go:Model.GatedActResident` returns, and knows two: tanh-GELU and SiLU.
- **Gated, exact-erf GELU (Spark-X2.5; `ActGelu`, `gegluExact`).** The feature list mapped every gated non-SiLU activation to
  `gated-gelu`, which every backend declares. Nothing ran wrong, because the runner shape and the output gate decline
  Spark-X2.5 first; once both were lifted it would have been admitted with the wrong activation.
- **Non-gated, exact-erf GELU: this one was live.** A GPT-2 checkpoint that declares `activation_function: "gelu"` passed every
  feature Metal declares, and Metal's non-gated kernel ran tanh-GELU for it. Measured on the pre-fix tree: Metal's logits were
  bit-identical between the `gelu` and `gelu_new` configs of the real GPT-2, while the CPU's differed (the baseline record, §3).
  The released GPT-2 checkpoints use `gelu_new`.
- **Now:** `gated-gelu` is tanh-GELU only. `gated-gelu-erf` and `non-gated-gelu-erf` are features no backend declares, and
  `gated-act-unknown` catches any other gated activation, so a gated model past admission hands a kernel only the two ordinals
  it knows (`decoder/features.go:Architecture.residentFeatures`).
- **What is left** is in §4.5: `non-gated-mlp` still means a different activation on Metal and on WebGPU.

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
  - MLA on Metal brings DeepSeek-V2 and DeepSeek-V3 there. Kimi K2 also needs Metal's router to route its 384 experts; the
    kernel routes 256 (`decoder/features.go:ResidentBackendMoECap`).
- **Performance where it matters most.** A family leaving the CPU fallback gets roughly the 15× decode difference. That is the
  largest single change available to a user of that family.
- **New families get cheaper.** A family made of existing operations needs an adapter (descriptor → program) and a parity gate,
  not code in three runners.
- **The families whose layers differ stop being special.** Each becomes a program of mostly existing kinds plus its own
  missing operations:
  - **Llama 4:** chunked attention on its RoPE layers, NoPE layers, parameter-free L2 QK-norm, attention temperature, and
    dense/MoE layer interleaving;
  - **LFM2.5:** a gated short convolution (`FeatShortConv`) on 22 of its 30 layers;
  - **Laguna:** per-layer query heads, per-layer RoPE rotary width, and attention output gate (`g_proj` with softplus,
    `FeatAttnOutputGate`);
  - **Spark-X2.5:** per-layer RoPE rotary width, attention output gate (`g_proj` with sigmoid, `FeatAttnOutputGate`), and
    exact-erf gated GELU (`ActGelu`, `gegluExact`);
  - **Granite-4.0-H:** Mamba-2 (`FeatSSM`) and attention layers.

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

- **OP0 — the decline table, the operation census, prior art (S). DONE 2026-10-13; the summary follows this list.**
  - `docs/hardware-matrix.md` prints each declining cell's reasons: the runner shape, the operations missing, or the gate. It
    is generated from both checks, and a runner-shape decline does not hide the operations `residentGateReason` also names.
  - A row whose default load runs resident in another configuration says so: Phi-3/4 at `--quant q4k` on CUDA.
  - Map every `ResidentFeature` to an operation kind or parameter.
  - Split `gated-gelu` into tanh GELU (`ActGeluTanh`, Gemma) and exact-erf GELU (`ActGelu`, Spark-X2.5), so a backend without
    an exact-GELU kernel declines Spark-X2.5 by name (§3.2's hazard).
  - Read the prior art (§10).
- **OP1 — attention softcapping on every GPU (M).** Worth doing before the program exists.
  - Softcapping is a parameter of every attention kernel variant per backend: decode, split-KV and flash-decode, fused and
    exact prefill, and the MC3 rows.
  - It is two steps per backend: route every kernel's score through one function, then put the softcap in that function.
  - **Its gates are registered in §4.6.**
- **OP2 — the next operations, in order of demand (M–L each).** Each has its own gates, as OP1 does.
  - MLA on Metal: DeepSeek-V2/V3. Kimi K2 needs it and a router that routes 384 experts (Metal's routes 256).
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

**OP0, done 2026-10-13.** What it changed, and what it found:
- **The matrix says why.** `docs/hardware-matrix.md` has a generated section, "Why a cell reads CPU", built from
  `decoder/residency.go:Architecture.runnerShapeDecline` and `decoder/features.go:residentGateReasons`. It lists every gate
  that refuses, and the generator fails if a cell and its reasons disagree. A note covers Phi-3/4's other configurations.
- **Its reasons are the baseline record's, with two additions:** Spark-X2.5's activation, and Kimi K2's router capacity on
  Metal, which the missing `mla` had hidden. §3.2's table and OP2's line are corrected for it.
- **The activation hazard is closed** (§3.2): `gated-gelu-erf`, `non-gated-gelu-erf` and `gated-act-unknown`, declared by no
  backend. The non-gated case was live on Metal, and the baseline record's §3 has the run before and after.
- **A runner-shape decline now names the shape** in `DecodePath()` and `serve check`.
- **The census is §4.5.** Its seven numbered findings are where the operation boundary does not hold, and it sizes OP1.
- **The prior art is §10a,** with §10's lines corrected in place.
- **Two things for the phases that follow:**
  - Before OP2's SSM on Metal, the non-gated activation needs a feature per value (§4.5, item 2).
  - Before OP5, `attn-output-gate` needs splitting by activation (§4.5, item 3).
- **Three things in backend modules were read and not fixed** (§4.5's last list). The CUDA one may be live.

### 4.5 Features mapped to operations (OP0, read 2026-10-13)

Every `ResidentFeature`, the operation it would become, and what implements it on each backend today.

- **Basis.** Read on branch `op0-2026-10` (main `bf62dd71` plus OP0's own changes). Nothing here was run.
  - "Declared" is read from `decoder/features.go:ResidentBackendFeatures`' source map. "—" means the backend does not declare
    the feature.
  - The kernels were read in three passes, one per backend module. Every kernel name below was then checked to occur in the
    file named. A statement that was not read in the code is marked INFERRED.
- **Kernel files.** CUDA kernels are `.cu` files in `cuda/`. Metal's are Go string constants: `metal/kernels/kernels_msl.go` (decode), `metal/kernels/prefill_msl.go` (the f16
  prefill), `metal/kernels/moe_msl.go` and `metal/kernels/deltanet_kernels_msl.go`. WebGPU's are WGSL constants in `gpu/`.
- **What "operation" means here.** §4.1's kinds: `norm`, `proj`, `qk-norm`, `rope`, `attention`, `out-gate`, `ffn` (gated,
  non-gated or MoE), the recurrent mixers, and the embedding and head stages. A feature is an operation kind, a parameter of
  one, a property of the layer list, or a bundle of several.

**Norms**

| feature | operation | CUDA | Metal | WebGPU |
|---|---|---|---|---|
| `rms-add-one` | parameter of `norm(rms)`: the (1+w) offset | `addOne` of every RMS kernel: `rmsnorm_quant`, `rmsnorm_f32` (`cuda/glue.cu`), `fused_rms_qkv`, `fused_rms_gu`, `qk_norm` (`cuda/fused_qkv.cu`), their batched twins (`cuda/prefill_batched.cu`) | `addOne` of `rmsnorm_quant`, `rmsnorm_f32`, `qk_norm`; `rmsnorm_f16`, `qk_norm_f16` in the prefill | `addone` of `rmsnormQuantWGSL` (`gpu/decodefuse.go`), `rmsnormShaderWGSL` (`gpu/layer.go`), `qkNormWGSL` (`gpu/qknorm.go`) |
| `layer-norm` | a second `norm` kind, with a bias parameter | `layernorm_quant` (`cuda/glue.cu`), bias-free; `layernorm_quant_batched` (`cuda/layernorm_quant.cu`) | `layernorm_quant` with `hasBias`; the prefill's `layernorm_f16` is bias-free | — |
| `qk-norm` | `qk-norm`, per head | `qk_norm` (`cuda/fused_qkv.cu`), `qk_norm_batched` (`cuda/prefill_batched.cu`) | `qk_norm`; `qk_norm_f16` | `qkNormWGSL` (`gpu/qknorm.go`) |
| `qk-norm-whole` | parameter of `qk-norm`: one group over all heads | the same kernels launched as one head; declines unless nH = nKV | the same; declines unless nH = nKV | — |
| `sandwich-norm` | the layer list: a `norm` after each sublayer | no new kernel: `rmsnorm_f32`, and the o and down projections lose their fused residual add | `rmsnorm_f32`; `rmsnorm_f16` in the prefill | reuses `rmsnormShaderWGSL` and `residualShaderWGSL` (`gpu/layer.go`) |
| `post-only-norm` | the layer list: no pre-norm | no new kernel: `quant_vec` (`cuda/glue.cu`) replaces the pre-norm | no new kernel: `quant_vec` | — |
| `parallel-block` | the layer list: one `norm` feeds attention and FFN, one residual add | no kernel; the attention input is reused | no kernel; the same | — |

**Positions**

| feature | operation | CUDA | Metal | WebGPU |
|---|---|---|---|---|
| `partial-rotary` | parameter of `rope`: the rotary width | `rhalf` of `rope_kv` (`cuda/gemv_fwd.cu`), `rope_kv_batched` (`cuda/prefill_batched.cu`) | `rhalf` of `rope2`; `rope_f16` | `half` in the rope shaders (`gpu/attention.go`); the batched prefill declines it |
| `yarn-mscale` | parameter of `rope`: a scale | `mscale` of the same kernels | `scale` of `rope2`; `rope_f16` | `scale` in the rope shaders; the batched prefill declines it |
| `per-layer-rope` | the layer list: `rope`'s table differs by layer | a buffer bound per layer | a buffer bound per layer | a buffer bound per layer |
| `nope` | the layer list: a layer with no `rope` | no kernel: the decoder hands the layer an all-zero table | the same | — (the table mechanism is there; not declared) |
| `pairwise-rope` | parameter of `rope`: the pairing; its own kernels | `rope_kv_pw`, `rope_kv_batched_pw` (`cuda/rope_pairwise.cu`) | `rope_pw`, `rope2_pw`; `rope_f16_pw` | — |
| `pairwise-mrope` | pairing × a (t, h, w) position; prefill only | `rope_kv_mrope_batched_pw` (`cuda/rope_pairwise.cu`) | `rope_mrope_f16_pw` | — |
| `learned-pos` | an embedding-stage operation; no `rope` | — | host add (`metal/model.go:resident.addLearnedPosTo`) | — |

**Attention**

| feature | operation | CUDA | Metal | WebGPU |
|---|---|---|---|---|
| `sliding-window` | parameter of `attention`, and a KV property | `window` of `attention` (`cuda/glue.cu`), `attn_batched`, `attn_fused_hd64` and the tiled kernels; a host-computed start for `splitkv_scores` and `fa_partial` | `window` of `attention`, `attention_i8`, `attention_prefill`; the batched step declines it | `start` in every decode attention shader's uniform (`gpu/attention.go`) |
| `attn-logit-softcap` | parameter of `attention` | — | — | — |
| `attn-temp` | parameter of `attention` (a query scale), implemented in the rope kernel | `qTempScale` of `rope_kv`; a per-row table for `rope_kv_batched` | `qTempScale` of `rope2`; not in the f16 prefill | — |
| `attn-sink` | **bundle of four** (below) | `sinks` of `splitkv_softmax` (`cuda/decode_splitkv.cu`), `attn_batched`, `attn_fused_hd64`; `glu_quant_gptoss`, `route_gptoss` (`cuda/gptoss_act.cu`); `gemv_w4a8_moe_wacc_bias` (`cuda/moe.cu`) | `sinks` of `attention`, `attention_i8`; `swiglu_quant_gptoss`, `route_gptoss_sg`, `gemv_w4a8_moe_wacc_bias` | `sinks` in the nine decode attention pipelines (`gpu/attention.go`); `routeGptOssWGSL`, `gptossGluQuantWGSL`, `moeExpertGptOssDownGEMVWGSL` (`gpu/moe.go`) |
| `attn-output-gate` | `out-gate`, with two activations under one name, plus a per-layer query-head count | — | — | — |
| `mla` | a second `attention` kind, with its own projections and rope | `mla_latent_store`, `mla_head_matvec`, `mla_q_rope`, `mla_attn` (`cuda/mla.cu`); decode only | — | `mlaLatentStoreWGSL`, `mlaHeadMatvecWGSL`, `mlaQRopeWGSL`, `mlaAttnWGSL` (`gpu/mla.go`) |
| `out-bias` | parameter of `proj(o)`: a bias | the bias argument of the GEMVs | `gemv_w4a8_sa_bias_resid` | no new shader: a bare GEMV, then two adds |

**Feed-forward**

| feature | operation | CUDA | Metal | WebGPU |
|---|---|---|---|---|
| `gated-gelu` | parameter of `ffn(gated)`: tanh-GELU | `act` of `glu_quant` (`cuda/glue.cu`), `glu_quant_batched` (`cuda/prefill_batched.cu`), `glu_quant_g32` (`cuda/actgroup.cu`) | `act` of `swiglu_quant`, `swiglu_f32`; `swiglu_f16` | its own shader, `gegluQuantWGSL` (`gpu/decodefuse.go`), chosen by one model-level flag |
| `gated-gelu-erf` | parameter of `ffn(gated)`: exact GELU | — | — | — |
| `gated-act-unknown` | none: the catch-all for any other gated activation | — | — | — |
| `non-gated-mlp` | `ffn(non-gated)`; **the activation is not named** | — | `act_quant`: tanh-GELU or SiLU, with biased up and down GEMVs | `relu2QuantWGSL` (`gpu/relu2.go`): squared ReLU, reached only through Nemotron's block kinds |
| `non-gated-gelu-erf` | parameter of `ffn(non-gated)`: exact GELU | — | — | — |
| `moe` | `ffn(moe)`: a router, routed experts, an ungated shared expert | `moe_route`, `gemv_w4a8_moe`, `gemv_w4a8_moe_wacc`, `shared_gate_combine` (`cuda/moe.cu`) | `moe_route_sg`, `gemv_w4a8_moe`, `gemv_w4a8_moe_wacc` | `moeRouteWGSL`, `moeExpertGEMVWGSL` (`gpu/moe.go`) |
| `moe-gated-shared` | parameter of `ffn(moe)`: a sigmoid gate on the shared expert | the `ungated` argument of `shared_gate_combine` | `shared_gate_combine` | `sharedGateWGSL` (`gpu/moe.go`) |

**Recurrent mixers**

| feature | operation | CUDA | Metal | WebGPU |
|---|---|---|---|---|
| `ssm` | `mamba2`, for two families with different block layouts | — | — | `mambaConvShaderWGSL`, `mambaSSMShaderWGSL`, `mambaGNormShaderWGSL` (`gpu/mamba.go`) |
| `deltanet` | **bundle of two**: the `deltanet` mixer, and a fused output gate on the family's softmax layers | `delta_conv`, `delta_gates`, `delta_norm`, `delta_rule`, `delta_gnorm`; `delta_qsplit`, `delta_attn_gate` (`cuda/deltanet.cu`) | the same seven names | `deltaRuleShaderWGSL`, `deltaNormShaderWGSL`, `deltaGatesShaderWGSL`, `deltaGNormShaderWGSL`; `deltaQSplitShaderWGSL`, `deltaAttnGateShaderWGSL` (`gpu/deltanet.go`) |
| `kda` | `kda` | — | — | — |
| `short-conv` | `short-conv` | — | — | — |

**Embedding and head**

| feature | operation | CUDA | Metal | WebGPU |
|---|---|---|---|---|
| `embed-scale` | parameter of the embedding stage | host, in the decoder for every backend (`decoder/residency.go:Model.embedResidentInto`) | the same | the same |
| `final-logit-softcap` | head post-processing, on the host | `cuda/softcap.go:applySoftcap` | `metal/model.go:softcapParallel` | `gpu/softcap.go:applySoftcap` |
| `logit-scale` | head post-processing | host multiply (`cuda/softcap.go:applyLogitScale`) | host multiply (`metal/model.go:resident.finalizeLogits`) | folded into the head's weight scale at build, **for the Granite hybrid only** |
| `gemma4-e-model` | **bundle of three**: a per-layer embedding operation, cross-layer shared KV, a per-layer FFN width | no new kernel | `ple_gelu_mul`, plus existing kernels (`metal/model.go:resident.encodePLE`) | no new shader |

**What does not map cleanly.** These are the places the operation boundary of §4.1 does not hold today.

1. **Four features are bundles.** A backend declares all of a bundle or none, so a model that needs one part waits for the rest.
   - `attn-sink` is gpt-oss's four changes: the sink (an `attention` parameter), a router whose bias reaches the mixing weight,
     a clamped SwiGLU expert with a bias, and a biased weighted down-projection.
   - `deltanet` is the mixer plus an output gate on the softmax layers, fused into a double-width query projection.
   - `gemma4-e-model` is one operation, one KV-layer property and one property of the layer list.
   - `moe` covers every routing flavour as kernel parameters, and each backend's build then declines some shapes by hand: on
     CUDA, a non-SiLU expert, MoE with sandwich norms, and widths that are not multiples of 32
     (`cuda/backend.go:cudaBackend.BuildResident`).
2. **One name, a different operation per backend.**
   - `non-gated-mlp` is tanh-GELU or SiLU on Metal and squared ReLU on WebGPU. OP0 named exact GELU; the rest is open.
     - A non-gated squared-ReLU model derives nothing for its activation, and Metal's `act_quant` would run tanh-GELU for it.
       Today only Nemotron-H is that shape, and `ssm` declines it on Metal first. **OP2's "SSM on Metal" lifts that**, so the
       non-gated activation needs a feature per value before then.
     - WebGPU reaches its kernel only through Nemotron's block kinds. INFERRED: another non-gated family that passed the
       gates would fail the build with a projection error, not a decline.
   - `logit-scale` is a host multiply for any family on CUDA and Metal. On WebGPU it is a fold into the head that only the
     Granite hybrid's build performs (`gpu/residency.go:webgpuBackend.BuildResident`). Another family with a logit scale would
     be admitted there and left unscaled. No released checkpoint reaches it: dense Granite 4.2 ships a scale of 1.0, and
     Command-R declines on WebGPU for other features (both read from this repo's notes, not from the checkpoints).
   - `layer-norm` is biased on Metal's decode kernel, and bias-free on CUDA's decode kernel and in Metal's prefill. GPT-2's
     biased norm is not on CUDA; `learned-pos` and `non-gated-mlp` decline it there first.
3. **One name, two parameter values: the class OP0's activation fix belongs to.** `attn-output-gate` is softplus for Laguna and
   sigmoid for Spark-X2.5, and also carries Laguna's per-layer query-head count. A backend that builds one gate and declares
   the feature admits the other. Split it before OP5.
4. **Five features are not operations.** `sandwich-norm`, `post-only-norm`, `parallel-block`, `nope` and `per-layer-rope`
   say which operations a layer has, in what order, and with which table. In a layer program they are the list itself.
   - They still cost code, because fusions assume one order. Sandwich and post-only norms give up the residual add fused into
     the o and down projections. On CUDA, post-only norms and the parallel block turn the fused norm-plus-QKV kernel off.
   - This is the argument for §4.1's explicit fusion tables: the kernels already know which sequences they claim.
5. **Where the math lives is not where the code lives.** `attn-temp` scales the query, so it is an `attention` parameter, and
   both backends implement it inside the rope kernel. It is missing from Metal's f16 prefill rope and from CUDA's m-RoPE and
   MLA kernels.
6. **A second admission system sits beside the features.** These decide admission and are not features:
   - in the decoder: the runner shape, the activation hazard and the activation group, the router's capacity, per-layer
     attention geometry, and Gemma 4's parallel dense and MoE block (`decoder/features.go:residentGateReasons`);
   - in each backend's build: hand-written declines. Metal's include widths that are not multiples of 8 and the q4k lane's
     shape (`metal/model.go:buildResident`). WebGPU's include a head dimension above its kernel's and a KV
     precision per family.
   - OP4's "admission by operation" must take these in, or it replaces one of two systems.
7. **The feature sets describe decode. Every other request shape keeps its own, smaller list.**
   - Metal: `metal/model.go:prefillFeatures` for the f16 prefill, and `metal/batch.go:resident.batchIneligible` for the batched
     step, which declines eleven model-level variants.
   - WebGPU: `gpu/prefillrunner.go:runModelToPrefill` declines MoE, MLA, the mixers, partial rotary and a rope scale.
   - CUDA: `cuda/prefill.go:cudaResident.prefillStaticDecline` and `cuda/batchstep.go:cudaResident.BatchStepRange`. MLA runs
     in decode only (its prefill and verify fall back to sequential: INFERRED from the code, not run).
   - So an operation table needs a request-shape column until RF removes the shapes. This is §3.3's cost, counted per feature.

**What OP1 has to touch.** Softcapping is one `attention` parameter, and the score is formed in this many places on the text
path:

| backend | hand-written score implementations | derived or twin forms |
|---|---|---|
| CUDA | 10, in 9 `.cu` files; 8 without MLA and the block drafter | f16-KV builds of six of those modules |
| Metal | 10: 6 decode and 4 prefill | 4 batched-step forms, derived as text (`metal/batch_rows.go`) |
| WebGPU | 10: 6 decode, 1 MLA and 3 prefill | 3 derived as text; 14 pipeline slots in all |

Metal's Gemma 4 audio tower already has a softcapped attention kernel, off the text path.

**Found while reading, in backend modules, and not fixed** (a backend change is a two-step tag; each needs its owner's call).
All are read from the code, and none was run.
- **CUDA: a batched-step row that returns logits appears to get the final softcap and the logit scale twice.**
  `cuda/prefill.go:cudaResident.prefillCore` applies both inside the job for a step row, and its closing loop applies both
  again to every row. It would touch concurrent conversations that sample on the host, on a model with a final softcap
  (dense Gemma 4) or a logit scale (Command-R). Both transforms keep the argmax, so a greedy test cannot see it.
- **WebGPU: `logit-scale`**, item 2 above.
- **WebGPU and Granite-4.0-H.** INFERRED: a Granite hybrid checkpoint whose config says `position_embedding_type: "nope"`
  derives `nope`, which WebGPU does not declare. The registry's representative config is roped, so the matrix cannot show it.
  If so, lifting the bring-up gate is not the only step left for the released tiny checkpoint, which the registry says ships
  NoPE.
- **Comments in `decoder/features.go`'s declared sets that the code has moved past:** CUDA's "per-layer rotary width is not
  implemented" (the layer struct carries one); CUDA's attention temperature "recomputed per row device-side" (a host-built
  table); Metal's "the f16 prefill declines" the E-model (it has a batched route); and several kernel file names.

### 4.6 OP1: the registered gates (2026-10-13, before any OP1 code)

Registered at main `0fb622ff`. Nothing below has been run. A backend's rows may be amended only before that backend's OP1 code
starts, in a commit of their own; after that a change to a rule is a deviation and is reported as one.

**The two steps, per backend.**
- **Step 1 is a refactor.** §4.5 counts about 10 hand-written places per backend where an attention score is formed. Each is
  routed through one score function per backend. The function returns what the kernel computes today, so nothing changes.
- **Step 2 puts the softcap in that function:** `cap · tanh(score / cap)` on the scaled score, before the mask and the softmax,
  which is what the CPU computes (`decoder/attention.go:softcapScore`).

**What the function has to respect** (Metal, read 2026-10-13 in `metal/kernels/kernels_msl.go` and
`metal/kernels/prefill_msl.go`). The kernels do not form the score the same way:
- `attention`, `attention_f32`, `attention_fa`, `attention_prefill`, `attention_prefill_img` and `attention_prefill_fused`
  multiply the dot product by the scale;
- `attention_i8` multiplies by the key's scale first, then by the scale;
- `attention_fa_blk` and `attention_fa_blk64` multiply the scale into the query before the dot product;
- `attention_prefill_steel` folds log2(e) into the scale and uses a base-2 exponential.

So the function takes the score as the kernel has it, and each kernel keeps its own order of multiplies. Where a constant is
folded into the scale, step 2 folds the cap's constants the same way (FlashInfer's form, §10a).

**Amended 2026-10-13, before any Metal OP1 code: one Metal kernel stays out of step 1.**
- `attention_fa_blk` (head dimension 128) is pinned to the source that was graded: `TestAttnFABlkIsTheGradedKernel` hashes its
  text, and its rule is that an edit needs the R17 grading re-run (fidelity, kernel accuracy, speed).
- Routing its score through a function edits that text. A byte-identity digest is evidence about the inputs it ran, and it is
  not the grading the pin asks for, so step 1 does not touch this kernel and does not move the pin.
- Nothing is lost for OP1: a capped layer never takes this kernel (the rule below), and Gemma 2's head dimension is 256.
- **Owed, and not in OP1:** route it together with a re-run of the R17 grading, on the night queue.
- G1-K still records this kernel's digests, so the record shows it did not move.
- `attention_fa_blk64`, its head-dimension-64 twin, has no source pin and is routed in step 1.

**Metal's mechanism, registered because the gates lean on it: the softcap is a compile-time variant.**
- A model with a cap compiles its kernel library with the cap defined. A model without one compiles step 1's source.
- **Why not a run-time parameter:**
  - a branch per score sits in the hottest loop of every model;
  - Metal's output has moved with code shape before, so "the cap = 0 path is identical" would rest on the compiler;
  - the batched step's split-KV rows kernel already binds all 16 buffers it can.
- **What this buys the identity gate:** for a model without a cap, step 2's kernel source is step 1's, byte for byte. G2-I
  checks that as text, and then checks the bits.
- CUDA and WebGPU choose their mechanism on nobara and write it here before their code. CUDA's kernels are prebuilt PTX, so a
  compile-time variant there means more PTX files, as its f16-KV twins are.

**Which kernels a capped model may run.**
- Gemma 2 is the only family with a cap. INFERRED from Metal's lane selectors, to be confirmed by step 2's tests: its layers
  (sandwich norms, head dimension 256) take `attention`, `attention_i8` under `--kv i8`, and `attention_prefill`.
- **The rule:** every kernel that can be compiled with the cap is covered by G2-K, and a lane selector keeps a capped layer
  off any kernel that is not. End-to-end evidence (G2-T, G2-R) exists only for the kernels Gemma 2 reaches, and the record
  lists which those are.

**Boxes.**

| backend | box | notes |
|---|---|---|
| Metal | the MacBook: M1 Pro, 16 GB | 9.0 GB of disk free on 2026-10-13 |
| CUDA | nobara-pc: RTX 2070 SUPER, 8 GB, the pinned driver | |
| WebGPU | nobara-pc, over Vulkan | |

**Checkpoints.**
- **Tiny:** `testdata/gemma2-tiny` and `testdata/gemma2-hd-tiny`.
- **Real 2B:** `gemma-2-2b-it`, the Q8_0 GGUF the CPU's real gate reads (`TestGemma12Real_gate`).
  - **Corrected 2026-10-13, a fact and not a rule:** this line said the file comes from the archive with `models-pull`. It is
    not in the archive. It is in nobara's `~/models` only (listed 2026-10-13), and `models-pull` on the Mac found nothing.
  - **Copied 2026-10-14 08:16** from nobara's `~/models` to the Mac's `~/models`, with nobara's timing lock free, and verified:
    sha256 `2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe` on both boxes.
- **Real 9B:** not in the archive or on either box today. It runs where a box has the disk and the memory for it, and
  otherwise the record says "9B not run" by name. The Mac does not have the disk today.
- **The peer cell's file** is the GGUF Ollama ships for `gemma2:2b`, copied to `~/models` and loaded by both engines, with
  `scripts/gguf_same_weights.py` reporting every tensor identical.

**The gates.** "Base" is the commit before the backend's OP1 code. Identity gates have no ambiguous band: equal or not.

| id | step | gate | tier | instrument | rule |
|---|---|---|---|---|---|
| G1-K | 1 | every attention kernel's output bytes are unchanged | day | a kernel digest test, built at base before the refactor: each text-path attention kernel driven on fixed inputs, its output hashed. Shapes: the head dimensions each kernel serves (64, 128, 256, 512); key counts on both sides of each kernel's tile bound; windowed and not; with and without a sink | every digest at the candidate equals base's. The test fails if the source has an attention kernel it does not drive |
| G1-M | 1 | every family's logits are unchanged | day (tiny), night (real) | `go run ./cmd/gate identity <base> <candidate> -backend <b>`, `-assets tiny` then `-assets real`. The tool has no CUDA backend today; CUDA's half names its instrument before its code | bytes equal for every family, every step |
| G1-S | 1 | the backend's own suites still pass | day (the tests the change touches), night (the device gate) | the attention kernel tests and the batched step's bit-identity tests by name; then `go run ./cmd/gate gpu` | green, with no test that passed at base skipped |
| G2-I | 2 | a model without a cap is untouched | day (tiny), night (real) | G1-K and G1-M again, base against the step-2 tree; on Metal also a test that the library source built without the cap equals step 1's | equal |
| G2-K | 2 | each capped kernel computes the softcap | day | each kernel that can be compiled with the cap, against an f64 reference, with scores on both sides of ±cap and peaked attention | cosine ≥ 0.9999 and max abs error ≤ 1e-2, `TestAttention_ShippedKernelShapes`' bar |
| G2-T | 2 | Gemma 2 tiny, resident against the CPU | day | `TestGemma1ResidentParityMetal`'s shape on both tiny fixtures: 16 positions, both sides int4. The batched prefill against the token-by-token one, as `TestPrefillParityGemma` does | worst logit cosine ≥ 0.999. `TestGemma2DeclinesResidentMetal` is replaced by this |
| G2-R | 2 | Gemma 2 real, resident against the CPU | day if estimated under 10 min, else night | the real-checkpoint harness (`TestGemma3ResidentParity`'s: greedy lockstep off the CPU, 24 steps, positions from 2), at int8int8 and at int4, both sides the same | min logit cosine ≥ 0.88, argmax equal at ≥ 50% of positions, no NaN: the Gemma 3 gate's bar |
| G2-D | 2 | the gates can see a dropped cap | day | the cap removed from the build (a planted defect), then G2-K, G2-T, G2-R | G2-K must go red. For G2-T and G2-R the result is recorded: one that stays green does not vouch for the cap, only for the rest of Gemma 2's path |
| G2-M | 2 | the matrix | day | `go test ./decoder -run HardwareMatrix -update` | Gemma 2 reads resident on the backend, and only after G2-I, G2-K, G2-T, G2-R pass there |
| G2-S | 1+2 | no model got slower | night | two arms, new against base (below) | below |
| G2-P | 2 | Gemma 2's decode against Ollama and against our CPU | night | `scripts/bench_peer.py`, one timed run per box (below) | below |

**G2-R's band.** `TestDenseResidentParity` (the control, bar 0.95) and `TestGemma3ResidentParity` run in the same session.
- The control failing voids the session.
- **Parked, for the owner:** Gemma 2 clears 0.88 but its min cosine is more than 0.03 under Gemma 3's from the same session.
  The feature is not declared while parked.

**G2-R's baseline step, added 2026-10-13 on the owner's ruling, before any G2-R run.** G2-T's bar sat within 0.0003 of
its fixture's own no-softcap agreement, which nobody had measured. This step measures G2-R's before its graded arm runs.
- **The checkpoint:** `gemma-2-2b-it-Q8_0.gguf`, sha256 `2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe`, in
  the Mac's `~/models`.
- **Arm B, the baseline.** G2-R's harness and metric (greedy lockstep off the CPU, 24 steps, the min logit cosine over
  positions from 2), at each registered quant (int8int8 and int4), with the softcap removed on BOTH sides.
  - **How it is removed:** a test hook, built only under `goinfer_testhooks`, zeroes the model's attention softcap before
    the resident is built. It is applied to the CPU model and to the Metal model alike.
  - **The hook is checked before B uses it (added 2026-10-13, before the hook was written):** on the CPU, the 2B's logits
    with the hook must differ from its logits without it, on the same tokens. If they do not, the hook is broken and the
    run stops there.
  - **What is recorded:** the min cosine per quant.
- **The stopping rule on B:** if 0.88, G2-R's bar, is within 0.001 of B's min cosine at either quant (B under 0.881), stop
  and report. The graded arm does not run and the bar is not changed.
- **Otherwise, two arms, once each, in this order, in B's session:**
  - **Arm G, graded:** G2-R as registered, the softcap on both sides, with its control and its band.
  - **Arm P, the planted defect:** the softcap dropped from the Metal build only, the CPU keeping it. P must FAIL G2-R's
    rule (min cosine under 0.88, or argmax equal at under half the positions) at both quants.
  - **P also records how much the softcap acts on the gate's prompts (added 2026-10-13, before the hook was written).**
    All three are read on the CPU, on the same tokens P feeds:
    - the share of attention scores, over every layer, head and key, with |score| > 15 before the cap;
    - the largest |score| before the cap;
    - the worst per-position logit cosine between the capped and the uncapped CPU runs.
- **The outcome:**
  - G passes and P fails: G2-R passes, and the declaration may merge.
  - G fails or is parked: stop and report, with no retuning.
  - **P does not fail G2-R's rule at a quant: G2-R is VOID as a softcap test at that quant, not a pass.** Report it with
    P's three numbers. Do not declare, and do not change the bar or the prompts after the fact. The next step would be a
    prompt set registered in advance for long contexts, where the scores are larger.
- **Void, in addition to G2-R's own conditions:**
  - B, G and P not in one session;
  - a resident that reports a cap that is not its arm's (0 in B, the model's in G and P);
  - the Gemma 3 control's checkpoint missing. The test reads `~/models/gemma-3-4b-it-Q4_K_M.gguf`, and on the Mac the file
    is in `~/models/gemma-3-4b-it-gguf/`. The run links it to the path the test reads for the session, and removes the
    link after.
- **Tier: by day.** The 2B loads twice per arm and quant, and Gemma 3 4B and the 1.5B once each; about 15 minutes in all
  (INFERRED from the tiny runs and the load sizes, not measured). Under the timing lock, with no night job running.

**G2-D is why the tiny fixture is not enough alone.** A tiny random model's scores are far inside ±cap, where
`cap · tanh(score / cap)` is nearly the identity. A dropped cap may leave G2-T green. That is `CLAUDE.md`'s "a minimal repro
can be minimal in exactly the dimension that hides the bug", so the claim rests on G2-K, whose inputs are built to cross
the cap.

**G2-S, the speed guard** (`docs/measurements/noise-registry.md` §8's template):
```
Tier: night.
Instrument: served, two arms (new, base), no peer arm: scripts/bench_peer.py for decode, scripts/bench_prefill_stall.py's
  solo arm for time to first token. A refactor has no in-process A/B: the two builds cannot share a process.
Cells (Metal): decode on the 1.5B at depth 128 and 3900 (the exact kernel, then the split-KV ones), and on Gemma 3 4B at
  depth 128 (head dimension 256, the kernel Gemma 2 shares); time to first token on the 1.5B at K = 3900.
Noise: decode sd 0.0129 (metal-r17-e2e-control-crossbuild, a cross-build A/A, n = 2); time to first token sd 0.0112
  (the chunked-prefill solo arm).
power.py N: `power.py served-decode mac/metal/1.5B/d128 --bar 0.97 --expect 1.0 --paired` -> N = 4 blocks (power 0.87);
  `power.py served-ttft mac/metal/1.5B/k3900 --bar 0.97 --expect 1.0 --paired` -> N = 4 blocks (power 0.94).
  A bar of 0.99 cannot be resolved: power.py wants N = 13 against a feasible 12.
Stopping rule: fixed N = 4 blocks per cell, no early stop.
Rule: PASS if every cell's interval clears 0.97. FAIL if any cell's interval lies under 0.97.
Ambiguous band: an interval that straddles 0.97. Parked for the owner; the refactor stays, and the cell is re-read.
Speed bar: this is a no-regression guard, not a ship bar. The refactor is expected to read 1.00.
Cost: 4 cells x 4 blocks x 2 arms, about 2 min an arm-cell: about 65 min. Queued after the KV1 jobs.
```
- What 0.97 leaves open: a regression under 3% is not seen by this gate. The argument that there is none is G2-I's: without a
  cap the compiled source is unchanged by step 2, and step 1 changes it only by one inlined function.
- CUDA and WebGPU register their own cells and N from nobara's registry entries before their code.

**G2-P, the peer cell** (the same template):
```
Tier: night. One timed run per box, under the timing lock.
Instrument: scripts/bench_peer.py, same-session interleaved, greedy, server restart between cells. Three arms: goinfer on
  the GPU backend, goinfer on the CPU backend (the do-nothing arm: where Gemma 2 decoded before OP1), and Ollama.
Model: gemma-2-2b-it, the GGUF Ollama ships as gemma2:2b, loaded by both engines; 4-bit both sides.
Cells: decode at depth 128 and 2048.
Noise: for the peer ratio, served-decode-peer-arm sd 0.0099 (seed-mac-metal-peer-all-P, n = 2); a cross-session ratio is
  not a ratio.
power.py N: not a bar to resolve for the peer ratio, which is a read. For GPU against CPU the expected ratio is several
  times 1, so the harness default (3 runs x 8) resolves the bands below at N = 1 block.
Stopping rule: one pass at the harness defaults. No second pass unless the first is void.
Rule, GPU against our CPU (this one gates): ship >= 1.02x at both depths; park 1.00-1.02x; off < 1.00x. "Off" means the
  softcap is not declared on that backend, because Gemma 2 would get slower by default.
Rule, against Ollama (this one is a read, and never gates): reported as the ratio with its provenance. It enters
  docs/benchmarks.md under that file's own rules.
Ambiguous band: 1.00-1.02x against our CPU, for the owner.
Cost: 3 arms x 2 depths, about 2 min a cell with restarts, plus idle-gate waits: about 25 min. Queued after G2-S.
```

**Void conditions, for every gate above.**
- A model path under `/srv/models` or `/Volumes/`.
- A resident arm whose `DecodePath()` is not the backend's resident path (a silent CPU fallback passes every parity bar).
- A skipped test counted as a pass. Each gate's record names what ran, by `--- PASS`.
- The two sides of an identity gate run on different boxes, OS builds or drivers.
- For G2-R: the two sides at different quants.
- For the timed gates: an idle-gate or load deviation, a timed run without the lock, weights that are not the same on both
  engines, or a stop before the fixed token count on either engine.

**Order, and what a failure does.**
1. Build G1-K's digest test at base and record base's digests. It is the first OP1 commit on a backend.
2. Step 1, then G1-K, G1-M (tiny), G1-S (by name). Any failure: step 1 is not merged.
3. Step 2, then G2-I (tiny), G2-K, G2-D, G2-T, G2-R. Any failure, or G2-R parked: the feature is not declared.
4. Declare `attn-logit-softcap` on the backend, regenerate the matrix (G2-M), merge.
5. Night: G1-M and G2-I on real checkpoints, the device gate, G2-S, G2-P.
   - An identity or device-gate failure reverts the step that caused it, the next morning.
   - G2-S failing reverts both steps. A second night's run, base against step 1 alone, then says which step it was.
   - G2-P "off" against our CPU removes the declaration, which is one line.

**Metal's half: the day gates, 2026-10-13.** The record is `docs/measurements/op1-metal-2026-10-13/README.md`.

| gate | result |
|---|---|
| G1-K | PASS: all 1,095 kernel digests equal base's. A planted one-ulp change turns every routed kernel's digests red, except the steel prefill kernel's, which a 1% change does |
| G1-M, tiny | PASS for the 25 families that ran; 18 not run (Metal declines them, or the asset is not on the Mac) |
| G1-S, by name | PASS: 140 passed, 91 skipped for want of real checkpoints |
| G2-I, tiny | PASS: digests and the same 25 families identical; step 2 does not touch the two kernel source files |
| G2-K | PASS: 152 cases inside the bar against float64 |
| G2-D | PASS at kernel level: all 152 cases miss the bar with the cap dropped. The tiny fixtures fall to 0.9785 and 0.9799 |
| **G2-T** | **FAIL as registered:** worst logit cosine 0.998997 and 0.998657, against 0.999 |
| G2-R, G2-M | not run: they follow the declaration |

- **What is on main:** the digest instrument; step 1 (`metal/kernels/attn_score_msl.go:AttnScoreMSL`); and the softcap as a
  variant of the kernel source (`metal/kernels/attn_score_msl.go:WithAttnSoftcap`), which no model reaches while the feature
  is undeclared. Gemma 2 still decodes on the CPU under Metal.
- **G2-T: FAILED.** The 0.999 bar was copied from Gemma 1's test without first measuring this fixture's no-softcap baseline
  at int4, which is 0.9991 to 0.9993. The softcap's own contribution is shown by the planted defect: with the cap dropped
  from the Metal build the fixtures read 0.978 and 0.980. The record has the diagnostics.
- **Owner decision OP1-M1 (2026-10-13): the bar stays, and it is not amended against the numbers already seen.** The tiny
  fixtures are now diagnostics, and the real Gemma 2 2B gate (G2-R) decides the declaration (below). The declaration, the
  matrix and the Gemma 2 parity tests stay on the branch `op1-metal-declare-2026-10`, unmerged, until G2-R passes.
- **Step 1 left one kernel out,** as amended above: `attention_fa_blk`. Routing it is owed with a re-run of the R17 grading.
- **Queued on the Mac's night queue 2026-10-13, after the two KV1 jobs,** at main `c02205b8` against `d03da21f`:
  - `op1-identity-real-metal` (35 min): G1-M and G2-I on the real checkpoints in `~/models`;
  - `op1-gate-gpu-metal` (25 min): the Metal device gate at the merge;
  - `op1-speed-metal` (75 min): G2-S, by `docs/measurements/op1-metal-2026-10-13/run-op1-night.sh`.
  - The queue is 4 h 35 min. A job that would end after 06:30 is not started and carries to the next night.
- **G2-S runs two cells the registration did not have,** reported and not gating: decode on the 0.5B at depth 128 and 3900.
  At depth 3900 the 0.5B takes `attention_fa_blk64`, the one split-KV kernel step 1 routed; the registered 1.5B cell at that
  depth takes the unrouted block kernel.
- **Not queued:** G2-R (no declaration, and the checkpoint is not on the Mac) and G2-P, the peer cell (no declaration, and
  the harness has no Gemma 2 key, calibrated prompts or Ollama import yet).
- **Graded 2026-10-14: all three PASS.**
  - G1-M and G2-I on real checkpoints: byte-identical on the three cells that ran (Qwen2.5-Coder 0.5B and 1.5B, Qwen3 0.6B).
  - The Metal device gate: 8 pass, 0 fail.
  - G2-S: every registered cell's interval clears 0.97; the lowest bound is 0.9919.

**What decides the declaration (owner's ruling, 2026-10-13): no G2-T2. The tiny fixtures are diagnostics, not a gate.**
- **Why not G2-T2.** Metal cannot run f32 weights: a resident build declines them by design
  (`metal/model.go:f32Projection`). The nearest runnable setting, int8int8, was already read in G2-T's diagnostics
  (0.999255 and 0.999334), so a gate on it could not fail on anything new.
- **What the tiny fixtures show, kept as a record:**
  - the planted defect separates: with the cap dropped from the Metal build the fixtures read 0.978 and 0.980;
  - the int8int8 readings are 0.999255 and 0.999334. That setting runs native int8 weights
    (`metal/model.go:w8Weights`) with the library compiled without fast math (`metal/model.go:w8PreciseMath`);
  - **the tanh:** at int4 the library is compiled with fast math, so the capped function's plain `tanh` is the fast one
    (INFERRED from the math-mode rule; the compiled code was not disassembled). An explicit `precise::tanh` moved
    `gemma2-tiny` by 0.00005 and `gemma2-hd-tiny` by nothing.
- **The softcap's correctness evidence is the kernel-level float64 checks:** all 152 cases pass, and all 152 fail with
  the softcap dropped (G2-K, G2-D).
- **The real Gemma 2 2B parity gate, G2-R, decides the declaration.** Its baseline step is below.

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
  - Prior art: what §10a lists as not read.
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

## 10. Prior art, read in OP0

Each line was written before the reading and is corrected here by it. §10a has the reading, and what RF0 still has to read.

- **llama.cpp and ggml:** a closed set of 101 operation kinds, mostly primitives, with some fused and architecture-specific ones.
  - Each backend answers per graph node whether it supports it, and an unsupported node runs on the CPU. No model is refused.
  - The Metal and CUDA backends carry explicit fusion tables over those primitives.
  - One C++ builder file per architecture (160 of them) sits over shared helpers with parameter enums.
  - `llama_batch` is ragged as an interface: a position, sequence ids and an output flag per token. It is not ragged in
    execution: a batch is split into micro-batches, and attention is dense over a contiguous cell range with a mask.
- **Ollama's Go engine** was model structs with an imperative `Forward` over an `ml` backend that mirrored ggml. It was removed
  between 2026-05-29 and 2026-09-16; GGUF models now run through llama-server. What remains is a Go engine over MLX, of the same
  shape, with hand-written kernels only for the recurrent mixers.
- **vLLM V1's scheduler:** it has no prefill or decode phase; each step assigns tokens so that every request's computed count
  catches up, within one token budget. It gives decode rows no priority: running requests are scheduled in list order, and an
  optional threshold is the guard against a prefill taking the budget.
- **Sarathi-Serve** (OSDI 2024): chunked prefills with stall-free batching. A batch takes the running decodes first, then
  partly finished prefills, then new requests, within a budget profiled against the time-between-tokens target.
- **FlashAttention's variable-length interface and FlashInfer:** ragged and paged attention kernels. FlashAttention marks
  ragged rows with cumulative lengths and pages KV through a block table. FlashInfer uses a compressed-row page table, plans on
  the CPU and runs a fixed-grid kernel, and still ships decode and prefill as separate kernels.
- **MLX and MLC:** the path not taken (E1). MLX is a lazy graph built on every call, with compilation optional and custom
  kernels allowed. MLC compiles a model to an IR per target, with one entry function per call shape. Both lose attention
  softcapping from their fused attention: mlx-lm composes Gemma 2's by hand, and MLC's Gemma 2 ignores it.

## 10a. What we take and what we leave (read for OP0, 2026-10-13)

Everything below was read on 2026-10-13 at the revisions in the last table, from shallow clones and the papers' text. Four of
§10's six lines were corrected and the other two gained detail: see "What the OP0 reading corrected in §10" at the end.

| source | what we take | what we leave | basis |
|---|---|---|---|
| llama.cpp and ggml | a per-backend fusion table that the planner and the encoder both read; a probe for each fused operation, with a composed fallback; softcap, sinks and a KV bound as parameters of the one attention operation; dropping non-output rows before the last layer's FFN; the extended batch entry (token, embedding or both; a position tuple; an output flag) | per-node CPU fallback; a primitive graph that backends pattern-match to recover structure; dense attention with a mask; the equal-length micro-batch split | READ |
| Ollama | an optional Go interface for a fused operation with a composed fallback; per-layer operator interfaces chosen at load; a kernel type with per-backend sources and a graph fallback | a ggml-mirrored tensor interface as the model-definition surface (deleted upstream); a padded (B, L) batch | READ: the Go engine at `f63eea3d`, the MLX engine at main |
| vLLM V1 | "catch up the computed count" as the one rule; one token budget; a per-request cap so one prefill cannot take it; a flat token batch with query offsets, sequence lengths and a block table, built once per step | scheduling the running list in list order; one module per architecture; preemption | READ |
| Sarathi-Serve | the fill order; a budget profiled once against the time-between-tokens target; chunk sizes rounded to the kernel's tile; the measured cost of chunking | its capacity multipliers (datacenter GPUs against a 2024 vLLM) | READ (arXiv v3) |
| FlashAttention | the interface vocabulary: cumulative lengths for ragged queries, per-row cache lengths and a block table for paged KV; softcap as a compile-time flag | a rectangular block table; FlashAttention 2's page size, a multiple of 256 | READ |
| FlashInfer | the compressed-row page table; plan on the CPU and run on the GPU with a fixed grid and fixed buffers, for CUDA-graph replay; its variant hooks as the list of places a softcap, a window and a sink touch a kernel | compiling CUDA source strings at run time; separate decode and prefill wrappers | READ (code and paper v2) |
| MLX, MLC | MLX's custom-kernel escape inside a lazy graph | a lazy graph or a compiled IR as the unit a model is defined in | READ (docs and a few source files); TVM not read |

### llama.cpp and ggml

[Repository](https://github.com/ggml-org/llama.cpp) at `091b6a7f` (master, 2026-10-13): `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-backend-impl.h`,
`ggml/src/ggml-backend.cpp`, `ggml/src/ggml-metal/ggml-metal-fusion.h`, the CUDA backend's fusion code, `src/llama-graph.h`, `src/llama-graph.cpp`,
`src/llama-context.cpp`, `src/llama-kv-cache.cpp`, `src/models/`, `include/llama.h`, `tools/server/server-context.cpp`, `common/common.h`.

READ:
- **The operation set.** `enum ggml_op` has 101 entries (`static_assert(GGML_OP_COUNT == 101)`, counting `GGML_OP_NONE`). It mixes primitives (`MUL_MAT`, `SOFT_MAX`, `RMS_NORM`,
  `ROPE`, `GET_ROWS`) with fused and architecture-specific kinds: `FLASH_ATTN_EXT`, `GLU`, `MUL_MAT_ID` (MoE), `SSM_CONV`, `SSM_SCAN`, `GATED_DELTA_NET`, `GATED_LINEAR_ATTN`.
- **Attention is one operation with parameters.** `ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap)`, plus `ggml_flash_attn_ext_add_sinks` and a setter for a KV bound.
  `build_attn_mha` has both forms: the one fused node, or `mul_mat`, an optional scale-tanh-scale softcap, `soft_max_ext`, `mul_mat`.
- **Support is a function per node.** `supports_op(dev, op)` in the backend interface. `ggml_backend_sched_split_graph` assigns a backend to each node in passes, splits the graph and copies tensors
  between the pieces; the last backend is "assumed CPU".
- **Fusion is explicit, per backend.**
  - Metal: `ggml-metal-fusion.h` opens "single source of truth for the fusions supported by the Metal backend". Its table has 27 rows over 10 fusion ids, each an operation sequence and a check.
  - CUDA: `ggml_cuda_try_fuse` matches fixed operation lists, among them `{MUL_MAT, MUL_MAT, GLU}`, `{RMS_NORM, MUL, ROPE}` and the softcap's `{SCALE, UNARY(tanh), SCALE}`.
- **A fused operation is probed, not assumed.** `llama_context::resolve_fused_ops` reserves a probe graph for each fused kind. If the node did not land on its layer's device, the kind is
  disabled and the builder emits the composed form.
- **Builders.** `src/models/` has 160 `.cpp` files, one architecture each, over shared helpers: `build_norm`, `build_qkv`, `build_ffn`, `build_moe_ffn`, `build_attn_mha`, `build_rs`.
  Parameters are enums: `llm_norm_type`; `llm_ffn_op_type` with ten values and an unset sentinel, among them `GEGLU`, `GEGLU_ERF`, `RELU_SQR` and `SWIGLU_OAI_MOE`; `llm_ffn_gate_type`.
- **Layers that differ are C++ control flow.** In `lfm2.cpp`: `hparams.is_recr(il) ? build_shortconv_block(...) : build_attn_block(...)`.
- **`llama_batch`** has `token`, `embd`, `pos`, `n_seq_id`, `seq_id`, `logits`. Its comments: "can contain input about one or many sequences"; of `logits`, "TODO: rename this to "output"".
  A newer extended batch adds token and embedding entries in one batch and a position tuple "in order [t, y, x, z]".
- **Execution is not ragged.** With one stream per sequence the batch is split by `split_equal`, which takes the same number of tokens from every sequence. The attention mask is filled per
  cell as 0 or `-INFINITY` by sequence, causality and window, over a contiguous view of the cells.
- **Output rows.** At the last layer the builder gathers only the rows that asked for output, before that layer's FFN.
- **The server.** `update_slots` first adds each generating slot's sampled token, then walks the slots "without exceeding n_batch" to add prompt tokens. Defaults: `n_batch` 2048, `n_ubatch` 512.

INFERRED:
- **Our operation kinds sit at the level of llama.cpp's builder helpers and their enums, not at `ggml_op`.** Their FFN enum already separates `GEGLU` from `GEGLU_ERF`, the split OP0 made.
  Its zero value is a sentinel an architecture must replace; our `ActKind`'s zero value is tanh-GELU, so a forgotten activation here is silent.
- **Both ggml GPU backends arrived at explicit fusion tables,** as matchers over a primitive graph. A layer program that already names `[norm, proj]` needs the table and not the matcher.
- **Admission differs in kind.** ggml moves a node to the CPU and never refuses a model. Its fused-operation probe is the nearest thing to our whole-model admission, per fused kind.
  A per-operation CPU fallback is what OP4's "admission by operation" would make possible; this doc does not propose it.
- **`llama_batch` is RF's row as an interface and not as a kernel.** With one stream per sequence, a 500-token prefill and three decode tokens become one micro-batch of four tokens and then
  prefill-only ones. RF2's ragged attention is what llama.cpp does not have.
- **The server fills decode first and does not budget it.** Prompt slots fill in slot order, so the first prefilling slot takes what is left.

### Ollama

[Repository](https://github.com/ollama/ollama). Main is `8718c2bd` (2026-10-13). The Go engine was read at `f63eea3d` (2026-05-24), its last complete revision.

READ:
- **At main the Go engine is gone.** `model/` holds only parsers and renderers, and `ml/` only device and path files. Commits `9db4bdba` (2026-05-29), `7b22ac96` (2026-07-02) and
  `bef41f71` (2026-09-16) removed the runner, `ml/nn`, the model interface and `ml/backend.go`. `2e036e7c` (2026-09-16) moved the MLX engine to top-level `mlx/` and `mlxrunner/`.
- **At `f63eea3d`, `ml/backend.go`** has `Backend`, `Context` and a `Tensor` interface with about 66 operation methods that track ggml (`Mulmat`, `MulmatID`, `RMSNorm`, `SSMConv`, `SSMScan`).
- **Fused attention is an optional interface.** `ScaledDotProductAttention` is type-asserted by `nn.Attention`, which otherwise composes it from matmul, scale, mask, softmax and matmul.
  The ggml implementation passes no softcap.
- **A model definition** is a Go struct with tagged weight fields, filled by reflection, and a `Forward(ml.Context, input.Batch)` that builds a lazy graph. 24 architectures were registered.
- **Layers that differ are Go interfaces chosen at construction:** an `MLP` interface with sparse and dense forms (qwen3); an `Operator` set to a short convolution or attention (lfm2).
- **`input.Batch`** has `Inputs`, `Outputs` (indices of the inputs to return output for), `Positions`, `Sequences` and `Multimodal`.
- **The MLX engine at main** keeps the shape: a model is `LoadWeights`, `NewCaches`, `Forward`, `Unembed`, with 17 architecture packages. Its batch is padded: `InputIDs` is "(B, L)".
- **`gpuKernel`,** in `mlx/gpu_kernel.go`, is "a custom kernel with per-backend sources and a graph fallback". It is used for the gated delta rule and the Mamba-2 scan.

INFERRED:
- **§10's line described an engine that no longer exists,** and whose unit was a struct with an imperative `Forward`, never an operation list.
- **The optional fused interface is the mechanism goinfer's resident seam already uses.** Ollama's shows the cost when the fused signature lacks a parameter, here the softcap.
- **Hand-written kernels survive in the MLX engine only for the recurrent mixers.** Those are the operations OP2 and OP5 have to write as well.

### vLLM V1's scheduler

[Repository](https://github.com/vllm-project/vllm) at `a1f51a02` (main): `vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/sched/output.py`, `vllm/config/scheduler.py`,
`vllm/v1/attention/backend.py`, `vllm/v1/attention/backends/utils.py`, `vllm/model_executor/models/registry.py`.

READ:
- **The wording,** in `Scheduler.schedule()`: "There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the num_computed_tokens and num_tokens_with_spec."
  And: "At each step, the scheduler tries to assign tokens to the requests so that each request's num_computed_tokens can catch up its num_tokens_with_spec. This is general enough to cover
  chunked prefills, prefix caching, speculative decoding, and the "jump decoding" optimization in the future."
- **The budget.** `token_budget = self.max_num_scheduled_tokens`, which defaults to `max_num_batched_tokens` (2048).
- **The order.** "First, schedule the RUNNING requests." walks `self.running`, a list in admission order, while the budget lasts. "Next, schedule the WAITING requests." runs only if nothing was preempted.
- **The guard.** `long_prefill_token_threshold` (default 0, off) "exists to stop a long prefill from starving other requests of the token budget". `enable_chunked_prefill` is on by default.
- **One forward.** The scheduler's output is a token count per request. The attention metadata is `query_start_loc`, `seq_lens`, a block table and a slot mapping.
  The rows whose logits are read are `query_start_loc[1:] - 1`.
- **Kernel dispatch still splits.** `reorder_batch_to_split_decodes_and_prefills` reorders a batch into decode, extend and prefill regions, and the FlashInfer backend keeps separate
  prefill and decode wrappers.
- **Coverage.** `vllm/model_executor/models/` has 284 Python files, and the registry has 401 names. It is one module per architecture over shared layers.

INFERRED:
- **§10's claim holds on the point that matters for RF:** prefill and decode are not kinds of step.
- **vLLM does not put decode first.** With the threshold off, a prefill early in the running list can take the whole budget. RF3's "decode rows first" is Sarathi-Serve's rule, not vLLM's.
- **Even with one scheduler, the kernels are split by row kind.** RF's one call need not mean one attention kernel.

### Sarathi-Serve

[arXiv 2403.02310](https://arxiv.org/abs/2403.02310) v3, 17 Jun 2024 (OSDI 2024): the abstract, §4.2, §4.3 and §5.4.1.

READ:
- **The two ideas,** from the abstract: "chunked-prefills which splits a prefill request into near equal sized chunks and creates stall-free schedules that adds new requests in a batch
  without pausing ongoing decodes."
- **The fill order** (§4.2): "we first pack all the running decodes in the next batch … After that, we include any partially completed prefill … Only after all the running requests have been
  accommodated, we admit new requests."
- **The budget** (§4.3) comes from "a one-time profiling of batches with different number of tokens and setting the token budget to maximum number of tokens that can be packed in a batch
  without violating TBT SLO".
- **The cost of chunking** (§4.3): "if a prefill sequence is split into N chunks, then the first chunk's KV-cache is loaded N − 1 times". And: "using chunk size of 257 can increase prefill
  time by 32% compared to that with chunk size 256".
- **Measured** (§5.4.1, Yi-34B): "even with the smallest chunk size of 512, we observe a moderate overhead of at most ∼25%. Whereas with the larger token budget of 2048, chunked prefills
  have almost negligible overhead."
- **Headline:** 2.6× the serving capacity of vLLM on Mistral-7B with one A100, and up to 3.7× on Yi-34B with two.

INFERRED:
- **RF3's scheduler is Sarathi-Serve's with one tier merged:** Sarathi separates partly finished prefills from new requests. RF3 should keep the tier, so a prefill in progress finishes
  before another starts.
- **A "fill what is left" chunk must be rounded to the kernel's tile,** as the 257-against-256 result shows. §5.1 does not say so yet.
- **The capacity multipliers do not transfer.** They are for many users on datacenter cards. The stall figure RF3 is gated on is ours (§5.4).

### FlashAttention

[Repository](https://github.com/Dao-AILab/flash-attention) at `1fc02211` (2026-10-12, package 2.8.4): `flash_attn/flash_attn_interface.py`, `csrc/flash_attn/flash_api.cpp`, `hopper/flash_attn_interface.py`.

READ:
- **Ragged rows.** `flash_attn_varlen_func` takes `q` as "(total_q, nheads, headdim)" and `cu_seqlens_q` as "(batch_size + 1,), dtype torch.int32. The cumulative sequence lengths".
  It also takes `softcap=0.0`, a window, and `block_table`.
- **Paged KV in version 2.** The table is rectangular, batch × the most blocks per sequence, and "Paged KV cache block size must be divisible by 256".
- **`flash_attn_with_kvcache`** appends the new K and V in place at `cache_seqlens` and applies RoPE in the kernel.
- **Version 3** (`hopper/`) adds a page table of any block size and ragged queries to that call, with a scheduler-metadata step computed ahead.
- **Softcap** is a template flag in version 3, and it is an input to the choice of tile size.

INFERRED:
- **Version 3's cache call, not version 2's varlen call, is the shape of RF2's kernel:** ragged queries over paged KV, with the append.
- **A softcap is one tanh, and still a separately compiled kernel variant that can change the tiling.** OP1 should expect a variant per kernel and not a run-time branch.
- **Version 2's kernels could not be taken as they are:** its page size must be a multiple of 256, and the KV doc's block is 64. Version 3 lifts the limit.

### FlashInfer

[Repository](https://github.com/flashinfer-ai/flashinfer) at `53e3168a` (main): `flashinfer/decode.py`, `flashinfer/prefill.py`, `flashinfer/pod.py`, `docs/tutorials/kv_layout.rst`,
`include/flashinfer/attention/scheduler.cuh`, `variant_helper.cuh`, `variants.cuh`, `flashinfer/jit/attention/variants.py`. Paper: [arXiv 2501.01005](https://arxiv.org/abs/2501.01005) v2, §3.1, §3.2.3, §3.3, appendix D.

READ:
- **The page table is a compressed-row sparse matrix:** `indptr`, `indices` and `last_page_len`. A sequence's length is `page_size * (pages - 1) + last_page_len`.
- **Plan, then run.** `plan(...)` runs on the CPU once per step; `run(q, kv_cache)` runs per layer, and the plan "can be reused across multiple batch decode attention calls (e.g. different
  Transformer layers)".
- **Decode and prefill are separate wrappers,** and a third (`flashinfer/pod.py`, arXiv 2410.18038) exists for mixed batches.
- **CUDA graphs.** With `use_cuda_graph`, "The `batch_size` cannot change during the lifecycle of this wrapper", and queries per row are "part of the frozen shape (like the batch size)".
  The grid is padded to a fixed size with a validity mask. From the paper: "plan function is not captured by CUDAGraph because it's on CPU".
- **Determinism.** A fixed split size gives "batch-size invariant outputs", but "compatibility with CUDA graph is NOT guaranteed".
- **Variants are hooks:** a query transform, a key transform, a logits transform, a logits mask, an update of the running softmax state, and an output transform.
  - A softcap is the logits transform with two constants refolded: the kernel computes `tanh(logits * sm_scale / cap)`.
  - A sliding window is the mask predicate.
  - An attention sink uses the softmax-state update and the output transform, and has a second implementation for the FlashAttention 3 template.
- **The paper on variants** (§3.2.3): "we specialize the kernel for each variant for maximum performance, and the number of variants is growing rapidly".

INFERRED:
- **Ragged queries over paged KV are one data model and still three kernels.** RF2's kill rule, which keeps the old decode call, has precedent.
- **For CUDA's graph replay (HL, RF2):** the batch size and the queries per row are fixed per captured graph, lengths travel in buffers at fixed addresses, and the grid is padded.
  The plan on the host and a fixed-shape kernel is the split our decode graphs need.
- **FlashInfer does not promise batch-invariant output and graph replay together.** RF's one kernel-class rule (§5.1) asks for both on CUDA, so RF2 has to measure it.
- **The six hooks are OP1's checklist** for each hand-written kernel in §4.5's count: where the softcap goes, and that it must compose with the window and the sink already there.

### MLX and MLC

MLX at `243bdf09`, mlx-lm at `ba8a08fd`, MLC-LLM at `8978ea9a`. Docs and a few source files; TVM itself was not read.

READ:
- **MLX is lazy and does not compile by default.** Its docs: "no computation actually happens. Instead a compute graph is recorded", and "Currently, MLX does not compile and rerun compute
  graphs. They are all generated dynamically." `mx.compile` is opt-in.
- **MLX owns the fused kernels** (`rms_norm`, `rope`, `scaled_dot_product_attention` with a mask and sinks) and offers `metal_kernel` and `cuda_kernel` for a custom one.
  Its fused attention has no softcap parameter.
- **mlx-lm** has one Python file per architecture (136 files in its models directory). Its Gemma 2 writes attention by hand: `scores = mx.tanh(scores / self.attn_logit_softcapping)`.
- **MLC** defines a model as a Relax module (44 architecture directories) that exports one entry function per call shape: `embed`, `prefill`, `decode`, `batch_prefill`, `batch_decode`,
  `batch_verify`. Attention is a call into the paged KV cache's runtime.
- **MLC's Gemma 2** says: "NOTE: We ignore attn_logit_softcapping in the gemma2 implementation for now."

INFERRED:
- **The cost E1 asks about shows on the exact feature OP1 adds.** A framework-owned attention kernel without a softcap parameter leaves two choices: compose attention from primitives and
  lose the fused kernel (mlx-lm), or drop the feature (MLC).
- **MLC is the opposite of RF:** it compiles one function per call shape, which is the seam RF removes.

### What the OP0 reading corrected in §10 (applied above)

- **llama.cpp, "a fixed operation set implemented per backend".** The set is closed, but at primitive granularity; support is per node with a CPU fallback, not per model; and the
  backends fuse through explicit tables. "`llama_batch` already is the ragged call" is true of the interface and false of the execution.
- **Ollama, "model definitions in Go over an `ml` backend. Confirm the shape".** The shape was right and the engine is deleted. Ollama today is llama-server plus a Go engine over MLX.
- **vLLM, "a step is a token budget across requests".** The wording about phases is confirmed, with the quote. The budget does not favour decode: the order is list order.
- **Sarathi-Serve.** Confirmed. Added: three tiers in the fill order, and the measured cost of chunking.
- **FlashAttention and FlashInfer.** Confirmed. Added: the paged ragged call is FlashAttention 3's; FlashInfer keeps decode and prefill kernels apart; a softcap is a compile-time variant.
- **MLX and MLC, "lazy graphs and compiled IR".** MLX is a lazy graph that is not compiled by default, and it allows custom kernels. MLC is compiled, one function per call shape.

### Revisions read

| source | revision | date |
|---|---|---|
| llama.cpp | `091b6a7fc561b78da90a0ed2d524ec995458c657` (master) | 2026-10-13 |
| Ollama | main `8718c2bd2363c3f9f422d82bbad322bc2b791d60`; Go engine at `f63eea3d273816ffc27fbeb0662ab44d236abc45` | 2026-10-13; 2026-05-24 |
| vLLM | `a1f51a0226a8f5e97a5ed340e091a447c8b35911` (main) | 2026-10-13 |
| FlashAttention | `1fc0221183e799363001394a12dd9e78e15f5dda` | 2026-10-12 |
| FlashInfer | `53e3168a630bc9ca55f0321fb3cee50242770b1c` (main) | 2026-10-13 |
| MLX; mlx-lm; MLC-LLM | `243bdf09c0cccbb85ca6a686dab3cc0102cf1f7e`; `ba8a08fd2ed72bbc53de84f89933f91fe4743e4a`; `8978ea9aec626d64096aaa92fbc89bda2e3ea802` | 2026-10-13; 2026-10-10; 2026-10-07 |
| Papers | arXiv 2403.02310 v3 (Sarathi-Serve); arXiv 2501.01005 v2 (FlashInfer) | read 2026-10-13 |

The llama.cpp, Ollama and vLLM mains had moved since the KV doc's reading the same day (`77c4540c`, `665f9e07`, `ac71b175`); those revisions were not re-read. The vLLM and FlashInfer
commits are dated 2026-10-14 in their own time zones.

**How it was read.** A delegated reading pass made the clones and reported each fact as read or inferred. Eleven of its facts were then checked against the clones by search, and all eleven matched:
the operation count, the Metal fusion header, the `logits` comment, the softcap parameter, the builder count, the FFN enum, the vLLM scheduler comment and its two ordering comments, both
Gemma 2 softcap lines, FlashInfer's softcap constant and FlashAttention's 256 check. The rest stands on the pass's own reading.

**Not read, and left for RF0:** TVM and Relax, so MLC's attention kernels are unverified; FlashInfer's and FlashAttention's kernel bodies and the mixed-batch (POD) internals; vLLM's model
runner beyond the logits rows, and its waiting-queue loop; llama.cpp's Vulkan fusion code and its hybrid and recurrent memories; the Ollama MLX runner's scheduler; the Sarathi-Serve code.

## 11. What this absorbs and changes elsewhere

- **`docs/tasks/task-kv-architecture-2026-10.md`:**
  - KV4/KV5 supply the tables RF2/RF4 read;
  - its KV11 takes the expert half of memory;
  - OP4's planner prices through its §5.10.
- **`docs/tasks/task-turn-telemetry-2026-09.md`:** RF0 builds its X1.
- **`docs/tasks/parked/task-concurrency-2026-09.md`:** "spec inside a batch" becomes RF6. MC5's continuous batching stays
  parked (E6).
- **`docs/tasks/task-recompute-audit.md` §5:** its class becomes a planner property (OP4).
- **`docs/hardware-matrix.md`:** decline reasons and the Phi-3/4 configuration note (OP0, done).
- **`docs/ARCHITECTURE.md`:** §1 ("One decoder, read from a descriptor") and §3 ("Registration and the runner seam",
  "Admission: which models run resident") are rewritten as phases land. §3's count of the optional interfaces was corrected to 34
  in OP0's doc pass.

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
  - added by OP0: `decoder/residency.go:Architecture.runnerShapeDecline`, `decoder/features.go:residentGateReasons`,
    `decoder/features.go:Architecture.residentFeatures`, `decoder/residency.go:Model.GatedActResident`;
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
