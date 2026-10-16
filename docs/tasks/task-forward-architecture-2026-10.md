# Task: one forward architecture — models as operation lists, one ragged forward pass (OP0–OP6, RF0–RF6, HL) — 2026-10

> **Status: OP0 DONE 2026-10-13. OP1's gates are registered (§4.6). OP1 on Metal: step 1 and the softcap's mechanism are on main
> and pass their day gates; the softcap is NOT declared, because its tiny parity gate failed as registered (§4.6, the last part);
> the real Gemma 2 2B gate decides it. The three night gates passed on 2026-10-14 (the record's last section). OP1 on CUDA: steps 1 and 2 are built and the softcap is declared on branch `op1-cuda-declare-2026-10`; the day gates pass (`docs/measurements/op1-cuda-2026-10/README.md`), the night gates are not run (§4.6a). OP1 on WebGPU, and every other phase, are not started. OP3's design and gates are registered (§4.8, 2026-10-16), before any OP3 code.** OP0's summary is under §4.4, its feature census is §4.5 and its prior-art reading is §10a. Written on the owner's ask after a Cowork conversation the same day ("we have
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
- **Backend findings are §4.5's last list.** The CUDA double softcap was confirmed and fixed; a Cohere step row's one-ulp
  LayerNorm difference was found by that fix and is open, as are the other two.

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

**Found in backend modules** (a backend change is a two-step tag; each needs its owner's call). The first two were run; the
rest are read from the code and not fixed.
- **CUDA: a batched-step row that returns logits got the final softcap and the logit scale twice. Confirmed and fixed
  2026-10-14.** `cuda/prefill.go:cudaResident.batchedHeadFull` applied both to a step row, and
  `cuda/prefill.go:cudaResident.prefillCore`'s closing loop applied both again to every logits row; the closing loop is now
  the only place. It touched concurrent conversations that sample on a model with a final softcap (dense Gemma 4) or a
  logit scale (Command-R, Granite): those models always draw on the host. `TestCUDAStepBatch_matchesForward` now runs
  `gemma4-dense-twogeom-tiny` (softcap) and `granite-dense-tiny` (logit scale) and compares a step row's logits with
  `Forward`'s bit for bit; before the fix it read logit 1 at -0.2842141 against -0.2842226 (Gemma 4) and logit 0 at
  -0.0024854098 against -0.014912458 (Granite, the scale of 1/6 applied twice).
- **CUDA: a Cohere step row is not bit-identical to `Forward`, found by the fix above and not fixed.** With the double
  transform gone, `cohere-tiny` and `cohere2-tiny` still differ from `Forward` by one ulp in a logit (cohere-tiny at its
  third step, cohere2-tiny at its first). The batched LayerNorm, `cuda/layernorm_quant.cu`'s `layernorm_quant_batched`,
  rounds `((x-mean)*inv)*w`, while decode's `layernorm_quant` in `cuda/glue.cu` rounds `((x-mean)*w)*inv`, so a normed value
  can differ in its last bit and move an int8 code. The step, `PrefillLastN` and speculative verify on Cohere inherit it.
  `cuda/prefill.go:cudaResident.bLayerNormQuantB`'s fallback, which loops decode's kernel per row, would be exact. The
  batched kernel is shared with the SigLIP tower, so the fix is the CUDA owner's call.
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

**G2-R moved to the night queue (owner's decision, 2026-10-14).** The registration says "Tier: by day"; by day the Mac
cannot admit the loads.
- **Run 1, 08:49:** the fit guard refused the first load of the 2B (6.7 GB needed, 6.4 GB available with the owner's
  apps open). Nothing was measured.
- **Run 2, 08:50, with the fit guard bypassed on the owner's word** (`GOINFER_NO_FIT_GUARD=1`):
  - the hook check passed: removing the softcap changes the CPU's logits;
  - arm B at int8int8 read a min cosine of 0.982217, with argmax equal at 22 of 24 positions;
  - Metal's resident memory guard then declined the int4 build (3.63 GB needed, a budget of 2.66 GB), and the test was
    skipped. **The run is void, and the 0.982217 is recorded as not a result:** the registration needs B, G and P in one
    session.
- **Tonight, as `op1-g2r-metal`:** all three arms in one session, by
  `docs/measurements/op1-metal-2026-10-13/run-g2r-night.sh` at the declaration branch's `78633e7f`. No guard is bypassed.
  The gate is not timed: night only means nothing else holds memory.

**G2-R amendment, 2026-10-15 (owner), before any further run: the CPU reference runs in its own process.**

**Why.** On the night of 2026-10-14 the fit guard refused the job's first load, the 2B on the CPU: 6.7 GB priced, with
6.6 GB available. No guard is bypassed (owner, 2026-10-15). Within one process, `residentParityAt` holds a Metal model
and a CPU model at once, so each load's guard sees what the other holds.

**What changes.** The job runs two processes, one after the other, at the same build:

1. **The CPU reference.**
   - It loads only CPU models, one at a time, and computes everything the arms read on the CPU:
     - the hook check's two logit sets;
     - for each of B (softcap off), G and P (softcap on), at int8int8 and int4: the token sequence (the seed, then the
       CPU's greedy continuation, 24 steps) and the CPU's logits at every step;
     - P's score statistics and its capped-against-uncapped logits;
     - the control's and Gemma 3's tokens and logits.
   - It writes them to one file under the job's output directory, with each checkpoint's sha256 and the build's commit.
2. **The Metal arms.**
   - They load only Metal models, one at a time.
   - They feed each arm the saved tokens (greedy lockstep off the CPU, as before) and compare their logits with the
     saved CPU logits.
   - B's stop rule is applied before G and P run, as before.

**Unchanged:** the arms and their order of reading, the prompts, the 24 steps, the metric (the min logit cosine from
position 2, and argmax), the bars (0.88; the control's 0.95; the 0.03 band), B's 0.001 stop margin, and every void rule.

The CPU logits are the same computation as before (`decoder.Load` on the CPU at the arm's quant, `ForwardForTest` over
the same tokens), saved as raw float32 bits.

**Void, in addition:**

- the saved file's checkpoint hashes or commit differ from the Metal process's;
- an arm is missing from the file;
- the file does not exist.

**The memory this buys, and what it does not.**

- **Before, in one process:** the largest moment was a Metal model and a CPU model held together, about 10 GB in all.
  Metal's own guard declined the 2B's int4 build at 3.63 GB needed against a 2.66 GB budget, with the CPU model beside it.
- **After:** the peak is the larger of the two processes, about 7 GB, and each load's guard sees only itself.
- **The fit guard still prices the CPU process's largest loads at about 6.7 GB** (estimated from the guard's formula:
  resident weights + KV at its fitted context + the `.gguf` mapped during the load):
  - the 2B at int8int8: 2.5 + 1.6 + 2.6 GB;
  - Gemma 3 4B at int8int8, which the old order never reached: about 4 + KV + 2.5 GB.
  - The guard's budget is 70% of available memory, so these loads need about 9.6 GB available. The two nights on
    record had 6.6 GB and 7.8 GB.
  - Splitting the processes alone therefore does not clear the guard on such a night, and a refused load is void.
  - What would clear it, for the owner to choose:
    - a CPU process that loads `.giw` sidecars: no mapping term, so about 4.1 GB priced for the 2B and about 4.5 GB for
      Gemma 3, which need roughly 6 to 6.5 GB available;
    - or a night with 9.6 GB free.

**G2-R amendment 2, 2026-10-15 (owner's choice from amendment 1's options), before any run: the CPU arms load `.giw` files.**

**What changes.**

- The CPU process loads each CPU model from a CPU `.giw` (`cmd/prequant -quant <q>`, this box's own target), built from
  the same `.gguf` at each quant the arms use:
  - the 2B at int8int8 and int4;
  - Gemma 3 4B at int8int8;
  - the dense 1.5B at int8int8, and at int4 for the check below.
- The night job builds any that are missing, at the job's revision, under `~/goinfer-bench/op1-metal-2026-10/giw/`, and
  records each one's sha256 in the reference file.
- The Metal process still loads the `.gguf` files, and every guard stays as shipped.

**Why it is the same computation.** A `.giw` holds the weights already quantized by the same quantizer a direct `.gguf`
load runs at load time.

- On record: greedy output byte-identical between the two routes (`docs/tasks/task-never-swap-2026-09.md`, S1), and
  CPU logits through the sidecar bit-identical to the direct load's (`docs/tasks/task-multimodal-support-2026-10.md`).
- **Checked again each run, as a precondition.** Before any arm, the CPU process loads the dense 1.5B both ways (its
  `.gguf` fits the fit guard directly) at int8int8 and int4. It runs the gate's prompt and 24 steps, and requires the
  logits to be equal bit for bit at every step. **A difference voids the run.** It is not a fail of G2-R.

**Memory.** By the guard's formula, the CPU process's largest priced loads drop to about 4.1 GB (the 2B at int8int8)
and about 4.5 GB (Gemma 3 4B), because there is no `.gguf` to map during the load. That needs roughly 6 to 6.5 GB
available, where the two nights on record had 6.6 and 7.8 GB.

**Disk.** The four bundles take about 10.6 GB. The job refuses to build when less than 14 GB is free.

**Unchanged:** amendment 1's split, the arms, prompts, bars, voids and the order of reading.

**G2-R amendment 3, 2026-10-15 (owner-approved), before any run: the Metal arms load Metal `.giw` files, and a refused
load is VOID.**

**Why.** On 2026-10-15 the CPU process passed (its precondition held). Then the fit guard refused the Metal process's
first load, the 2B's `.gguf`: 5.5 GB priced against a 4.9 GB budget, with 7.0 GB available. The script labelled the run
FAIL. A refused load measures nothing, so that label was wrong: it was VOID.

**What changes.**

- **Every Metal load reads a `.giw`** of the same `.gguf` at the arm's quant, with no `.gguf` mapped during the load.
  - **int8int8 has one bundle for both targets.** `cmd/prequant -target metal` and the CPU's own target wrote the dense
    1.5B's int8int8 bundle byte for byte the same (`cmp`, 1,787,283,843 bytes, 2026-10-15). The target changes only int4
    tensors. So the Metal process reads amendment 2's int8int8 bundles, under a `.metal.giw` hard link the job makes.
  - **int4 needs its own Metal bundle**, built with `-target metal`: the CPU's int4 bundles hold the row4-only layout,
    which Metal cannot read. Two are added: the 2B, and the 0.5B for the precondition below. About 2.7 GB of disk with
    the 0.5B's int8int8 bundle; the job refuses to build under 6 GB free.
- **A Metal precondition, like amendment 2's on the CPU.** Before any arm, the Metal process loads Qwen2.5-Coder-0.5B
  both ways at int8int8 and int4. Its `.gguf` is small enough to load directly. It runs the gate's prompt and 24 steps:
  the `.gguf` route's greedy tokens, fed to both routes. The logits must be equal bit for bit at every step. **A
  difference voids the run.**
- **A refused load is VOID, not FAIL.** A load refused by the fit guard or by Metal's memory guard, in either process,
  ends the run as VOID, with the refusal's numbers in the log. The script grades a log that holds a `VOID` line as VOID
  (exit 2) before it looks for a FAIL. A FAIL is reserved for G2-R's own rule: arm G failing, or arm G parked.
- **A fit check of every load, by day, before the job is queued** (CLAUDE.md, "Testing tiers"). Each load the two
  processes make is run once, alone, on its backend, and admitted or refused is recorded. The test is
  `TestG2R_fitCheck`, which takes seconds. The job is queued only if every load is admitted by day. Day memory is the
  owner's apps plus this session, so a load admitted by day is admitted at night.

**Considered, not changed: whether a smaller model covers the same operations** (owner's question, 2026-10-15).

- **The 2B cannot shrink.** G2-R tests the attention softcap. Gemma 2 is the only family goinfer reads that has one
  (`decoder/gemma12.go`). The 2B is the smallest Gemma 2 release (2B, 9B, 27B), and no 0.5B-class checkpoint carries the
  operation.
- **Gemma 3 4B stays.** It is not covering an operation. It is the reference for G2-R's band: parked when Gemma 2 is
  more than 0.03 under Gemma 3 from the same session. Gemma 3 1B would change the band's reference after the bar was
  set, and that is a bar change, so it stays at 4B. With `.giw` loads, the 4B's int8int8 load is priced at about 4.5 GB.
- **The dense 1.5B control stays.** Its 0.95 bar was registered on it.
- **The 0.5B enters only as the Metal precondition's model.** Its `.gguf` fits Metal's guards directly, and the
  precondition checks the load route, not the gate.

**Unchanged:** amendments 1 and 2, the arms, prompts, bars, voids and the order of reading.

**G2-R, read 2026-10-15 by day (the owner moved it off the night queue): VOID as a softcap test at both quants. Not
declared.**

- **The run:**
  - Declaration branch `ecf58621`, the script at main `205e3b0d`, under the timing lock, 15:20 to 15:21.
  - The CPU process passed, with amendment 2's precondition holding at both quants. The Metal process passed amendment 3's
    precondition and every load.
  - The logs are in `~/goinfer-logs/op1-metal-2026-10-13/g2r-split-night-2026-10-15/`.
- **Two earlier attempts the same afternoon measured nothing:**
  - The Metal `.gguf` load was refused by the fit guard. That run was labelled FAIL by the old grading.
  - The Metal precondition's `.giw` route read `seed[0]` with no seed and panicked. That was a harness bug, fixed in
    `ecf58621`.
  - After the second attempt, the grading was tightened (`205e3b0d`): FAIL comes only from G2-R's own verdict lines, and
    anything else that ends the run reads VOID.

| arm | int8int8: min cosine, argmax | int4: min cosine, argmax |
|---|---|---|
| B, the softcap removed on both sides | 0.982217, 22/24 | 0.976711, 23/24 |
| the control, the dense 1.5B (bar 0.95) | 0.997749, 21/24 | — |
| Gemma 3 4B, the band's reference | 0.974482, 23/24 | — |
| G, graded: the softcap on both sides | 0.991436, 23/24: pass | 0.982765, 20/24: pass |
| P, the softcap dropped from the Metal build only | 0.990859, 23/24: **passes the rule** | 0.978495, 21/24: **passes the rule** |

- **B's stop rule did not fire:** 0.88 is far below B at both quants.
- **G passes at both quants and is inside the band:** G is not more than 0.03 under Gemma 3's 0.974482.
- **P must fail and does not, at either quant.** The softcap barely acts on the gate's prompts. P's three numbers, read on
  the CPU on the tokens P feeds:

  | quant | scores over \|15\| | largest \|score\| | worst capped-vs-uncapped logit cosine |
  |---|---|---|---|
  | int8int8 | 0.0032% of 62,400 | 15.72 | 0.992485 |
  | int4 | 0.0064% of 62,400 | 16.97 | 0.979200 |

  At a cap of 50, `50·tanh(s/50)` moves a score of 17 by about 0.6, so dropping it costs less than the gate's tolerance.
- **The outcome, as registered:** VOID as a softcap test at both quants, not a pass. Do not declare; change no bar and no
  prompt after the fact.
- **The registered next step:** a prompt set for long contexts, registered in advance, where the scores are larger. It is
  for the owner to commission; nothing is registered yet.

**G2-R closed as void (owner, 2026-10-15).** No long-context prompt set will be registered, and no CPU comparison is added.

- **The reason:** on these prompts the cap barely acts, so a dropped cap passes the model-level rule.
- **What tests the Metal kernels instead:** G2-K. It is graded against a float64 reference, on inputs built so the scores
  cross ±cap, and it reads 152 of 152 inside its bar, with all 152 red once the softcap is dropped (G2-D).
- **G2-K and its plant now decide the Metal softcap declaration,** in G2-R's place in the order (§4.6, step 3) and among
  the gates G2-M follows.

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

### 4.6a OP1 on CUDA: mechanism and rows (amended 2026-10-16 on nobara, before any CUDA OP1 code)

Read at main `7e130317`. This section amends §4.6 for CUDA only, in a commit of its own, as §4.6 allows. It applies Metal's
two rulings (OP1-M1, 2026-10-13, and "G2-R closed as void", 2026-10-15) to CUDA's rows from the start, so CUDA does not repeat
the run that found them out. The owner may overrule either before the first CUDA code commit.

**CUDA's mechanism: the softcap is a compile-time variant, and the cap is a kernel argument of the variant.**
- **The variant.** `build_ptx.sh` builds a second module from the same source with `-DATTN_SOFTCAP`: `<module>_cap.ptx`, and
  `<module>_kv16_cap.ptx` beside the f16-KV twin. The variant's kernels take a trailing `float cap`; the plain module's
  signatures and source do not change. A capped model loads the variants of the three modules below, and no other model loads
  them (as the `_kv16` twins are loaded only by an f16 build).
- **Why a variant and not a run-time parameter.**
  - A parameter puts a branch on every score in every attention kernel of every model, and changes the PTX of all of them.
  - Here the identity gate can be stronger than Metal's: the plain modules are PTX files, so G2-I compares step 2's plain PTX
    with step 1's **byte for byte**, then checks the bits. With a parameter it would rest on the compiler.
- **Why an argument and not a baked `-D` constant.** CUDA has no NVRTC at run time, so a baked cap would be one number in a
  prebuilt file. An argument lets G2-K sweep caps, including a small one so that ordinary scores cross it.
- **The covered kernels** (the ones a capped layer takes): `attention` (`glue.cu`, the fallback when batched prefill is not
  ready), `attn_batched` (`prefill_batched.cu`: exact prefill, and decode through `decodeAttnGap`, the batched step's rows
  included), and `splitkv_scores` (`decode_splitkv.cu`; its softmax and V-sum kernels read the stored, already capped scores).
  That is three modules at two KV types: six variant files.
- **The kernels a lane selector keeps a capped layer off**, each by a named decline and each tested (G2-L below):
  flash-decode and its verify rows (`faEligible` and `flashVerifyAttn`: the lane is default-on at 2,048 keys and serves head
  dimension 256), `attn_fused*` and `attn_fused_bm*` (`useAttnFused`: head dimensions 64 and 128 only, guarded anyway), the
  drafter's `attn_block_full`, the image prefill's `attn_img_batched`, `mla_attn`. The vision and audio towers (`attn_fused_vit`,
  `tower_base`, `audio_encoder`) are not text-path kernels and are not touched.
  **Owed, not in OP1:** the flash-decode lane with the cap. Gemma 2 then decodes at depth on the exact split-KV lane; G2-P's depth
  cell shows what that costs.

**Step 1 on CUDA.** One new header, `attnscore.cuh`, beside `kvtype.cuh`. Each of the 10 hand-written score sites (§4.5) calls it:
`glue.cu`, `prefill_batched.cu`, `attn_block.cu`, `attn_img_prefill.cu` (`dot *= scale`), `decode_splitkv.cu` (`dot * scale`),
`decode_fa.cu` (`d * scale`, two sites), `attn_fused.cu` and `attn_fused_bm.cu` (`__fmul_rn(s, scale)`), `mla.cu`
(`__fmul_rn(dot, scale)`). The header has one function per multiply the sites use today, so each kernel keeps its own order of
operations and a `__fmul_rn` stays a `__fmul_rn` (a fused multiply-add must not appear).
- **`glue.ptx` is an audited artifact** (`cuda/testdata/REGEN.md`). Editing `glue.cu` means regenerating it at the pinned NVRTC
  12.6.85 with that file's control. **Registered fallback:** if the per-kernel hash audit shows any kernel other than `attention`
  moving, step 1 does not touch `glue.cu`, `attention` stays unrouted, and a capped model requires `prefillReady` or declines by
  name. G1-K records `attention`'s digests either way. **Which modules were built at 12.6.85 (read from each PTX header, 2026-10-16):**
  `glue.ptx`, `glue_kv16.ptx` and `mla.ptx`. The other seven modules and their f16 twins were built at the ambient 12.9.86 and are
  regenerated there. Each is rebuilt unchanged first, at its own toolchain, and must come back byte for byte.
- **G1-K's instrument** is `TestAttnKernelDigests` (`goinfer_testhooks`, `cuda` tags), built and run at base first.
  - It drives each of the 10 sites' kernels, in every module they ship in (f32 and f16 KV), on fixed pseudo-random inputs, and hashes
    the output bytes. Shapes: head dimensions 64, 128, 256 (512 where a kernel has it); key counts at, one under and one over each
    kernel's tile bound; windowed and not; with and without a sink.
  - The kernel list is a table in the test. **A `__global__` in `cuda/*.cu` whose name says attention, and that is not in the table,
    fails the test.** A kernel in the table that no case drives fails it too.
  - The digests are committed in `cuda/testdata/attn_digests.json`, keyed by device name and driver version. A run on another
    driver is VOID, not a pass.
  - **Planted defect:** one ulp added to a score in each routed kernel turns that kernel's digests red.
- **G1-M's instrument.** `cmd/gate identity` gets `-backend cuda`: a dumper in `cuda/` with `-tags cuda`, an ephemeral
  `go.work` of `. ./cuda`, the same three-run judging as the other GPU backends, and a decline that reads as NOT RUN. It lands as its
  own commit, before step 1, with the tool's tests. Tiny fixtures at int4 and int8int8; real checkpoints at int4, as the other backends.

**G1-S on CUDA, by name** (no test that passed at base may skip): `TestAttnBatched_bitIdentical`, `TestSplitKV_bitIdentical`,
`TestSplitKV_bitIdentical_gemma3`, `TestAttnFused_vsExact`, `TestAttnFusedTile_32x64BitIdentical`,
`TestAttnFusedTile_defaultBitIdenticalWholeModel`, `TestAttnBlockFull_nonCausal`, `TestAttnImgBatched_*`, `TestAttnSink_matchesReference`,
`TestFlashDecode*` (the host-only ones included), `TestMLAAttn_cuda`, and the batched step's bit-identity tests; the night device gate is
`go run ./cmd/gate gpu`.

**G2-K on CUDA.** Each covered kernel, at f32 and at f16 KV, with a head dimension of 64 and of 256, windowed and not, with a sink
(`attn_batched` and `splitkv_scores` carry one), against a float64 reference, with caps of 50 and of 2 and queries scaled so that scores
land on both sides of ±cap, plus a peaked row. Bar: cosine ≥ 0.9999 and max abs error ≤ 1e-2.
- **G2-D** removes the cap's application from the variant source (the argument is read and ignored): every G2-K case must go red.
  It is the same planted defect for G2-T and G2-R, whose results are recorded.

**G2-L (new, CUDA): the lane selectors keep a capped layer off the uncovered kernels.** On a capped fixture, decode at a depth past
2,048 keys and a prefill with `fastAttn` on launch none of the kernels in the list above. The instrument is the executor's launch record by
kernel name; if it has none, step 2 adds a counter behind `goinfer_testhooks`. **Planted defect:** `faEligible` ignores the cap, and G2-L
goes red.

**G2-T on CUDA is a diagnostic from the start, with its baseline measured first.** Metal's run found a bar copied from Gemma 1 sat
inside the fixture's own no-softcap agreement. For CUDA, before any graded reading, both tiny fixtures are read at int4 and at int8int8
with the cap removed on both sides (`SetAttnSoftcapOffForTest`), 16 positions, the batched prefill against the token-by-token one. Both
numbers are recorded with the capped ones. The 0.999 bar is not applied: the declaration does not rest on these fixtures (G2-D).

**G2-R on CUDA is a model-level sanity gate, not a softcap test.** Metal found the cap barely acts on the gate's prompts (a plant that
drops it passes). CUDA runs the registered arms B, G and P anyway, since the harness exists, and records them: G must clear 0.88 with
argmax equal at ≥ 50% of positions and no NaN, inside the band of §4.6's G2-R. P's result is recorded and does not gate. **The CUDA
softcap declaration rests on G2-K, G2-D and G2-L**, as Metal's does. The checkpoint is `gemma-2-2b-it-Q8_0.gguf` in nobara's `~/models`
(sha256 `2d448a9a…debe`); G2-R loads it from there, never from `/srv/models`.

**Rows for the night gates.**
- **G1-M and G2-I on real checkpoints, CUDA:** `cmd/gate identity -backend cuda -assets real`. Estimate 25 min; the families the box
  holds, int4.
- **G2-S, CUDA** (the template in §4.6; the numbers from `scripts/power.py` and the registry, 2026-10-16):
  ```
  Tier: night. One timed run per box, under the timing lock (the 8 GB card holds one model at a time).
  Instrument: scripts/bench_peer.py served decode, two arms (new, base), no peer arm, restart between cells.
  Cells: decode on the 1.5B at depth 128 and 3900 (the single-block and split-KV lanes, then flash-decode, whose kernels step 1
    routes), on Gemma 3 4B at depth 128 (head dimension 256, the kernels Gemma 2 shares), and on the 0.5B at depth 128.
  Noise: 1.5B and Gemma 3 cells: seed-nobara-cuda-goinfer-all-S, sd 0.0081, a same-binary A/A (registry gap 1: it cannot see a build
    effect). 0.5B cell: cuda-fused-rms-0.5B-crossbuild, sd 0.0164 with a floor of 0.0144, the only cross-build A/A on this box.
  power.py N: `served-decode nobara/cuda/1.5B/d128 --bar 0.97 --expect 1.0 --paired` -> N = 3 (power 0.88), run at N = 4;
    the same at d3900 and for gemma-3-4b/d128; the 0.5B cell with `--entry cuda-fused-rms-0.5B-crossbuild` -> N = 11 (power 0.83).
  Stopping rule: fixed N per cell, no early stop.
  Rule: PASS if every cell's interval clears 0.97; FAIL if any lies under it. A straddling interval is parked for the owner.
  Not gating: time to first token on the 1.5B at K = 3900 (the registry has no CUDA entry: gap 4), reported with its spread.
  Cost: (3 cells x 4 blocks + 1 cell x 11 blocks) x 2 arms x about 2 min = about 90 min, plus idle-gate waits. Queued after the other night jobs.
  ```
- **G2-P, CUDA:** §4.6's template with `bench_peer.py` on nobara, `gemma-2-2b-it` Q4 both sides, depth 128 and 2048, three arms (goinfer
  CUDA, goinfer CPU, Ollama). The CPU arm gates (ship ≥ 1.02x at both depths). Estimate 25 min. The depth-2048 cell runs on the exact
  split-KV lane because flash-decode is kept off a capped layer; a result under 1.02x there is parked and names that lane.

**Boxes and costs.** nobara-pc, RTX 2070 SUPER, 8 GB, driver 595.91.07. Day gates need the card for seconds to minutes each, under
`timing_lock.py run`; while anyone is logged in, `kwin_wayland` holds about 16 MiB of the card, which goes in the record. The 2B at int4
and f16 KV fits; its fit is checked with the server's own plan (the `fit:` and `KV plan:` banner) before the night job is queued.

**Order on CUDA** (§4.6's order, with the two additions): (1) the `-backend cuda` tool commit; (2) G1-K's digest test and base's digests,
committed with the digest file; (3) step 1, then G1-K, G1-M (tiny), G1-S; (4) step 2, then G2-I (plain PTX byte-equal to step 1's, then
G1-K and G1-M again), G2-K, G2-D, G2-L, the G2-T baseline and readings, G2-R; (5) declare `attn-logit-softcap` on CUDA, regenerate the
matrix (G2-M), merge; (6) night: G1-M and G2-I on real checkpoints, the device gate, G2-S, G2-P.

### 4.7 The remaining plan for OP2–OP6 (proposed 2026-10-15; docs only, nothing built)

Written on the owner's ask, after G40 stopped on its tier 1 memory run and G2-R was queued for the night of 2026-10-15. Read at
main `ee80cf2e`. **A proposal for the owner, not a decision.** Sizes are §0's. Gates are in the tiered form
(`CLAUDE.md`, "Testing tiers"): each is registered at the lowest tier that can answer it, and a night run states why the day
tiers cannot. The day sizes and the fit notes below are INFERRED from checkpoint sizes, not fit-checked. Every night load is
fit-checked by day before it is queued.

**KV work in flight, and its production files** (from `git diff --stat` against main on each branch, tests excluded):

- **KV2** (the Mac's `kv2-2026-10`, nobara's `kv2-cuda-2026-10`):
  - `decoder/model.go`, `decoder/resident_reuse.go` and `decoder/blockspec.go`, plus the decoder's session and VL
    generate files;
  - `internal/kvmgr/`;
  - `internal/serveapp/` (main, openai, sessions);
  - a new KV-copy file in each of `metal/`, `gpu/` and `cuda/`;
  - on CUDA, also `cuda/resident.go`.
- **KV3:** CUDA's half on `kv3-cuda-2026-10` (`cuda/prefill.go` and the prefill scratch). Metal's half is KV-pre's
  (`metal/prefill.go`).
- **The CPU's KV:**
  - G40, stopped: `decoder/forwardn.go`, `decoder/kvcache.go` and `decoder/weightmat.go`;
  - the CPU KV pricing fix: `decoder/model.go`, `decoder/fitguard.go` and `decoder/prefill_budget.go`;
  - CPU f16 KV, approved and not started: `decoder/kvcache.go` and the CPU attention.
- **Next on the KV track:**
  - KV4, paged CUDA storage: every CUDA KV kernel goes through a block table, including `mla_attn` in `cuda/mla.cu`;
  - KV5, the same on Metal;
  - KV8, a recurrent state pool.
- **Proposal 2** (the integration harness's id carry, nobara's next build): `internal/serveapp/` and one accessor on the
  decoder's `Generation`. It shares serve's files with KV2 and no file with OP.

#### OP2 — the next operations by demand

Two prerequisites from §4.5, both S, on the Mac, tier 0 only, in `decoder/features.go`:

- **(a) A feature per non-gated activation** (§4.5, item 2), before SSM on Metal. Metal's `act_quant` would otherwise run
  tanh-GELU for Nemotron-H's squared ReLU.
- **(b) Split `attn-output-gate` by activation** (§4.5, item 3), before OP5.

Their gate:

- **Tier 0:** the hardware matrix regenerated, with every cell's decline reasons unchanged except the split names.
- **Plant:** a backend declaring the wrong half of a split must decline the family by name.

**(a) and (b) built 2026-10-15; both gates pass.**

- **(a)** One feature per non-gated activation, in `decoder/features.go`: `non-gated-gelu-tanh`, `non-gated-silu`,
  `non-gated-relu2` and `non-gated-act-unknown`, beside `non-gated-gelu-erf`. Each backend declares the kernels it has:
  - Metal's `act_quant` (`glu_act`'s two branches): GELU-tanh and SiLU;
  - WebGPU's `relu2Quant`: squared ReLU;
  - CUDA: no non-gated kernel, as before.
- **(b)** `attn-output-gate` is three features:
  - `attn-output-gate-softplus` (Laguna);
  - `attn-output-gate-sigmoid` (Spark-X2.5);
  - `per-layer-query-heads` (Laguna XS's per-layer query-head count, which the old feature carried along).

  The gate features follow `hasAttnOutputGate`'s two cases, so the features and the forward cannot drift.
- **Tier 0, the matrix:** `docs/hardware-matrix.md` changes in six rows, each only by the split names. No cell changed
  whether it runs resident, and the capability matrix is unchanged.
  - **The row that matters:** Nemotron-H on Metal now reads "missing: `non-gated-relu2`, `ssm`". When SSM on Metal lands,
    Nemotron-H stays declined until Metal has a squared-ReLU kernel, instead of running `act_quant`'s GELU-tanh on it.
- **The plant:** `TestFeatureSplits_wrongHalfDeclinesByName` gives each split's family a backend that declares the other
  half: GPT-2 with SiLU, Nemotron-H with GELU-tanh, Laguna with sigmoid, Spark-X2.5 with softplus. Each declines,
  naming exactly the half it lacks.
  - **Planted:** Spark-X2.5 deriving softplus, and squared ReLU deriving GELU-tanh. Both went red, then restored.

**OP2a, MLA on Metal.**
- **Delivers:** DeepSeek-V2/V3 and Moonlight resident on Metal. Kimi K2 also needs a 384-expert router; that is not here.
- **Size and box:** M–L, on the Mac.
- **Depends on:** OP0 only. CUDA's `cuda/mla.cu` and WebGPU's `gpu/mla.go` are the models to port.
- **KV:**
  - MLA's latent cache is a second KV shape, a layer group of its own in the KV design (§5.2 of the KV doc).
  - KV2's Metal KV-copy has to copy it, or decline MLA for cross-conversation reuse, and it shares `metal/model.go`'s
    slot allocation with KV2.
  - KV5 will route Metal's MLA kernel through the block table, so OP2a writes the kernel KV5 then edits.
- **Gates:**
  - **Tier 0:** `deepseek-tiny` and `kimi-tiny` on Metal against the CPU, by the resident identity suite's bars. A plant: the
    rope split of q and k swapped must go red.
  - **Tier 1:** none fits by day as a whole model. The smallest real MLA checkpoint is DeepSeek-V2-Lite (16B MoE), whose
    Q4_K_M file is about 10 GB, against the Mac's 16 GB. Tier 1 is a layer slice of V2-Lite against `deepseek_v2lite_golden.json`'s
    hidden states, as a correctness gate only, never quoted as a speed.
  - **Night, only if the slice passes:** the whole V2-Lite resident through Metal's expert paging. It is the only run that
    sees the router with the real expert count. Fit-check by day first, and if it is refused, it is an owner question, not
    a bypass.

**OP2a on Metal, registered 2026-10-16 before any code** (branch `op2a-mla-metal-2026-10`).

- **Design:** WebGPU's absorbed MLA (`gpu/mla.go`, `gpu/decoderunner.go`), ported to Metal kernels:
  - the q-LoRA path or the direct q projection;
  - the KV down-projection with its latent norm and the rope slice, stored per position as one latent row
    (`kvLoRA + qkRope` values);
  - q's nope part absorbed through W_UK; q's rope part rotated;
  - attention over the latent cache;
  - the lift through W_UV, then the output projection.
  - Metal's MoE router already routes DeepSeek-V2/V3 (sigmoid, bias, group-limited top-k), so the MoE is unchanged.
  - Kimi K2's 384-expert router stays out, as above.
- **The KV:**
  - an MLA layer's cache is its latent rows, priced so by `ResidentKVBytes`;
  - the MC3 batched step, the f16 prefill, KV2's slot copy and int8 KV decline an MLA model, each pinned by a test.
- **Tier 0, by day:**
  - **M, the tiny models against the CPU at the same quant:** `deepseek-tiny` and `kimi-tiny` resident on Metal at int4,
    in greedy lockstep for 32 steps off the CPU, graded from position 2.
    - **The control,** in the same session and harness: `qwen3moe-tiny`, a MoE already resident on Metal. A routing flip
      under rounding is then something both arms can show.
    - **The bar:** each MLA fixture's min logit cosine at least the control's minus 0.001, and argmax agreement at least the
      control's count minus 2. That is gate M's rule (OP2b).
    - **Planted defect:** q's and k's rope and nope slices swapped in the Metal kernel only. M must fail.
  - **Guards:** each decline above, with a test; `ResidentKVBytes` for an MLA layer equals its latent size per position.
- **Tier 1, by day:** a layer slice of DeepSeek-V2-Lite (`scripts/slice_gguf_layers.py`, its first layers including the
  dense one and one MoE layer) resident on Metal against the CPU at int4, by G2-R's bars (min cosine ≥ 0.88, argmax ≥ 50%)
  with the dense 1.5B control ≥ 0.95. The checkpoint is pulled to `~/models` first.
  - This replaces the hidden-state comparison against `deepseek_v2lite_golden.json` written above. That golden holds the
    whole model's output, which a slice does not produce.
- **Night, only if the slice passes:** the whole V2-Lite, as above.

**OP2a gate M, read 2026-10-16, and two amendments after the data (owner, 2026-10-16).**

- **As registered:**
  - The control `qwen3moe-tiny` read min cosine 0.999458, argmax 31/32.
  - `deepseek-tiny`: 0.999880, 32/32, **PASS**.
  - `kimi-tiny`: 0.994251 against 0.998458, **FAIL**, with argmax 31/32. Its cosine held about 0.9997 except steps 8 and 9
    (0.994, 0.997), then recovered.
  - The guards PASS.
- **Amendment 1, gate M for the MoE fixtures is graded with routing forced.** Every MoE layer on Metal takes the experts
  and weights the CPU chose for the same token (`TestMLAResident_routingForced`), on all three fixtures, control included.
  - Read: control 0.999885, 31/32; `deepseek-tiny` 0.999884, 32/32; `kimi-tiny` 0.999751, 32/32. **PASS.**
  - The step 8–9 dip was a routing near-tie: Metal's router sums its softmax in f32, the CPU's in f64, and kimi-tiny picks
    4 of 16 experts. A fault in the MLA path would persist through the latent cache, and none shows once routing is forced.
- **Amendment 2, the tiny fixtures cannot see a rope defect.** The registered plant (q's and the latent's rope and nope slices
  swapped, on the branch these interleaved fixtures run) read 0.999866, 32/32, not red. Their rope contributes almost nothing
  to the scores (`decoder/features.go`'s note on flat tiny weights). Gate M stays as the model-level check of the rest of the
  path. The rope is gated at the kernel instead, registered here before it runs:
  - **K-MLA:** `mla_store` and `mla_qrope` against a float64 reference on non-flat synthetic inputs (values spanning ±4, a
    rope table of base 10,000), both branches (interleaved and not), at positions 0, 1, 7, 255 and 4,095.
    - **Bar:** max |error| ≤ 2e-3 × max |reference| per output, which allows for fast-math `sin`/`cos` at angles up to
      about 4,095 radians.
    - **Plant:** the rope and nope slices swapped in each branch must turn it red.

**OP2a read 2026-10-16: K-MLA, tier 1 PASS; Metal declares `mla`.** Record: `docs/measurements/op2a-mla-metal-2026-10-16/`.

- **K-MLA (`TestMLAKernels_rope`):** worst max |error| / max |reference| is 7.8e-5 for `mla_store` and 2.9e-4 for `mla_qrope`
  across both branches and every position, against 2e-3. The rope/nope swap reads 1.3 to 2.1. **PASS.**
- **Tier 1, the V2-Lite slice:** the first two layers (dense, then MoE) of `DeepSeek-V2-Lite-Chat-Q4_K_M.gguf`, cut by
  `scripts/slice_gguf_layers.py` on nobara (774 MB, sha256 `7396629f…`), int4, by day under the timing lock. The fit guard
  and Metal's memory guard admitted both loads.
  - Control 0.997134, 27/32.
  - The slice: min cosine 0.995465 against 0.88; argmax 30/32 against 16. **PASS.**
- **The declaration:** Metal declares `mla`.
  - It rests on gate M with routing forced, K-MLA, the guards and the tier 1 slice.
  - The hardware matrix reads DeepSeek-V2 and V3 resident on Metal.
  - Kimi K2 stays on the CPU, held by its 384-expert router (Metal's routes 256), which this work does not touch.
- **Not run:** the whole V2-Lite at night, through Metal's expert paging, as registered above.

**OP2b, SSM (Mamba-2) on CUDA and on Metal.**
- **Delivers:** Nemotron-H resident on CUDA and Metal, and Granite-4.0-H once its bring-up gate lifts.
- **Size and box:** M–L for each of CUDA (nobara) and Metal (the Mac).
- **Depends on:** OP2(a) for Metal.
- **KV:**
  - The SSM state is not KV. It lives in today's per-slot recurrent buffers until KV8's state pool.
  - KV2's reuse already treats hybrids as one slot with exact-extension reuse only. OP2b keeps that, so it changes nothing
    in KV2's files.
  - It shares `cuda/resident.go` (CUDA) and `metal/model.go` (Metal) with KV2's slot code.
- **Gates:**
  - **Tier 0:**
    - `nemotron-tiny` and `granite-tiny` on the backend against the CPU, plus `nemotron3nano-tiny` where its MoE is
      admitted;
    - a plant: the conv window shifted by one must go red. This needs ≥ 2 decode steps past the window, per `CLAUDE.md`'s
      minimal-repro rule.
  - **Tier 1:**
    - **CUDA:** NVIDIA-Nemotron-Nano-9B-v2 at int4 on nobara's 8 GB card, against `nemotron_real_golden.json.gz`.
    - **Metal:** the same checkpoint if it fits by day. If it doesn't fit, a layer slice by day, and the whole model at
      night after a by-day fit check.
  - **No speed gate:** the do-nothing arm is the CPU, about 15× slower, so the direction cannot be in doubt.

**OP2b on Metal: the gates, registered 2026-10-15 before any code.**

**The build:**
- The Mamba-2 decode kernels in MSL, ported from `gpu/mamba.go`'s shaders: the conv step with its window, the SSM step,
  and the gated RMSNorm, per group (Nemotron-H) and over the full inner width (Granite-4.0-H).
- A squared-ReLU branch in `glu_act`. It is a new ordinal (2); 0 and 1 stay as they are.
- Metal's resident builder learns Nemotron-H's one-operation blocks: Mamba, NoPE attention and the non-gated MLP. The
  recurrent state is kept per slot and reset with it.
- `ssm` and `non-gated-relu2` are declared on Metal only after every gate below passes.

**Tier 0, by day: synthetic kernels and tiny fixtures, seconds.**

- **K, each new kernel against the CPU's own computation** (`decoder/mamba2.go:mamba2Step`'s pieces), on synthetic
  inputs. The bar is `gpu/mamba_test.go`'s: cosine ≥ 0.99999 and max |diff| ≤ 1e-3.
  - The gated norm at one group and at several.
  - The conv step over ≥ 2 steps past its window. One step inside the window cannot see a window error (CLAUDE.md's
    minimal-repro rule).
  - The SSM step over 64 steps, for drift.
  - The squared-ReLU branch.
- **D, plants: each must turn K red.**
  - The conv window shifted by one.
  - The softplus on dt dropped.
  - The group norm taken over the full width where groups are asked for.
  - Squared ReLU as plain ReLU.
- **I, identity:** every existing kernel's digest equals base's (G1-K's instrument). Adding the new ordinal changes no
  existing path.
- **M, the tiny models against the CPU at the same quant:** `nemotron-tiny` and `granite-tiny` resident on Metal, at
  int8int8 and int4, in greedy lockstep for 32 steps. Positions from 2 are graded.
  - **The bar is relative, set by a control in the same session:** `llama-tiny` at the same quant through the same
    harness.
  - Each SSM fixture's min logit cosine must be at least the control's minus 0.001, with argmax equal on at least the
    control's count minus 2.
  - G2-T is why: its bar was copied without measuring the fixture's own baseline, and sat just under it.
  - With the conv window plant from D in the Metal build only, M must fail.
- **S, slots:**
  - A recurrent family is declined by MC3's batched step and by KV2's sharing, as now.
  - A test pins both declines.
  - Two conversations taking turns on one slot, the state reset between them, each give the reply they give alone.

**Tier 1, by day.**

- **Fit-check first:** NVIDIA-Nemotron-Nano-9B-v2 at int4 on Metal, about 5.5 GB.
- **If it is admitted by day:** its resident run against the CPU at int4, by G2-R's harness and bars (min cosine ≥ 0.88,
  argmax ≥ 50%), with the dense 1.5B control at 0.95, in one session.
- **If not:** the same on a layer slice by day, and the whole model at night after a by-day fit check of every load.
- The checkpoint is pulled to `~/models` first; the archive is never a read path.

**Step 1, the kernels, read 2026-10-15** (branch `op2b-ssm-metal-2026-10`). All tier 0 gates pass.

- **Built:**
  - `metal/kernels/mamba_kernels_msl.go`: `mamba_conv`, `mamba_ssm` and `mamba_gnorm`, ported from `gpu/mamba.go`'s
    shaders and appended to the library.
  - `act_quant`'s squared-ReLU branch: `nongated_act`, ordinal 2, local to `act_quant`, so `glu_act`'s other callers keep
    their code.
  - Nothing reaches the new kernels yet.
- **Two amendments, made before any run:**
  - **K reads real inputs, not synthetic ones.** The CPU's own `mamba2Step` captures each step's in_proj output and gated
    output on the tiny fixtures (`decoder.SetMambaCapHook`), and the Metal chain replays the in_proj output from a fresh
    state.
  - **The norm-group plant needs its own kernel case.** Both tiny fixtures have one group, so the grouped norm is gated at
    4 groups against the CPU's `gatedRMSNormGrouped`, through a new test hook.
- **K:**
  - The chain against `mamba2Step` over 72 steps, worst cosine 1.000000000: `nemotron-tiny` max |diff| 1.2e-7,
    `granite-tiny` 7.2e-7.
  - The gated norm at 4 groups: 2.4e-7.
  - `act_quant`'s squared ReLU: exact.
- **D:**
  - Conv taps shifted by one: red on both fixtures.
  - Softplus on dt dropped: red on `nemotron-tiny`. `granite-tiny`'s dt sits where softplus is nearly the identity, so
    there it moved the output by 4e-5 only; logged as that fixture's blind spot.
  - One group over the full width: red (cosine 0.79).
  - Plain ReLU for squared ReLU: red.
  - K fails under every plant, as registered.
- **I:**
  - G1-K's 1,095 attention digests are equal on main (`69b37261`) and the branch.
  - `act_quant` at ordinals 0 and 1 is bit-identical to the kernel rebuilt with `glu_act` in the branch's place.
- **Found on the way:**
  - The test's first run read Metal's gated output as all zeros, because the fixture's weight slices aliased a model the
    test had already closed (fixed by copying them).
  - It showed as a cosine of 1.0, because the shared `cosF32` returns 1 when one side is all zeros. The Mamba tests use
    `mambaCos`, which scores that case 0. Other callers of `cosF32` still have the guard.
- **Next, step 2:** the resident builder learns Nemotron-H's one-operation blocks, with the per-slot state, then M and S.

**Step 2, Nemotron-H on Metal, read 2026-10-15** (branch `op2b-step2-nemotron-metal-2026-10`). Nothing is declared yet.
Gates S and the tier 0 checks pass. **Gate M fails at int8int8**: stopped for the owner's call (below).

- **The NoPE check:** Nemotron-H's adapter sets `RotaryDim: 0` and no RoPE base, so it builds no RoPE table, and
  `Model.RopeInvFreqLayerResident` already returns an all-zero table (the identity rotation) for its attention layers.
  - No change was needed. A clause added for it was dead code and was removed.
  - `TestRopeInvFreqLayer_nemotronIsNoPE` pins the property, and a RoPE base planted in the adapter turns it red.
- **The general form:** `decoder.Model.ResidentLayerOps(i)` gives each layer's mixer (attention, DeltaNet, Mamba-2 or
  none) and whether a feed-forward block follows.
  - This is §4.1's layer program in its smallest form: each layer is a mixer operation, then an optional FFN operation,
    from a closed set of mixer kinds.
  - Metal's build and encode switch on it and upload and encode only that layer's own weights. A hybrid of existing
    kinds (Granite-4.0-H, once its multipliers are folded) needs no family branch on Metal.
  - What it does not yet do: per-operation norms and fusions are still the runner's own (§4.1's fusion tables are
    OP4's), and the FFN kind is still read from model-level flags.
- **Mamba-2 on the resident** (`metal/mamba.go`):
  - in_proj and out_proj run W8A8, as on WebGPU;
  - the conv window and SSM state are per layer, reset with the slot (`resetRecurrent`);
  - Granite-4.0-H declines on Metal, its multipliers not yet folded.
- **The guards:** the twelve paths that declined DeltaNet now decline any recurrent mixer.
  - `TestRecurrentGuards_declineMamba` checks all twelve on a hand-built resident with one Mamba-2 layer. For the four
    where the recurrent check decides, a control without the Mamba layer is admitted.
  - Plant: the KV-copy guard left DeltaNet-only. Red.
- **KV pricing:** `ResidentKVBytes("metal")` prices attention layers only.
  - `TestResidentKVBytes_attentionLayersOnly`: nemotron-tiny 2 attention layers of 5, qwen35-tiny's DeltaNet layers,
    llama-tiny all.
  - Plant: only DeltaNet skipped. Red: 640,000 bytes against 256,000.
- **Gate S: pass.**
  - One KV slot, and the batched step declines.
  - Two conversations taking turns on the slot give their solo replies.
  - Plant: the reset leaving the Mamba-2 state. Red: both later replies change.
- **Gate M** (`TestNemotronResident_tinyAgainstControl`, against `llama-tiny` in the same run):

  | quant | control (llama-tiny) | nemotron-tiny | verdict | conv plant |
  |---|---|---|---|---|
  | int4 | 0.999906, argmax 32/32 | 0.999699, argmax 31/32 | pass | red (0.0336) |
  | int8int8 | 0.999888, argmax 32/32 | 0.994279, argmax 30/32 | **FAIL** | red (0.2451) |

- **Why int8int8 fails: not the Mamba-2 mixer.**
  - With the CPU's Mamba projections also W8A8 (`SetSSMQ8CPU`), nemotron-tiny still reads 0.994730.
  - The native int8 path (`w8Eligible`) covers no recurrent family. Its doc comment scopes slice 1 to "no DeltaNet", and
    the converted guard now declines Mamba too.
  - So at int8int8, Nemotron-H's attention and MLP projections take the int4 re-quantization (`int4ConcatA`) while the CPU
    runs int8. The control is on the native int8 path, so it matches.
  - This agrees with the decoder's existing rule (`decodeRunnerDecline`): Nemotron-H is resident at int4 only by default,
    with int8 behind `GOINFER_SSM_RESIDENT`.
- **Not run (at the time of this reading):** tier 1 (the real 9B) and the declaration. Both waited on gate M; see the
  amendment below.
- **Regression checks:**
  - The Metal set (DeltaNet, Qwen3.5, Olmo, GPT-2, Gemma, the parity, KV and batch tests): 86 pass, 45 skip.
  - Its two failures: gate M's int8int8 row, and `TestGemma2ResidentParityMetal`, which fails identically on main
    (0.998997 and 0.998657, G2-T's recorded readings: OP1-M1 made the tiny fixtures diagnostics, and the test still
    asserts 0.999).
  - The decoder's feature, matrix, residency and KV tests: 55 pass, 0 fail.

**Gate M amended after the data (owner, 2026-10-15): graded at int4 only, with a condition.**

- **The reason:** the gate assumed a native Metal int8 path, which no recurrent family has. So int8int8 measured the int4
  re-quantization of Nemotron-H's attention and MLP weights, not this work.
- **Gate M at int4: PASS:** 0.999699 against the control's 0.999906, argmax 31/32; the conv plant red.
- **The condition, built:** at int8, Metal declines Nemotron-H as resident, and the CPU runs it at real int8, instead of
  re-quantizing to int4.
  - It is `metal/mamba.go:buildMamba`, on the decoder's own test (`decoder.Model.ResidentProjsInt4`).
  - `TestNemotronResident_int8Declines`: int8int8 is refused with the int4-only reason, and int4 builds.
  - Plant: the decline removed. Red: the int8int8 build went through.
- **The ssm and non-gated-relu2 declaration is scoped to int4:** Metal builds Nemotron-H at int4 only.
- **Checked once: the DeltaNet families have the same re-quantization.**
  - `qwen35-tiny` at int8int8 runs resident on Metal as "int8int8→int4", with native int8 off.
  - Filed as G44 (`docs/queue-correctness.md`) with the same fix.
- **Native int8 for recurrent families** is slice 6 of `docs/tasks/task-metal-int8-2026-10.md`, later.
- **Tier 1 next, after a by-day fit check.**

**Tier 1's whole model cannot load on the Mac, by day or by night (2026-10-15).** The int4 .giw that nobara built from
the registered Q8_0 GGUF (`make-nemotron-giw.sh`, at `6622df20`) is 19.2 GB. Its Mamba projections stay f32 in the
bundle, about 0.59 GB a Mamba layer over 27, and only the attention and MLP matrices take int4. That is more than the
Mac's 16 GB RAM, so the fallback's night half cannot run here. The fit-check line's 5.5 GB above assumed every matrix
at int4.

**Tier 1 on a layer slice, registered 2026-10-15 before any run (the registered fallback, by day).**

- **The slice:** layers 0 (Mamba), 1 (MLP) and 14 (the first attention layer) of the Q8_0 GGUF, renumbered 0 to 2, with
  the embedding, final norm and head (`scripts/slice_gguf_layers.py`, bytes checked against the source). nobara built
  its int4 .giw (`nemotron-nano-9b-v2.slice-0-1-14.int4-metal.giw`, 1.88 GB, sha256 `13c38555b9ac…`) and the Mac pulled
  it into `~/models/nemotron/`.
  - **Why these three:** a first-N slice that reaches an attention layer needs 15 layers, 8 of them Mamba, about 6 GB.
    That does not fit by day. These three hold real weights of every block kind the step builds.
  - **What it is not:** the model. Its logits are what three layers make of the prompt.
- **The arms, `TestNemotronResident_tier1Slice`:** one load of the .giw at int4; the Metal resident is built from it and
  run against the CPU model it was built from. That is gate M's harness: greedy lockstep for 32 steps off the CPU, graded
  from position 2.
- **Bars, G2-R's:** min logit cosine ≥ 0.88 and argmax agreement ≥ 50%.
- **The control, same process, before the slice:** the dense Qwen2.5-Coder-1.5B through its int4 Metal .giw, the same
  harness. It must reach min cosine ≥ 0.95, or the run is VOID.
- **A refused load** (the fit guard or Metal's memory guard) is VOID, not FAIL. No guard is bypassed.
- **Not answered by the slice:** the whole model's agreement. That needs a smaller bundle, quantized Mamba projections
  (B27), or a box that holds 19.2 GB.

**Tier 1 on the slice, read 2026-10-15 22:36 PDT: PASS.** By day, under the timing lock. The fit guard and Metal's memory
guard admitted both loads; nothing was bypassed. Record: `docs/measurements/op2b-step2-2026-10/tier1-slice-2026-10-15/`.

- **Control, the dense 1.5B:** min cosine 0.997134, argmax 27/32. Over 0.95, so the run counts.
- **The Nemotron-H slice:** min cosine 0.994410 against 0.88; argmax 29/32 against 16. **PASS.**
- **What stays open:** the whole model, for which this Mac has no room, and filed as B27 (`docs/queue-engineering.md`; filed as B26, renumbered
  because main's B26 is another item).

**Declared, 2026-10-15 (owner: the slice stands in for tier 1 for now): Metal declares `ssm` and `non-gated-relu2`,
int4 only.**

- **What it rests on:**
  - gate M at int4;
  - gate S;
  - the tier 0 kernel tests (K, the norm groups, ReLU²), the guard gate and the int8 decline, each with its plant;
  - the three-layer slice of the real 9B against the CPU, above;
  - the whole 9B against the CPU, below (2026-10-16, PASS).
- **The whole-model run is done,** no longer owed: B27's int8 Mamba projections brought the 9B's bundle to 7.27 GB, and it
  passed by day against the bars above.
- **The matrices:** Nemotron-H reads resident on Metal. Granite-4.0-H stays on the CPU: its runner-shape bring-up gate holds
  it, `buildMamba` declines it, and only its "missing: `ssm`" reason goes. The admission golden lists Metal for both,
  as it tracks the feature-only model.

**Tier 1 on the whole 9B, read 2026-10-16 06:28–06:33 PDT: PASS.** Run by day on the owner's word, as the fit check
admitted both loads (B27's v17 int4 bundle, 7.27 GB). Pinned at `3faaa5bb`; the fit guard and Metal's memory guard were
on and admitted both. Peak RSS 5.78 GB, 0 swaps, swap flat at about 1.86 GB. Record:
`docs/measurements/op2b-step2-2026-10/tier1-whole-2026-10-16/`.

- **Control, the dense 1.5B:** min cosine 0.997134, argmax 27/32. Over 0.95, so the run counts.
- **NVIDIA-Nemotron-Nano-9B-v2, whole:** min cosine 0.929161 against 0.88; argmax 27/32 against 16. **PASS.**
- **This closes what the slice stood in for.** The declaration now rests on the whole model as well.

**Not in this build:**
- Granite-4.0-H's own bring-up gate (§4.5).
- Nemotron-3-Nano's MoE.
- A speed gate. The do-nothing arm is the CPU, about 15× slower.

**OP2b on CUDA: the gates, registered 2026-10-16 before any code** (read at main `d183f2c9`; Metal's step 1 and step 2 records above).

**What exists and what does not.**
- **On the CPU and the decoder side, ready:**
  - the reference, `decoder/mamba2.go:mamba2Step`, and its hooks (`SetMambaCapHook`, `SetSSMQ8CPU`);
  - the layer program, `decoder.Model.ResidentLayerOps`, and the accessors `NemotronResidentParams`, `MambaProjInt8`,
    `ResidentProjsInt4`;
  - the fixtures `nemotron-tiny`, `granite-tiny`, `nemotron3nano-tiny` and the real goldens.
- **On CUDA, nothing:**
  - no Mamba kernel, no non-gated MLP, and none of `ssm`, `non-gated-mlp`, `non-gated-relu2` is declared. Nemotron-H derives exactly
    those three (`decoder/features.go:Architecture.residentFeatures`);
  - the only recurrent state is DeltaNet's `dnetParams` (`cuda/resident.go`), model-level and shaped for DeltaNet, so it is not reused;
  - the recurrent guards test `r.dnet` at eight sites: `cuda/batchstep.go`, `cuda/kvsnap.go`, `cuda/kvcopy.go`, `cuda/prefill.go` (two),
    `cuda/resident.go` (four). A Mamba layer must read as recurrent at every one.
- **The shape that carries over from Metal:** Nemotron-H resident at int4 only. Its int8 projections would be re-quantized to int4
  without notice, so int8int8 declines by name, on the decoder's own test (`ResidentProjsInt4`), and the CPU runs it at real int8
  (Metal's gate-M amendment). Granite-4.0-H stays on the CPU: its runner-shape bring-up gate holds it, and the backend declines it by name.

**The build, by step.**
- **Step 1, kernels and the non-gated MLP, nothing reaches them yet.**
  - `mamba_conv`, `mamba_ssm`, `mamba_gnorm` (per group for Nemotron-H, over the full inner width for Granite), in a new `cuda/mamba.cu`,
    from `gpu/mamba.go`'s shaders and `decoder/mamba2.go`. Conv window ≤ 8 taps, the window and the SSM state persistent in place.
  - The squared-ReLU activation and the non-gated structure (up GEMV, activation, quantize, down GEMV) in a new `cuda/nongated_act.cu`.
  - **Mechanism, chosen here because the gates lean on it: new files and new PTX, no ordinal added to `glue.cu`,
    `prefill_batched.cu` or `actgroup.cu`.** No audited or shipped PTX moves, so gate I below is a byte comparison of PTX and not a
    claim about a compiler. Built at the ambient NVRTC (12.9.86) per `cuda/testdata/REGEN.md`; a new `.cu` needs no pinned toolchain.
  - `in_proj` and `out_proj` run W8A8 through the existing `gemv_w8a8` (int8 rows with per-row scales from `MambaProjInt8` when the
    load holds them, else quantized from the f32 as Metal and WebGPU do); attention and MLP projections are W4A8 as today.
- **Step 2, the resident:** the builder walks `ResidentLayerOps`, uploads and encodes only a layer's own weights, keeps the conv window
  and SSM state per layer and per slot, and resets them with the slot. The eight guard sites above read "recurrent".
  Prompts are processed token by token (the batched prefill declines a recurrent family, as for DeltaNet). `KVBytes` and the fit plan
  price the attention layers only, and add the states.
- **Declaration:** `ssm`, `non-gated-mlp` and `non-gated-relu2` on CUDA, int4 only, after every gate below passes; the
  matrix and the admission golden regenerate with it. On a branch, unmerged, until then (OP1's way).

**Tier 0, by day: synthetic and tiny, seconds. In every K and M reading both outputs are asserted non-zero before the cosine is taken,
and a cosine where exactly one side is all zeros is scored 0.**
CUDA's `cosF32` returns 1 for an all-zero side (`cuda/deltanet_test.go`), which is the guard that made Metal's first kernel test read
1.0 on an all-zero output (step 1's "found on the way"). The non-zero assertion fails the test by name ("output is all zeros: the
reading is void"), so a kernel that writes nothing, or a replay of a model closed under it, cannot pass as agreement. Tier 1's harness
applies the same check to each step's logits.
- **K, each new kernel against the CPU's own computation, on real inputs.** `SetMambaCapHook` captures each step's `in_proj` output
  and gated output from `nemotron-tiny` and `granite-tiny` (the kernels serve Granite's geometry too; the family stays declined), and
  the CUDA chain replays the `in_proj` output from a fresh state. Bar `gpu/mamba_test.go`'s: cosine ≥ 0.99999 and max |diff| ≤ 1e-3
  (Metal read 1.2e-7 and 7.2e-7).
  - The chain over 72 steps (≥ 2 past the conv window: one step inside it cannot see a window error; 72 for drift).
  - The gated norm at 4 groups against `gatedRMSNormGrouped` (both fixtures have one group, so the grouped case needs its own).
  - The squared-ReLU activation: exact.
- **D, plants, each must turn K red.** The kernels are prebuilt PTX, so each plant is a temporary edit of `mamba.cu` or
  `nongated_act.cu` with the affected modules rebuilt, run, and restored (as OP1's G2-D was); the record lists the red cases.
  - The conv taps shifted by one.
  - The softplus on dt dropped. `granite-tiny`'s dt sits where softplus is nearly the identity (Metal moved it by 4e-5), so this
    plant is read on `nemotron-tiny` and `granite-tiny`'s blind spot is logged, not hidden.
  - One group over the full width where groups are asked for.
  - Squared ReLU as plain ReLU.
- **I, identity.** Every existing PTX file is byte for byte base's (`git diff` adds only new files), and G1-K's 1,864 attention digests
  are equal to base's. `go run ./cmd/gate identity <base> <new> -backend cuda` on the tiny fixtures: IDENTICAL for every family that ran.
- **M, the tiny models against the CPU, int4 only,** in greedy lockstep for 32 steps, positions from 2 graded.
  - **The bar is relative, set by a control in the same run:** `llama-tiny` at int4 through the same harness. Each SSM fixture's min
    logit cosine ≥ the control's minus 0.001, argmax equal on at least the control's count minus 2 (G2-T's lesson: a bar copied
    from another model sat just under this fixture's own baseline).
  - `nemotron-tiny` always; `nemotron3nano-tiny` where its MoE is admitted on CUDA, else its decline reason is recorded.
  - **int8int8 is not graded:** it must decline by name, test pinned (`buildMamba`'s counterpart), planted by removing the decline.
  - **Plant:** the conv window shifted in the CUDA build only, the CPU keeping the right math: M must fail.
- **G, the guards.** A resident built from `nemotron-tiny` is declined by each of the eight sites: the batched step, the prefix
  snapshot, KV copy, both prefill sites, and the four in `resident.go`. A control (`llama-tiny`) is admitted by the ones where the
  recurrent check decides. **Plant:** one guard left DeltaNet-only (`kvcopy.go`, then `kvsnap.go`): red.
- **S, slots.** One KV slot; the batched step declines. Two conversations taking turns on the slot with the state reset between them
  each give the reply they give alone. **Plant:** the reset leaving the Mamba state: red, both later replies change.
- **P, pricing.** The KV plan prices attention layers only plus the states (`nemotron-tiny`: 2 attention layers of 5; `qwen35-tiny`'s
  DeltaNet layers; `llama-tiny` all). **Plant:** only DeltaNet skipped: red.
- **F, fit, by day, before any tier 1:** every load tier 1 makes is started alone with the server's own flags and its `fit:` and
  `KV plan:` banner read, or its refusal. A refused load is VOID, not FAIL, and no guard is bypassed.

**Tier 1, by day, under the timing lock, about 10 minutes.** The checkpoint is `nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf` in
nobara's `~/models` (never `/srv/models`); the CUDA int4 bundle is built from it on nobara with `cmd/prequant -target cuda`.
- **The whole-model cell, decided now (owner, 2026-10-16), in this order, before any data:**
  1. **Look for a smaller real Nemotron-H first.** `nvidia/Nemotron-H-4B-Instruct-128K` exists (52 layers, hidden 3072, 112 Mamba heads
     of 64, per its model card and config as found 2026-10-16; its file size was not confirmed and is INFERRED to be about 8 GB in
     bf16, so about 2.5 to 3 GB as an int4 bundle). The steps, each stopping the list on a failure:
     - pull it to nobara's `~/models` (never read from the archive);
     - it must load and generate coherently on the **CPU** through today's `nemotron_h` adapter, which is a decoder finding in its
       own right if it does not (a different layer pattern or head geometry the adapter does not know);
     - gate F's plan must admit its CUDA int4 bundle whole.
     If all three hold, **it is tier 1's whole-model cell:** its resident run against the CPU by G2-R's harness and bars (greedy
     lockstep 32 steps off the CPU, min cosine ≥ 0.88, argmax ≥ 50%), the dense 1.5B control at ≥ 0.95 in the same session (a control
     failure voids the session). The 9B's three-layer slice runs beside it as an extra, not as the cell.
  2. **If none fits whole,** the declaration may rest on gates K, D, I, M, G, S and P, **the three-layer 9B slice on CUDA** (layers 0, 1
     and 14 of the Q8_0, renumbered 0 to 2, with the embedding, final norm and head, `scripts/slice_gguf_layers.py`; same harness and
     bars), **and Metal's whole-9B pass** (2026-10-16, min cosine 0.929161, argmax 27/32), recorded as exactly that: the same CPU
     reference and the same Mamba-2 math, checked whole on Metal and in slice on CUDA, **not** a whole-model CUDA measurement. The
     record says so in its first line, and the matrix row carries no stronger claim.
  3. **The 9B on the card is still tried:** gate F reads the plan for Metal's 7.27 GB bundle's CUDA counterpart (on an 8 GB card with
     about 0.45 GB held by the desktop and a context, it may not fit). If F admits it, the whole 9B joins as a further cell. A refusal
     is recorded, is VOID for that cell and not a FAIL, and changes nothing in rule 2.
- A smaller bundle would also answer it for the 9B (B27's int8 Mamba projections took Metal's from 19.2 to 7.27 GB; int4 Mamba
  projections would go further and need their own quality gate, `docs/ssm-int8-quality.md`); that is not part of this build.
- **Tier 2: none.** There is no speed gate: the do-nothing arm is the CPU, about 15× slower, so the direction cannot be in doubt. One
  decode rate and the time to first token of the sequential prompt are recorded for the 9B, not graded.

**Void, for every gate above:** a model path under `/srv/models` or `/Volumes`; a resident arm whose `DecodePath()` is not
`cuda-resident` (a silent CPU fallback passes every bar); a skipped test counted as a pass (the record names `--- PASS`); the two
sides on different drivers; the control failing; a refused load.

**Order, and what a failure does.**
0. **OP2b CUDA's build starts after #58 (OP1 on CUDA) merges** (owner, 2026-10-16): both edit `cuda/resident.go`, `cuda/backend.go` and the PTX lint tests.
1. Merge main into the branch (B27, `.giw` v17, KV7's snapshot code are all in `cuda/resident.go`'s neighbourhood). Gate F's table,
   and the CUDA int4 bundle's size, are written into this section before step 1's code.
2. Step 1, then K, D, I. A failure: step 2 does not start.
3. Step 2, then M, G, S, P. A failure: not declared.
4. Tier 1. Declaration on its own branch; the matrix regenerates (`go test ./decoder -run HardwareMatrix -update`).
5. Not in this build: Granite-4.0-H, Nemotron-3-Nano's MoE unless admitted, int8int8, a batched prefill for recurrent families, a
   speed gate.

**OP2c, KDA.**
- **Delivers:** Kimi Linear resident.
- **Size:** M–L.
- **Box:** none today. The released Kimi Linear is a 48B-A3B MoE (INFERRED from its model card, not checked here), which
  fits neither box resident.
- **Gates:** tier 0 only: `kimi-tiny` plus `kda_rehearsal_golden.json`.
- **Proposed: park OP2c** until a checkpoint that fits a box exists. A tiny-only gate would leave the real path unseen,
  the gap §4.6 closed for Gemma 2 with G2-R.

#### OP3 — the layer program, on the CPU

**Registered 2026-10-16 in §4.8, with the design and the steps.** The sketch below is the plan it registered from.

- **Delivers:** every registry family built as a per-layer program that the CPU executes, with its logits bit-identical to
  today's generic loop and own forwards.
- **Size and box:** L, on the Mac. The CPU is either box; the Mac, because it stays on OP.
- **Depends on:** OP0.
- **Shares files with KV work in flight:**
  - `decoder/forwardn.go` and `decoder/kvcache.go` with G40 and CPU f16 KV;
  - `decoder/model.go` with KV2 and the pricing fix.
  - **Start it after G40 is decided and CPU f16 KV has merged,** or it is rebased across both kernels of the CPU
    attention.
- **Gates:**
  - **Tier 0:** `go run ./cmd/gate identity` on the tiny assets, every family, every step, bit for bit, against the commit
    before OP3.
  - **Plants, each of which must go red:** a dropped operation, two swapped operations, and a parameter off by one (the
    registration in §4.4).
  - **Tier 1:** the 0.5B, bit for bit, at prompt sizes that cross the CPU's size-switched paths (the fast-attention floor,
    and G40's chunking if it merges). Tiny fixtures never reach those paths.
  - **No night run.** Identity inherits each family's validation (TE6(b)). A real checkpoint adds a shape only where a path
    switches on size, and tier 1 covers those.

#### OP4 — the residents walk the program

- **Delivers:**
  - each GPU resident runs the program layer by layer;
  - admission by operation, which absorbs the decoder's and each backend's hand-written declines (§4.5, item 6);
  - per-backend fusion tables, each fusion gated;
  - one memory planner pricing through the KV doc's sizing function (its §5.10).
- **Size:** L per backend.
- **Box:** CUDA on nobara, Metal on the Mac, WebGPU on the Mac (its discrete half on nobara).
- **Depends on:** OP3.
- **KV:**
  - The memory planner is KV3's admission, generalised, so it follows KV3 on that backend.
  - CUDA's walk rewrites the launch paths that KV4 routes through block tables (`cuda/resident.go`, `cuda/prefill.go`).
    **Proposed: CUDA's OP4 after KV4a lands,** so one of the two rebases onto the other once, not repeatedly.
  - Metal's OP4 shares `metal/model.go`, `metal/prefill.go` and `metal/batch.go` with KV-pre, KV2 and later KV5.
- **Gates:**
  - **Tier 0:**
    - every family's tiny identity suite on the backend, bit-identical to the pre-walk runner;
    - each fusion bit-identical to the operations it replaces;
    - a plant: a fusion that claims one operation too many must go red.
  - **Tier 1:** the 0.5B and the 1.5B, bit for bit against the pre-walk runner, by day.
  - **Speed:** §4.4's "within 0.99 at depth 128 and 3,900".
    - A whole-token in-process A/B, interleaved, by day (TE5(b)), first.
    - **Night only if it does not resolve:** a 0.99 bar is a small-effect decision, which is tier 2's stated case.

#### OP5 — the families whose layers differ

**What it delivers, size and order:**

- **Delivers:** each family resident, bringing only its missing operations.
- **Size:** M each.
- **Depends on:** OP4 on that backend, and OP2(b) (the out-gate split).
- **Order by what a box can measure:**
  1. Spark-X2.5 (1.7B, on the Mac): exact-erf GELU, out-gate sigmoid, per-layer rotary width.
  2. LFM2.5 (2.6B, on the Mac): short conv.
  3. Granite-4.0-H, after OP2b SSM and its gate.
  4. Laguna (XS-2.1 Q4_K_M), after a by-day fit check.
  5. Llama 4 last. Scout's smallest released GGUF here is Q2_K at about 40 GB, which neither box holds, so it is tier 0
     only, and is proposed parked like OP2c.

**KV:**

- LFM2.5's conv state and Granite's SSM state are recurrent state: per-slot until KV8.
- Llama 4's chunked attention is a KV property (a released-entry rule like the window's), so it touches KV4/KV5's null-block
  handling.

**Gates:**

- **Tier 0:** the family's tiny fixture on the backend, against the CPU.
- **Tier 1:** the real checkpoint on the box that holds it, against its committed golden: `spark2_5_real_golden.json.gz`
  and `lfm2_real_golden.json.gz` on the Mac.
- **No night run** for the families that fit by day.

#### OP6 — Phi-3/4 on the fast paths (E5: its own small task)

**Delivers:**

- Metal's default load becomes `--quant q4k` for Phi-3/4.
- A q4k batched prefill on CUDA and on Metal.
- Per-32 activations, or a q4k lane, on WebGPU.

**Size and box:** M. The Metal half is on the Mac, the CUDA half on nobara.

**Depends on:**

- The Metal decode lane is built and gated (`docs/tasks/task-metal-q4k-2026-10.md`: G-Q1 to G-Q3 PASS).
- The default switch waits on the owed peer cell M1, a night run. That is a published number, which is tier 2's case.

**Shares with KV:**

- the prefill path: `cuda/prefill.go` with KV3's CUDA half, and `metal/prefill.go` with KV-pre;
- the q4k prefill sizes its scratch through KV3's admission.

**Gates:**

- **Tier 0:** `phi3-tiny` q4k prefill against q4k decode, bit for bit on the logits of the last prompt row.
- **Tier 1:** Phi-3 mini on the Mac (`phi3-mini-4k-gguf`), the prefill against `phi3_mini_golden.json`.
- **Night:** M1 only.

**OP6's Metal half, registered 2026-10-15 before any code** (branch `op6-q4k-step-metal-2026-10`).

- **Design: the q4k lane joins MC3's batched step,** so its prefill runs on the step route that every other dense Metal
  model below the floor already takes (`prefillByStep`, eight rows a step, the sequential loop's bits).
  - **New rows kernels:**
    - `rmsnorm_f32_out` and `swiglu_f32` rows forms, derived from production's text as `batch_rows.go` derives the others;
    - `gemv_q4k_f32` and `gemv_w8_f32` rows forms, which read each weight row once for B activation rows and keep
      decode's per-lane loop and `simd_sum` for each row.
  - **The head:** production's `gemv_w8f32_head`, once per row.
  - **`promptStepOK` gains one case:** a resident whose batched pass never runs (`prefillOK` false and no layer-major
    route) takes the step for a prompt of any length. Its rows are the loop the decoder would otherwise run.
  - **One consequence, in scope:** concurrent batched decode turns on for q4k models (`StepBatch`), with the same per-row
    bits.
- **Amendment before any code: tier 0 runs on Qwen2.5-Coder-0.5B q4_k_m, not `phi3-tiny`.** No tiny Q4_K fixture
  exists (`phi3-tiny` is safetensors). The 0.5B is the lane's own non-heavy test model and loads in seconds. Its attn_v
  is Q6_K in half the layers, so both rows kernels run, and so do the multi-segment projections.
- **Tier 0, by day, all bit for bit (`math.Float32bits`):**
  - **(a) Prompt:** `PrefillLast` on the step against the sequential loop (`promptStepOff`). Measured at lengths
    2, 7, 8, 9, 17 and 64, on the last row's logits and every K/V byte written.
  - **(b) Verify rows:** consecutive rows on one slot at depths 128 and 2048, M = 2, 4 and 8, against sequential decode
    (`TestMC3Verify_sameSlotRowsBitIdentical`'s shape).
  - **(c) Two sequences:** two slots stepped together for 8 steps, each row against that sequence's decode alone.
  - **Planted defect:** the min term dropped from the q4k rows kernel only. It must turn (a) red.
  - **`PrefillPath`** on a q4k resident must name the step route.
- **Tier 1, by day (≤ 10 min):** Phi-3 mini (`phi3-mini-4k-gguf`) at q4k on Metal.
  - **Graded:** the step against the sequential loop, bit for bit on the last row and the K/V, on the golden's prompt
    and on a 200-token prompt.
  - **Graded:** argmax equal to `phi3_mini_golden.json`'s, as `TestPhi3GGUFReal_gate` grades this file. Its cosine is
    logged only, because the GGUF is an earlier Phi-3 snapshot than the golden.
  - **Speed:** an in-process A/B of `PrefillLast` on the 200-token prompt, step against sequential, interleaved, 5 pairs
    after a warm-up.
    - **Ship:** every pair's ratio above 1 and the median at least 1.02.
    - **Park:** a median of 1.00 to 1.02.
    - **Off:** a median below 1.00.
- **Night:** M1, unchanged, after this merges.

**OP6's Metal half, read 2026-10-15.** Tier 0 and tier 1 PASS.

- **Tier 0, by day, on the 0.5B q4_k_m:** (a) to (c) are 0 differing logits and 0 differing K/V elements in every case.
  The planted defect turns (a) red: every logit and 91,101 K/V elements differ. `PrefillPath` names the step route.
  - The first run failed every case, with differences up to 1.9e-6. The int8 rows kernel computed its row scale once,
    above its mode branch, where decode's kernel writes `simd_sum * scale + bias` as one statement and the compiler
    fuses it. Each mode now writes decode's statement. That is a kernel fix, not a bar change.
- **Tier 1 needed the fit guard's q4k price fixed first.** By day the guard refused Phi-3 at 6.6 to 7.0 GB, pricing q4k as
  int4 with its repacks. PR #41 prices it at what the load keeps; Phi-3's weights are 2.81 GB measured against 2.83 GB
  estimated. Tier 1's context was set to 256 positions, what its prompts use.
- **Tier 1, the Mac's night queue, 2026-10-15 21:30–21:32 PDT,** at `d1b340ad` (its `metal/` and `decoder/` equal this
  branch's after the rebase onto #41). The fit guard was on and admitted both loads. Swap stayed at 849 MB with 0 swaps;
  the tier 1 process peaked at 5.49 GB RSS. Records: `docs/measurements/op6-q4k-step-2026-10/night-2026-10-15/`.
  - **Bit identity:** 0 of 32,064 last-row logits and 0 K/V elements differ from the sequential loop, on the golden's
    5-token prompt and on the 200-token prompt. **PASS.**
  - **Golden:** argmax 3681, the golden's 3681. **PASS.** Cosine 0.7554, logged only (an earlier snapshot).
  - **Speed:** 200 tokens in 4.42 s on the step (45 tok/s) against 8.78 s on the sequential loop (23 tok/s). Ratios
    1.982 to 1.986 over 5 pairs, every pair above 1, median 1.985. **SHIP.**
- **Next:** the PR, merged by hand on green CI. Then M1 at night, which decides Metal's default `--quant q4k` load.

#### Proposed order

**The Mac stays on OP continuously:**

1. OP1 on Metal: tonight's G2-R. On a pass, the declaration merges and G2-M runs, by day.
2. OP2(a) and (b), the two feature splits: S, tier 0.
3. OP2b, SSM on Metal.
4. OP6's Metal half: the q4k batched prefill by day, and M1 queued for a night.
5. OP2a, MLA on Metal. It is behind SSM because its real gate needs a slice and paging, while Nemotron-H's may fit whole.
6. OP3, once G40 is decided and CPU f16 KV has merged. If those are still open when step 5 ends, OP1 on WebGPU fills the
   gap.
7. OP4 on Metal.
8. OP5 on Metal: Spark-X2.5, LFM2.5, then the rest as above.

**nobara, after proposal 2's build:**

1. OP1 on CUDA (§4.6's CUDA rows).
2. OP2b, SSM on CUDA.
3. OP6's CUDA half: the q4k batched prefill.
4. KV4a, the KV track's next CUDA phase. That order is the owner's call; it is here because OP4 on CUDA is proposed after it.
5. OP4 on CUDA, after OP3 (the Mac's) and KV4a.
6. OP5 on CUDA, by family.

**Two departures from §4.4 and §7, both approved by the owner on 2026-10-15:**

- **OP4 on Metal before OP4 on CUDA** (§4.4 says CUDA first). The Mac reaches OP4 while nobara is in KV4a, and CUDA's walk
  would otherwise rebase across KV4's kernel rewrite.
- **OP2c (KDA) and Llama 4 parked,** because no box can run their real checkpoints. They wait for a checkpoint that fits a
  box.
  - **A way to test them later without the real weights:** a small, randomly initialised Hugging Face checkpoint of the
    family's own config class. Build it with `transformers` at a reduced width and depth, with the real layer pattern:
    Llama 4's chunked and NoPE layers, and Kimi Linear's KDA and MLA layers. It gives a full-model reference from the
    family's own modelling code, at a size either box holds.
  - Such a checkpoint is still not trained weights. Its scores sit near zero, as `CLAUDE.md`'s minimal-repro rule warns. Its
    registration has to say which effects a random init leaves identically zero, as G2-D did for the softcap.

### 4.8 OP3: design and registered gates (2026-10-16, before any OP3 code)

Read at main `2c56d9c6`, with CPU f16 KV merged (#55) and G40 merged, which §4.7 makes OP3's precondition.

**What the CPU forward is today.** Three generic paths, not two:
- the per-token loop, `decoder/model.go:Model.runLayersFromEmbed`;
- the batched loop, `decoder/forwardn.go:Model.runLayersFromEmbedNFloor`;
- MC3c's multi-sequence step, `decoder/cpu_batch.go:Model.decodeMultiStep`.

Beside them are the nine own loops of `decoder/arch.go:ownForwards`, two of which have batched twins:
`decoder/forward_gemma4_batched.go:Model.runLayersGemma4FromEmbedN` and `decoder/forward_qwen35_n.go:Model.runLayersQwen35N`.

**The same mathematics is computed in more than one floating-point order across them:**
- **Attention:** `attendBatchedHeads` with float64 accumulation in the generic paths; `attendQuery` in the qwen35, lfm2,
  granite, nemotron and llama4 loops; a float32 `MatmulBT` in gemma4's; an inline float64 softmax with sinks in gpt-oss's;
  and the absorbed and naive MLA forms.
- **Norms:** four-way partial sums in `rmsNormInto` and `layerNormInto`, sequential sums in `rmsNormNoWeight`,
  `l2NormHeads` and the recurrent mixers' gated norms.
- **RoPE:** `mlaRope` for llama4 and MLA, gemma4's zero-padded table, and `applyRoPE` called directly by lfm2 and granite.

OP3 moves no number, so each of these orders survives it.

**The design.**
- **The program** is a value built once at load from the resolved `Architecture`. It holds an embedding stage, one list of
  operations per layer, and a head stage.
  - Each operation has a kind from a closed set, its parameters, and the operands it reads and writes.
  - The operands are a small fixed set of per-stream registers: the residual, the normed input, and two sublayer outputs. So
    the parallel block is a list (norm, attention into one output, FFN from the same input into the other, one residual add of
    both), not a branch.
  - The types are unexported in OP3. OP4 exports what the backends read, when they first read it.
- **The kinds for the generic families,** following §4.1:
  - the embedding stage: `embed` {scale} and `learned-pos`;
  - `norm` {slot, rms or layer or none, eps, (1+w), bias};
  - `proj-qkv` {bias};
  - `qk-norm` {per head or whole vector, eps, (1+w)};
  - `rope` {table, mscale, interleave, m-RoPE section and interleave};
  - `attn-temp` {beta, original length};
  - `attention` {window, softcap, scale, order};
  - `out-gate` {activation, per head or per element};
  - `proj-o` {bias};
  - `ffn` {gated with its activation, non-gated with its activation and biases, or MoE with its router's parameters};
  - `residual` {one output or two};
  - the head stage: the final norm, the head (tied or not), the final softcap and the logit scale.
- **The own loops add their mixers and their variants as kinds:**
  - `deltanet`, `mamba2`, `kda`, `short-conv`;
  - MLA as an `attention` order with its own projections;
  - gemma4's per-layer embedding and shared KV;
  - gpt-oss's sinks and expert;
  - llama4's chunked and NoPE layers.
  
  Each is added in the step that moves its family.
- **The CPU's reduction order is a parameter of each operation that has more than one.** `attention`'s order is one of the
  forms listed above, and a norm's is four-way or sequential. Each family keeps the order it computes in today. A GPU backend
  ignores the parameter. Unifying the orders would move numbers, so it is a separate change with its own validation, not
  part of OP3.
- **Every implementation reads its parameters from the operation, not from the `Architecture`.** That is why a planted
  parameter can go red, and gate L checks it for every parameter.
- **What is not an operation stays outside the program,** as a hook in the walker around the operations:
  - LoRA;
  - the sublayer capture and the layer capture;
  - the layer pager, the expert pager and per-layer cancellation;
  - `decodeTiming` and the test seams.
- **Three walkers,** one per generic path, run the one program.
  - A path that declines today declines by operation kind or parameter instead: `canBatchN`, and the multi-sequence
    step's eligibility.
  - The declines' outcomes do not change in OP3.
- **The old generic loops stay in the tree until OP3's last step,** behind a test hook. Gate S needs both arms in one
  process. The last step deletes them.

**Steps.** Each step is its own PR, merged by hand on green CI, and each passes gates I, P and L before it merges.
- **OP3a:** the program, the builder for the generic families, and the per-token walker.
- **OP3b:** the batched walker and the multi-sequence walker.
- **OP3c to OP3k:** the own loops, one family per step, each with its batched twin where it has one, and its
  `ownForwards` entry removed as it moves. The order is gpt-oss, llama4, deepseek, bailing_hybrid, lfm2, granitemoehybrid,
  nemotron_h, qwen3_5_moe, gemma4: the smallest loops first, the two with batched twins last.
- **OP3z:** the old loops and the table go.

**Gates.**
- **I, identity (tier 0, every step).**
  - The instrument: `go run ./cmd/gate identity <base> <head> -backend cpu -assets tiny`, all families, quants f32,
    int8int8 and int4, the default steps.
  - The base is the commit the step branched from. For OP3a that is main `2c56d9c6`, or the merge it branches from if main
    has moved.
  - **Pass:** every family that runs is IDENTICAL. A DIFFERENT or WITHIN TOLERANCE family is red.
  - NOT RUN cells are listed and never counted. Today's are mellum (a 4.2 GB slice), qwen2 and qwen3 (no tiny asset),
    deepseek_v2 (no tiny fixture: deepseek_v3's covers the MLA loop), and qwen3_asr and voxtral (their goldens are zips).
  - **Identity runs `Generate`, so it does not reach every path.** It misses the multimodal seams, LoRA, the captures, MC3c
    and speculative verify. Those are covered by the touched packages' own tests (`scripts/touched.py`) and, once before each
    push, by the goldens-gated parity refresh.
- **P, plants (tier 0, every step, on the walkers it adds).** A test hook changes the built program before a run on the tiny
  fixtures. Each plant must change the logits of at least one family the step covers:
  - **a dropped operation:** layer 0's second `residual`;
  - **two swapped operations:** layer 0's `qk-norm` and `rope` on a family with a QK-norm, and its `norm` and `proj-qkv`
    on one without;
  - **a parameter off by one:** `rope`'s rotary width one pair short, and `attention`'s window one key short on a family
    with a window.
- **L, every parameter is live (tier 0, every step).**
  - For each (kind, parameter) pair the builder emits for a family the step covers, one test changes that parameter alone
    and requires the logits to change. An integer parameter goes off by one; any other is doubled, or toggled if it is a
    flag. Each change runs on a family whose fixture carries the parameter.
  - A pair that no fixture reaches is listed by name in the step's record, never counted as passing.
  - This gate is what shows an implementation reads the program and not the `Architecture`. P checks three parameters;
    L checks all of them.
- **T1, the size-switched paths (tier 1, by day, on the steps that change a walker: OP3a, OP3b, and qwen3_5_moe's
  batched twin).**
  - The instrument: `gate identity -assets real -only coder-0.5b-q4km`.
  - The real prompts are 145 and 621 tokens. Those cross the grouped kernel's 128 keys and the fast-attention floor at
    512 (on by default). A third, long prompt is added to the set, about 2,500 tokens, so the prefill crosses G40's
    2,048-row chunk.
  - **Pass:** IDENTICAL. About 3 minutes.
- **S, speed (tier 1, by day, OP3a and OP3b).**
  - The instrument: a whole-token in-process A/B, the walker against the old loop (the test hook), interleaved ABBA in
    one process. The shape is F3's harness (`decoder/kv_f16_ab_test.go`). The cells are the 0.5B at depths 128 and 4,096,
    with 6 and 4 pairs. The quantity is the decode-rate ratio, walker ÷ old.
  - **Pass:** the median is at least 0.99 at both cells.
  - **Fail:** a cell resolves slower (every pair under 1) with its median under 0.99. The walker is fixed before the
    step merges.
  - **Parked:** a cell does not resolve and its median is under 0.99. It goes to the served two-arm gate
    (`BENCH_ENGINES=goinfer,goinfer_old`) at night, fit-checked first.
  - About 8 minutes, under the timing lock.
- **No other night run.** Identity inherits each family's validation (TE6(b)).

**Void conditions:**
- an identity run whose base is not the named commit;
- a NOT RUN cell counted as a pass;
- gate S run while another timed run holds the lock, or run without the lock.

**Stopping rule.** Each gate runs once per step, at the step's head, after its last merge of main. A red is fixed and the
gate is re-run. The bar never moves.

**Predictions:**
- I is IDENTICAL at every step (0.9). The risk is an own loop whose order this census misread.
- S is within 0.99 at both cells (0.85). The walker adds one dispatch per operation, about 15 per layer per token.

**For the owner,** if either should be otherwise:
- the reduction order as an operation parameter, rather than unifying the orders inside OP3;
- the old loops kept until OP3z, for gate S.

**OP3a's readings, 2026-10-16 (branch `op3-layer-program-2026-10`). P, L and T1 ran at `439bd3bf`. S ran at `d5752b99`, which adds only the S test. I ran at `1ce5f447`, which has the same code as `d5752b99` plus this record.**

- **I, identity: PASS.** Against `2c56d9c6` on the tiny assets: 37 families IDENTICAL at f32, int8int8 and int4, none
  different. The six NOT RUN are as listed above. Log: `docs/measurements/op3-2026-10/op3a-identity-tiny.log`.
- **P, plants: PASS.** Each plant changed the logits on every fixture it applies to:

  | plant | fixtures changed |
  |---|---|
  | layer 0's second `residual` dropped | 33 of 33 |
  | `qk-norm` and `rope` swapped, or `norm` and `proj-qkv` | 33 of 33 |
  | `rope` one pair short | 32 of 32 |
  | `attention`'s window one key short | 9 of 9 |

- **L, every parameter is live: PASS.**
  - Every (kind, parameter) pair the builder emits changes some fixture's logits, or errors when changed (`attention.Order`, read
    and refused by the walker), except six.
  - **Listed, with the reason in the test (`decoder/program_test.go:programParamUnreached`):**
    - `ffn.MoE.NGroup` and `TopkGroup`: no generic fixture routes by group;
    - `ffn.MoE.NumExperts`: a shape, which the per-token path never reads;
    - `ffn.MoE.PerExpertScale` and `RouterPreNorm`: Gemma 4's router extras, read only by its own loop;
    - `head.Norm.Slot`: the head has one norm weight.
  - **Five pairs only GPT-2 reaches:** learned positions, the norm and head-norm biases, and the q/k/v and o biases.
    - Every tiny fixture's biases are exactly zero, as HF initialises them, including `glm-tiny-bias`'s. Their goldens cannot
      see a bias.
    - `testdata/gpt2` is gitignored, so where it is absent those five report NOT RUN, never pass
      (`programParamNeeds`).
  - The m-RoPE pairs are reached by giving the VL fixtures' cache distinct (t, h, w) positions; a text prompt's are equal.
- **T1, the size-switched paths: PASS.** `gate identity -assets real -only coder-0.5b-q4km -long`: prompts of 145, 621 and
  2,593 tokens, read from the kept dumps, 32 decode steps each, IDENTICAL. Log:
  `docs/measurements/op3-2026-10/op3a-T1-real-0.5b.log`.
- **S, speed: depth 4,096 PASS, depth 128 PARKED.** The in-process A/B on the 0.5B, decode-rate ratio walker ÷ old:

  | cell | pairs | median | range | pairs under 1 | read |
  |---|---|---|---|---|---|
  | depth 128 | 6 | 0.980 | 0.905–1.011 | 4 | **parked:** unresolved, under 0.99 |
  | depth 4,096 | 4 | 0.997 | 0.982–1.026 | 3 | pass |

  - Load average was 7.2 at the end (Spotlight indexing). Log: `docs/measurements/op3-2026-10/op3a-S.log`.
  - **Diagnostics, by day, not a grade:**
    - allocations per token are equal, 69 against 69 on `llama-tiny`;
    - per-token time on `llama-tiny`, where the walker's dispatch is the largest share of a token, is 247–252 µs walker
      against 249–260 µs old, over four alternating rounds.
- **Found while building, and fixed in OP3a:**
  - Four tests changed the `Architecture` after load to simulate another configuration, so they now rebuild the program
    (`rebuildProgramForTest`).
  - OP1's softcap-off test hook zeroed the softcap after the program was built, so the CPU would have kept capping. It
    rebuilds the program now.

**Gate S's served cell (amendment, 2026-10-16, before it runs: the details the registration left open).** The parked cell
runs at night as registered.
- The instrument is `docs/measurements/op3-2026-10/run-S-served.sh`: CPU decode on the 0.5B at depth 128, two arms and no
  peer.
  - The walker is `d5752b99`, and the old arm is `2c56d9c6`.
  - Both binaries were built by day and fit-checked: each admits the 0.5B, with 0.3 GB of weights against a 4.1 GB budget.
- `BENCH_RUNS=5`, two passes, the engine order reversed in the second.
- **The quantity:** the median decode rate, walker ÷ old, per pass. The reading is the geometric mean of the two passes.
- **Pass:** at least 0.99. **Fail:** under 0.99, and the walker is fixed before OP3a merges. Pass 2 is not skipped.
- OP3a merges after this cell reads. OP3b builds on the branch meanwhile.

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
