# decoder: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `decoder`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## QuantBackend

Moved from `decoder/backend.go` (the comment above `QuantBackend`) on 2026-10-10.

```text
QuantBackend is an optional Backend extension: a backend that can run the
int8×int8 (W8A8) weight matmul on-device. linalg.WeightMat type-asserts for it and
routes the W8A8 path through it — keeping the weight resident, keyed by the q8
slice's backing pointer — falling back to the CPU kernel when the backend
doesn't implement it or a call declines (returns false on any GPU error, so
results stay correct). The fused qkv / gate-up batch dispatches
(MatmulBTW8A8Batch) are a CPU optimization and are NOT routed here yet; full
decode coverage needs a batch equivalent (a follow-on).
```

## QuantBackend4

Moved from `decoder/backend.go` (the comment above `QuantBackend4`) on 2026-10-10.

```text
QuantBackend4 is QuantBackend's int4 (W4A8) counterpart — G6 (docs/tasks/task-gpu-paths-2026-09.md),
the "staged int4" item: matmulInto's int8 branch already checked QuantBackend before falling
back to the CPU kernel, but its int4 branch never consulted a backend at all, so an int4 model
on the STAGED (non-resident) path ran every projection on the CPU regardless of which backend
was active. Keyed by the packed-nibble slice's backing pointer, same residency convention as
QuantBackend. Declines (false) fall back to linalg.WeightMat's own CPU W4A8 kernel, so results
stay correct either way.
```

## QuantBatchBackend4

Moved from `decoder/backend.go` (the comment above `QuantBatchBackend4`) on 2026-10-10.

```text
QuantBatchBackend4 is QuantBatchBackend's int4 (W4A8) counterpart (P-16, audit-2026-09-10):
staged int4 had no batch dispatch on any GPU backend, so a fused q/k/v or gate/up call on an
int4 model paid one sync PER PROJECTION instead of one for the whole group — exactly the
per-dispatch overhead QuantBatchBackend exists to remove for int8, never extended to int4.
Same decline contract: false falls back to the CPU batch kernel.
```

## AutoBackend

Moved from `decoder/backend.go` (the comment above `AutoBackend`) on 2026-10-10.

```text
AutoBackend resolves a backend of "auto": the first GPU backend this binary links whose device probe answers, cuda
then metal, else "cpu". A GPU backend it picks that then cannot build a model resident still declines to the CPU path
with its own message, as an explicit -backend cuda or metal does.

R17 (docs/tasks/task-first-hour.md, cold-user run 3): the CLIs defaulted to "cpu", so on a machine with an NVIDIA GPU
the first run took 19.3 s at 14.4 tok/s where -backend cuda took 9.1 s at 173.8 tok/s, under a README that says the
GPU is built in.
```

## CompiledBackends

Moved from `decoder/backend.go` (the comment above `CompiledBackends`) on 2026-10-10.

```text
CompiledBackends lists the backends this BINARY can actually run: "cpu" (always) plus every
name a linked module registered from init(). It is the compiled-in truth, not a menu of
accepted flag values — --backend accepts "metal" on a Linux CPU-only build and falls back.

R2 (docs/measurements/cold-user-2026-09-06.md): the released darwin asset was built from the
root cmd/serve, so it contained no Metal backend at all, and nothing on the binary said so.
A user could only discover it by loading a model and reading a warning that scrolled past.
```

## RegisterMemoryProbe

Moved from `decoder/backend.go` (the comment above `RegisterMemoryProbe`) on 2026-10-10.

```text
RegisterMemoryProbe registers a live free-memory query for a named backend, for Model.Plan
(decoder/fitplan.go) and `goinfer-chat fit` to call without decoder importing the cgo/GPU
packages that know how to ask (the same "register from init(), decoder stays clean" shape
RegisterBackend already uses, for the same reason). metal/backend.go registers "metal" with
70% of `hw.memsize` — NOT a live query, matching its OWN resident guard's existing budget
exactly (darwin's UBC makes "available" memory unreliable, so the guard never asks for it —
see metal/backend.go's residentMemFraction comment); cuda/backend.go registers "cuda" with
the CUDA driver's live MemInfo(). ok=false means "unknown" (no device, no driver, a query
error) — Plan's own contract treats an unknown freeBytes as "cannot judge, proceed", so a
probe should never fabricate a number to avoid returning ok=false.
```

## cpuBackend

Moved from `decoder/backend.go` (the comment above `cpuBackend`) on 2026-10-10.

```text
cpuBackend dispatches the hot matmul to the shared linalg package
(M7): SIMD dot kernels (AVX2/NEON) parallelized across output columns. The
math is identical to the previous naive triple-loop — the decoder parity
tests (which match HF exactly) still pass — just multiple-× faster.
```

## backendNames

Moved from `decoder/backend.go` (the comment above `backendNames`) on 2026-10-10.

```text
backendNames resolves what the caller REQUESTED against what will actually execute, and the
reason they differ.

NewBackend answers a not-built-in request by returning the CPU backend AND an error — a
deliberate fallback, not a failure. Nothing recorded which of the two names was true, so
callers printed the requested one. A cold-user run against v0.16.0 caught the result on two
consecutive lines: "decoder: metal backend not built in … using cpu", then
"loaded 28-layer model … [backend=metal quant=int4]". The warning scrolls; the status line is
what gets pasted into an issue. On that Mac it was the difference between 37.9 and 82.3 tok/s
(docs/measurements/cold-user-2026-09-06.md, finding #3).

req is normalised ("" means cpu) so a caller that passed nothing does not report an empty
backend. reason is "" when nothing was declined.
```

## Session.cleanCache

Moved from `decoder/session.go` (the comment above `Session.cleanCache`) on 2026-10-10.

```text
cleanCache: the cache was fully reset and nothing has run through it since, so a second TruncateTo(0) would find nothing to do (audit R-16). True only from
NewSession (a born-zero cache) and Reset; rewindForReuse consumes it. The zero value is the safe side: a session restored from a snapshot, and every session
after its first generation, takes the full reset. It is NOT a general "the state is clean" claim; the extra checks in rewindForReuse (no tokens, pos 0) exist
because tests and snapshot loaders build caches by hand.
```

## Session.UseAdapter

Moved from `decoder/session.go` (the comment above `Session.UseAdapter`) on 2026-10-10.

```text
UseAdapter activates a compute-time LoRA adapter (Model.LoadAdapter, #7) for
this session's subsequent Generate calls, so a single resident base serves many
fine-tunes without paying its RAM per adapter. Switching adapters changes the
projections, so any KV built under a different (or no) adapter is stale: the next
Generate notices and prefills cold rather than reusing it (since 2026-10-08; before,
the caller had to Reset).
```

## Session.rewindForReuse.clean

Moved from `decoder/session.go` (the comment above `Session.rewindForReuse.clean`) on 2026-10-10.

```text
R-16: a session that was just created, or Reset (sessionLRU.fresh's eviction path), has already had everything TruncateTo(0) clears cleared, and nothing has
run since, so the reset below would zero a recurrent state (about 63 MB for a 30-layer DeltaNet model, an estimate) that is already zero. Every guard fails
toward doing the reset: the flag is consumed here whatever happens, and a session with tokens or a cache past position 0 never skips.
```

## Session.reconcile.recurrent

Moved from `decoder/session.go` (the comment above `Session.reconcile.recurrent`) on 2026-10-10.

```text
Recurrent (Mamba-2 / Gated DeltaNet) rolling state is not positional: TruncateTo cannot rewind
it (it reports inexact), and a mid-sweep error can leave it over-advanced past the committed KV.
Clamping seq to cache.Pos() makes the truncate a no-op (exact), so the corrupt state would be
warm-reused on the next call and decode a new sequence from leaked state (C-01 class). On any
rollback, reset a recurrent session to cold so the next call re-prefills (audit R-14).
```

## Session.reconcile.truncate

Moved from `decoder/session.go` (the comment above `Session.reconcile.truncate`) on 2026-10-10.

```text
CONSUME THE BOOL. TruncateTo reports whether the rewind was EXACT, and this line discarded it
— the same answer rewindForReuse above has always acted on.

It is reachable, and G18 is what made it so (3a16a4b, 2026-08-25: the batched sweep aborts per
LAYER on a cancelled context). Layers below the abort point have already commitBatch'd, so
their ring count is startPos+K while c.pos is still startPos — advanceTo runs only at the end
of a completed sweep. Truncating to c.pos then rewinds those rings by K on a WRAPPED window,
which cannot restore the rows the commit evicted: ring.truncate returns false. Ignoring that
left the aborted turn's K/V — RoPE'd at positions startPos.. — physically resident, and
s.tokens pointing at the prefix, so the next request that extends the conversation matched,
rewound to a no-op "exact", and read those rows as history for EARLIER positions. Silently
wrong attention on every local layer below the abort point, and plausible text
(audit-2026-09-02 C-05). A cancelled request reports a clean end, so nothing else notices.

Rings cannot restore evicted rows, so cold reset is the only exact answer — same remedy,
same reason, as rewindForReuse.
```

## Session.Generate.PrefillReused

Moved from `decoder/session.go` (the comment above `Session.Generate.PrefillReused`) on 2026-10-10.

```text
generateInto only sets g.PrefillReused itself on the resident/GPU path (prefillFrom == 0,
so matched == 0 here too — this write is a harmless no-op there, since that branch may
still overwrite it with the resident's own reuse count once the goroutine below runs). A
plain session (commit != nil, no adapter) always takes generateInto's CPU/staged path
instead (see its own "G3" comment), which never touches g.PrefillReused at all — without
this line, every session's real, correctly-computed prefix reuse (rewindForReuse above)
silently reported 0 to every caller, including usage.prefill_reused_tokens in the API.
```

## Session.Generate.go

Moved from `decoder/session.go` (the comment above `Session.Generate.go`) on 2026-10-10.

```text
P-01 (audit-2026-09-10): nil newCache — s.cache always exists already (the session
owns its cache's whole lifetime), so generateInto's lazy-allocation branch (cache ==
nil) is never taken for a session call; nothing here needs to construct one.
```

## Session.GenerateNgramSpeculativeAdaptive

Moved from `decoder/session.go` (the comment above `Session.GenerateNgramSpeculativeAdaptive`) on 2026-10-10.

```text
P-16: see *Model.GenerateNgramSpeculativeAdaptive's comment — Depth()
always returns 0, so every round would still pay to draft and verify
nothing. Session.Generate reuses the warm KV prefix exactly as genSpec
does, so this loses no session behavior.
```

## Session.genSpec

Moved from `decoder/session.go` (the comment above `Session.genSpec`) on 2026-10-10.

```text
Reconcile: seq == prompt + every token committed to the cache, so the session's token
list mirrors the cache exactly for the next call's prefix match — clamped to what the
cache actually holds if a forward errored (the final pending token was emitted but not
committed — one behind, same as a fresh prefill would leave it). Skip on an empty prompt:
genNgramInto rejected it without touching the cache, so reconcile(seq=[]) would TruncateTo(0)
and wipe a warm session's KV — the same guard Session.Generate has (audit R-13 / N-01).
```

## Architecture.NormPlacementLinear

Moved from `decoder/arch.go` (the comment above `Architecture.NormPlacementLinear`) on 2026-10-10.

```text
NormPlacementLinear overrides NormPlacement on layers where isLinearLayer(i) is
true — Olmo Hybrid's real departure from every other DeltaNet hybrid in this
tree (qwen3_5/qwen3_next/granitemoehybrid all use ONE scheme for both their
linear and full-attention layers). Verified against the real
modeling_olmo_hybrid.py: full-attention layers use NormPostOnly (olmo3's own
scheme, confirmed identical), but the DeltaNet layers use plain NormPre2 — two
placements in ONE model, keyed by the SAME layerIsLinear hook that already
selects the mixer. nil (every family so far, including olmo3 itself, which has
no linear layers at all) ⇒ NormPlacement applies uniformly, exactly as before
this field existed. See normPlacementAt.
```

## Architecture.AttnGate

Moved from `decoder/arch.go` (the comment above `Architecture.AttnGate`) on 2026-10-10.

```text
AttnGate selects the head-wise output gate applied to the attention context BEFORE the
output projection: GateNone (default), GateSoftplus (Laguna's g_proj, unchanged — Laguna
itself still gates via `arch.laguna != nil`, not this field, so its behavior is byte-for-byte
unaffected), GateSigmoid (Spark-X2.5's g_proj — verified against the real modeling_spark.py:
`gate = torch.sigmoid(gate_score); attn_output = attn_output * gate`, applied before
out_proj, exactly Laguna's STRUCTURE with a different activation). Distinct from
mlaParams.GateGranularity, which is MLA-only (DeepSeek/Bailing Hybrid forward,
forward_deepseek.go) and reached through a completely separate call site; this field is for
the GENERIC (non-MLA) attention forward's gate hook (applyAttnGate, attention.go/forwardn.go).
```

## Architecture.NoPositionEncoding

Moved from `decoder/arch.go` (the comment above `Architecture.NoPositionEncoding`) on 2026-10-10.

```text
NoPositionEncoding (Olmo Hybrid): true when the family genuinely has NO positional
encoding at all — neither RoPE nor learned — on any layer. The released checkpoint's
rope_parameters is {"rope_theta": null}; modeling_olmo_hybrid.py's own comment says so
explicitly ("Released ckpt don't use any ROPE"). A fourth legitimate "no RoPEGlobalBase"
reason alongside LearnedPosEmbed/nemotron/mla in validateResolved's M-06 check, named
explicitly for the SAME reason those three are: so a family that simply forgot to read
rope_theta cannot look like one that deliberately has none.
```

## Architecture.AttnTempBeta

Moved from `decoder/arch.go` (the comment above `Architecture.AttnTempBeta`) on 2026-10-10.

```text
AttnTempBeta/AttnTempOrigMaxPos (Ministral 3): a position-dependent multiplicative scale on
the query, applied AFTER RoPE, on every layer — get_llama_4_attn_scale in Ministral3's own
HF source (modular_ministral3.py), literally named after Llama 4's attention-temperature
tuning: scale = 1 + beta·ln(1 + floor(pos/origMaxPos)). It is IDENTICAL in shape to the
attnTemp/floorScale primitive llama4Architecture already has (decoder/forward_llama4.go),
but Llama 4 applies it INSTEAD of RoPE on NoPE layers only; Ministral 3 applies it ON TOP OF
RoPE on every layer, which llama4's own dedicated forward has no path for. Generalized here
as generic Architecture fields (0 ⇒ off, so every existing family is unaffected) rather than
copying llama4's own-forward path, since the formula is a straightforward postfix to the
generic causalAttention RoPE step. scale ≡ 1 for pos < origMaxPos (floor(pos/origMaxPos)=0),
so a SHORT test prompt exercises nothing — this is the "minimal repro hides the bug" trap
this repo's own culture warns about; the tiny fixture's prompt is deliberately longer than
origMaxPos.
```

## Architecture.kda

Moved from `decoder/arch.go` (the comment above `Architecture.kda`) on 2026-10-10.

```text
kda, when non-nil, marks Bailing Hybrid's (Ling 3.0) Kimi Delta Attention linear-attention
layers, alternating with MLA (mla, above) every LayerGroupSize-th layer — layerIsLinear
picks which, the SAME hook qwen35's Gated-DeltaNet hybrid uses. Structurally a delta-rule
recurrence like Gated DeltaNet, but with a PER-CHANNEL decay (one value per row of the state
matrix) where Gated DeltaNet's is one scalar per head — verified against fla-org/
flash-linear-attention's actual source, not the HF modeling file's opaque Triton-kernel
call. Own forward (forward_bailing.go). nil for every other family.
```

## llama4Params.chunkSize

Moved from `decoder/arch.go` (the comment above `llama4Params.chunkSize`) on 2026-10-10.

```text
chunkSize is attention_chunk_size (8192 on Scout/Maverick): the RoPE layers use a
BLOCK-DIAGONAL chunked mask, so a query at position p attends only to keys in its own
chunk, [(p/C)*C, p]. NoPE layers stay full-causal.

M-05: this was read from config and then dropped, and the forward attended [0, pos] on
every layer. Below C that is identical to chunked — which is why the parity gates, which
use short sequences, never saw it — and from position C on, the RoPE layers saw keys HF
masks out. 0 means no chunking (a checkpoint that does not set the field).
```

## mlaParams

Moved from `decoder/arch.go` (the comment above `mlaParams`) on 2026-10-10.

```text
mlaParams carries DeepSeek Multi-head Latent Attention geometry. The cached state
is the compressed latent [KVLoRARank + QKRopeHeadDim] per position (the KV-memory
payoff: ~576 floats/token vs the ~41k a reconstructed full K+V would need); per-head
K/V are rebuilt from it each step via kv_b_proj. QLoRARank 0 ⇒ a direct q_proj (the
V2-Lite path) instead of the q_a/q_b LoRA bottleneck. forward_deepseek.go consumes this.
```

## mlaParams.AttnPrefix

Moved from `decoder/arch.go` (the comment above `mlaParams.AttnPrefix`) on 2026-10-10.

```text
AttnPrefix/DenseSuffix override the tensor-name prefix/output-projection suffix. ""
(deepseek_v2/v3, kimi_k2) ⇒ "self_attn"/"o_proj.weight". Bailing Hybrid (Ling 3.0) uses
"attention"/"dense.weight" instead — verified against the real modeling_bailing_moe_v3.py,
whose BailingMoeV3DecoderLayer assigns BOTH its MLA and KDA mixers to self.attention (not
self.self_attn), and whose MLA class names its output projection self.dense.
```

## mlaParams.GateGranularity

Moved from `decoder/arch.go` (the comment above `mlaParams.GateGranularity`) on 2026-10-10.

```text
GateGranularity ("" | "head_wise" | "element_wise"): Bailing Hybrid's optional per-head or
per-element sigmoid output gate (self.g_proj) applied to the attention context BEFORE the
output projection — the same STRUCTURE Laguna's own attention-output gate already ships, but
sigmoid-activated where Laguna's is softplus (verified against source, not assumed). ""
(every DeepSeek family) ⇒ no gate.
```

## kdaParams

Moved from `decoder/arch.go` (the comment above `kdaParams`) on 2026-10-10.

```text
kdaParams carries Bailing Hybrid's (Ling 3.0) Kimi Delta Attention geometry for the
linear-attention layers; the MLA layers use mlaParams above. HeadDim/NumHeads are shared by
q/k/v (no GVA — verified against the real modeling_bailing_moe_v3.py, where
BailingMoeV3KimiDeltaAttention sets head_k_dim = head_dim and num_k_heads = num_heads
unconditionally). NoLora selects a single f_proj/g_proj linear per gate (Ling-3.0-tiny's own
value) over a LoRA'd a/b-split pair — the LoRA'd path is NOT implemented (no released
checkpoint needs it yet; see kdaArchitecture's own "what was deliberately not done").
```

## qwen35Params.AttnGate

Moved from `decoder/arch.go` (the comment above `qwen35Params.AttnGate`) on 2026-10-10.

```text
AttnGate: this family's full-attention (non-DeltaNet) layers use qwen3.5's own
double-width q_proj scheme — [query ‖ gate] per head, interleaved, with the attention
context multiplied by sigmoid(gate) before o_proj. True for qwen3_5/qwen3_5_moe/
qwen3_next (verified against their real modeling_qwen3_5*.py: Qwen3_5Attention Projects
q_proj to 2*num_heads*head_dim). FALSE for Olmo Hybrid: its full-attention layer is
olmo3's own PLAIN scheme (ordinary q_proj, no gate) — verified against the real
modeling_olmo_hybrid.py, which reuses Olmo3Attention verbatim for these layers, not
qwen3.5's gated one. G5 (docs/tasks/task-gpu-paths-2026-09.md): this field did not exist before
Olmo Hybrid — the resident backends assumed EVERY qwen35Params-carrying family's softmax
layer was qGate, which was true of every family that had reached residency until now.
```

## qwen35Params.SeparateConv

Moved from `decoder/arch.go` (the comment above `qwen35Params.SeparateConv`) on 2026-10-10.

```text
SeparateConv (Olmo Hybrid): the checkpoint stores the depthwise causal conv
as THREE separate tensors, q_conv1d/k_conv1d/v_conv1d, split at the SAME
q/k/v channel boundaries the mixed_qkv activation uses (keyDim, keyDim,
valueDim rows respectively) — verified against a real Olmo-Hybrid-7B
checkpoint's safetensors header, not assumed from source (the modeling code
alone shows one combined self.conv1d; only the real file's actual tensor
names and shapes revealed the three-way split). loadQwen35Attn concatenates
them in q,k,v order to reconstruct the same [convDim,1,K] layout
gatedDeltaNetStep's conv step already expects.
```

## Architecture.kvDimAt

Moved from `decoder/arch.go` (the comment above `Architecture.kvDimAt`) on 2026-10-10.

```text
kvDimAt returns the number of f32-equivalent elements layer i's K (or V — the two are always
equal width) cache actually stores PER POSITION it holds — zero for a linear/mamba/conv mixer
layer (M-28, docs/audit-2026-09-10.md: these hold no position-indexed K/V array at all, a
small fixed-size recurrent state instead, not something that grows with context), MLA's real
compressed latent width (KVLoRARank+QKRopeHeadDim — what forward_deepseek.go's cache actually
stores) instead of the full reconstructed per-head width when a.mla != nil, else the ordinary
kvHeadsAt(i)*headDimAt(i) every ordinary softmax-attention family already used. This is the
WIDTH only; kvPositionsAt below is the COUNT (a sliding-window layer holds fewer positions
than ctx once ctx exceeds its window).
```

## Architecture.hasNoAttentionKVAt

Moved from `decoder/arch.go` (the comment above `Architecture.hasNoAttentionKVAt`) on 2026-10-10.

```text
hasNoAttentionKVAt reports whether layer i holds no ordinary softmax-attention K/V array at
all: the generic linear/mamba/conv mixer cases (isLinearLayer/isMambaLayer/isConvLayer) PLUS
Nemotron's own per-layer block-kind classification. Nemotron needs its own check because its
mixer identity is per-layer RUNTIME DATA (nemotronParams.blockKind, read from
layers_block_type), not a closure registered once at resolve time the way Granite's
layerIsMamba is — isMambaLayer never fires for Nemotron's mamba layers at all, and its mlp/moe
block kinds (single-op-block: exactly one of {mamba, attention, mlp, moe} per layer,
decoder/forward_nemotron.go's own switch) touch no K/V either, which neither isMambaLayer nor
any other generic predicate was ever positioned to catch.

Found as a residual gap in M-28 (docs/audit-2026-09-10.md) while implementing P-02: the fix
there zeroed KV pricing/allocation for isLinearLayer/isMambaLayer/isConvLayer layers, but
Nemotron's mamba AND mlp AND moe layers all slipped through priced as full attention, since
none of those three generic predicates ever return true for a Nemotron layer regardless of its
real kind (confirmed directly: testdata/nemotron-tiny's `mamba,attention,mlp,mamba,attention`
layers all reported kvDimAt=32 before this fix, including the two mamba and one mlp layer that
hold no attention K/V at all).
```

## Architecture.kvPositionsAt

Moved from `decoder/arch.go` (the comment above `Architecture.kvPositionsAt`) on 2026-10-10.

```text
kvPositionsAt returns how many of ctx cache positions layer i's K/V actually needs to hold
resident: ctx itself for an ordinary layer, or SlidingWindow once ctx exceeds it for a LOCAL
(non-global) layer under a sliding-window architecture — the ring buffer never grows past its
own window regardless of how long the context gets (M-28, docs/audit-2026-09-10.md). Meaningless
but harmless for a layer kvDimAt already prices at zero (the caller multiplies the two).
```

## kvBytesForCtx

Moved from `decoder/arch.go` (the comment above `kvBytesForCtx`) on 2026-10-10.

```text
kvBytesForCtx sums the REAL per-layer KV cost at ctx positions across the whole architecture —
kvDimAt (WIDTH: zero for linear/mamba/conv, MLA's compressed latent instead of the
reconstructed per-head width) combined with kvPositionsAt (COUNT: capped at SlidingWindow for a
local layer) — the fix for M-28 (docs/audit-2026-09-10.md): the flat NumLayers×NumKVHeads×
headDim formula this replaces overpriced hybrid (DeltaNet/conv/Mamba), sliding-window, and MLA
models 3-7x by charging every layer full softmax-attention KV regardless of what it actually
caches. Used by decoder/fitguard.go's load-time host-RAM guard and decoder/prefill_budget.go's
request-time guard, both Config-only (no loaded weights yet) via resolveArchitecture. NOT used
by decoder/fitplan.go's device-VRAM Plan(), which needs only the WIDTH half (kvDimAt) via its
own per-layer loop — see that file's kvBytesPerPositionAllLayers doc comment for why the
sliding-window COUNT cap is deliberately not applied there.
```

## ownForwards

Moved from `decoder/arch.go` (the comment above `ownForwards`) on 2026-10-10.

```text
ownForwards is THE list of families that do not use the generic layer loop — one table, so that
"runLayers dispatches here" and "the batched path must not touch this" are the same fact.

THEY WERE TWO FACTS, AND THEY DISAGREED. runLayers dispatched LFM2 to runLayersLFM2 while
canBatchN's hand-written exclusion list — gemma4, qwen35, granite, nemotron, mla, llama4, gptoss
— simply did not mention it. So every prompt of ≥2 tokens ran the DENSE ATTENTION STACK over
LFM2's 22 conv layers, whose QProj/KProj/VProj/OProj/QNorm/KNorm are never loaded: rmsNorm
indexed a nil weight slice at layer 0 and the process died, since the panic is in the Generate
goroutine where net/http's handler recover cannot reach it. PrefillPath() published "batched
shape" for the same model at startup. Reproduced on the committed testdata/lfm2-tiny fixture
with a 4-token prompt (audit-2026-09-02 C-01; three reviewers found it independently).

This is audit §0 theme 1 — "one predicate, seven consumers" — applied to the first two. A family
added to this table is excluded from the batched path by construction, and
TestOwnForward_tableNamesEveryFamilyForward fails if a runLayersXxx is written that is not here.
```

## Architecture.hasAttnOutputGate

Moved from `decoder/arch.go` (the comment above `Architecture.hasAttnOutputGate`) on 2026-10-10.

```text
hasAttnOutputGate reports whether this architecture gates the attention context before o_proj:
Laguna's softplus gate, or any family with AttnGate == GateSigmoid (Spark-X2.5). One predicate for
the forward's two dispatch sites (attention.go, forwardn.go) and the resident feature it derives
(features.go's FeatAttnOutputGate) — it was the same OR written three times.
```

## Architecture.kvInt8OK

Moved from `decoder/arch.go` (the comment above `Architecture.kvInt8OK`) on 2026-10-10.

```text
kvInt8OK reports whether NewCache may store this architecture's K/V as int8. A family with its own
layer loop only when its ownForwards entry says KVInt8; the generic loop always reads K/V through the
int8-aware attention path, except for MoE, where attention runs the acc64 kernel so expert routing
stays bit-stable, and a quantized cache would reopen that.

It replaces a hand-written list (gemma4, qwen35, granite, nemotron, lfm2, llama4, and MoE) that a
second list — the ring-buffer one, kvRingsOK — contradicted on lfm2, llama4 and gpt-oss. Neither
contradiction was live: lfm2 and llama4 have no sliding window, so rings were a no-op for them, and
gpt-oss is MoE, so int8 was already off. Two guards that are right by coincidence are how the next
family breaks; this is one rule, in the table.
```

## Config.MoeSharedExpertIntermediateSize

Moved from `decoder/config.go` (the comment above `Config.MoeSharedExpertIntermediateSize`) on 2026-10-10.

```text
MoeSharedExpertIntermediateSize (Nemotron-H MoE, e.g. Nemotron 3 Nano) is its
OWN explicit shared-expert width — verified NOT derivable as
NSharedExperts*MoeIntermediateSize the way DeepSeek's is (Nano ships
n_shared_experts=1, moe_intermediate_size=1856, but
moe_shared_expert_intermediate_size=3712 — the shared expert is 2x a routed
expert's width, not 1x). A distinct field, not reused, to avoid silently
mis-deriving it for this family.
```

## Config.MoeRoutedScalingFactor

Moved from `decoder/config.go` (the comment above `Config.MoeRoutedScalingFactor`) on 2026-10-10.

```text
Laguna (poolside). MoeRoutedScalingFactor is its spelling of
routed_scaling_factor (2.5 on the XS generations, 1.0 on M.1).

Gating is json.RawMessage because the SAME field ships with three different
JSON types across three releases of one model_type: "per-head" (XS-2.1),
true (XS.2), "per-element" (M.1). Decoding it into a string or a bool would
fail on the other spellings, so it is held raw and resolved by
lagunaGatePerHead, which mirrors the vendor's own two-line rule.

NumAttentionHeadsPerLayer is layer i's QUERY head count ([48,64,64,64,…] —
48 on full_attention, 64 on sliding_attention). Absent on M.1, which is
uniform. MlpOnlyLayers lists the layers that are plain dense MLPs rather
than MoE ([0] on XS, [0,1,2] on M.1) — contiguous from the top in every
released config, so it maps onto FirstKDense.
```

## Config.LayerGroupSize

Moved from `decoder/config.go` (the comment above `Config.LayerGroupSize`) on 2026-10-10.

```text
Bailing Hybrid (Ling 3.0, model_type "bailing_hybrid"): MLA (above) alternating with Kimi
Delta Attention (KDA) every LayerGroupSize-th layer being MLA instead — verified against the
real modeling_bailing_moe_v3.py's BailingMoeV3DecoderLayer.__init__, not assumed from the
task brief's paraphrase: `layer_types` is NOT a config.json field at all for this family (no
released checkpoint carries it); the pattern is COMPUTED from LayerGroupSize, same shape as
Qwen3-Next's FullAttentionInterval. NumSharedExperts/ShortConvKernelSize etc. use this
family's OWN JSON key spellings (num_shared_experts, not DeepSeek's n_shared_experts) —
confirmed against the real inclusionAI/Ling-3.0-tiny config.json, which uses num_experts/
num_shared_experts throughout, not deepseek_v3's n_routed_experts/n_shared_experts.
```

## Config.HybridOverridePattern

Moved from `decoder/config.go` (the comment above `Config.HybridOverridePattern`) on 2026-10-10.

```text
Nemotron-H (NemotronH): single-op-per-block hybrid (layers_block_type entries
"mamba" | "attention" | "mlp"), NoPE attention, non-gated relu² MLP. Its
Mamba-2 uses its own key spellings (mamba_num_heads / mamba_head_dim /
ssm_state_size / n_groups / conv_kernel) and layer_norm_epsilon for eps.

HybridOverridePattern is the SAME layer sequence in NVIDIA's released spelling:
one character per block ("M" mamba, "*" attention, "-" mlp), e.g.
"M-M-M-MM-M-M-M*-…". Every released NemotronH config.json carries this and NOT
layers_block_type — which is transformers' internal spelling, and therefore the
only one the tiny fixtures (built by instantiating NemotronHConfig) ever emitted.
Reading just the fixture spelling made the loader reject every real checkpoint.
```

## Config.MRopeSection

Moved from `decoder/config.go` (the comment above `Config.MRopeSection`) on 2026-10-10.

```text
MRopeSection is Qwen2.5-VL's m-RoPE head_dim/2 split across the (temporal,
height, width) position components. The qwen2_5_vl and qwen3_vl adapters extract it
from the nested rope_parameters or rope_scaling (clearing rope_scaling), so it is
serialized under its own key: with `json:"-"` a .giw dropped it and a Qwen-VL loaded
from one ran plain RoPE on image positions, silently (found 2026-10-09 by S16's real gate
through a directory sidecar; text alone cannot show it). nil = plain scalar RoPE. (P5)
```

## Config.UseBidirectionalAttention

Moved from `decoder/config.go` (the comment above `Config.UseBidirectionalAttention`) on 2026-10-10.

```text
UseBidirectionalAttention (gemma4, P7 vision serving): "vision" on
26B-A4B/31B checkpoints enables a blockwise bidirectional attention mask
over image/audio blocks on SLIDING (local) layers only — global layers
stay strictly causal (create_masks_for_vision_model; docs/multimodal.md's
P7 entry, verified against modeling_gemma4.py, not assumed). Empty/null
on E2B/E4B, where image-block attention is plain causal — exactly what
GenerateGemma4VL's sequential embed-by-vector prefill
(decoder/generate_gemma4_vl.go) already produces. A non-empty value ("vision",
26B-A4B/31B) is SERVED, not refused: GenerateGemma4VL dispatches it to the genuinely
batched, blockwise-masked forward (prefillLogitsGemma4VLBidirectional /
runLayersGemma4FromEmbedN, decoder/forward_gemma4_batched.go) rather than the
sequential causal path E2B/E4B use. Lives under text_config in a real checkpoint,
picked up automatically by loadConfig's existing text_config merge — same shape as
PadTokenID above.

Typed as gemma4BidirectionalAttention, NOT plain string: Config is one
flat struct shared by every family, and a REAL, unrelated field of the
SAME NAME already exists on Gemma 3's own config (use_bidirectional_attention
as a bool — confirmed on testdata/gemma-3-270m/config.json and every
gemma3-vl-tiny fixture, a pre-existing field, different semantics). A
plain `string` field failed to unmarshal that bool with a hard error,
breaking gemma3 loading entirely — found by running the full decoder
suite, not assumed safe. The tolerant type accepts a real string
(gemma4's own case) and silently resolves anything else (bool, null,
absent) to "" — exactly the "not gemma4's own field, or genuinely unset"
case this check needs to treat as "no bidirectional attention."
```

## Config.NormEps

Moved from `decoder/config.go` (the comment above `Config.NormEps`) on 2026-10-10.

```text
LFM2 (model_type lfm2): the gated short-convolution block's geometry. Its
per-layer pattern rides on LayerTypes above ("conv" | "full_attention").

ConvLCache is the conv KERNEL WIDTH (3), named for the rolling state it implies
rather than for the filter — upstream calls it conv_L_cache. ConvDim is the
channel count the block operates on, which equals hidden_size on every released
checkpoint but is configured separately, so it is read rather than assumed.
ConvBias is false on the released weights and there is no bias tensor to load;
a true here is refused rather than silently ignored.

ConvDim is OPTIONAL and often absent. Upstream Lfm2ShortConv builds its Conv1d and
in_proj on config.HIDDEN_SIZE and never reads conv_dim at all, so hidden_size is the
authority: the released LFM2.5-2.6B config.json carries conv_dim (2048, equal to
hidden_size), while a checkpoint written by Lfm2Config.save_pretrained carries no
conv_dim key whatsoever. Absent ⇒ default to hidden_size (what the reference uses);
present-and-different ⇒ refused, because the reference would ignore it and we would
not, which is a silent divergence rather than a shape error.
NormEps is LFM2's RMSNorm epsilon. IT HAS ITS OWN JSON KEY: LFM2 writes
"norm_eps" where every other RMSNorm family here writes "rms_norm_eps", so
reading cfg.RMSNormEps for this family yields 0, not the checkpoint's 1e-5.
That is not a rounding difference. Measured 2026-08-31 on LFM2.5-2.6B: eps=0
scaled the first operator_norm output by a uniform 1.0185x (the embedding's
variance is ~2.9e-4, so rsqrt(v)/rsqrt(v+1e-5) is a visible factor), and the
error compounded through 61 norms into logits at cosine 0.897 vs HF -- with a
MATCHING argmax, so a greedy-decode smoke test would have called it correct.
The checkpoint also carries "block_norm_eps"; upstream Lfm2Config reads
norm_eps, so that one is deliberately not used.
```

## Config.gemma4GlobalGeometry

Moved from `decoder/config.go` (the comment above `Config.gemma4GlobalGeometry`) on 2026-10-10.

```text
gemma4GlobalGeometry returns the full-attention layers' head_dim and KV-head count. The flat
global_head_dim / num_global_key_value_heads win when present (every released checkpoint and every
config written before transformers 5.16). A config saved by 5.16 or later carries per_layer_config
instead, so a re-saved checkpoint would otherwise read 0 and load every layer at the local geometry
(found 2026-10-06 pinning gemma4-emodel-tiny: a shape error on layer 2's q_proj). goinfer has exactly
two attention geometries, so the overrides must name only full-attention layers and agree with each
other; anything else is refused rather than half-applied.
```

## Config.EOSIDs.null

Moved from `decoder/config.go` (the comment above `Config.EOSIDs.null`) on 2026-10-10.

```text
A JSON null is "no id here" (transformers writes eos_token_id: null when generation_config.json carries it). Unmarshalled into an int it
would read as 0, and id 0 would end generation: "!" in Qwen's vocabulary, so every Qwen3-ASR transcription stopped at its first "!".
```

## Model.generateInto.resident

Moved from `decoder/model.go` (the comment above `Model.generateInto.resident`) on 2026-10-10.

```text
Prefill the (divergent suffix of the) prompt and seed the first token's
logits. On the batched archs this runs the layers at M=len in one pass (each
weight streamed once — ~1.7–2× faster TTFT than sequential), LM head on the
last position only. Reuse means len here is the suffix, not the whole prompt.
GPU full-residency decode (webgpu + eligible arch + plain stateless
Generate). Prefix-reuse (prefillFrom > 0) keeps the CPU/staged path — the
session's prefix-reuse cache is CPU-side, the resident's positional KV is
GPU-side, and the two cannot both be the source of truth for a reused prefix.
Prefill = option (a): run the prompt through the resident DecodeRunner
sequentially to build its GPU KV (also warms the pipelines); the last
token's logits seed decode. O(prompt-len) GPU Runs — fast for typical
prompts since a GPU Run ≫ a CPU int4 forward (the K/V-upload bridge stays
for future prefix-reuse; batched on-device prefill, the long-prompt fix,
is deferred).

G3 (docs/tasks/task-gpu-paths-2026-09.md): a plain session (commit != nil, no adapter) still
keeps the CPU/staged path — sessions exist for prefix reuse, which the prior paragraph
already rules out combining with resident. But an ADAPTER session (cache.lora != nil) is
the one case where going through Session.Generate is NOT about prefix reuse at all — it is
the ONLY way compute-time LoRA gets applied at all (Session.UseAdapter → cache.lora →
applyLoRA in the CPU forward) — and with prefillFrom==0 it never touches the reused-prefix
conflict either. So it is safe to admit here too, PROVIDED the resident backend actually
implements ResidentAdapter; a backend that doesn't declines to CPU exactly like any other
missing resident capability, never silently running an adapter session's tokens through
the base model's resident weights (which would return correct-looking but WRONG,
base-model output — audit R-01's whole reason for existing).
P-01 (audit-2026-09-10): cache may be nil here (Model.Generate now defers allocation to
whichever branch below actually needs it) — nil-safe read, not a change in meaning: only
Session.Generate's cache ever has .lora set (an adapter is bound through a Session, never
through the plain Model.Generate path this nil case is for), so this reads exactly the
same lora value either caller would have produced.
```

## Model.generateInto.ctxcap

Moved from `decoder/model.go` (the comment above `Model.generateInto.ctxcap`) on 2026-10-10.

```text
M-01 (docs/audit-2026-09-10.md): decline BEFORE ever claiming the resident — a
prefill past this fixed KV cap fails mid-write below (residentPrefillSeed's error
a few lines down just sets g.err and returns; there is no CPU fallback once this
commits), so refuse here and fall through to the staged CPU path below, exactly as
"not resident" already does. Chiefly the adapter case: an adapter's first turn
(prefillFrom==0) reaches this branch too, and prepare() enforces MaxPositions for
it unless the caller derives residentPath from ResidentActive() (serveapp's own
M-01 fix) — this is the decoder-seam half, defense in depth for any caller that
doesn't.
```

## Model.generateInto.greedy

Moved from `decoder/model.go` (the comment above `Model.generateInto.greedy`) on 2026-10-10.

```text
Greedy fast path: when the resident can pick the argmax on-device AND the sampler's
choice is exactly argmax(raw logits) with nothing else reading them, skip the
full-logits readback (594 KB/token at a 151936 vocab). Emitted tokens are identical;
GOINFER_NO_GREEDY_FASTPATH forces the logits path (escape hatch / A-B check).
fastNext >= 0 means "the resident already picked the next token"; the first token
still comes from the prefill logits through the sampler.
P1: `top_k=1` takes this path too (GreedyEquivalent), at ANY temperature — monotone scaling
preserves ordering and a one-token distribution is deterministic, so the emitted tokens are
the same ones greedy emits. Both predicates are consulted; neither is widened, so this
routing decision is the ONLY behaviour that changes.

SCOPE — the speculative paths are deliberately NOT affected. They gate on `sp.Temperature <= 0`
directly (speculative.go, spec_grammar.go, spec_ngram.go), never on these
predicates, so `top_k=1` with a temperature stays speculative-INELIGIBLE exactly as before.
That is the conservative half of P1: making it eligible would be correct (argmax verification
reproduces greedy, which top_k=1 equals) but is a second behaviour change, and it does not
ride along silently here.

RNG: this path skips the per-token rng.Float64() draw that SampleWithInfo would make. That is
unobservable rather than merely harmless — under top_k=1 every step is deterministic, so no
later draw's VALUE can depend on the skipped ones, and no emitted token can differ. (The RNG
stream position does advance differently, which is why nothing may depend on it downstream.)
```

## Model.generateInto.topkmc3

Moved from `decoder/model.go` (the comment above `Model.generateInto.topkmc3`) on 2026-10-10.

```text
MC3: a TopKRow's Full() reads the resident's logits, and a token later another generation's step may have overwritten
them. So under MC3 the draw is resolved inside the resident call (topKPre / topKPreFull below), while this generation
still holds the resident, instead of at the top of the next iteration. It was simply off under MC3 until 2026-09-30,
which sent every CUDA top-p request, alone or not, down the full-row path: a 151,936-logit readback and a host sort per
token, 0.74× the top-K path's speed (docs/measurements/topp-regression-2026-09-30.md).
```

## Model.generateInto.cancel

Moved from `decoder/model.go` (the comment above `Model.generateInto.cancel`) on 2026-10-10.

```text
V-04 (docs/review-2026-09-04.md): a cancel that arrives during the PREVIOUS
iteration's Forward -- the dominant per-iteration cost, milliseconds against the
send-select's microseconds -- is not observed until here, at the top of the next
iteration, because nothing between generated=append(...,next) and Forward
returning is select-guarded. By the time we reach this select, that iteration's
`next` has both been appended to `generated` AND had its own Forward already run
(the resident cache write happens synchronously inside it), so the cache is
exactly as consistent as the natural-completion commit below -- this was the
ORIGINAL R-02 fix's blind spot: it only committed at the send-select exit, which
is the rarer of the two cancel-observation points, not the common one.
```

## Model.generateInto.cancelsend

Moved from `decoder/model.go` (the comment above `Model.generateInto.cancelsend`) on 2026-10-10.

```text
audit R-02: at this exit `next` has been sampled but never forwarded (its own
K/V write is later in this same iteration, which cancellation skips), so the
resident cache is consistent with exactly prompt+generated — record it instead
of leaving resIDs nil and forcing the next turn to cold-prefill an interrupt
that left nothing inconsistent behind. Agent harnesses cancel constantly
(interrupts, timeouts, disconnects), so today every one of them pays this cost.
```

## Generation.DecodeResident

Moved from `decoder/model.go` (the comment above `Generation.DecodeResident`) on 2026-10-10.

```text
DecodeResident reports whether a multimodal turn's DECODE ran on the resident after its CPU prefill was
uploaded (GenerateGemma4VL's bridge). Diagnostic, for the same reason as ImgPrefillResident: S1's G4
(docs/tasks/task-multimodal-support-2026-10.md) must show the image turn decoded resident, not on the CPU.
```

## Model.ForwardCapture

Moved from `decoder/model.go` (the comment above `Model.ForwardCapture`) on 2026-10-10.

```text
ForwardCapture runs one forward for token id and returns the next-token logits
PLUS the residual stream after each layer in `layers` (cloned) — the read-only
hidden-state seam a draft head reads (05's EAGLE-3 head fused it until its removal on
2026-09-24; block drafters read it now). The forward is byte-identical
to forward(id): the captures are copies that never feed back. Layer indices are
0-based into [0, NumLayers); out[i] corresponds to layers[i].

Wired for the generic decode path plus the own-runLayers families whose loops call
cache.captureResidual (see decoder/capture.go): qwen3_5_moe, gemma4, gpt-oss — the three
P10 block-drafting targets we hold locally with a licensed drafter. The rest still return
an error rather than silently producing nothing: granite and nemotron_h interleave recurrent
mixers whose "residual after layer l" needs deciding rather than assuming, mla and llama4_text
are simply not done. A family is wired only when BOTH its loop captures and it leaves this
list, so a half-wired one fails here loudly instead of handing back nil rows.
```

## Model.ForwardCapture.derived

Moved from `decoder/model.go` (the comment above `Model.ForwardCapture.derived`) on 2026-10-10.

```text
Derived from the dispatch table's Captures bit rather than re-listed: the families whose own
loop calls captureResidual are wired, every other own-forward family is not. LFM2 was in
neither list, so runLayersLFM2 — which never captures — returned nil rows through a seam
documented to fail loudly instead (audit-2026-09-02 C-02; the since-removed EAGLE head's fuseAt panicked on them).
```

## softcapParallel

Moved from `decoder/model.go` (the comment above `softcapParallel`) on 2026-10-10.

```text
softcapParallel applies Gemma's final-logit softcap sc·tanh(x/sc) in place, fanning the loop
out via parallelElementwise. Every element is independent and math.Tanh is deterministic, so
splitting the loop is BYTE-IDENTICAL to the serial form (disjoint writes, no reduction) --
gated by TestSoftcapParallel_bitIdentical. Mirrors metal/model.go's softcapParallel (Metal
already had this; the generic CPU decode path didn't -- task-moe-streaming.md's re-ranked
lever #3). At Gemma's 256k vocab this is a real per-token tax fanned out over GOMAXPROCS.
```

## Model.Generate

Moved from `decoder/model.go` (the comment above `Model.Generate`) on 2026-10-10.

```text
Generate streams generated token ids over the returned channel until EOS,
a stop id, maxTokens, or ctx cancellation. prompt is already-tokenized
ids (the demo runs the tokenizer). The channel closes when generation
ends; check Err after the range loop for a terminal error.

This is a RAW COMPLETION primitive: it continues prompt verbatim and knows nothing about
chat turns, roles, or a checkpoint's own template — encoding a user message directly, with no
formatting, is why an instruct-tuned model degenerates into repetition (R10, docs/measurements/
cold-user-2026-09-06-nobara-pc.md). Rendering a template is the caller's job: see
github.com/townsendmerino/goinfer/chat (chat.Detect resolves a checkpoint's own template from
its tokenizer metadata) and examples/embed/main.go for the whole sequence.

Sampling is greedy at Temperature 0, else temperature/top-k/top-p (see
Sampler). A SamplingParams.LogitProcessor, if set, masks each step's logits
before sampling — the seam for constrained/structured decoding.
```

## Model.residentPrefillSeed.shared

Moved from `decoder/model.go` (the comment above `Model.residentPrefillSeed.shared`) on 2026-10-10.

```text
residentPrefillSeed ingests the whole prompt into the resident KV and returns the
LAST token's logits — the seed for decode. Both resident generation paths call it:
generateInto (plain) and genNgramInto (speculative).

IT IS SHARED ON PURPOSE. These were two copies, and they drifted: batched prefill
was wired into generateInto by c36698a (2026-07-16) and NOT into the speculative
twin, which had carried its own per-token loop since 0fd54e8 (2026-06-20). The
speculative path therefore prefilled one token at a time on every backend that
implements Prefiller, measured at +2.66 ms per prompt token (R^2 0.9977) against
the batched path's 0.42 — a 6.3x per-token penalty that made `serve --spec ngram`
on resident CUDA 3-4.5x SLOWER than no drafter at all on realistic prompts. It went
unseen for six weeks because the only GPU speculative harness used 36-74 token
prompts and logged its speedup without asserting on it (docs/spec/02). One
function, so the next optimisation cannot land on one path and miss the other.

Batched prefill (optional Prefiller) ingests the prompt in one pass — much faster
TTFT for long prompts. It declines (falls back) past the backend's cap, is absent
for backends without a batched forward (WebGPU implements no Prefiller; CUDA and
Metal do), and is skipped for tiny prompts.

DEFAULT ON — bit-identical to sequential decode again (restored 2026-08-04). It was
briefly default-off after an 84% token-stream divergence traced to a compiler
fma-vs-mul+add contraction difference between the separately-compiled batched and
decode GEMV/RMS kernels. FIXED: every float MAC in both paths is now an explicit
__fmaf_rn (no compiler discretion), enforced at build time by cuda.TestKernelFMALint.
The decode-side half shipped in aikit/gpu@v0.25.0 (gemv_w4a8_fwd); with the dep
bumped, TestPrefillDivergenceRate is 0/50 on the real 1.5B (was 42/50), gap
byte-identical. GOINFER_BATCHED_PREFILL=0 force-disables.
See docs/task-batched-prefill-bitidentity.md.
prefillDeclineDigitsRE normalizes a decline error's varying numbers (prompt length, floor,
byte counts) out of the dedup key below, so e.g. every below-floor prompt — a different
promptLen each time — collapses to the SAME reason instead of re-triggering the warning.
```

## prefillDeclineSeen

Moved from `decoder/model.go` (the comment above `prefillDeclineSeen`) on 2026-10-10.

```text
prefillDeclineSeen is the per-reason dedup set warnPrefillDeclined reports through — see N-35
(audit-metal-2026-09-12.md). It used to be a single process-lifetime sync.Once (deliberately,
per the test this replaces), which meant the FIRST decline of any kind — on Metal, the routine
below-floor case, which fires on nearly every short prompt — permanently silenced every later
decline, including a genuinely different one (a resident-cap refusal, an OOM) an operator would
want to see. Keying per normalized reason keeps the routine case to one line (every below-floor
promptLen normalizes to the same key) while still surfacing a later, differently-worded decline.
```

## warnPrefillDeclined

Moved from `decoder/model.go` (the comment above `warnPrefillDeclined`) on 2026-10-10.

```text
warnPrefillDeclined reports, once per distinct reason, that a backend's batched prefill refused
a prompt at call time and this prompt (and every one after it with the same reason) is being
ingested one token at a time instead.

It exists because the load-time report and the runtime behaviour could disagree with nothing
saying so. PrefillPath() answers from the model's static properties, so a serve banner and
/v1/models both said "batched (one weight-stationary CUDA pass)" while every long prompt hit an
M-dependent decline — the CUDA scratch is O(M·inter), and an 8k prompt on an 8 GB card asked for
2.28 GB it did not have. Measured on qwen2.5-7b at int4: the fallback runs at 12.5 ms/token
against the batched path's 2.8, and the only visible symptom was a slow benchmark cell.
prefillChunked now keeps that case on the fast path, so this should be rare — which is exactly why
it is worth a line when it happens rather than another silent 4.5×.

Stderr and not an error: the fallback is CORRECT, just slow, and failing the request over a
performance decline would be worse than serving it.
```

## Model.residentPrefillSeed.params

Moved from `decoder/model.go` (the comment above `Model.residentPrefillSeed.params`) on 2026-10-10.

```text
from is the first position to compute: prompt[:from] is already committed to the resident
KV (prefix reuse, resident_reuse.go) and positions carry through unchanged because the cache
is positional. from == 0 is the cold path.

hasAdapter must be true whenever this call runs under a bound resident adapter (cache.lora
!= nil at generateInto, the only caller where that is possible — GenerateVL, the n-gram
spec target and GenerateSpeculative's target and draft never bind one). The batched Prefiller path is a separate encoded launch per
backend (cuda/prefill.go, metal/prefill.go) that never reads the bound delta at all, so an
adapter session that reaches it would prefill the prompt's K/V from the BASE weights and only
start applying the adapter at decode — plausible, wrong, HTTP 200 (audit C-01, 09-10). Mirrors
the CPU sequential/batched split's own rule: decoder/forwardn.go's canBatchN caller declines
batched prefill whenever cache.lora != nil, because compute-time LoRA is wired only into the
sequential forward.
```

## Model.residentPrefillSeedMRoPE

Moved from `decoder/model.go` (the comment above `Model.residentPrefillSeedMRoPE`) on 2026-10-10.

```text
residentPrefillSeedMRoPE is residentPrefillSeed's m-RoPE-aware sibling — used by
GenerateQwenVL's image-reuse fast path (P9a, docs/multimodal.md), where decode past the
(already-resident) image block needs pos+mropeDelta for the rotation, not plain pos.
Deliberately simple (no batched/KV-only-prefill path, unlike residentPrefillSeed): the
reused-suffix case is, by construction, an agent turn's short trailing extension (the 45-51
token deltas resident_reuse.go's own doc comment measures), not a long cold prompt worth the
batched machinery's complexity.
```

## dtAttn

Moved from `decoder/model.go` (the comment above `dtAttn`) on 2026-10-10.

```text
decodeSplit* are decodeTiming's fine split of forward — attention (q/k/v matmuls, the
rope+KV+scores/softmax/AV core, o-proj), MLP (gate+up matmuls, the activation, down), LM head —
accumulated in ns behind the same env gate (one bool check per site when off) and printed with
the DECODE TIMING line. R9's attribution (docs/measurements/cpu-decode-attribution-2026-09-22-linux.md)
found the shares this names — the activation fan-out and the grouped-attention fallback — from
exactly these lines; keeping them means the other box can read its own split without a patch.
```

## Model.reqBackend

Moved from `decoder/model.go` (the comment above `Model.reqBackend`) on 2026-10-10.

```text
reqBackend / effBackend record what the caller ASKED for and what is actually executing.
They differ when NewBackend falls back — `--backend metal` on a build without the metal
submodule runs on CPU. Before 2026-09-06 nothing recorded the difference, so the load
banner printed the REQUESTED name: a cold-user run saw "…using cpu" and
"[backend=metal quant=int4]" on consecutive lines, and the second is the one that gets
screenshotted (docs/measurements/cold-user-2026-09-06.md, finding #3). On that machine it
was the difference between 37.9 and 82.3 tok/s.
prof is this load's phase timing, when the loader instrumented it. nil otherwise.
```

## Model.ExtraResidentBytes

Moved from `decoder/model.go` (the comment above `Model.ExtraResidentBytes`) on 2026-10-10.

```text
ExtraResidentBytes returns Options.ExtraResidentBytes — VRAM a companion allocation will claim
on the SAME device AFTER this model's own residency is built (a --drafter's weights today; a
vision tower is the same class of term, tasks/task-fit-to-hardware.md §2), priced ahead of time so the
elastic terms a backend sizes against live free VRAM (CUDA's capSlots expert cache, its
resolveCtxCapFit context-by-default) leave room for it instead of claiming everything free VRAM
offers and having the later attach fail with no room left — §2's own motivating example,
measured 2026-09-02: a 26B auto-sized to 31 slots/layer, the server came up, then --drafter
attached and NewBlockSpec failed on a 15.9 MB buffer because the cache had already taken the
room. 0 means "nothing else is attaching" (today's behavior, unchanged).
```

## Model.ExtraResidentKVPerPosition

Moved from `decoder/model.go` (the comment above `Model.ExtraResidentKVPerPosition`) on 2026-10-10.

```text
ExtraResidentKVPerPosition returns Options.ExtraResidentKVPerPosition — M-22's own fix
(docs/audit-2026-09-10.md): a companion allocation's ExtraResidentBytes above prices only its
FIXED terms (a drafter's weights); its device K/V, when the companion has any, scales with
whatever resident context THIS model ends up choosing, which ExtraResidentBytes's own caller
(loadDecoder, priced before this model's residency is built at all) cannot know yet. This field
carries the RATE instead (decoder.DrafterKVBytesPerPosition's own doc comment has the shape and
why it is a rate, not a total) — a residency builder that knows its own candidate/final ctx
multiplies by it locally. 0 means "the companion, if any, has no ctx-scaling K/V term" (today's
default, and every model without a drafter attached).
```

## Model.MoECacheSlotsRequest

Moved from `decoder/model.go` (the comment above `Model.MoECacheSlotsRequest`) on 2026-10-10.

```text
MoECacheSlotsRequest returns the requested per-layer expert-slot count, or 0 for "as many as
fit". Only meaningful with MoECacheExperts.

0 means ask for ALL experts and let the builder cap to measured free VRAM — deliberately, and
this is a change from the env-var-only behaviour, where an unset value meant topK. topK is the
WORST setting for the only situation in which this applies: it degenerates to fresh-loading
every routed expert every token (~714 MB/token on the 26B, ~5 tok/s instead of ~17). A user who
asked for expert streaming and said nothing about slots wants it to work, not to be safe; the
safety is already provided by allocSlots, which measures free VRAM and caps-and-logs rather
than OOMing.

Falls back to GOINFER_MOE_CACHE_SLOTS.
```

## Options.EmbedInt4

Moved from `decoder/model.go` (the comment above `Options.EmbedInt4`) on 2026-10-10.

```text
EmbedInt4 relaxes the int8 pin on the token-embedding/LM-head table in int4
mode, storing it at int4 too — halving the single largest resident tensor on a
big-vocab small model. Lossy (~2.3 pts top-1, mostly on rare tokens). Off in a zero
Options, which keeps the bit-exact int8 pin; the CLIs turn it on by default since
2026-09-28 (internal/loadflags, the owner decision in task-never-swap-2026-09.md;
E-D01, audit-metal-2026-09-30.md). GGUF load path only.
```

## Options.DisableFit

Moved from `decoder/model.go` (the comment above `Options.DisableFit`) on 2026-10-10.

```text
DisableFit is tasks/task-fit-to-hardware.md's --fit=off: restores every "fit by default" behavior
to its pre-Phase-2 default exactly. Currently: CUDA's unpinned resident context stays the
flat historical constant instead of asking Plan for more when there's room —
cuda/resident.go's resolveCtxCapFit; and (decoder/fitguard.go's guardFit, read by
internal/modelload's Load, not by Load itself) a dense .gguf that will not fit
resident RAM stays a plain refusal instead of getting an automatic -stream-weights retry.
Does NOT affect a genuine bug fix shipped alongside Phase 2 work (Metal now honoring an
explicit -ctx at all, docs/tasks/task-gpu-paths-2026-09.md's G6 entry) — that is correctness, not
an opinionated default, and stays on either way. An explicitly PINNED request
(ResidentContext, MoECacheSlots, StreamWeights itself, etc.) is never affected by this flag
in either direction: fit-by-default only ever acts on the UNPINNED case.
```

## Options.ExactPrefill

Moved from `decoder/model.go` (the comment above `Options.ExactPrefill`) on 2026-10-10.

```text
ExactPrefill forces bit-exact prompt ingestion on every backend that has a faster,
non-exact default: CUDA's tensor-core batched prefill (GOINFER_CUDA_FAST_PREFILL),
Metal's f16-MMA batched prefill (GOINFER_METAL_FAST_PREFILL), and CPU's f32-attention
fast path (GOINFER_CPU_FAST_ATTENTION) — all three default ON above their own
thresholds. false (the default) leaves whichever env state the process already has
untouched, so a caller managing these knobs itself (serve's own --exact-prefill/
--cpu-exact-prefill/--cpu-fast-attention, which are more granular than this single
bool and set the env vars directly) is not overridden. M-26 (docs/audit-2026-09-10.md):
this is the library-level chokepoint docs/completed/task-prefill-gap.md already
documented as existing; chatapp/gemmaapp's own --exact-prefill flag sets it.
```

## Options.ResidentKVSlotsDefault

Moved from `decoder/model.go` (the comment above `Options.ResidentKVSlotsDefault`) on 2026-10-10.

```text
ResidentKVSlotsDefault says ResidentKVSlots is the caller's default, not a count the operator chose (serve sets it
when -kv-sessions was not given). A backend may then lower it to its own default: Metal keeps 2 slots (E-P09,
docs/audit-metal-2026-09-30.md), since every slot's KV is resident from the first token on unified memory, where
the extra slots cost about 224 MB on the 1.5B and 470 MB on the 7B. CUDA and WebGPU keep the count asked.
```

## Options.ResidentPrefillChunk

Moved from `decoder/model.go` (the comment above `Options.ResidentPrefillChunk`) on 2026-10-10.

```text
ResidentPrefillChunk, under MC3 (EnableResidentConcurrency), prefills a long prompt suffix in chunks of this many
tokens while other generations are decoding, one decode step between chunks, instead of in one pass that stalls
them all for the whole prompt (docs/tasks/task-concurrency-2026-09.md, chunked prefill). 0 = off: whole
prefill, the behaviour before it. Sound only where the resident's batched prefill is chunk-invariant (Metal's is:
TestMC5_prefillChunkInvariance). serve defaults it to 512, the graded value (the decoders' longest stall 0.23x, wall
1.045x, replies identical; a 256 candidate missed its wall gate — docs/measurements/chunked-prefill-2026-09-27.md).
```

## Options.ActQuantGroup

Moved from `decoder/model.go` (the comment above `Options.ActQuantGroup`) on 2026-10-10.

```text
ActQuantGroup selects per-group ACTIVATION quantization for the int8-activation projections
(int4 = W4A8, int8int8 = W8A8, int4mix): 0 (the default) scales each activation vector by one
max/127, 32 gives every 32 inputs their own scale. A family with massive activation outliers
(Phi-3: max/rms ~80-90) loses nearly the whole vector under one scale; per-32 keeps an outlier's
damage inside its group (docs/tasks/task-actquant-pergroup-2026-09.md). Per model: two models in
one process may differ. Honoured on the CPU and by CUDA residency; other resident backends
decline to the CPU path when it is set.
```

## Options.LoadAbort

Moved from `decoder/model.go` (the comment above `Options.LoadAbort`) on 2026-10-10.

```text
LoadAbort, if non-nil, is checked BETWEEN LAYERS during a direct (non-.giw) GGUF weight
build — S3 (docs/tasks/task-never-swap-2026-09.md): the swap tripwire's LOAD-TIME
consumer. Closing it aborts the load with an error satisfying errors.Is(err,
ErrLoadAborted); Load has no view of WHY it closed (it does not own or import
decoder.SwapWatch — the caller does), so a caller arming a watch for this is expected to
wrap the returned error with its own reason/pricing detail once it comes back. A nil
channel (the zero value — every existing caller) blocks forever in a select, so this is a
genuine no-op, not a special case every caller needs to opt out of.

SCOPE, STATED RATHER THAN HIDDEN: only the GGUF direct-build path
(loadGGUFWeights/buildWeightsFromGGUF's resident, non-streaming branch) checks this today
— the path S0's own mechanism table names as the dangerous one ("fresh heap copies... the
whole resident weight set") and the one the historical gpt-oss-20b incident took. The
safetensors direct-build path (decoder/weights.go, 5 separate buildXWeights entry points)
and StreamTranscodeGGUF's own transcode path (which already has its own M-21
ctx-cancellation, checked at a different granularity) do NOT check this yet — see the task
doc's own S3 status note for why this pass stopped here rather than threading it further.
```

## modelFromOptions

Moved from `decoder/model.go` (the comment above `modelFromOptions`) on 2026-10-10.

```text
modelFromOptions is a Model over w carrying every per-model field Options sets — the one place the
constructors (Load's .giw and direct paths, LoadGGUFBytes, NewModelWithOptions) read them. They
used to be three struct literals, and they had drifted: LoadGGUFBytes never read the drafter's
ExtraResident* reservation, and NewModel read no option but the backend, so a baked-in chat model
silently ignored --kv, --fit and --exact-prefill. Callers set what is specific to their path (the
requested quant, the resolved EOS ids, the file mapping), then apply backend names, knobs,
streaming and residency in the order their path needs.
```

## Load.exactprefill

Moved from `decoder/model.go` (the comment above `Load.exactprefill`) on 2026-10-10.

```text
Options.ExactPrefill is recorded on the Model (exactPrefill, set in each constructor below
BEFORE withResidency, because CUDA reads it while building its resident) and consulted by
each backend's fast-prefill switch alongside its env var. It used to be applied by
os.Setenv here (M-26), which is process-global and never undone: every model loaded later
in the same process inherited exact prefill whether it asked for it or not. Serve no longer
sets the env vars either: its flags travel as this field and Options.Knobs (phase 5,
docs/tasks/task-env-config-2026-09.md).
Whether the CALLER chose the resident context, read before either fit guard below can auto-pin one into
opts.ResidentContext (R13) — that pin is a one-slot ceiling, not a choice (Model.ResidentContextPinned). Stamped
on the Model after modelFromOptions and before withResidency, where a backend reads it.
```

## Load.giwcrc

Moved from `decoder/model.go` (the comment above `Load.giwcrc`) on 2026-10-10.

```text
The trailing CRC reads every byte of the mapping, which is the whole load time of a large
streamed .giw (27-28 min for a 22 GB file over a slow link). It is a property of the file,
so check it once per (size, mtime) and skip it on later loads — see giwverify.go. The stat
is taken BEFORE the load so a file replaced mid-load cannot inherit the marker.
```

## Load.giwfit

Moved from `decoder/model.go` (the comment above `Load.giwfit`) on 2026-10-10.

```text
S4 (task-never-swap-2026-09.md), item 1: a .giw's weights are file-backed (no
fitCheckFor call here at all, by design — see fitguard.go's own srcFileBytes doc
comment), but its KV cache and prefill scratch ARE real anonymous allocations this
path never priced before. guardGIWFit refuses or auto-pins exactly like the .gguf
path's guardFit does, against a flat margin over live available memory rather than
fitMemFraction's 70%-of-available (sized for a load that commits its weights too).
Not everything else a .giw load allocates is file-backed: the paged MoE scale cache and
the rest of the heap (about 1.95 GB on the 26B MoE) are anonymous, and nothing here
prices them (C-D01 and C-P01 in audit-metal-2026-09-30.md).
```

## Load.moeworkingset

Moved from `decoder/model.go` (the comment above `Load.moeworkingset`) on 2026-10-10.

```text
S4 item 5: the arithmetic the M35 run needed before it started — predict
the working-set rate and require an explicit acknowledgement below the
registered floor, rather than let a 2h10/zero-completions run discover it.
```

## Load.eos

Moved from `decoder/model.go` (the comment above `Load.eos`) on 2026-10-10.

```text
M-04 (docs/audit-2026-09-10.md): write the RESOLVED EOS set — config.json plus any extra
ids generation_config.json adds — back into w.Cfg.EOSTokenID, not just onto this Model's
own eosIDs field. w.Cfg is what a .giw bundle serializes (internal/prequant, via
SerializeWeightsToForTarget), and a .giw's own Load branch reads eosIDs straight from
w.Cfg.EOSIDs() with no directory to re-resolve generation_config.json from — so without
this, a checkpoint whose stop ids live only in generation_config.json (Qwen3:
<|endoftext|> 151643 beside config.json's <|im_end|> 151645) loses the extra id the moment
it round-trips through `cmd/prequant`, and a completion that emits it runs to max_tokens
instead of stopping. Cfg.EOSTokenID has no other reader that needs the UNRESOLVED
config.json-only value (EOSIDs() is its only consumer anywhere in the tree), so
overwriting it here is safe.
resolveEOSIDs looks for generation_config.json via os.DirFS(eosDir) — a real DIRECTORY.
For a .gguf load, dir is the FILE path, so os.DirFS(dir) can never open anything inside it
(the fallback decoder/gguf.go:ggufEOS's own comment claims); generation_config.json for a
GGUF conversion lives beside the file, in its parent directory, same as M-04 found.
```

## Model.Config

Moved from `decoder/model.go` (the comment above `Model.Config`) on 2026-10-10.

```text
Config returns a SNAPSHOT of the loaded architecture config — a copy, not the live struct
(audit M-23). The forward pass reads a derived, unexported *Architecture plus precomputed RoPE
tables built from this config AT LOAD; the config itself is not re-read per token. Returning
the live &m.w.Cfg let a caller do m.Config().NumLayers = N and silently desync those caches
from the config — wrong logits on the path the project calls its stable contract. The copy
makes such a write land on a throwaway value instead. (Scalar fields — the realistic footgun —
are fully isolated; the copy is shallow, so its slice/pointer fields still alias the model's
and must be treated as read-only. There is no supported way to reconfigure a loaded model.)
```

## Model.NewCache

Moved from `decoder/model.go` (the comment above `Model.NewCache`) on 2026-10-10.

```text
NewCache allocates a KV cache sized for this model. capHint pre-sizes for
a known max length (0 = grow on demand).

R13: prefillEnters counts every call, so a test can OBSERVE that a request AdmitPrefillMemory
refused never reached here — the same discipline weightAllocs (fitguard.go) applies to
loadWeights, for the same reason: a check placed one line too late produces the identical
error text and the identical swap storm.
```

## Model.NewCache.mla

Moved from `decoder/model.go` (the comment above `Model.NewCache.mla`) on 2026-10-10.

```text
P-02 (audit-2026-09-10): an MLA family (DeepSeek-V2/V3, Kimi K2/V3) never writes
c.keys[l]/c.vals[l] on ANY layer — the per-layer compressed latent (c.mlaLatent, set up
below) is the whole store — but NewKVCache reserved full capHint*kvDim capacity for them
on every layer regardless: ~0.66 MB/position dead weight at DeepSeek-V2-Lite's geometry
(~3.4 GB for a 4k+1k request), 6-12 MB/position for Kimi K2/V3. kvCapHint=0 makes the
reservation itself (not just its later use) match what actually happens: an empty slice
that ordinary append would still grow correctly if anything ever DID write to it (nothing
does), so this is a pure allocation elimination, not a new correctness constraint.
```

## Model.NewCache.recurrent

Moved from `decoder/model.go` (the comment above `Model.NewCache.recurrent`) on 2026-10-10.

```text
P-02's second half (docs/audit-2026-09-10.md, "recurrent families reserve the same dead
capacity on their non-attention layers"): a linear/mamba/conv mixer layer (or one of
Nemotron's own mlp/moe block kinds) reserves the SAME dead capHint*kvDim capacity as an
ordinary attention layer despite never writing c.keys[l]/c.vals[l] at all — its own
recurrent state (c.delta/c.mamba/c.conv/c.kda, set up below) is the whole store for that
layer. Found while verifying this fix: Nemotron's mamba/mlp/moe layers weren't caught by
any of isLinearLayer/isMambaLayer/isConvLayer at all (its mixer identity is per-layer
runtime data, not a registry-time closure) — hasNoAttentionKVAt (decoder/arch.go) is the
single place that now knows all of these cases, shared with kvDimAt's own pricing use.
```

## Model.NewCache.kvi8

Moved from `decoder/model.go` (the comment above `Model.NewCache.kvi8`) on 2026-10-10.

```text
int8 KV storage (opt-in, Options.KVQuant=="i8"): the uniform dense families
only — MoE routes attention through the acc64 kernel for bit-stable expert
routing (quantized KV would reopen that), and gemma4/qwen3_5_moe have their
own forward. Must precede enableRings so local layers inherit the mode.

Every family forward that sizes its scores buffer from the f32 key store
(`len(cache.Keys(layer))` — forward_granite/qwen35/nemotron/llama4/lfm2.go) must be
excluded: with int8 on, that store is empty, the buffer has length 0, and attendQuery
hands it to attendQueryI8, which indexes past it on the first decode step. LFM2 was
missing and panicked (TestLFM2_kvQuantI8_generates); llama4 was excluded only through
a.MoE, so it is named too.
```

## Architecture.QKNormWhole

Moved from `decoder/arch.go` (the comment above `QKNormWhole`) on 2026-10-10.

```text
QKNormWhole (Olmo 3/Olmo Hybrid): when QKNorm is also set, normalize the WHOLE projected
q/k vector as one RMSNorm (num_heads*head_dim elements, one statistic) instead of per-head
— verified against the real modeling_olmo3.py, not the standard per-head convention every
other QK-norm family uses. The underlying rmsNorm(x, weight, rows, dim, ...) already
supports this: per-head calls it with (rows=nHeads, dim=headDim); whole-vector calls it
with (rows=1, dim=nHeads*headDim) — same function, different split.
```

## Architecture.lfm2

Moved from `decoder/arch.go` (the comment above `lfm2` in `Architecture`) on 2026-10-10.

```text
lfm2, when non-nil, marks an LFM2/LFM2.5 hybrid: every layer has a SwiGLU FFN,
and its mixer is either a gated short convolution (layerIsConv true, 22 of 30 on
LFM2.5-2.6B) or GQA softmax attention with per-head RMSNorm on Q and K.

The conv layers carry a rolling per-channel window instead of a KV cache, which
```

## qwen35Params.FusedDeltaNetProj

Moved from `decoder/arch.go` (the comment above `FusedDeltaNetProj`) on 2026-10-10.

```text
FusedDeltaNetProj: qwen3_5_moe's checkpoint stores in_proj_qkv/in_proj_z/
in_proj_b/in_proj_a as four separate tensors; qwen3_next's checkpoint fuses
them into in_proj_qkvz/in_proj_ba instead (same math, different packing —
verified against modular_qwen3_next.py's Qwen3NextGatedDeltaNet.torch_forward).
loadQwen35Attn splits the fused tensors into the same four deltaNetWeights
fields so the rest of the pipeline (forward, gguf, serialize) is untouched.
```

## qwen35Params.SeparateQKVProj

Moved from `decoder/arch.go` (the comment above `SeparateQKVProj`) on 2026-10-10.

```text
SeparateQKVProj (Olmo Hybrid): the checkpoint stores q_proj/k_proj/v_proj as
THREE fully independent tensors — more unfused than qwen3_5_moe's own
in_proj_qkv (which is already pre-concatenated on disk into one [convDim,
hidden] tensor). loadQwen35Attn concatenates them at load time into the same
internal inProjQKV layout, so gatedDeltaNetStep and every downstream consumer
stay untouched. Verified against the real modeling_olmo_hybrid.py
(OlmoHybridGatedDeltaNet.__init__: separate self.q_proj/k_proj/v_proj
nn.Linear modules, vs. qwen3.5's single mixed_qkv projection).
```

## MoEConfig.IntermediateDim

Moved from `decoder/arch.go` (the comment above `IntermediateDim` in `MoEConfig`) on 2026-10-10.

```text
IntermediateDim is the per-expert FFN width. Mixtral's experts use the
model's intermediate_size; Mellum gives them a narrower moe_intermediate_size
(896 vs the vestigial 7168), so the expert width is tracked here rather than
read from arch.IntermediateDim.
```

## Architecture.NormPostOnly

Moved from `decoder/arch.go` (the comment above `NormPostOnly`) on 2026-10-10.

```text
NormPostOnly: NO pre-norm at all — attention/MLP read the RAW residual
stream directly — and the sublayer's OUTPUT is normalized before the
residual add (residual = x + post_attn_norm(attn(x)); same for MLP).
Olmo 3, verified against the real modeling_olmo3.py: no input_layernorm
exists at all, only post_attention_layernorm / post_feedforward_layernorm,
applied to the SUBLAYER OUTPUT before the add. Genuinely different from
Sandwich4, which normalizes the input AND (separately) the output; here
there is no input norm to skip past. Olmo Hybrid's full-attention layers
use this SAME scheme, but its DeltaNet layers use NormPre2 instead — see
NormPlacementLinear, not a second value of this enum.
```

## NewBackend.entrypoint

Moved from `decoder/backend.go` (the comment in `NewBackend`) on 2026-10-10.

```text
Since v0.10.0 (audit M-19) the backend lives in a submodule ENTRYPOINT, not a build tag on
the root binary — `-tags gpu|cuda|metal` on cmd/serve does nothing. Point at the real one.
```

## Config.GatedAttentionProjGranularity

Moved from `decoder/config.go` (the comment above `GatedAttentionProjGranularity`) on 2026-10-10.

```text
GatedAttentionProjGranularity ("head_wise" | "element_wise" | absent): MLA's optional
per-head or per-element output gate (self.g_proj, sigmoid-activated, applied to the
attention context BEFORE the output projection — the same STRUCTURE Laguna's own
FeatAttnOutputGate already ships, but sigmoid where Laguna's is softplus, a real difference
verified against source, not assumed identical).
```

## Config.HeadwiseAttnOutputGate

Moved from `decoder/config.go` (the comment above `HeadwiseAttnOutputGate`) on 2026-10-10.

```text
HeadwiseAttnOutputGate/GateAttnActMode (Spark-X2.5): its own, differently-shaped spelling of
the same "gate before out_proj" idea — a plain bool (always per-head when on; no
per-element variant, unlike MLA's granularity string above) plus an explicit activation-mode
string ("sigmoid" is the only released value; "silu" is a legal-but-unreleased option per
the real modeling_spark.py, which goinfer does not implement — validateResolved rejects it
rather than silently mis-running).
```

## Config.ShortConvKernelSize

Moved from `decoder/config.go` (the comment above `ShortConvKernelSize`) on 2026-10-10.

```text
KDA's own wrapper geometry: a depthwise short causal conv (kernel ShortConvKernelSize,
SiLU-activated, THREE separate q/k/v convs — modeling_bailing_moe_v3.py's
BailingMoeV3KimiDeltaAttention has independent self.q_conv1d/k_conv1d/v_conv1d modules, not
one combined conv like Gated DeltaNet's), NoKDALora selecting a single f_proj/g_proj linear
per gate (true on the release) vs a LoRA'd a/b-split pair, and the safe (lower-bounded)
decay gate Ling-3.0-tiny selects.
```

## Config.LinearAllowNegEigval

Moved from `decoder/config.go` (the comment above `LinearAllowNegEigval`) on 2026-10-10.

```text
LinearAllowNegEigval (Olmo Hybrid only): doubles the write-gate beta from
sigmoid's [0,1) range to [0,2), widening the delta-rule's eigenvalue range to
include negative — verified against the real modeling_olmo_hybrid.py
(OlmoHybridGatedDeltaNet.forward: `if self.allow_neg_eigval: beta = beta * 2.0`,
gated on this exact config field, default true). Absent (false) for every
other DeltaNet family.
```

## Config.FullAttentionInterval

Moved from `decoder/config.go` (the comment above `FullAttentionInterval`) on 2026-10-10.

```text
FullAttentionInterval (Qwen3-Next only — qwen3_5_moe ships the per-layer
pattern explicitly via LayerTypes instead). The real released config has
NO layer_types field at all; the pattern is COMPUTED: layer i (0-indexed)
is full_attention when (i+1)%FullAttentionInterval==0, else
linear_attention — verified against transformers'
configuration_qwen3_next.py __post_init__ directly, not assumed from the
qwen3_5_moe precedent. normalizeQwen3NextLayerTypes turns this into the
same LayerTypes list every other consumer already reads, so it's the
ONLY place that needs to know this family computes rather than states.
```

## Config.PadTokenID

Moved from `decoder/config.go` (the comment above `PadTokenID`) on 2026-10-10.

```text
PadTokenID (gemma4, P7): the real multimodal forward substitutes THIS id's
embedding/per-layer-embedding at every image/video/audio position before
computing PLE's token-identity term — not the placeholder token's own id,
and not a skipped/zeroed term (verified against modeling_gemma4.py's real
multimodal forward, not assumed — see docs/multimodal.md's P7 entry).
Lives under text_config in a real checkpoint; loadConfig's text_config
merge picks it up via this tag with no special-casing needed.
```

## Config.NoRopeLayerInterval

Moved from `decoder/config.go` (the comment above `NoRopeLayerInterval`) on 2026-10-10.

```text
NoRopeLayerInterval (SmolLM3): HF's generation formula for NoRopeLayers (above) when the
checkpoint's config.json omits the explicit per-layer list: NoPE exactly when
(layer_idx+1) % interval == 0. SmolLM3 reuses the SAME NoRopeLayers field and the SAME
"1 = has rope, 0 = NoPE" convention Llama 4 already established above — verified against
the real modeling_smollm3.py (`self.use_rope = config.no_rope_layers[layer_idx]`), not
assumed to match from the shared field/JSON-key name alone.
```

## Config.validateQwen3Moe

Moved from `decoder/config.go` (the comment above `validateQwen3Moe`) on 2026-10-10.

```text
validateQwen3Moe pins the Qwen3-MoE assumptions (Qwen3-30B-A3B /
Qwen3-Coder-30B-A3B-Instruct, both model_type "qwen3_moe"): the qwen3 dense
constraints (QK-norm, no q/k/v bias, single-base RoPE) plus a valid sparse
MoE on every layer (num_experts / num_experts_per_tok / moe_intermediate_size).
Unlike qwen2_moe there is NO shared expert — confirmed against the real
released config.json, which carries no shared_expert_intermediate_size field
at all, and against a real GGUF's tensor list, which has no ffn_*_shexp.
```

## Config.validateBailingHybrid

Moved from `decoder/config.go` (the comment above `validateBailingHybrid`) on 2026-10-10.

```text
validateBailingHybrid pins Bailing Hybrid's (Ling 3.0) assumptions: a valid MLA geometry
(same checks as validateDeepseek), a valid KDA geometry (conv kernel, per-head dim), and a
routed+shared MoE using THIS family's own field spellings (num_experts/num_shared_experts, not
DeepSeek's n_routed_experts/n_shared_experts — verified against the real config.json).
```

## Config.validateLFM2.eps

Moved from `decoder/config.go` (the comment in `validateLFM2`) on 2026-10-10.

```text
Every other RMSNorm family validates its eps >0 right here, and this family
did not until an eps of 0 shipped a silently-wrong forward. LFM2 spells the
key "norm_eps"; a checkpoint that omits it (or a parse that looks for
rms_norm_eps) must fail loudly rather than normalise by rsqrt(variance).
```

## Config.validateGraniteDense

Moved from `decoder/config.go` (the comment above `validateGraniteDense`) on 2026-10-10.

```text
validateGraniteDense pins the dense Granite 4.2 assumptions (ibm-granite/granite-4.2-{3b,8b,30b},
model_type "granite"): a plain llama skeleton (GQA, SwiGLU, single-base RoPE, no bias, no
QK-norm) plus Granite's scalar multipliers. residual_multiplier is the one multiplier the
generic forward path cannot apply (granitemoehybrid's own-forward does, via graniteParams —
see graniteDenseArchitecture's comment); every released 4.2 size ships it at 1.0, confirmed
directly, so anything else is rejected loudly rather than silently dropped.
```

## Config.normalizeQwen3NextLayerTypes.tail

Moved from `decoder/config.go` (the comment above `normalizeQwen3NextLayerTypes` (second paragraph)) on 2026-10-10.

```text
0-indexed i, so layer 3 (not layer 4) is the first full-attention layer at
the default interval=4. A no-op if LayerTypes is already populated (a
transformers-instantiated config carries it directly, same asymmetry
normalizeNemotronBlocks handles for that family).
```

## Config.normalizeBailingLayerTypes

Moved from `decoder/config.go` (the comment above `normalizeBailingLayerTypes`) on 2026-10-10.

```text
normalizeBailingLayerTypes synthesizes LayerTypes from LayerGroupSize for Bailing Hybrid
(Ling 3.0), whose real released config.json has NO layer_types field at all. Formula verified
against the real modeling_bailing_moe_v3.py's BailingMoeV3DecoderLayer.__init__:
```

## Config.normalizeBailingLayerTypes.tail

Moved from `decoder/config.go` (the comment above `normalizeBailingLayerTypes` (second paragraph)) on 2026-10-10.

```text
0-indexed i. The second clause is a tail-cleanup for a NumLayers that isn't a clean multiple of
LayerGroupSize (irrelevant for Ling-3.0-tiny's exact 24/4, but replicated anyway rather than
dropped, since the release-verified formula is the authority, not a simplification of it).
Synthesizes "full_attention" (this tree's own spelling) for HF's "attention", so the existing
IsGlobalLayer/IsLinearLayer helpers read it with no new predicate. A no-op if LayerTypes is
already populated.
```

## Config.normalizeNemotronBlocks.moe

Moved from `decoder/config.go` (the comment in `normalizeNemotronBlocks`) on 2026-10-10.

```text
Nemotron 3 Nano's MoE FFN layer (sparse routed + shared expert, replacing
the plain "-" dense-MLP block at this position). Verified against the real
checkpoint's hybrid_override_pattern, not assumed from the "M"/"*"/"-"
alphabet documented for plain Nemotron-H.
```

## Config.validateOlmoHybrid

Moved from `decoder/config.go` (the comment above `validateOlmoHybrid`) on 2026-10-10.

```text
validateOlmoHybrid checks Olmo Hybrid's (model_type olmo_hybrid) core dims: the same
Gated-DeltaNet geometry checks as qwen3_5, no MoE (it is a dense hybrid, verified
against the real released config — no num_experts field at all), and MHA rather than
GQA on every released size fetched (num_attention_heads == num_key_value_heads).
```

## Config.validateQwen3Next

Moved from `decoder/config.go` (the comment above `validateQwen3Next`) on 2026-10-10.

```text
validateQwen3Next pins the same shape assumptions as validateQwen35 — this
family shares every other dimension field-for-field with qwen3_5_moe,
verified against the real config, not assumed — EXCEPT the RoPE check: the
real released config never carries a rope_parameters object at all (flat
rope_theta instead), so requiring it unconditionally (validateQwen35's own
check) would reject every real Qwen3-Next checkpoint. Accepts either shape,
matching qwen3NextArchitecture's own dual-path RoPE resolution.
```

## loadConfig.textconfig

Moved from `decoder/config.go` (the comment in `loadConfig`) on 2026-10-10.

```text
Composite/VL checkpoints (e.g. Qwen3.6-35B's qwen3_5_moe, shipped as a
*ForConditionalGeneration with a vision tower) nest the TEXT decoder's dims
under "text_config" rather than at the top level. Flatten it: decode
text_config into c first so its dims (hidden_size, num_hidden_layers,
num_experts, rope_parameters, layer_types, …) populate the otherwise-zero
fields, then re-apply the top-level keys so anything authoritative there
(model_type, tied-head signals) wins. json.Unmarshal only writes keys that
are present, so a flat config.json is unaffected (text_config absent).
```

## Config.lagunaFirstKDense

Moved from `decoder/config.go` (the comment above `lagunaFirstKDense`) on 2026-10-10.

```text
lagunaFirstKDense turns Laguna's dense-layer declaration into goinfer's
FirstKDense prefix count.

TWO SPELLINGS, and mlp_layer_types is the reliable one: all three released
configs carry mlp_layer_types (["dense","sparse",…]), but XS.2 DROPS
mlp_only_layers entirely. Reading only mlp_only_layers yields FirstKDense=0 on
XS.2, which would make the loader treat its dense layer 0 as MoE and demand
expert tensors that do not exist. So mlp_layer_types wins when present.

Either way the dense layers must form a CONTIGUOUS PREFIX — that is what
```

## Model.Generate.newCache

Moved from `decoder/model.go` (the comment in `Model.Generate`) on 2026-10-10.

```text
P-01 (audit-2026-09-10): the host KV cache (numLayers*2*(prompt+maxTokens)*kvDim*4B — 5.2 GB
of heap CAPACITY for a 7B at 16k+4k) is not allocated here at all: generateInto only calls
newCache when it actually reaches the CPU path (never resident, or lost the resBusy race),
so a resident-and-won call never pays for host KV capacity it never touches.
```

## Model.generateInto.topk

Moved from `decoder/model.go` (the comment in `generateInto`) on 2026-10-10.

```text
Device top-K fast path (R7, sampler_topk.go): a FILTERED sampler (top_k / top_p / min_p at
temperature > 0) needs only the K best logits, so the resident reduces the row on-device and reads
back ~2 KB instead of the whole vocab-wide row, and the host filters K candidates instead of V.
Excluded: anything that needs or rewrites the full row (bias, penalties, logprobs, a
LogitProcessor), the greedy and optimistic-forward paths, and backends whose logits are
transformed on the host after readback. GOINFER_NO_TOPK_FASTPATH forces the full-row path
```

## Model.generateInto.sample

Moved from `decoder/model.go` (the comment in `generateInto`) on 2026-10-10.

```text
Device temperature-only sampling (R7b, sampler_gumbel.go): the resident draws the next token by
Gumbel-max on-device and returns just the id, reusing the greedy fast path's fastNext mechanism. Same
exclusions as the top-K path; GOINFER_NO_SAMPLE_FASTPATH forces the host draw (A/B check, escape hatch).
```

## Model.generateInto.batchGreedy

Moved from `decoder/model.go` (the comment in `generateInto`) on 2026-10-10.

```text
E-P08 (docs/audit-metal-2026-09-30.md): a greedy generation's batched token asks the step for its argmax id (a
Greedy draw) where the resident offers that, instead of its whole logits row; the id is the argmax the sampler
would take from that row. Rows run alone keep residentCall's own path.
```

## Model.generateInto.mc3Logits

Moved from `decoder/model.go` (the comment in `generateInto`) on 2026-10-10.

```text
mc3Logits holds an MC3 solo token's logits (E-C01, docs/audit-metal-2026-09-30.md): a resident's Forward returns its
one host buffer, reused by every call, and this generation reads the logits after leaving the resident — in its
LogitProcessor and its sampler — while another generation's solo token may already be rewriting that buffer.
```

## Model.generateInto.clamp

Moved from `decoder/model.go` (the comment in `generateInto`) on 2026-10-10.

```text
Clamp the decode length to the resident KV cap up front (C3/M20). A Forward past
the cap is refused mid-generation (the silent-corruption guard), but a resident
backend that exposes its cap lets us stop cleanly AT it instead of erroring after
N tokens. gpuPos is the next decode position; the last valid one is ctxCap-1, and
the prefill loop already guarantees gpuPos <= ctxCap.
```

## Model.generateInto.embScratch

Moved from `decoder/model.go` (the comment in `generateInto`) on 2026-10-10.

```text
embScratch is reused across iterations (P-08, audit-2026-09-10): each token's embedding is
consumed synchronously by Forward/ForwardArgmax below before the next one is requested, so
one buffer for the whole loop replaces a fresh [hidden]float32 allocation every token.
```

## Model.generateInto.cache

Moved from `decoder/model.go` (the comment in `generateInto`) on 2026-10-10.

```text
P-01 (audit-2026-09-10): allocate the host KV cache HERE, not before the CAS above —
covers both "never going to be resident" (useGPU started false) and "lost the race for
the shared resident KV" (useGPU flipped false in the CAS block just above), the only two
ways this function reaches the CPU path below. A resident-and-won call never allocates one
at all; newCache is nil (never called) whenever cache is already non-nil, which is every
Session.Generate call — its own cache always exists already, this branch is simply never
taken for it.
```

## Model.generateInto.ctxdone

Moved from `decoder/model.go` (the comment in `generateInto`) on 2026-10-10.

```text
P-09 (audit-2026-09-10, sibling of R-02): a request cancelled while waiting for the
model lock (tryEnter blocks with no context) reaches here with ctx ALREADY done —
residentForgetIDs below would discard a warm cache for a prefill that residentPrefillSeed
is about to refuse anyway, cold-prefilling the whole conversation on the next live turn
for no benefit. Checked before the forget, not left to residentPrefillSeed's own ctx
check, which runs too late to matter.
```

## Model.generateInto.claim

Moved from `decoder/model.go` (the comment in `generateInto`) on 2026-10-10.

```text
The resident path drives the model's ONE shared positional KV; two concurrent
generations would interleave writes at overlapping positions and corrupt it.
Claim it non-blockingly — a loser falls back to the staged CPU path, which uses
this call's own cache, so both still complete correctly (M9). The doc's
"distinct sequences can run concurrently" holds; only resident speed is lost.

This is ALSO what makes binding an adapter here race-free even though N adapters of
one base share this single resident runner (internal/serveapp/main.go): only the
resBusy winner's SetAdapter/Forward/SetAdapter(nil) sequence ever runs at a time, so
two adapter sessions (or an adapter session and a base-model session) can never
```

## Model.mc3Prefill.seed

Moved from `decoder/model.go` (the comment in `mc3Prefill`) on 2026-10-10.

```text
The seed may be the resident's shared host buffer (a Forward, or a prefill path that returns it), and the
generation reads it after this exclusive section, when another generation's token may be rewriting it:
copy it while the resident is still ours (E-C01, docs/audit-metal-2026-09-30.md; generateInto's mc3Logits).
```

## Load.weightcache

Moved from `decoder/model.go` (the comment in `Load`) on 2026-10-10.

```text
S4 item 2 (task-never-swap-2026-09.md): resolve an "auto" (0) weight-cache request
from this platform's own live probe BEFORE either pager sees it, so darwin gets a
real figure instead of aikit's Linux-only /proc probe + fixed 8 GB darwin fallback.
```

## Load.minherit

Moved from `decoder/model.go` (the comment in `Load`) on 2026-10-10.

```text
Keep the weights out of any fork()ed child (forkinherit_darwin.go): once a GPU backend has
wired a page of this private mapping, a fork would otherwise copy ALL of it eagerly — the
M26 collapse. Non-fatal: failing it only restores the old behaviour.
```

## Load.nilbackend

Moved from `decoder/model.go` (the comment in `Load`) on 2026-10-10.

```text
A nil backend means the name was genuinely unknown (not a registered/fallback backend) —
abort rather than proceed and panic at the first matmul (M14). A non-nil be with a
non-nil beErr is the CPU-fallback note (webgpu/cuda/metal not built in); keep the (cpu)
backend and surface the note rather than abort.
```

## Load.fitguard

Moved from `decoder/model.go` (the comment in `Load`) on 2026-10-10.

```text
The fit guard runs HERE — after the quant is resolved (it moves the weight term more than
anything else) and before loadWeights allocates a byte. Refusing after the allocation would
be refusing after the swap storm, which is the failure it exists to prevent.

R13: the guard may return a SMALLER context to pin than what was requested (0 = unrequested,
the common case) — applied to opts here, before opts.ResidentContext is read again below
(resCtxReq) and by the banner, so a cold user gets a working server with a visible, honest
```

## Load.giwint4

Moved from `decoder/model.go` (the comment in `Load`) on 2026-10-10.

```text
L2 (docs/tasks/task-int4-layout-2026-09.md): a .giw bakes its int4 representation
in at WRITE time (giwWriter.target), so unlike a GGUF/safetensors load —
where wantsCanonicalInt4 decides needCanonical from THIS opts.Backend before
a byte is quantized — the reader has to check the file's promise against
what THIS Load actually needs. giwReader.weightMat already refuses a kind-5
tensor this core can't run (wrong arch/shape); this catches the other named
mismatch — a kind-5 file loaded under a backend that needs canonical (e.g.
Backend:"metal") — which withResidency's own decline does NOT fail loudly
for (it logs and falls back to CPU/staged, which is non-fatal by design but
not the "fails at load" contract L2 requires for a wrong-representation file).
```

## Model.adapters

Moved from `decoder/model.go` (the comment above `adapters` in `Model`) on 2026-10-10.

```text
adapters holds compute-time LoRA adapters loaded against this base (#7), behind a POINTER so
*Model stays value-copyable — the kvi8 test seam does `mm := *m` to flip kvI8, and a sync.Mutex
field would make `go vet` reject the copy (and resBusy is a raw int32 for the same reason). nil
until the first LoadAdapter. The mutex inside guards concurrent LoadAdapter vs UseAdapter/
HasAdapter (audit C-29).
```

## adapterRegistry.retired

Moved from `decoder/model.go` (the comment above `retired`) on 2026-10-10.

```text
retired holds runtimes displaced by a re-registration of the same name. A live Session may
still hold the old *loraRuntime in its cache.lora and read its mmap'd deltas mid-generation, so
it must NOT be munmap'd on re-register (that SIGSEGVs the reader, audit C-29). Released only at
Model.Close — a small, bounded leak (one entry per re-registration of a live name) traded for
```

## Model.resDrafterSynced

Moved from `decoder/model.go` (the comment above `resDrafterSynced`) on 2026-10-10.

```text
resDrafterSynced identifies which *BlockSpec's own drafter context is currently in sync
with resIDs (P-05, audit-2026-09-10) — nil means no drafter context is trustworthy for
reuse. resIDs alone is not enough: a plain Generate or n-gram-speculative turn can commit
resIDs without ever touching a block drafter's own context, so the TOKEN prefix can match
while the DRAFTER's state does not reflect it at all. Cleared by residentForgetIDs (every
resident write invalidates it) and set only by BlockSpec.generate's own fully-completed
exit — the one path that keeps resIDs and the drafter's context advancing together.
```

## Model.ResidentContextPinned

Moved from `decoder/model.go` (the comment above `ResidentContextPinned`) on 2026-10-10.

```text
ResidentContextPinned reports whether the caller chose the resident context (Options.ResidentContext > 0), as
opposed to leaving it to the backend. A context the load-time fit guard auto-pinned for an unrequested load (R13)
is NOT pinned: it is a ceiling that fits one KV slot, and ResidentContextRequest still reports it as the upper
bound. A backend that trades context for KV slots (MC1 "slots before context") shrinks only an unpinned context,
never an explicit -ctx.
```

## Model.FitDisabled

Moved from `decoder/model.go` (the comment above `FitDisabled`) on 2026-10-10.

```text
FitDisabled is tasks/task-fit-to-hardware.md's --fit=off (Options.DisableFit), true when either the
Options field or the pre-existing GOINFER_NO_FIT_DEFAULT env var (cuda/resident.go's original,
narrower escape hatch — kept working rather than orphaned) says to restore every "fit by
default" behavior to its pre-Phase-2 exact default. Checked by backend packages that implement
a fit-by-default policy (cuda.resolveCtxCapFit today; a future Metal/CPU equivalent would read
the same accessor) — decoder itself has no fit-by-default logic of its own to gate.
```

## Options.KVPrecision

Moved from `decoder/model.go` (the comment above `KVPrecision`) on 2026-10-10.

```text
KVPrecision selects the GPU residency KV cache precision: "" / "f32"
(default, bit-exact, 16k context cap), "f16" (lossy, 2× context to 32k), or
"i8" (lossy, 4× vs f32 → ~64k context). Ignored off the residency path. See
task-gpu-f16-kv.md / task-gpu-kv-i8.md.
```

## Options.KVQuant

Moved from `decoder/model.go` (the comment above `KVQuant`) on 2026-10-10.

```text
KVQuant selects the CPU KV cache storage precision: "" / "f32" (default,
bit-exact) or "i8" (per-(position,KV-head) symmetric int8, 4× smaller +
SDOT decode). Lossy, opt-in; excluded on MoE / gemma4 / qwen3_5_moe in v1.
See task-cpu-kv-quant.md.
```

## Options.AcceptSlowMoE

Moved from `decoder/model.go` (the comment above `AcceptSlowMoE`) on 2026-10-10.

```text
AcceptSlowMoE is S4 item 5's (task-never-swap-2026-09.md) explicit acknowledgement: a
paged-MoE StreamWeights load whose predicted working-set rate falls below
moeSlowTokPerSecThreshold (decoder/moeworkingset.go) is refused unless this is true. Only
meaningful for a .giw MoE load under StreamWeights; a no-op everywhere else.
```

## Options.ExtraResidentBytes

Moved from `decoder/model.go` (the comment above `ExtraResidentBytes` in `Options`) on 2026-10-10.

```text
ExtraResidentBytes prices a companion allocation that will claim VRAM on the SAME device
AFTER this model's own residency is built — a --drafter's weights today
(internal/serveapp's loadDecoder computes this via decoder.DrafterResidentBytesEstimate
before calling Load, so it is known before BuildResident runs). See
Model.ExtraResidentBytes's own doc comment for why this exists and what it fixes. 0 (the
default) is today's behavior, unchanged.
```

## Options.ExtraResidentKVPerPosition

Moved from `decoder/model.go` (the comment above `ExtraResidentKVPerPosition` in `Options`) on 2026-10-10.

```text
ExtraResidentKVPerPosition is ExtraResidentBytes' ctx-scaling twin (M-22,
docs/audit-2026-09-10.md): a companion allocation's device K/V, when it has any, scales with
whatever resident context THIS model ends up choosing — unknowable at loadDecoder's pricing
point, unlike the fixed weight bytes ExtraResidentBytes carries. See
Model.ExtraResidentKVPerPosition's own doc comment and decoder.DrafterKVBytesPerPosition for
the shape and rationale. 0 (the default) is today's behavior, unchanged.
```

## Options.ResidentKVSlots

Moved from `decoder/model.go` (the comment above `ResidentKVSlots` in `Options`) on 2026-10-10.

```text
ResidentKVSlots asks a GPU-resident backend for this many independent KV caches ("slots"), so several
interleaved conversations each keep their own prefix resident instead of evicting one another's
(docs/tasks/task-concurrency-2026-09.md MC1). Still one generation at a time: a slot is bound per generation,
never batched. 0 or 1 = one slot (the behaviour before MC1). A backend clamps it to what its fit guard allows
and says so; a backend that does not implement ResidentKVSlotter, and every family with recurrent state, keep
one slot. serve sets it from -kv-sessions.
```

## Model.KVCacheF16

Moved from `decoder/model.go` (the comment above `KVCacheF16`) on 2026-10-10.

```text
KVCacheF16 reports whether the GPU residency path should use an f16 KV cache
(Options.KVPrecision == "f16"): 2× context (32k) on the same VRAM, lossy. The
residency builder reads it; off the residency path it has no effect.
```

## Model.KVCacheI8

Moved from `decoder/model.go` (the comment above `KVCacheI8`) on 2026-10-10.

```text
KVCacheI8 reports whether the GPU residency path should use an int8 KV cache
(Options.KVPrecision == "i8"): 4× vs f32 / 2× vs f16, ~64k context on the 8 GB
card. Lossy; f32 + f16 paths unchanged. Distinct from KVQuant (the CPU cache).
```

## Generation.PrefillReused

Moved from `decoder/model.go` (the comment above `PrefillReused`) on 2026-10-10.

```text
PrefillReused is how many leading prompt tokens this generation skipped because they
were already committed to the resident positional KV (resident_reuse.go). 0 on a cold
prefill and on every non-resident path. Diagnostic: it is what makes an agent loop's
per-turn prefill cost visible without timing it. GenerateVL/GenerateQwenVL set it only
on P9(a)'s full-image-reuse fast path (docs/multimodal.md) — 0 there means that turn's
image (or everything before it) was NOT fully reused, whether because nothing matched
or because a partial match stopped short of the image block's own start.
```

## Generation.ImgPrefillResident

Moved from `decoder/model.go` (the comment above `ImgPrefillResident`) on 2026-10-10.

```text
ImgPrefillResident reports whether GenerateVL's resident image-prefill fast path (the
bidirectional image-block CUDA kernel, decoder.ResidentImagePrefill) actually ran this
turn's PREFILL on the GPU — false means the turn fell through to the CPU-prefill+UploadKV
bridge (gap 0), whether because no resident implements the capability, the prompt was too
long for one chunk, or any other decline. Diagnostic, same reasoning as PrefillReused: a
real-checkpoint gate asserting end-to-end correctness needs this to confirm the fast path
actually fired rather than passing vacuously via the (already-correct) fallback.
```

## Generation.DeviceSampled

Moved from `decoder/model.go` (the comment above `DeviceSampled`) on 2026-10-10.

```text
DeviceSampled counts decode steps whose token the resident drew on-device by Gumbel-max (R7b); 0 unless
that fast path was active.
```

## Generation.Budget

Moved from `decoder/model.go` (the comment above `Budget`) on 2026-10-10.

```text
Budget is the effective max-token budget after the resident context-cap clamp
(audit M-04). It equals the requested maxTokens unless prompt+maxTokens would
exceed the resident KV cap, in which case it is the remaining room (may be 0).
A caller that reports finish_reason must compare the emitted count against this,
not the requested value, or a context-clamped generation is mis-reported as a
clean "stop" and the client never continues. Set before the first token is sent;
read it after the channel closes (like Err). 0 on generation paths that don't
clamp (speculative/VL).
```

## Generation.BudgetClamped

Moved from `decoder/model.go` (the comment above `BudgetClamped`) on 2026-10-10.

```text
BudgetClamped is true iff the resident context cap (not the request) bounded this turn, so
Budget is the authoritative limit — including a clamp to 0 when the prompt fills the whole
context. A caller must judge finish_reason against Budget only when this is set; otherwise it
falls back to the requested max_tokens. Without it, a genuine clamp-to-0 is indistinguishable
from an unclamped Budget-0 turn and an empty-context-full response mis-reports "stop" (R-09).
```

## Model.generateInto.chain

Moved from `decoder/model.go` (the comment in `generateInto`) on 2026-10-10.

```text
The greedy chain (C-B01, ResidentGreedyChain): the same tokens with the next token's forward queued on the device
before the host has seen this one. With no processor at all (a gated one would need the full row mid-chain) and no
adapter, and only where the embedding lookup is a plain table row, which the resident's gather reproduces. Under
MC3 it runs only on the tokens this generation decodes alone, with the resident held across them (holdSolo); every
other token takes the batcher as before, so fastGreedy stays the path for those.
```

## Model.generateInto.optfwd

Moved from `decoder/model.go` (the comment in `generateInto`) on 2026-10-10.

```text
Optimistic forward: sampled decode's (Temperature>0) sibling of the greedy fast path
above, but overlapping rather than skipping the CPU sampler -- see spec_optfwd.go.
Excludes fastGreedy's own (rare: Temperature>0 AND top_k==1) overlap with this predicate
so the two mechanisms never both try to drive the same step. GOINFER_NO_OPTFWD forces
the plain sequential path (escape hatch / A-B check), same convention as
GOINFER_NO_GREEDY_FASTPATH.
MC3 (mc3 != nil): off — optFwdStep drives the resident itself, outside the batcher's exclusive section.
```

## Model.generateInto.samplechain

Moved from `decoder/model.go` (the comment in `generateInto`) on 2026-10-10.

```text
The sampled chain (C-P02, ResidentSampleChain): where the device draw serves the token, the same draw with the next
token's forward queued on the device first, as the greedy chain does for the argmax. Same exclusions as the greedy
chain's, and not where ForwardSample itself takes the argmax (a temperature so small 1/T is infinite).
```

## resetPrefillDeclineDedup

Moved from `decoder/model.go` (the comment above `resetPrefillDeclineDedup`) on 2026-10-10.

```text
resetPrefillDeclineDedup clears the per-reason dedup state; test-only (mirrors the old
`prefillDeclineOnce = sync.Once{}` reset tests used before this was keyed per reason).
```

## Model.ForwardSubCapture

Moved from `decoder/model.go` (the comment in `ForwardSubCapture`) on 2026-10-10.

```text
EVERY own-forward family, derived: this seam needs runLayersFromEmbed's uniform block, which
no own-forward loop routes through. The hand-written list was that set minus lfm2.
```

## Model.generateInto.budget

Moved from `decoder/model.go` (the comment in `generateInto`) on 2026-10-10.

```text
Publish the effective budget so the caller reports finish_reason "length" when this
clamp (not an EOS) ends the turn (audit M-04). Set before any send; read after close.
BudgetClamped disambiguates a genuine clamp-to-0 (prompt fills the cap → "length") from an
unclamped turn whose Budget is coincidentally 0 (R-09).
```

## Model.NewCache.lfm2

Moved from `decoder/model.go` (the comment in `NewCache`) on 2026-10-10.

```text
Hybrid cache: KV for the 8 attention layers + a rolling conv window for each of
the 22 conv layers. manualPos because the conv layers never Append, so position
cannot be inferred from the KV length.
```

## Model.NewCache.rings

Moved from `decoder/model.go` (the comment in `NewCache`) on 2026-10-10.

```text
Ring-buffer storage on sliding-window (local) layers: keep only the W most
recent positions, the only ones a future query can read. Restricted to the
uniform-stride families whose forward uses attendQuery/attendBatchedHeads;
gemma4 (per-layer widths + KV-sharing) and qwen3_5_moe (linear attention)
have their own forward and keep append-forever for now (a later increment).
```

## Model.NewCache.kda

Moved from `decoder/model.go` (the comment in `NewCache`) on 2026-10-10.

```text
Bailing Hybrid (Ling 3.0): a THIRD hybrid-cache shape, alongside qwen35's and mla's
above — the MLA layers' latent cache is already handled by the a.mla block above
(bailingHybridArchitecture sets both), so only the KDA linear layers' recurrent
state needs its own array here.
```

## Session.rewindForReuse

Moved from `decoder/session.go` (the comment above `rewindForReuse`) on 2026-10-10.

```text
rewindForReuse rewinds the cache to the longest reusable prefix shared with prompt and returns
how many tokens to reuse. On an INEXACT rewind — a wrapped sliding-window ring can't restore the
positions it dropped (C1) — it Resets and returns 0 (cold prefill), so prefix reuse never reads
stale history. Callers must skip it (and reconcile) for an empty prompt, so a rejected call
```

## Session.reconcile

Moved from `decoder/session.go` (the comment above `reconcile`) on 2026-10-10.

```text
reconcile sets s.tokens to EXACTLY what the cache holds after a generation. seq mirrors the cache
(prompt + each committed token); on a prefill/forward error the cache may hold fewer positions
than seq claims, so clamp — otherwise the next call "reuses" KV that was never written and
prefills at the wrong position (M10).
```

## RegisterBackend

Moved from `decoder/backend.go` (the comment above `RegisterBackend`) on 2026-10-10.

```text
RegisterBackend registers a named Backend factory. The goinfer/gpu module
calls this from init() (under `-tags gpu`) to make "webgpu" available
without the decoder importing the cgo WebGPU implementation. Safe for
concurrent use; a later registration of the same name replaces the earlier.
```

## NewBackend

Moved from `decoder/backend.go` (the comment above `NewBackend`) on 2026-10-10.

```text
NewBackend returns the named backend. "" and "cpu" always resolve to the
pure-Go CPU backend. Other names resolve through the registry; "webgpu"
falls back to CPU with an explanatory error (rather than hard-failing) when
goinfer/gpu has not been imported, so a `--backend webgpu` flag still runs
on a build without the GPU module.
```

## Inline id labels

Trailing and in-body comment labels removed by the comment diet (2026-10-10) from `decoder/model.go`, `decoder/session.go` and
`decoder/arch.go`; the code and the sentence they sat on are unchanged. Audit and tracker ids, with the sentence each labelled:

- `resBusy` (`Model`): "(M9)", the single-claim concurrency fix.
- `resIDsLora` (`Model`): "(audit C-02; resident_reuse.go rule 4)".
- `Options.Quant`: "(M8)".
- `residentPrefillSeed`: "A-P05" on the 8-token batched-prefill floor; "G18" on the per-token ctx check in the sequential loop
  ("the resident prefill loop is the GPU-side twin of the batched CPU path's per-layer check. Same failure without it — an abandoned
  client leaves the whole prompt streaming through the device"); "~1 MB readback" in the KV-only prefill note.
- `generateInto`: "(M-04/R-09)" on `BudgetClamped`, "(M8)" on the send-select, "(E-C01, mc3Logits)" on the MC3 solo copy.
- `Generation.ImgPrefillDecline`: "S11", the multi-image turn.
- `Session.reconcile` / `Session.genSpec` / `Session.GenerateGrammarSpeculative`: "(C-01)", "(N-01)", "(C1)".
- `ownForwards` (`bailing_hybrid`): "(audit C-03)" on the Recurrent bit.
