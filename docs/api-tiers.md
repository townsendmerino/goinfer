# API stability tiers — what v1.0 semver-binds

> **STATUS: SIGNED OFF 2026-08-18 (decider: Francis).** This is the v1.0 gate's §3 deliverable —
> the declaration §0 of `release-1.0-gate.md` says is owed. It is written from the **actual
> exported surface at HEAD**, enumerated with `go doc`, not from memory.
>
> **The split takes effect AT the v1.0 tag, not at this sign-off.** Until then goinfer is still
> pre-1.0 and everything can move; what is settled now is *which* surfaces v1.0 will bind, so the
> apidiff baseline has something to check against and the RC's docs have something to cite. The
> three open questions at the bottom are their own gate lines and are NOT decided here.

Two tiers, following aikit's precedent (its README §"Stability tiers"), for one reason: a 1.0 that
promises everything promises nothing, and a 1.0 that promises nothing is not a 1.0. The split says
which parts you can build on and which parts are still moving — and **the Experimental list is
explicit, never "whatever is not listed above"**, so a surface cannot end up outside the promise by
being forgotten.

## Hard — the 1.0 compatibility guarantee

From v1.0 these follow semver: no breaking change before a v2.0.

### `decoder` — load and generate

- `decoder.Load`, `decoder.LoadGGUFBytes`
- `decoder.Options` — the field SET is Hard, and specifically **`Options.Quant`** (with its
  values `""`, `int8`, `int8int8`, `int4`; new values may be ADDED) and **`Options.LoRA`** (the
  load-time adapter path). **Individual fields that name an Experimental subsystem are
  Experimental** (see below), because the option exists only to reach it.
- `decoder.Model`: `Close`, `Config`, `Dims`, `Quant`, `NewCache`, `NewSession`, `Generate`
- `decoder.Session`: `Generate`, `Reset`, `Tokens`, `Snapshot`, and `Model.LoadSession`
- `decoder.SamplingParams`, `decoder.Generation` (`Err`), `decoder.KVCache` as an opaque handle
- `decoder.Config` as a READ surface (what a loaded model reports)

### `tokenizer`

- `tokenizer.Load`, `LoadGGUF`, `LoadGGUFBytes`, `LoadJSONBytes`
- `tokenizer.Tokenizer`: `Encode`, `EncodeSegments`, `Decode`, `DecodePiece`, `TokenText`,
  `TokenID`, `Has`, `Special`, `ChatTemplate`
- `tokenizer.Segment`, `tokenizer.SpecialTokens`

### `chat`

- `chat.Detect`, `chat.Meta`, `chat.ErrUnknownTemplate`
- `chat.Template`: `Name`, `Render`, `RenderSegments`, `ParseToolCalls`
- The named constructors: `ChatML`, `Gemma3`, `Gemma4`, `Harmony`, `Llama3`, `Mellum2`, `Mistral`
- `chat.Turn`, `chat.Tool`, `chat.ToolCall`, `chat.Stops`, `chat.Segment`

**Adding a template constructor is not a breaking change**; changing what an existing one renders
for a given input IS, because callers pin prompts against it.

### `constrain`

- `constrain.Grammar` (interface), `constrain.NewMasker`, `constrain.Masker`'s
  `Process`, `MaskAt`, `Commit`, `Reset`, `CanEnd`, `StopWhenComplete`, `ForcedRun`,
  `ForcedBytesRun`, `GrammarClone`
- `constrain.TokenBytes`, `constrain.SchemaFromStruct`

### `cmd/serve` — the operator surface

- **HTTP**: `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`,
  `/v1/messages`, `/v1/messages/count_tokens`; `GET /v1/models`, `/health`. Request/response
  shapes follow the upstream OpenAI and Anthropic specs they implement — **their compatibility is
  the promise; our field ORDER and any extension fields are not.**
- **Flags**: `-addr`, `-api-key`, `-tls-cert`, `-tls-key`, `-name`, `-quant`, `-backend`,
  `-ctx-size`, `-max-inflight`, `-max-queue`, `-max-body-bytes`, `-kv-sessions`, `-session-dir`.
  Removing or renaming one of these, or changing its default in a way that changes behaviour, is
  breaking.
- `/admin/models/load` and `/admin/models/unload` are Hard **only in that `-allow-admin` gates
  them**; their bodies are Experimental (see below).

### `docs/env-vars.md` becomes contract, not documentation

The **operator-facing** variables the registry curates (its "Serving", "Residency & MoE" and
equivalent sections — 26 documented rows at the time of writing) are Hard: removing one, or
changing what it does, is breaking. **The ~130 grep-derivable test/diagnostic knobs are
Experimental** and the registry already separates them; that separation is now load-bearing rather
than editorial.

## Experimental — ships in 1.0, explicitly OUTSIDE the guarantee

Real, supported, and used in production paths here — but they may change in any release, minor or
patch. Pin a version if you depend on them. Each graduates when it settles.

- **The backend/residency seam.** `decoder.Backend`, `RegisterBackend`, `QuantBackend`,
  `QuantBatchBackend`, `ResidencyBackend`, `Prefiller`, `ResidentForward`, `ResidentGreedy`,
  `ResidentPrefillKV`, `ResidentCapped`, `ResidentDrafterHost`, `ResidentBlockDrafter`,
  `ResidentFeature`, `ResidentEligible`, `ResidentBackendFeatures`, and **every `*Resident*`
  method on `Model`**. This is the seam the `gpu/`, `cuda/` and `metal/` submodules bind to; it
  moves whenever a family is ported.
- **Family descriptors and weights.** `decoder.Architecture`, `Weights`, `LayerWeights`,
  `MoEConfig`, `ActKind`, `NormKind`, `NormPlacement`, and the `qwen35Params`-class internals they
  carry. A new family routinely adds fields here.
- **Speculative decoding and drafters.** `Drafter`, `NgramDrafter`, `GrammarDrafter`,
  `RouterDrafter`, `DFlashDrafter`, `DSparkDrafter`, `ConfidentDrafter`, `EagleHead`,
  `EagleState`, `AdaptiveDepth`, `BlockSpec`, `BlockSpecOptions`, `BlockDrafterWeights`,
  `DrafterGeometry`, `DrafterLayerWeights`, `SpecStats`, `SpecTrace`, `DraftInfo`,
  `OutcomeRecorder`, and every `Model.Generate*Speculative*` entry point.
- **Multimodal.** `Model.GenerateVL`, `GenerateQwenVL`, and the `multimodal` package.
- **Adapters at compute time.** `Model.LoadAdapter`, `HasAdapter`, `Session.UseAdapter`,
  `Session.ClearAdapter`, and `Options.LoRA`'s multi-adapter behaviour.
- **Diagnostics and capture.** `Model.ForwardCapture`, `ForwardSubCapture`, `HiddenLast`,
  `DecodePath`, `PrefillPath`, `PrefillPathReporter`, `ResidentDecline`, `MmapByteOffset`.
- **Serialization plumbing.** `SerializeWeights`, `SerializeWeightsTo`, `StreamTranscodeGGUF`,
  `SerializeError`, `SnapshotError`, `NewModel`, `CheckGiwQuantMatch`, `GiwPath`. (The `.giw`
  FORMAT promise is a separate line — see Open questions.)
- **Tuning knobs.** `SetDecodeParallelThreshold`, `DefaultDecodeParallelThreshold`,
  `ErrBlockSpecUnsupported`.
- **`Options` fields that reach the above**: `Backend`, `KVPrecision`, `KVQuant`,
  `MoECacheExperts`, `MoECacheSlots`, `StreamWeights`, `WeightCacheBytes` and their `serve` flag
  twins (`--backend`'s non-cpu values, `--kv-prec`, `--kv-quant`, `--moe-cache-*`,
  `--stream-weights`, `--drafter`, `--spec`, `--adapter`, `--vision-*`, `--metal-fast-prefill`,
  `--embed-*`, `--require-be`, `--allow-admin` bodies). `Options.Quant` and `Options.LoRA` are
  Hard and listed above.
- **The submodules themselves** — `gpu/`, `cuda/`, `metal/`, `demo/agent` — are Experimental as
  packages regardless of their tag numbers (see Open questions on posture).

## What this page does NOT decide

1. **Submodule tag posture** — do `gpu/`, `cuda/`, `metal/`, `demo/agent` tag `v1.0.0` alongside
   the core or stay on their own 0.x series? `release-1.0-gate.md` §3 carries it; aikit kept
   `gpu/` at 0.x. Either is fine, undecided is not — the tag-day script needs to know.
2. **The `.giw` / `.giw-kv` format promise** — read-N−1, reserved-header forward-compat, or
   documented rebuild-on-minor. §3 owes one sentence that both the code comment and the README
   cite. Today they say "rebuild per minor" with a version guard that fails loud.
3. **The apidiff baseline** — §3 requires it clean across at least one minor, in CI, against this
   Hard list. That is what turns this page from a promise into a gate.

## Why the split falls where it does

The Hard list is small on purpose, and it is exactly the surface the demos, `cmd/serve` and the
README examples use: load a model, tokenize, render a chat prompt, generate, optionally constrain.
That surface has been stable in practice for several minors — it is the part where a break would
strand users who never asked for a backend detail.

Everything in the Experimental list moves for a REASON that is still live: a new family adds
descriptor fields, a ported family changes the residency seam, a drafter's economics change its
interface. Freezing those at 1.0 would either stop the work or make the promise a lie within a
minor. Naming them is the honest alternative — and it is naming, not omission, so a reader can tell
the difference between "excluded" and "forgotten".
