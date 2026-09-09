# Task: serving paths that run on the CPU when a GPU path exists — 2026-09 (G1–G11)

> **Status: OPEN, drafted 2026-09-08** from the code at `c9db2ec`, not from the docs — R9
> (`docs/task-first-hour.md`) showed the docs can say "GPU" where the code says CPU, so every item
> below cites the line that decides. Companion to `docs/gpu-residency-coverage.md` (the WebGPU
> runner's per-family scoping) and `docs/hardware-matrix.md` (the generated admission table).
>
> The question this answers: after R9 (CUDA/Metal have no staged GPU path; WebGPU's staged path
> reaches the GPU for f32/int8 only) and R13, what *else* on a GPU box runs on the CPU when it
> could run on the CUDA, Metal or WebGPU code we already have — or on code that is one kernel
> away? Three kinds: **A** requests that run entirely on the CPU while the resident path sits
> idle; **B** requests that are on the GPU but on a slower GPU path than exists; **C** packaging.
>
> Suggested order: G1 → G2 (folded into multimodal P6) → G3+G4 together → G5 (four one-kernel
> families) → G6 → G7 (Metal TTFT) → the rest as they come up. Items are independent; each lands
> with its own gate and hardware-matrix/CHANGELOG touch, in the pattern of task-families-2026-09.

## Ground rules (same as every task doc here)

- Nothing is declared resident without an end-to-end gate on that backend (features.go's rule:
  "a backend adds an entry ONLY when it ships the kernel that implements it").
- Bit-identity contract holds: a path moved from CPU to GPU must reproduce the CPU stream on the
  existing parity harness, or decline by default the way Metal batched prefill does.
- Every item updates the thing that lied, if anything did: hardware-matrix footnote, `serve check`
  row, `BackendReport()` label, or the release-asset assertion.
- One doc. Findings from doing the work go into this file's per-item status line, not new docs.

---

## A. Whole-request CPU fallbacks on a GPU box

### G1 — `goinfer-chat-*` release assets have no backend (packaging, same class as v0.16.0's serve asset)

**Where.** `.github/workflows/release-assets.yml:52–54` builds every `goinfer-chat-<os>-<arch>` and
`goinfer-chat-0.5b-<os>-<arch>` asset from `./demo/chat` with no `-tags`. The workflow's own comment
at line 83 says so ("the chat binaries above are built with NO -tags, so the darwin asset cannot
use the GPU"). `metal/cmd/chat` and `cuda/cmd/chat` already exist and are `chatapp.Main()`.

**Evidence.** Cold run 2 (nobara) and 2b (Mac) both saw `[backend=cpu quant=int8int8]` from the
chat binary on machines with a GPU. The v0.17.0 fix covered `goinfer-serve` only.

**Fix.** Build darwin chat assets from `./metal/cmd/chat`, linux from `./cuda/cmd/chat`, windows
from the root (no GPU backend for it), with the same `go version -m` main-module assertion the
serve assets got in v0.17.0 (`release-assets.yml:164–169`), extended to the chat assets and to the
embedded `-0.5b` one-file variant. The embedded-model step (`release-assets.yml:62–71`) must take
its runtime from the per-backend binary, not the root one.

**Gate.** The existing asset assertion loop, extended: every darwin chat asset reports
`mod github.com/townsendmerino/goinfer/metal`, every linux one `.../cuda`. Add `--version` output
to the readme-smoke job for the chat binary too.

**Size.** Small — workflow only. Do this first.

### G2 — image turns run the whole text decoder on the CPU, not just the tower

**Where.** `decoder/generate_vl.go:18–30`: `GenerateVL` (and `GenerateQwenVL`) are "stateless and
CPU-only by design — never touches m.resident at all". `internal/serveapp/openai.go:1056–1077`
(`driveVL`) is the only caller from serve; `prepare()` is told `residentPath=false` for vision
(`openai.go:660–663`).

**Effect.** On the Mac or a CUDA box, a Gemma 3 image request runs the *text* decode at CPU speed
even though `gemma3` text is resident on both backends. `-tags gpu` moves only the SigLIP tower
(`docs/multimodal.md`); the cgo-free release binaries move nothing.

**Fix.** Two seams, both already designed in `docs/multimodal.md` §5 for the CPU path and now
needed on the resident one: (a) an embed-by-vector override on the resident runners' prefill
(`cudaResident.PrefillLast` / `metalResident.PrefillLast` / the WebGPU sequential prefill) so the
projected image block is written into the positions its placeholders occupy; (b) the bidirectional
image-block mask in each backend's prefill attention. Then `driveVL` takes the resident path when
`ResidentActive()`, claiming `resBusy` like `Generate`. Prefix reuse for image turns is P9's
image-byte-hash item and stays out of scope here.

**Gate.** The existing end-to-end VL parity fixture (Gemma 3 4B, precomputed `pixel_values`) run
through the resident path on each backend; text-only prompts through the new seams must stay
bit-identical (the multimodal doc's inertness rule).

**Size.** Medium-large per backend. This is the first item of multimodal P6 in all but name —
record it there as P6a and do it with the tower move rather than after.

### G3 — LoRA adapter requests drop to the staged path (100% CPU on CUDA/Metal)

**Where.** `internal/serveapp/openai.go:986`: `if lm.model.ResidentActive() && lm.adapter == ""` —
adapter models take the session path below it, and `decoder/model.go:1045` makes a session
generation ineligible for the resident KV (`useGPU = resident != nil && prefillFrom == 0 &&
commit == nil`). The comment at `openai.go:960–967` records the cost: 13 tok/s vs ~460 resident on
a 0.5B (RTX 2070 SUPER). Documented as audit R-01 and left there.

**Fix.** Apply the compute-time LoRA on the resident path: the adapter is a per-projection
low-rank delta applied to the activations (`Session.UseAdapter` → cache's `lora`), so the resident
runners need one extra GEMV pair per adapted projection per token, with the delta weights uploaded
at `bindAdapter` time. Alternative that is cheaper and may be enough: merge the adapter into the
resident weights at bind time (re-pack the affected projections) and treat "switch adapter" as a
re-pack; one adapter per loaded model at a time, which is what `lm.sessions.adapter` already
assumes (`main.go:829`).

**Gate.** An adapter-vs-merged parity test on the tiny fixture, then the R-01 measurement
re-run on the 0.5B.

**Size.** Medium. Shares its seam with G4.

### G4 — `/v1/embeddings` on decoder models never uses the resident path

**Where.** `decoder/embed.go:36–71`: `HiddenLast` → `hiddenLastBatched` (CPU `forwardN`) or
`hiddenLastSequential`; neither consults `m.resident`. `internal/serveapp/decoder_embedder.go` is
the only serving caller.

**Fix.** A resident `HiddenLast`: the resident prefill already exposes hidden-state capture for the
block drafter (`hidCapTaps`, `cuda/prefill.go:311–319`), so a "prefill and return the last row's
pre-LM-head hidden state" entry is mostly wiring on CUDA; Metal and WebGPU need the same tap. Must
respect `ownForward` families (they error today, keep that), claim `resBusy`, and forget `resIDs`
after (it drives the shared positional KV).

**Gate.** Cosine ≥ 0.9999 against the CPU vector on the embedding fixture, per backend.

**Size.** Small-medium on CUDA, medium elsewhere. Do with G3.

### G5 — families CPU-only on CUDA and Metal for one or two small features

**Where.** `decoder/features.go:390–541` (the three backend tables) against
`decoder/features.go:129–221` (`residentFeatures`). Everything below declines to the staged path,
which on CUDA/Metal is entirely CPU (R9), so each missing kernel costs the whole model's speed.

| Family | Missing on CUDA | Missing on Metal | What the kernel is |
|---|---|---|---|
| SmolLM3 | `FeatNoPE` | `FeatNoPE` | skip RoPE on the listed layers (`no_rope_layers`, inverted) — a per-layer flag on the existing rope launch |
| Ministral 3 | `FeatAttnTemp` | `FeatAttnTemp` | one per-position scalar on Q (`llama_4_scaling_beta`), applied before attention |
| Olmo 3 | `FeatPostOnlyNorm`, `FeatQKNormWhole` | same | norm placement (post-sublayer only) + QK RMSNorm over the whole projection width instead of per head — a reduction-width variant of `qk_norm` |
| Olmo Hybrid | the two above + `FeatNoPE` | same | its Gated-DeltaNet half is already declared on both backends |
| Command-R / R7B | `FeatLayerNorm`, `FeatParallelBlock`, `FeatLogitScale` | `FeatParallelBlock`, `FeatLogitScale` | parallel attn‖MLP from one normed input, summed; logits scale is a host-side multiply; Metal already has the LayerNorm (generalized for Cohere, `features.go` note) |
| Nemotron-H | `FeatSSM`, `FeatNonGatedMLP`, `FeatLogitScale`… | `FeatSSM`, `FeatLogitScale` | the Mamba-2 engine exists on WebGPU (`gpu/`); a port, not a design |
| DeepSeek-V2/V3, Kimi K2 | `FeatMLA` | `FeatMLA` | exists on WebGPU; **gate the nGroup/topkGroup mapping first** (the CUDA TRAP comment, `features.go:396–405`) |
| Laguna | `FeatAttnOutputGate` | same | not on any backend; WebGPU's DeltaNet has a fused output gate to crib from |
| LFM2.5 | `FeatShortConv` + "own forward, not bridged" | same | `residency.go:207` declines it before features are consulted |
| Llama 4 | own forward, not bridged | same | `residency.go:205` |
| Ling 3.0 | `FeatKDA` | same | not on any backend |
| Gemma 4 E2B/E4B | `FeatGemma4EModel` | same | PLE + shared-KV + per-layer FFN — not on any backend; the 26B/31B are resident |

The first five rows are each one small kernel or a wiring change on backends that already run the
rest of the family, and the tiny-oracle fixtures from task-families-2026-09 are the gates. Do
those five, in that order, and treat the rest as the standing residency backlog
(`gpu-residency-coverage.md`).

### G6 — WebGPU side of the same table: Gemma 3, Gemma 4, gpt-oss, and staged int4

**Where.** `features.go` "webgpu" table lacks `FeatSandwichNorm`, `FeatGatedGELU`,
`FeatEmbedScale`, `FeatFinalLogitSoftcap` (the Gemma set) and `FeatAttnSink`, `FeatOutBias`
(gpt-oss) — every one of which CUDA and Metal already implement. Pure ports. Also unchanged since
R9: the WebGPU staged path never consults a backend for int4 (`decoder/weightmat.go` staged
dispatch), and `gpu/gemv_w4a8.go` is wired only into the resident runner.

**Size.** Medium (ports). Lower priority than G5 because the cgo-free release binaries don't carry
WebGPU.

### G7 — Nemotron 3 Nano / 3.5 Lightning are CPU on every backend, and the matrix says otherwise

**Where.** `decoder/residency.go:242`: `if a.nemotron != nil { return a.MoE == nil }` — the
MoE block kind has no resident builder on any backend (comment at 234–240). `docs/hardware-matrix.md`
row "Nemotron-H → WebGPU ✅ resident" is generated from the *dense* representative config, so it is
true of Nemotron-H and false of the two models people download. task-families-2026-09 F2
"verified" Lightning against the adapter without noting it runs CPU-only.

**Fix, two parts.** (1) Today: a footnote on the matrix row and a line in the Lightning/Nano
family docs; `serve check` should say "CPU (MoE block not resident)" for these. (2) The real one:
add the MoE FFN case to the WebGPU Nemotron block switch (`gpu/residency.go:509`, cases 0/1/2;
the `default` that residency.go's comment says is missing is there now at line 583, so an
unknown kind declines cleanly), gated on the real Nano checkpoint on the Linux box.

**Size.** (1) trivial; (2) medium on WebGPU, and moot on CUDA/Metal until G5's SSM row lands.

---

## B. On the GPU, but on a slower GPU path than exists

### G8 — Metal prefill is sequential for every non-plain-dense family, flag or no flag

**Where.** `metal/model.go:63–67`: `prefillFeatures` is exactly `{FeatQKNorm, FeatSlidingWindow,
FeatPartialRotary}`; `metal/model.go:535` sets `prefillOK` from it; `metal/backend.go:255` declines.
Separately, `metal/backend.go:248` declines batched prefill unless `GOINFER_METAL_BATCHED_PREFILL=1`
(the 54% stream divergence, §A2-Metal). So MoE, Gemma, DeltaNet, gpt-oss and GPT-2 prompts on the
Mac are one forward per prompt token regardless of `--metal-fast-prefill`. CUDA's batched prefill
covers dense and MoE (`cuda/prefill.go:289–320`) and declines only f32 projections and the
per-token debug seams.

**Fix.** Two levers, in order: (a) extend the f16-MMA prefill to MoE (per-row FFN off the batched
residual, exactly what CUDA does) and the Gemma set — the kernels exist for decode; (b) the
bit-identity work that would let it default on, tracked in `task-metal-batched-verify-kernel.md`
and `task-int4-int8-exact-mma.md`. (a) alone is worth it as an opt-in: measured 3.9–4.6× TTFT on
dense (`docs/ollama-chase.md`), and the Mac's remaining gap to Ollama is mostly TTFT.

**Size.** Medium. This is the Metal-specific lever if the Mac is the target.

### G9 — WebGPU has no batched prefill at all

**Where.** `decoder/model.go:919`: "WebGPU implements no Prefiller"; `gpu/residency.go:1028`
seeds the caches via sequential `Forward`. Every prompt on WebGPU is one submit per token.

**Fix.** A `Prefiller` on the WebGPU runner, dense first, following the CUDA shape
(`task-gpu-batched-prefill.md`). **Size.** Medium-large; lower priority (see G6).

### G10 — Metal has no int8 weight kernel: `int8int8` is requantized to W4A8 on device

**Where.** `metal/model.go:346–363` (`int4Buf`): an int4 weight is packed directly; an int8 weight
is dequantized to f32 and re-packed as 4-bit/group-32. There is no W8 GEMV in `metal/`. So
`-quant int8int8` on Metal runs int4 numerics on the GPU while holding the int8 host copy — more
RAM, not more precision. `metal/backend.go:52`'s comment ("weights must be int8-loaded…") is stale
in the other direction.

**Fix.** Either a W8A8 GEMV on Metal (CUDA has `gemv_w8a8_batched`), or make the requant explicit:
the load banner and `BackendReport()` should say "metal resident (int8 weights repacked to
W4A8)", and `docs/quantization.md` should say int8int8 on Metal is an int4 path. The "try 7B at
int8int8 on the Mac" loose end from task-first-hour is void until this is decided.

**Size.** Reporting: trivial, do now. Kernel: medium.

### G11 — no layer placement on CUDA/Metal (resident-or-CPU)

**Where.** `cuda/backend.go` / `metal/backend.go` `BuildResident` admit the whole model or decline
it; there is no per-layer split. This is where llama.cpp `--fit` beat goinfer on the 8 GB card
(M35 32.9 vs 23.5, M26 27.8 vs 24.6 tok/s; `docs/benchmarks.md` peer matrix). Already scoped as
`task-fit-to-hardware.md`; listed here for completeness, not re-scoped.

---

## Things checked and found fine

- The `resBusy` CAS loser falls to the staged/CPU path (`decoder/model.go:1046–1056`), but serve
  serializes each model's generations (`internal/serveapp/openai.go:62` `mu`), so it never fires
  through the HTTP surface; only direct library callers running two generations on one `Model`
  see it.
- Constrained/tool requests keep the plain resident `Generate` (`openai.go:977`).
- The n-gram and block drafters claim `resBusy` and verify on the resident batched `ForwardN`;
  the CPU block drafter is measured-negative and deliberately not wired (`blockspec_cpu.go`).
- Sampling, argmax readback, grammar masking and tokenization are per-token host work by design.

## Status log

- 2026-09-08 — drafted from `c9db2ec`. Nothing started.
- 2026-09-08 — G1 done. `build()` in `release-assets.yml`'s chat-asset step now mirrors the serve
  step: darwin from `./metal/cmd/chat`, linux from `./cuda/cmd/chat` (both via the same
  ephemeral-checkout `go mod edit -replace`), windows from the root; the assertion step gained the
  `go version -m` main-module + stale-proxy-root checks the serve assets already had, plus a
  `backends:` grep on the native binary. `build-embed.sh`'s per-target loop got the same
  metal/cuda entrypoint switch for the `-0.5b`/`-1.5b` embedded assets, gated on `go env GOWORK`
  so local (non-CI) invocations — this script is also run by hand per README.md/RELEASE_TEMPLATE.md
  — skip the replace entirely in the workspace-mode case and, when the replace IS needed, drop it
  again right after the build so the tree comes back clean either way. Verified end-to-end on this
  Mac for both branches (workspace-mode and `GOWORK=off`, darwin+linux targets): `go version -m`
  reports the right submodule, `--version` reports the right `backends:` line, `git status` on
  `metal/go.mod`/`cuda/go.mod` is clean after. Also added a `--version` smoke check for the plain
  `go install .../demo/chat@latest` path in `scripts/readme_smoke.sh` (the Gate's last ask); ran
  the full script before and after — two pre-existing, unrelated failures (`-backend` missing from
  `goinfer-serve --help`, and the smoke-model ref-extraction misreading a couple of README lines)
  reproduce identically on `c9db2ec` with none of this branch's changes, so left alone.
- 2026-09-08 — G4 partially done, decoder+Metal VERIFIED, CUDA WRITTEN BUT UNCOMPILED (no CUDA/
  Linux box available this session — `aikit/gpu`'s CUDA half is hard-gated `//go:build linux`, so
  even `go vet -tags cuda ./cuda/...` fails on darwin with `undefined: gpu.Library` etc. before it
  ever reaches this branch's own changes). G3 was parked for now instead of done alongside it: its
  tractable fix (merge-at-bind, reusing the already-tested `loraAdapter.merge()`/`loadWeights`
  pipeline, zero new kernel code) is a real product tradeoff — N adapters sharing one base model's
  RAM today vs. one full resident copy per bound adapter — that needs a decision, not just code.
  - **Decoder (backend-agnostic, fully verified here).** New `decoder.ResidentHiddenLast` optional
    interface (`residency.go`); `HiddenLast` (`embed.go`) tries it first — claims `resBusy` (CAS,
    same as `Generate`), calls the new `hiddenLastResident` helper (embeds via `embedResident`,
    calls `residentForgetIDs` first), falls through to the existing CPU path on ANY resident
    decline or a lost CAS, exactly like every other resident decline in this codebase. New seam
    test `decoder/resident_embed_seam_test.go` (fakeResidentHiddenLast, mirrors
    `resident_seam_test.go`'s pattern) gates the wiring itself with no GPU: resident-used, CPU-
    fallback-when-busy, CPU-fallback-on-decline, and `resIDs` left nil after — all pass.
  - **Metal (fully verified on this Mac, real hardware).** New `resident.forwardHiddenNoHead`
    (`model.go`) generalizes `forwardHeadForTest`'s dequant recipe into a production, head-skipping
    primitive; new `metalResident.HiddenLast` (`backend.go`) loops it per token (Metal's batched
    `PrefillLast` declines by default — not bit-identical to decode, §A2-Metal — so this reuses the
    sequential per-token kernels instead, like decode does). New real-checkpoint gate
    `metal/hiddenlast_resident_parity_test.go` against the committed (gitignored, already-present)
    `testdata/gpt2` fixture — chosen specifically because GPT-2 exercises learned positional
    embeddings, the one family-specific wrinkle `forwardHiddenNoHead` has to reproduce. **Finding**:
    the doc's stated "cosine ≥ 0.9999" gate does not hold for a CPU-vs-Metal-resident int8
    comparison — measured 0.9991–0.9993, consistent with `gpt2_resident_parity_test.go`'s own
    accepted 0.95 floor for the SAME comparison one step further downstream (post-LM-head). Bar
    recalibrated to 0.998 in the new test, documented inline; both PASS, plus a Reset-then-replay
    check (a second, shorter sequence must not see KV left over from the first). Full `metal/...`
    suite (`-tags goinfer_testhooks`) reran clean after, ~119s, no regressions.
  - **CUDA (written from the research + first-principles reading of `resident.go`/`prefill.go`, ZERO
    compile/run verification).** `prefillChunked` refactored into `prefillChunkedTail(..., finalTail
    int)` (mechanical, behavior-preserving for the existing `tailLastLogits` call — `prefillChunked`
    is now a one-line wrapper); new `cudaResident.HiddenLast` calls it with a new `tailHiddenLast`
    tail mode added to `prefillCore`'s per-row tail switch: runs the identical `r.rms` final-norm+
    quant dispatch the LM head already reads from, then — instead of `r.doG` (the head GEMV) —
    downloads `r.aq`/`r.aSc` and dequantizes host-side via `linalg.DequantizeRowInt8` (confirmed by
    reading `aikit/gpu`'s `Download[T]`: a raw byte copy with no coupling to the buffer's original
    element type, so downloading `r.aq` — allocated `int32`-typed — as `[]int8` is exactly the H
    quantized bytes, no repacking needed). Added an explicit guard so the final Gemma softcap loop
    skips `tailHiddenLast` output (it holds a hidden state, not logits — applying a logit-only
    transform there would have been a silent, family-scoped correctness bug). **This entire
    paragraph needs a real CUDA box before it can be trusted**: run `go vet -tags cuda ./cuda/...`
    first (catches anything the darwin gofmt-only check below could not), then a CUDA twin of
    `hiddenlast_resident_parity_test.go` against a real dense checkpoint, then decide whether the
    0.998-vs-0.9999 recalibration found on Metal also applies here (CUDA's kernels are the ones the
    codebase already made FMA-bit-identical for batched prefill — `docs/task-batched-prefill-
    bitidentity.md` — so it may legitimately clear 0.9999 where Metal cannot; do not assume either
    way). Verified only: `gofmt -l` (valid Go syntax, correctly formatted) and a manual re-read of
    every touched line against `resident.go`'s existing `r.aq`/`r.aSc` M=1 decode-path usage.
- 2026-09-08 — G5 row 1 (SmolLM3/`FeatNoPE`) DONE on decoder+Metal, WRITTEN-UNVERIFIED on CUDA
  (same "no CUDA box this session" limitation as G4). `RopeInvFreqLayer` (`decoder/residency.go`)
  returns an all-zero per-layer invFreq table for a NoPE layer (`arch.isNoPELayer`) instead of a
  new kernel path — identity rotation by construction at `invFreq==0` (verified directly against
  the shipped kernel math, `cuda/gemv_fwd.cu` and `metal/kernels.go`'s `rope2`:
  `c=cos(pos·invf)·scale, s=sin(pos·invf)·scale`), holding only where `mscale==1` on the NoPE
  layers — true of every family that sets `layerNoPE` today, documented as a real caveat rather
  than a general claim. `FeatNoPE: true` declared on both `residentBackendFeatures["cuda"]` and
  `["metal"]` (`decoder/features.go`); `admissionGolden["smollm3"]` (`decoder/features_test.go`)
  updated `{} → {"cuda", "metal"}` (SmolLM3's ONLY required feature) and two stale comments
  elsewhere in that file fixed (they still said FeatNoPE was undeclared everywhere).
  - **Real finding, not assumed — the gate this row needed was NOT the resident-vs-CPU cosine
    floor the other rows use.** `testdata/smollm3-tiny` is seeded/synthetic (hidden=64, per
    `scripts/pin_smollm3_tiny.py`), and measured directly: with the real fix (zero only the NoPE
    layer), with the fix reverted (every layer wrongly ropes, including the one that shouldn't),
    and with EVERY layer's invFreq forced to zero (layers 0-2 wrongly skip rope too), the worst
    cosine against the CPU reference over 32 tokens on Metal was 0.9619 / 0.9617 / 0.9624 — a
    ~0.0006 spread across "correct", "wrong one way", and "wrong the other way", meaning a cosine
    floor on this fixture would pass or fail independent of whether the fix is right. Same
    "cannot discriminate a real bug from quantization noise on unstructured weights" finding this
    file's own Mellum-on-Metal note (`features.go`) already recorded for a different family —
    confirmed here first-hand rather than taken on faith from that precedent.
  - **The real gate**: `decoder.TestRopeInvFreqLayer_NoPEIsZero` (`decoder/smollm3_test.go`) — a
    pure, backend-agnostic unit test of the actual changed function (exact zero/non-zero per
    layer against the fixture's `no_rope_layers=[1,1,1,0]` pin), no GPU, no quantization noise.
    PASSES. The resident tests (`metal/smollm3_resident_parity_test.go`'s
    `TestSmolLM3ResidentSmokeMetal`, `cuda/smollm3_resident_smoke_test.go`'s
    `TestSmolLM3ResidentSmokeCUDA`) were downgraded from a numeric floor to a smoke check
    (resident admission + no NaN over 32 tokens) with the finding documented inline, rather than
    shipping a test whose name claims coverage its assertions don't have.
  - Metal smoke test run on real hardware: PASS. Full `metal/...` suite (`-tags
    goinfer_testhooks`) and `staticcheck` both reran clean after. CUDA smoke test written,
    `gofmt`-clean, not run anywhere — same CUDA-box dependency as G4.
- 2026-09-08 — G5 row 2 (Ministral 3/`FeatAttnTemp`) DONE on decoder+Metal+**CUDA**, the first G5
  row with a real CUDA-hardware run. Session had no CUDA box of its own, so this used `ssh nobara`
  into an ISOLATED `git clone` at `/tmp/ministral3-cuda-check` (bundle-transferred, never touching
  the OTHER session's live checkout at `~/mycode/goinfer`, which had its own uncommitted edits to
  `cuda/gemv_fwd.cu`/`cuda/resident.go` mid-test-run at the time — checked first, deliberately not
  gone near). Cleaned up after.
  - **The fix.** `RopeInvFreqLayer`'s twin for a genuinely position-dependent (not per-layer-
    constant) feature: `decoder.Model.AttnTempScale(pos)` (decode, one scalar per call) and
    `AttnTempParams()` (raw beta/origMaxPos, for a batched kernel that must recompute the scale
    PER ROW device-side since position varies within one launch). Folded into the EXISTING rope
    launch's Q output on both backends rather than a new kernel — `rope2`/`rope_kv` gained one
    trailing `qTempScale` parameter (Q-only, post-rotation, guarded so `beta==0` never evaluates
    the `pos/origMaxPos` division that would otherwise poison every non-Ministral3 family's Q
    with NaN), `rope_kv_batched` gained the two raw params and computes it per row.
  - **A real safety finding along the way.** CUDA's kernel is NOT the `.cu` source — it's a
    checked-in, `go:embed`'d PTX binary (`cuda/testdata/*.ptx`) only NVRTC can regenerate, and
    Apple Silicon cannot run any part of the CUDA toolchain at all. Editing the Go call site to
    pass a new argument BEFORE regenerating the PTX would have shipped an argument-count mismatch
    against every family's decode/prefill on real hardware — not "feature missing," a crash or
    memory corruption in `rope_kv`, which every family uses. Surfaced to the user before writing
    any Go wiring; resolved by actually reaching CUDA hardware via nobara rather than shipping
    that risk. `build_ptx.sh` regenerated both `.ptx` artifacts clean via NVRTC
    (`~/cuda-toolkit`), confirmed via a byte-diff (not just "the script exited 0").
  - **A second real finding: two more direct `rope_kv`/`rope_kv_batched` launch call sites this
    session's own search almost missed.** Beyond `resident.go`/`prefill.go`, three TEST files
    construct raw kernel launches with hardcoded argument lists (`rope_partial_test.go`,
    `moe_route_demand_test.go`) and — the one that actually matters for correctness, not just test
    hygiene — `cuda/drafter.go`'s speculative-decode block-drafter path calls `bRopeKV` from TWO
    production call sites with no `decoder.Model` reference available (drafters build from a raw
    geometry). All fixed the same way the file's own existing YaRN-mscale precedent already
    established for exactly this "no model reference" situation: hardcoded `0`/`0` (off), with a
    comment pointing at what a future Ministral-3-shaped drafter would need. `reale2e_test.go`
    carries its own scar tissue for this exact class of bug ("this file's last real drift (rope_kv
    missing its `rhalf` argument)") — also fixed. Found by grepping for every `"rope_kv"` /
    `"rope_kv_batched"` string in the package AFTER the signature change, not by trusting the two
    call sites the initial research turned up.
  - **Real numbers, on real hardware, PASS**: `TestMinistral3ResidentParityCUDA` — 32/32 exact
    argmax, worst cosine 0.999923 across all four `floor(pos/8)` values the fixture's
    `AttnTempOrigMaxPos=8` exercises. Unlike G5 row 1's Metal finding (pure noise-dominance on
    synthetic weights), the control experiment here (fix vs. fix force-disabled) showed the SAME
    ~0.9999 either way — but a direct probe of CUDA's own resident logits (bypassing the CPU
    reference) showed positions with `floor=0` bit-identical between configs (correct — scale is
    exactly 1 there either way) and positions with `floor>0` genuinely, reproducibly DIFFERING
    between configs. Real, small effect; too small for a whole-model cosine floor on this tiny
    fixture to isolate, not absent. Documented inline in the test rather than left as a silent
    footnote. Broader regression: `TestRopePartial`, `TestKernelLocalMemoryCensus`,
    `TestGLMResidentParity`, `TestRopeMscale` all reran clean after the signature change; three
    unrelated failures (`TestPrefillLast_e2e`, `TestSlidingWindowLongContext`,
    `TestGraphsSafeGate`) are a missing `testdata/mistral-tiny-window` fixture in the scratch
    clone, not a code defect — confirmed by the exact error text, not assumed.
  - **Metal**: `resident.setPos(pos)` centralizes `uPos`/`uNKeys`/`uQTempScale` mutation (new
    helper, replacing three separate call sites that each set the first two by hand) — the SAME
    guarded formula as CUDA's host-side computation, since Metal has no per-row batched path for
    this (its batched prefill declines by default regardless, per G4's Metal finding). `rope2`
    gained the same trailing `qTempScale` buffer arg at its one production dispatch site
    (`encodeAttention`) plus two test-helper dispatch sites that share the pipeline. Same Metal-
    specific finding as SmolLM3 (G5 row 1): `testdata/ministral3-tiny` is ALSO seeded/synthetic,
    and the SAME fix-vs-disabled control experiment landed within ~0.0008 cosine either way — so
    the Metal resident test is a smoke check (admission + no NaN), not a numeric floor, exactly
    like row 1. Full `metal/...` suite reran clean after the kernel signature change (which touches
    every family's rope dispatch, not just Ministral 3).
  - `admissionGolden["mistral3"/"ministral3"]` updated `{} → {"cuda", "metal"}`;
    `decoder.TestAttnTempScale_matchesSequentialFormula` is the pure, backend-agnostic gate for the
    formula itself (no GPU, no quantization noise) — the same "prove the actual changed function
    directly" pattern G5 row 1 established.
  - Remaining G5 rows (Olmo 3/`FeatPostOnlyNorm`+`FeatQKNormWhole`, Olmo Hybrid, Command-R/R7B) not
    started.

- 2026-09-08 — G5 row 3 (Olmo 3/`FeatPostOnlyNorm`+`FeatQKNormWhole`, plus Olmo Hybrid getting the
  same two "for free" on top of its already-declared `FeatDeltaNet`+`FeatNoPE`) DONE on
  decoder+Metal+**CUDA**, all four family×backend combinations verified on real hardware (Metal
  locally, CUDA again via the isolated `ssh nobara` clone — now at `/tmp/olmo3-cuda-check`, the
  other session's live `~/mycode/goinfer` checked first and left untouched throughout). Chosen
  over shipping Olmo 3 alone and filing Olmo Hybrid separately, per explicit instruction to fix
  the qGate bug found along the way rather than defer it.
  - **The feature itself.** `FeatPostOnlyNorm`: skip the pre-sublayer norm entirely, normalize only
    after (both backends already had a `sandwich` — pre AND post — code path from another family;
    this widens that same gate to also cover post-only, and forces `fuseQKV` off since the fused
    path assumes a pre-norm exists to fuse into). `FeatQKNormWhole`: QK RMSNorm over the WHOLE
    projection width in one reduction instead of per-head — reused the existing `qk_norm`
    kernel/launch UNCHANGED on both backends, just with different geometry arguments (`1, 1,
    nH*headDim` instead of per-head), so no new kernel and (unlike row 2) no PTX regeneration.
  - **Real bug #1: `Qwen35ResidentParams()` hardcoded `attnGate=true`.** Both backends assumed
    every family sharing the `qwen35Params` struct (Gated-DeltaNet hybrids) used qwen3.5's own
    double-width query-gate scheme on its full-attention layer. Olmo Hybrid's full-attention layer
    is plain (Olmo 3's own scheme, no gate) — the struct had no field to say so. Root-caused from
    Metal's decline message ("qwen35 softmax layer has empty q_norm/k_norm... the loader did not
    populate them") back to the hardcoded `true`. Fixed with a new `qwen35Params.AttnGate bool`
    field (true for qwen3_5/qwen3_5_moe/qwen3_next, false for Olmo Hybrid), consumed by both
    backends' dispatch (`case dnetOK && dnAttnGate:` on CUDA, `else if dnetOK && attnGate` on
    Metal — previously both bare `dnetOK`, silently discarding the gate flag CUDA's tuple already
    returned). The pre-existing plain-attention branch needed zero changes; it already handled the
    Olmo-Hybrid shape correctly once actually reached.
  - **Real bug #2, in this session's OWN earlier G5 row 1 fix: `RopeInvFreqLayer` degrades to
    zero-LENGTH, not zero-VALUE, when a whole model has no RoPE table at all.** Row 1's fix
    (`make([]float32, len(inv))` then zero-fill for a per-layer NoPE flag) assumed `inv` was always
    the real, correctly-sized table for layers that DO use RoPE. Olmo Hybrid sets `rope_theta:
    null`, so `Architecture.finalizeRoPE()` early-returns and NEVER populates any rope table for
    ANY layer — `ropeInvFreq(i)` comes back empty even for layers that are not per-layer-NoPE.
    CUDA has no length guard on the upload and crashed; Metal happened to have an UNRELATED,
    incidental `len(invf) > 0` guard (originally added for GPT-2) that avoided the crash by
    skipping the buffer build entirely, leaving an uninitialized zero-value buffer rather than a
    deliberate all-zero one — working by accident, not by design. Root-caused on nobara with a
    temporary `fmt.Printf` in `cuda/backend.go` (added, used, reverted before any commit) that
    isolated the failure to layer 3 (full-attention), not layers 0–2 (DeltaNet, unaffected). Fixed
    by falling back to `a.rotaryDim()/2` as the table width whenever the real table is empty, so
    the zero-fill is always the correct length; also makes Metal's behavior deliberate instead of
    incidental. No regression: `TestSmolLM3_forwardParity`, `TestGPT2ResidentParityMetal`,
    `TestRopeInvFreqLayer_NoPEIsZero` all rerun byte-identical.
  - **Real bug #3, a REGRESSION in the ALREADY-COMMITTED G5 row 2, found only by running the full
    Metal suite instead of a targeted `-run`.** `metal/rope2_test.go` and `metal/rope2_kv_test.go`
    construct their own `rope2` pipeline and dispatch it with a hardcoded argument list,
    independent of `model.go`'s dispatch helpers — row 2's grep for production call sites never
    found them, so both tests were passing the OLD (pre-`qTempScale`) argument count against the
    NEW kernel and failing (`TestRope2Kv_matchesRope2ThenKv`: `max|diff| 1.157e+00`, not a rounding
    artifact). Fixed by adding the same trailing no-op `qTempScale=1.0` argument at all three call
    sites (two in `rope2_test.go`, one in `rope2_kv_test.go`'s `twoDispatch`). Full `metal/...`
    suite reran clean afterward: 105 real passes, 0 failures, 50 skips (device/fixture-gated).
    **Lesson applied**: after any kernel-signature change, run the FULL test suite, not a targeted
    subset — this is now the second time in this session a targeted check missed a real call site.
  - **Correctness evidence, since neither fixture supports a resident-vs-CPU cosine floor** (both
    `testdata/olmo3-tiny` and `testdata/olmo_hybrid-tiny` are seeded/synthetic, same class as rows
    1–2's Metal findings, and this feature has no new pure-Go formula to unit-test the way
    `FeatNoPE`/`FeatAttnTemp` did): `metal/qknorm_whole_test.go`'s `TestQKNorm_wholeVector` is an
    isolated kernel-geometry proof — compiles `allKernels`, dispatches the real `qk_norm` kernel
    directly with the whole-vector geometry, compares against a precise CPU reference (maxAbs
    ~1.19e-07/2.38e-07, plus a non-vacuousness sanity check) — and both backends' postOnly/
    qkNormWhole changes reuse already-shipped, unmodified kernels with different launch geometry
    rather than new kernel code. `TestOlmo3ResidentSmoke{Metal,CUDA}` and
    `TestOlmoHybridResidentSmoke{Metal,CUDA}` (admission + N-token NaN check, the same smoke
    pattern as rows 1–2's Metal tests) all PASS on real hardware after both bug fixes; each asserts
    the fixture actually requires both features, so the gate can't silently stop meaning anything
    if the fixture changes.
  - **Pre-existing, unrelated, confirmed out of scope**: `TestOlmo3_forwardParity` fails at cosine
    0.98997287 (want ≥0.9999) on the CPU path, on already-committed `main` code with none of this
    session's changes applied (checked via `git stash`) — not introduced here, not fixed here.
  - Full CUDA suite on the isolated nobara clone reran clean after all three fixes: the only
    failures (`TestGraphsSafeGate`, `TestPrefillLast_e2e`, `TestSlidingWindowLongContext`) are the
    same `testdata/mistral-tiny-window` fixture gap row 2 already logged above — confirmed still
    missing on the MAC'S OWN real checkout too (not gitignored-but-present, genuinely absent), so
    this predates both G5 rows and isn't this session's to fix.
  - `admissionGolden["olmo3"]` and `["olmo_hybrid"]` updated `{} → {"cuda", "metal"}`.
  - Remaining G5 row: Command-R/R7B not started.

- 2026-09-08 — G5's LAST row (Cohere/Command-R + Cohere2/Command-R7B, `FeatLayerNorm` +
  `FeatParallelBlock` + `FeatLogitScale` on CUDA; `FeatParallelBlock` + `FeatLogitScale` on Metal,
  which already had `FeatLayerNorm` from GPT-2) DONE on decoder+Metal+**CUDA**, all four
  family×backend combinations verified on real hardware. Largest of the five G5 rows: three
  features, one of them (`FeatLayerNorm` on CUDA) a genuinely new kernel, and the other two a
  real per-layer STRUCTURAL change (a different layer shape, not a kernel parameter) rather than a
  wiring tweak — the pattern every other G5 row was.
  - **The feature itself.** `NormPlacement == NormParallel`: ONE shared input norm feeds attention
    AND MLP INDEPENDENTLY, both summed into a single residual add (`x_final = x_orig + attn_out +
    mlp_out`) — no post-attn or post-MLP norm exists for this family at all (`decoder/weights.go`'s
    cohere tensor schema leaves `PreMLPNorm: ""` on purpose: "parallel: MLP reads the shared input
    norm, no separate pre-MLP norm"). `Model.ParallelBlockResident()` and `Model.LogitScaleResident()`
    (new decoder-level accessors, `decoder/residency.go`) expose this the same way
    `SandwichNormResident`/`PostOnlyNormResident` do for their own placements; `FeatLayerNorm`
    already had `Model.LayerNormResident()` from GPT-2.
  - **The structural insight that made this NOT need a new kernel on either backend.** Both
    backends already compute the pre-attn norm into a live scratch buffer that survives to the end
    of the attention block untouched (CUDA's `r.aq`/`r.aSc` from `segA`; Metal's `r.aq`/`r.aSc`
    from `encodeAttention`'s entry dispatch) — because the MLP's `x_final` formula needs the SAME
    normed value attention already computed, not a fresh norm of the post-attention residual,
    `segBFFN`/`encodeLayer`'s FFN half was changed to REUSE that scratch buffer as the gate‖up
    projection's input directly, skipping its own norm dispatch (and skipping the pre-MLP norm
    weight's build/upload entirely — it doesn't exist for this family). Since quantization is a
    deterministic function of its input with no per-call randomness, reusing the ALREADY-quantized
    shared norm is bit-identical to quantizing it a second time from the same source — there is no
    precision cost to the reuse, only a wiring-correctness question. The attention side's residual
    add is deferred (project into scratch, defer the add) exactly the way `sandwich`/`postOnly`
    already do, reusing that same code shape — CUDA's `normF32`/Metal's guarded `pRmsF32` call
    naturally no-op (CUDA) or are explicitly skipped (Metal, which has no Go-level no-op guard on
    that dispatch) since this family has no post-attn/post-MLP norm weight to apply. Two
    sequential residual adds (attn then MLP) into the same accumulator equal one combined add,
    since addition doesn't care which order two independent terms land in.
  - **CUDA's new kernel**: `layernorm_quant` (`cuda/glue.cu`) — mean-centered LayerNorm+quant,
    structurally `rmsnorm_quant` plus a mean-subtraction pass ahead of the variance pass, same
    warp-shuffle maxabs epilogue. BIAS-FREE ONLY (no `hasBias` parameter at all, unlike Metal's
    kernel): Cohere's LayerNorm carries no learned bias term, and no other family needs one on
    this backend yet. Regenerated PTX via NVRTC on the isolated nobara clone (byte-diff confirmed,
    not just "the script exited 0" — same discipline as G5 row 2's PTX regeneration).
  - **Isolated kernel-level proof, both backends, before any family was declared on the strength
    of it** — the same discipline G5 row 3's `TestQKNorm_wholeVector` established: `TestLayerNormQuant`
    now exists on BOTH backends (Metal's already existed, built for GPT-2, and ALREADY covered the
    bias-free branch explicitly commented "Cohere's hasBias=0 path" — this row needed to write
    only CUDA's twin, `cuda/layernorm_quant_test.go`). Both compare a real dispatch against an
    exact float64 CPU reference at a 0.9999 cosine floor (not exact equality — `rsqrtf` is not
    IEEE-exact, the same reason `quant_vec_exact_test.go`'s own comment gives for why
    `rmsnorm_quant` has no exact-equality gate either): Metal 0.9999887, CUDA 0.9999884 on real
    hardware.
  - **`FeatLogitScale`**: a plain host-side multiply after readback — `applyLogitScale`
    (`cuda/softcap.go`) / `finalizeLogits`'s inline loop (`metal/model.go`), the exact same site
    and shape as `FeatFinalLogitSoftcap`'s existing softcap. Unlike softcap's tanh, a multiply is
    memory- not compute-bound at any vocab size this repo has seen, so it stays a single serial
    pass with no parallel-fan-out threshold to measure or maintain. `LogitScaleResident()` is
    deliberately INDEPENDENT of `GraniteResidentParams()`'s own copy of the same `arch.LogitScale`
    field (Granite's SSM path bundles it alongside `embMul`/`residMul`/`attnScale` for its own
    WebGPU-only resident path) — this is the first consumer reading it generically. Positive
    multiplicative constant, so on-device greedy argmax needs it not, same as softcap's own note.
  - **Resident admission validation, both backends**: `BuildResident` requires a real pre-norm
    weight (`hl.preNorm`/`L.preNorm`, still required non-empty) but must NOT require a pre-MLP norm
    weight for a `parallelBlock` arch (previously one combined check required both together) —
    split into two independent checks on CUDA; Metal's build loop gained one `&& !r.parallelBlock`
    guard on each of the two `L.postNorm`/`L.postNormBias` builds, since `NewBufferFloats`/`r.up32`
    on the family's genuinely-empty `PreMLPNorm` slice is a hard build-time error, not a no-op —
    the exact hazard `postOnly`'s own comment already named for a different empty-tensor case.
    `fuseQKV` forced off for `parallelBlock` on CUDA (for a DIFFERENT reason than `postOnly`: not a
    missing pre-norm weight, but that the fused K1 kernel never materializes the intermediate
    normed activation `segBFFN` needs to reuse).
  - **Correctness evidence, since neither fixture supports a resident-vs-CPU cosine floor**
    (`testdata/cohere-tiny`/`testdata/cohere2-tiny` are "tiny-random" per
    `scripts/pin_cohere_tiny.py`, the same seeded/synthetic class as every other G5 fixture, and
    unlike rows 1–2 this feature has no new pure-Go formula to unit-test): the kernel-level
    `TestLayerNormQuant` proof above, plus `TestCohereResidentSmoke{Metal,CUDA}` and
    `TestCohere2ResidentSmoke{Metal,CUDA}` (admission + N-token NaN check, the same smoke pattern
    every G5 row's Metal test already used) — all four PASS on real hardware. Each also asserts
    `FeatSlidingWindow` presence matches the family (cohere2 interleaves it, cohere does not), so
    the gate stays honest about which shape it's actually exercising.
  - `decoder/residency.go` CPU-forward path is completely UNTOUCHED by this row — `TestCohere_forwardParity`/
    `TestCohere2_forwardParity` (cosine 1.00000000 both) reran unchanged, confirming the resident
    work is additive.
  - **Three stale-test findings, all from EARLIER already-committed G5 rows, surfaced only by
    running the full suite again for this row** (the same lesson G5 row 3 already recorded once
    this session — evidently once was not enough): (1)
    `TestResidentBackendFeatures_noOverclaim`'s hand-maintained per-backend "want" pin lists and
    its `known`-feature whitelist were never updated when rows 2–3 landed `FeatAttnTemp`/
    `FeatPostOnlyNorm`/`FeatQKNormWhole` — fixed alongside this row's own three additions, in the
    same edit, since leaving a known gap while adding a new one would be indefensible. (2)
    `docs/capability-matrix.md`/`.json` and `docs/hardware-matrix.md` were stale from rows 1–3
    too (SmolLM3/Ministral 3/Olmo 3/Olmo Hybrid all still showed CPU-only) — regenerated via each
    doc's own `-update` test, which is the intended mechanism, just never invoked after those
    rows' commits. `pull/capability-matrix.json` needed the same `cp docs/capability-matrix.json
    pull/` resync an earlier, unrelated CI break (`296ab78`) already had to do once this session.
    (3) `testdata/parity_manifest.json`'s deps_hash was stale for every family sharing the shared
    core/CUDA/Metal files this row touched — refreshed via `scripts/refresh_parity_hashes.sh`'s
    underlying mechanism (`go test ./decoder -run TestParityManifest -update`), NOT the wrapper
    script itself: it hard-refuses on ANY forward-golden failure, and `TestOlmo3_forwardParity`'s
    pre-existing failure (logged in this row's own predecessor entry above) trips that refusal
    unconditionally. Independently re-verified via `git stash` that this failure is byte-for-byte
    identical with none of this row's changes applied before proceeding — the same verification
    the wrapper script performs, just done manually since its blanket gate can't distinguish
    "pre-existing" from "caused by you." `validated_at` untouched for every family (confirmed via
    diff: 0 changed lines contain `validated_at`, 66 contain `deps_hash`) — a real re-validation
    moves both, this refresh legitimately moves only one.
  - **G5 is now COMPLETE**: all five rows (SmolLM3, Ministral 3, Olmo 3, Olmo Hybrid, Cohere,
    Cohere2 — six families across five rows) resident on both cuda and metal.

- 2026-09-08 — G6 DONE (WebGPU: the Gemma set, gpt-oss, and staged int4), all in one pass at the
  user's explicit request ("all six features + staged int4"). By far the largest single item in
  this doc: CUDA and Metal already had every one of these; WebGPU had none. Full suites green on
  both `gpu` (93 pass/0 fail/38 skip) and `decoder` (413 pass/0 fail/134 skip, the one exception
  being `TestOlmo3_forwardParity`'s already-logged pre-existing CPU-reference issue, unrelated).
  - **FeatEmbedScale**: zero new code. `decoder/residency.go`'s `embedResident` already applies
    √hidden host-side, generically, before calling `resident.Forward` — this backend just
    consumes the already-scaled embedding like CUDA/Metal do. Pure declare-and-gate.
  - **FeatFinalLogitSoftcap**: a byte-identical port of `cuda/softcap.go`'s `applySoftcap` to
    `gpu/softcap.go`, applied in `residentDecoder.Forward`/`ForwardN` after readback — same site
    and shape as CUDA's `step()`/Metal's `finalizeLogits`. Ported CUDA's own bit-identity +
    mutation-check test suite (`TestApplySoftcap_bitIdentical/_disabled/_mutation`) verbatim
    rather than writing a new one, since the code itself is a verbatim port.
  - **FeatSandwichNorm**: reuses `rmsnormF32` (previously only the f16-Mamba path's own norm) —
    defeats the fused `gemvAdd` residual epilogue at the o-proj and MLP down-proj sites (bare
    `gemv` → norm → separate residual add via `biasAdd`, which is really "`xd += out`"), the same
    shape CUDA/Metal already use for this feature. No new kernel.
  - **FeatGatedGELU**: a genuinely new kernel pair (`gegluShaderWGSL` in `gpu/layer.go`,
    `gegluQuantWGSL` in `gpu/decodefuse.go`) — this backend had no GELU-tanh-gated activation
    before, only SiLU. Clamps the tanh argument to ±15 before calling `tanh`, matching Metal's own
    fix for the exact f32-overflow-before-saturation defect that cost it a real cosine regression
    (0.818→0.994) the first time this math shipped there — applied preemptively here, not found
    the hard way a third time.
  - **FeatOutBias**: composed from two already-existing kernels (a bare `gemv`, then `biasAdd` —
    itself the general "`vec[i] += other[i]`" residual kernel already used for Qwen2's q/k/v
    bias) — no new kernel at all.
  - **FeatAttnSink (gpt-oss), by far the hardest of the six** — comparable in scope to Metal's
    whole gpt-oss port, as scoping research predicted:
    - The sink itself: seeding the online-softmax state (`m`/`l`) with `(sink, 1.0)` instead of
      `(-1e30, 0.0)` before the key loop — an imaginary zero-value key whose score is the sink,
      exactly CUDA's/Metal's own algebra — threaded through ALL FIVE attention WGSL sources
      (`attnShaderWGSL`, `attnKeysShaderWGSL`, `attnF16ShaderWGSL`, `attnI8ShaderWGSL`, and the
      shared `attnWideTemplateWGSL` instantiated 3×) — 7 compiled pipelines total, since gpt-oss's
      tiny fixture's dims default to the `attn-keys` kernel and a real deployment's could hit any
      of the others depending on `--kv-precision`/head_dim.
    - **A real design problem, caught before it shipped wrong**: `hasSink` is a genuinely
      PER-LAYER property (Metal's own `L.attnSinks`/`L.uHasSink` are per-layer for exactly this
      reason), but WebGPU's attention uniform (`P`) is CACHED PER GEOMETRY TUPLE across layers
      (`geomFor`'s dedup, an optimization CUDA/Metal don't have) — baking `hasSink` into `P` would
      have let one layer's flag leak onto another sharing its `{hd,nKV,half,kEqV}` tuple. Fixed by
      adding a genuinely separate per-layer uniform (`struct HS`) rather than widening `P`.
    - Three brand-new MoE kernels (`gpu/moe.go`): `routeGptOssWGSL` (gpt-oss's router disagrees
      with the generic `moeRouteWGSL` about what the bias means — biased logits select AND weight,
      vs "bias steers selection only, weight is the unbiased score" — mirroring
      `cuda/gptoss_act.cu`'s `route_gptoss`/`metal/moe.go`'s twin exactly), `gptossGluQuantWGSL`
      (the clamped interleaved-SwiGLU expert, with a per-expert gate‖up bias table), and
      `moeExpertGptOssDownGEMVWGSL` (the down-combine, with a per-expert down bias — `wgt[slot]*(r
      + bias[idx[slot]*N+row])`, NOT a plain `+= bias[row]`, matching Metal's own documented
      slot-vs-expert-id warning even though this backend's lack of expert paging makes that
      specific confusion structurally moot here).
    - **A build-time wiring bug found by the crash it caused, not by review**: the three new
      kernels' pipelines were never added to `newDecodeRunner`'s `ensures` list (the mechanism
      that lazily compiles each precision's WGSL once, gated on which features a plan actually
      needs) — this crashed `wgpu-native` itself ("invalid bind group layout for bind group
      descriptor", a Rust panic, not a Go error) rather than returning a catchable error, because
      the layout object being bound was never created. Fixed by adding
      `ensureRouteGptOss`/`ensureGptOssGluQuant`/`ensureMoEExpertGptOssDown` to the `m.moe.gptoss`
      branch alongside the existing MoE ensures.
    - **Verified against the REAL, non-seeded `decoder/testdata/gptoss_tiny.gguf`** (not a
      synthetic fixture): resident vs CPU, int4 both sides, minCosine 0.9943 across 8 positions —
      the same 0.95 floor Metal's own gate uses on this exact fixture, not a looser bar invented
      for this backend (`TestGptOssResidentParityWebGPU`).
  - **Gemma set verified against a REAL checkpoint too**: `testdata/gemma3-vl-tiny`'s text tower
    (a real small Gemma3 VL model, not tiny-random) — resident vs CPU, int4 both sides, minCosine
    0.9998 across 8 positions (`TestGemma3ResidentParityWebGPU`). Stronger evidence than every G5
    smoke test this session wrote, which were all seeded/synthetic fixtures by necessity.
  - **A second real bug, same shape as the first**: `runModel.gatedGELU`'s first attempt mirrored
    `decoder.ActKind`'s own ordinal (0=GELU-tanh, 1=SiLU) directly — but every `gpu/*_test.go` file
    that builds a `runModel{}`/`runLayer{}` literal by hand (not through `BuildResident`) leaves
    new fields at their Go zero value, and 0 under that convention silently meant "apply an
    untested new GELU kernel to every existing SiLU family." Caught immediately by the full suite
    (`TestDecodeRunnerW4A8_parity` et al., 6 failures, cosine ~0.9995 not ~1.0) — fixed by
    inverting the field to a bool with SiLU as the zero value, matching the codebase's own
    established convention that a new field's default must be the SAFE, COMMON case. The identical
    root cause recurred for `runLayer.attnSinks` (a `nil` per-layer buffer another 6 hand-built
    tests never set) — fixed the same way in spirit, but since a `*wgpu.Buffer` has no safe
    non-nil zero value, the fix is a fallback AT THE DISPATCH SITE (one shared `noAttnSinks` dummy
    substituted for any `nil` `lw.attnSinks`) rather than a field-default flip.
  - **A THIRD real bug, this one an admission-taxonomy gap, not a wiring bug — caught by an
    EXISTING regression test, not a new one**: satisfying every `ResidentFeature` dense Gemma 4
    nominally requires (identical to Gemma 3's set plus `FeatFinalLogitSoftcap`, both now declared)
    silently ADMITTED it too — `TestGemma4Admission_unconditional`/`TestResidentAdmission_matrix`
    caught it immediately. Dense Gemma 4's local/global attention layers genuinely have DIFFERENT
    head_dim (256 vs `gemma4.GlobalHeadDim` 512), which needs a per-layer geometry seam
    (`runLayer.ghd`/`gnKV`/`ghalf` — the fields exist, CUDA/Metal populate their own twins,
    `gpu/residency.go`'s per-layer builder never has for ANY family). This is NOT expressible as a
    `ResidentFeature` — Gemma 3 (uniform head_dim) and dense Gemma 4 derive the IDENTICAL required
    set otherwise — so `MissingResidentFeatures` structurally cannot catch it, the same class of
    blind spot `residentMoECapacityOK` already exists to patch for MoE expert/group counts. Fixed
    by adding a parallel, backend-scoped predicate (`decoder.residentPerLayerGeomOK` /
    `Model.PerLayerGeomOK`, `cuda`+`metal` declared capable, `webgpu` not) wired into BOTH
    `decoder.ResidentEligible` (the doc-generation/admission-golden predicate) and
    `gpu/residency.go`'s own hand-rolled `BuildResident` admission check (which does NOT call
    `ResidentEligible` and would otherwise still have shipped the crash). Confirmed the fix by
    re-running the exact `gemma4-dense-twogeom-tiny` load that previously crashed
    `wgpu-native` ("unsupported projection precision \"\"") — now a clean, named decline instead.
  - `decoder/features_test.go`'s `admissionGolden` for `gemma3`/`gemma3_text` now correctly says
    `{cuda,metal,webgpu}`; `gemma4`/`gemma4_text`/`gemma4_unified_text` ALSO say `{cuda,metal,webgpu}`
    in that table specifically because it's the simplified FEATURE-LIST-ONLY model (documented
    inline, same shape as the pre-existing deepseek_v2/kimi_k2 MoE-cap precedent) — the REAL
    runtime (`ResidentEligible`, `TestGemma4Admission_unconditional`) correctly still declines
    Gemma 4 on webgpu. `gpt_oss` golden updated to `{cuda,metal,webgpu}` for real, no caveat.
    `decoder/gptoss_decline_test.go` (kept its historical name and file, per its own established
    precedent for CUDA's 2026-08-31 promotion) moved webgpu from the decline assertions to the
    admit ones alongside metal/cuda.
  - **Staged int4** (the separate, independently-scoped item): `decoder/weightmat.go`'s
    `matmulInto` AND its sibling `matmul` (the non-Workspace variant used by e.g. Gemma 4's own
    forward — fixed for consistency, not just the doc's literally-named function) never consulted
    a backend for int4 weights on the staged (non-resident) path, unlike int8, which already did
    via `QuantBackend`. Added `decoder.QuantBackend4` (int4's `QuantBackend` twin) and
    `webgpuBackend.MatmulW4A8` — M=1 (decode) only, mirroring `MatmulW8A8`'s residency-cache
    pattern exactly (keyed by the packed-nibble slice's pointer), M>1 (prefill) declines cleanly
    since this backend has no int4 tiled/GEMM kernel yet. Required generalizing `GEMVRunner` from
    a `*ResidentW8A8`-specific type to the existing `decodeWeight` interface (W8A8 or W4A8) — every
    piece needed (the WGSL kernel, the upload paths, the interface) already existed from the
    RESIDENT path; only the STAGED path's `GEMVRunner`/`webgpuBackend` never used them. Verified
    against the CPU reference (`linalg.MatmulBTW4A8Into`) at K=517 (deliberately not a multiple of
    32, to exercise both the fallback unpack-and-repack upload path and `GEMVRunner`'s zero-tail-
    padding invariant): cosine 1.000000 (`TestWebGPUBackend_MatmulW4A8_matchesCPU`). Also newly
    establishes a direct-call test pattern `MatmulW8A8` itself never had.
  - Also fixed along the way (found by re-running the doc-generation tests, not by design):
    `docs/hardware-matrix.md` regenerated (Gemma 3 and gpt-oss now show `✅ resident` under
    WebGPU; Gemma 4 correctly still shows `CPU`); `testdata/parity_manifest.json`'s deps_hash
    refreshed for every family sharing the shared core/decoder files this row touched (same
    goldens-verified-first discipline as every G5 row, `TestOlmo3_forwardParity`'s pre-existing
    failure independently re-confirmed unrelated via `git stash` before proceeding);
    `TestResidentBackendFeatures_noOverclaim`'s webgpu pin list updated to the new 19-feature set.
  - **G6 is now COMPLETE.** Remaining items in this doc: G3 (LoRA on resident path), G7-G11, and
    G2 (handed to the Linux box session).

- 2026-09-08 — G10 DONE (Metal's `int8int8` silently runs as int4; the report lied about it).
  `metal/model.go`'s `int4Buf` re-quantizes ANY int8-kind weight through the W4A8 packer
  (dequant-then-repack) because Metal has no int8 GEMV kernel at all — int4 IS the resident
  precision on Metal whether loaded directly (`--quant int4`, cheaper — no dequant-then-repack) or
  arrived at via `int8int8`. `decoder.Model.DecodePath()` used to just echo the requested quant
  string back, so `metal-resident (int8int8)` claimed a precision this backend never runs. Fixed
  with a small pure helper, `decoder.residentQuantLabel(backend, quant string) string` — passes
  every (backend, quant) pair through unchanged except `("metal", "int8int8")`, which becomes
  `"int8int8→int4, no Metal int8 GEMV kernel"` — wired into `DecodePath()`'s resident-case
  `Sprintf`. Also corrected two stale/wrong claims this same investigation turned up:
  `metal/backend.go`'s `BuildResident` doc comment (previously implied int8 gets a real kernel and
  int4 declines to CPU on Metal — both false) and `docs/quantization.md`'s `int8int8 (W8A8)`
  section (same two false claims, now explains the silent re-quantization and points at the new
  `DecodePath()` string). `TestResidentQuantLabel` (`decoder/staged_device_note_test.go`) pins the
  full table (metal×{int8int8,int4,int8,f32}, plus int8int8 on cuda/webgpu/cpu as pure-passthrough
  controls). Verified live on real Metal hardware via a throwaway probe:
  `DecodePath: metal-resident (int8int8→int4, no Metal int8 GEMV kernel)`. Full `decoder`+`metal`
  suites clean (only `TestOlmo3_forwardParity`, pre-existing/unrelated). Committed alongside G3's
  first commit below rather than standalone, since both were accepted together
  ("G3 as the main pick, G10's reporting fix as a trivial quick win alongside it").

- 2026-09-08 — G3 IN PROGRESS: decoder-level plumbing done and seam-tested; no backend
  implementation yet.
  - **Why "merge into resident weights at bind time" (this doc's own listed alternative) is
    architecturally wrong**, found while scoping: `internal/serveapp/main.go`'s
    `lm.sessions.adapter` design shares ONE base `*decoder.Model` — and its ONE `m.resident`
    object — across every `--adapter` served name. Merging would need either N resident copies (one
    per adapter, defeating the whole point of a shared resident base) or unsafe concurrent mutation
    of one shared weight buffer. The "extra GEMV pair per adapted projection, applied additively at
    call time" fix this doc already named is the one that actually fits: additive means each call
    computes its own delta and touches no shared state permanently, and resident access is already
    serialized end-to-end by `resBusy`'s CAS (one bind→forward→clear sequence in flight at a time,
    across however many adapter sessions share the base).
  - **Decoder plumbing** (`decoder/residency.go`): a new OPTIONAL `ResidentForward` extension,
    `ResidentAdapter` (one method, `SetAdapter(layers []ResidentAdapterLayer) error`, `nil` clears),
    matching the established optional-capability pattern (`ResidentHiddenLast`, `PrefillPathReporter`).
    `ResidentAdapterProj`/`ResidentAdapterLayer` are exported twins of the package-private
    `loraDelta`/per-layer struct `decoder/lora.go` already has — needed because a resident backend
    lives in another module and can't see unexported fields — with `residentAdapterProj`/
    `residentAdapterLayers` converting one to the other (nil ⇒ nil, matching `loraLayerDelta`'s own
    untargeted-projection convention exactly, so a backend's dispatch loop is a plain nil-check per
    projection).
  - **Gate widening** (`decoder/model.go`, `generateInto`): `useGPU` was `resident != nil &&
    prefillFrom == 0 && commit == nil` — which is exactly what made every session-driven request
    (including adapter ones — `Session.Generate` always sets `commit`) drop to CPU, this item's
    whole premise. Widened to also admit `commit != nil` when `cache.lora != nil` (an adapter
    session) AND the resident backend implements `ResidentAdapter` — a plain (non-adapter) session
    still declines exactly as before, since prefix-reuse's CPU-side cache and the resident's GPU-side
    KV still can't both be the source of truth for a reused prefix (the pre-existing reason session
    generations avoid resident at all). The bind/clear itself happens INSIDE the existing `resBusy`
    CAS block, wrapping the whole generation: `SetAdapter(layers)` right after winning the CAS,
    `SetAdapter(nil)` via the same `defer` that releases `resBusy` — so a bind failure falls back to
    CPU cleanly (releases `resBusy`, does not proceed resident with no delta bound) and two adapter
    sessions sharing one base's resident can never observe each other's bound delta, for the same
    reason two plain generations can't race today.
  - **Seam test** (`decoder/resident_adapter_seam_test.go`, new): the G4 fake-backend pattern
    (`resident_embed_seam_test.go`) applied to this gate. `resident_seam_test.go`'s own GGUF-based
    `tinyFixture` (glm-tiny.gguf) turned out unusable here — GLM is MoE-shaped and GGUF-based, and
    `Model.LoadAdapter` rejects both — so the fixture is a small in-memory synthetic llama
    (safetensors, 2 layers) plus a PEFT adapter targeting q/v/gate/down, following
    `TestLoRACompute_forwardParity`'s own construction exactly (factored into a new
    `buildLoRAFixture` helper since that test's version is inline and this needs the raw
    directories, not a pre-built `*Model`). Two tests: `TestSeam_AdapterSessionBindsAndClearsResidentAdapter`
    (a fake `ResidentAdapter`-capable backend must see exactly one bind before decode with the
    right per-layer deltas — asserted per-projection, confirming untargeted k/o/up stayed nil — and
    one matching clear after, `binds == clears`, decode actually ran on the resident `Forward`) and
    `TestSeam_AdapterSessionDeclinesToCPUWithoutResidentAdapter` (a resident backend WITHOUT
    `ResidentAdapter` — reusing the existing plain `fakeResidencyBackend` — must never call its
    resident `Forward` for an adapter session, and the CPU-fallback output must match a direct
    `prefillLogits` call with the same adapter bound). Both pass.
  - **`testdata/parity_manifest.json` deps_hash refresh**: `decoder/model.go` is a `core` file, so
    every family sharing it went stale. Ran the goldens-first check by hand
    (`scripts/refresh_parity_hashes.sh` refuses unconditionally on ANY forward-golden failure, and
    `TestOlmo3_forwardParity` still fails, pre-existing/unrelated per every prior row this session)
    — re-confirmed via `git stash` that the failure is byte-identical with none of G3's changes
    applied, then ran the script's own underlying mechanism directly
    (`go test ./decoder -run TestParityManifest -update`), same as every prior row this session.
    33 families' deps_hash refreshed, 0 `validated_at` lines touched (confirmed via diff).
  - Full regression: `decoder` 416 pass/1 fail(`TestOlmo3_forwardParity`, pre-existing)/134 skip;
    `metal` 107 pass/0 fail. `gofmt -l`, `go vet` (plain and `-tags realckpt`), and CI's pinned
    staticcheck (`-tags goinfer_testhooks`, since the U1000s without it are pre-existing
    testhook-only helpers) all clean.
- 2026-09-08 — G3's Metal backend DONE (the decoder plumbing above now has a real implementation
  and a numeric parity gate). No commit yet.
  - **Two new kernels** (`metal/kernels.go`): `lora_delta_down` (ONE threadgroup, loops over rank
    r serially, standard tree-reduction of `A[r,:]·dequant(aq,asc)` into a small `t[R]` scratch
    buffer — same shape as `rmsnorm_quant`'s own reduction) and `lora_delta_up` (one thread per
    output row, `out[row] += scale·Σ_r B[row,r]·t[r]`, dispatched with an exact non-uniform grid
    so no bounds guard is needed). `t` (`r.loraT`) is shared across every projection/layer within
    one command buffer — safe because dispatches within one `MTLComputeCommandEncoder` observe
    each other's writes, the same guarantee every other multi-stage GEMV in this file already
    relies on (rmsnorm→GEMV, GEMV→SwiGLU, etc).
  - **New file `metal/lora.go`**: `residLoRAProj`/`residLoRALayer` (device-resident A/B buffers +
    uniforms per projection per layer), `resident.SetAdapter` (releases whatever was previously
    bound FIRST, then uploads the new deltas or leaves `r.loraLayers` nil — matching prefill.go's
    C5 fix for the same class of per-call-buffer leak: a resident backend's `BuildResident`-time
    buffers are tracked for `Close`, but a bind/rebind buffer created OUTSIDE that isn't, unless
    released explicitly), `applyResidentLoRA` (the shared down-then-up dispatch helper, no-op when
    the target projection's delta is nil). `metalResident.SetAdapter` (`metal/backend.go`) is a
    thin forward to `resident.SetAdapter`, with the usual compile-time
    `_ decoder.ResidentAdapter = (*metalResident)(nil)` seam.
  - **Dispatch sites**: `encodeAttention` (q/k/v into their `r.qkv` slots, right after the base
    fused QKV GEMV; o-proj into whichever buffer the base o-proj GEMV just wrote — `r.oO` for
    sandwich/postOnly/parallelBlock BEFORE their post-attn norm, `r.x` directly otherwise) and
    `encodeLayer` (gate/up into `r.gu` BEFORE the SwiGLU activation — matching
    `decoder/mlp.go`'s own comment, "delta into gate/up before the activation", word for word;
    down into `r.dO` or `r.x` by the same before-any-subsequent-norm rule). Every site takes the
    SAME quantized activation buffer (`aq`/`aSc`, `cq`/`cSc`, or `mq`/`mSc`/`dq`/`dSc`) the base
    projection it rides beside already consumed — matching `applyLoRA`'s CPU reference exactly,
    which takes the identical input the base matmul does.
  - **Known, documented gap, not a silent one**: the qGate branch (Qwen3.5-style fused
    double-width q_proj+gate) in `encodeAttention` is NOT wired — `Model.LoadAdapter` doesn't
    exclude qGate families by name, only by its three structural checks (own-forward/MoE/non-gated
    MLP), so an adapter loaded against a qGate family would silently apply no q/k/v delta on Metal
    today. No family with qGate=true is known to pass `LoadAdapter`'s checks currently; recorded in
    `metal/lora.go`'s file comment as the gap to close first if one arrives.
  - **A real bug, caught by the parity test's own vacuousness check failing, not by inspection**:
    `lora_delta_down`'s first version declared its reduction scratch as a DYNAMIC
    `threadgroup float* red[[threadgroup(0)]]` PARAMETER but was dispatched via `Dispatch` (not
    `DispatchTG`, which is what actually calls `setThreadgroupMemoryLength:atIndex:`) — so the
    buffer's length was never set. This didn't crash or NaN; it silently read/wrote a
    zero-length/undefined threadgroup allocation, producing a small-but-wrong, self-consistent
    delta (first measured cosine against CPU: 0.9647 — inside the 0.95 floor, so a NUMERIC-ONLY
    gate would have shipped this). Fixed by switching to a STATIC `threadgroup float red[256]`
    declared inside the kernel body, matching `rmsnorm_quant`'s own convention exactly (and
    avoiding the whole `Dispatch`-vs-`DispatchTG` footgun rather than just fixing this one call
    site). Cosine after the fix: 0.999960.
  - **A second issue, this one in the TEST, not production**: the first parity-test draft compared
    `got` (with-adapter logits) against `gotNoAdapter` (without-adapter) and found them
    IDENTICAL — looked exactly like the adapter silently no-op'ing (the G6 zero-value bug class).
    Root cause was the test, not the kernels: `metalResident.Forward`'s own doc says its returned
    slice "is reused across calls" (aliases resident-owned storage), and the test's first loop
    just captured the returned slice directly (`got = lr`) instead of copying it — so the SECOND
    loop's calls silently overwrote `got`'s backing array in place, making the two comparisons
    trivially identical regardless of what the kernels did. `gpt2_resident_parity_test.go` already
    has the right pattern (`append([]float32(nil), lr...)`) for exactly this reason; missed it on
    the first pass. Fixed by copying at the end of each run.
  - **`TestLoRAResidentParityMetal`** (new, `metal/lora_resident_parity_test.go`): drives
    `decoder.ResidentAdapter` directly via `ResidentForwardForTest`/`ResidentAdapterLayersForTest`
    (two new CPU-exported test hooks in `decoder/fidelity_testhook.go` —
    `PrefillLogitsWithAdapterForTest` and `ResidentAdapterLayersForTest`, since `KVCache.lora` and
    `Model.adapter` are both unexported and this test lives outside package `decoder`), isolating
    the KERNEL correctness question the way `hiddenlast_resident_parity_test.go` isolates G4's —
    the WIRING question is `decoder/resident_adapter_seam_test.go`'s job, not this test's. Uses
    `testdata/llama-tiny` (a real committed safetensors checkpoint, GQA 4-heads/2-kv-heads so
    o-proj and the differently-widthed q/k/v sites are all genuinely exercised) plus a synthetic
    PEFT adapter targeting all seven projections across all four layers, at `--quant int4` (not
    int8int8 — G10 already established int8int8 silently becomes int4 numerics on Metal anyway).
    Gates at cosine ≥ 0.95 against the CPU compute-time-LoRA reference — the SAME floor
    `gpt2_resident_parity_test.go` established for whole-model resident-vs-CPU decode logits, not
    a bar invented for this test — plus the vacuousness check above. Measured: cosine 0.999960,
    argmax match, adapter genuinely non-vacuous (CPU with/without-adapter cosine was -0.0485 on
    this fixture, confirming the adapter has a real, large, correctly-reproduced effect, not a
    coincidentally-tiny one the parity bar couldn't have caught).
  - Full regression: `decoder` 416 pass/1 fail (`TestOlmo3_forwardParity`, pre-existing)/134 skip;
    `metal` 108 pass/0 fail (107 prior + this row's new test). `gofmt -l` and `go vet` (plain and
    `-tags goinfer_testhooks`) clean; CI's pinned staticcheck (`-tags goinfer_testhooks`) clean.
  - **Still open**: CUDA and WebGPU implementations (Metal-first precedent, same as G4's
    HiddenLast — CUDA/WebGPU are the natural next continuation once Metal is verified, not done in
    this pass). No commit yet.

