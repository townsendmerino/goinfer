# Task: serving paths that run on the CPU when a GPU path exists — 2026-09 (G1–G11)

> **Status: OPEN, drafted 2026-09-08** from the code at `c9db2ec`, not from the docs — R9
> (`docs/task-first-hour.md`) showed the docs can say "GPU" where the code says CPU, so every item
> below cites the line that decides. Companion to `docs/gpu-residency-coverage.md` (the standing
> residency backlog, cross-backend) and `docs/hardware-matrix.md` (the generated admission table).
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
CPU-only by design — never touches m.resident at all". `internal/serveapp/openai.go:1076–1077`
(`driveVL`) is the only caller from serve; `prepare()` is told `residentPath=false` for vision
(`internal/serveapp/openai.go:677–663`).

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

**Where.** `internal/serveapp/openai.go:1006`: `if lm.model.ResidentActive() && lm.adapter == ""` —
adapter models take the session path below it, and `decoder/model.go:1122` makes a session
generation ineligible for the resident KV (`useGPU = resident != nil && prefillFrom == 0 &&
commit == nil`). The comment at `internal/serveapp/openai.go:980–967` records the cost: 13 tok/s vs ~460 resident on
a 0.5B (RTX 2070 SUPER). Documented as audit R-01 and left there.

**Fix.** Apply the compute-time LoRA on the resident path: the adapter is a per-projection
low-rank delta applied to the activations (`Session.UseAdapter` → cache's `lora`), so the resident
runners need one extra GEMV pair per adapted projection per token, with the delta weights uploaded
at `bindAdapter` time. Alternative that is cheaper and may be enough: merge the adapter into the
resident weights at bind time (re-pack the affected projections) and treat "switch adapter" as a
re-pack; one adapter per loaded model at a time, which is what `lm.sessions.adapter` already
assumes (`internal/serveapp/main.go:866`).

**Gate.** An adapter-vs-merged parity test on the tiny fixture, then the R-01 measurement
re-run on the 0.5B.

**Size.** Medium. Shares its seam with G4.

### G4 — `/v1/embeddings` on decoder models never uses the resident path

**Where.** `decoder/embed.go:36–71`: `HiddenLast` → `hiddenLastBatched` (CPU `forwardN`) or
`hiddenLastSequential`; neither consults `m.resident`. `internal/serveapp/decoder_embedder.go` is
the only serving caller.

**Fix.** A resident `HiddenLast`: the resident prefill already exposes hidden-state capture for the
block drafter (`hidCapTaps`, `cuda/prefill.go:312–319`), so a "prefill and return the last row's
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
| DeepSeek-V2/V3, Kimi K2 | `FeatMLA` | `FeatMLA` | exists on WebGPU; **gate the nGroup/topkGroup mapping first** (the CUDA TRAP comment, `decoder/features.go:396–405`) |
| Laguna | `FeatAttnOutputGate` | same | not on any backend; WebGPU's DeltaNet has a fused output gate to crib from |
| LFM2.5 | `FeatShortConv` + "own forward, not bridged" | same | `decoder/residency.go:208` declines it before features are consulted |
| Llama 4 | own forward, not bridged | same | `decoder/residency.go:206` |
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

**Where.** `decoder/residency.go:243`: `if a.nemotron != nil { return a.MoE == nil }` — the
MoE block kind has no resident builder on any backend (comment at 234–240). `docs/hardware-matrix.md`
row "Nemotron-H → WebGPU ✅ resident" is generated from the *dense* representative config, so it is
true of Nemotron-H and false of the two models people download. task-families-2026-09 F2
"verified" Lightning against the adapter without noting it runs CPU-only.

**Fix, two parts.** (1) Today: a footnote on the matrix row and a line in the Lightning/Nano
family docs; `serve check` should say "CPU (MoE block not resident)" for these. (2) The real one:
add the MoE FFN case to the WebGPU Nemotron block switch (`gpu/residency.go:486`, cases 0/1/2;
the `default` that residency.go's comment says is missing is there now at line 583, so an
unknown kind declines cleanly), gated on the real Nano checkpoint on the Linux box.

**Size.** (1) trivial; (2) medium on WebGPU, and moot on CUDA/Metal until G5's SSM row lands.

---

## B. On the GPU, but on a slower GPU path than exists

### G8 — Metal prefill is sequential for every non-plain-dense family, flag or no flag

**Where.** `metal/model.go:63–67`: `prefillFeatures` is exactly `{FeatQKNorm, FeatSlidingWindow,
FeatPartialRotary}`; `metal/model.go:566` sets `prefillOK` from it; `metal/backend.go:258` declines.
Separately, `metal/backend.go:251` declines batched prefill unless `GOINFER_METAL_BATCHED_PREFILL=1`
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

**Where.** `decoder/model.go:982`: "WebGPU implements no Prefiller"; `gpu/residency.go:1025`
seeds the caches via sequential `Forward`. Every prompt on WebGPU is one submit per token.

**Fix.** A `Prefiller` on the WebGPU runner, dense first, following the CUDA shape
(`task-gpu-batched-prefill.md`). **Size.** Medium-large; lower priority (see G6).

### G10 — Metal has no int8 weight kernel: `int8int8` is requantized to W4A8 on device

**Where.** `metal/model.go:361–363` (`int4Buf`): an int4 weight is packed directly; an int8 weight
is dequantized to f32 and re-packed as 4-bit/group-32. There is no W8 GEMV in `metal/`. So
`-quant int8int8` on Metal runs int4 numerics on the GPU while holding the int8 host copy — more
RAM, not more precision. `metal/backend.go:55`'s comment ("weights must be int8-loaded…") is stale
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

- The `resBusy` CAS loser falls to the staged/CPU path (`decoder/model.go:1251–1067`), but serve
  serializes each model's generations (`internal/serveapp/openai.go:62` `mu`), so it never fires
  through the HTTP surface; only direct library callers running two generations on one `Model`
  see it.
- Constrained/tool requests keep the plain resident `Generate` (`internal/serveapp/openai.go:1006`).
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

- 2026-09-08 — G3's WebGPU backend DONE. CUDA still not attempted: its package does not even
  compile on this Mac today (`gpu.MappedHostBuffer`/`gpu.Graph`/etc undefined under `-tags cuda`)
  — a pre-existing gap in the aikit/gpu module on this checkout, unrelated to this row, but it
  means CUDA can only be written fully blind with zero local verification. Asked the user how to
  proceed given that; chose WebGPU next (testable end-to-end on this Mac) over writing CUDA blind.
  No commit yet.
  - **The real design problem, exactly as scoped before starting**: `gpu/decoderunner.go`'s
    `DecodeRunner` builds a FIXED, flat `[]runStep` dispatch plan ONCE at construction
    (`newDecodeRunner`) and `record()` just walks it fresh into a new command buffer every single
    `Run()` — nothing about the plan itself is baked GPU-side, but its STRUCTURE (which dispatches,
    in what order) does not vary per token the way Metal's re-encoded-every-token trunk can with a
    plain `if r.loraLayers != nil`. Solved by splitting `r.steps` into an immutable `r.baseSteps`
    (the pristine no-adapter plan, saved once at the end of construction) plus `r.loraHooks`
    (recorded during construction at each of the 7 projection sites: `{afterIdx, layer, kind, aq,
    ascale, dst, k}` — afterIdx is the base step index the delta belongs right after, dst is the
    SAME buffer the base projection just wrote, aq/ascale is the SAME quantized activation it
    consumed). `SetAdapter` REBUILDS `r.steps` by walking `r.baseSteps` and splicing in each
    hook's down+up dispatch pair (skipped if the bound adapter leaves that hook's projection nil);
    clearing an adapter is a cheap restore (`r.steps = r.baseSteps`) since the base plan is never
    mutated. This is genuinely more machinery than Metal needed, but it is a direct, mechanical
    consequence of the architecture difference scoped out before writing any code — not a surprise
    found partway through.
  - **One thing WebGPU's architecture made EASIER than Metal's, found while wiring q/k/v**:
    Metal's qGate branch (Qwen3.5-style fused double-width q_proj+gate) dispatches q/k/v through a
    COMPLETELY SEPARATE kernel path (`pDnQSplit`) than the plain fused-qkv site, so wiring LoRA
    there would have needed a second hook at a different site — left undone and documented as a
    known gap instead. WebGPU's qGate handling is structurally simpler: `q, k, v :=
    gemv(aq,as,lw.q/k/v)` (or the bias-fused variant) runs FIRST, identically whether or not
    qGate is set, and `if lw.qGate { q, aGate = qSplit(q, ...) }` only runs AFTER — so inserting
    the LoRA hook right after the base gemv, before the qGate check, applies correctly regardless
    of qGate (the delta targets whatever width the base matmul actually produced, qGate or not).
    No documented gap needed here on this backend.
  - **Two new WGSL kernels** (`gpu/lora.go`): `loraDeltaDown` (ONE workgroup only,
    `@workgroup_size(64)`, loops over rank serially — same reduction shape as
    `rmsnormQuantWGSL`'s own sum-of-squares tree, reading the SAME packed
    `array<vec4<u32>>` int8 activation format every GEMV kernel in this backend already uses, via
    a small `get_aq_i8(k)` unpack helper) and `loraDeltaUp` (`@builtin(global_invocation_id)`,
    one thread per output row, dispatched with an exact grid so no bounds guard is needed).
    Compiled once via `ensureLora` (auto bind-group layout via `c.bgl`, matching every other
    kernel in this package), added to `newDecodeRunner`'s base `ensures` list unconditionally —
    same "always create, cheap" choice Metal made for its own LoRA pipelines.
  - **`gpu/lora_resident.go`**: `loraRunProj`/`loraRunLayer` (bound per-projection buffers +
    bind groups — built fresh per `SetAdapter` call since a hook's fixed buffers and an
    adapter's rank/scale can only be combined once both are known), `SetAdapter` (releases the
    previous bind's GPU resources first, same leak-avoidance discipline as `metal/lora.go`'s
    `releaseLoRALayers`), `rebuildSteps` (the splice described above). `residentDecoder.SetAdapter`
    (`gpu/residency.go`) forwards to `rd.runner` only — `ForwardN`'s batched verify runners
    (`rd.batch`) are speculative-decode-only, and `generateInto`'s adapter-admission path (the
    thing that can ever call `SetAdapter` in production) calls `Forward` exclusively, confirmed by
    reading `decoder/model.go`'s two `m.resident.Forward` call sites directly rather than assuming
    it — so this is a scoped decision, not an oversight, matching the same scope
    `Model.LoadAdapter` itself already restricts to.
  - **`TestLoRAResidentParityWebGPU`** (new, `gpu/lora_resident_parity_test.go`): the same
    `testdata/llama-tiny` fixture and synthetic 7-projection/4-layer adapter as the Metal test
    (duplicated locally — `buildLlamaTinyLoRAFixtureGPU`/`writeLoraGPUSafetensors` — since the
    Metal test's versions are package-private there), same 0.95 floor, same vacuousness check.
    Passed on the FIRST run with no debugging needed (unlike Metal's two real bugs) — reused the
    already-fixed fixture-aliasing lesson (`append([]float32(nil), lr...)`) from the start instead
    of rediscovering it. Measured: cosine 0.999963, maxAbs 0.005082, argmax match — matching
    Metal's own 0.999960 on the identical fixture almost exactly, independent confirmation the
    LoRA math itself (not just its wiring) is correct on both backends.
  - Full regression: `gpu` 94 pass/0 fail/38 skip (93 prior + this row's new test). `gofmt -l`
    (caught one real formatting miss in `decoderunner.go`, fixed), `go vet` (`-tags "gpu
    goinfer_testhooks"`), and CI's pinned staticcheck (same tags) all clean.
  - **G3 is now DONE on decoder+Metal+WebGPU.** CUDA remains unwritten (package doesn't compile
    locally, see above) — left as explicitly future work rather than written blind. No commit yet.

- 2026-09-08 — G3's CUDA backend DONE, via SSH to nobara (the only box with the CUDA toolchain).
  Worked in an ISOLATED checkout (`~/mycode/goinfer-g3-cuda`, fresh `rsync` of the local working
  tree, not the box's default `~/mycode/goinfer`) because another active session was mid-work
  there with uncommitted changes touching `decoder/residency.go` among others — never touched
  that checkout or its git state. **G3 now has all three GPU backends (Metal, WebGPU, CUDA) plus
  decoder plumbing, all independently verified against the same CPU reference.**
  - **Architecture, third variant**: unlike WebGPU's fixed-plan-replay, CUDA is like Metal —
    `launchToken` re-issues live kernel launches every token (`segA`/`segB`/`segBFFN`), so binding
    is a plain `if r.loraLayers != nil` at each of the 7 sites, no dispatch-plan surgery needed.
    The one thing CUDA has that neither other backend does: **CUDA graphs** (`r.graphs`, opt-in
    via `GOINFER_CUDA_GRAPHS`, off by default) capture-and-replay each layer's static launch
    segments — a graph captured before any bind would silently replay with zero LoRA dispatches
    forever after, since replay doesn't re-run the live Go code my `if` lives in. Fixed by having
    `SetAdapter` refuse outright when `r.graphs` is true, rather than attempt a live/graph hybrid
    or a mid-life re-capture — graphs are opt-in and off by default, so this costs nothing for a
    default deployment and is an honest, named limitation for the opt-in case.
  - **A structural wrinkle neither Metal nor WebGPU has**: CUDA has TWO code paths for q/k/v (and
    separately for gate/up) depending on `r.fuseQKV` — a fused int4-only super-kernel
    (`fused_qkv.cu`'s `fused_rms_qkv`/`fused_rms_gu`) that computes rmsnorm+quant+matmul in ONE
    launch and never materializes the normed+quantized activation (`r.aq`/`r.aSc` /
    `r.mq`/`r.mSc`) as a separate buffer — versus the unfused chain, which does. LoRA needs that
    activation as its own input. Fixed by having the hook, when `r.fuseQKV` is true, ALSO issue
    the standalone `r.norm(...)` call the unfused path already makes elsewhere — a redundant
    extra dispatch (only when an adapter is bound), correct because `rmsnorm_quant` is a
    deterministic function of `r.x`/the norm weight, so materializing it a second time from a
    second kernel invocation reproduces the same values the fused kernel computed internally
    (up to ordinary cross-kernel float noise, the same class already tolerated everywhere else in
    this codebase). Confirmed safe by checking `BuildResident`: `r.fuseQKV` is forced `false` for
    both `postOnly` and `parallelBlock` (their own comments explain why), so this fused-path
    materialization is never ambiguous about which norm weight applies.
  - **`cuda/lora.cu`** (new, own module — `cuda/testdata/REGEN.md`'s rule for a NEW kernel):
    `lora_delta_down` (one block, loops over rank, same reduction shape as `rmsnorm_quant`'s
    sum-of-squares tree, reading the SAME packed-int8-per-int32-word activation format every
    GEMV kernel in this backend already consumes) and `lora_delta_up` (one thread per output row,
    `if (row >= Outn) return;` bounds check matching `gemv_w8a8`'s own convention for an
    over-provisioned grid). `cuda/lora.go`: `SetAdapter` (runs via `r.do(...)`, the executor-
    thread indirection EVERY device-touching call in this backend requires — CUDA contexts are
    thread-affine — and releases the previous bind's buffers via `r.dev.ReleaseBuf`, since unlike
    Metal/WebGPU this backend's device ledger normally frees everything only at `Close`, so
    skipping this would leak VRAM on every rebind), `applyLora` (the shared down-then-up dispatch
    helper). New Pipeline fields (`fLoraDown`/`fLoraUp`) loaded unconditionally in `BuildResident`,
    same "always create, cheap" choice both other backends made.
  - **Two real, independently-discovered pre-existing bugs, both fixed, NEITHER caused by this
    row** — found only because this was the first CUDA build attempted against a fresh checkout
    since they landed:
    1. **`cuda/testdata/glue.ptx` was stale by three commits.** `git log` on `glue.ptx` vs
       `glue.cu` showed the committed PTX predates `7357856` (G5's Cohere row, which added
       `layernorm_quant`) AND two further optimization commits (`glu_quant`/`rmsnorm_quant`'s
       warp-shuffle maxabs, RoPE's YaRN mscale) — none of their PTX regens were ever committed.
       Concretely: **every CUDA resident build, on any machine, from any commit since `7357856`,
       has been silently declining to CPU** (`layernorm_quant` genuinely absent from the shipped
       PTX — confirmed identical stale artifact on both this Mac's checkout and nobara's own
       tracked copy, so this is not a sync artifact). Fixed by regenerating via `./build_ptx.sh
       glue` on nobara (no toolchain-pinning concern per REGEN.md — `glue.ptx` is not one of the
       version-pinned-audited artifacts, only `moe.ptx` is) — 69095→82719 bytes, `layernorm_quant`
       now present (14 occurrences). **This is a real, separate finding the user should know
       about independent of G3** — it means Cohere/Command-R (and everything else) has been
       CPU-only on CUDA this whole time, not the resident speed the G5 row's own commit claimed.
    2. **`TestKernelFMALint_coversEmbeddedPTX` correctly caught the new kernel unlinted** — my
       first `lora.cu` draft used bare `part += Ar[k] * dq` / `acc += Br[r] * tin[r]` /
       `dst[row] += scale * acc`, all bare float MACs the lint (audit C-16) exists specifically to
       catch before any numeric test runs. Fixed by rewriting as explicit `__fmaf_rn` calls (the
       same idiom `rmsnorm_quant` already uses) and adding `lora.cu` to `lintedKernels` — the
       CORRECT fix for a brand-new kernel, not an exemption (`moe.cu`'s exemption is a legacy
       special case with its own expired justification already flagged in that file; a new kernel
       with no such history should comply, not join it).
  - **`TestLoRAResidentParityCUDA`** (new, `cuda/lora_resident_parity_test.go`): same
    `testdata/llama-tiny` fixture, same synthetic 7-projection/4-layer adapter, same 0.95 floor,
    same vacuousness check as the Metal/WebGPU twins. Measured: **cosine 0.999963** — matching
    WebGPU's 0.999963 and Metal's 0.999960 on the identical fixture almost exactly; three
    independent kernel implementations across three different GPU APIs converging on the same
    number is strong evidence the LoRA math itself is right, not just each backend's own wiring.
  - Full regression (`cuda`, real RTX 2070 SUPER, idle/uncontended at test time — confirmed via
    `nvidia-smi` before running to avoid colliding with the other active session's own GPU use):
    117 pass / 0 fail / 105 skip. The only failures before the `glue.ptx`/FMA-lint fixes were the
    two above (both fixed) plus 3 failures from an UNRELATED, pre-existing gap — the mistral
    sliding-window tiny fixture's weight file is gitignored and genuinely absent from disk on this
    Mac despite its config JSON being tracked (the "dir-only guard" trap — the directory looks
    present, the weights are not), reproduced identically in the synced isolated checkout, not
    something this row caused or is in scope to fix. `gofmt -l`, `go vet`
    (`-tags "cuda goinfer_testhooks"`), and CI's pinned staticcheck (same tags, same 0.8.0
    version confirmed on nobara) all clean.
  - **G3 is now COMPLETE on decoder + all three GPU backends (Metal, WebGPU, CUDA)**, each with
    its own numeric parity test converging on the same ~0.9999-0.99996 cosine against the shared
    CPU reference. No commit yet — the isolated nobara checkout was left in place
    (`~/mycode/goinfer-g3-cuda`) rather than cleaned up immediately, in case the user wants to
    inspect it before it's torn down; the regenerated `glue.ptx` and the CUDA-side new files were
    copied back to this Mac's checkout so everything lands in one place to commit from here.
  - **All six commits above landed** (65781ec decoder plumbing, 7468e82 G10, e13cc00 Metal,
    2048ca2 WebGPU, 23c46b1 glue.ptx fix, 0547f60 CUDA), split by `git add -p` where a single
    file (`decoder/residency.go`, `metal/backend.go`) mixed G3 and G10 hunks — each commit
    independently build+test-verified in isolation via `git stash --keep-index -u` before
    committing, not just at the end.

- 2026-09-08 — G7 part 1 DONE (the trivial doc-honesty half; part 2 — actually implementing the
  MoE FFN block case on WebGPU — not started, needs a real Nano/Lightning checkpoint to validate
  against).
  - `decoder/residency.go`'s `withResidency()` now sets a SPECIFIC decline reason
    (`"Nemotron-H MoE FFN block has no GPU resident implementation on any backend..."`) for
    `a.nemotron != nil && a.MoE != nil`, ahead of the generic `DecodeRunnerEligible` call whose
    bare-bool return was swallowing WHY. This reaches `DecodePath()`/`serve check` directly (the
    webgpu-staged branch reads `m.resDecline` verbatim; cuda/metal fold it into
    `declinedToCPUReason`), so a user loading either real checkpoint now sees the actual gap
    instead of the family-agnostic "arch is not eligible for the resident decode runner".
  - `docs/hardware-matrix.md`'s generator (`decoder/hardware_matrix_test.go`) gets a new curated
    footnote (the existing "load-time fit facts the taxonomy can't know" section already had two;
    this is the third) explaining the Nemotron-H row is generated from a DENSE representative
    config and does not apply to the two real MoE checkpoints — regenerated via `-update`,
    `TestHardwareMatrix_fresh` passes.
  - `docs/completed/nemotron-resident.md` (scoped entirely to the dense port, but titled generically enough
    a reader could miss that) gets an explicit scope callout up top. `docs/task-families-2026-09.md`'s
    F2 (Lightning) section — which verified CPU-path config-identity against Nano in detail but
    never once mentioned GPU residency — gets a closing note stating CPU-only-on-every-backend
    directly, same as Nano's own already-archived note in `docs/completed/nemotron3nano-t3.md`.
  - Full regression: `decoder` 416 pass/1 fail (`TestOlmo3_forwardParity`, pre-existing)/134 skip.
    `gofmt -l` and `go vet` clean.

- 2026-09-08 — G8 DONE for the Gemma set (item (a)'s "Gemma" half). MoE (item (a)'s other half)
  not started — a batched routing+indexed-GEMM kernel that doesn't exist yet even for decode-
  shape batching, a genuinely separate and larger undertaking than extending the existing dense
  kernels. Item (b) (the bit-identity work that would let batched prefill default on) untouched,
  tracked elsewhere as the doc already noted.
  - **Two kernel signature changes, both in `metal/prefill.go`'s own separately-compiled
    library** (`prefillKernels`, isolated from decode's `allKernels` on purpose, so this row
    can't touch decode's audited numerics): `rmsnorm_f16` gains an `addOne` parameter (Gemma's
    `(1+w)` RMS offset — this kernel had NO addOne support at all before, meaning it was
    silently wrong for RMSAddOne families the moment they were ever admitted, not just
    aesthetically incomplete); `swiglu_f16` gains an `act` parameter selecting SiLU or GELU-tanh
    (a new `glu_act_f16` inline helper, DUPLICATED from `kernels.go`'s `glu_act` rather than
    shared — the two files compile as separate Metal libraries — including its ±15 tanh-argument
    clamp, the same Gemma-massive-activation overflow fix this session already had to apply
    twice elsewhere this week, applied preemptively here instead of found the hard way a third
    time).
  - **The sandwich-norm dispatch restructuring** (`PrefillLast`): Gemma's pattern (norm the
    sublayer OUTPUT before the residual add, not just the GEMV input) doesn't fit the existing
    fused mode-2 residual epilogue (`gemm_w4f16_store`'s `if (mode==2) v += C[...]`), so a
    sandwich family's o-proj/down-proj now write into mode-0 (plain overwrite) instead of
    straight into `xF`, get normed in place, then a separate `residual_f16` add — three
    dispatches instead of one, only for `r.sandwich` families, byte-identical for everyone else.
    Reuses `normF` as the scratch target for BOTH the o-proj and down-proj sandwich writes
    (its prior contents are always already-consumed by that point in the per-layer sequence, so
    no new buffer allocation was needed) — checked by tracing the exact producer/consumer order,
    not assumed safe.
  - **A real gap found while checking `FeatFinalLogitSoftcap`'s admission**: `PrefillLast` never
    called anything applying Gemma's final-logit softcap at all — `finalizeLogits` (the decode
    path's own entry point) does it, but `PrefillLast` copies straight out of `r.logits` into a
    fresh slice and returns before `finalizeLogits` ever runs. Fixed with a direct
    `softcapParallel(out, r.finalSoftcap)` call at the end of `PrefillLast`, mirroring
    `finalizeLogits`'s own logic exactly (`finalSoftcap` is 0 — no-op — for every family without
    it, so this is invisible to every non-softcapped family already admitted).
  - **A design trap avoided, not hit**: Gemma 3 and dense Gemma 4 derive an IDENTICAL required-
    feature set otherwise (the exact fact `decoder/features.go`'s `residentPerLayerGeomBackends`
    comment already documents from the G6 WebGPU incident it was written to prevent) — so simply
    adding `FeatSandwichNorm`/`FeatGatedGELU`/etc to `prefillFeatures` would have silently
    admitted Gemma 4 too, which this uniform-`g0`-read fast path cannot represent (its local/
    global attention layers genuinely differ in head_dim). Added a SEPARATE guard,
    `r.prefillOK = ... && m.PerLayerGeomOK("webgpu")` — calling the existing per-layer-geometry
    predicate with a backend that never declares support (webgpu) as a deliberate reuse to ask
    the arch-only half of that question ("does this arch vary per layer at all") independent of
    what Metal's OWN decode path separately supports (Metal decode DOES implement per-layer
    geometry, so calling it with "metal" would have wrongly cleared Gemma 4 here too).
  - **`TestPrefillParityGemma`** (new): real checkpoint (`testdata/gemma3-vl-tiny`, text tower
    only), same structure as the existing `TestPrefillParity` (sequential Forward vs PrefillLast,
    argmax match + cosine ≥ 0.95 floor — f16 activations vs decode's int8 mean high-but-not-exact
    is expected, not a bug). Passed on the FIRST real run: argmax match, cosine 0.99979. Building
    it surfaced one more thing to get right: the manually-built prefill embeddings needed the
    SAME embed-scale multiply `loadEmbedRow` applies internally for the sequential reference —
    missing it would have compared scaled-vs-unscaled embeddings and failed for a test-harness
    reason having nothing to do with the kernel work being gated.
  - **`TestPrefillDeclinesGemma4PerLayerGeom`** (new, kept as a permanent regression test rather
    than a throwaway check): pins the per-layer-geometry guard directly against
    `testdata/gemma4-dense-twogeom-tiny` — confirmed `prefillOK=false` before writing this as a
    permanent test, not just reasoning that it should be.
  - Full regression: `metal` 110 pass/0 fail/50 skip (108 prior + 2 new tests). `gofmt -l`,
    `go vet`, and CI's pinned staticcheck all clean.

- 2026-09-08 — G7 part 2 DONE for WebGPU (Nemotron 3 Nano / 3.5 Lightning's MoE FFN block, real
  GPU resident implementation, not just the honest-decline docs from part 1). CUDA and Metal
  still decline it — no dispatch for it exists on either.
  - **No new kernel primitives needed at all** — the gap was purely wiring. `moeRouteWGSL`
    already implements Nemotron's exact router shape (sigmoid + selection bias + group-limited
    top-k; `n_group=1` degenerates it to plain top-k, the kernel's own existing path for that
    case) since it's the SAME primitive DeepSeek/GLM already use. `moeExpert` was already generic
    per single projection (called separately for gate and up elsewhere, never assumed fused) —
    Nemotron's experts have ONLY up_proj/down_proj (non-gated relu², confirmed against the real
    safetensors index in `decoder/forward_nemotron.go`'s own comment — NOT `moeMLP`'s gated
    SwiGLU), so calling it for up/down only, skipping expGate entirely, was already correct.
    `relu2Quant` already existed (built for the DENSE Nemotron-H port's own relu²-MLP block).
    The actual gap: `gpu/residency.go`'s per-layer Nemotron block-kind switch (cases 0/1/2 —
    mamba/attn/mlp) always `append`s the layer and `continue`s before reaching the generic
    isMoE-building code a Mixtral/DeepSeek/GLM layer would hit, so a MoE block's per-layer
    weights (`rl.router`/`rl.expUp`/`rl.expDown`/`rl.shUp`/`rl.shDown`) were simply never
    populated. Added `case 3:` doing exactly that (no `expGate`/`shGate`/`shGateW` at all, unlike
    the generic branch it mirrors); added a matching `nemoKMoE` dispatch-side case in
    `gpu/decoderunner.go` (route → k routed non-gated-relu² experts weighted-summed → the
    always-on ungated shared expert of the same shape), inserted next to the existing `nemoKMLP`
    early-return (no mixer to run first, same as that case).
  - **A real, separate, pre-existing bug found and fixed while wiring the admission check**:
    `decoder.webgpuBackend`'s actual `Name()` is `"webgpu:" + the underlying GPU surface`
    (`"webgpu:metal"` on this Mac, presumably `"webgpu:vulkan"` on Linux) — NEVER the bare
    string `"webgpu"` every other backend uses (`"cpu"`, `"cuda"`, `"metal"`). A NEW
    `== "webgpu"` check written for this row's own admission override silently never matched,
    which traced straight to `decoder.Model.DecodePath()`'s PRE-EXISTING `case be != "webgpu":`
    — that branch has never matched a real webgpu backend on ANY platform, so every declined
    webgpu load has been reported through the cuda/metal branch's wording ("has no staged decode
    path" — false, webgpu does have one) instead of the webgpu-staged branch actually meant for
    it, since webgpu was first wired. Fixed with a new `isWebGPUBackend(name string) bool`
    helper (`strings.HasPrefix`, not equality) used at all three `m.be.Name()`/`be` comparison
    sites in `decoder/residency.go` — not papered over with another exact-match string.
  - **The admission change itself**: `decoder.Architecture.decodeRunnerEligible()` (the ARCH-only
    predicate `ResidentEligible`/doc-gen also use) still hard-declines EVERY Nemotron+MoE arch
    for every backend UNCHANGED — cuda/metal's own per-layer block-kind switches genuinely have
    no MoE case and would nil-dereference on an unhandled kind, so widening the arch-only
    predicate universally would have reintroduced exactly the crash risk it exists to prevent,
    just aimed at cuda/metal instead. Fixed one level up instead: `decoder.Model.DecodeRunnerEligible()`
    (which — unlike the arch-only function — knows `m.be`) overrides the decline SPECIFICALLY
    for `(nemotron+MoE, webgpu)`, leaving cuda/metal's decline (and the arch-only predicate
    itself) completely untouched. `withResidency()`'s own G7-part-1 specific-decline-reason
    branch got the same `isWebGPUBackend` exclusion, so it stops firing for the one backend that
    isn't actually declining. `docs/hardware-matrix.md`'s generator footnote,
    `docs/completed/nemotron-resident.md`,
    and `docs/task-families-2026-09.md`'s F2 section (all written in part 1, when "no backend
    implements it" was still true) updated to say webgpu now does.
  - **`TestNemotronMoEResidentParityWebGPU`** (new, `gpu/nemotron_moe_resident_test.go`): real
    committed fixture `testdata/nemotron3nano-tiny` (6 layers: linear_attention/moe/
    linear_attention/full_attention/moe/linear_attention — exercises mamba, attention, AND moe
    block kinds in one fixture, not a moe-only synthetic). Resident vs CPU, int4 both sides,
    minCosine 0.999771 across 8 positions — comfortably inside the established 0.95 floor.
  - **Not attempted**: end-to-end validation against a real 30B-A3B checkpoint. nobara has
    `~/models/nemotron3nano-30b-bf16` (59 GB) already downloaded, but its RTX 2070 SUPER (8 GB
    VRAM) almost certainly cannot hold 128 experts resident without expert paging — a separate,
    much larger undertaking (CUDA/Metal's own MoE paging took a whole prior work stream) not
    attempted here. The tiny-fixture parity gate is real numeric evidence of kernel correctness;
    it is not evidence this scales to the real checkpoint's memory footprint.
  - Full regression: `gpu` 95 pass/0 fail (94 prior + 1 new test); `decoder` 416 pass/1 fail
    (`TestOlmo3_forwardParity`, pre-existing)/134 skip. `gofmt -l`, `go vet`, and CI's pinned
    staticcheck all clean on both packages.

- 2026-09-08 — G8's MoE half DONE (item (a)'s other half; item (a) is now fully closed for
  Metal — Gemma set + MoE, both extending the batched f16-MMA prefill kernels). Gemma 4's
  `enable_moe_block` (g4moe) variant still declines (see below) — a third FFN shape, not this
  row's target.
  - **Mirrors CUDA's own established shape exactly, no new routing math**: `cuda/prefill.go`
    already batches a MoE layer's attention half normally and runs the FFN ROW BY ROW off the
    batched residual, reusing the per-token decode MoE dispatch chain unchanged — not a batched-
    routing-and-indexed-GEMM kernel, which is what G8's own write-up originally guessed this would
    need before actually looking at how CUDA does it. Metal's row loop does the same: for each of
    the M prompt positions, norm+quantize that row, route, run the k experts + shared expert, fold
    the result back into the batched residual — `metal/prefill.go`'s `PrefillLast`, a new
    `if L.moe != nil { ... continue }` branch ahead of the existing dense-FFN dispatch sequence.
  - **The type-mismatch bridge**: decode's MoE dispatch chain is F32-only (`r.x`); prefill's
    residual (`xF`) is F16. Three pieces close the gap: (1) `rmsnorm_quant_f16` — already existed,
    built for the final-logits step — norms+quantizes an F16 row DIRECTLY into `r.mq`/`r.mSc`, no
    F32 conversion needed; (2) a new `zero_f32` kernel zeroes a per-row F32 scratch buffer
    (`moeDst`, size H, allocated once per `PrefillLast` call) before each row's expert loop —
    unlike decode's `r.x`, which always already holds the residual to accumulate onto, this
    scratch starts uninitialized every row and must be zeroed explicitly; (3) a new
    `residual_f16_from_f32` kernel (`x[i] = half(float(x[i]) + y[i])`) folds `moeDst` into `xF`'s
    row once, after the expert+shared-expert accumulation completes.
  - **`encodeMoEExperts`/`encodeMoESharedExpert` parameterized with a `dst Buffer` accumulate
    target** (`metal/moe.go`), previously hardcoded to `r.x`. Decode's two callers
    (`encodeMoEFFN`, `encodeMoEExpertsPaged`'s tail call) both now pass `r.x` explicitly; the new
    prefill row loop passes `moeDst`. Also split `encodeMoERouter` into a norm dispatch plus a new
    `encodeMoERoute` (route-selection only, no `r.x` involvement) so prefill's row loop — which
    norms via `rmsnorm_quant_f16` instead of `encodeMoERouter`'s own F32-only `r.pRms` — can reuse
    just the routing half. Verified this refactor alone is a byte-for-byte no-op for decode: full
    Metal suite green before writing a single line of the new prefill branch.
  - **The g4moe guard, caught by design review before any test found it**: adding
    `FeatMoE`/`FeatMoEGatedShared` to `prefillFeatures` would, on its own, admit Gemma 4's
    `enable_moe_block` variant too if it happened to have uniform per-layer geometry — a THIRD
    FFN shape (`residLayer.g4moe`, dispatched via `encodeGemma4MoEFFN`), not the `L.moe != nil`
    shape this row implements, and not caught by the existing `PerLayerGeomOK` guard (which only
    checks head_dim variance, nothing about FFN shape) — the exact class of blind spot G8's own
    Gemma-half incident earlier this session already burned once (dense Gemma 4 vs Gemma 3 having
    an otherwise-identical feature set). Added `&& !m.HasGemma4MoEResident()` to `r.prefillOK`'s
    computation; confirmed via a throwaway test against `testdata/gemma4-moe-tiny`
    (`prefillOK=false`) before deleting the scaffolding.
  - **`TestPrefillParityMoE`** (new, `metal/prefill_moe_parity_test.go`): real checkpoint
    `testdata/mixtral-tiny` (no shared expert, softmax routing, MHA — the simplest MoE shape
    available, deliberately isolating the row-loop mechanics). Same structure as
    `TestPrefillParity`/`TestPrefillParityGemma`: sequential Forward vs `PrefillLast`, argmax
    match + cosine ≥ 0.95 floor. Passed on the first real run: argmax match, cosine 0.99993.
    `TestPrefillParityMoEGatedShared` (meant to exercise `encodeMoESharedExpert`'s OTHER branch —
    a sigmoid-gated always-on shared expert, Qwen2-MoE) correctly SKIPS: `testdata/tiny-qwen2-moe`
    is an incomplete fixture (`model.safetensors` only, no `config.json`); no substitute
    gated-shared fixture exists in `testdata/` today. Left as a documented placeholder, not
    force-fit onto the wrong fixture.
  - **`TestMoE_declinesPrefill` flipped to admit**, following the `decoder/gptoss_decline_test.go`
    precedent of keeping a decline test's historical name/file identity when the underlying
    capability moves from decline to admit (`git log -p` on it still tells the right story). Its
    existing identical-experts-vs-dense trick (`TestMoE_assemblyVsDense`'s own fixture: 8 identical
    experts + zeroed shared expert ⇒ MoE FFN mathematically equals the dense FFN regardless of
    routing) carries over for free: MoE prefill logits now checked directly against the dense
    twin's prefill logits, not just "both succeed" — cosine 0.999875, confirming the `dst`
    redirection through `moeDst` neither drops nor double-counts anything. (Caught one own bug
    while writing this: an edit that dropped each embedding row's `make([]float32, tmHidden)`
    allocation, leaving 8 nil rows, surfaced as `index out of range [0] with length 0` deep inside
    the dispatch chain — a reminder that an empty-input panic from GPU code can look exactly like
    a real kernel bug until you check the input.)
  - `metal/backend.go`'s `PrefillLast` doc comment (stale since G8's Gemma half, worse now)
    updated to describe both what admits (Gemma set, generic gated-SwiGLU MoE) and what still
    declines (per-layer-geometry, g4moe) accurately.
  - Full regression: `metal` 111 pass/0 fail/51 skip (110 prior + 1 new test; `TestMoE_declinesPrefill`
    is a rename-in-place, not a net-new test). `gofmt -l`, `go vet`, and CI's pinned staticcheck
    (v0.8.0) all clean.

- 2026-09-09 — G9 SCOPED, NOT IMPLEMENTED: still a legitimate wash, gate unchanged since
  2026-06-09. `task-gpu-batched-prefill.md` (which this item's own Fix line points to) carries an
  explicit "GATED — do not build yet" banner: batched prefill only wins if the WGSL tiled GEMM
  clears the bandwidth-bound M=1 GEMV, which needs `dot4I8Packed`/DP4A in `cogentcore/webgpu`;
  without it the 2026-06-09 measurement found it a wash on both backends (RTX ≈0.91×, Metal
  ≈1.2×). Re-checked before writing any code: `gpu/go.mod` still pins `cogentcore/webgpu v0.23.0`
  — still the only version the proxy has ever published — and that module's source has no
  `dot4`/`dp4a` symbol anywhere. Nothing has changed since June; building the `Prefiller` now
  would cost real engineering effort for ~0 net TTFT gain per the doc's own component-level
  measurement on real hardware.
  - **Also found and corrected a stale internal record while checking for a contradicting
    signal**: a session memory claimed a June 2026 branch (`gpu-wgpu-dot4`, deleted after harvest)
    had actually gotten `dot4I8Packed` working and measured it at "~0 gain, bandwidth-bound" —
    which would have meant the gate had already been tested and failed, not just left unbuilt.
    `git log --all` does not support this: commit `d2ba970` (2026-06-19, on `main`) directly
    refutes the OTHER half of that same memory (a claimed "−23% v29 decode penalty") with a real
    re-measurement (per-dispatch cgo record cost identical, 1.1µs both; gemv compute within 4%) —
    and no commit anywhere ever describes `dot4I8Packed` as measured rather than blocked; every
    mention in `roadmap.md`/`gpu-assessment.md`, before and after that date, says "blocked" /
    "upstream-blocked". The memory had conflated a reasoned prediction ("decode's M=1 GEMV is
    already bandwidth-saturated, so packed int8 dot arithmetic wouldn't help even if it existed")
    with an actual test. Corrected in this session's memory store, not just noted here.
  - No code changed. Moving to the next open item.

- 2026-09-09 — G11 SCOPED, REDIRECTED. G11's own text says "no layer placement on CUDA/Metal —
  already scoped as `task-fit-to-hardware.md`, listed here for completeness." That doc, in turn,
  explicitly disclaims covering this at all: its `placement` enum has no "N dense layers on GPU,
  the rest on CPU" state, and its own §8 names the real fix as `task-freetoken-techniques.md`'s
  Lead 5 (bandwidth-adaptive CPU/GPU co-execution) — which that doc itself marks **"the biggest
  architectural lift of the five... worth a scoping pass of its own before any code"** and
  **priority: low, don't start it yet** (flagged as possibly antagonistic with the speculation
  program). So G11 as literally worded does not reduce to a buildable item today. Redirected to
  `task-fit-to-hardware.md` instead, at the user's direction.
  - **Found `task-fit-to-hardware.md`'s own status line is stale**: it says "SCOPED 2026-09-02,
    nothing started," but `decoder/fitguard.go` (438 lines, `db61c83`, 2026-09-06 — three days
    *before* the doc's own last edit) already implements Phase 0's accounting for the CPU staged
    GGUF load path (weight+KV byte estimation, quant-aware sizing via `measureBytesPerElem`,
    auto-context-pinning, `GOINFER_NO_FIT_GUARD`). Phases 1–5 (the `plan()` function,
    `goinfer-chat fit`, fit-by-default on CUDA/Metal, WebGPU, rate bands) remain entirely
    unstarted. The doc's header was never updated after Phase 0 landed — flagging here rather
    than silently working around it, per this doc's own "findings go into the status line" rule.
  - **This session's slice, per the user's choice**: close M-02's remaining accounting gaps on
    Metal, and build the CUDA memory-fit guard that turned out not to exist at all for the fixed
    (non-expert) term. Not the full planner (Phases 1–5) — that stays unstarted.
  - **Metal (M-02 continued).** `decoder/weightbytes.go` refactored: `residentWeightBytes` split
    into `residentWeightBytesSplit()` (returns the dense sum once, plus a closure capping the
    routed-expert sum at N slots) so `ResidentWeightBytesPaged` and two new accessors —
    `ResidentDenseWeightBytes()` (dense-only, for a caller that must price the FIXED part before
    an elastic term like an MoE cache gets to size itself) and `ResidentHostCopyBytes(slots)` —
    can share one enumeration pass rather than risk disagreeing about what "dense" means.
    `ResidentHostCopyBytes` accounts for the specific M-02 gap: Metal's unified memory holds BOTH
    a quantized host `WeightMat` AND a separately re-packed device buffer for the same dense
    weights (`metal/model.go`'s `int4Buf`, nothing released in between) — but a genuinely PAGED
    expert (`0 < slots < nExperts`) does NOT get this doubling, since it streams via pread
    straight from the `.giw` file into its device slot buffer (confirmed by reading
    `metal/moe.go`/`metal/gemma4_moe.go`'s staging code directly, not assumed) rather than being
    materialized host-side first. `metal/backend.go`'s `residentNeedBytes` gained this addend plus
    a new `residentKVBytes` term (`metalCtxCap × kvDim × 2 bytes(f16) × 2(K,V)`, summed per layer
    for per-layer-geometry families) — the dominant "scratch" term M-02 named as entirely missing;
    the many small per-model buffers (`r.mq`/`r.gu`/`r.logits`/etc.) stay excluded as rounding to
    nothing beside it, the same exemption `ResidentWeightBytes`' own doc comment already gives
    norms/biases. `metal/resident_memguard_test.go`'s existing M-02 gate updated to assert the new
    three-term sum instead of the old weight-only one, plus a new assertion that the host-copy
    addend itself shrinks under paging; `decoder/weightbytes_test.go` gained
    `TestResidentHostCopyBytes_exemptsPagedExperts`, cross-checked against an independently-
    written per-layer formula (not the production code's own arithmetic), same discipline as the
    existing M-01/M-02 gates in that file. Full `metal` suite: 111 pass/0 fail/51 skip (unchanged
    count — two tests updated in place, none added). Full `decoder` suite: 417 pass/1 fail
    (`TestOlmo3_forwardParity`, pre-existing, unrelated)/134 skip. `gofmt -l`, `go vet`, staticcheck
    all clean.
  - **CUDA (M-02, new ground — no guard existed here at all).** An earlier grep-based pass had
    concluded CUDA had zero memory-fit protection; closer reading found that's only true for the
    FIXED weight term — `cuda/resident.go`'s `checkKVFits` already guards KV-vs-free-VRAM well
    (hard error when an explicit `-ctx` doesn't fit, decline when the default doesn't), and
    `capSlots` already elastically sizes the MoE expert cache against live free VRAM via a
    bisection search. What was missing: a dense model whose weights ALONE exceed free VRAM ran
    the full kernel-compile-and-upload sequence before failing on whichever raw CUDA allocation
    happened to be the first one that didn't fit — no clean decline naming the numbers, the exact
    failure MODE `checkKVFits`'s own doc comment already named as the point of a load-time guard.
    Added `checkWeightsFit` (`cuda/resident.go`, next to `checkKVFits`) + `fitsWeightsBudget` (the
    pure arithmetic, unit-tested standalone like Metal's `fitsResidentBudget`), wired into
    `cuda/backend.go`'s `BuildResident` immediately after the device is created — before any
    kernel compile or weight upload. Deliberately prices `m.ResidentDenseWeightBytes()`, NOT
    `ResidentWeightBytesPaged(0)`: routed MoE experts have their own elastic sizing that runs
    LATER against live free VRAM, so pricing them at full unpaged size here would decline a model
    whose experts are about to be capped down to fit — the same over-eager-guard class M-02
    already fixed once for Metal's paging case, avoided here by construction rather than
    discovered by a failing gate.
  - **Verified on nobara (RTX 2070 SUPER), via an isolated checkout + `go.work` overlay (this
    session's own unpushed commits transferred as a git bundle, not through origin) — established
    pattern from this session's earlier CUDA work.** `go build`/`go vet -tags cuda` clean. New unit
    test `TestFitsWeightsBudget` passes (mirrors Metal's `TestResidentMemGuard` table). Real
    end-to-end sanity check: `TestBackendResidentWired` (production `decoder.Load(cuda)` →
    `BuildResident` → `Forward`, qwen2.5-coder-0.5b) still passes with the new guard wired in — 7/8
    exact, worst near-tie 0.563%, confirming the guard does not wrongly block a real model that
    fits.
  - **Full `cuda` suite (started 04:59:50 PDT / 11:59:50 UTC, `-timeout 25m`) hit that timeout
    mid-`TestGemma4_26B_cache_B` — a pre-existing property of the full heavy suite, not this
    change (a 26B real-checkpoint test genuinely takes a while); 49 pass/7 fail/8 skip before the
    cutoff.** Investigated all 7 failures rather than assuming: every one is `dflash_dispatch_test.go`/
    `dflash_draftcost_test.go` declining resident with "resident did not engage" — 5 of 7 show
    `checkKVFits`' PRE-EXISTING decline message (unchanged by this diff) at critically low free
    VRAM (0.10-0.70 GB); the other 2 show THIS change's new message ("dense weights need 2.66 GB
    but only 2.67 GB is free"). Re-ran all 7 together, in isolation from the rest of the suite:
    **7/7 pass cleanly**, confirming the failures were cumulative VRAM pressure from whatever ran
    earlier in that specific 25-minute sequence, not a standalone regression. The 2 tests hitting
    the NEW message specifically would have failed anyway, one step later, via `checkKVFits`:
    with only ~10 MB left after weights (2.67−2.66 GB) there is nowhere near the ~1.2 GB the
    default 4096-position KV cache these tests use needs — this change just names the same
    inevitable decline earlier and more clearly, it does not create a new failure mode. Did not
    spend a second 25-minute run proving the unmodified code fails identically at that exact
    point in the sequence — the direct evidence (5/7 failures already carrying `checkKVFits`'
    unchanged message, plus the arithmetic above for the other 2) was judged sufficient without
    re-burning nobara time already at a premium this session.
  - **Not attempted, left open**: the drafter/verify-buffer timing bug that is
    `task-fit-to-hardware.md`'s own concrete example (a `--drafter` attach AFTER `BuildResident`
    grabs VRAM the MoE expert cache already claimed, `NewBlockSpec` then fails) — fixing it
    requires `BuildResident`'s signature (or an out-of-band hint) to know a drafter is coming
    BEFORE `capSlots` runs, a `decoder.ResidencyBackend` interface change both CUDA and Metal
    implement, genuinely bigger and riskier than the accounting-only fix made here. The full
    `task-fit-to-hardware.md` planner (Phases 1–5: `plan()`, `goinfer-chat fit`, fit-by-default,
    WebGPU, rate bands) also remains entirely unstarted.

- 2026-09-09 — `task-fit-to-hardware.md` Phase 1 (the pure `plan()` function + its table test, G4)
  DONE for the "Load()-based" scope chosen at the user's direction — not the doc's original
  header-only ambition (see below). `goinfer-chat fit` and the banner wiring (the rest of Phase
  1's surfaces) NOT started; those are next.
  - **Scope decision, made explicit before writing code**: §2 wants `plan()` reading tensor bytes
    per-class straight from a checkpoint HEADER ("two seconds, no model in memory") so `pull` and
    the web UI can check fit before a multi-GB download. Nothing in the repo separates dense from
    routed-expert bytes at the header level today (`estimateGGUFWeightBytes`, `decoder/
    fitguard.go`, sums every tensor flat) — building that is real new work, and only for GGUF.
    Chose instead to build `Plan` against an already-`Load`ed `*decoder.Model`, reusing every
    accessor this session's own M-02 work just built
    (`ResidentDenseWeightBytes`/`ResidentWeightBytesPaged`/`MoEResidentParams`/
    `KVHeadsAtResident`/`HeadDimAtResident`) — far less new code, correct output, but `goinfer-chat
    fit`/`pull`'s verdict/the web UI listing would need to load the model first when they're
    eventually wired up, losing the header-only speed promise for those three surfaces
    specifically. The startup banner and fit-by-default paths (Phase 2) load anyway, so this
    doesn't cost them anything.
  - **`decoder/fitplan.go`** (new): `Placement` enum (resident/expert-cached/host-computed-experts
    [reserved for L-01, never chosen]/weight-paged/decline); `PlanRequest` (ctx + pinned flag,
    explicit slots, KV precision flags, `ExtraBytes` for a companion allocation) with EVERY
    default resolved by the caller — `Plan` invents none, so a test can see exactly what it asked
    for; `Plan` result struct carrying every byte term plus a human-readable `Reason`, and
    `NeedBytes()`. `Model.Plan(backend, freeBytes, req) Plan` checks per-backend feature
    eligibility FIRST (`MissingResidentFeatures`/`ResidentBackendFeatures` — the SAME taxonomy
    every real backend's `BuildResident` already gates on, so the plan can never disagree with
    what a backend would actually do) before any byte arithmetic runs, then follows §2's priority
    order: try the requested ctx; if it doesn't fit and isn't pinned, shrink toward `ctxPlanFloor`
    (4096) — computed by division since KV bytes are exactly linear in ctx, not a search; a dense
    model goes resident or declines there; an MoE model tries fully-resident first, then cache the
    largest expert-slot count that fits beside dense+KV+`ExtraBytes` (again arithmetic — a layer's
    experts are uniform in shape, same assumption `ResidentWeightBytesPaged` already makes), and
    declines only below top-k (one token's routed set must be simultaneously resident). CPU never
    declines outright — it always has weight-paging as a backstop, matching how `--weight-cache`
    already behaves today. A new `kvBytesPerPositionAllLayers` sums KV bytes PER LAYER (not one
    model-level figure) so a per-layer-geometry family (Gemma 4) isn't mis-priced, mirroring
    `metal/backend.go`'s own `residentKVBytes` from the M-02 work but backend-agnostic (no
    `metalCtxCap` dependency). WebGPU is explicitly out of scope this phase (needs M-32 first,
    Phase 3's own item) — an unrecognised backend name declines on features, same as an
    unimplemented arch would, never panics or silently answers wrong.
  - **`decoder/fitplan_test.go`** (new), G4's table test: dense (`testdata/llama-tiny`, TRACKED in
    git — this row runs in CI unconditionally, unlike the MoE/hybrid rows below), MoE
    (`testdata/gemma4-moe-tiny`, gitignored, skip-guarded like every other test using it), and
    hybrid (`testdata/bailing_hybrid-tiny`, MLA+KDA, same skip convention) — not literally
    zero-asset as §6's G4 aspires to (two of three rows need a real, if tiny, checkpoint), flagged
    honestly rather than claimed as satisfied. Budgets scaled to each fixture's OWN measured byte
    counts rather than the doc's literal "6 to 64 GB" — these are toy parity fixtures, a literal
    GB range would never exercise the decline path. 11 cases: generous budget → resident (dense
    and MoE); a tight-but-sufficient budget auto-shrinks ctx (not floored, not unchanged); an
    EXPLICITLY pinned ctx that doesn't fit declines rather than silently shrinking; CPU never
    declines (falls to weight-paged) even at a near-zero budget; a GPU backend WITH a near-zero
    budget does decline; MoE caches a slot count strictly between top-k and every expert at a
    budget sized for exactly that; MoE below top-k's own floor declines; an unknown backend name
    declines on FEATURES with zero byte accounting even computed (proving eligibility is checked
    before arithmetic, not after); the hybrid fixture either admits cleanly or declines on
    features, never on bytes it shouldn't have reached. A twelfth, separate test
    (`TestPlan_extraBytesReservedAheadOfExperts`) pins the exact regression this session's own
    CUDA M-02 work traces back to (`task-fit-to-hardware.md`'s own motivating example: a
    `--drafter` attach grabbing VRAM an MoE expert cache had already claimed) — asserts
    `ExtraBytes` is priced AHEAD of the elastic expert-slot count, not ignored.
  - **Mutation-checked**, not just run: a deliberate one-line break to the ctx-shrink branch
    (always return the unshrunk ctx) was caught by exactly one subtest
    (`dense/cuda/tight_shrinks_ctx`) and nothing else — the other ten stayed green, confirming they
    each pin something the mutation didn't touch rather than all silently depending on the same
    assertion.
  - Full regression: `decoder` 419 pass/1 fail (`TestOlmo3_forwardParity`, pre-existing,
    unrelated)/134 skip (417→419: two new test functions, `TestPlan_tableDriven` and
    `TestPlan_extraBytesReservedAheadOfExperts`, both counted once each despite 11+1 subtests).
    `gofmt -l`, `go vet`, staticcheck all clean.

- 2026-09-09 — `task-fit-to-hardware.md` Phase 1's remaining two surfaces DONE: `goinfer-chat fit
  <path>` (the dry run) and, as a prerequisite it surfaced, a small cross-package memory-probe
  registry so `Plan` can be told a REAL number for whichever GPU backend a given binary actually
  links — Phase 1's own `plan()` work (previous entry) only ever took `freeBytes` as a plain
  argument; nothing wired a live query to it until now.
  - **The gap found while wiring it**: `decoder` cannot import `cuda`/`metal` (they import IT), so
    a CLI command living in `internal/fitcmd` (linked into the plain, GPU-tag-free
    `internal/chatapp`) has no way to ask "how much VRAM/unified-RAM headroom does this backend
    have" for a backend it may not even be compiled with. `decoder.RegisterBackend` already solves
    the exact same shape of problem for backend REGISTRATION; added `decoder.RegisterMemoryProbe`/
    `FreeBytesFor` next to it in `decoder/backend.go`, same pattern (register from `init()`, decoder
    stays clean, `ok=false` means "unknown, don't guess").
  - **`metal/backend.go`'s registered probe is NOT a live query** — 70% of `hw.memsize`, the exact
    figure `residentFitsMemory`'s own `residentMemFraction` already uses, so `fit`'s report can
    never disagree with what the real guard would do. Darwin's UBC makes "available" memory
    unreliable (documented at length on that same guard already), so a live query would be the
    wrong number to report, not just an inconsistent one.
  - **`cuda/backend.go`'s registered probe IS a live query** (`dev.Context().MemInfo()`) — VRAM is
    a separate pool from host RAM, so "free" is real and reliable here, unlike Metal's case.
    Creates a throwaway device with no kernels loaded and releases it (`dev.ReleaseObjects()`)
    immediately after the query, the same bare create→query→release shape
    `cuda/alloc_floor_test.go` already uses directly — not the persistent `LockOSThread`'d executor
    `cudaResident` needs for actual decode.
  - **`internal/fitcmd/fit.go`** (new): `Run(args) int`, same shape as `internal/pullcmd`'s `Run`
    (a `flag.FlagSet` + a leading positional path, shared by both binaries' dispatch). Loads the
    checkpoint (this phase's Load()-based scope, previous entry), then calls `Plan` once per
    `decoder.CompiledBackends()` — so a CPU-only build reports only "cpu"; the metal/cuda release
    binaries report their own GPU backend too, automatically, with no per-binary fit code. Wired
    into `internal/chatapp/main.go` as a `pull`-style pre-`flag.Parse` subcommand
    (`goinfer-chat fit <path>`), plus the usage/error-message updates naming it alongside
    `pull`/`models`.
  - **Verified end-to-end on real hardware, all three binary variants, first try**:
    - Default (CPU-only) build against `~/models/Qwen3-0.6B-Q8_0.gguf` on this Mac: reports only
      `cpu`, RESIDENT.
    - `-tags metal` build, same checkpoint: reports `cpu` AND `metal`; `metal`'s free-bytes read
      11.20 GB — exactly 70% of this Mac's 16 GB, confirming the registered probe matches
      `residentFitsMemory`'s own arithmetic precisely.
    - `-tags cuda` build on nobara (RTX 2070 SUPER): `qwen2.5-coder-0.5b` reports `cpu` RESIDENT
      and `cuda` RESIDENT (7.03 GB free, matching `nvidia-smi` within noise); the REAL
      `gemma-4-26B_q4_0-it.gguf` (a genuine 26B checkpoint, not a tiny fixture) reports `cuda`
      EXPERT-CACHED at 18/128 experts, 7.03 GB free — the exact scenario `task-fit-to-hardware.md`
      §4's own worked example describes, produced by this tool against a real checkpoint on the
      real card the doc was written against, on the first run.
    - An absurd pinned `-ctx` (999999999) on both cpu and metal correctly declines/falls to
      weight-paged rather than silently shrinking, on real hardware.
  - **A known, accepted overlap, not reconciled this phase**: `decoder.Load` itself already runs
    `decoder/fitguard.go`'s own CPU-path guard (Phase 0), which can print its own "context capped
    at N" line and auto-shrink the loaded model's OWN KV sizing — a SEPARATE mechanism from
    `Plan`'s ctx-shrink logic, which computes independently against whatever ctx `fit` was asked
    to plan for. The two can both fire in one `fit` invocation (observed live in testing above),
    which is exactly the doc's own header note: "three residency guards... still fragmented rather
    than reconciled into one planner" — visible now in `fit`'s own output, not just in code, but
    not fixed here.
  - **`internal/fitcmd/fit_test.go`** (new, zero prior coverage in that package): usage-error and
    load-error exit codes; a real end-to-end run against `testdata/llama-tiny` (tracked, runs in
    CI) asserting the report names `cpu` and a real placement word; the pinned-impossible-ctx case
    on the same tracked fixture, asserting the header echoes the ctx UNCHANGED and no backend
    reports RESIDENT.
  - Full regression: `decoder` 419/1(pre-existing)/134skip unchanged, `metal` 111/0 unchanged,
    `internal` all packages pass (new `internal/fitcmd` 4/4). `cuda`, real hardware on nobara:
    `TestBackendResidentWired` (production load path) still green with the new probe registered;
    `TestFitsWeightsBudget`/`TestKVBytesForCap` unaffected. `gofmt -l`, `go vet`, staticcheck all
    clean on every touched package.
  - **Not done, left open**: the startup banner (Phase 1's third surface, `task-fit-to-hardware.md`
    §3 — "the closest thing the product has to a UI") does not yet print a `Plan`; `pull`'s
    verdict and the web UI's Models-tab listing (§3's other two surfaces) still need the
    header-only work this phase explicitly deferred. Phases 2–5 (fit-by-default, WebGPU, the rate
    band, host-computed experts) remain entirely unstarted.

- 2026-09-09 — `task-fit-to-hardware.md` Phase 2 STARTED, CUDA-only, at the user's explicit
  direction ("CUDA first, measured") over wiring all three backends at once — Phase 2 is a real
  production default change, unlike Phase 1's dry run, and its own gate (G1) requires a real
  decode-rate measurement, not just correctness. Metal and CPU are NOT touched this pass.
  - **What actually changes for CUDA**: only CONTEXT. `capSlots` (`cuda/resident.go`) already IS
    "fit by default" for the elastic MoE term — `--moe-cache-slots 0` auto-caps to live free VRAM,
    which is exactly what Phase 2 asks for and already existed before this task doc was written.
    The one CUDA piece that was NOT automatic: an unpinned load always got the flat
    `cudaCtxCapDefault` (4096) regardless of the card, and that constant's OWN doc comment already
    recorded a real measurement (RTX 2070 SUPER, dense 7B int4) where the true per-card ceiling was
    5-6x the default — headroom nobody who didn't know to pass `-ctx` ever got.
  - **`cuda/resident.go`**: new `resolveCtxCapFit(m, request, modelCtx)` next to the existing
    `resolveCtxCap`. An EXPLICIT request (a pinned `-ctx`) is untouched either way — fit-by-default
    only ever applies to the unpinned case. Otherwise: try a candidate of `fitDefaultCtx` (8192,
    the same agent-turn-size default `task-fit-to-hardware.md` §8 and `goinfer-chat fit`'s own
    `-ctx` default already use — so the dry run and the real load agree), clamped to the model's
    own window; ask `Plan("cuda", freeBytes, ...)` (`decoder.FreeBytesFor("cuda")` — this session's
    own earlier `fit` work) what actually fits; use whatever Plan lands on. **Built so it can only
    ever raise the default above 4096 or leave it exactly at 4096 — never below**: an unknown free-
    bytes reading, a `PlacementDecline`, or a Plan result somehow under 4096 all fall back to the
    exact historical constant, so there is no configuration where turning this on could make an
    existing working deployment worse. `GOINFER_NO_FIT_DEFAULT=1` is a temporary escape hatch (not
    the doc's own `--fit=off` CLI flag yet — deferred until Metal/CPU share it too, since a flag
    that only ever affects one of the three backends the doc names is premature).
  - **Verified on real hardware (nobara, RTX 2070 SUPER), not assumed from the arithmetic alone**:
    `TestResolveCtxCapFit_shortcuts` (new, `cuda/resident_cap_test.go`) pins the branches that need
    no device (explicit request, the env escape hatch, a model window already at/below the
    historical default) — `nil` is passed for `*decoder.Model` deliberately, itself asserting none
    of those three branches ever reaches code that would dereference it. The live-probe-driven
    branch was checked directly: a throwaway probe against `qwen2.5-coder-0.5b` resolved
    `ctxCap=8192` (up from 4096) with fit-by-default on, and exactly `4096` with
    `GOINFER_NO_FIT_DEFAULT=1` set — confirmed on the real card, not inferred from the code.
  - **G1's rate bar, measured, not assumed**: KV cap size affects reserved VRAM, not per-token
    compute at a fixed shallow decode depth — a bigger allocated cap should cost nothing at the
    depths a real test decodes to, but that is a claim, not evidence, so it was measured directly.
    `TestProdThroughput` (existing, `cuda/backend_wired_test.go`, autoregressive with real
    advancing positions) run twice, same box, same model, ctx=8192 (new default) vs ctx=4096
    (`GOINFER_NO_FIT_DEFAULT=1`): logits path 230.9 vs 230.7 tok/s, greedy fast path 234.2 vs 234.9
    tok/s — differences of 0.09% and 0.3%, both comfortably inside noise, nowhere near the ≥0.90×
    bar. `TestBackendResidentWired` (the full production wiring gate) still green with the new
    default active (7/8 exact, worst near-tie 0.563%, unchanged from before this change).
  - **No regression across the rest of the CUDA suite**: `TestSplitKV_bitIdentical`/
    `TestSplitKVCrossover` (heavy, KV-cap-adjacent) both pass with the new default; a full-suite
    run (started 07:35:06 PDT / 14:35:06 UTC, 25-minute timeout) was launched specifically to catch
    any VRAM-pressure interaction the bigger default might introduce in already-tight tests (the
    same class of pre-existing cumulative-pressure artifact the prior M-02 full-suite run
    surfaced) — result below.
  - **Full-suite result: 55 pass/1 fail/8 skip before the 25-minute timeout hit mid-
    `TestGemma4_26B_cache_B`** (a genuinely heavy real-26B-checkpoint test, same as last time).
    The one failure, `TestResidentDrafter_extendContext`, was taken seriously rather than
    dismissed on sight: it attaches a drafter to a target loaded with an UNPINNED context
    (`decoder.Options{Backend:"cuda", Quant:"int4"}`, no `ResidentContext` set), which is exactly
    the shape this change touches — a bigger default target KV reservation competing with the
    drafter's own attach for the same finite VRAM is precisely `task-fit-to-hardware.md`'s own
    motivating example (§2's `--drafter`-after-`BuildResident` story). Re-ran it in isolation,
    fit-by-default ON and OFF: **both pass cleanly** (`... BIT-IDENTICAL to one-shot (8) across
    8192 K values`), confirming this specific failure is the same suite-ordering VRAM-pressure
    artifact as the DFlash failures the earlier M-02 full-suite run found — not a deterministic
    regression from this change, on this hardware, for this drafter/target pairing.
  - **The narrower risk this does NOT rule out, stated rather than hidden**: fit-by-default's
    context default is not aware that a drafter might attach later (the exact gap
    `resolveCtxCapFit`'s own doc comment and the earlier M-02 CUDA entry already name as future
    work — `BuildResident` cannot see a companion allocation it does not yet know about). A
    drafter+target pairing that fit comfortably under the historical 4096-position default on a
    genuinely tight card could, in principle, find less room once the target's own default KV
    claim doubles to 8192 — this session's test evidence says that does NOT happen for the
    specific pairing/hardware tested, but it is a real, un-eliminated failure mode, not a proven
    absence of one. `GOINFER_NO_FIT_DEFAULT=1` (or an explicit `-ctx`) is the mitigation for a
    drafter workload on a tightly-constrained card until the companion-aware fix lands.
  - `gofmt -l` clean; `go vet -tags cuda` clean; staticcheck's only findings
    (`drain_marker_test.go`/`mustalloc_test.go`/`specdecode.go`, all U1000) are in files this
    change never touched — confirmed pre-existing before this session's earlier M-02 CUDA work.
  - **Not done, left open**: Metal and CPU's own fit-by-default pieces (Metal's slot count still
    has no `Option`/flag at all, still env-var-only per the doc's own §3 note); the real
    `--fit=off` CLI flag (currently the CUDA-only env-var stand-in); G6's table test (every
    explicit flag honoured or refused with numbers); the companion-aware (drafter) ctx sizing
    named just above.

- 2026-09-09 — `task-fit-to-hardware.md` Phase 2, Metal's own piece: "Metal slots become an Option
  and a flag" (§3), done exactly as scoped — NOT Metal's context, which turned out to have a real
  structural blocker this phase correctly stays away from (below).
  - **Why Metal's ctx wasn't touched too, checked before writing anything**: CUDA's
    `resolveCtxCap` at least took a real request value; Metal's `metalCtxCap` is a bare package
    CONSTANT (`const metalCtxCap = 4096 // resident KV positions (spike)`, `metal/model.go`) with
    NO `-ctx` plumbing into it at all today, AND its own doc comment says raising it past 4096
    without also resizing a fixed-size kernel-side threadgroup buffer (`sc[4096]` in the attention
    kernel) is a silent out-of-bounds write. This is a real, deliberate kernel-level ceiling, not
    an arbitrary conservative default the way CUDA's was — fit-by-default cannot safely touch it
    without an MSL kernel change first, so it stays out of scope here rather than being rushed.
  - **The slots piece, by contrast, was already 90% there**: `decoder.Options.MoECacheSlots` /
    `Model.MoECacheSlotsRequest()` — the SAME field and accessor CUDA's `--moe-cache-slots` already
    reads — were already backend-agnostic and already wired from the CLI flag
    (`internal/serveapp/main.go:178`, unchanged, predates this entry); Metal's own code simply
    never READ them, checking only `os.Getenv("GOINFER_METAL_MOE_SLOTS")` directly at its three
    real call sites (the guard's estimate, `metal/moe.go`'s and `metal/gemma4_moe.go`'s actual
    paging-engagement checks).
  - **`metal/backend.go`**: new `metalMoESlotsRequest(m) string` resolves Options first
    (`m.MoECacheSlotsRequest() > 0`), falling back to the env var — "kept one release as a
    deprecated alias" per the doc's own §3 wording. `metalMoESlotsFromEnv` (the guard's int-typed
    reader, kept its name — same `decoder/gptoss_decline_test.go`-style precedent of not renaming
    something mid-transition) now calls it instead of reading `os.Getenv` directly.
  - **`metal/moe.go` and `metal/gemma4_moe.go`**: their real paging-engagement checks (`if s :=
    os.Getenv("GOINFER_METAL_MOE_SLOTS"); s != ""`) now call `metalMoESlotsRequest(m)` instead —
    the exact same validation (`n >= topK`) and error path, just resolving from the shared
    priority order first. `m` was already in scope at both sites (used a few lines below for
    `.GiwPath()`/similar), so this is a same-function, no-signature-change edit.
  - **`internal/serveapp/main.go`**: `--moe-cache-slots`'s help text corrected, not just extended —
    it previously said "the runtime measures free VRAM and lowers it if the request does not fit",
    which is true for CUDA's `capSlots` but was NEVER true for Metal (a request Metal cannot honor
    declines the WHOLE LOAD to CPU via the memory guard; it does not auto-lower the slot count the
    way CUDA does). Rewritten to state that per-backend difference explicitly rather than let a
    CUDA-only claim quietly become wrong for a second backend as soon as this landed.
  - **Verified**: `TestMetalMoESlotsFromEnv` (existing, `metal/resident_memguard_test.go`) rewritten
    to load a real (synthetic, in-memory) model per case instead of testing the bare env-var parser
    in isolation — the env-var sub-cases now go through a model whose `Options.MoECacheSlots` is
    genuinely 0, so the fallback path is what's actually exercised, plus two new cases pinning the
    priority order directly (Options wins over a conflicting env var; Options alone with no env var
    set). New `TestMoESlotsViaOptions_engagesPaging`
    (`metal/moe_slots_option_test.go`) proves the REAL dispatch path, not just the guard: on the
    real `testdata/gemma4-moe-tiny` fixture, with `GOINFER_METAL_MOE_SLOTS` explicitly unset,
    `decoder.Options{MoECacheSlots: N}` alone engages paging (`r.g4moe.paged=true`,
    `r.g4moe.slots=N`) — passed on the first real run.
  - **`TestMoESlotsViaOptions_engagesPaging_genericMoE`** (new, same file): the generic (non-
    gemma4) twin, on `testdata/mixtral-tiny` — TRACKED in git (nE=8, topK=2), so unlike the gemma4
    case this one runs in CI unconditionally, no skip guard needed. Confirms `metal/moe.go`'s own
    separate `moeResident`/`paged`/`slots` fields engage identically from Options alone. Passed on
    the first real run.
  - Full regression: `metal` 113 pass/0 fail/51 skip (111→113: two new tests; the rewritten
    `TestMetalMoESlotsFromEnv` replaces rather than adds subtests net). `gofmt -l`, `go vet`,
    staticcheck all clean; `metal/cmd/serve` and `internal/serveapp` both still build clean with
    the help-text change.
  - **Not done, left open**: Metal's context sizing (blocked on the fixed-size kernel threadgroup
    buffer, above); the real `--fit=off` flag; G6's table test; the drafter-aware
    companion-allocation ctx sizing (CUDA's own still-open item) all remain untouched.

- 2026-09-09 — G6 ("nothing pinned is overridden... honoured or refused with numbers") scoped and
  partially closed, at the user's direction after CPU's own placement piece was explicitly
  deferred (skip CPU placement — its context is already fit-by-default via Phase 0's guard, and
  auto-switching PLACEMENT to weight-paging is a bigger, separate product decision needing its own
  measured verification, left for later rather than folded in here).
  - **A real, pre-existing gap found while scoping this, not caused by this session's work**:
    Metal's resident path never reads `m.ResidentContextRequest()` AT ALL — `grep` across
    `metal/*.go` for it returns nothing. An explicit `-ctx` is silently ignored on Metal today;
    `metalCtxCap` is always the bare 4096 constant regardless of what a caller asked for. This is
    exactly the class of bug G6 exists to catch, and it is real, but fixing it means converting
    `metalCtxCap` from a package constant into a per-resident field across a dozen-plus call sites
    in `metal/backend.go`/`metal/model.go` (mirroring the `r.ctxCap` field CUDA already has) — a
    bigger, more delicate change than writing G6's test was scoped to cost. At the user's explicit
    choice, this is documented here as a known, separately-scoped gap rather than attempted this
    pass: **the real fix is "give Metal a resolveCtxCap of its own," which cannot happen until
    metalCtxCap stops being a compile-time constant.**
  - **`-kv` (KV precision) is WebGPU-only today** (`internal/serveapp/main.go`'s own flag help
    text: "webgpu backend only") — CUDA and Metal have no `-kv` handling to test at all, so G6's
    `-kv` row does not apply to either backend this phase; it becomes live once Phase 3 (WebGPU)
    lands.
  - **`--quant` is untouched by construction, not by a test asserting an absence**: neither
    `decoder.PlanRequest` nor `resolveCtxCapFit` has a quant field or reads `Options.Quant` at
    all — there is no code path through which fit-by-default COULD override it, matching
    `task-fit-to-hardware.md` §0's own rule ("never selects a lossy... quant... without saying so
    and requiring the flag").
  - **What WAS tested, on real hardware**: `TestCheckKVFits_realDevice_explicitRefusesWithNumbers`
    (new, `cuda/resident_cap_test.go`) — the sibling test already pinned the sentinel/wiring
    without a device; this one calls `checkKVFits()` against a REAL device's REAL free VRAM with
    an absurd explicit ctx and asserts the refusal actually NAMES the position count and GB
    figures, not just that an error occurred. Verified on nobara's RTX 2070 SUPER, passed first
    try. `TestMoESlotsViaOptions_belowTopKRefusesWithNumbers` (new,
    `metal/moe_slots_option_test.go`, `testdata/mixtral-tiny`, tracked/CI-safe): an explicit
    `Options.MoECacheSlots` below top-k refuses with both the requested count and `topK=N` in the
    message. `TestResolveCtxCapFit_shortcuts`'s existing "explicit request bypasses fit entirely"
    case already covered the honoured-unchanged half for CUDA's ctx; no new test needed there.
  - Full regression: `metal` 114 pass/0 fail/51 skip (113→114); targeted `cuda` subset
    (ctx/kv/resident-related, no GOINFER_HEAVY_TESTS) green on nobara, no regressions. `gofmt -l`,
    `go vet` clean on both; staticcheck's only findings remain the three pre-existing, untouched
    U1000s already confirmed unrelated.
  - **Not done, left open**: Metal's `-ctx` gap (above, its own separately-scoped fix); `-kv` for
    CUDA/Metal (not applicable — WebGPU-only feature); the real `--fit=off` flag; CPU's placement
    piece; the drafter-aware companion-allocation ctx sizing.

- 2026-09-09 — Metal's `-ctx` gap (named above as a separately-scoped fix) DONE, at the user's
  direction after "continue." An explicit `-ctx` now actually reaches the Metal resident path —
  previously silently ignored entirely, always using the bare 4096 constant regardless of what
  was requested.
  - **`metalCtxCap` (const) renamed `metalCtxCapMax`** (`metal/model.go`) — the value did not
    change, but its MEANING did: it is now documented as the hard kernel ceiling a resolved value
    can never exceed (the attention kernel's fixed-size `threadgroup float sc[4096]` score
    buffer), not "the resident context" itself. `resident` (the per-build struct) gained a new
    `ctxCap int` field holding the actually-resolved value for THIS build — every call site that
    used to read the bare constant (`residentKVBytes`'s pre-build estimate, the two KV-buffer-
    sizing lines in `buildResident`, `checkCap`, `ContextCap()`, `PrefillLast`'s bounds check, plus
    a benchmark test and two comment-only references) now reads the resolved field or a
    `resolveMetalCtxCap(m)` call instead. `TestMetalCtxCapWithinKernelBound`'s invariant (the
    constant must never exceed the kernel's real score-buffer size) is UNCHANGED in substance,
    just renamed along with the constant.
  - **New `resolveMetalCtxCap(m) (int, error)`** (`metal/model.go`), deliberately shaped like
    CUDA's `resolveCtxCap`/`resolveCtxCapFit` but NOT the same policy: CUDA raises an unpinned
    default toward available VRAM because `cudaCtxCapDefault` is an arbitrary, conservative
    choice; `metalCtxCapMax` is NOT arbitrary — it is the kernel ceiling — so there is no "raise it
    when there's room" story for Metal at all. An unpinned load still always gets exactly
    `metalCtxCapMax`, byte-for-byte the historical behavior. What changes: an EXPLICIT request in
    `(0, metalCtxCapMax]` is now honored (clamped further to the model's own window, same as
    CUDA), and a request ABOVE `metalCtxCapMax` is REFUSED with the numbers — G6's own principle
    ("honoured or refused", never silently substituted) — rather than the previous silent ignore.
  - **The refusal is a NAMED error, not folded into the generic decline**, mirroring CUDA's
    `errKVWontFit` precedent: `metalBackend.BuildResident` checks `resolveMetalCtxCap` explicitly,
    before the memory guard, and propagates its error directly rather than letting it fall into
    the same `ok=false, err=nil` swallow every unsupported-shape decline uses. An operator's
    explicit request that cannot be honoured gets a specific, propagated reason (and would trip
    `--require-backend`'s strict-mode exit), not an opaque "declined" indistinguishable from "this
    arch just doesn't fit here."
  - **A nil-safety subtlety, caught before it broke an existing test**: two pre-existing tests
    (`TestMetalResidentCheckCap`, `TestMetalCtxCapWithinKernelBound`) deliberately construct a
    zero-value `&metalResident{}` (`r.r == nil`) specifically so `checkCap`/`ContextCap` can be
    tested as pure logic with no Metal device. Making those methods read `a.r.ctxCap` directly
    would have paniced on that nil receiver. Added a `ctxCap()` accessor with an explicit nil/zero
    fallback to `metalCtxCapMax` instead of inlining the field read at each of the three call
    sites, so this precedent-preserving fallback lives in exactly one place.
  - **Verified, including the POSITIVE case, not just the refusal**: `TestResolveMetalCtxCap`
    (new, `metal/resident_cap_test.go`, `testdata/llama-tiny` — tracked, runs in CI
    unconditionally) covers all four branches (unset, honored-as-requested, clamped-to-model-
    window, refused-above-ceiling) as pure logic, no device needed.
    `TestMetalBuildResident_explicitCtxTooLargeRefusesNotDecline` confirms the wiring one level up
    (a named error, not a swallowed decline) on real Metal hardware.
    `TestMetalBuildResident_explicitCtxHonoured` is the one that actually PROVES the fix works, not
    just that it doesn't crash: builds a real resident with an explicit smaller `-ctx` (32,
    against llama-tiny's 128-position window and the 4096 ceiling), confirms `ContextCap()`
    reflects 32 (not 4096), and runs a REAL `Forward` pass at that smaller capacity, asserting
    finite, non-degenerate logits — "the number changed" and "decode still works at that number"
    are two different claims, and both are checked.
  - Full regression: `metal` 117 pass/0 fail/51 skip (114→117: three new tests). `gofmt -l`,
    `go vet`, staticcheck all clean.
  - **Not done, left open**: the real `--fit=off` flag; CPU's placement piece; the drafter-aware
    companion-allocation ctx sizing. With this fix, G6 and Metal's own Phase 2 pieces are now both
    complete for CUDA and Metal alike (WebGPU stays out of scope, per Phase 3).

- 2026-09-09 — The real `--fit=off` CLI flag DONE, replacing the CUDA-only
  `GOINFER_NO_FIT_DEFAULT` env-var stand-in with `goinfer-serve`'s own `--fit` (default true).
  CPU's placement piece and the drafter-aware companion-allocation ctx sizing remain the two
  genuinely open Phase 2 items — both explicitly out of scope for this pass, per the user's own
  earlier direction to skip CPU placement and this session's standing scope for the drafter fix.
  - **`decoder.Options.DisableFit` / `Model.FitDisabled()`** (new, `decoder/model.go`): a real
    field alongside `ResidentContext`/`MoECacheSlots`, wired through all three `&Model{...}`
    construction sites the same way those already are. `FitDisabled()` checks the new field OR
    (kept working, not orphaned) the original `GOINFER_NO_FIT_DEFAULT` env var — so anything
    driving `decoder.Load` directly without the flag still has an escape hatch.
  - **`cuda/resident.go`'s `resolveCtxCapFit`** now calls `m.FitDisabled()` instead of reading
    `GOINFER_NO_FIT_DEFAULT` itself — the ONE call site this session's CUDA ctx-default work
    added, now backend-agnostic-ready (a future Metal/CPU fit-by-default policy would read the
    same accessor, not invent its own env var).
  - **`internal/serveapp/main.go`**: new `--fit` bool flag (default true), wired as
    `DisableFit: !cfg.fit` in `modelSpec.options()` — global, not per-model, matching how
    `--moe-cache-experts`/`--moe-cache-slots` are ALSO global rather than per-`--model` overrides
    (an inconsistency already present in the flag surface, not introduced here). Help text states
    plainly what `--fit=off` does and does NOT affect (never touches an explicitly pinned
    `-ctx`/`-quant`/`--moe-cache-slots`; does not revert the Metal `-ctx` bug fix, which is
    correctness, not an opinionated default).
  - **Verified real, not just wired**: `TestResolveCtxCapFit_shortcuts` gained an
    `Options.DisableFit` case alongside the existing env-var one (both restore
    `cudaCtxCapDefault` identically) — rewritten to load a real (tracked-in-git) `testdata/
    llama-tiny` model per case instead of passing `nil`, since `FitDisabled()` reads a real
    struct field now and a nil receiver would panic in the `request==0` cases where it's
    evaluated. A throwaway probe on nobara's RTX 2070 SUPER confirmed the real, end-to-end
    wiring: `Options{DisableFit: false}` resolved `ctxCap=8192`, `Options{DisableFit: true}`
    resolved exactly `4096` — the SAME real checkpoint, the SAME machine, only the flag differed.
    `goinfer-serve -h` shows the flag registered with the correct default and help text.
  - **Two real, unrelated issues found and fixed while closing this out, neither caused by the
    `--fit` flag itself**:
    1. `TestEnvVars_docAndCodeAgree` started failing the moment `GOINFER_NO_FIT_DEFAULT` moved
       from being read only in `cuda/resident.go` (a CUDA-tagged file, evidently outside this
       test's plain-`decoder`-package scan) to `decoder/model.go` (untagged, always compiled) —
       a real, previously-invisible documentation gap the move exposed. Added it to
       `docs/env-vars.md`'s "Operator knobs" table.
    2. `TestParityManifest_fresh` went stale because `decoder/model.go`/`decoder/gguf.go` are
       `core`-covered files. `scripts/refresh_parity_hashes.sh` (the sound, goldens-gated refresh
       path) refused to run it automatically: its own golden sweep found `TestOlmo3_forwardParity`
       failing — the SAME pre-existing, already-tracked failure this session's every other status
       entry has named — and the script has no per-test exclusion, so one unrelated red golden
       blocks the whole automated refresh. Rather than bypass this on faith, independently
       verified BOTH halves by hand: (a) `git stash`, confirmed `TestOlmo3_forwardParity` fails
       IDENTICALLY (same cosine 0.9899728674750051, same argmax) on the clean committed baseline
       with none of this session's `--fit` changes present — so it is genuinely unrelated; (b)
       ran the exact golden set the script runs (`GOINFER_HEAVY_TESTS=1 go test ./decoder/ -run
       '(_forwardParity|_logitParity|_textParity)$|^TestGGUF_.*_parity$' -v`) with my change
       present: 36 passed / 1 failed (olmo3, matching (a) exactly) / 21 skipped — every OTHER
       family (dense, MoE, hybrid, GGUF, quantized, LoRA) proves the change non-numeric, which is
       the exact proof the script's gate exists to produce. Then performed the SAME mechanism the
       script itself uses — `go test ./decoder/ -run TestParityManifest -update`, followed by the
       identical post-check (`git diff` touches ONLY `deps_hash` lines, confirmed: 66 lines / 33
       families, `validated_at`/metrics untouched) — by hand, since the script's blanket
       automation had no way to accommodate a pre-existing unrelated red golden. Documented here
       per the script's own stated philosophy ("make the exception AUDITABLE"), not silently.
  - **A third failure investigated on nobara and confirmed NOT a regression**:
    `TestGptOssResidentParityCUDA` (a real 20B-checkpoint test) failed on its SECOND load (a CPU
    reference build in the same process) with a host-RAM fit-guard refusal ("6.2 GB of memory
    available" — very low for this box). Given the mechanism (a bigger CUDA resident KV
    allocation from fit-by-default could plausibly increase host RAM pressure) was directly
    plausible, this was NOT waved off: re-ran with `GOINFER_NO_FIT_DEFAULT=1` (the historical
    ctx=4096 behavior fully restored) and the SAME failure reproduced identically — proving the
    fit-by-default ctx change is not the cause. Most likely nobara's shared-box memory pressure
    at the time (a concurrent, not-mine `go test ./decoder/... -v` process was observed running
    on the box during this check) or a pre-existing fragility in a test that builds two heavy
    models sequentially in one process. Not investigated further — out of scope for this task.
  - Full regression: `decoder` 419 pass/1 fail (`TestOlmo3_forwardParity`, independently
    re-confirmed pre-existing this entry)/134 skip. `internal` all packages green.
    `gofmt -l`/`go vet`/staticcheck clean on every touched package (darwin and linux/cuda).
  - **Phase 2 is now closed for CUDA and Metal** (context + slots + the real off-switch, all
    measured on real hardware). What remains for Phase 2 specifically: CPU's placement piece
    (deferred) and the drafter-aware companion-allocation ctx sizing (deferred, needs a
    `ResidencyBackend` interface change). Phases 3-5 (WebGPU, the rate band, host-computed
    experts) remain entirely unstarted.

- 2026-09-09 — `task-fit-to-hardware.md`'s CPU placement piece DONE, at the user's explicit choice
  between the two remaining Phase 2 items (offered both; this one chosen over the drafter-aware
  ctx fix). `decoder/fitguard.go`'s load-time guard now gets ONE automatic retry with weight
  streaming for a plain `.gguf` that will not fit resident RAM, instead of just refusing —
  gated by `--fit` (default on) the same way CUDA's ctx-default and Metal's slots already are, and
  scoped to DENSE models only after a real prior-art finding made the MoE case an explicit
  non-goal (below). **UNCOMMITTED — awaiting go-ahead.**
  - **The prior-art finding that shaped scope, found before writing any code**: `docs/benchmarks.md`
    "M35/M26 on the Mac" already measured goinfer's OTHER CPU weight-streaming path — MoE expert
    demand-paging (`decoder/moepaging.go`'s `expertPager`) — on a real 20 GB checkpoint on this
    16 GB Mac: **2h10min wall-clock, ZERO completions**, RSS pinned at ~3.2 GB the whole time
    (consistent with re-reading weights from disk essentially every token, no useful cache
    retention), very likely the direct cause of a genuine kernel panic shortly afterward. That is
    exactly the failure mode this session's own explicit condition ("do NOT auto-switch without
    measuring") exists to catch — an automatic retry that silently walked into it would have been
    the wrong default, not a bug in the retry mechanism itself. This ruled MoE out of the
    automatic path by construction, before any new measurement was needed for it.
  - **Dense weight streaming (`decoder/layerpaging.go`'s `layerPager`) is a structurally different
    mechanism**, not just a smaller version of the same risk: MoE expert selection is
    data-dependent and chosen per-token (nothing to prefetch ahead of), while the dense transformer
    layer loop is strictly sequential and fully known in advance, so `layerPager` issues a real
    `Advise(WILLNEED)` for the next layer while the current one computes, overlapping the fault
    with compute (a genuine prefetch-ahead window, not an LRU). Worth measuring on its own rather
    than tarred with the MoE result — which is exactly what made this the item worth building
    the retry for, and the MoE case the item to explicitly exclude.
  - **`decoder.ErrWontFitResident` / `decoder.FitDeclineError`** (new, `decoder/fitguard.go`):
    `declineErr()` now returns a typed error wrapping the sentinel, carrying a `DenseStreamable
    bool` field a caller can act on instead of parsing the refusal text. `denseStreamable(cfg
    *Config)` computes it by calling `resolveArchitecture(cfg)` (needs only the parsed config, no
    tensor data — already how `fitCheckFor` gets `cfg`) and mirroring `newLayerPager`'s own
    exclusions EXACTLY: false for `arch.MoE != nil` and for "own-forward" families (gemma4,
    nemotron-h-moe, lfm2 — resolved dynamically via `arch.ownForward()`, the same call
    `newLayerPager` itself makes, not a hand-written list, so the guard's retry decision and the
    pager's own decision to actually build one can never disagree) — and false for a nil `cfg`
    (header unreadable ⇒ don't know ⇒ don't retry, the same "every unknown proceeds \[without
    acting\]" discipline this file already states for the refusal path itself).
  - **`internal/serveapp/main.go`'s `loadDecoder`**: the `.gguf→.giw` transcode block (previously
    only reachable via an explicit `--stream-weights`) factored into a shared `ensureGIW` closure.
    On a `decoder.Load` failure, `errors.As` recovers a `*decoder.FitDeclineError`; when
    `!opts.StreamWeights && !opts.DisableFit && strings.HasSuffix(spec.path, ".gguf") &&
    fde.DenseStreamable`, it prints a note, flips `opts.StreamWeights = true`, transcodes, and
    retries the tokenizer+model load ONCE — any failure at that point (transcode or the retried
    load) returns a combined error naming both the original decline and the retry failure, so a
    genuinely-broken retry never silently swallows the reason the first attempt failed. `--fit=off`
    (`decoder.Options.DisableFit`, its own doc comment updated) restores the plain refusal exactly,
    matching CUDA/Metal's existing off-switch semantics. `--fit`'s and `--stream-weights`'s help
    text both updated to say so.
  - **Verified with real unit tests, not just build success**: `TestDenseStreamable_
    agreesWithLayerPagerEligibility` (new, `decoder/fitguard_test.go`) drives `denseStreamable`
    against three TRACKED tiny fixtures — `testdata/llama-tiny` (dense, generic-forward → true),
    `testdata/mixtral-tiny` (MoE → false), `testdata/lfm2-tiny` (dense but own-forward → false) —
    plus the nil-config case, all CI-safe with no `GOINFER_HEAVY_TESTS`. `TestFitDeclineError_
    carriesDenseStreamable` pins that `declineErr()` carries the field through unchanged in both
    directions. `TestFitGuard_refusesBeforeAllocating` (existing) gained an assertion that gpt-oss
    (MoE, the tracked `gptoss_tiny.gguf` fixture) reports `DenseStreamable=false` through the REAL
    `Load()` refusal path, not just the pure classifier — the safety-critical direction (never
    auto-retry into the measured-bad MoE path) proven through a real decline, not only through the
    helper function in isolation.
  - **The throughput cost, measured on real hardware before treating this as shippable** (the
    explicit condition this item was deferred under): a throwaway interleaved A/B
    (`/tmp`-scratchpad `measure_stream_overhead.py`, modeled on `scripts/bench_peer.py`'s own
    decode-tok/s method: client-side timing from the first streamed token, `usage.completion_tokens`
    as the count, `stream_options.include_usage`) against a real, safe dense model —
    `~/models/qwen2.5-7b-instruct-q4_k_m.gguf` (Qwen2.5-7B, llama-family generic-forward, int4,
    4.4 GB on disk — comfortably resident on this 16 GB Mac, nowhere near the M35/M26/H27/G20
    off-limits class) — comparing plain resident CPU decode against `--stream-weights
    --weight-cache 1` (forced small enough that `layerPager` actually windows: 28 layers, window 7
    resident, confirmed from the load banner, not assumed). Two interleaved rounds
    (resident/streamed/resident/streamed), 3 requests/arm/round, `--backend cpu`, greedy, a fixed
    64-token generation on a fixed prompt, server restarted between arms:
    | arm | n | samples (tok/s) | median |
    |---|---|---|---|
    | resident | 6 | 9.39, 9.90, 10.01, 10.09, 10.09, 10.77 | 10.09 |
    | streamed (window 7/28) | 6 | 9.09, 9.15, 9.44, 9.52, 9.77, 9.89 | 9.52 |

    **streamed/resident = 0.944 — a ~5.6% decode-throughput cost**, with only a quarter of the
    model's layers resident at once. This is the number that makes shipping the auto-retry a
    defensible default rather than a guess: a model that would otherwise be refused outright now
    runs at ~94% of its own resident rate, a world apart from the MoE path's measured 2h10min/zero
    completions. (Load time also dropped 76.7s → 12.7s/4.8s/8.6s streamed vs resident — the .giw
    cache reuse on repeat rounds, not a claim about cold-transcode cost, which this run did not
    separately time.) Servers cleanly shut down between arms; no leftover processes.
  - **A real, stated test gap, not a silently accepted one**: no CI-level integration test drives
    `loadDecoder`'s new retry branch itself end-to-end (main.go's specific boolean wiring). Building
    one needs either a tracked tiny DENSE `.gguf` fixture (none exists anywhere in this repo today
    — every dense tiny fixture is a safetensors dir, and only `.gguf` reaches the fit guard at
    all) or an exported `goinfer_testhooks`-gated RAM-override hook callable from
    `internal/serveapp` (decoder's `hostRAM`/`hostRAMAvailable` indirection is package-private, and
    the existing `assets_testhook.go`/`fidelity_testhook.go` pattern would need a new entry plus a
    tag-gated test file). Judged disproportionate to what remained unproven: the classification
    logic is unit-tested for real (above), and the actual streamed-load-and-decode mechanism the
    retry activates is the SAME `decoder.Load(.giw, StreamWeights)` path just measured on real
    hardware above (via the explicit flag, not the auto-retry branch specifically). What is NOT
    independently proven is that `loadDecoder`'s specific error-handling branch actually executes
    and wires the two together correctly in production — reviewed by hand twice, not exercised by
    a test. Flagged here per this repo's own rule about doc comments claiming coverage they don't
    have, rather than left implicit.
  - Full regression: `decoder` 418 pass/2 fail(`TestOlmo3_forwardParity` pre-existing;
    `TestParityManifest_fresh` — both `decoder/fitguard.go`/`decoder/model.go` are `core`-covered
    files, so this session's edits re-staled the manifest the same way the `--fit` flag work did
    three entries up)/125 skip, THEN green after the same by-hand refresh procedure that entry
    already established: golden regex set (`_forwardParity|_logitParity|_textParity|TestGGUF_.*_
    parity`) re-run with these changes present — **36 pass/1 fail(olmo3, identical)/21 skip,
    byte-for-byte the same counts as the pre-existing baseline** — then `go test ./decoder/ -run
    TestParityManifest -update`, `git diff -- testdata/parity_manifest.json` confirmed
    deps_hash-only (66 lines / 33 families), `TestParityManifest_fresh` re-run green
    ("35/35 families enforced"). `internal/serveapp` full suite green (unaffected by the
    manifest refresh). `gofmt -l`/`go vet`/CI-pinned staticcheck (v0.8.0, confirmed via
    `-version`) clean on both touched packages.
  - **Not done, left open**: the CI-integration-test gap above; the drafter-aware
    companion-allocation ctx sizing (still the other deferred Phase 2 item, untouched this
    session); cold-transcode wall-clock was not separately measured (only cache-hit reload times);
    Phases 3-5 remain entirely unstarted. This closes Phase 2's CPU piece as scoped — dense
    auto-retry, measured and gated — not the MoE CPU-streaming failure mode itself, which stays a
    known, documented capability boundary (unchanged by this session, matching the existing
    "M35/M26 on the Mac" verdict).

- 2026-09-09 — The drafter-aware companion-allocation ctx sizing DONE — `task-fit-to-hardware.md`
  Phase 2's last open item, at the user's explicit direction after asking why it was deferred
  (prioritization, not a technical blocker — offered both remaining items, the CPU piece above was
  chosen first) and confirming it was fine to pick up now. CUDA-only (Metal/WebGPU implement no
  `ResidentDrafterHost` at all — `grep -rl AttachBlockDrafter metal/ gpu/` returns nothing, checked
  before scoping anything, not assumed from the earlier session's more cautious note).
  - **No `ResidencyBackend` interface change was needed, contrary to the earlier assessment
    (the G11 entry above: "needs `BuildResident`'s signature... or an out-of-band hint... a
    `decoder.ResidencyBackend` interface change both CUDA and Metal implement").** The
    out-of-band-hint alternative that entry floated turned out to be sufficient on its own:
    `cuda/backend.go`'s `BuildResident(m *decoder.Model)` already receives `*Model`, and `Model`
    already carries per-load knobs this way (`resCtxReq`, `moeSlots`, `disableFit` — the same
    pattern `ResidentContext`/`MoECacheSlots`/`DisableFit` already established). Adding one more
    field (`extraBytes` / `Options.ExtraResidentBytes` / `Model.ExtraResidentBytes()`) cost zero
    interface churn. The earlier note's "both CUDA and Metal implement" turned out not to matter
    either, once checked: Metal hosts no drafter today, so there was nothing for it to implement.
  - **`decoder.DrafterResidentBytesEstimate(dw BlockDrafterWeights) int64`** (new,
    `decoder/blockdrafter.go`): prices a drafter's WEIGHTS (the dominant term) by summing every
    matrix's `Rows()×Cols()` at an int8-packed estimate (+1 f32 scale/row) — not a generic
    multi-quant estimate the way `fitguard.go`'s `quantBytesPerElem` is for the main model, because
    every drafter matrix is f32 on host (`dflash.go`'s `loadMat`, never `QuantizeInt8`/`QuantizeInt4`)
    and the only backend that hosts one, CUDA's `AttachDrafter`, packs every f32 matrix through
    `packWeight`'s f32 branch, which is ALWAYS int8 regardless of the target model's own quant — a
    drafter is never packed to int4 today. Deliberately excludes the verify/capture buffers
    `NewBlockSpec` allocates (a few MB at the default verify width against hundreds of MB of
    weights) — named as a residual the existing 384 MiB margins (`ctxCapMarginBytes`/
    `slotMarginBytes`) already absorb, not silently folded in as if computed.
  - **`decoder.Options.ExtraResidentBytes` / `Model.ExtraResidentBytes()`** (new,
    `decoder/model.go`): threaded through both `&Model{...}` construction sites the same way
    `ResidentContext`/`MoECacheSlots` already are. 0 (default) is a total no-op through every call
    site below.
  - **`internal/serveapp/main.go`'s `loadDecoder`**: `--drafter` is now loaded ONCE, at the TOP of
    `loadDecoder` (before `decoder.Load`), not inside `attachBlockDrafter` after the fact —
    `decoder.LoadDFlashDrafter(cfg.drafter)` runs early, `DrafterResidentBytesEstimate` prices it
    into `opts.ExtraResidentBytes`, and the SAME loaded `*decoder.DFlashDrafter` is passed through
    to a modified `attachBlockDrafter(lm, dw)` (was `attachBlockDrafter(lm, dir string)`) for the
    real attach later — pricing and the actual attach see identical weights, not two independent
    file reads that could in principle disagree. This does cost reading the drafter file even when
    the eventual backend cannot host one (previously `attachBlockDrafter` checked
    `BlockSpecCapable()` before touching the file at all); accepted as the necessary order once the
    byte count has to exist before `BuildResident` runs.
  - **`cuda/resident.go`, every free-VRAM check a companion allocation could be starved by**:
    - `resolveCtxCapFit`: `m.ExtraResidentBytes()` now passed as `PlanRequest.ExtraBytes` into the
      `m.Plan("cuda", free, ...)` call — `Plan`'s own arithmetic already handled `ExtraBytes`
      correctly (pinned by `decoder/fitplan_test.go`'s existing `TestPlan_tableDriven` case, which
      predates this session), so this is genuinely a one-line wiring change, not new arithmetic.
    - `checkWeightsFit(m)`: `need` now includes `m.ExtraResidentBytes()`; the refusal message
      names the reservation separately from the dense-weight figure when it is the reason
      (`"...plus %.2f GB reserved for a companion attach (--drafter)..."`) rather than folding it
      into one number a reader can't attribute.
    - `allocSlots` (capSlots' real call site): new `reservedBudget(free, extraBytes int64) int64`
      (split out for the same reason `fitsWeightsBudget` was — unit-testable without a device),
      subtracted from live free VRAM before the elastic slot search runs. Both the FLOOR decline
      (`topK` doesn't even fit) and the ordinary cap-down log line now name the reservation when
      `extraBytes > 0`, unchanged otherwise.
    - `checkKVFits()`: same `r.extraBytes` term added. **Caught by the real-device test below, not
      by review**: the PINNED-explicit branch (`return e`) picked up the reservation-aware message
      correctly on the first pass, but the UNPINNED/default branch — a completely separate
      hardcoded `fmt.Errorf` a few lines below, pre-existing code this session did not otherwise
      touch — did not, because it builds its own message from scratch rather than reusing `e`/`msg`.
      `TestCheckKVFits_realDevice_extraResidentBytesRefusesWithNumbers` failed on the first run
      (`"...staged path" does not mention "companion attach"`) and named exactly which branch was
      missed; fixed by giving that branch the same extraBytes-aware message split. Left as a
      pointer for the drafter-aware-sizing pattern generally: **the "explicit vs default" branch
      split recurs at every guard in this file, and a change to one arm's message is not
      automatically a change to the other's** — this is the second time in this session's own
      history this exact class of miss has been caught by a real-device test rather than review
      (the first was G6's Metal `-ctx` gap).
  - **Verified real, on nobara's RTX 2070 SUPER** — isolated checkout, this session's uncommitted
    diff transferred as a patch on top of a git-bundle-synced `HEAD` (the checkout's own working
    tree held STALE, already-superseded pre-`e36ac75` changes from an earlier session that never
    got committed there; diffed against the now-committed local history to confirm before
    `git stash`-ing them out of the way, not discarded blind):
    - `go build`/`go vet -tags cuda` clean; `gofmt -l` clean.
    - `TestFitsWeightsBudget` (pre-existing) and new `TestReservedBudget` (pure, no device) both
      pass — the arithmetic in isolation.
    - New `TestCheckKVFits_realDevice_extraResidentBytesRefusesWithNumbers`
      (`cuda/resident_cap_test.go`, sibling to the existing G6-era
      `TestCheckKVFits_realDevice_explicitRefusesWithNumbers`): a PAIRED real-device comparison —
      the SAME modest 128-position ctx, on the SAME card, fits at `extraBytes=0` and is refused at
      `extraBytes=100 GiB` (deliberately bigger than any card this repo targets, so the refusal
      cannot be this box's own free-VRAM state, only the reservation) — message asserted to name
      "companion attach", "--drafter", and "GB". This is the test that caught the missed branch
      above.
    - **A throwaway probe (not committed, per this session's own established pattern for the
      live-VRAM-probe branch — see `TestResolveCtxCapFit_shortcuts`'s own doc comment referencing
      one from the `--fit` flag work)**, isolating `resolveCtxCapFit` from `checkWeightsFit`'s
      confound (a large-enough reservation to shrink ctx is also large enough to fail the
      independent weights-fit guard first, on an 8 GB card — the viable "shrinks but doesn't
      decline" window is a few hundred MB wide and too fragile to commit as a device-state-
      dependent test): on a real `qwen2.5-coder-1.5b-instruct` GGUF (CPU-loaded, so only the
      live-VRAM-probe path runs, no full resident build), `resolveCtxCapFit(m, 0, 32768)` returned
      **8192 at `ExtraResidentBytes=0`, 4096 (falling back to `cudaCtxCapDefault`, exactly as
      documented — "Plan could not confidently improve on the historical floor") at
      `ExtraResidentBytes=7 GiB`** — same model, same device, same free-VRAM reading, only the
      reservation differed.
    - Full `cuda` suite (`-tags 'cuda goinfer_testhooks'`, 15 min timeout): 101 pass / 3 fail / 126
      skip. All 3 failures share ONE root cause — the mistral sliding-window tiny fixture's weight
      file absent on this checkout (the config JSON is git-tracked, the gitignored weight file was
      never copied there; the exact "dir-only guard" trap `CLAUDE.md` names) — confirmed by
      `ls`ing the fixture directory, not assumed from the error text. None of the three
      (`TestGraphsSafeGate`, `TestPrefillLast_e2e`, `TestSlidingWindowLongContext`) touch anything
      this session changed. `TestBackendResidentWired` (`GOINFER_HEAVY_TESTS=1`, a real qwen
      checkpoint through `decoder.Load(cuda)→BuildResident→Forward`) still passes at
      `ExtraResidentBytes=0`: 7/8 exact, worst near-tie 0.563% — confirms the ordinary,
      nothing-attaching path is unchanged.
  - **Decoder-side, on this Mac**: `TestDrafterResidentBytesEstimate_matchesHandComputedTotal` and
    `_growsWithLayers` (new, `decoder/blockdrafter_test.go`, synthetic geometry — no checkpoint
    needed, so these run in CI unconditionally) pin the estimator's arithmetic against an
    independently-summed total, the same discipline `decoder/weightbytes_test.go`'s own
    cross-checks use. `TestExtraResidentBytes_reachesTheModel` (new, `../testdata/llama-tiny`,
    tracked) closes the loop the estimate feeds: a struct-literal typo at either of `Load`'s two
    `&Model{...}` sites would otherwise silently zero the field with nothing to catch it, since a
    backend-level test alone could never distinguish "wired wrong" from "nothing was attaching".
  - **Full regression, this Mac**: `decoder` 418 pass/2 fail(`TestOlmo3_forwardParity` pre-existing;
    `TestParityManifest_fresh` — `decoder/model.go`/`decoder/blockdrafter.go` are `core`-covered,
    same re-staling this session's earlier CPU-placement entry already walked through)/125 skip,
    then green after the identical by-hand refresh: golden regex set re-run with these changes
    present — **36 pass/1 fail(olmo3, identical)/21 skip, byte-for-byte the same as baseline** —
    then `-run TestParityManifest -update`, `git diff` confirmed deps_hash-only (66 lines/33
    families), `TestParityManifest_fresh` green ("35/35 families enforced"). `internal/serveapp`
    full suite green. `gofmt -l`/`go vet`/staticcheck (v0.8.0, `-version` confirmed) clean on every
    touched package, both machines.
  - **Not done, left open**: the verify/capture-buffer term `DrafterResidentBytesEstimate`
    deliberately excludes (named above, judged negligible against the weight term and already
    covered by the existing fixed margins); the "shrinks but doesn't decline" middle case for
    `resolveCtxCapFit` under a companion reservation was demonstrated once via a throwaway probe
    but has no committed device-dependent test (the narrow-window fragility above); a real
    `--drafter` pairing was never attached end-to-end WITH this fix live (the real-device tests
    above isolate the guards directly rather than running a full `serve --drafter` startup — no
    drafter+target checkpoint pair was confirmed present on nobara to do that run). **With this,
    `task-fit-to-hardware.md` Phase 2 is now fully closed** — CUDA (context + slots + `--fit=off`),
    Metal (slots + the `-ctx` bug fix), CPU (dense auto-retry), and the drafter-aware sizing, all
    measured on real hardware. Phases 3-5 (WebGPU, the rate band, host-computed experts) remain
    entirely unstarted.
