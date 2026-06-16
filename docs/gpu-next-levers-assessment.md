# GPU backend — what's left worth doing (next-levers assessment)

> Internal planning doc (gitignored, like `gpu-assessment.md`). Question:
> **the GPU decode arc is CLOSED (`gpu-assessment.md` §0.0) — so what GPU work
> remains that is actually worth a maintainer's weeks?** Grounded in the live
> `gpu/` module and `decoder/` seam as of go.mod `v0.7.0` / gpu `v0.23.0`
> (read 2026-06-15). Same rule as `gpu-assessment.md`: no payoff claim without a
> traceable source — either an in-repo measurement (cited) or marked as a
> *prediction to measure on the 2070S*, never a guess dressed as a number.

## TL;DR

Decode is at the WGSL wall (~90 tok/s int8 on the 2070 SUPER, 61% of
Ollama-CUDA at equal quant — `gpu-assessment.md` §0.0). That wall is structural,
not a grind target: WGSL cannot express the single-dispatch megakernel that is
CUDA/Metal's edge. **Stop optimizing single-stream decode.** Three levers remain
that are *not* walled, ranked by (certainty × payoff) / effort:

1. **Unblock `dot4I8Packed`** — the upstream blocker the repo records as open is
   now *merged upstream*. Cheapest, highest-certainty. Gates batched prefill.
2. **GPU speculative decoding** — the explicitly-parked lever; the exact machinery
   already exists but falls back to staged on GPU. Biggest decode upside, real
   uncertainty against the glue wall.
3. **wasm/browser inference** — `demo/gemma-web` is already a scaffold; turning it
   client-side is outsized marketing for bounded effort, low perf risk.

Everything else GPU (more decode fusion, K-quant-in-shader, MoE indirect routing,
native CUDA path) is either walled, deferred-with-reason, or off-strategy. Section
6 says why, so they don't get re-litigated.

---

## 0. Where the backend actually is (live inventory, 2026-06-15)

Confirmed by reading the code, not the older docs:

- **Resident decode path exists and is the default GPU path** for eligible archs.
  `decoder/residency.go` defines the seam: `ResidentForward` (one interface,
  `Forward(embedding, pos) → logits`, plus `UploadKV` / `Reset` / `Close`) and
  `ResidencyBackend.BuildResident(m) (rf, ok, err)`. `DecodeRunnerEligible()`
  gates it to the dense Qwen2/Llama shape (no MoE/Gemma4/qwen3_5, gated SwiGLU,
  pre-2 RMSNorm, full RoPE, standard GQA; q/k/v bias + (1+w) RMS offset allowed).
- **The GPU runner is M=1 only.** `gpu/decoderunner.go:277` — `Run(x []float32,
  pos int) ([]float32, error)`. One token in, one logit row out, appends one KV
  position. There is no batched on-device forward.
- **`dot4I8Packed` is probed but assumed absent.** `gpu/spike_test.go` compiles a
  one-line `dot4I8Packed` shader at adapter init and logs SUPPORTED / NOT. The
  W8A8 GEMV (`gpu/quant.go:15`) ships the *unpacked* int8 fallback ("no
  dot4I8Packed builtin, so the kernel unpacks four sign-extended int8 per u32").
  `gpu/attention.go:273` flags attention scores as f32 "no integer dot / DP4A — a
  free future upgrade."
- **Spec-decode machinery is complete and exact, but CPU-bound.**
  `decoder/speculative.go` — `GenerateSpeculative(ctx, prompt, maxTokens, draft,
  K, sp)`; greedy-only, vocab-matched draft/target, verify = one target pass over
  `[cur, draftTok...]`, greedy accept-prefix, exact-vs-plain-greedy gate
  (`TestSpeculativeGreedyParity`). The verify pass is `Model.forwardN`
  (`decoder/forwardn.go:386`) — a batched M=K forward that **only exists on CPU**.
- **`residency.go` documents the exact reason spec-decode can't use the GPU:**
  "Session prefix-reuse and GenerateSpeculative drive a CPU-resident KVCache that
  the GPU-resident KV can't transparently share, so those requests fall back to
  the staged path." So GPU spec-decode is blocked on two missing resident
  capabilities: a **batched verify** and a **KV rollback**.
- **wasm scaffold is real.** `demo/gemma-web/` already has `main.go` + `index.html`
  + README — an HTTP demo, not yet client-side WebGPU.

---

## 1. Lever 1 — Unblock `dot4I8Packed` (DP4A). Cheapest, do first.

**The finding that changes the call.** Every prior doc (`gpu-assessment.md` §0.5,
`roadmap.md` "GPU decode performance", `task-gpu-batched-prefill.md`) records
`dot4I8Packed` as **upstream-blocked on `cogentcore/webgpu`**. That status is
stale. The underlying `gfx-rs/wgpu` (which `cogentcore/webgpu` wraps) **merged**
the builtins — `dot4I8Packed`/`dot4U8Packed` in PR #7494, and the
`NATIVE_PACKED_INTEGER_DOT_PRODUCT` path lowering to `VK_KHR_shader_integer_dot_product`
(exactly the 2070S's TU104 DP4A units) in PR #7595. Chrome's own writeup measures
**1.7–2.9×** on 8-bit ML workloads. The repo pins `gpu v0.23.0`; the question is
purely whether a current `cogentcore/webgpu` tag re-exports a `wgpu` new enough to
carry these.

**Why this is the first move.** It is the only lever whose risk is *a version bump
and a re-run of a test you already wrote* — `TestSpike_capabilities` already prints
SUPPORTED/NOT for `dot4I8Packed`. No new numerics surface, no architecture change.

**What it unblocks (in priority order):**

- **Batched on-device GPU prefill.** `task-gpu-batched-prefill.md` + `roadmap.md`
  concluded prefill batching is "a wash until DP4A lands" because the naive WGSL
  GEMM is bandwidth-bound at ~748 GB/s and loses to the M=1 GEMV; the in-repo
  measurement was 680 GFLOP/s (only 1.54× the naive path). DP4A is the documented
  precondition to "clear the 748 wall." This is the **long-prompt TTFT win** — the
  thing that most helps the agent/serve loop (re-sending long prompts), which
  matters more than raw decode for the daily-tool story.
- **A modest decode bump.** The W8A8 GEMV currently hand-unpacks 4×int8/u32; on
  DP4A hardware the packed builtin replaces that ALU with one instruction. Decode
  is glue-bound not gemv-bound (`gpu-assessment.md` §0.0: gemv is 4.3 ms of a
  9.7 ms token, *at roofline*), so do **not** expect a token-rate jump — the gemv
  is already bandwidth-bound, and packing helps ALU not bandwidth. Predict ≤1.05×
  decode, real prefill gain. Bill it as a prefill lever.

**Plan (≈3–6 days):**

1. Bump `cogentcore/webgpu`; run `go test -tags gpu -run TestSpike_capabilities`
   on the 2070S. Branch on the logged result:
   - SUPPORTED → proceed.
   - NOT → size the upstream gap (is it the Go binding not enabling the WGSL
     extension, or the wrapped wgpu being too old?) and stop; the rest is moot.
2. Add a packed-int8 GEMV variant in `gpu/quant.go` behind the runtime feature
   flag the spike already detects; keep the unpacked shader as fallback (the spike
   pattern is built for exactly this dual-path).
3. **Parity gate first, speed gate second** — same discipline as `gpu-assessment.md`
   §3: packed vs unpacked vs CPU oracle at the quant tolerance, *then* a
   `matrix_bench` microbench showing the packed path wins at M=1 and M=64.
4. Only if (3) clears, wire the batched M=K prefill GEMM onto the packed path and
   re-measure TTFT on a 256- and 1024-token prompt vs today's O(prompt-len)
   option-(a) prefill.

**Honest risk:** the binding may export the wgpu version but not surface the WGSL
language-feature enable, in which case this becomes a small upstream PR to
`cogentcore/webgpu` (the §4 single-binding risk `gpu-assessment.md` always
flagged). Even then it's the cheapest lever and de-risks the others.

---

## 2. Lever 2 — GPU speculative decoding. Biggest upside, measure before believing.

**The parked lever, and why now is when it unparks.** `roadmap.md` parks
MTP/spec-decode with a precise condition: *"Stays parked; revisit with a
bandwidth-bound (GPU) backend."* `gpu-assessment.md` closes with the same: the
unpark condition "is not yet met; revisit after the full-token build lands a real
E2E number." **That build landed** (residency decode wired into `decoder.Generate`,
`gpu-assessment.md` §0.0). So the stated precondition is satisfied — but a second
finding from the same campaign complicates it (below).

**Why it should win.** Plain decode runs K sequential M=1 target passes, each
paying the full per-token cost: 9.7 ms GPU + ~1 ms cgo encode of ~420 dispatches
(`gpu-assessment.md` §0.0). Spec-decode replaces those with K cheap *draft*
decodes + **one** batched M=K target verify. A verify pass streams the resident
weights **once** (the 4.3 ms gemv roofline cost amortizes over K tokens) **and**
re-encodes the dispatch graph **once** (the ~1 ms cgo encode amortizes over K).
So spec-decode amortizes *both* the bandwidth floor and the encode-glue wall — the
two things that cap single-stream decode — over K. That is structurally why it
beats the wall that fusion couldn't.

**Why to measure before believing it.** `gpu-assessment.md` §0.5's later finding
is that decode is **glue-serialization-bound, not purely bandwidth-bound**. Spec
decode's textbook win is framed for the bandwidth-bound regime. The amortization
argument above says it *also* helps the glue-bound regime (encode is amortized
too), but the magnitude depends on how much of the M=K verify's glue stays K-wide
vs fixed. The field caveat in `roadmap.md` is real: "even llama.cpp sees *no* MoE
single-stream win on consumer hardware." Net: predict **1.4–2.0× on dense models
at decent acceptance**, but this is a *prediction to measure on the 2070S*, not a
claimed number. The cost discipline: gate on a real E2E tok/s, kill if <1.3×.

**What has to be built (the two missing resident capabilities):**

1. **`ForwardN(embeddings [][]float32, startPos int) ([][]float32, error)`** on
   `ResidentForward` / `gpu/decoderunner.go` — a batched M=K forward that appends K
   KV positions and returns K logit rows. This is the GPU analogue of
   `Model.forwardN` (`decoder/forwardn.go:386`). It is genuinely new shader work:
   the resident attention (`gpu/attention.go`, warp-per-head) is written for one
   query row; M=K needs K query rows against the resident K/V (a small GEMM-shaped
   attention, K≪context so still cheap). The MLP/gemv kernels already handle M>1
   conceptually (prefill uses them) — the new surface is mainly attention + the
   KV-append-K plumbing.
2. **`TruncateTo(pos int)`** on `ResidentForward` — drop the rejected draft KV
   positions after a partial accept. On the resident cache this is just moving the
   write cursor back (positions get overwritten next round), so it is cheap; the
   CPU side already has the exact-rollback contract (`decoder/kvcache.go:298`
   "rejected draft positions were appended but aren't real"). Mirror that.

**Wiring:** in `GenerateSpeculative`, when `target` is on the webgpu backend and
`DecodeRunnerEligible()`, route the verify through `ForwardN` and the rollback
through `TruncateTo` instead of the CPU `forwardN`/`KVCache`. The draft model is
tiny — keep it CPU-resident at first (simplest, and draft cost is a small fraction
of the win); a resident draft is a later refinement, not v1.

**Parity is nearly free** — the killer feature of this lever. `GenerateSpeculative`
is already gated exact-vs-plain-greedy by `TestSpeculativeGreedyParity`. The new
gate is: *GPU spec-decode output == GPU plain `Generate` output, token-for-token*,
reusing that harness against the resident path. No new HF-golden burden (GPU parity
runs vs CPU, per `gpu-assessment.md` §4 policy).

**Effort:** ~2–4 weeks, dominated by the M=K resident attention shader and its
numerics gate (the GQA-head / mask drift tail `gpu-assessment.md` §3 warns about
lives here). Bounded because correctness is defined by an existing exact test.

**Stretch (don't scope into v1): MTP heads.** Models that ship multi-token-predict
heads (Qwen 3.6, Gemma 4 — `roadmap.md` line 44) make the *draft* free (no separate
draft model). Once `ForwardN` + `TruncateTo` exist, MTP is "draft = the target's own
MTP head" — the same verify/rollback path. Note it in the design so `ForwardN`
isn't built in a way that precludes it; build it after the draft-model version
measures a real win.

---

## 3. Lever 3 — wasm/browser inference. Marketing per unit effort.

**State:** `demo/gemma-web/` already exists (`main.go`, `index.html`, README) as an
HTTP demo. `gpu-assessment.md` §3 Stage 4 scoped the client-side version at 2–3
weeks and named the honest framing: **"a 0.5B in a tab, pure Go," not the 7B story
in a browser.** `cogentcore/webgpu` targets browser WebGPU under wasm without cgo —
the whole point of having chosen WebGPU over native APIs.

**Why it's worth it despite being demo-not-product:** `gpu-assessment.md` calls it
"the killer adjacent payoff … a demo nobody else in the Go world can ship." It is
the cheapest way to convert the entire GPU investment into something a reader can
*click*, and it has no decode-wall risk because nobody benchmarks a tab demo
against Ollama.

**Honest bounds (write them into the demo, don't hide them):**

- Browser WebGPU int8-packed-path support is uneven; assume the wasm build runs the
  **f32 or unpacked-int8** shaders (couples to Lever 1's fallback path, not its fast
  path).
- wasm memory caps + model-download size bound it to ~0.5B. This is a tab demo, not
  a product surface — say so in the UI.
- The resident decode path is the natural thing to compile; eligibility gate
  already excludes the hard archs.

**Effort:** ~2–3 weeks, low *numerical* risk (same shaders, new target), the tail
is wasm build plumbing + model fetch/caching in the page.

---

## 4. Ranked recommendation

| # | Lever | Effort | Payoff | Certainty | Gate to stop |
|---|---|---|---|---|---|
| 1 | `dot4I8Packed` unblock → batched prefill | 3–6 d | TTFT (big on long prompts); ≤1.05× decode | **High** (upstream merged; spike already written) | spike logs NOT *and* the gap is a real upstream port |
| 2 | GPU speculative decode (`ForwardN`+`TruncateTo`) | 2–4 wk | 1.4–2.0× decode (predicted, dense) | **Medium** (glue-bound regime; measure) | real E2E < 1.3× |
| 3 | wasm/browser tab demo | 2–3 wk | marketing / reachability | **High** (no perf risk) | — (justified by visibility alone) |

**Sequence:** 1 → 2 → 3. Lever 1 first because it is days not weeks, its result is
a logged boolean, and it de-risks 2 and 3 (both lean on the int8 kernel path).
Lever 2 second because it is the only remaining *decode* win and its machinery is
80% built — but enter it with the kill-gate armed. Lever 3 last (or in parallel,
by a different sitting) because it is independent and justified by reach, not speed.

---

## 5. The one-week spike that resolves the most uncertainty

If only a few days are available, run **Lever 1 step 1 + a Lever 2 paper-cut**:

1. Bump `cogentcore/webgpu`, run `TestSpike_capabilities` on the 2070S — resolves
   the single biggest stale assumption in the GPU docs (DP4A availability) with one
   command.
2. Instrument a *CPU-side* spec-decode acceptance-rate measurement on the target
   1.5B + a 0.5B draft over a real prompt set (the machinery and telemetry —
   `Drafted`/`Accepted`/`AcceptanceRate` in `decoder/speculative.go` — already
   exist; this needs no GPU). Acceptance rate is the dominant term in the spec-decode
   speedup and is **architecture/quant-determined, measurable today on CPU.** If
   acceptance is poor on this model pair, Lever 2's ceiling is low *before* any GPU
   work — a cheap way to right-size the 2–4 week bet.

Both produce a number that the current docs only estimate. Neither needs new
kernels.

---

## 6. Explicitly NOT worth doing (so it isn't re-litigated)

- **More single-stream decode fusion / a megakernel.** Walled: WGSL has no
  persistent-thread single-dispatch whole-layer kernel; GPU-alone is ~103 tok/s and
  the irreducible cgo encode floors the token near ~93 (`gpu-assessment.md` §0.0).
  The 4.3 ms gemv is *at roofline*. Nothing to grind.
- **K-quant (Q4_K/Q5_K/Q6_K) in-shader.** `gpu-assessment.md` §3 Stage 1b: the CPU
  loader already dequant-requants GGUF into resident int8/int4, so this is a
  bandwidth optimization, not a capability. Defer.
- **MoE GPU residency / indirect routing.** Real and interesting
  (`gpu-assessment.md` §0.5 "design requirement"), but the eligibility gate
  excludes MoE for good reason and the dense path is where the users are. Keep
  `ForwardN` *retrofittable* to `dispatchWorkgroupsIndirect` (don't bake "dispatch
  args known on CPU"), but don't build MoE residency now.
- **A native CUDA/Metal kernel path.** Off-strategy, permanently. `gpu-assessment.md`
  §4 + `roadmap.md` line 46: "Don't fight on kernels." The portability moat (one
  WGSL set, one static binary, no driver) is the position; a vendor-kernel arms race
  is llama.cpp's headcount.
- **Apple Neural Accelerators (M5).** MLX-only; WebGPU-on-Metal can't reach the
  matrix units. Apple users get the bandwidth win, not the MMU win. Acknowledged
  trade.

---

## Sources (in-repo + upstream)

- `gpu/decoderunner.go`, `gpu/quant.go`, `gpu/attention.go`, `gpu/spike_test.go` —
  live runner / kernel / probe state.
- `decoder/residency.go`, `decoder/speculative.go`, `decoder/forwardn.go`,
  `decoder/kvcache.go` — the resident seam, spec-decode machinery, batched forward,
  KV rollback contract.
- `docs/gpu-assessment.md` (§0.0 resolved decode, §0.5 glue-bound finding, §3 staged
  plan + Stage 4 wasm, §4 tradeoffs), `docs/roadmap.md` (MTP/spec-decode parked,
  "don't fight on kernels", DP4A gates prefill), `docs/task-gpu-batched-prefill.md`.
- Upstream DP4A: gfx-rs/wgpu PR #7494 (`dot4I8Packed`/`dot4U8Packed`), PR #7595
  (`NATIVE_PACKED_INTEGER_DOT_PRODUCT` → `VK_KHR_shader_integer_dot_product`);
  Chrome for Developers "WebAssembly and WebGPU enhancements for faster Web AI,
  part 2" (1.7–2.9× on 8-bit ML).
