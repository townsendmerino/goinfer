# Gemma 3 on the Metal backend — debug report (UNRESOLVED)

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


A port of Gemma 3 to goinfer's cgo-free Metal resident decoder produces wrong output on the
real model. Every kernel is unit-tested correct, a synthetic Gemma 3 passes at the real model's
exact per-layer dimensions, and ~9 hypotheses have been measured and killed. This is everything
known, written for someone with no prior context.

**Status:** kernels are committed but DORMANT — Metal does not declare the Gemma features, so
`gemma3` correctly declines to the CPU path. Nothing ships broken. The parity gate skips until
the features are declared.

---

## 1. Background: what the code does

goinfer runs LLMs in pure Go. The Metal backend (`metal/`, darwin, cgo-free via purego-objc)
keeps a model resident on the GPU and runs one token per command buffer.

**Quantization asymmetry — important for reading every number below.** `BuildResident` requires
an **int8**-loaded model and re-quantizes weights to its own **W4A8** (int4, group=32, scale =
maxabs/7) for the GPU. The CPU reference runs **int8**. So Metal-vs-CPU is *always*
**int4-GPU vs int8-CPU** — never like-for-like. (CUDA's equivalent test loads int4 on *both*
sides, so its bars do not transfer here.)

**Gemma 3 needs five features beyond the dense Qwen/Llama block** (all derived from arch flags,
not hand-listed):

| feature | what it is | Metal implementation |
|---|---|---|
| `sandwich-norm` | Gemma norms each **sublayer OUTPUT** before the residual add: `y=proj(...)`, `y=rms(y)·(1+w_post)`, `x+=y` | new `rmsnorm_f32` kernel; **splits the fused `_resid` GEMV epilogue** into proj→scratch, norm, add |
| `gated-gelu` | gated MLP with GELU-tanh instead of SiLU | `act` selector in `swiglu_quant`; ordinals = `decoder.ActKind` iota (0=GELU-tanh, 1=SiLU) |
| `rms-add-one` | `(1+w)` RMS offset | `addOne` uniform threaded into `rmsnorm_quant`, `rmsnorm_f32`, and the **final** norm |
| `embed-scale` | ×√hidden on the embedding | **nothing** — `decoder.embedResident` applies it host-side |
| `per-layer-rope` | local base 10k vs global base 1M | per-layer inv-freq buffer; **rope kernel unchanged**, width stays model-level |
| *(+ per-layer window)* | Gemma mixes local/global layers 5:1 | per-layer window uniform (was model-level) |

Reference implementation: the same five features landed on the CUDA backend (commit `03e9816`)
and work there — Gemma 3 generates coherently on CUDA at 91 tok/s.

---

## 2. The symptom

Real model: **`gemma-3-4b-it` Q4_K_M GGUF** (34 layers, H=2560, nH=8, hd=256, nKV=4, I=10240,
V=262144, embed scale √2560=50.6, sliding window 1024 with a 5:1 local:global pattern).

Harness: load twice (one Metal-resident, one CPU), drive both in greedy lockstep (the CPU's
argmax feeds both sides), 24 steps.

| | argmax exact | min logit cosine |
|---|---|---|
| **dense control** (qwen2.5-coder-1.5b — the shipped, known-good Metal path, same harness) | 15/24 | **0.962** |
| **gemma-3-4b-it** | **14/24** | **0.699** (0.764 at pos 0) |

**The central tension:** Gemma's **argmax agreement is essentially the control's** (14 vs 15 of
24) — a badly broken 34-layer model should not pick the CPU's token 58% of the time out of a
262,144-token vocabulary. But its cosine is far worse. Both metrics are reported because
neither alone is trustworthy here (a lesson learned the hard way on both backends).

---

## 3. The localization (the strongest clue)

`KVCache.Keys(l)`/`Vals(l)` are exported, and Metal stores per-layer K/V, so the **hidden path**
can be compared layer by layer instead of only the final logits. At **pos 0**:

```
layer  0: K cos 0.999515   V cos 0.998092     <- layer 0's INPUTS are correct
layer  1: K cos 0.403001   V cos -0.046768    <- catastrophic
layer  2: K cos 0.439291   V cos  0.427412
layer  3: K cos 0.686639   V cos  0.337995
layer  4: K cos 0.849539   V cos  0.465292
layer  5: K cos 0.945979   V cos  0.697585    <- partially "recovers"?!
layer 12: K cos 0.536798   V cos  0.059271
```

Layer L's K/V are computed from `x` *after* layer L−1. So:

- **Layer 0's KV is correct** ⇒ the embedding, `rmsnorm_quant(preNorm)` (with addOne=1!), and
  the fused QKV projection are all **right**.
- **Layer 1's KV is catastrophic** ⇒ **`x` after layer 0 is wrong**.
- Therefore the bug is in **layer 0's post-attention path** — exactly the sublayer this port
  rewrote:
  `o-proj → rmsnorm_f32(postAttnNorm) → residual → rmsnorm_quant(preMLPNorm) → gate|up → GeGLU
  → down → rmsnorm_f32(postMLPNorm) → residual`

Two oddities worth a fresh eye:
- **V cos = −0.047 is orthogonal**, not merely degraded. That is a structural break, not drift.
- The subsequent trend is **non-monotonic** (layer 1 worst, layer 5 back to 0.95, layer 12 bad
  again). Speculation: each layer's RMSNorm renormalizes, so a direction error can partially
  cancel — but this pattern is not understood.

### Why pos 0 is powerful

At pos 0 the following are **provably eliminated** — they cannot affect the output:
- **RoPE** — θ = pos·invfreq = 0 ⇒ cos=1, sin=0 ⇒ the rotation is the identity. (So *per-layer
  rope* is also eliminated.)
- **QK-norm, attention scale, q, k** — with one key, `softmax(single element) = 1`, so the
  attention output is **exactly `v0`**, independent of q/k/scale.
- **Sliding window** — nKeys=1 < window=1024, so `winStart=0` regardless.

So the pos-0 divergence lives in: **norms, the MLP, and the projections.**

---

## 4. Hypotheses tested and KILLED (with evidence)

Each was measured, not reasoned about.

1. **OOM masquerading as a parity bug.** Metal's buffer constructors silently kept a nil id on
   allocation failure, so an OOM would surface as garbage numerics. Made allocation failures
   panic loudly → **no allocation failed, and the cosine was bit-identical (0.699119)**. Not OOM.
   *(The unchecked-alloc landmine was real and is now fixed.)*

2. **The individual kernels.** All unit-tested against CPU references:
   - `rmsnorm_f32` — both addOne modes ✓
   - `rmsnorm_quant` — both addOne modes ✓ (and layer 0's correct KV independently proves it
     works at H=2560 in situ)
   - `swiglu_quant` act selector — **both** GELU-tanh and SiLU vs CPU refs ✓ (a swapped ordinal
     would silently run Gemma as SwiGLU)

3. **The uniforms.** Dumped from the live model: `uAddOne=1`, `uAct=0` (GELU), `sandwich=true`,
   embedScale=50.5964=√2560, window=1024, local/global pattern correct (layers 0,1,6 local;
   layer 5 global), `half=128`. All correct.

4. **The dispatch order.** Checked line-for-line against the CPU forward (`decoder/model.go`
   440–461) and against `normalize`/`rmsNorm`. Matches exactly, including that the sandwich
   post-norms use the same `RMSAddOne` as everything else and take a nil bias.

5. **The pipelined executor.** `ForwardEmbPipe` pre-encodes token t+1 while t runs. Compared
   against the direct `ForwardEmb`: **bit-identical (both 0.764, max|Δ|=0)**. Not the driver.

6. **The test harness.** The real gate loads the model **twice** (Backend:"metal" + a CPU load);
   the synthetic loads once. Made the synthetic mirror the real harness exactly (two loads) →
   **still passes**. Not the harness.

7. **Shape.** A synthetic Gemma 3 was built at progressively closer shapes, ending at
   **gemma-3-4b's exact per-layer dims** (H=2560, nH=8, hd=256 so nH·hd=2048≠H, kvDim=1024,
   I=10240) × 4 layers → **cosine 0.997, PASSES**. Not the shape — including the `nH·hd ≠ H`
   class that caused a previous real bug (Qwen3's activation-staging cap).

8. **Unbound uniforms.** Adding `addOne`/`act` changed two kernel signatures; Metal leaves
   unbound buffer slots holding whatever the previous dispatch left, so a stale `act` silently
   runs GeGLU. This **was** a real bug — but only in two *test* dispatch sites (caught by the
   pre-tag gate: a legacy test went 0.9627→0.2473, now restored). **All 10 kernels in the
   production Gemma path were audited: every uniform is bound.**

9. **int4 requant damage from Gemma's weights** (the "Gemma has outliers int4 destroys"
   hypothesis). Measured the int8→int4 requant cosine per projection, layer 0:

   | | q | k | v | o | gate | up | down |
   |---|---|---|---|---|---|---|---|
   | Gemma-3-4B | .9947 | .9950 | .9951 | .9949 | .9948 | .9950 | .9948 |
   | Qwen control | .9934 | .9945 | .9946 | .9956 | .9947 | .9953 | .9952 |

   **Identical.** Gemma's weights quantize to int4 exactly as well as the control's. Refuted —
   and this *strengthens* the case for a real bug: same quant damage, 10× the divergence.

---

## 5. The competing hypothesis (weakened, not dead)

**"It's int4-vs-int8 error accumulating over a deep model, and the 0.95 bar is simply wrong."**

Supporting: the synthetic at gemma's per-layer dims decays ~**0.6%/layer** (0.9977 → 0.9810
over 4 layers); extrapolated over **34** layers ≈ **0.79** — close to the observed 0.764. And
Gemma's argmax (14/24) matches the control (15/24), which is what a *working* model looks like.

Against — and this looks decisive:
- The real model's decay is **not gradual**: layer 0 is 0.998 and layer 1 is **−0.047**. That is
  a step change at a single sublayer, not accumulation.
- A **narrow but 34-layer-deep** synthetic reaches only 0.965, not 0.76.
- The wide × 34-layer synthetic could not be run (at those dims it *is* a 4B model — the f32
  safetensors alone would be ~16 GB; the process was killed).
- Hypothesis 9 shows the per-weight int4 damage is identical to the control's.

---

## 6. What is different between the PASSING synthetic and the FAILING real model

This is the crux. The synthetic reproduces the architecture, the five features, the exact
per-layer dims, and the harness — and passes at 0.997. Remaining differences:

1. **Real weights vs uniform-random weights** (int4 damage is measured identical — hypothesis 9)
2. **GGUF vs safetensors** — the real model is a Q4_K_M GGUF dequantized to int8; the synthetic
   is F32 safetensors quantized to int8. *Note: a weight-mapping error should affect CPU and GPU
   equally and therefore cannot produce a GPU-vs-CPU divergence — but this is the largest
   untested difference.*
3. **Depth** 34 vs 4 layers (should not affect layer 0's output)
4. **Vocab** 262144 vs 256 (cannot affect layer 0's KV)
5. **Window/pattern** 1024/6 vs 16/2 (inert at pos 0)

---

## 7. The open question

> Layer 0's inputs are correct and every kernel in layer 0's output path is individually
> correct, the uniforms are right, the order matches the CPU line-for-line, and a synthetic
> model with the same architecture, features, dims and harness passes at 0.997 — yet on the real
> checkpoint `x` after layer 0 comes out **orthogonal** to the CPU's.
>
> What differs between a random-weight synthetic and a real GGUF checkpoint that could break
> layer 0's post-attention path *on the GPU only*, when both sides read the same loaded weights?

## 8. Untried next steps

- **Coherent generation** — the test CUDA used to validate its Gemma ("The capital of France is
  Paris."). Blocked here: the un-gated GGUF mirror lacks the tokenizer merges array (Gemma 3 is
  SentencePiece), and the safetensors are license-gated. **This is the highest-value next step**
  — if it generates coherently, the bar is wrong; if it emits garbage, the bug is real.
- **Read `x` directly.** Add a test hook that encodes only N layers and reads the residual
  buffer, then bisect *within* layer 0's sublayers (after o-proj, after the sandwich norm, after
  the residual, after the MLP) instead of inferring from layer 1's KV.
- **Ask CUDA for its Gemma cosine at int4-GPU vs int8-CPU** (not its native int4-vs-int4). If
  CUDA also shows ~0.76 under that asymmetry, there is no Metal bug and the bar is wrong.
- **Run the real Gemma through the WebGPU backend** for a third opinion on the same weights.

## 9. Reproduction

```bash
# the model (un-gated mirror, 2.3 GB)
curl -sL -o ~/models/gemma-3-4b-it-Q4_K_M.gguf \
  "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf"

# declare the 5 Gemma features for metal in decoder/features.go, then:
go test -run 'TestGemma3ResidentParity' ./metal/ -v   # subject   → cosine 0.699
go test -run 'TestDenseResidentParity'  ./metal/ -v   # control   → cosine 0.962
```

Kernels: `metal/kernels.go` (`rmsnorm_f32`, `rmsnorm_quant`, `swiglu_quant`/`glu_act`).
Assembly: `metal/model.go` `encodeTrunkInto` (the `r.sandwich` branches).
Design + status: `docs/task-metal-gemma.md`.
