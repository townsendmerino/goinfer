# Positioning after the cgo-free CUDA result: sell the intersection, not a speed pivot

> **Audience:** internal strategy — written after the same-box, pinned-Ollama
> anchor (`task-cuda-cgofree-spike.md`, `0de55d3`) showed goinfer's **cgo-free**
> CUDA decode at **parity-to-ahead of fresh Ollama-CUDA** (190 int4 vs ~149,
> ~1.6× WebGPU), driver-only, `CGO_ENABLED=0`. The perf question is closed. This
> doc records the *strategic* reading so the excitement doesn't set direction by
> default. Two decisions: **how much to build**, and **how to promote**.

## What the result actually is (and isn't)

**Is:** the one structural weakness that defined goinfer's GPU posture — "we don't
compete on GPU speed; portability is the trade" — is **dissolved**. goinfer can now
credibly claim the **intersection no one else has**: native-CUDA-class decode **and**
zero cgo **and** no toolkit / driver-only **and** a single static binary. llama.cpp/
Ollama are fast but need the toolkit + native libs; the Go bindings (yzma, gollama)
still ship llama.cpp's `.so`. The *intersection* is the novel, defensible claim.

**Isn't (yet):** a shipped capability. The anchor is a **bare-forward GEMV loop,
synthetic weights, fixed pos=128, dense 1.5B** — a micro-benchmark, not a correct
end-to-end decode of a real checkpoint. And the win is **dense-residency-only**.

## Decision 1 — how much to build

**Build (B), scoped tight — YES.** Ship the *dense* CUDA-residency path with
real-weight extraction + **argmax parity on a real checkpoint**. This is what turns
the micro-benchmark into a capability you can honestly claim, and it's bounded (one
card, dense residency, opt-in). It is the honest completion of the spike.

**The full CUDA-backend treadmill — NO (gate on demand).** The load-bearing caveat:
the win is on the **residency path = dense Qwen/Llama only**. goinfer's *headline*
families — DeepSeek/GLM/Kimi (MLA), Granite/Nemotron (Mamba-2) — are **not**
residency-eligible on CUDA any more than on WebGPU. Accelerating them means
hand-authoring MoE/MLA/Mamba CUDA kernels, each separately. So "build it out fully":

- **fragments the payoff** — the frontier families everyone's excited about don't get
  the speedup until many more kernels exist;
- **is a permanent maintenance treadmill** against llama.cpp's full-time team (parity
  today ≠ parity as CUDA/models/drivers evolve);
- **drifts goinfer's identity** from "portable embeddable runtime" toward "another
  fast GPU runtime" — a crowded, well-funded fight goinfer loses at full-portfolio /
  server scale (no continuous batching, MoE unaccelerated, single maintainer).

Each additional CUDA kernel is its own decision, gated on a real adopter asking — not
on the excitement of the anchor number.

## Decision 2 — how to promote

**More: yes.** goinfer is under-marketed vs its substance. **0.8.0 + the
coverage-axis story + the Mellum2 GPU-resident showcase** are promotable *now*.

**Differently: the CUDA result is a proof point that makes the existing pitch
airtight — not a new headline that replaces it.** The trap is leading with "goinfer
is a fast CUDA runtime now": that moves the fight onto the *speed* axis, invites a
head-to-head with Ollama/vLLM on their turf (full portfolio, server scale), and
goinfer loses that as a whole product. Instead, **sell the intersection**:

> *The embeddable, cgo-free, single-binary Go runtime that runs every modern
> attention family — softmax, gated-linear, state-space, latent-KV — **and doesn't
> even give up native-CUDA-class GPU speed to do it: driver-only, no toolkit, no
> cgo.***

The CUDA win **deletes the one apology** from the portability pitch ("but it's slow
on GPU") and turns it into "and it's fast too." **"The portable one that's *also*
fast" beats "the fast one that's *also* portable"** — because on raw speed at full
scale goinfer is contestable, but on the *combination* it is alone. Promote the
intersection nobody else has; never either axis by itself.

## Timing discipline (same rigor as the parity work)

- Promote **0.8.0 + coverage axes now** — shipped, real, under-sold.
- Add the **CUDA claim only after (B) ships with argmax parity.** Promoting a spike
  benchmark as a shipped feature is the projection-optimism the spike itself keeps
  catching — don't front-run your own standard. "Native-CUDA-class, cgo-free" becomes
  true-and-promotable the day (B) produces correct tokens at speed, not before.
- When it's true, the honest phrasing is **"dense-model GPU decode at native-CUDA-
  class speed, cgo-free"** — keep the *dense* qualifier so the claim survives contact
  with a DeepSeek/GLM user (whose model isn't on that path yet).

## One-line synthesis

**Build (B) to make the claim real; refuse the all-families CUDA treadmill; promote
the intersection — portability that no longer sacrifices GPU speed — not a speed
pivot goinfer would win only on a dense micro-benchmark.** The result is a gift to
the *existing* moat, not a reason to trade it for a new one.

## What would change this

Flip to "build the CUDA backend out broadly" only if a concrete adopter needs
*fast NVIDIA decode of a specific MoE/hybrid family in a cgo-free Go binary* — i.e.
the intersection demand shows up for a non-dense family. Absent that, dense (B) +
the intersection pitch is the whole play.
