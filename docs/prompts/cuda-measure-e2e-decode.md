# goinfer spike (cont.): measure the real end-to-end cgo-free CUDA decode

## Why this task (read first)

The CUDA spike (`task-cuda-cgofree-spike.md`) reached **GO**, then a reality-check
tempered it: the 244 tok/s was a **GEMV-only ideal-streaming projection** (83% of the
~293 tok/s bandwidth ceiling); Ollama-CUDA does the **same 1.5B int8 end-to-end at
147** (50% of ceiling). A hand megakernel is **not** more optimized than llama.cpp,
so its realistic landing is **near Ollama (~147 ≈ 1.3× WebGPU), not 244**. But that
~1.3× is *still a projection* — the real end-to-end goinfer-CUDA decode has **not been
measured**. This task measures it, so the "worth the CUDA-kernel maintenance burden?"
decision rests on a number, not an inference.

**Do NOT build the production backend, real-weight `BuildResident` extraction,
argmax-on-real-checkpoint parity, or any family beyond dense 1.5B.** Those are gated
on *this* measurement, not the reverse.

## What to build (a measurement harness, not a backend)

A minimal **real N-layer decode loop** on the cgo-free CUDA path — the GEMV you
already have PLUS the per-token work the 244 projection omitted, so the number is
end-to-end, not a streaming ceiling:

- Reuse the validated W8A8/W4A8 GEMV (cosine-1.0 gated already).
- Add the elementwise/glue kernels: RMSNorm, RoPE (q/k), residual, SwiGLU — and a
  **minimal attention** (QKᵀ · softmax · V) over a small real KV, + activation
  requant + sampling + the syncs. These are exactly the ~33%-of-ceiling costs the
  projection dropped. Correctness: each kernel cosine-validated vs the CPU reference;
  greedy output token-identical to the CPU decode over a short run (synthetic weights
  are fine for the *perf* number — bandwidth is value-independent — but the loop must
  do all the real per-token work).
- Run a real **1.5B-shaped, N-layer** decode (dense Qwen2/Llama residency shape), not
  a single GEMV chain.

## The shippable-config requirements (so the number is decision-grade)

The measurement must reflect what would actually ship, or it's another optimistic
projection:

1. **Driver JIT, not runtime NVRTC.** Compile `.cu`→PTX at build (nvcc/NVRTC on the
   dev box), `go:embed` the PTX, JIT via the **driver** (`cuModuleLoadDataEx`, in
   libcuda). **Verify the running binary has NO `libnvrtc.so` dependency** (`ldd` /
   the dlopen list) — else "driver-only" is false. (The spike's "NVRTC JIT" is the
   thing to close here.)
2. **Include the thread-safety executor cost.** A correct multi-goroutine CUDA path
   needs the context pinned to an OS thread (CUDA context is per-OS-thread; Go
   migrates goroutines) — the `LockOSThread`-worker + channel design. Its **per-call
   channel hop must be in the measured number**, because the production backend can't
   skip it. If the spike measured a single-threaded direct path, add the executor and
   re-measure.
3. **`CGO_ENABLED=0`** end to end, one static binary (libdl/libpthread dynamic-link is
   fine; libcuda dlopen'd at runtime).

## Measure + report (CUDA events, warm, same discipline as the spike)

- End-to-end decode **tok/s**, CUDA-event-timed, warm, best-of-N, on the 2070 SUPER,
  same **1.5B int8**, vs the two anchors on the same model: **WebGPU 111.6** and
  **Ollama-CUDA 147**. Report as tok/s **and** % of the ~293 tok/s bandwidth ceiling
  (so it slots into the spike's table).
- Break the token down (GEMV vs glue vs attention vs sync) so the gap to the 244
  ceiling is *attributed*, not hand-waved.
- Confirm requirements 1–3 held (no libnvrtc; executor in the loop; cgo-free).

## Go / no-go (this is the decision, finally on a real number)

- **Clearly above Ollama** (≳1.5× WebGPU, i.e. the megakernel genuinely beats native
  CUDA end-to-end) ⇒ surprising and strong; the track is worth building. Document why
  it beat llama.cpp (fusion? packing?) — that's a real finding.
- **~Ollama parity** (~1.3× WebGPU, ~147) ⇒ **PAUSE the track.** Confirms the
  recalibration: the prize is *matching* native CUDA, cgo-free. That's a genuine
  capability, but whether it's worth the permanent CUDA-kernel maintenance burden is a
  **business/judgment call for the maintainer, not an engineering step** — park it as a
  measured, documented option; don't build the production backend on a tie.
- **≤ WebGPU** (glue + channel-hop tax ate the GEMV advantage once real per-token work
  is included) ⇒ **NO-GO**, close the CUDA lane; WebGPU + buffer-coalescing stays the
  NVIDIA story, now measured-closed rather than assumed.

Timebox: a focused session — you already have the GEMV, the executor pattern, and the
CPU reference; this is the glue kernels + the decode loop + the measurement.

## Deliverable

Append the measured end-to-end number, the token breakdown, the config-1–3
confirmation, and the recalibrated GO/PAUSE/NO-GO to `task-cuda-cgofree-spike.md`
(same provenance discipline: no perf claim without a CUDA-event-timed run, versioned
by driver + card + commit). This measurement supersedes both the 244 projection and
the ~1.3× inference — it's the number the track's future rests on.
