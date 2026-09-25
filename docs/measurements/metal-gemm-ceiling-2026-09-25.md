# Metal prefill GEMM ceiling — what the M1 Pro sustains on the 1.5B's shapes (S1a, 2026-09-25)

**Question.** S0 (`metal-prefill-decomp-2026-09-25.md`) put the GEMM at 92.7% of Metal prefill GPU time at K=512,
with goinfer's `gemm_w4f16_store` running the MLP projections at ~0.75 TFLOPS in production, and priced parity at
K=512 at a ≈2.85× GEMM speedup. Whether that is reachable depends on what this GPU can sustain at these exact shapes.
This measures it with Apple's own tuned GEMM as the reference.

**Answer.** `MPSMatrixMultiplication` (f16) sustains **3.24–3.43 TFLOPS on all four projection shapes, at both M** —
2.4–4.5× goinfer's in-sequence rate, **4.5× on gate/up**. At that rate the GEMM category would be 3.83× faster at K=512,
more than the 2.85× parity needs. **And MPS shows no post-idle burst**, so the ~2× burst/sustained split S0 found is a
property of goinfer's kernel, not of the GPU's clock or power state.

## Provenance

- **Machine:** Apple M1 Pro, 16 GB, macOS 26.6.2. Idle at start (load1 1.95), 80% memory free.
- **Instrument:** [`metal-gemm-ceiling-2026-09-25/mps_gemm_ceiling.swift`](metal-gemm-ceiling-2026-09-25/mps_gemm_ceiling.swift),
  `xcrun swiftc -O` (Apple Swift 6.4). Run 2026-09-25 14:15:15–14:16:16 local (21:15–21:16 UTC), 5 reps. Raw:
  [`run.log`](metal-gemm-ceiling-2026-09-25/run.log).
- **What it computes:** C = A · Wᵀ, f16 in and out, A = [M × K] activations, W = [N × K] weights — the layout
  goinfer's prefill uses. Shapes are the 1.5B's (H=1536, I=8960): qkv N=2048/K=1536, o 1536/1536, gate/up 17920/1536,
  down 1536/8960; M = 512 and 3904 (the padded 3900).
- **How:** each timed command buffer encodes the shape 28 times (the layer count), cycling 4 distinct weight
  matrices, each larger than the system-level cache, so weights stream from DRAM as in production. After ≥ 3 s of
  warm-up, the timed loop runs back to back with no idle gaps: **sustained load**, the condition S0's run 4 showed any
  timing on this GPU needs. GPU time = `gpuEndTime − gpuStartTime`.

## Result (medians of 5)

| M | shape | MPS f16 GPU ms (28×) | spread | **MPS TFLOPS** | goinfer in sequence (S0) | headroom |
|---:|---|---:|---:|---:|---|---:|
| 512 | qkv | 27.8 | 2.8% | **3.25** | 66.75 ms, 1.35 TFLOPS | 2.4× |
| 512 | o | 20.8 | 4.0% | **3.25** | 49.20 ms, 1.38 | 2.4× |
| 512 | gate/up | 232.5 | 4.9% | **3.39** | 1050.5 ms, 0.75 | **4.5×** |
| 512 | down | 121.8 | 4.1% | **3.24** | 378.6 ms, 1.04 | 3.1× |
| 3904 | qkv | 200.6 | 6.2% | **3.43** | 578.2 ms, 1.19 | 2.9× |
| 3904 | o | 151.8 | 3.4% | **3.40** | 394.2 ms, 1.31 | 2.6× |
| 3904 | gate/up | 1768.4 | 1.5% | **3.40** | 8032.5 ms, 0.75 | **4.5×** |
| 3904 | down | 887.6 | 0.6% | **3.39** | 4037.3 ms, 0.75 | **4.5×** |

goinfer's column: K=512 qkv/o from S0 run 1 (whose medians were all in the sustained mode), gate/up and down from the
run-3 leave-one-out; K=3900 all from run 3's leave-one-out, i.e. each category's cost inside production.

**Burst check** (gate/up, M=512): 242.1 ms after 2 s of idle (spread 1.5%), 232.5 ms immediately after other work
(spread 2.6%). MPS is, if anything, ~4% *slower* right after idle, where goinfer's kernel is ~1.9× faster
(`metal-prefill-decomp-2026-09-25.md`, run 4).

## What it implies

TTFT fraction after a GEMM speedup s = (1 − X) + X/s, X = the GEMM's share of wall (S0: 0.960 at K=512, 0.761 at
K=3900). With goinfer's GEMM category at MPS's sustained rate:

| K | goinfer GEMM (ms) | at MPS rate (ms) | s | TTFT vs today | vs Ollama (today 0.377 / 0.239) |
|---:|---:|---:|---:|---:|---:|
| 512 | 1544.3 | 402.9 | 3.83× | 0.29 | **~1.30×** (ahead) |
| 3900 | 13042.2 | 3008.4 | 4.34× | 0.415 | ~0.58× (still behind) |

- **Parity at K=512 is inside what the hardware sustains.** The ≈2.85× S0 derived is below the 3.83× an MPS-rate GEMM
  would deliver. This is an **optimistic bound** in one direction and a conservative one in another: MPS's f16
  weights need no dequantization, but they read 4× the bytes int4 does. It does not say what an int4 kernel reaches;
  that is S1b below.
- **At K=3900 the GEMM cannot close the gap alone**, as S0 found: even at MPS's rate TTFT reaches ~0.58× Ollama's,
  with attention (25% today) left.
- **The burst is goinfer's kernel, not the GPU.** A global clock or power drop under sustained load would slow MPS
  too, and it does not. So something in how `gemm_w4f16_store` uses the GPU makes it fast only right after idle. The
  current kernel's ~0.75 TFLOPS is still its production rate; the mechanism is not identified, and a replacement
  kernel should be measured sustained and after idle, so it is shown not to inherit it.

## S1b: what an int4 kernel sustains here — llama.cpp on the same file (same day)

`llama-bench` (llama.cpp `c1d0e7a00`, build 10621, Homebrew, ggml 0.22.0; Metal, `-ngl 99`, flash attention at its
default `auto`, f16 KV, `n_ubatch` 512) on **the same `~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`**, `-p 512,3904
-n 0 -r 5`. It repeats back to back after its own warm-up, so it is sustained. 14:21:00–14:21:27 local. Raw:
[`run-llamabench.log`](metal-gemm-ceiling-2026-09-25/run-llamabench.log).

| prompt | llama.cpp prefill (5 samples) | GEMM FLOPs (4 projections × 28) | implied GEMM rate | goinfer `PrefillLast` |
|---:|---|---:|---|---:|
| 512 | **453.9 ms** (sd 1.7; 452.3–456.3) | 1.342 T | **≥ 2.96 TFLOPS** | 1608 ms |
| 3904 | **4009.2 ms** (sd 4.4; 4002.1–4013.6) | 10.230 T | **≥ 2.55 TFLOPS** | 17,618 ms |

The implied rate is a **lower bound** on llama.cpp's GEMM throughput: the time also covers its attention, norms and
everything else. Q4_K_M is not pure 4-bit (some tensors are Q6_K), and the kernel dequantizes in-kernel as goinfer's
does, so this is the int4-class reference S1a's f16 number is not.

- **In-kernel dequantization costs little of the ceiling.** An int4-class kernel sustains ≥ 2.96 TFLOPS here, within
  ~13% of MPS's f16 3.39 and ~4× goinfer's gate/up. The gap is the kernel's structure, not the weight format.
- **Parity at K=512 is inside what a known kernel does on this machine.** goinfer's GEMM category at llama.cpp's
  lower-bound rate is ~453 ms against 1544 (s ≈ 3.4×), giving a TTFT of ~0.32 of today's — ~1.17× Ollama's rate at
  K=512, against the ≈2.85× parity needs.
- **At K=3904, llama.cpp's whole prefill (4.0 s) is shorter than goinfer's attention alone (4.4 s, S0).** The deep
  gap is GEMM and attention in similar measure. llama.cpp also prefills in 512-token micro-batches (`n_ubatch`
  512), so its GEMMs run at M=512 however long the prompt is.

## Scope

One machine, one model's shapes (1.5B), f16 only, one vendor kernel. MPS is a ceiling reference, not a peer: Ollama
runs llama.cpp's Metal kernels on Q4_K weights, not MPS. The 7B's shapes (H=3584, I=18944) are not measured.
