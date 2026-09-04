# Task — FP4 microscaling formats: native MXFP4 compute, and NVFP4

**Status:** proposed, gates before code. Filed 2026-09-02.
**Venue:** `mac` and `linux` for the reachable half; a Blackwell-class device for the rest.
**Relates to:** `docs/task-mxfp4-gptoss.md` (the existing MXFP4 work — this doc extends it, does not
replace it).

---

## The distinction this whole doc turns on

**A quantization format can be a storage format or a compute format, and we currently treat MXFP4
only as the first.**

`decoder/mxfp4.go` reads MXFP4 blocks — 32 elements, one e8m0 scale byte plus 16 nibble-packed
e2m1 bytes, 17 bytes per block — and dequantizes them through a lookup table
(`mxfp4DequantSplitInto`) into float32. The layout is transcribed from the reference `gguf` Python
library and verified bit-for-bit against a real gpt-oss checkpoint (`mxfp4_test.go`). That is solid
work and it is a *loader*.

The file's own comment names what is missing:

> e2m1 has 16 representable values; the table below is those values DOUBLED (so the lookup stays
> integer), paired with a HALF-scale e8m0 — d*half · value*2 == the true product, and keeping both
> integer is what lets the eventual SIMD kernel avoid a float dequant in the inner loop.

**"The eventual SIMD kernel" does not exist.** The representation was deliberately chosen to make
one possible and nobody has written it. That is the single most actionable item in this doc, and it
needs no hardware we do not already own.

---

## Gate 0 — hardware, and it splits this doc in two

The reason FP4 is interesting industry-wide is that Blackwell tensor cores consume it **natively**,
without a dequant step. NVIDIA's own positioning at CES 2026 put NVFP4 as the compression story for
DGX Spark, claiming up to 70% model compression with performance gains.

That acceleration requires Blackwell-class silicon (FP4 tensor cores). **Confirm what our CUDA box
actually is before proceeding** — read `docs/hardware-matrix.md` rather than assuming — but the
working expectation is that it is generations older than Blackwell and has no FP4 tensor path at
all.

So the work splits cleanly:

| | needs new hardware | value |
|---|---|---|
| A. Native MXFP4 SIMD kernel on CPU | **no** | the anticipated kernel, unmeasured |
| B. MXFP4 beyond gpt-oss | **no** | loader/registry scope, not arithmetic |
| C. NVFP4 read support | **no** | load checkpoints published in NVFP4 |
| D. NVFP4/MXFP4 native tensor-core compute | **yes — Blackwell** | the actual industry win |

**Only A, B and C are in scope.** D is recorded here so the reasoning survives, and so that if the
DGX Spark question ever resolves, this doc is where the follow-on starts. Do not begin D.

---

## A — the native MXFP4 kernel (do this first)

**The question:** does computing directly on MXFP4 blocks beat the current dequant-to-f32 path?

The design is already implied by the storage choices. Values are stored doubled as `int8`, scales
are half-scale e8m0, and the product of the two is exact — so the inner loop can stay integer and
skip the float dequant entirely. That is the same shape as the existing `int4` and `w4a8` kernels,
which is where to look for the pattern rather than inventing one.

**Why it might not pay, stated before measuring.** Decode is memory-bound. MXFP4 at 4.25 bits per
weight against int4's block-scaled 4-bit is not obviously fewer bytes, and if the byte counts are
close the kernel change buys arithmetic on a path that is waiting on memory. The win, if there is
one, is avoiding the dequant pass and its intermediate f32 buffer — not bandwidth.

**So the comparison is against `int4`, not against dequant-MXFP4.** The relevant question for a user
is "which format should I run," and int4 is the shipped answer on Apple Silicon CPU decode
(matching or beating int8int8 at half the weight RAM). A native MXFP4 kernel has to beat *that*,
not merely beat its own dequant path.

Report bytes-per-weight for both formats alongside throughput, so it is visible whether any
difference is bandwidth or arithmetic.

---

## B — MXFP4 beyond gpt-oss

Today MXFP4 handling is entangled with the gpt-oss family (`forward_gptoss.go`,
`gptoss_safetensors.go`, `stackedExperts` routing in `decoder/gguf.go:1755`). Establish what is
family-specific and what is format-general.

Note one real trap already recorded in the tree: `decoder/gptoss_safetensors.go:17` documents that MXFP4
nibbles are **sequential** in safetensors (byte j holds elements 2j and 2j+1) where GGML uses a
different order. Any generalization must carry that distinction, and a format-general path that
assumes one ordering will silently produce wrong weights rather than an error.

This is scoping work, not a kernel change. It may be small. Find out before committing to it.

---

## C — NVFP4 read support

NVFP4 and MXFP4 are both 4-bit microscaling formats and they are **not the same format**. Verify
the following against the OCP Microscaling spec and NVIDIA's own documentation before building
anything on it — it is stated here as the expectation to check, not as established fact:

- **MXFP4**: 32-element blocks, e8m0 (power-of-two) scale.
- **NVFP4**: smaller blocks (16 elements), an FP8 e4m3 scale rather than power-of-two, plus a
  second-level per-tensor FP32 scale.

If that holds, NVFP4's finer blocks and non-power-of-two scales should track local weight variation
better, which is the accuracy argument for it. That is also why it is not a drop-in of the MXFP4
reader: two-level scaling is a different dequant path.

**The case for doing this is portability, not speed.** Reading a format costs us nothing at run
time and means checkpoints published in NVFP4 load without waiting for someone to convert them.
That is the same argument that justifies safetensors support — Chapter 5 of `docs/book/` makes it
explicitly: reading what models are *published* as is what lets goinfer run a checkpoint on release
day.

**Establish demand before building.** How many checkpoints are actually distributed in NVFP4 today?
If the answer is "almost none outside NVIDIA's own releases," this is premature and should be
parked with the finding recorded.

---

## Pre-registered gates

**Gate A1 — the kernel is worth writing.** Before writing it, state the bytes-per-weight for MXFP4
and int4 on a representative tensor, and what throughput difference would justify a new kernel and
its parity gating. If MXFP4 is *more* bytes per weight than int4 and the workload is memory-bound,
say so and consider closing.

**Gate A2 — it beats int4.** A native MXFP4 kernel must beat the shipped int4 path on the same
model and box, by a stated margin, with `off` (the current dequant path) as a third arm. Beating
only its own dequant path is not sufficient — nobody would choose MXFP4 over int4 on that basis.

**Gate C1 — demand exists.** A count of real, publicly distributed NVFP4 checkpoints that goinfer
would otherwise be unable to load. Below a stated threshold, park it.

Ambiguous → parked, in every case.

---

## Constraints

- Measure before building. Gate A1 is arithmetic on tensor sizes and needs no code.
- No cgo, on any path.
- Parity gating applies: a new compute kernel must pass the HuggingFace forward-pass gate and
  preserve bit-identical decode, same as every other format.
- Quiet box, settle gate, repeat runs, spread reported. Take the box explicitly.
- `off` is an arm in every comparison.
- Label the regime at the point of recording. A CPU MXFP4 kernel result does not transfer to a GPU
  path, and an Apple Silicon result does not transfer to x86 — the existing MXFP4 work already
  notes that x86 speed and bench numbers were deferred (`decoder/forward_gptoss.go:17`).
- Do not use the words "honest" or "honesty".
- Leave uncommitted for review.

---

## Deliverable

Extend `docs/task-mxfp4-gptoss.md` for A and B — same subject, and that page already carries the
format transcription and the phase history. Open a new section or a new page for C only if gate C1
passes.

Record the gate-0 hardware finding either way. "Native FP4 compute needs Blackwell and we do not
have it" is a durable fact that should not have to be rediscovered, and it is the same hardware
gate that governs the DGX Spark question.
