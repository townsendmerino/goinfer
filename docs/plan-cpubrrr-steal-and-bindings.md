# Plan: steal from cpubrrr, ship multi-language + mobile bindings

**Owner:** vscode-claude · **Repos:** `~/tmcode/{aikit,goinfer}` · **Drafted:** 2026-07-26

Two workstreams, deliberately decoupled. **A** is high-confidence throughput work with a
known payoff. **B** is a bet with a measurement gate in front of it.

Run A and B-0 in parallel. Do not start B-1..B-4 until B-0's gate is passed or the
fallback is chosen.

---

# Status (2026-07-31) — five days in

| task | state | where |
|---|---|---|
| **A1** Q8_K integer-accum kernel | ◐ **built, measured, DECLINED for Q6_K — Q4_K left open** | aikit `linalg/kquant.go` |
| **A2** MXFP4 + gpt-oss | ✅ **shipped** | `decoder/mxfp4.go`, `gpt_oss` in `decoder/registry.go:57`, GGUF loader + parity + decline |
| **A3** yielding spin-barrier | ☐ not started | — |
| **B** bindings (sidecar + c-archive) | ☐ not started | `docs/task-bindings.md` still reads *Status: NOT STARTED* |
| A2 deliverable §6.6 — gpt-oss tok/s row | ☐ not published | `docs/benchmarks.md` has no gpt-oss entry |

**A1 is a negative result, not a skip — and it is the most important line here.**
`linalg/kquant.go` implements cpubrrr's algorithm (Q4_K/Q6_K weights × Q8_K activations,
integer accumulation, one float conversion per 256-superblock), bit-exact and benchmarked,
deliberately **not** wired into `WeightMat` dispatch. The ≥1.3× decode gate is *provably*
unreachable for Q6_K: byte-ratio ceiling 256/210 = **1.22×** against the int8 W8A8 path,
regardless of unpack quality (aikit's CPU-acceleration note, uncommitted in that repo → "Native K-quant
matmul — evaluated, NOT shipped"). The file states the revisit condition explicitly:

> a future revisit (e.g. a bandwidth-bound **Q4_K variant, ceiling 1.78×**) starts from a
> validated base rather than scratch.

**That revisit condition is now met — by a different task.** See below.

> **↔ Cross-reference (2026-08-04): `docs/ollama-chase.md` §D5 is the campaign that gives this
> declined lever a reason to exist.** A1 was declined *because the CPU path did not matter*. The
> hybrid GPU/CPU **layer split** for oversized models (the 26B-on-8GB case) runs ~58% of layers on
> the CPU, so its throughput is gated directly by CPU-kernel speed — that is exactly what A1's Q4_K
> variant would buy. **Honest bound, so this reads straight:** even at the full **1.78×** Q4_K
> speedup the split lands ~**16–18 tok/s** against Ollama's 24.5 — it *revalidates* A1's Q4_K door
> (worth reviving) but does **not** make the 26B winnable; only more VRAM does. So revive A1-Q4_K
> for the capability + the CPU-decode ceiling, not as a way to beat Ollama on this model.

---

# The convergence: A1's Q4_K door and the Gemma 4 quant blocker are one task

> **UPDATE (2026-08-01, `da5a6ec`): the "convergence" premise is void — the Gemma 4 quant blocker
> was a PROMPTING artifact, not a quantizer deficit.** Real int4 (sym W4A8) is fully coherent under
> the chat template (`TestGemma4_26B_gate`, distinct-trigram 0.841); the "garbage at int4" below was
> a raw completion prompt on an instruction-tuned model, and int8 is equally garbage there. So
> **MXFP4/Q4_K are NO LONGER NEEDED for Gemma 4 coherence**, and the "Action" reconstruction-table
> work below is moot *for coherence*. **But A1's Q4_K THROUGHPUT door stays OPEN — do not file it as
> resolved.** The bandwidth-bound Q4_K variant (measured byte-ratio ceiling **1.78×** vs the int8
> W8A8 path) is untouched by this result; it is a throughput lever independent of the coherence
> question, and the shared 4-bit encoder work should be scoped to that, not to unblocking Gemma 4.

`docs/completed/task-gemma4-moe.md` is stuck at Phase 5. The real 26B-A4B is coherent at int8
(~26 GB, needs the 64 GB box) and **garbage at int4** (~13 GB) — which is the size that has
to work, because the fieldfare-comparable rig is an M1 Pro 16 GB. Measured reconstruction
(commit `bcadd44`): symmetric-g32 **0.99514** (garbage) vs affine-g32 **0.99690** vs int8
**0.99995**. The diagnosis is that goinfer's 4-bit is **group-wise symmetric**,
`int4GroupSize = 32`, codes `[1,15]` with 8 = zero → 15 levels, no zero-point, maxabs
scale (`linalg/quant.go:414`, `QuantizeGroupInt4Row` — retargeted 2026-08-15 after the aikit v1.19.0
bump; the prior `:327–331` citation had drifted onto an unrelated `q8Span` line via a non-discriminating
match on `for i := range M {`, not this function).

cpubrrr's two 4-bit formats are precisely the alternatives:

| | structure | zero-point | block | goinfer has |
|---|---|---|---|---|
| goinfer int4 | uniform, symmetric `[-7,7]`, maxabs scale | no | 32 | quantize + dequantize + W4A8 |
| **Q4_K** | uniform, `d(f16)` + `dmin(f16)` + 6-bit sub-scales/mins | **yes (affine)** | 256 superblock / 32 sub | **dequantize only** (`gguf.go`, mirrored in `kquant.go`) + the validated Q4_K×Q8_K matmul |
| **MXFP4** | **non-uniform ladder** `{0,1,2,3,4,6,8,12}` × E8M0 scale | no (symmetric, non-uniform) | 32 | **dequantize only** (`mxfp4.go`, bit-verified vs `gguf/quants.py`) |

Both handle the outlier-driven-maxabs failure mode that uniform symmetric int4 does not.
Both dequant paths already exist and are parity-verified, so **adding MXFP4 and Q4_K
columns to that reconstruction table is nearly free** — do it before building a bespoke
affine int4 quantizer plus a `.giw` v5.

**One shape constraint likely settles it.** Q4_K requires `K % 256 == 0`
(`QuantizeActQ8K` panics otherwise; `qkK = 256`). Gemma 4's `down_proj` has
K = `moe_intermediate_size` = **704**, and 704 / 256 = 2.75 — **it does not tile**.
MXFP4's 32-element blocks give 704 / 32 = 22 ✓ (the same divisibility that makes today's
group-32 int4 fit). So on this model's shapes MXFP4 fits and Q4_K does not, absent padding
or a mixed scheme.

Honest caveats before anyone treats MXFP4 as the answer: its E8M0 scale is power-of-two
only (coarser than Q4_K's f16 `d`), and it has no zero-point — it is non-uniform
*symmetric*, not affine, so it is not a drop-in substitute for the affine hypothesis.
Whether it wins on *these* tensors is empirical. And goinfer has **no encoder** for either
format — both are read-only today — plus `.giw` representation needs checking. That
encoder is still a smaller lift than a new format, and it is shared with A1's revisit.

**Action:** add `mxfp4` and `q4_k` columns to the Step 3 reconstruction table in the Gemma 4
quant investigation. If MXFP4 clears the coherence bar, A1's Q4_K revisit and the Gemma 4
blocker resolve together, and goinfer gains a 4-bit encoder it needs for both.

---

# A3 should be ranked higher than this plan originally put it

cpubrrr runs **12 persistent workers on a yielding spin-barrier** and measured condvar
wakeups at **7.5 ms/token**. Set that against goinfer's own conclusion in
`docs/completed/perf-campaign.md`:

> **The scaling is the problem, not the kernel.** Serial = 51 tok/s; 8-core ~1.3× scaling
> … ~68 tok/s is the practical ceiling for pure-Go batch=1 CPU decode with this approach.

The campaign identified thread scaling as the wall and stopped. cpubrrr shipped the fix for
that exact wall. This is not a speculative transfer — it targets the number goinfer's own
perf work already named as the ceiling, and it is independent of every quantization
question above.

---

# The first legal peer cell

Unlike `turbo-fieldfare` (where no comparable measurement exists until Gemma 4 MoE lands),
cpubrrr's headline model is one goinfer **already supports**: gpt-oss:20b at MXFP4, same
model, same quant. Only the hardware differs — cpubrrr measured an M4 Max; goinfer's rigs
are an M1 Pro 16 GB and a Ryzen 7 3700X.

cpubrrr's published figures (M4 Max, CPU-only):

| model | quant | cpubrrr | llama.cpp CPU | ratio |
|---|---|---|---|---|
| gpt-oss:20b | MXFP4 | **~77 tok/s** | ~14 | **5.5×** |
| Qwen3-Coder-30B | Q4_K/Q6_K | **~92 tok/s** | ~82 | 1.1–1.2× |

Per `docs/benchmarks.md`'s methodology rule this still needs same-machine numbers, but the
*checkpoint and quant* axes are already matched — which is exactly what A2 deliverable §6
item 6 (a gpt-oss:20b decode row with full provenance) would close. It remains unpublished.
**Peer figures are unpinned**: re-read cpubrrr's README and pin its commit before any
published comparison row.

---

## Workstream A — take the CPU kernel win from cpubrrr

`arizqi/cpubrrr` beats llama.cpp's own CPU path on Q4_K (~92 vs ~82 tok/s) and by ~5× on
MXFP4 MoE, on an M4 Max, CPU-only, in 5.5k lines of Rust with one dependency. The wins are
algorithmic and portable. Three tasks:

| task | repo | file | why |
|---|---|---|---|
| **A1** Q8_K integer-accumulation kernel | aikit | its Q8_K integer-accum task note (uncommitted in aikit) | The kernel that took cpubrrr from losing to winning on Q4_K. Lands in `linalg`, Experimental tier. |
| **A2** MXFP4 + gpt-oss support | goinfer | `docs/task-mxfp4-gptoss.md` | goinfer has **zero** MXFP4/gpt-oss support today (`git grep -il mxfp4` → empty). This is a missing model family, not just a speed knob. |
| **A3** Yielding spin-barrier for the CPU decode pool | goinfer | folded into A2 §5 | cpubrrr measured condvar wakeups costing **7.5 ms/token**. Pure spin collapses under jitter; spin-then-yield fixes both. |

### The strategic point on A1

You already have aikit's native-Q6_K kernel note (uncommitted in that repo), **DEFERRED 2026-06-13**,
covering a fused native-K-quant kernel. It was deferred because the justification examined
was *fidelity and footprint vs int4-requant* — and those numbers came back weak (native
Q6_K is 31% **larger** than int4; your selective-int8-head already captures most of the
fidelity win).

That analysis was correct and A1 does not contradict it. **A1 un-defers the task on the
axis that doc explicitly set aside** — §5's compute number, which it said "only matters if
that capability is the goal." cpubrrr is the evidence that it is: their win was not
fidelity or footprint, it was **decode throughput**, and it was large enough to beat a
mature hand-tuned kernel on its home turf.

So A1 is scoped as a *throughput* task with a *throughput* gate, not a revival of the
fidelity argument. Read the deferred doc first — its §4 architecture notes (`linalg` and
`embed` as independent siblings, the `WeightMat` seam) are pick-up-ready and still valid.

---

## Workstream B — bindings: sidecar (desktop) + c-archive FFI (mobile)

Full detail in `docs/task-bindings.md`. The short version:

**A WASM-for-everything plan was drafted and withdrawn.** Three facts killed it:
`GOARCH=wasm` satisfies `//go:build !arm64 && !amd64`, so `linalg/dot_other.go` aliases
every kernel to `dotGeneric` and **all SIMD goes inert**; wasm cannot `dlopen(libcuda)` or
reach Metal, so **both GPU backends disappear**; and wasm32 caps usable memory near
**2–3 GiB**. On iOS the JIT ban then forces AOT compilation to a native static library
anyway — reproducing the per-platform artifact WASM was chosen to avoid, with worse
codegen. It paid the full cost of portability and got none of it.

**The replacement splits by deployment shape, not by language:**

| half | mechanism | targets | properties |
|---|---|---|---|
| **desktop / server** | **sidecar binary** | Python, Node | native speed, CUDA + Metal, no size ceiling, zero cgo, ~500 lines of pure-language client each |
| **mobile** | **c-archive / c-shared FFI** | Swift, Flutter, React Native | native ARM64 + NEON, **Metal reachable on iOS**, in-process, one C ABI for all three |

Both halves consume the OpenAI/Anthropic JSON `cmd/serve` already speaks, so there is no
new API surface to design, test, or document — and **every binding inherits constrained
decoding on day one**, which no llama.cpp wrapper can offer.

**On the cgo-free promise:** `-buildmode=c-archive` needs `CGO_ENABLED=1` in *your CI*, at
build time. The library and default build stay pure Go; consumers of the `.xcframework`
never see a C toolchain. Same quarantine pattern as the existing `gpu` submodule.

**Browser/WASM is not dead, it is reclassified.** `docs/roadmap.md`'s `GOOS=js` +
`syscall/js` → `navigator.gpu` demo — a 0.5B in a tab, cgo-free — is a genuine capability
no other Go runtime can ship. It is a demo, not a binding strategy.

---

## Phases

```
A1 (aikit Q8_K) ──┐
A2 (MXFP4/gpt-oss)─┼── independent, ship on their own merits
A3 (spin-barrier) ─┘

B: A0 server prereqs ─> A1 Python ─> A2 Node          (weeks, near-zero risk)
   B0 spikes ─────────> B1 C ABI ─> Swift ─> Flutter ─> RN
```

**The one measurement that matters:** B0.1 — does the cgo-free `metal` backend work under
`c-archive` on a real iPhone? The backend is purego + Obj-C at `CGO_ENABLED=0`, and
c-archive forces cgo on; they should coexist, but iOS also restricts `dlopen` to system
frameworks (`Metal.framework` qualifies). Verify on device, not simulator. **If it works,
iOS becomes the fastest binding in the set** — GPU-accelerated on-device inference with no
Python and no llama.cpp. Run this spike early, even out of order.

Ship the sidecar half first regardless: it is cheap, it is risk-free, and it gets real
binding users while the mobile spikes resolve.

---

## Sequencing note

A1 and A2 make the mobile numbers better and cut bytes-per-token, but nothing in B depends
on them. The two workstreams are genuinely parallel.

Suggested order: A1 → A2 alongside B's A0 → Python → Node, with B0.1 run as early as a
device is available.

---

## Where these files live

- **this file** — `~/tmcode/goinfer/docs/plan-cpubrrr-steal-and-bindings.md`
- **A1** — aikit's Q8_K integer-accum task note (uncommitted in that repo)
- **A2 + A3** — `~/tmcode/goinfer/docs/task-mxfp4-gptoss.md`
- **B** — `~/tmcode/goinfer/docs/task-bindings.md`

The plan lives in goinfer because three of the four tasks land there; the aikit task doc
cross-references it. Follow the existing convention — in-flight tasks live in `docs/`,
finished ones move to `docs/completed/`.
