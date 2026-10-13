# Task: f16 KV on the CPU as the command-line default, with f32 kept as the reference (2026-10)

> **Status: SCOPED 2026-10-12. Not started, and not scheduled.** The owner asked for it that day ("can we keep f32
> as an option for kv on cpu, but make default f16? f32 would be important for our reference") and then: "we'll work
> on it at some future point, not sure when". Nothing here is measured. The facts about the code were read at main
> `88665dbe`; the gates are sketched, not registered. Whoever starts it registers them first.

## 1. What is asked, and what is still open

- **Asked:** a half-precision (f16) KV cache on the CPU path, used by default, with f32 still selectable because the
  float32 CPU forward is the project's reference.
- **Why it is worth having:** KV is the memory term that grows with the context, and on the CPU it is allocated per
  request at 4 bytes an element. Half of that is the same saving Metal has always had (its resident KV is f16).
  - Per position, K and V, f32: 57 KB on Qwen2.5-1.5B and 115 KB on Qwen2.5-7B, so 0.94 GB and 1.88 GB at 16,384
    positions (`docs/completed/task-cpu-kv-quant.md`, "Problem"). f16 halves each.
- **Open, for the owner, before any code:**
  1. **Which default moves.** Recommended: the command lines (`goinfer-serve`, `goinfer-chat`) default to f16 on a CPU
     load, and the library's zero value stays f32. Then every test, golden, parity record and fidelity reference that
     calls `decoder.Load` keeps running at f32 untouched, and `--kv f32` is the command line's reference setting.
     Moving the library default as well would shift every CPU record the project has.
  2. **Its place in the order.** Unscheduled.

## 2. What is true today

Read in the tree, not run:

- **The CPU cache is f32, or int8 on request.** `decoder/kvcache.go:KVCache` holds each layer's keys and values as
  `[]float32`, with a ring for a sliding-window layer; `decoder/kvcache.go:kvQuant` has two values, f32 and int8.
  `decoder/model.go:Options` carries two separate fields: `KVQuant` for this cache and `KVPrecision` for a GPU
  resident's.
- **`--kv f16` means nothing on a CPU load, and two places speak as if it did.**
  - `internal/loadflags/loadflags.go:CPUKV` maps `--kv f16` to f32 for the CPU cache ("which has no f16 form").
  - The load-time fit guard still prices the KV at f16 when it sees the flag (`decoder/fitguard.go:kvPricingFor`,
    `decoder/fitguard.go:kvBytesPerPosition`), so a CPU load given `--kv f16` is priced at half the KV it allocates.
  - serve's banner prints the requested precision off the resident path (`internal/serveapp/banner.go:modelBannerFrom`),
    so the same load says "KV f16 (lossy)" over an f32 cache.
  - Both go away once the CPU cache has a real f16 form. Until then they are a small under-count and a wrong line,
    worth confirming with one run and fixing on their own if this task stays parked.
- **The other backends:** Metal allocates f16 (or int8) whatever is asked; CUDA allocates f32 whatever is asked;
  WebGPU takes f32 (its default), f16 or int8 (`docs/ARCHITECTURE.md`, the K/V precision table).
- **Who reads the cache**, each of which needs an f16 form or a widening step:
  - decode: `decoder/attention.go:attendQuery`, the scalar dot that accumulates in float64 over the stored history.
    This is the path the reference runs, and the reason it must stay reachable at f32.
  - decode at int8: `decoder/attention.go:attendQueryI8`, the model for a second storage form.
  - batched prefill: `decoder/forwardn.go:attendBatchedHeads`, which copies cache rows into per-head scratch and
    multiplies there.
  - the bridge that uploads a CPU prefill's K/V to a GPU resident: `decoder/kvcache.go:KVCache.LayerKV`.
  - rewind and sessions: `decoder/kvcache.go:KVCache.TruncateTo`, and the snapshot format in `decoder/kvsnapshot.go`,
    whose header has a storage byte with two values today (f32, int8). A third value is a format change; snapshots
    are a cache, so an older reader refusing a newer file costs one cold prefill.
- **The half-float kernels exist in aikit:** `linalg/f16scale.go:F16ToF32Slice` widens with NEON on arm64 and F16C
  on amd64; `linalg/f16scale.go:F32ToF16Slice` narrows one value at a time, which is enough for appending one
  position's rows.

## 3. Prior art, and what has changed since

`docs/completed/task-cpu-kv-quant.md` built the int8 CPU cache and rejected f16 on the CPU by name ("dominated on
every axis; skip it"). Read it before designing anything. Its reasons, and which still hold:

- *"Go has no hardware f16: every read would be a software unpack."* **No longer true.** aikit has had vectorised
  widening since the f16 weight work.
- *"int8 is smaller (4x against 2x) and faster to read (a quarter of the bandwidth, and an integer dot in place of
  scalar float64 dots)."* **Still true.** f16 does not beat int8 on memory or, probably, on speed.
- **What that doc did not weigh is a default.** Its int8 cache is opt-in because it is lossy: it measured an
  average logit cosine of 0.993 and 92.5% argmax agreement on Gemma 3 4B. That is fine for a knob and not for what
  every CPU user gets unasked. f16 is the candidate for a default because it should cost almost nothing in fidelity;
  that "should" is gate 2 below.
- So the question this task answers is narrower than the old one: **is f16 near enough to f32, and at least as fast,
  to be what the command lines use when nobody chooses?** int8 stays the choice for someone who wants the most
  context in a fixed amount of memory.

## 4. Design sketch

- **Storage:** a third `kvQuant` value. Keys and values held as `[]uint16`, in the global store and in the rings,
  narrowed as each position is appended.
- **Decode read:** the dot over the history has to read half-floats. Two ways, to be chosen by measurement:
  - widen a layer's keys into scratch each step and run today's dot. Simple, and it adds a pass over the history.
  - a dot that widens in the register as it goes. No extra pass; it is a new aikit kernel on each architecture, and
    aikit's pushes and tags are the owner's.
- **The float64 accumulation stays** in the f16 path's score, so the only difference from the f32 path is the
  rounding of the stored rows. That is what makes gate 1 an exact check.
- **Prefill read:** `attendBatchedHeads` already copies rows into scratch; it widens as it copies.
- **Options:** `KVQuant` gains "f16". The library's zero value stays f32 (decision 1). The command lines' `--kv`
  default becomes f16 for a CPU load, and `CPUKV` stops mapping f16 away.
- **Pricing and the banner** then describe what is allocated: the fit guard's CPU branch, the per-request prefill
  budget and the banner's context line all read the cache's real precision.
- **Snapshots:** the storage byte's third value, and the version guard's decision about older files.
- **Not in scope:** MoE families and the others the int8 cache excludes need checking one by one, not assuming;
  a GPU resident's precision does not change; int8's own default does not change.

## 5. Gates to register before the code

Sketched here so their cost is visible. The numbers and bands are the registrant's to set.

1. **Exactness, by day.** With every stored row rounded to f16 and back by hand, the f32 path's logits equal the f16
   path's bit for bit, on tiny fixtures across families (a sliding window, shared K/V, m-RoPE, an image block),
   through decode, batched prefill, rewind and a snapshot round trip. This isolates "the storage is f16" from "the
   arithmetic changed". Planted defects: a row narrowed twice, a ring read at the wrong offset.
2. **Fidelity, at night.** f16 against f32 KV, both scored against the float32 forward with f32 KV, on two or three
   real models, by the non-inferiority protocol in `internal/fidelity`. Include Gemma 3 4B, the family that moves most
   under a small change. The reference arm must be loaded with f32 KV explicitly, which is the reason f32 has to stay.
3. **Speed, by day and then at night.** A whole-token in-process A/B on both machines, at a short depth and a deep
   one (128 and several thousand positions), since the widening costs most where the history is long and the saving
   is bandwidth at that same place. Then a served gate with two arms. The bar is "not slower": this change is for
   memory. The owner's standing rule for speed bars applies to anything claimed as a gain.
4. **Memory.** The heap at a fixed context before and after, read from the process, against the priced figure.
5. **The reference is unmoved.** Every existing golden and parity record passes unchanged with the library default
   at f32, and the fidelity gates' references are shown to be loading f32 KV (an assertion, not an assumption).

## 6. Risks

- **A slower default.** If the widening pass costs more than the bandwidth it saves at ordinary depths, f16 loses
  gate 3 and stays an option, not the default. That is a legitimate outcome.
- **A drifting reference.** If any reference path picks up the new default, measurements made against it stop
  being comparable with earlier ones without anything failing. Gate 5 exists for this.
- **Near ties.** At int4 a model's logits sit near ties often enough that a rounding-level change moves greedy
  replies (the wide-head attention kernel did exactly that on Gemma 3 4B, 2026-10-12). Replies will differ between
  f16 and f32 KV; the fidelity gate, not reply equality, is what says whether that matters.

## 7. Size, and what it touches

A few days with its gates. It edits the decoder core (`decoder/kvcache.go`, the attention read paths, the
snapshot format), so the parity manifest's hashes go stale and are refreshed as usual; it may need one aikit
kernel on each architecture; and it changes resource use for command-line CPU loads, which the CHANGELOG entry has
to say in its first words.
