# Task: f16 KV on the CPU as the command-line default, with f32 kept as the reference (2026-10)

> **Status: SCOPED 2026-10-12. Not started, and not scheduled.** The owner asked for it that day ("can we keep f32
> as an option for kv on cpu, but make default f16? f32 would be important for our reference") and then: "we'll work
> on it at some future point, not sure when". Nothing here is measured. The facts about the code were read at main
> `88665dbe` and re-read at `9f4ecdc3` on 2026-10-13, after a review pass whose claims were each checked against the
> tree (§2 and §4 changed most: decode does not run the path the first draft named). The gates are sketched, not
> registered. Whoever starts it registers them first.

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
- **The other backends:** Metal allocates f16 unless int8 is asked; CUDA takes `--kv auto` (the default) as f16 for
  the dense llama, mistral, qwen2, qwen3 and phi3 families and as f32 for every other (`decoder/kvpolicy.go:Options.kvF16For`);
  WebGPU takes f32 (what `auto` gives it), f16 or int8 (`docs/ARCHITECTURE.md`, the K/V precision table).
- **Who reads the cache.** The int8 cache is the model for a second storage form, and it does not work the way the
  first draft of this document said: nothing in the generic path reads a stored row with its own dot. Every reader is
  handed `[]float32`.
  - **Decode, the generic loop:** `decoder/attention.go:causalAttention` calls `decoder/forwardn.go:attendBatchedHeads`
    for one row, accumulating in float64, so that a decode step is bit-identical to the batched forward. This is the
    path the reference runs.
    - At f32 it is handed the cache's own slices.
    - At int8 a global layer appends the new row (quantised) and then widens the whole stored history into scratch
      (`decoder/kvcache.go:KVCache.dequantGlobalLayer`, which splits the rows across goroutines in
      `decoder/kvdequant_parallel.go:dequantKVRows`; the scratch is `decoder/scratch.go:decodeScratch.localBufs`,
      grown on first use).
    - A sliding-window layer goes another way at int8: `decoder/kvcache.go:KVCache.batchReadLocal` copies the window
      into scratch and round-trips the new row through int8, then `decoder/kvcache.go:KVCache.commitBatch` stores it.
    - At f32 a sliding-window layer reads its ring in place, from a second copy of the window kept beside the first
      (`decoder/kvcache.go:ring.mirrored`, switched by `decoder/kvcache.go:ringDirectDecode`). Both test
      "not int8", which today means f32.
  - **Batched prefill and speculative verify:** `decoder/forwardn.go:Model.runLayersFromEmbedN`, the same two routes
    for more rows. A global layer appends the new rows and widens history and new rows together; a ring layer
    round-trips the new rows in scratch, so prefill attends over the values decode will read back.
  - **`attendBatchedHeads` copies K and V rows only on its fast path.** With float64 accumulation it reads them in
    place, at a stride (aikit's `linalg/matmul_qk_acc64.go:MatmulQKAcc64` and `linalg/matmul_av_acc64.go:MatmulAVAcc64`).
    That is decode, verify, any forward of fewer than 512 rows (`decoder/forwardn.go:fastAttnMinPrompt`), and
    everything under `--exact-prefill` or `--cpu-exact-prefill`. Only a fast prefill of 512 rows or more gathers rows
    into per-head scratch. So "widen as it copies" covers one case, not the common ones.
    - On arm64, decode with six query heads to a KV head and at least 128 keys uses grouped kernels
      (`linalg/matmul_qk_acc64_group.go:MatmulQKAcc64Group`, `linalg/matmul_av_acc64_group.go:MatmulAVAcc64Group`),
      which also read in place.
  - **The own-forward families:** `decoder/attention.go:attendQuery` is called only by the Granite, LFM2, Llama 4,
    Nemotron and Qwen3.5 forwards, and `decoder/forward_gemma4.go` and `decoder/forward_gptoss.go` read the cache's
    slices directly. None of the nine entries in `decoder/arch.go:ownForwards` allows an int8 cache, so
    `decoder/attention.go:attendQueryI8` is reached by tests only.
  - **Code that tests "int8, else f32":** `decoder/prefix_share.go:KVCache.copyStateFrom`,
    `decoder/cpu_batch.go:cpuBatchCacheEligible`, `decoder/forward_qwen35_n.go:Model.runLayersQwen35N`,
    `decoder/cachestate.go:KVCache.holdsState`, `decoder/arch.go:kvBytesForCtx` (it prices the ring's second copy
    for anything that is not int8), and in `decoder/kvcache.go` the methods `KVCache.Append`, `KVCache.storedRows`,
    `KVCache.TruncateTo`, `KVCache.LayerKV` and `ring.write`. A third storage value has to be handled at each.
  - **The bridge that uploads a CPU prefill's K/V to a GPU resident:** `decoder/kvcache.go:KVCache.LayerKV`.
  - **Rewind and sessions:** `decoder/kvcache.go:KVCache.TruncateTo`, and the snapshot format
    (`decoder/kvsnapshot.go:Session.Snapshot`, `decoder/kvsnapshot.go:Model.LoadSession`). The header's storage byte
    is the cache's `kvQuant` value itself (0 for f32, 1 for int8), and a loader refuses a file whose byte differs
    from the cache it is loading into, as a geometry mismatch. A third value needs no version bump for that; an older
    reader refuses the newer file, and snapshots are a cache, so the cost is one cold prefill.
  - **The field tables:** a field added to `KVCache` must be declared in `decoder/cachestate.go:kvCacheFieldState`,
    or `decoder/cachestate_test.go:TestKVCache_everyFieldHasAState` fails. Nothing checks the ring's fields the same
    way.
- **A cache built with a capacity hint reserves f32 it may never use.** `decoder/kvcache.go:NewKVCache` reserves
  capacity for each attention layer's f32 keys and values, and `decoder/model.go:Model.NewCache` switches the cache to
  int8 afterwards (`decoder/kvcache.go:KVCache.setQuant`), which reserves the int8 arrays and leaves the f32 ones.
  - Nearly every caller passes a hint, an ordinary generation included.
  - The f32 arrays have length 0 and are never written. Whether reserved, untouched capacity costs resident memory
    is not something the code says; gate 4 measures it.
  - A sliding-window layer at f32 has the same unused reservation beside its ring today.
- **Two wrong lines that exist today for int8,** and that f16 would repeat unless they are fixed with it:
  - **Pricing reads the option, not the cache.** `decoder/fitguard.go:kvPricingFor` and
    `decoder/prefill_budget.go:Model.AdmitPrefillMemoryShare` price int8 whenever `--kv i8` was given. A family the
    int8 cache excludes (`decoder/arch.go:Architecture.kvInt8OK`: any MoE, any own-forward family) keeps an f32 cache
    and is priced at 1.125 bytes an element against the 4 it allocates.
  - **The banner prints the request.** Off the resident path `internal/serveapp/banner.go:modelBannerFrom` prints
    the `--kv` value, so the same load says "KV i8 (lossy)" over an f32 cache.
- **The half-float kernels exist in aikit, the widening one vectorised.** `linalg/f16scale.go:F16ToF32Slice` widens
  with NEON on arm64 and F16C on amd64. `linalg/f16scale.go:F32ToF16Slice` narrows one value at a time.
  - That is enough for one position's rows in decode. A 2,048-row prefill on Qwen2.5-7B narrows 58.7 million values
    (2,048 rows, 512 wide, K and V, 28 layers), so prefill should split the narrowing by rows, as `dequantKVRows`
    splits the widening.
  - `linalg/f16scale.go:F32ToF16` rounds half up, not ties-to-even, and the hardware narrowing instructions round
    ties to even. A vectorised narrow would differ from this one on ties, so gate 1's "rounded by hand" arm must use
    the same function the cache does.

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
  narrowed as each position is appended. Each new field on `KVCache` is declared in `kvCacheFieldState`. Every
  "int8, else f32" site in §2 handles the third value by name; the ring's in-place read and its second copy stay
  f32-only unless they are built for f16 too, and `kvBytesForCtx` prices whichever was built.
- **Reading, two ways, to be chosen by measurement:**
  - **A. Widen into scratch, as int8 does. Build and measure this first.** A global layer narrows the new rows on
    append and widens the stored history into the scratch the int8 path already uses; a ring layer round-trips the
    new rows through f16 in scratch. Attention then runs on `[]float32` exactly as it does now, on every path:
    in-place float64, the fast prefill, the grouped arm64 kernels.
    - It needs no new aikit kernel.
    - It adds a pass over the whole history on every decode step. The int8 path pays the same pass, and its record
      is `docs/measurements/r15-kvi8-widen-2026-10-05.md`.
  - **B. Widen in the register, inside the kernels. Only if A fails gate 3.** Because the float64 path reads K and V
    in place, this is not one kernel: it is an f16 form of each of the four above (two of them arm64-only) on each
    architecture, plus widening in the fast prefill's gather. aikit's pushes and tags are the owner's.
- **The new rows of a prefill are attended as stored.** Whichever way, the rows a batched forward adds are rounded to
  f16 before that forward attends over them, as the int8 path does for int8. Otherwise prefill and decode see
  different values for the same position.
- **The float64 accumulation stays** in the f16 path's score, so the only difference from the f32 path is the
  rounding of the stored rows. That is what makes gate 1 an exact check.
- **Options:** `KVQuant` gains "f16", and `decoder/model.go:Options.Validate` accepts it (it accepts f32 and i8
  today). The library's zero value stays f32 (decision 1). `internal/loadflags/loadflags.go:CPUKV` is where the
  command lines choose: `--kv auto` (the default) and `--kv f16` give f16 on a CPU load, `--kv f32` gives f32 and
  `--kv i8` gives int8. serve's deprecated `--kv-quant`, when given, still wins.
- **Which families get it:** a rule beside `Architecture.kvInt8OK`, starting from the same set (the generic loop,
  no MoE). A family outside it keeps f32, and says so.
- **Pricing and the banner read the cache's real precision,** not the option: the fit guard's CPU branch, the
  per-request prefill budget and the banner's context line. That fixes the two wrong int8 lines in §2 as well, and
  it is what keeps an excluded family from being priced at half the KV it allocates.
- **Snapshots:** the storage byte's third value. No version bump is needed for an older reader to refuse it.
- **Not in scope:** MoE families and the others the int8 cache excludes need checking one by one, not assuming;
  a GPU resident's precision does not change; int8's own default does not change.

## 5. Gates to register before the code

Sketched here so their cost is visible. The numbers and bands are the registrant's to set.

1. **Exactness, by day.** With every stored row rounded to f16 and back by hand, the f32 path's logits equal the f16
   path's bit for bit, on tiny fixtures across families (a sliding window, shared K/V, m-RoPE, an image block),
   through decode, batched prefill, rewind and a snapshot round trip. This isolates "the storage is f16" from "the
   arithmetic changed". The hand rounding uses the cache's own narrowing function (§2, the rounding rule).
   - **Prefill equals decode.** A batched forward over N tokens against a batched forward over N-1 and one decode
     step, on a global layer and on a sliding-window layer. This is what catches new prefill rows attended at f32.
   - Planted defects: a row narrowed twice, a new prefill row left at f32 while it is attended, a ring read at the
     wrong offset.
2. **Fidelity, at night.** f16 against f32 KV, both scored against the float32 forward with f32 KV, on two or three
   real models, by the non-inferiority protocol in `internal/fidelity`. Include Gemma 3 4B, the family that moves most
   under a small change. The reference arm must be loaded with f32 KV explicitly, which is the reason f32 has to stay.
3. **Speed, by day and then at night.** A whole-token in-process A/B on both machines, at a short depth and a deep
   one (128 and several thousand positions), since the widening costs most where the history is long and the saving
   is bandwidth at that same place. Then a served gate with two arms. The bar is "not slower": this change is for
   memory. The owner's standing rule for speed bars applies to anything claimed as a gain.
4. **Memory.** The heap at a fixed context before and after, read from the process, against the priced figure.
   Measured for a cache built with a capacity hint, which is the ordinary case: this is where the unused f32
   reservation of §2 would show. If it does, the cache releases it when it changes storage.
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

A few days with its gates. It edits the decoder core, so the parity manifest's hashes go stale and are refreshed as
usual:

- storage and its readers: `decoder/kvcache.go`, `decoder/kvdequant_parallel.go`, `decoder/attention.go`,
  `decoder/forwardn.go`, `decoder/forward_qwen35_n.go`;
- the sites that assume two storage forms: `decoder/cachestate.go`, `decoder/prefix_share.go`, `decoder/cpu_batch.go`,
  `decoder/kvsnapshot.go`, `decoder/arch.go`;
- pricing: `decoder/fitguard.go`, `decoder/prefill_budget.go`;
- the command lines: `internal/loadflags/loadflags.go`, `internal/serveapp/banner.go`.

Way A needs no aikit kernel; way B needs four on each architecture. It changes resource use for command-line CPU
loads, which the CHANGELOG entry has to say in its first words.
