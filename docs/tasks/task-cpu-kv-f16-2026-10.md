# Task: f16 KV on the CPU as the command-line default, with f32 kept as the reference (2026-10)

> **Status: APPROVED 2026-10-15 (owner), to build after G40 passes; gates registered 2026-10-14 (§5).** The owner asked for it on 2026-10-12 ("can we keep f32
> as an option for kv on cpu, but make default f16? f32 would be important for our reference"). On 2026-10-14 the owner
> asked for a recommendation on the two open decisions (§1, answered in §1a) and for the gates, with building to follow
> G40 by day if approved. Nothing here is measured yet. The facts about the code were read at main `88665dbe`, re-read
> at `9f4ecdc3` on 2026-10-13 after a review pass, and spot-checked at `afd83410` for §5.

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

## 1a. Recommendation on the two decisions (2026-10-14, for the owner)

**Owner, 2026-10-15: approved as recommended below.** Building starts once G40's night gates (G40-2, G40-3) pass.

**Decision 1, which default moves: the command lines, not the library.** This is §1's recommendation, with two
refinements.

- **The library's zero value stays f32.** Every test, golden, parity record and fidelity reference that calls
  `decoder.Load` keeps running at f32 without being touched. The reference is the reason f32 exists at all, and a
  library default that moved it would shift every CPU record at once.
- **On the command lines, `--kv auto` becomes f16 only once the gates have read, in a second step.**
  - First, `--kv f16` works on a CPU load: a real f16 cache, priced and printed as f16.
  - Then `auto` flips in its own commit, only if gates F2, F3 and F4 read as registered.
  - That is the shape the CUDA f16 default had (`docs/tasks/task-cuda-f16-kv-2026-10.md`, "Decision"). A PARKED read
    leaves `--kv f16` available and `auto` at f32. The owner's override stays the owner's.
- **The families start as `Architecture.kvInt8OK`'s set,** the generic loop with no MoE. Every other family keeps
  f32, and the banner says which and why. MoE and the own-forward families are each checked separately later, not
  assumed.

**Decision 2, the order: after G40, the next day if you approve; independent of KV2's merge.**

- **G40 first, because it changes what f16 is worth.**
  - Today the KV is a minority of a long CPU prompt's footprint. At 20K tokens on the 0.5B it is 0.47 GiB of 3.1 GiB
    (`docs/tasks/task-cpu-prefill-memory-2026-10.md` §1); the rest is prefill scratch and garbage that G40 removes.
  - Once G40 lands, the KV is the largest term. f16 halves it: 0.24 GiB saved of a predicted ~1.1 GiB on that cell, and
    0.55 GiB of the 1.5B's 1.10 GiB of KV at the same length.
  - Before G40, f16 would be measured against a footprint it cannot move much.
- **G40 and f16 both touch `decoder/kvcache.go` and `decoder/forwardn.go`,** in different places: G40 the reserve
  and the chunk loop, f16 the storage and the widening. Building them one after the other avoids two branches in the
  same functions.
- **KV2 (`kv2-2026-10`) adds one more "int8, else f32" site:** its CPU copy between sessions
  (`KVCache.copyPrefixFrom`, on that branch only). Whichever of KV2 and f16 merges second
  handles the third storage form there. Gate F1 includes that copy once KV2 is on `main`.
- **int8 stays the choice for the most context per gigabyte.** KV9 step 0 noted that `--kv i8` already holds four
  sessions' KV in about one f32 session's bytes. f16 is for a default nobody chooses, int8 for the user who does.

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

## 5. Gates (registered 2026-10-14, before any code)

These replace the sketch this section held. The fidelity criteria are the CUDA f16 gate's, unchanged
(`docs/tasks/task-cuda-f16-kv-2026-10.md`, "The gates"), so the two backends' defaults are judged by the same rule.
Tiers are CLAUDE.md's ("Gate tiers").

### F1. Exactness (day, tiny fixtures; seconds)

**The comparison.** In the f32 path, every stored K and V row is rounded to f16 and back by hand, with the cache's
own narrowing function (§2, the rounding rule). That path's logits must equal the f16 path's bit for bit.

**Fixtures:** `llama-tiny` (dense), `gemma2-tiny` (a sliding window, so the ring) and `qwen25vl-tiny` (m-RoPE and an
image block). `mixtral-tiny` asserts the opposite: an excluded family keeps an f32 cache and the banner names it.

**Paths:**

- decode, and batched prefill at under and over 512 rows (the exact path and the fast gather);
- rewind (`KVCache.TruncateTo`);
- a snapshot round trip (`Session.Snapshot`, `Model.LoadSession`);
- the GPU-upload bridge (`KVCache.LayerKV`);
- KV2's session copy, once that is on `main`.

**Prefill equals decode:** a batched forward over N tokens against a batched forward over N − 1 and one decode step,
on a global layer and on a ring layer.

**Plants, each of which must go red:**

- a row narrowed twice;
- a new prefill row attended at f32;
- a ring read at the wrong offset;
- the snapshot's storage byte left at f32's value.

**PASS:** zero differing bits in every cell. There is no tolerance.

### F2. Fidelity (night; nobara for the 7B, either box for the rest)

**Instrument.** The CUDA gate's protocol, run on the CPU:

- arms `--kv f32` and `--kv f16`, in one binary and one session;
- teacher-forced on the reference tokens;
- against the CPU f32/f64 reference rows of held-out prompt set B (`testdata/prefill-gate-prose-b/`, the rows in
  `~/goinfer-logs/prefill-ref-b`), ten prompts of 64 positions.

The f16 cache has no tunable parameter, so set B was not fitted to it.

**Cells:**

- **Decision cells:** Qwen2.5-7B-Instruct q4_k_m at K = 8,000, and Gemma 3 4B at K = 3,900. Gemma 3 is included
  because it moves most under a small numeric change (§6, near ties).
- **Confirmation cell:** Qwen2.5-Coder-1.5B at K = 3,900. It cannot rescue a failed decision cell.

**Criteria, all three per cell:**

- (a) hard flips: f16 ≤ f32 + 2·√f32.
- (b) mean teacher-forced agreement: f16 ≥ f32 − 1.0 pt, AND f16 ≥ f32 on at least half the prompts.
- (c) mean continuation KL: f16 ≤ 1.10 × f32.

**Ambiguous is parked:** (b) within its last 0.2 pt, or (c) in 1.05 to 1.10×. A re-run needs a named mechanism; it is
never a re-roll.

**Preconditions (a failed one voids the cell; it is not a fail):**

- the f16 arm's cache holds half the f32 arm's KV bytes, read from the cache;
- its rows differ from f32's at one position or more in every prompt;
- an A/A of two f16 runs is bit-identical on the first prompt;
- the reference arm's cache is asserted to be f32 (`cache.quant`), not assumed;
- **an overflow survey before any gate is read:** max|K| and max|V| at every layer on the gate prompts must be under
  65504 / 2. A family within 2× of f16's range is refused f16 by the default, whatever the gate says, and the survey
  is a table in the record.

**Prediction (written now):** the 7B and 1.5B cells PASS (0.8); Gemma 3 4B PASSES (0.55); every cell's KL ratio is
under 1.03 (0.6).

### F3. Speed (in process by day; served at night)

**First, by day:** a whole-token in-process A/B of decode, f16 against f32, interleaved (TE5(b)), on the 0.5B and
1.5B at depths 128 and 4,096, on the Mac. A resolved direction may be acted on; an unresolved one goes to the night.

**Then, at night, served on both boxes:**

- the two-arm form of `bench_peer.py` (`BENCH_ENGINES=goinfer,goinfer_old`; arms `--kv f32` and `--kv f16`, same
  binary, fresh serve per arm, alternating order);
- at 128, 512, 2,048 and 3,900 on the 0.5B and 1.5B, and 8,000 on the 7B (nobara only);
- the do-nothing arm is `--kv f32`.

**The quantity:** median decode rate, f16 ÷ f32.

**Bars.**

- **Ships, as the default:** ≥ 0.98 at every cell. This change is for memory; the bar only forbids a regression.
- **Parked:** the worst cell in [0.95, 0.98).
- **Killed:** any cell under 0.95.

Way B (§4) is built only if way A is parked or killed by this gate.

**Prediction:** the 128-key cells within ±2% of 1.00 (0.7); a deep cell under 0.98 under way A, from the widening
pass (0.35).

### F4. Memory (night, Mac; with G40-2's harness)

- **The cache's bytes,** read from the cache at 16,384 positions on the 1.5B: f16 holds f32's ÷ 2, exactly (to the
  byte, for the rows written).
- **The process footprint** at that context, f16 against f32: f16 ≤ f32 − 0.8 × (f32 KV bytes ÷ 2). That is at
  least 80% of the halving, seen by the process.
- **The unused f32 reservation (§2):** an f16 cache built with a capacity hint has a footprint within 32 MiB of the
  same cache built without one. If not, the cache releases the reservation when it changes storage, and this is
  re-read.

### F5. The reference is unmoved (day, then as part of the next sweep)

- Every forward golden and parity record passes unchanged with the library default at f32
  (`scripts/refresh_parity_hashes.sh`, and the next `gate parity`).
- F2's reference arm asserts f32 KV.
- One new test asserts that `decoder.Load` with zero options builds an f32 CPU cache. That is the guard against the
  library default ever following the command lines.

### The decision

**`--kv auto` becomes f16 on CPU loads only if all of these hold:**

- F1 passes;
- every F2 decision cell passes;
- F3 ships;
- F4 and F5 pass.

**Otherwise:**

- a PARKED read leaves `--kv f16` built and available, with `auto` at f32;
- a FAIL on F2 leaves it opt-in, with the reason recorded.

The owner may override, as on CUDA. The override is recorded with the gates that did not pass.

## 5a. Registered 2026-10-16, before the build: what f16 changes against f32, and the parity manifest (owner's ask)

**What f16 changes.** One thing: each stored K and V row is rounded to f16 when it is appended (and a prefill's new rows
are attended as stored, §4). Scores still accumulate in float64 and attention still runs on `[]float32`, so an f16 run
equals an f32 run whose stored rows were rounded by hand. That is F1's exact check. Nothing else in the forward moves.

**What it must not change: the f32 path, bit for bit.** The library's zero value stays f32 (decision 1), so every test,
golden and parity record keeps running at f32. Two checks pin that, before any parity hash is refreshed:
- **F0, identity (tier 0, by day, seconds to minutes):** `go run ./cmd/gate identity` between main before the change
  and the branch, on the tiny assets, every family at f32: full logits byte-identical (TE6(b)). The f32 storage and
  every reader of it are unchanged code paths, and F0 proves the third storage form did not leak into them.
- **F5,** as above: the forward goldens and parity records pass unchanged, and one test asserts `decoder.Load` with zero
  options builds an f32 CPU cache.

**The parity manifest.** It records the f32 reference only; an f16 cache never enters it.
- The decoder edits stale its deps hashes. They are refreshed by `scripts/refresh_parity_hashes.sh`, which runs the
  forward goldens as its proof and refuses on any failure, after F0 has passed. No family is re-validated, since F0 shows
  its f32 numerics did not move.
- If F0 shows any family's f32 logits moved, the change is numeric for that family: stop and report, and nothing is
  refreshed.

**Tiers.**
- **Tier 0, by day:** F0, F1, F5.
- **Tier 1, by day:** F3's in-process A/B on the 0.5B and 1.5B.
- **Tier 2, at night:** F2 and F4, and F3's served cells. They decide a default that will be published, F2 on a small
  numeric effect at depths of 3,900 to 8,000 keys, and F4 at 16,384 positions. Each is estimated and fit-checked by day
  before it is queued.

**The default stays f32** on the command lines as well, until F1–F5 read as registered and the owner says so (§1a).
The build adds `--kv f16`, a real f16 cache on a CPU load, priced and printed as f16.

## 5b. Built and read, 2026-10-16 (way A; branch `cpu-kv-f16-2026-10`)

- **Built:**
  - A third storage form (`kvqF16`): rows narrowed to binary16 on append (`linalg.F32ToF16`) and widened into the int8 path's
    scratch on read, at every cache site of §2, the two session copies and the snapshot (storage byte 2).
  - `Options.KVQuant` `"f16"`, built where `Architecture.kvF16OK` allows (the generic loop, no MoE).
  - On the command lines `--kv f16` builds it, and `--kv auto` stays f32.
  - The fit guard and the per-request budget price the CPU cache at its real precision. That fixes §2's under-count of an
    f32 cache priced at f16.
  - The banner prints the CPU cache's precision, and why an f16 request was held at f32.
- **F0, identity: PASS.** `gate identity` between the branch's base and the build, on this Mac's CPU:
  - 37 families IDENTICAL at f32, int8int8 and int4 on the tiny assets, none different;
  - qwen2 IDENTICAL on the 0.5B at f32 and int4;
  - five not run (missing tiny assets, or real-model slices).
  - The f32 path did not move, so the parity hashes are refreshed and nothing is re-validated.
- **F1, exactness: PASS,** zero differing bits in every cell. Fixtures as registered, plus one choice made at the run: the
  tiny fixtures' windows hold 64 to 128 positions, so the cell at 512 rows or more (the fast gather) ran on
  Qwen2.5-Coder-0.5B, 600 rows.
  - Cells: decode, batched prefill (with qwen25vl's image block and m-RoPE positions), the GPU-upload bridge, rewind,
    snapshot round trip, KV2's session copy, and prefill-equals-decode on a global and a ring layer.
  - mixtral-tiny keeps f32, and says why.
  - **Plants:**
    - new prefill rows attended at f32: 33,213 values differ, red;
    - the ring read one slot late: 17,374, red;
    - the snapshot's storage byte left at f32: refused on restore, red.
    - **A row narrowed twice cannot go red,** and is not built: f32 to f16 to f32 to f16 gives the same bits, because every f16
      value is exact in f32.
- **F5, the library default:** `TestKVF16_F5LibraryDefaultIsF32` PASS, and the forward goldens pass in the parity refresh.
- **Still owed, before `auto` can move:** F3's in-process A/B by day, then F2, F4 and F3's served cells at night, each
  fit-checked before it is queued. Until they read, and until the owner says so, the default stays f32 everywhere.
  Record: `docs/measurements/cpu-kv-f16-2026-10-16/`.

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
