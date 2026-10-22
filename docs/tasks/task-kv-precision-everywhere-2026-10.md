# Task: `--kv f16` and `--kv i8` delivered on every backend, family and path (2026-10)

> **Status 2026-10-21 (the close-out sweep).** Phase 0 is done and on main, in no tag yet: five commits, ca49bcf1 to
> c51e13cf (#114, 2026-10-20). One resolver (`decoder.ResolveKV`) decides the precision, an explicit precision that cannot be
> delivered refuses the load by name (D1), the banner and the request log print requested and delivered, and
> `docs/kv-precision-matrix.md` is generated and ratcheted. **Decided by the owner on 2026-10-21:** KV4
> (`docs/tasks/task-kv-architecture-2026-10.md`) goes first. Phase 1 (CUDA f16 for the ten families that refuse it) folds
> into KV4b, and phase 2 (CUDA int8, B39) waits for KV4's pools, so no KV kernel is written twice. That revises D4's order;
> phases 3 to 6 stay this doc's. **Stale below:** the status line's "Phase 0 is next"; §2, which is the tree at 8494a91d
> (the generated matrix is the truth now); §2's `cuda/kvpolicy.go`, since moved to
> `decoder/kvresolve.go:CUDAKVF16Decline`; and "used by every backend's build", since Metal's and WebGPU's builds still read
> the raw request (P0a′ below).

**Ledger, 2026-10-21.** One row per item, the current state; it supersedes the status notes below where they
disagree, and is updated in the same commit as the work. `python3 scripts/task_docs.py check` reads it.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| P0 | done | — | ca49bcf1 to c51e13cf (#114) |
| P0a′ | done | — | branch `p0a-resolvekv-2026-10`; P0A-G1, G2 PASS, `docs/measurements/kv-precision-2026-10/p0a-prime.md` |
| §1.6 | blocked | nothing to build until a Mamba-only family exists | `keepsNoKV` |
| P1 | moved | KV4b in `docs/tasks/task-kv-architecture-2026-10.md` (owner, 2026-10-21) | — |
| P2 | blocked | KV4's pools (owner, 2026-10-21); the sizing stays in B39 | B39 |
| P3 | open | Mac: Metal int8 everywhere and f32 back; start after `kv8a-metal-2026-10` merges, which edits `metal/backend.go` and `metal/model.go`; M–L | 12 matrix cells "until phase 3" |
| P4 | blocked | P3, by the doc's order | 38 matrix cells "until phase 4" |
| P5 | blocked | P4; a park candidate (trigger: phase 4 lands, or a request for WebGPU f16 or i8) | no matrix cells yet |
| P6 | blocked | P1–P5 | 12 matrix cells "until phase 6" |
| G1 | blocked | P1 and P2, with KV4 | — |
| G2 | blocked | P1–P5 | — |
| G3 | open | a tier-0 readback of delivered bytes for every cell: CPU and Metal cells on the Mac, CUDA cells on nobara; S–M | `TestResidentKVBytes_matches*Allocation` covers a few fixtures |
| G4, G5, speed | blocked | each phase | the instrument is in `docs/tasks/task-cuda-f16-kv-2026-10.md` |
| D1–D3 | done | — | §6 |
| D4 | done | revised 2026-10-21: KV4 first, phase 2 after the pools | §6; the status block above |


> **Status: APPROVED 2026-10-20 (owner: D1 refuse, D2 honor, D3 WebGPU in scope, D4 the recommended order; §6). Phase 0 is next. Filed 2026-10-20 at the owner's request: a KV precision the caller asks for
> is the precision the caller gets, on every backend, every family and every path. Supersedes B39 (CUDA int8 KV, a
> candidate), which becomes phase 2 here.** The facts about the code were read at main `8494a91d`. Nothing here is
> measured; the sizes are estimates.

## 1. The contract

1. `--kv f32`, `--kv f16` and `--kv i8` each name a storage precision for the KV cache. A load given one of them either
   runs at that precision on every path the request takes, or **refuses to load and says why**. It never runs another
   precision without saying so.
2. `--kv auto` keeps its meaning: the backend picks (today f16 on CUDA for the cleared families, f16 on Metal, f32 elsewhere).
   `auto` changes only through its own gate and the owner's word, as the CUDA and CPU f16 defaults did. Nothing in this
   plan moves a default.
3. The library's zero value stays f32. f32 stays the exact reference every gate compares against.
4. **An explicit request is honored whatever its quality reads.** The kernel gates (§4, G1 to G3) decide whether a
   precision is *implemented correctly* on a path; the quality read (G4) is a published number, not a refusal. Quality
   decides only what `auto` may pick and what the docs and the banner say. The one exception is a numeric failure, not a
   quality one: f16 overflow (§4, G5), which is fixed by design, not declined.
5. Every load prints the precision **requested and delivered** for each cache it built (resident and CPU), and serve's
   request log carries the delivered precision, so a run's record states it.
6. A model with no KV cache (a Mamba-only family) reports `KV n/a` and accepts any `--kv` value.

One quantization scheme everywhere: int8 is symmetric per (position, KV head), one f32 scale each for K and V, the scheme
the CPU, Metal and WebGPU int8 caches already use (`decoder/kvcache.go`, `metal/kernels` `attention_i8`, `gpu/decoderunner.go`
`kScale`/`vScale`). The same scheme on every backend is what lets G2 compare backends at equal precision.

## 2. Where it stands today (main `8494a91d`)

"Silent" means the load runs another precision and nothing says so.

| Backend, path | f32 | f16 | i8 |
|---|---|---|---|
| **CPU** decode and batched prefill | all | dense families on the batched attention kernel, no MoE (`Architecture.kvF16OK`); others stay f32, banner names why | same set (`kvInt8OK`); others stay f32, **silent** (no i8 decline reason exists) |
| **CPU** batched decode (several sequences) | yes | declined at `cpuBatchCacheEligible`, falls back per sequence | same |
| **CUDA** resident decode and prefill | all | llama, mistral, qwen2, qwen3, phi3, dense, no sliding window, pairwise RoPE, m-RoPE, hybrid, MLA, MoE or paged experts (`cuda/kvpolicy.go`); others f32, banner names why | **none: silent f32** (`ResidentKVPrecision`'s `cuda` branch ignores i8; the `--kv` help says it works) |
| **Metal** resident decode | **silent f16** (`metal/model.go` sets `r.kvF32 = false`; the f32 kernels are compiled out) | all resident families | decode attention exists (`attention_i8`) |
| **Metal** batched prefill, batched decode, layer-major paths | as above | yes | declined: the MMA prefill writes f16 K/V (`metalResident.PrefillPath` in `metal/backend.go`, `TestPrefill_declinesInt8KV`), batched decode (`resident.batchIneligible` and `canUseAttnFAAt` in `metal/batch.go`), Gemma 4, MoE and emodel layer-major paths; each falls to the per-token path, so i8 is delivered but prefill is slow |
| **WebGPU** resident | yes | decode kernels exist (`gpu/decoderunner.go` `kvF16`) | decode kernels exist with scales; the faster "keys" attention declines both (`attnKeysEligible` in `gpu/attention.go`, speed only); family and prefill coverage **unsurveyed** |
| Speculative verify | tested on Metal only (option grid) | as the target's cache | CPU tested (`TestOptionPath_specVerify`) |
| Session reuse, KV7 snapshots | yes | yes; precision is part of the share root (`resident_reuse.go`) | CPU snapshot tested (`TestKVI8_snapshotRoundtrip`); CUDA none |
| MLA latent cache (DeepSeek, Kimi) | yes | none (CUDA declines, named) | none |

Also wrong today: the `--kv` help text (`kvHelp` in `internal/loadflags/loadflags.go`) claims i8 for "the GPU residency cache", and
B39's "State today" line says i8 is honored on Metal and WebGPU without the prefill and family declines above.

**aikit** loads no goinfer model and has no KV setting (`go.mod` has no goinfer dependency), so it has nothing to build.
**tack** starts `goinfer-serve` with its own arguments (tack's `startPilotServer`) and passes `--kv` through when it sets one.

## 3. Phases

Owners: CUDA work is nobara's; Metal, CPU and WebGPU are the Mac's. The two tracks run in parallel. Each phase lands as
its own PR on the product-code rule (the owner's yes), and each removes refusals from §2's table; none adds one.

### Phase 0 — no silent precision, anywhere (S: a day, Mac; tooling and a small product change)

- One resolver, `decoder.ResolveKV(backend, arch, opts) (delivered string, reason string)`, used by every backend's build,
  the fit guard's pricing, the banner and the request log. Today the answer is spread over `kvF16For`, `CPUKVPrecision`,
  `ResidentKVPrecision` and `kvBytesForCtx`, and they can disagree.
- An explicit `f32`, `f16` or `i8` the resolver cannot deliver **refuses the load** with the reason (owner decision D1).
  Everything §2 marks silent or declined becomes a named refusal until its phase lands.
- Fix the help text and B39's state line. Metal `--kv f32` refuses until phase 3 restores it.
- **The ratchet:** `TestKVPrecisionDelivered`, a table over backend × family fixture × path × requested precision,
  asserting delivered == requested or a named refusal, plus a generated `docs/kv-precision-matrix.md` (the pattern of
  `docs/option-state-grid.md` and `TestOptionStateGrid_fresh`). Each later phase turns refusals in that table into
  deliveries; a delivery may never turn back into a refusal without the owner's word.
- Add CUDA's columns to the option grid's `KVPrecision` row (it lists Metal only today).

### Phase 1 — CUDA f16 for every family (M–L: about a week, nobara)

The `KV_T` and `kvtype.cuh` machinery from the f16 build already templates the kernels on the element type. Widen, one
family group at a time, each with its own gate pair (G1 identity, G4 quality): MoE and the paged-experts build;
sliding-window layers and windowed KV's ring and rollback; pairwise RoPE (Cohere); m-RoPE and the image prefill kernels;
the hybrids (Qwen3.5's attention layers, Gemma 4, Laguna, Granite, LFM2); then MLA's latent cache, last, because it is a
different cache and its range needs G5 first.

### Phase 2 — CUDA int8 (L–XL: about two weeks, nobara; this is B39)

- **Make `KV_T` three-way (f32, f16, i8) in one place** rather than writing `_kv8` twins kernel by kernel: the store
  kernel quantizes per (position, head) and writes the two scales; every reader (`attn_fused`, `decode_fa`,
  `decode_splitkv`, the batched, exact and KV3 row-batched prefill attention, the rope and KV-store kernels, the image
  paths) dequantizes on load. The fused and exact prefill attention and the KV3 row-batched kernels are about half the
  work; without them a 16k prompt would prefill per token (minutes).
- Go side: scale buffers, `cudaKVBytes`, the fit plan, the banner, windowed KV and its rollback, KV7 snapshots and
  prefix reuse with the scales stored beside their rows (the state grid's rule), and the attention-digest census entries.
- v1 covers the dense families first; each family phase 1 widened then gets i8 as a second arm of the same gate, so the
  widening is paid once.
- Expected effect on the 8 GB card (arithmetic in B39): Qwen3-8B Q4_K_M from about 20k positions at f16 to about 39k.

### Phase 3 — Metal: int8 on every path, and f32 back (M–L: about a week, Mac)

- int8 in the MMA prefill: the prefill's K/V write quantizes on store (the same store kernel shape as decode's), so
  `TestPrefill_declinesInt8KV` becomes a delivery test; then int8 in batched decode (`metal/batch.go`) and the Gemma 4,
  MoE and emodel layer-major paths.
- f32 KV: rebuild the f32 kernels that were compiled out, behind `--kv f32` only, so Metal has its reference precision
  again. `auto` on Metal stays f16.

### Phase 4 — CPU: f16 and int8 for every family and path (M: 3 to 5 days, Mac)

- MoE: the acc64 attention kernel reads f16 and int8. `kvInt8OK` excludes MoE because a quantized cache "would reopen"
  expert-routing stability; G4 measures that rather than assuming it, and an explicit request is honored either way (§1.4).
- The own-loop families (Gemma 4, Qwen3.5 and the other hybrids) read f16 and int8 through their own attention.
- CPU batched decode: `cpuBatchCacheEligible` admits f16 and int8 caches.

### Phase 5 — WebGPU (M, Mac; survey first)

Survey with phase 0's table, then fill the family and prefill gaps the survey finds. Lowest priority of the backends:
owner decision D3 can move it out.

### Phase 6 — cross-cutting paths (M, both boxes)

- Speculative decode and drafters: the drafter's cache follows the target's request unless a drafter KV flag says otherwise;
  verify runs at the target's precision.
- Session save and restore on disk: the precision and the scales are in the format, and a restore at another precision
  refuses by name (the share root already separates precisions).
- The option-state grid: every `KVPrecision` and `KVQuant` cell reads tested; none reads declined.

### P0a′ gates, registered 2026-10-22 before the code (P0A-G)

The build: Metal's resident build (`metal/model.go`, `r.kvI8` and `r.kvF32`) and WebGPU's (`gpu/residency.go`, `kvF16` and
`kvI8`) take the cache precision from `decoder.ResolveKV` for their backend (a new `Model.KVDeliveredFor`), not from the raw
request (`KVCacheF16`, `KVCacheI8`). For every request either backend honours today the two agree, so nothing a user sees
changes; the point is that phase 3 (Metal f32 back) and any later resolver change reach the build through one function. A
test-only hook (`goinfer_testhooks`) lets a test make the resolver disagree with the request.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| P0A-G1 plumbing | 0, Mac | the resolver hooked to deliver a precision the request does not ask for (`auto` → i8 on Metal; `auto` → f16 and `auto` → i8 on WebGPU), on `llama-tiny` resident on each backend | each build's cache precision is the hooked one (Metal `r.kvI8`; WebGPU's runner `kvF16`/`kvI8`). Plant, once and reverted: either build reading the raw request again turns it red |
| P0A-G2 unchanged | 0 | unhooked: `TestKVPrecisionDelivered` and the generated `docs/kv-precision-matrix.md` (`TestKVPrecisionMatrix_fresh`), the Metal and WebGPU KV tests the touched packages carry | all PASS, and the matrix is byte-unchanged |

Stopping rule: one pass each. Cost: seconds.

## 4. Gates (registered here, before any code; changing one after the data needs the owner's word and is recorded as such)

- **G1, exact-input identity (every new kernel, CI):** on inputs exactly representable at the precision (f16 values;
  int8 values times a power-of-two scale), the f16 or i8 kernel equals the f32 kernel bit-exactly, as the f16 build's
  `_kv16` identity tests do.
- **G2, cross-backend agreement at equal precision (every path):** the same model and prompt at `--kv i8` (or `f16`) on
  the GPU path against the CPU path at the same precision: logits within the tolerance the backend's f32-vs-f32 parity
  test already uses for that model, top-1 agreement over 64 steps at least the f32-vs-f32 rate less 2 points. This is
  what proves "you got int8", not "you got something".
- **G3, delivered-bytes readback:** the bytes allocated per position equal the precision's arithmetic (f32 4 B, f16 2 B,
  i8 1 B plus 8 B of scales per position, head and layer) for every cell in phase 0's table. Catches a silent fallback
  that the banner might still misreport.
- **G4, quality against f32 (published, not a refusal):** the CUDA f16 gate's instrument and cells
  (`docs/tasks/task-cuda-f16-kv-2026-10.md`: 7B at K=8000, 1.5B at K=3900) plus one long-context cell per family group,
  agreement and mean KL ratio recorded per family and precision. `auto` may move to a precision only on a PASS under that
  doc's bars and the owner's word.
- **G5, f16 range:** the overflow survey from the CUDA f16 gate (max|K| and max|V| under 65504/2 on the gate prompts).
  A family that fails it still gets f16 on request, stored with the same per-(position, head) scale int8 uses, so it
  cannot overflow. The survey decides which layout the family gets, not whether it gets f16.
- **Speed (paired, interleaved A B B A, by day unless over about 2 hours):** a served check per backend at 4k and 16k;
  the bar only forbids a regression against the same backend's f16 at short context. Not a gate on fidelity.

## 5. Sizes and order

| Phase | Size | Box | Can start |
|---|---|---|---|
| 0 no silent precision | S, a day | Mac | now |
| 1 CUDA f16, all families | M–L, ~1 week | nobara | after 0 |
| 2 CUDA int8 (B39) | L–XL, ~2 weeks | nobara | after 1's dense refactor, or in parallel with 1 if `KV_T` goes three-way first |
| 3 Metal int8 everywhere, f32 back | M–L, ~1 week | Mac | after 0 |
| 4 CPU all families and paths | M, 3–5 days | Mac | after 3 |
| 5 WebGPU | M, survey first | Mac | after 4 |
| 6 cross-cutting | M | both | as each backend lands |

Roughly three to four weeks of agent work on each box, run in parallel, plus gate nights. Estimates, not commitments;
phase 0's table gives the first real count of cells to move.

## 6. For the owner

**Owner, 2026-10-20, before any code: D1 refuse the load; D2 honor explicit requests whatever G4 reads; D3 WebGPU stays in
scope; D4 the recommended order (phase 0 now, CUDA int8 before the CUDA family widening).**


- **D1:** an explicit precision that cannot be delivered **refuses the load** (recommended), or loads at f32 with a warning
  on stderr and in the banner. Refusing is the contract; a warning is easy to miss in a script.
- **D2:** explicit requests are honored whatever G4 reads, with G4 published and only `auto` held to it (recommended,
  §1.4), or a family that fails G4 is refused that precision.
- **D3:** WebGPU in scope (phase 5 as written), or out of scope with a named refusal.
- **D4:** order. Recommended: phase 0 now; CUDA int8 (2) before the CUDA family widening (1) if the 8 GB card's context is
  the pull, since tack's Qwen3-8B is already a cleared dense family.

<!-- doc-reviewed: 2026-10-21 -->
