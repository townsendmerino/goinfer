# Task: MLA residency on the cgo-free CUDA backend

> Scoping doc. Opened 2026-08-09 out of the cap-bump leg (`0018114`), which established that
> K2-class models decline CUDA residency on **`FeatMLA`, not capacity** — and pinned that in
> `TestResidentMoECapacity_routerCap` so it cannot be mistaken for done. **No code in this leg.**
>
> **Headline: the port is probably much cheaper than the WebGPU reference implies, and the payoff is
> much thinner than "unblock Kimi-K2" implies.** Both of those cut against the obvious read, and both
> are the reason to scope before building.

## 1. What `FeatMLA` gates, precisely

| site | what it does |
|---|---|
| `decoder/features.go:48` | `FeatMLA ResidentFeature = "mla"` — the declaration |
| `decoder/features.go:126` | `add(a.mla != nil, FeatMLA)` — an arch *requires* it iff it has `mlaParams` |
| `decoder/features.go:330` | `"webgpu": { FeatMLA: true }` — **the only backend that declares it** |
| cuda / metal feature maps | `FeatMLA` absent ⇒ `missingFeatures(...)` non-empty ⇒ `ResidentEligible` false |
| `cuda/backend.go:89` | the runtime twin: `MissingResidentFeatures(...)` → `declined(...)` |

So lifting the decline is exactly: **implement the ops, then add one map entry.** The gate is honest
and there is no second hidden guard — the capacity cap is now 512 and K2's 384 already passes it.

### Op inventory — what an MLA decode step needs vs what CUDA already has

`mlaParams` (`decoder/arch.go:173-182`): `QLoRARank`, `KVLoRARank` (= *rank*), `QKNopeHeadDim`,
`QKRopeHeadDim`, `VHeadDim`; `qkHeadDim() = QKNope + QKRope`.

| op | shape | CUDA today |
|---|---|---|
| `q_a_proj` → `q_a_layernorm` → `q_b_proj` (q-LoRA) | GEMV + RMSNorm | ✅ **reuse** — W8A8/W4A8 GEMV + `rmsnorm` |
| `kv_a_proj_with_mqa` → `[rank ‖ qkRope]` | GEMV, output `rank+qkRope` | ✅ **reuse** (GEMV) |
| `kv_a_layernorm` over the `rank` prefix only | partial-width RMSNorm | ⚠ **narrow variant** — existing norm is full-width |
| decoupled RoPE on the `qkRope` tail (q per head, k once) | rope on a sub-slice | ⚠ **variant of `rope_kv`** — it ropes a *tail slice*, one shared K |
| **latent-cache write** | one row `[rank ‖ qkRope]` per position | ⚠ **new store** (trivial; it is a strided copy) |
| **W_UK absorb**: `qNope_h × W_UK_h → rank` | per-head matvec, nH× | ✅ **reuse** — this is a batched GEMV |
| **decode attention over the latent** | score width `rank+qkRope`, **value width `rank`**, one KV row shared by all heads | ❗ **the genuinely new shape — see §2** |
| **W_UV lift**: `wsum_h × W_UV_h → vHead` | per-head matvec, nH× | ✅ **reuse** |
| `o_proj` | GEMV | ✅ **reuse** |

**Everything except the attention (and three small variants) is an existing GEMV.** The strategy
question — absorbed vs materialized — is settled by the CPU path already: `decoder`'s
`mlaAttentionAbsorb` (`forward_deepseek.go:188`) is the absorb path, and WebGPU mirrors it. Building
CUDA on **materialize-per-token K/V** would mean expanding `kv_b_proj` every step for every head
(nH × (qkNope+vHead) floats per position) — strictly more work *and* it discards MLA's whole point.
**Absorb is the only sane choice; it is not really a fork.**

## 2. The WebGPU reference — and where it must NOT be ported on trust

`gpu/mla.go` adds four kernels, dispatched at `gpu/decoderunner.go:776-804`:

1. `mlaStore` — kv-down + narrow norm + rope → `lat[j] = cn_j[rank] ‖ krj_j[qkRope]`
2. `mlaHeadMV` — per-head matvec, used **twice**: `q × W_UK → qAbs`, and `wsum × W_UV → ctx`
3. `mlaQRope` — rope into `qAbs`'s tail
4. `mlaAttn` — `qAbs[h] · lat[j]` over `latDim`, value = `lat[j][:rank]`, one workgroup per head

Cache layout: `latDim = rank + qkRope`; **`qAbs` has the same width as a cache row**, and the value is
the row's own `rank`-wide prefix. That symmetry is the useful part of the design and it should carry
over.

> ### ⚠ FLAG — the one thing that must not be ported as-is
> **`mlaAttn` uses online (FlashAttention-style) softmax** (`gpu/mla.go:26-30`: *"online
> (FlashAttention-style) softmax over keys so no per-key score row is materialized"*), and it is
> gated on **tolerances** (`TestMLAAttn_parity`, `TestMLALatentStore_parity`: `cos ≥ 0.9999`,
> `maxAbs ≤ 1e-3`), not bit-identity.
>
> **CUDA's decode attention holds the opposite contract.** Campaign A explicitly rejected exactly
> this technique: *"We want the same occupancy — **without** FA's online rescale, which is not
> bit-exact"* (`docs/task-decode-splitkv-attention.md:17`), and both CUDA decode-attention kernels
> are gated **byte-identical** (`TestAttnBatched_bitIdentical`, `TestSplitKV_bitIdentical`).
>
> Porting `mlaAttn` verbatim would make MLA the **first tolerance-gated decode attention on CUDA** and
> silently retire a contract the backend has defended twice. That is a decision to take deliberately,
> not to inherit. WebGPU is not wrong — its backend never held the byte-identical contract — but this
> is precisely the "no diagnosis transfers" case.

### The cheaper path the reference obscures: reuse `attn_batched`

MLA's value is the **prefix of its own key row**, and its single latent row is shared by every query
head. That is structurally `nKV = 1` with `hd = latDim`, which `attn_batched` already expresses:

```
attn_batched(q, kc, vc, nH, nKV, hd, startPos, scale, window, M, ctx)   // cuda/prefill_batched.cu:145
    kvDim = nKV*hd;  group = nH/nKV;  kvh = h/group;     // nKV=1 ⇒ every head reads the same row ✅
    qDim  = nH*hd;                                       // ⇒ q laid out nH×latDim == qAbs ✅
```

Set `kc = vc = latentCache`, `nKV = 1`, `hd = latDim`, `scale = 1/√(qkNope+qkRope)`, `window = 0`.
The score pass is then **exactly right**, and the value pass computes a `latDim`-wide weighted sum
whose **first `rank` dims are exactly `wsum`** — the extra `qkRope` dims are computed and discarded
(~11% wasted value-accumulate at `rank=512, qkRope=64`). The `W_UV` lift then reads a strided prefix,
which is a host-side stride, not a kernel change.

**If that holds, MLA decode attention on CUDA needs no new attention kernel and inherits the existing
byte-identical gate** — collapsing the expensive half of this task.

> ### ✅ TESTED AND IT HOLDS (2026-08-09, micro-leg B) — `cuda/mla_latent_reuse_prototype_test.go`
>
> Test-only prototype at `testdata/deepseek-tiny`'s real MLA geometry (nH=4, `kv_lora_rank` 16,
> `qk_rope_head_dim` 8 ⇒ latDim 24), driving `attn_batched` at `nKV=1, hd=latDim, M=1` against an
> independently-written f64 CPU reference of the absorb-path math. The fixture cannot be loaded
> *resident* here — CUDA declines MLA on features, which is the task — so this drives the kernel at
> the fixture's dims rather than through a model. `kc`/`vc` are two **duplicated allocations**, not
> one aliased buffer (the `restrict` fix is production work, not a prerequisite for the answer).
>
> | criterion | result |
> |---|---|
> | (1)+(2) rank-prefix == absorb-path rank collapse | **max\|Δ\| = 3.18e-08, max relΔ = 2.77e-06** ✅ |
> | (2b) rope-tail accumulate is inert — *verified, not assumed* | zeroing the value buffer's rope tail leaves the first `rank` dims **bit-identical** ✅ |
> | (3) cost of the wasted dims | **1.127×** (hd=24 → 5692 ns vs hd=16 → 5049 ns), order-alternated |
>
> **⚠ Read the 1.127× in context — it is a worst case, not the production number.** The tiny
> fixture's rope tail is 8/24 = **33.3%** of the value width; V2-Lite/V3 geometry is 64/576 =
> **11.1%**, i.e. **3.0× less waste**. The measured overhead is at triple the production waste
> fraction, so real-geometry overhead should land well under it. (Not extrapolated to a number here —
> the honest statement is a bound, and the real geometry is one prototype run away.)
>
> **VERDICT: the reuse holds. The ~3-day path in §6 is the live one; the bespoke-kernel row does not
> activate.** MLA decode attention on CUDA needs no new attention kernel and inherits
> `TestAttnBatched_bitIdentical`'s byte-identical contract — so the §2 flag (WebGPU's tolerance-gated
> online softmax) is not merely avoided, it is *unnecessary*.
>
> The prototype is kept as the parity gate's skeleton: its CPU reference and per-dim-independence
> check are what a production resident-vs-CPU gate needs.

> **Blocker to check first, and it is real.** `attn_batched` declares `const float* __restrict__ kc`
> and `const float* __restrict__ vc`. Passing the same buffer to both **violates `restrict`** — both
> are read-only so it is benign in practice, but it is language-level UB and the compiler may key
> optimizations on the non-aliasing promise. The repo has already met this exact question: Gemma-4's
> K=V case was resolved by **not** aliasing the caches. The clean fix is a small
> `attn_batched_shared_kv` variant with the `restrict` dropped on the value pointer, in a **new file**
> (per the `router_f32.cu` policy — a new kernel goes in a new file, so the audited `moe.ptx`/`glue`
> artifacts stay untouched). That is a ~20-line kernel, not a new attention design.

**Do not assume `splitkv` transfers.** Its 3-kernel split materializes an `nH × nWin` score array in
global memory; at MLA's geometry (`nH = 128` on V3-class) that is a much larger array than the GQA
case, and P6a already showed split-KV is a *loss* on high-`nH` geometries (phi3, `nH=32`, never
crossed over). Assume `attn_batched` only, and re-characterize before enabling split-KV for MLA.

## 3. KV-cache interaction — and a correction to the premise

The latent cache is **one row of `rank + qkRope` floats per position per layer**, versus materializing
`nH×(qkNope+qkRope)` K plus `nH×vHead` V. Measured against real configs (fetched, not assumed):

| model | L | nH | latent KB/pos | materialized KB/pos | ratio | latent @32k |
|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | 27 | 16 | **60.8** | 540.0 | 8.9× | 1.90 GB |
| DeepSeek-V3 / Kimi-K2 | 61 | 128 | **137.2** | 9760.0 | **71.1×** | 4.29 GB |
| Kimi-K3 (24 MLA layers) | 24 | 96 | **54.0** | 2880.0 | 53.3× | 1.69 GB |

> **⚠ The "latent cache is small" premise needs qualifying.** It is 9–71× smaller than *materializing
> MLA's own K/V* — that is the win, and it is large. But **per position it is not small in absolute
> terms**: V2-Lite's 60.8 KB/pos is *larger* than qwen2.5-coder-1.5b's 56.0 KB/pos GQA cache, because
> MLA models carry more layers and a 576-float latent each. The capacity story improves dramatically
> **relative to running MLA the naive way**, not relative to small GQA models. Deep context on an 8 GB
> card is still a budget, not headroom.

**Concrete change needed in the cap arithmetic (`ca29d6c`).** `kvBytesForCap` computes
`Σ layers[i].kvDim × 2 (K+V) × 4 (f32)`. MLA caches **one** row, not a K and a V — so an MLA layer
reporting `kvDim = latDim` would be **over-counted 2×**, and the load-time fail-fast would refuse
configurations that actually fit. Fixing this is small but must not be skipped: the whole point of
that check is that it names a real number.

## 4. Parity plan

The **CPU MLA path is the oracle** — V2/V3/K2 all run there, gated by
`decoder/forward_deepseek.go` with `deepseek_v2` / `deepseek_v3` validated in the parity manifest.

- **T1 — tiny, and the fixture already exists.** `testdata/deepseek-tiny/` ships `config.json` **and
  `model.safetensors`** (4 layers, nH=4, `kv_lora_rank=16`, `qk_nope=16`, `qk_rope=8`, `v_head=16`,
  8 routed experts). A resident-vs-CPU gate needs **no new fixture** — mirror
  `gpu/mla_resident_test.go` (`TestMLAResidency_matchesCPU`). ⚠ `testdata/kimi-tiny/` is **config
  only, no weights** — do not plan a K2 tiny gate without generating one.
- **Per-kernel gates** mirroring WebGPU's: latent-store parity, head-matvec parity, attention parity.
- **Bit-identity gate — the contract decision from §2 made explicit.** If the `attn_batched` reuse
  works, MLA decode attention inherits `TestAttnBatched_bitIdentical` and should get its own
  MLA-shaped byte-identical assertion. If a bespoke kernel is written instead, the doc must state
  which contract it holds *before* it is written.
- **Real-model gate:** `deepseek_v2lite_golden.json` and `deepseek_moonlight_golden.json` exist as CPU
  references; a resident run against either is the T3 target **if the weights fit the box** (§5).
- **Explicitly OUT OF SCOPE:** batched prefill and split-KV for MLA. **Decode residency first;
  prefill stays staged**, consistent with the C1 table. Adding prefill would drag in the query-tiling
  work and a second bit-identity argument.

## 5. Payoff honesty — who actually benefits

This is the part that should decide the queue position, and it is **thin near-term**.

| model | params | fits the 8 GB dev box? | benefits? |
|---|---|---|---|
| `deepseek-tiny` fixture | toy | yes | tests only — no user-visible payoff |
| **DeepSeek-V2-Lite** (15.7B-A2.4B) | 15.7B | **int4 ≈ 7.9 GB — borderline alone; feasible via the shipped C′ host→VRAM expert streaming** (which already runs a 26B-A4B on this card) | ✅ **the one real near-term beneficiary** |
| Moonlight-16B (V3-shaped) | 16B | same class as V2-Lite | ✅ same |
| DeepSeek-V3 / Kimi-K2 | 671B / 1T | **no, and not with any residency work** | ❌ — residency is not their blocker; size is |
| DeepSeek-V4 (Pro/Flash) | — | — | ❌ **not MLA at all** (Phase 0, `e907058`): `kv_lora_rank` absent, sparse attention |
| Kimi-K3 | 2.8T | no | ⚠ its 24 MLA layers would ride this — **but the family is unbuilt and 69/93 layers are KDA** |

**The honest summary: MLA-on-CUDA buys V2-Lite-class checkpoints and essentially nothing else today.**
The trillion-scale models everyone associates with MLA do not fit regardless — for them residency was
never the constraint. K3 is the one future consumer that would make this leverage, and K3 is itself
gated behind fp8/mxfp4 loading and a KDA primitive (`task-model-family-deepseek-v4-kimi-k3.md`).

Counterweight, and it is not nothing: **CUDA and Metal are the only backends without MLA**, so every
MLA model on the two native GPU backends silently runs the whole forward on CPU. That is a coverage
hole, and this is the only task that closes it.

## 6. Effort class and recommended queue position

**Effort: days, not weeks — and the condition is DISCHARGED.** Micro-leg B tested the
`attn_batched` reuse and it holds (see §2), so the bespoke-kernel branch is closed.

| item | class |
|---|---|
| narrow-width RMSNorm + tail RoPE variants + latent store | ~1 day (small kernels, new file) |
| `attn_batched_shared_kv` (drop `restrict` on the value pointer) | ~½ day **if the reuse holds** |
| … or a bespoke MLA attention with a stated softmax contract | **+1–2 weeks** if it does not |
| wiring in `cuda/backend.go` + `resident.go`, feature-map entry | ~1 day |
| `kvBytesForCap` MLA branch (§3) | hours |
| parity gates (fixture exists) | ~1 day |

**Recommended queue position: AFTER the v1.0 cut, and behind the other CUDA depth work.**

1. ~~**The gate to run first**~~ — **DONE (micro-leg B).** The prototype ran; the reuse holds; this
   is the ~3-day task, not the ~3-week one. `cuda/mla_latent_reuse_prototype_test.go` is checked in
   and doubles as the parity gate's skeleton.
2. **Do not schedule the full build on K2/V3's account** — they do not fit, and the doc should stop
   implying otherwise.
3. **Reconsider immediately if K3 is funded**, or if a mid-size MLA checkpoint lands that the C′
   streaming path can host: that flips the payoff column from one model to a family.
4. Ahead of it in the queue: anything with a measured user-visible deficit. P6a/§B7 showed CUDA decode
   is **~25× the peer's per-position cost at depth** — that is a real, measured, every-model problem,
   where MLA residency is a coverage fix for one borderline model.

## Not in scope (recorded, so it isn't re-litigated)

- **MLA batched prefill / split-KV for MLA** — decode first; see §4.
- **Metal MLA** — same feature hole, its own leg, and Metal's shader caps and validation live on the Mac.
- **Making V3/K2 fit** — a weights problem (fp8, streaming), not a residency problem.
