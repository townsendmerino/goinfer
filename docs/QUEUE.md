# QUEUE — the index over four queues

The work list is split by **success criterion**, not by component. An entry lives in exactly one
queue, keeps its original ID, and keeps the section it was filed under.

| queue | holds | the question it answers |
|---|---|---|
| [**performance**](queue-performance.md) | throughput, latency, kernels, residency, memory | *how fast, how much memory* |
| [**correctness**](queue-correctness.md) | parity, numerics, goldens, quantization, families | *does it compute the right thing* |
| [**engineering**](queue-engineering.md) | gates, lints, censuses, tooling, process rules | *would we find out* |
| [**release**](queue-release.md) | release gates, tagging, v1.0 criteria, capability claims | *can we tag* |

**Task docs are not queues.** `docs/task-*.md` are design records — why a thing is built as it is —
and they are cited from **88 code comments**. A queue entry cannot carry that, so they stay where
they are and the queues hold only the open work. Finished ones are archived to `docs/completed/`.

**This file keeps** the cross-cutting material that is not any one queue's: the sweeps, the
sequencing notes, the release draft, and the generated citation indexes below.

# Work queue — the shared, claimable list

> **Why this file exists.** The queue used to live in conversation, where only the top of it gets
> restated each turn and everything below silently sinks. Three items aged out that way — the Metal
> consumer window, the out-of-tree consumer audit, the drain fix's CUDA verification — none through
> carelessness. And two boxes pulling from the same unstated queue independently built two
> mechanisms for running the heavy tier, because neither could see the other's progress.
>
> That makes the conversational queue an instance of the class this fortnight has been cataloguing:
> an artifact that exists and is not composed into any decision. A check that cannot fail.
>
> **This file is the queue.** If it is not written here, it is not queued.

## How to use it

- **Claim before starting.** Move the item to `In flight` and put your box and the date on it.
  A claim is what stops the other box duplicating it.
- **Release on finish** — move to `Done` with the commit, or back to `Queued` with what you learned.
- **Never delete an item to tidy up.** Strike it with a reason, so "we decided not to" is
  distinguishable from "it sank".
- **Add the whole item, not a title.** Enough that whoever picks it up does not have to reconstruct
  the context from a transcript they may not have.

Boxes: `linux` (nvidia-rtx2070s, CUDA) · `mac` (Apple Silicon, Metal).

## In flight

## Queued

Ordered roughly by priority within each group. Each item carries enough context to be picked up
cold. Where something is believed done but unconfirmed, it says so — **verify before striking**.

### A. Open investigation

## Struck — decided against, kept so the decision is visible

- ~~**Default `top_k`**~~ — truncating the distribution changes which tokens are reachable, which is
  a silent substitution of something other than what was asked. Document it; do not default it.
- ~~**Change the global `--quant` default**~~ — CPU inverts the CUDA quant ordering, so a single
  global default cannot be right for both, and the evidence is one model on one box, never
  reproduced at 1.5B.
- ~~**Force cross-architecture float agreement**~~ — explicit `math.FMA` everywhere is a software
  fallback on amd64 that costs the SIMD performance the CPU backend exists for. Scoped in the policy
  instead.
- ~~**Slab restructure for expert slots**~~ — the control produced the reverse of fragmentation's
  prediction: a fresh heap with ~10 large allocations had *worse* contiguity (32–64 MiB) than the
  slot-loaded heap (96–128 MiB) at the same free figure.
- ~~**aikit branch protection**~~ — required checks force PR-only merges, which is friction against
  a threat model aikit doesn't have. The gate ritual is the enforcement. Revisit at v1.0.
- ~~**Metal `ResidentGreedy` gap**~~ — measured **net-negative**. Kept here rather than under group
  P because it is not work. The 2026-08-12 audit reached the same conclusion **independently**, from
  code, without access to the measurement — recorded as a corroboration of that audit's calibration,
  which is the only reason the entry is worth keeping at all.

## Done

_(append with commit sha and date)_

## Sequencing — release BEFORE G2

**Revised, and D3 is OUT of the release — no rebase attempted.**

> **cut the release → G2 → D3 design read → B1, B2 → mac batch**

The README change in this release is a **retraction**: the workaround language goes away because the
cap holds. D3, if it survives its design read, is an *addition to adjacent text later* — not the same
edit made twice, which was the argument for including it.

A repo-wide mechanical diff immediately before a tag costs bisectability and reasoning room and buys
the modernizers nothing. G2 is not urgent and never was; it is cleared, which is different from being
next.


## Freshness sweep — C, D, E, G (2026-08-12)

F was **fifteen for fifteen already fixed**, because it was seeded from `docs/completed/`. These
groups were seeded from conversation, and **the rate is much lower**, which is the useful result:

| entry | state | evidence |
|---|---|---|
| C1 drain fix — CUDA verification | **open** | no CUDA unload/drain test found |
| C2 out-of-tree consumer audit | **open** | needs a fresh no-repo session by design |
| C3 Metal consumer window | **RUN 2026-08-15 (metal/v0.13.0)** | cgo-free/no-Xcode build ✅ (require path, otool-confirmed); `go install …/metal/cmd/serve@v0.13.0` **BROKEN** (committed `replace`); resident decode not public-API-drivable; tautology has no Metal analog. Full note: `docs/measurements/c3-metal-consumer-window.md` |
| C4 soak testing | **open** | `internal/serveapp/fuzz_test.go` and `internal/serveapp/chaos_test.go` exist; neither is an hours-long soak |
| D1 trace tap + launch-site table | **open** | no coverage table in `docs/` |
| D2 launch-wrapper commit 1 | **open** | no `cuda/internal/gen` |
| D3 parked flag-pair | **open**, design read done above | — |
| E1 v1.0 gate as written criteria | **open** | prose item, no tree anchor |
| E2 four per-family demotions | **open** | manifest still lists `gpt2`, `granitemoehybrid`, `kimi_k2`, `nemotron_h` as `pending` |
| E3 freeze re-declaration | **DONE `cda8cfe`** | re-declared as a proof requirement, with decider and date |
| E4 `scripts/bench_compare.sh` fix or retire | **FIXED** | it now opens with *"goinfer's OWN numbers only. NOT a peer comparison"* and points at `scripts/bench_peer.py`, which drives both sides |
| E5 promo drafts | **unverifiable** | held in conversation, nothing in the tree to check |
| E6 aikit release | **CLOSED 2026-08-12** — superseded by events, not by reversal | aikit cut `v1.17.0`/`gpu/v0.28.0` (`ada417e`); goinfer is on it (`f33fcaf`). The release met E6's own "a reason a consumer can receive" test |
| G1 LFM2.5 family | **open** | no LFM2 code in the tree |
| G2 `go fix` modernizers | **DONE `3d6ae1e`** | — |

**Rate: 1 of 13 previously-open entries was silently already fixed (E4), against F's 15 of 15.**
Two more (E3, G2) were closed by this campaign and were already recorded.

**That difference is the finding.** F was seeded from a *filed audit* — work done elsewhere, reported
once, never propagated back. C/D/E/G were seeded from *conversation*, where the person who did the
work was the person holding the list. **The burial folder is what produced the 15/15, not the passage
of time.** So the sweep paid for itself once, on the group that came from a document, and should not
be assumed to pay again on groups that did not.

E5 is recorded as **unverifiable** rather than open: nothing in the tree can confirm or deny it, which
is a different state and should read as one.


## Description sweep (2026-08-12) — does each entry match its source?

The status sweep found 1 of 13. **D3 shows description can be wrong while status is right, and
description is what someone acts on.** So: for every open entry with a source outside conversation —
a branch, a commit, an audit line, a script — does the entry describe it correctly?

*(A goldens run in a fresh `git worktree` is fixture-less: the same commit proved 33 goldens in the
main checkout and 7 in the worktree, because the fixture checkpoints are gitignored. The refresh
script now says so when skips outnumber passes — `scripts/refresh_parity_hashes.sh`. Found by running
D3's refresh in the rebase worktree and getting `goldens=7`.)*

| entry | source | description matches? |
|---|---|---|
| D3 | branch `flag-pair-moe-cache` + `BRANCH-NOTE.md` | **NO — corrected `2d28358`.** Called a "parked flag-pair" on a workaround premise; it is an API-surface promotion following `KVPrecision` |
| B4 | a stash that does not exist here | **unverifiable** — the description is all that survives, and it names a file that resolves nowhere |
| C1 | `588052b` (the drain fix) | matches — Metal-verified, CUDA arm untested |
| D2 | design recorded in-entry, no branch | matches; no external source to drift from |
| E2 | `testdata/parity_manifest.json` | matches — the four families are still `pending` |
| E4 | `scripts/bench_compare.sh` | **stale** — the entry says "status unconfirmed, may still measure the two sides differently"; the script now refuses that use and points at `scripts/bench_peer.py`. Corrected in the status sweep above |
| E6 | aikit tree + `gpu/v0.27.0` | **now stale by design** — the tag it pinned is superseded by `gpu/v0.28.0`. Re-checked at the bump: `be049df` is an ancestor of `gpu/v0.27.0`, and across `gpu/v0.27.0..gpu/v0.28.0` the quantized GEMV PTX is byte-identical |
| G1 | `docs/scoping-lfm2.md` | matches |
| P1, P2, P4, P5, P8 | audit lines + the cited source | match; each carries a measured figure or an explicit ESTIMATE label |

**Split: 9 entries had an external source and were checkable; 2 of those 9 were wrong (D3, E4).
4 entries — C2, C3, E1, E5 — have no source outside conversation and are recorded as unverifiable
rather than checked.**

**THE TRIGGER, not a cadence.** An entry's description **and the specific details inside it** — counts,
file names, measurements that no lint covers — are re-read against their source **at the moment the
item is picked up for work** — nothing schedules this and nothing lints it. That is when the
description matters, when someone is already loading the context anyway, and it is exactly what caught
D3: the read happened because the item was next, not because a sweep came due. The cost falls at the
only point where the drift would have changed what someone did.

**That rate (2 of 9) is higher than the status sweep's (1 of 13), and the two are not the same
population.** A description drifts silently because nothing re-reads it against its source; a status
drifts only when work lands elsewhere. The queue's SHA and path citations are now linted, but **no
lint reads an entry's prose against the branch note or audit line it describes** — that remains a
person's job, and this sweep is its baseline.

## Sequencing

**D3 (loaded and bounded) → the mac batch as one session → B1, B2.**

Within the mac batch, **C3 goes FIRST**, not last: it is the largest completely uncovered surface and
it sank once already. Batching it behind two chores is precisely how that happened. Then
`metal-rope-merge`'s push, then B4's stash check.

**C3 DONE 2026-08-15** (auto-pickup fired on the `v0.13.0` tag). cgo-free/no-Xcode build verified via
the require path; two real gaps found — `go install …/metal/cmd/serve@v0.13.0` is broken by the
committed `replace github.com/townsendmerino/goinfer => ../` in the tagged `metal/go.mod`, and
resident-metal decode isn't drivable via the public API (needs the serve binary, which `go install`
can't build). Full findings + the resolved dep set in `docs/measurements/c3-metal-consumer-window.md`.
**Follow-up worth queuing:** tag `metal/` from a replace-free tree so `go install` works.

## Draft: contents of the next release

**Not a version number** — that is a separate call. This is what has accumulated since
`demo/agent/v0.11.0` (93 commits) that a user would notice, and **none of it depends on the freeze
decision**.

### The headline: the 26B expert cache sizes itself correctly

The defect that opened this campaign was live in the product and is fixed. On an 8 GB card the
runtime auto-capped the MoE expert cache to **34 slots/layer, which allocates and then cannot
launch** — the forward produced **zero tokens**.

- **A5 (`6091e7a`)** — the cap is a **search over the granularity form**, not a division. The driver
  charges each of four buffers per layer its own whole 2 MiB quantum, so the requirement is a step
  function; at 34 all four tip at once, putting it 203,816,960 B over free. Verified through the
  shipping auto-cap path: `capping to 34` → 0 tokens becomes `capping to 33` → coherent output.
- **A9-FIX (`0103b49`)** — the deferred first-launch reservation (`moe_route` takes 138,412,032 B of
  local memory the first time it runs) is now paid **before** the free reading that sizes the cache,
  so the cap is correct by construction rather than covered by a margin. Costs two slots, and that is
  the point: 384 MiB now means 384 MiB.
- **A3 (`e42e83e`)** — a launch OOM now names the kernel and **both** the requested and effective
  slot counts, instead of a bare `cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY`.
- **README** — the manual-workaround section is retracted and replaced with what the cap now
  accounts for, plus a version test (`capping to 33` has the fix, `34` does not).

### Performance, all bit-identical

- **P3 (`4c26a58`)** — Gemma's final-logit softcap parallelised: **1.43 ms → 640 µs** per sampled
  token at 262,144 vocab. Sampling path only; greedy never paid it.
- **P6 (`eea7f29`)** — MoE experts share one gate/up buffer pair per token instead of one per expert:
  **16 allocations → 2** at top-k 8.
- **P7 (`91f359f`)** — W4A8 reaches the per-stream `Workspace` it was silently excluded from, ending
  a fresh allocation per projection per token.

### Verification a user can check

- **int4 forward goldens** (`1d0d1ed`) — 23 fixtures across 16 architectures. int4 is the documented
  default quantization and **nothing gated it** before this.
- The goldens refresh went from **19 passed / 0 quantized** to **33 passed / 14 quantized**, and now
  prints its composition rather than a bare count.

### Known-unfixed, disclosed

- **A10** — a ~150 MiB driver allocation floor: memory `cuMemGetInfo` reports as free and
  `cuMemAlloc` will not hand out, at any request size down to 1 MiB. Measured, unattributed. It is
  why the margin cannot simply be lowered to recover the two slots.

<!-- sha-lint: allow c8b65ba UNPUSHED — the PRE-REBASE D3 flag-pair commit, on the local-only branch `flag-pair-moe-cache`; never pushed, so CI cannot resolve it (it failed there 2026-08-12). DELIBERATELY NOT re-pointed to its rebased successor `bacc04c` (on main via the D3 merge): the two have DIFFERENT patch-ids, and the passage citing this one is historical — it describes the branch as it stood BEFORE the rebase ("its branch predates the fix it completes"), which `bacc04c` no longer illustrates. Re-pointing would keep the lint green by making the surrounding sentence false. Allowlisted rather than laundered. Flagged 2026-08-12 -->
<!-- sha-lint: allow d682315 UNPUSHED — Metal branch `metal-rope-merge`, mac-local; not on origin and not in any clone here. Owner: whoever cited it. P4's "already implemented, snapshot-golden byte-exact" rests on a commit only that machine can see; push the branch or the claim stays unverifiable from anywhere else. Flagged 2026-08-12 -->

<!-- SHA-INDEX: generated by scripts/queue_citation_lint.py --update; do not edit by hand -->

## SHA index

Generated. Every commit id cited above, with the subject it resolved to at the time
of generation. Regenerate with `scripts/queue_citation_lint.py --update`.

| sha | subject |
|---|---|
| `0103b49` | fix(cuda): pay the deferred reservation before sizing the cache (A9-FIX) |
| `0c54e35` | fix(gate): repo hygiene runs what CI runs, derived from ci.yml (B0) |
| `1d0d1ed` | test(decoder): int4 forward goldens — 23 fixtures, 16 architectures (Q1c) |
| `1f6dbe0` | fix(parity,fmt): gofmt the threshold sweep + refresh deps_hash after comment-only core edits |
| `a6c5b57` | fix(parity): the goldens refresh runs quantized goldens, and reports the split |
| `2e91607` | test: refresh parity deps_hash — non-numeric core-file drift (un-reds main) |
| `3d6ae1e` | chore: go fix modernizers, one deterministic pass (G2) |
| `4c26a58` | perf(cuda): parallelise the Gemma final-logit softcap, bit-identical (P3) |
| `588052b` | serve: drain in-flight requests before freeing an unloaded model (fixes the leak safely) |
| `6091e7a` | fix(cuda): size the expert cache by SEARCH over the granularity form (A5) |
| `6edd1ca` | parity: make "validated" MEAN T3 — method-tier gate + honest experimental tier (D2, pre-freeze) |
| `a15a394` | cuda+docs: decline floor, slot-cap gate, driver allocation facts, and seven rules |
| `7cc2f0d` | fix(parity,ci): refresh deps_hash after 38061b1's pread-staging core plumbing (non-numeric) |
| `7ccec1e` | fix(cuda): the expert cache sizes itself — topK was the worst possible default |
| `82b39cc` | docs(parity): document qwen3_5_moe's int8-vs-bf16 movement (v0.8.0 §1 — gate-backed pass) |
| `8fecfad` | ci: scripts/heavy_gate.sh — a runner for the real-checkpoint tier that no CI job executes |
| `91f359f` | fix(decoder): matmulInto dispatches on the property, not on W8A8 (P7) |
| `93eb7d4` | feat(decoder): gpt-oss real-model path — batched-prefill fix + real gates |
| `9624dd9` | chore(parity): refresh deps_hash for aikit v1.12.0 (goldens-proven non-numeric) |
| `98936cf` | test(goldens): strengthen mamba-2 + deltanet parity fixtures (kill identity weights) |
| `99b3f95` | chore(deps): pin aikit v1.12.0 — gpt-oss MXFP4 reproducible on main |
| `9e5f8fa` | fix(quant): reject --quant that conflicts with a prequant .giw at startup (T1-7) |
| `bd08936` | fix(gate): cannot-search is not not-found; cross-gate composition; B7 sweep |
| `be049df` | [aikit] gpu(gemv): explicit __fmaf_rn in the quantized GEMV — the bit-identity contraction rule |
| `c8b65ba` | feat(serve): --moe-cache-experts / --moe-cache-slots — PARKED on the freeze |
| `ca29d6c` | cuda: resident context cap becomes configuration-derived (-ctx), VRAM-checked at load |
| `cc238c6` | cleanup: consolidate GINFER_ env vars to GOINFER_ + add env-var registry |
| `e42e83e` | fix(cuda): name the kernel and both slot counts when a launch runs out of memory |
| `e58ac8a` | fix(parity): refresh deps_hash after f340d4e's guarded int4-scale seam — non-numeric, validated_at preserved |
| `ecc5af2` | chore(parity): refresh deps_hash after default-off diagnostic hooks (non-numeric) |
| `ed81e13` | P1: route top_k=1 to the on-device greedy fast path |
| `eea7f29` | perf(decoder): one gate/up pair per token in MoE, not one per expert (P6) |
| `bacc04c` | feat(serve): --moe-cache-experts / --moe-cache-slots (decisions 2+3) — HELD, trips the parity manifest |
| `f9d5d07` | feat(decoder): dispatch census (B6); close the GGUF-quant gap; reopen B4 |

<!-- /SHA-INDEX -->

<!-- CITATION-INDEX: generated by scripts/queue_citation_lint.py --update; do not edit by hand -->

## SHA index

Generated. Every commit id cited above, with the subject it resolved to at the time
of generation. Regenerate with `scripts/queue_sha_lint.py --update`.

| sha | subject |
|---|---|
| `0103b49` | fix(cuda): pay the deferred reservation before sizing the cache (A9-FIX) |
| `0c54e35` | fix(gate): repo hygiene runs what CI runs, derived from ci.yml (B0) |
| `1d0d1ed` | test(decoder): int4 forward goldens — 23 fixtures, 16 architectures (Q1c) |
| `1f6dbe0` | fix(parity,fmt): gofmt the threshold sweep + refresh deps_hash after comment-only core edits |
| `2d28358` | docs(branch-note): re-derive against the corrected cap (D3 design read) |
| `2e91607` | test: refresh parity deps_hash — non-numeric core-file drift (un-reds main) |
| `38061b1` | perf(gemma4-paging): pread expert nibbles straight into the slot buffers |
| `3d6ae1e` | chore: go fix modernizers, one deterministic pass (G2) |
| `4c26a58` | perf(cuda): parallelise the Gemma final-logit softcap, bit-identical (P3) |
| `588052b` | serve: drain in-flight requests before freeing an unloaded model (fixes the leak safely) |
| `6091e7a` | fix(cuda): size the expert cache by SEARCH over the granularity form (A5) |
| `6edd1ca` | parity: make "validated" MEAN T3 — method-tier gate + honest experimental tier (D2, pre-freeze) |
| `7cc2f0d` | fix(parity,ci): refresh deps_hash after 38061b1's pread-staging core plumbing (non-numeric) |
| `7ccec1e` | fix(cuda): the expert cache sizes itself — topK was the worst possible default |
| `82b39cc` | docs(parity): document qwen3_5_moe's int8-vs-bf16 movement (v0.8.0 §1 — gate-backed pass) |
| `8fecfad` | ci: heavy_gate.sh — a runner for the real-checkpoint tier that no CI job executes |
| `91f359f` | fix(decoder): matmulInto dispatches on the property, not on W8A8 (P7) |
| `93eb7d4` | feat(decoder): gpt-oss real-model path — batched-prefill fix + real gates |
| `9624dd9` | chore(parity): refresh deps_hash for aikit v1.12.0 (goldens-proven non-numeric) |
| `98936cf` | test(goldens): strengthen mamba-2 + deltanet parity fixtures (kill identity weights) |
| `99b3f95` | chore(deps): pin aikit v1.12.0 — gpt-oss MXFP4 reproducible on main |
| `9e5f8fa` | fix(quant): reject --quant that conflicts with a prequant .giw at startup (T1-7) |
| `a15a394` | cuda+docs: decline floor, slot-cap gate, driver allocation facts, and seven rules |
| `a6c5b57` | fix(parity): the goldens refresh runs quantized goldens, and reports the split |
| `ada417e` | [aikit] scripts: ptx-repro is n/a on darwin, keyed on the PLATFORM not on NVRTC's absence |
| `bacc04c` | feat(serve): --moe-cache-experts / --moe-cache-slots — PARKED on the freeze |
| `bd08936` | fix(gate): cannot-search is not not-found; cross-gate composition; B7 sweep |
| `be049df` | [aikit] gpu(gemv): explicit __fmaf_rn in the quantized GEMV — the bit-identity contraction rule |
| `ca29d6c` | cuda: resident context cap becomes configuration-derived (-ctx), VRAM-checked at load |
| `cc238c6` | cleanup: consolidate GINFER_ env vars to GOINFER_ + add env-var registry |
| `cda8cfe` | docs: re-declare the freeze as a proof requirement; clear G2 for amd64 alone |
| `e42e83e` | fix(cuda): name the kernel and both slot counts when a launch runs out of memory |
| `e58ac8a` | fix(parity): refresh deps_hash after f340d4e's guarded int4-scale seam — non-numeric, validated_at preserved |
| `ecc5af2` | chore(parity): refresh deps_hash after default-off diagnostic hooks (non-numeric) |
| `ed81e13` | P1: route top_k=1 to the on-device greedy fast path |
| `eea7f29` | perf(decoder): one gate/up pair per token in MoE, not one per expert (P6) |
| `f33fcaf` | chore(deps): aikit v1.16.0 -> v1.17.0, aikit/gpu v0.27.0 -> v0.28.0 |
| `f340d4e` | metal(9c Step 4): argmax-primary gate + f16-scale confound diagnostic (finding recorded) |
| `f9d5d07` | feat(decoder): dispatch census (B6); close the GGUF-quant gap; reopen B4 |

## Path index

Generated. Every `file:line` cited above, the repo it resolved in, and the trimmed
content of that line. A line that MOVED is reported with its new number; content that
has VANISHED is red, because the citation then claims something the file no longer
supports.

| doc \| path:line | repo | line content |
|---|---|---|
| `docs/benchmarks.md|cuda/resident.go:28` | goinfer | `anchor: var (` |
| `docs/cuda-megakernel-spec.md|gpu/attention.go:14` | goinfer | `// uses f64 accumulation; the GPU f32 — cosine ~1.0, not bit-exact).` |
| `docs/cuda-megakernel-spec.md|gpu/decoderunner.go:730` | goinfer | `// moeExpert records one indexed sparse-expert GEMV: dst[n] = expert[idx[slot]]·aq` |
| `docs/cuda-megakernel-spec.md|gpu/decoderunner.go:835` | goinfer | `// relu²→int8 → down + residual into xd. The other kinds fall through to the mixer.` |
| `docs/cuda-megakernel-spec.md|gpu/forward_parity_test.go:36` | goinfer | `func TestWebGPU_forwardParity(t *testing.T) {` |
| `docs/cuda-megakernel-spec.md|gpu/gemv.go:41` | goinfer | `@compute @workgroup_size(64)` |
| `docs/gpu-residency-coverage.md|decoder/registry.go:135` | goinfer | `IntermediateDim:   cfg.IntermediateDim,` |
| `docs/how-inference-works.md|decoder/attention.go:104` | goinfer | `if !arch.LearnedPosEmbed && !arch.isNoPELayer(layer) {` |
| `docs/how-inference-works.md|decoder/attention.go:144` | goinfer | `cache.Append(layer, k, v)` |
| `docs/how-inference-works.md|decoder/attention.go:59` | goinfer | `nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim` |
| `docs/how-inference-works.md|decoder/kvcache.go:126` | goinfer | `subCapture bool` |
| `docs/how-inference-works.md|decoder/kvcache.go:20` | goinfer | `func quantizeHeads(src []float32, q []int8, scales []float32, nKV, headDim int) {` |
| `docs/how-inference-works.md|decoder/model.go:545` | goinfer | `anchor: func (m *Model) runLayers(id int, cache *KVCache) ([]float32, error) {` |
| `docs/how-inference-works.md|decoder/model.go:586` | goinfer | `anchor: func (m *Model) runLayersFromEmbed(h []float32, cache *KVCache) ([]float32, erro` |
| `docs/how-inference-works.md|decoder/registry.go:19` | goinfer | `var registry = map[string]archAdapter{` |
| `docs/how-inference-works.md|decoder/sampler.go:109` | goinfer | `// can never silently diverge. They are separate predicates, not one widened one, so tha` |
| `docs/how-inference-works.md|decoder/sampler.go:116` | goinfer | `// though a temperature is set — the `top_k=1` shape. It is TRUE at any temperature, whi` |
| `docs/how-inference-works.md|decoder/sampler.go:118` | goinfer | `// distribution restricted to ONE token is deterministic regardless of that token's prob` |
| `docs/how-inference-works.md|decoder/session.go:71` | goinfer | `// stale history. Callers must skip it (and reconcile) for an empty prompt, so a rejecte` |
| `docs/ideas-weight-memory.md|decoder/mlp.go:69` | goinfer | `anchor: func mlp(h, out []float32, lw *LayerWeights, arch *Architecture, be Backend, scr` |
| `docs/measurements/c3-metal-consumer-window.md|metal/gemma_parity_test.go:84` | goinfer | `t.Fatal("metal resident DECLINED — admission says it should be admitted")` |
| `docs/multimodal.md|decoder/config.go:466` | goinfer | `case c.MoeIntermediateSize <= 0:` |
| `docs/multimodal.md|decoder/gguf_qwen35.go:77` | goinfer | `anchor: func ggufQwen35Config(g *embed.GGUFFile) (*Config, error) {` |
| `docs/multimodal.md|decoder/weights.go:344` | goinfer | `const shardIndexFile = "model.safetensors.index.json"` |
| `docs/ollama-chase.md|cuda/resident.go:1066` | goinfer | `// All of it runs ON the executor thread — that thread made the context current — and th` |
| `docs/ollama-chase.md|cuda/resident.go:340` | goinfer | `g4x1, g4x2, g4rn Buffer` |
| `docs/ollama-chase.md|cuda/resident.go:41` | goinfer | `anchor: const ctxCapMarginBytes = 384 << 20` |
| `docs/ollama-chase.md|cuda/resident.go:583` | goinfer | `// declined to the staged/CPU path upstream.` |
| `docs/ollama-chase.md|decoder/forwardn.go:378` | goinfer | `for kvh := range nKV {` |
| `docs/ollama-chase.md|decoder/mlp.go:82` | goinfer | `func moeMLP(h []float32, lw *LayerWeights, arch *Architecture, be Backend, pager *expert` |
| `docs/ollama-chase.md|decoder/model.go:825` | goinfer | `// logits. On the batched archs this runs the layers at M=len in one pass (each` |
| `docs/ollama-chase.md|decoder/model.go:973` | goinfer | `// sample. Identical to the logits path — guarded by ArgmaxEquivalent/GreedyEquivalent.` |
| `docs/ollama-chase.md|decoder/residency.go:677` | goinfer | `return false, "sequential — this backend has no batched prefill (per-token resident forw` |
| `docs/ollama-chase.md|decoder/weightmat.go:202` | goinfer | `var ws linalg.Workspace` |
| `docs/parity-coverage-policy.md|cuda/resident.go:910` | goinfer | `// always been allocated without one, and a hard failure here would regress every driver` |
| `docs/parity-coverage-policy.md|linalg/dot.go:25` | aikit | `sum += a[k] * b[k]` |
| `docs/plan-cpubrrr-steal-and-bindings.md|decoder/registry.go:46` | goinfer | `"gpt_oss":             gptOssArchitecture,     // gpt-oss (20b/120b): sparse MoE + per-h` |
| `docs/plan-cpubrrr-steal-and-bindings.md|linalg/quant.go:327` | aikit | `for i := range M {` |
| `docs/prompts/metal-close-leak-check.md|metal/backend.go:139` | goinfer | `return nil, fmt.Errorf("metal: batched prefill declined — not bit-identical to decode (5` |
| `docs/prompts/metal-close-leak-check.md|metal/model.go:350` | goinfer | `// expert weights, buffer OOM — model.go/moe.go/gemma4_moe.go) into the error this signa` |
| `docs/prompts/metal-close-leak-check.md|metal/model.go:46` | goinfer | `// same-op kernel inherit it. A byte-exact fixture for such an op MUST use context > the` |
| `docs/queue-engineering.md|cuda/argmax_tiebreak_test.go:19` | goinfer | `func TestArgmaxTieBreak(t *testing.T) {` |
| `docs/queue-engineering.md|cuda/backend.go:836` | goinfer | `// cache, so the cap is correct by construction rather than covered by a margin.` |
| `docs/queue-engineering.md|cuda/prefill.go:200` | goinfer | `defer func() {` |
| `docs/queue-engineering.md|cuda/resident.go:244` | goinfer | `// backend.go locals; the per-layer KV cache and UploadKV read r.layers[l].kvDim.` |
| `docs/queue-engineering.md|cuda/resident.go:397` | goinfer | `func (r *cudaResident) recordUpload(e error) {` |
| `docs/queue-engineering.md|decoder/features_test.go:146` | goinfer | `want, ok := admissionGolden[name]` |
| `docs/queue-engineering.md|decoder/forwardn.go:502` | goinfer | `logits[j] = sc * float32(math.Tanh(float64(val/sc)))` |
| `docs/queue-engineering.md|decoder/kvsnapshot_gemma4_test.go:10` | goinfer | `func TestSnapshot_refusesNonUniformKVWidth_C05(t *testing.T) {` |
| `docs/queue-engineering.md|decoder/layerpaging.go:42` | goinfer | `// mu guards the mutable paging state below (audit C-30). The pager lives on *Model, sha` |
| `docs/queue-engineering.md|decoder/model.go:731` | goinfer | `anchor: func (m *Model) ForwardSubCapture(id int, cache *KVCache) (attn, mlp, ctx, mlpPr` |
| `docs/queue-engineering.md|decoder/modelsdir_test.go:13` | goinfer | `root := os.Getenv("GOINFER_MODELS_DIR")` |
| `docs/queue-engineering.md|decoder/serialize.go:594` | goinfer | `// KV-snapshot fingerprint, accounting for MIXED bundles. It scans the BODY matmuls — th` |
| `docs/queue-engineering.md|decoder/serialize_shapecheck_test.go:15` | goinfer | `func TestValidateShapes_catchesArchMismatch(t *testing.T) {` |
| `docs/queue-engineering.md|decoder/serialize_test.go:436` | goinfer | `t.Fatalf("streamed length %d != buffered %d", n, len(want))` |
| `docs/queue-engineering.md|internal/giw/bundle.go:114` | goinfer | `if avail := fi.Size() - (tokOff + 4); tokLen > avail {` |
| `docs/queue-engineering.md|internal/serveapp/embeddings.go:26` | goinfer | `// Embedding request bounds (audit C-21). /v1/embeddings is deliberately un-queued (the ` |
| `docs/queue-engineering.md|internal/serveapp/main.go:432` | goinfer | `// write, so a long stream is unaffected. WriteTimeout stays 0: SSE responses are long-l` |
| `docs/queue-engineering.md|linalg/quant.go:113` | aikit | `for k := range K {` |
| `docs/queue-engineering.md|metal/model.go:728` | goinfer | `r.residencyBufs = pinned` |
| `docs/queue-engineering.md|metal/model.go:827` | goinfer | `r.logitsHost[j] = sc * float32(math.Tanh(float64(v/sc)))` |
| `docs/queue-engineering.md|metal/snapshot_golden_test.go:77` | goinfer | `func TestMetalEmbedScale_forwardMatchesForwardEmb(t *testing.T) {` |
| `docs/queue-performance.md|cuda/backend.go:463` | goinfer | `if r.dev, e = CreateSystemDefaultDevice(); e != nil {` |
| `docs/queue-performance.md|cuda/backend.go:591` | goinfer | `load(&r.bRopeKV, pbmod, "rope_kv_batched")` |
| `docs/queue-performance.md|cuda/backend.go:836` | goinfer | `// cache, so the cap is correct by construction rather than covered by a margin.` |
| `docs/queue-performance.md|decoder/forwardn.go:378` | goinfer | `for kvh := range nKV {` |
| `docs/queue-performance.md|decoder/mlp.go:82` | goinfer | `func moeMLP(h []float32, lw *LayerWeights, arch *Architecture, be Backend, pager *expert` |
| `docs/queue-performance.md|decoder/sampler_chunked.go:188` | goinfer | `return drawChunked(e, sums, z, r)` |
| `docs/queue-performance.md|decoder/scratch.go:38` | goinfer | `ws        *linalg.Workspace // W8A8 activation-quant scratch (zero-alloc Into/Batch)` |
| `docs/queue-performance.md|linalg/quant.go:113` | aikit | `for k := range K {` |
| `docs/scoping-lfm2.md|decoder/arch.go:156` | goinfer | `type nemotronParams struct {` |
| `docs/scoping-lfm2.md|decoder/attention.go:94` | goinfer | `if arch.QKNorm {` |
| `docs/scoping-lfm2.md|decoder/config.go:627` | goinfer | `case c.UseQKNorm:` |
| `docs/scoping-lfm2.md|decoder/deltanet.go:99` | goinfer | `// 1. Projection + depthwise causal conv (+ SiLU). Taps t-K+1..t: the last K-1` |
| `docs/scoping-lfm2.md|decoder/forward_qwen35.go:30` | goinfer | `if arch.isLinearLayer(l) {` |
| `docs/scoping-lfm2.md|decoder/kvcache.go:50` | goinfer | `type KVCache struct {` |
| `docs/scoping-lfm2.md|decoder/mamba2.go:89` | goinfer | `// 2. Depthwise causal conv over xBC (+ bias, + SiLU). Taps t-K+1..t: the last` |
| `docs/scoping-lfm2.md|decoder/mamba2_chunked.go:60` | goinfer | `// Depthwise causal conv over xBC (+bias, +SiLU), then split into x/B/C.` |
| `docs/scoping-lfm2.md|decoder/rmsnorm.go:49` | goinfer | `func layerNorm(x, weight, bias []float32, rows, dim int, eps float64) {` |
| `docs/task-gpu-batched-prefill.md|decoder/residency.go:54` | goinfer | `// ResidentGreedy is an optional capability on a ResidentForward: compute the token's gr` |
| `docs/task-moe-streaming.md|decoder/forwardn.go:14` | goinfer | `// MoE FFN itself stays per-row (router picks different experts per token).` |
| `docs/task-moe-streaming.md|decoder/forwardn.go:228` | goinfer | `// Sequential: add the attention residual, then re-norm the updated stream for the MLP.` |
| `docs/task-moe-streaming.md|decoder/mlp.go:81` | goinfer | `// Only the chosen experts are evaluated — the point of MoE.` |
| `docs/task-moe-streaming.md|decoder/moepaging.go:15` | goinfer | `// only K·L per token; the router's top-k selection is the demand signal. The` |
| `docs/task-moe-streaming.md|decoder/moepaging_test.go:11` | goinfer | `// it with the frequency-aware policy (TestSpanCache_evictsLeastRecentWithPolicy),` |
| `docs/task-moe-streaming.md|decoder/residency.go:130` | goinfer | `return m.residentProjsInt4()` |

## Bare file index

Generated. Every file referenced WITHOUT a line number, and the repo it resolves in.
Existence only — there is no line to key content against, which is recorded rather
than papered over.

| file | repo |
|---|---|
| `internal/serveapp/chaos_test.go` | goinfer |
| `internal/serveapp/fuzz_test.go` | goinfer |
| `scripts/bench_compare.sh` | goinfer |
| `scripts/bench_peer.py` | goinfer |
| `scripts/heavy_gate.sh` | goinfer |
| `scripts/queue_citation_lint.py` | goinfer |
| `scripts/refresh_parity_hashes.sh` | goinfer |

<!-- /CITATION-INDEX -->
