# Task: work computed, dropped, and computed again — the recompute inventory (2026-09-03)

> **BLUF.** Every large item below is the same shape: a state the runtime already had in hand — a
> KV cache, a recurrent state, a repacked weight, a decoded string, a count — thrown away and rebuilt
> from the token sequence next time. Ranked by what gets redone: **(R-01)** the hybrid families
> (Qwen3.5/3.6-MoE, Qwen3.8, Nemotron-H, Granite, LFM2) re-prefill the **whole conversation every
> turn** on the resident path, because `residentReuseLen` refuses any recurrent family — while the
> staged CPU `Session` already reuses an exact extension of the same state, correctly, by its
> `TruncateTo` exactness rule; the named blocker for the broader checkpoint track (no device-to-device
> copy in aikit's `gpu`) **is gone** — `CopyDevice`/`CopyDeviceBatch` shipped in gpu v0.30.1 and both
> GPU modules already pin past it. **(R-02)** a cancelled generation forgets the prefix although the
> cache is consistent at the cancel point. **(R-03)** every speculative generation forgets the prefix
> too, and **two of them never forget at all** — `BlockSpec.generate` and `GenerateSpeculative` write
> the resident KV without clearing `resIDs`, so a later plain turn can reuse rows another request
> overwrote (**R-00, a correctness bug, fix first**). Then the smaller, memo-shaped ones: paged experts
> unpacked on every use, the same activation quantised seven times per layer, an embeddings route that
> prefills one token at a time, and three O(n²)-per-token loops in serving.
>
> **Status: SCOPED 2026-09-03, nothing started as of the scoping.** Read at goinfer `3e45469` and
> aikit `438acfa` (v1.32.0). Static; every cost figure is quoted from the record that measured it
> and says so. **Updated 2026-09-04** against `docs/completed/audit-2026-09-02.md`'s full remediation, which
> landed after this doc was scoped (goinfer `f1d98d3`) — several of its cross-references were stale
> within a day: P-13 is FIXED; P-15 is 2-of-3 fixed (the penalty-map rebuild this doc's R-08 names is
> done; `forEachChunk` measured not-worth-it; batched-verify buffer churn deferred); P-17 is now
> 2-of-3 fixed as of this doc's own R-07 work below (the decoder-as-embedder batching AND the
> embeddings double-tokenize are both done; only `streamTokens`'s re-decode remains, deferred for
> genuine complexity — three interacting correctness invariants — not lack of ROI, tracked under
> R-08); P-18 is confirmed and measured (148× TTFT
> at 2k tokens on the real production path) with its fix explicitly assigned to L-15, not itself.
> P-09/P-10 stand as filed, not implemented — no change there. See each item below for the
> corrected status. Also: aikit was bumped to v1.33.0 in the same round (`gpu` stays v0.32.0,
> pinned consistently by every module now — metal's go.mod was itself stale at v0.30.1 until this
> bump); v1.33.0 ships S-03's NEON quantiser (built 2026-09-03, per aikit's own tracking, unmeasured
> as of that note) but not yet `MatmulBTW4A8Batch` — R-06 still needs the batch-matmul half. **Since
> shipped:** aikit v1.34.0 added it; R-06's own table row below has the current disposition (wired
> behind `GOINFER_W4A8_BATCH`, measured, PARKED default-off — inside the ambiguous 1.05×/1.15×
> park/ship zone). This paragraph is a snapshot of the v1.33.0 round and is left as the historical
> record of that state, not updated in place.
> Cross-references: `docs/completed/audit-2026-09-02.md` (P-06 and C-12 closed, as before), its L-05 and L-15,
> `docs/QUEUE.md` §A (the single-conversation limit), `docs/spec/09-mtp-heads.md` ("Pricing the
> narrow state snapshot"), `docs/tasks/task-freetoken-techniques.md` (Lead 1), aikit
> `docs/task-simd-audit.md` (S-02, S-03), and — **added 2026-09-13** — `docs/audit-2026-09-10.md`
> P-05, which found R-03's commit was never consumed by the speculative loops' own prefill and
> fixed the read side too; see the correction under R-03 below. **R-00 (the correctness bug, fixed
> 2026-09-03):**
> `BlockSpec.generate` now claims `resBusy` and forgets `resIDs` before any resident write;
> `GenerateSpeculative` already claimed `resBusy` and now forgets too. Mutation-checked; the full
> mixed-traffic CUDA scenario from R-00's own Gate bullet is still unrun (no CUDA hardware here) —
> a portable stub-host test covers the same postcondition instead. See R-00 below for detail.

## 0. What must not change

- **Bit-identity.** Reuse is never a numeric change: a reused prefix must produce the tokens a cold
  prefill of the same sequence produces. Every item here is gated on that, the way 3358e6b was.
- **Conservative where the state is unknown.** `resident_reuse.go`'s rule 2 — forget on any path
  that did not complete — stays the default. The items below narrow *unknown* to *known*: they commit
  where the cache is provably consistent and forget everywhere else. Nothing below reuses a state it
  cannot name.
- **Off is a competitor.** Each measurement has the do-nothing arm (today's cold prefill / today's
  forget) in the same session, paired and interleaved, per `CLAUDE.md`.

## 1. Inventory

| id | what is recomputed | where | size of the redo | fix shape | status |
|---|---|---|---|---|---|
| **R-00** | a plain turn reuses resident KV rows a block-spec or draft-model generation overwrote | `decoder/blockspec.go:201`, `decoder/speculative.go:135-135` — neither calls `residentForgetIDs` | wrong output, silently | forget (or commit) on both paths; a test that alternates the paths | **bug — fix first** |
| **R-01** | the whole conversation, every turn, on every recurrent family, resident path | `decoder/resident_reuse.go:120` refuses `hasRecurrentState()` outright | a full prefill per agent turn (8.85 s at 2.3k tokens on the 7B dense; the 35B-A3B is the model this hits) | phase 0: exact-extension reuse, no snapshot; phase 1: the narrow snapshot via `CopyDeviceBatch`; phase 2: parked checkpoints | **phase 0 fixed 2026-09-03** (exact-extension reuse; CUDA-hardware scenarios unrun); **phase 1 CLOSED 2026-09-23** (spec/09's own measurement, 11.0% of a decode step, killed 2026-08-28 — this doc missed it until now); phase 2 open; L-05 |
| **R-02** | the prefix, after a cancelled generation | `decoder/model.go` `generateInto`'s `select` on `ctx.Done` returns without committing | the next turn cold-prefills after every interrupt | commit `prompt+generated` at that exit — the cache is consistent there | **fixed 2026-09-03** |
| **R-03** | the prefix, after any speculative generation | `decoder/spec_eagle.go`, `decoder/spec_ngram.go` forget; R-00's two never clear | a `--drafter`/`--spec` agent loop gets no prefix reuse at all | commit the accepted sequence for attention-only families; forget (or restore, R-01 phase 1) for recurrent ones | **`spec_ngram.go` fixed 2026-09-03**; `spec_eagle.go` never touches resident state — the fix doesn't apply there (see below) |
| **R-04** | the prefix, when a second conversation interleaves, or a stop string fires | QUEUE §A "single-conversation"; P-18 / L-15 (`internal/serveapp/sessions.go` whole-containment) | a cold prefill per switch; ~8.9 s vs 43 ms to park 257 MiB | park per-conversation KV (+ state, phase 2) in host RAM; ask `rewindForReuse` for the partial prefix | **L-15/P-18 half FIXED 2026-09-23** (`bestExtend` now picks longest-common-prefix, not whole containment, guarded against hijacking a session that merely shares another's system-prompt preamble); the resident-GPU parking half (R-01 phase 2) stays open, pre-registered decision rule below |
| **R-05** | the int4 nibble unpack, per token, per paged expert | `decoder/moepaging.go:108-113` — a paged tensor is never repacked; the canonical kernel runs every use | row4 vs canonical is 1.33× on the M=1 GEMV; MoE is ~70% of a CPU-paged 35B token | repack into the slot on fetch (the owned-buffer fetch already copies) | **investigated 2026-09-03, not implemented**: the described mechanism belongs to the Metal pager, not this one; the CPU-paged equivalent (`.giw` kind-4 row4) already SHIPPED and its own performance case is UNRESOLVED per this repo's own measurement saga (swung between −49% and +49% across sessions) — see below |
| **R-06** | the same activation row quantised 7× per layer where 4 would do | `decoder/attention.go:98-110` (q, k, v as three `matmulInto`), the gate/up pair in `decoder/mlp.go` — W8A8 batches, W4A8 does not | ~509k elements/token on the 1.5B, plus 3 fork/joins per layer (fork/join measured 1.70× on decode, aikit S-09.1) | a `MatmulBTW4A8Batch` mirroring `MatmulBTW8A8Batch` (aikit S-02/S-03), wired where `qkvOps` already is | **wired and measured 2026-09-03/04, PARKED (default-off)**: aikit `MatmulBTW4A8Batch` shipped at v1.34.0; goinfer wired it behind `GOINFER_W4A8_BATCH` (default off) in q/k/v (`attention.go`) and gate/up (`mlp.go`); reproduced on two independent architectures via `bench_peer.py` (n=10 paired, idle-gated) — arm64/Metal 1.071× (stdev 0.009), amd64/CPU 1.066× (stdev 0.0008) — both squarely inside the pre-registered ambiguous zone between the 1.05× park / 1.15× ship thresholds, so it stays off by default per this repo's own "ambiguous → parked" rule. **Follow-up 2026-09-20** (red-october.md R9 step 1's own finding that MLP's token share grows with model size raised the question of whether this does better at 7B): it does not — OFF 58.98 ms/token (stdev 2.10), ON 59.0 (stdev 1.60), a difference an order of magnitude below either arm's own noise — an even cleaner null than the 1.5B result, not a size-dependent win, consistent with the remedy amortizing a roughly fixed per-barrier cost that matters proportionally *less* as the matmuls it's amortized against get bigger. See `docs/measurements/w4a8-batch-7b-2026-09-20.md`. Stays parked. |
| **R-07** | one forward per token on the embeddings route; every input tokenised twice | P-17's second half (`decoder/embed.go`, `internal/serveapp/embeddings.go`) | "sequential prefill", ~9× slower than batched | batched prefill through `forwardLayersN`; tokenise once | **fully fixed 2026-09-03**: `decoder/embed.go`'s per-token forward (`hiddenLastBatched`, ~12-14× measured) and `embeddings.go`'s double-tokenize (`embedBatchCounter`) are both done |
| **R-08** | per token: the whole generated text re-decoded and rescanned for stops; a penalty map rebuilt over the whole history; a full vocabulary sort for `top_logprobs` | P-17 (`internal/serveapp/openai.go` `streamTokens` and three copies), P-15, P-13 | O(n²) in output length; ~1–2 ms/token late in a 64k reply; 10–20 ms/token with logprobs on | incremental: keep the decoded tail, keep the counts, keep a top-k | **ALL FOUR COPIES FIXED.** Serving hot path 2026-09-03 (P-13, P-15, `openai.go` `streamTokens`); `chatapp`/the demo agent 2026-09-11 (`c0ab6ed3`, found already done this pass — this row's own prior text was stale, see below); `gemmaapp` 2026-09-23 (this pass, its own design needed to preserve the leading-space-strip semantic incrementally, as anticipated) |
| **R-09** | the whole `.giw` CRC on every start | P-10 | a full read of a >RAM bundle before the first token | per-layer CRCs | filed, not implemented (disposition 2026-09-03) |
| **R-10** | every KV head's history re-gathered and transposed per layer per token (Gemma-4 CPU) | P-09 | 2–3× attention traffic at long context | store V transposed at append | filed, contingent |

Checked and **not** recompute, or already closed: prefill does not run the LM head per position
(KV-only prefill skips it for all but the last token, `decoder/model.go`); the block-spec seed that
headed the whole prompt for one id is fixed (C-12, `ResidentSeedArgmax`), and the redundant
`xhost` download went with it; RoPE cos/sin per position per layer measured **0.08%** of a prefill
(P-06, closed); the grammar mask is precomputed (P-20, 2a44d4e); speculative verify's rejected
positions are inherent, not recompute.

## 2. Items

### R-00 · Correctness — two paths write the resident KV and never clear `resIDs`

- **Where:** `decoder/blockspec.go:201` (`BlockSpec.generate`: `PrefillLastNArgmax(embs, 0)` /
  `PrefillSeedArgmax` prefill the prompt from position 0, then every verify round writes rows) and
  `decoder/speculative.go:135-135` (`GenerateSpeculative` claims `resBusy` and runs the target's
  verify on the resident KV). The five files that forget are `generate_vl.go`, `model.go`,
  `resident_reuse.go`, `spec_eagle.go`, `spec_ngram.go`. `blockspec.go` does not claim `resBusy`
  either.
- **Mechanism:** `resIDs` is written only by `residentCommitIDs` at the end of a completed plain
  generation (`decoder/model.go:1885`) and read by `residentReuseLen` at the start of the next
  (`decoder/model.go:1591`). `serve` routes a greedy request to `BlockSpec.GenerateStream` and a
  sampled one to `Model.Generate` on the **same** `*Model` (`internal/serveapp/openai.go`, the
  `--drafter` branch). So: plain turn A commits A's ids → greedy turn B prefills B over the same
  positional rows → sampled turn C whose prompt extends A matches A's ids and skips the prefix,
  attending over B's K/V. No error anywhere — exactly the failure `resident_reuse.go`'s header calls
  the whole risk.
- **Fix:** `m.residentForgetIDs()` at the top of both paths (rule 2 as written), and claim
  `resBusy` in `BlockSpec.generate`. R-03 then upgrades the forget to a commit where it is safe.
- **Gate:** a test that runs plain-greedy(A) → block-spec(B, longer than A) → plain(A+suffix) and
  compares against a cold plain(A+suffix), token-identical. CUDA is the only `ResidentDrafterHost`,
  so it runs under `-tags 'cuda goinfer_testhooks'`; a stub host in `blockspec_test.go`'s shape can
  pin the `resIDs == nil` postcondition without a device.
- **Confidence:** high on the mechanism (both call sites read); the scenario needs mixed
  greedy/sampled traffic on one model, which an agent harness supplies.
- **Status (2026-09-03): fixed.** `BlockSpec.generate` (`decoder/blockspec.go`) now claims
  `m.resBusy` via CAS before any device write when `m.resident != nil` (gated to match
  `model.go`'s `useGPU` check, so `NewCPUBlockSpec`'s CPU-only host — which never touches the
  resident device KV — doesn't contend for a claim it doesn't need), returns the new
  `ErrBlockSpecResidentBusy` on a losing claim, and calls `m.residentForgetIDs()` immediately on a
  winning one, before `SetBatchedCapture`/`PrefillLastNArgmax` touch the cache. `speculative.go`'s
  `GenerateSpeculative` already claimed `target.resBusy` (C-03) but never forgot; it now calls
  `target.residentForgetIDs()` right after the claim succeeds, same ordering. Both mutation-checked
  (disabling the new guard makes `blockspec_residentbusy_test.go`'s two tests fail — one to a panic
  on the now-nil drafter, confirming the claim is load-bearing, not just present). The full
  mixed-traffic scenario in the Gate bullet above (plain(A)→block-spec(B)→plain(A+suffix) token
  parity) still needs CUDA hardware and is not run here; what's covered instead is the narrower,
  portable postcondition the Gate bullet itself proposed as a fallback: a stub
  `ResidentDrafterHost` pins `resIDs == nil` after a claimed generation and confirms a losing claim
  returns before touching the host or `resIDs` at all (`TestBlockSpecGenerate_forgetsResIDsBeforeAnyWrite`,
  `TestBlockSpecGenerate_residentBusyDeclines`). `GenerateSpeculative`'s one-line addition has no
  equivalent unit coverage — driving its full goroutine (draft/verify/accept loop, channel output)
  needs a much heavier harness than `BlockSpec.generate`'s synchronous call, and R-03 (which
  upgrades the forget to a commit) is the natural point to build that harness rather than
  duplicating it now for a single-line change whose shape is otherwise identical to the
  already-tested `decoder/model.go:1591` pattern.

### R-01 · The hybrid families re-prefill the whole conversation every turn (resident path)

- **Where:** `decoder/resident_reuse.go:120` — `if m.hasRecurrentState() {`, added
  2026-09-02 after repeated identical greedy prompts on qwen3.6-35B-A3B decoded from the previous
  generation's tail state. `decoder/forwardn.go:200-145` is the shared predicate;
  `cuda/resident.go:395` holds the per-layer `dnWin`/`dnState` that are mutated in place and
  re-zeroed only at pos 0.
- **What the staged path already does, and the resident path should copy:** the CPU `Session`
  reuses through `rewindForReuse` (`decoder/session.go:73-80`) → `KVCache.TruncateTo`
  (`decoder/kvcache.go:539`), whose rule for recurrent state is: `pos == 0` resets, `pos < c.pos`
  is **inexact** (cold prefill), and `pos == c.pos` is **exact**. An agent turn is `previous prompt +
  reply + tool result`, so `commonPrefixLen == c.pos` and the staged cache reuses it warm — the
  recurrent state after the committed sequence *is* the live state, nothing to rewind. The only
  hybrid-specific refusal on that path is `reconcile`'s reset after a mid-sweep rollback
  (`decoder/session.go:98-102`). So `docs/completed/qwen3_5_moe.md:132` ("falls back to full
  recompute") was stale for the case that matters; corrected 2026-09-12 alongside that doc's
  archival, with the test below already the evidence for the fix.
- **Phase 0 — exact extension, no snapshot.** Replace the blanket refusal with the staged rule:
  for a recurrent family, `residentReuseLen` returns `len(m.resIDs)` when the prompt extends the
  entire committed sequence by at least one token, else 0. The existing cap (`len(prompt)-1`) makes
  an identical re-send fall to 0 by construction, which is the repro that motivated the guard. The
  invariant this rests on — *the live recurrent state equals the state after `resIDs` whenever
  `resIDs != nil`* — holds because `resIDs` is set only at the completed-generation commit, every
  token in `generated` was forwarded before the next was sampled, and every other writer forgets
  (after R-00). Cost of the change: a few lines and a test. Prize: the whole per-turn prefill on the
  models the audience runs for MoE.
  - **Gate:** `TestPagerDeterminism` stays green (the identical-prompt repro); new: on
    qwen3.5-0.8b (CPU-resident WebGPU or CUDA) and qwen3.6-35B-A3B (CUDA), a two-turn exact
    extension equals the cold prefill of the concatenation token for token, and a three-turn one
    with a *non*-extending middle turn falls to cold and still matches.
  - **Decision rule (L-05's, pre-registered):** TTFT of turn N in a 10-turn agent transcript on
    Qwen3.6-35B-A3B, resident CUDA, with vs without; fund at ≥2× on turn-3+ TTFT. The do-nothing
    arm is today's cold prefill in the same session.
- **Phase 1 — the narrow snapshot, for speculation on hybrids.** `specRollbackSafe`
  (`decoder/forwardn.go`) refuses every recurrent family because a verify advances the state by K
  tokens and a partial rejection needs it at an earlier one. `docs/spec/09-mtp-heads.md` priced
  the remedy — snapshot before the verify, restore on rejection, one reused buffer — at **2.9 ms on
  the CPU 0.8B, <1% of a K=4 round**, and named the resident blocker: "`aikit/gpu` exposes only
  `Upload`/`Download`, so state already on the device crosses PCIe twice — ~100% of a decode step".
  **That blocker no longer exists.** aikit's `gpu` module has `CopyDevice` (one `cuMemcpyDtoD_v2`)
  and `CopyDeviceBatch` (every copy issued, one synchronize, adjacent pairs coalesced) on CUDA and
  Metal, since gpu v0.30.1; goinfer's `cuda/go.mod` and `metal/go.mod` both pin gpu v0.32.0 as of the
  2026-09-04 aikit bump (`metal/go.mod` had drifted to v0.30.1 before that — already fixed, not a
  blocker either way since `CopyDeviceBatch` predates both pins).

  > **Correction, 2026-09-23: this whole framing is stale. `docs/spec/09-mtp-heads.md` itself went
  > on to measure the real decision number and it KILLED this track — the paragraph above only
  > quotes that same document's EARLIER sections.** The "well under a millisecond against a
  > 60–95 ms decode step" comparison above is exactly the invalid extrapolation spec/09's own later
  > section warns against by name ("the 0.8B resident denominator is not a proxy for a 27B resident
  > denominator... do not extrapolate either way, measure it there") — the 0.8B's own in-situ decode
  > step measured 5.69–7.92 ms, not 60–95 ms; the larger figure belongs to a bigger model this
  > pricing was never run on. **The number that matters is the paired ratio through the real
  > `CopyDeviceBatch` primitive, in situ, against the real decode step it competes with: 623 µs of
  > snapshot+restore against a 5.69 ms decode step, 11.0% (p50, 3 paired runs, sd 2.0–2.9 pp).**
  > Against the pre-registered rule (">5% — the narrow version does not pay on its own. Report and
  > stop.") that fires even under the more favorable K-step framing (5.5–11%, since a batched
  > verify on a bandwidth-bound decode costs nearer one step than K). spec/09's own words: **"the
  > narrow snapshot does NOT pay on its own... it does not pay."** The composition cost is also
  > 2× the isolated-primitive number (real, interleaved buffer layout coalesces far less than a
  > synthetic consecutively-allocated one) — a second reason the "446 µs" figure this section
  > quoted was never the real cost. The one thing that could rescue it — a weight-bandwidth-bound
  > decode step on a 27B+ trunk, where the same absolute snapshot cost would be a smaller fraction
  > — needs hardware that does not fit an 8 GB card, which is every CUDA box this repo currently
  > has. **Phase 1 is CLOSED, not open-and-unblocked; do not build it here.** If it resumes, the
  > entry condition is a resident measurement on an actual 27B+ trunk, not a better acceptance rate
  > on a small one — spec/09's own words, restated because this doc's own prior text obscured them.

  Kept below for whoever eventually re-opens this on qualifying hardware — every design note remains
  correct even though the track itself does not currently pay:

  Sizes from spec/09's own record: 62.8 MiB for the 35B-A3B, 149.6 MiB for the 27B. Consumers:
  Qwen3.8-27B's native MTP head (spec/09), DFlash pairings on hybrid targets, and R-03's
  commit-after-speculation for recurrent families. `specRollbackSafe` is `decoder/forwardn.go:212`
  exactly.

  Three design notes from spec/09's own pricing record, carried forward so they aren't re-derived
  if this resumes: (1) **reuse one buffer across rounds** — allocating fresh each time more than
  doubles the cost (5.2-6.2 ms at int8 vs 2.9 ms reused); (2) **copy `convWin`, not only `S`** — a
  width-4 verify replaces the whole conv window, so skipping it is only 6.6% of the bytes but gives
  WRONG logits with no error, not a slower-but-correct path; (3) **make the windows contiguous**
  (per-layer or one arena) so `CopyDeviceBatch`'s adjacent-pair coalescing actually collapses them —
  spec/09 measured this specific gap costing 2× (174 vs 347 GB/s on a synthetic probe; the REAL,
  interleaved layout measured even worse, 65 GB/s in situ). CUDA's `DeltaNet` layer holds
  `dnWin`+`dnState` at `cuda/resident.go:395`; Metal has the same `CopyDeviceBatch` available.
  WebGPU is NOT covered by this plumbing at all -- its `dnState` lives in `gpu/decoderunner.go`
  (`*wgpu.Buffer`, transposed `[nv*hv*hk]` relative to the CPU's `[hk,hv]`) and would need its own
  copy path; not scoped here.
  - **Gate, if this resumes:** restore is bit-exact (`convWin` included — spec/09 shows a width-4
    verify replaces the whole window), so speculative output on a hybrid equals greedy;
    `TestDFlashLoop_lossless`'s shape on a hybrid target.
- **Phase 2 — parked checkpoints.** For non-extending reuse (an edited last message) and for
  R-04's multi-conversation case: at commit, `Download` the recurrent state beside the parked KV
  (~5 ms for 62.8 MiB at PCIe 3 ×16) keyed by conversation; restore = `Upload` both. Bytes are
  bounded by turns, not tokens — which is L-05's "semantic anchors" in the form the code already has.
  - **Pre-registered decision rule, before any code (2026-09-23) — do not repeat the phase-1
    mistake of building on a projected number.** The honest do-nothing arm is NOT a cold prefill:
    per the R-04 finding above, a lost resident slot already falls back to the staged CPU path
    for that conversation's own cache, and with L-15 shipped that fallback finds its own prefix.
    So the number that decides Phase 2 is: **(park round-trip once per commit + resident-GPU
    decode for the turns until this conversation reclaims the slot) vs. (staged CPU decode for
    those same turns, on the L-15-fixed path).** For a model whose CPU decode is dramatically
    slower than its resident GPU decode (a large MoE, say) this plausibly favors parking; for a
    model fast enough on CPU that the gap is small, the ~5-30ms round-trip (scales with recurrent
    state + KV size, not just the 62.8 MiB figure above) may not be worth the eviction/budget
    machinery it requires. Measure on a real multi-conversation interleave on this box's own
    hardware — not projected from spec/09's single-round numbers, which priced a different event
    (per-verify-round DtoD, killed in phase 1) than this one (per-commit DtoH/HtoD). Ambiguous →
    park, same as everywhere else in this repo's discipline.
  - **Measured 2026-09-23** (`docs/measurements/r01-phase2-cpu-fallback-cost-2026-09-23.md`), real
    hardware, the model this row names: Qwen3.6-35B-A3B int4 CPU decode-only on nobara-pc is
    **4.19 tok/s** (streaming, first-to-last-chunk, prefill excluded) against `docs/benchmarks.md`'s
    M35 CUDA-resident row of **23.5 tok/s** — **5.6× slower, real but not catastrophic** (RAM
    headroom is why: 21 GiB model against 52 GiB available, unlike the Mac's catastrophic same-path
    result elsewhere in `docs/benchmarks.md`, where the model didn't fit RAM at all). Arithmetically
    this makes parking look like an overwhelming win for a several-hundred-token turn (~98s
    difference against a sub-second round-trip) — **but the measurement exposes that the arithmetic
    answers a narrower question than Phase 2 needs answered.** Parking only makes a LATER restore
    cheap (`Upload` vs. a cold GPU reprefill); it does not change what a conversation does WHILE it
    does not hold the single resident slot, which today is exactly the CPU fallback just measured —
    that's a **scheduling/concurrency policy question** (should a losing `resBusy` CAS block briefly
    for the slot instead of falling back immediately?), genuinely separate from the snapshot/restore
    mechanism Phase 2 was scoped to build, and not obviously worth it on its own: at 5.6× (not two or
    three orders of magnitude), a losing conversation blocking even a few seconds hoping for the slot
    could easily lose to just running on CPU immediately, depending on the current holder's own turn
    length — unmeasured, not assumed either way. **Two distinct, independently-fundable candidates
    now, not one:** (a) the bounded-wait scheduling policy (small, needs a turn-length-distribution
    measurement, no new snapshot machinery), and (b) the parking/snapshot mechanism as originally
    scoped (bigger, still needs the real interleaved measurement named above, now with a real CPU
    baseline to measure against instead of a projection). Neither built here.
- **Confidence:** high on phase 0 (the staged path is the existence proof, and the invariant is the
  one 3358e6b already relies on); phase 1 is CLOSED per the 2026-09-23 correction above, not a
  confidence question any more — the device-side numbers were measured (spec/09) and the answer
  was no.
- **Status (2026-09-03): Phase 0 fixed.** `residentReuseLen` (`decoder/resident_reuse.go`) no
  longer blanket-refuses a recurrent family: it now returns `len(m.resIDs)` when the prompt is an
  exact, strict extension of the entire committed sequence (`m.resIDs` is a full prefix of
  `prompt` and `len(prompt) > len(m.resIDs)`), else 0 — narrower than the generic branch's
  longest-common-prefix match, on purpose, since a recurrent state has no per-position history to
  rewind into. Mutation-checked against the generic branch's own test cases carried over
  (a mid-sequence divergence the generic branch WOULD reuse now correctly returns 0 for a recurrent
  model — `TestResidentReuseLen_recurrentExactExtensionOnly`). Full decoder suite green. The Gate
  bullet's CUDA half (`TestPagerDeterminism` on a real qwen3.6-35B-A3B checkpoint) is unrun here
  (no CUDA hardware) but verified by inspection: it repeats an IDENTICAL prompt, and
  `len(prompt) <= len(m.resIDs)` for a resend still returns 0 under the new rule exactly as the old
  blanket refusal did, so its pass/fail is unaffected by this change. The two-turn/three-turn
  token-identity-vs-cold-prefill scenarios also need real hardware and are not run; the portable
  test instead pins the id-arithmetic those scenarios exercise (which is all `residentReuseLen`
  is — no backend interaction). L-05's decision rule (TTFT funding gate) needs the same hardware
  and is unattempted.

### R-02 · A cancelled generation forgets a prefix that is intact

- **Where:** `generateInto`'s `select { case <-ctx.Done(): g.err = ctx.Err(); return ... }` before
  `out <- next` (`decoder/model.go:1707`, the send that M8 made cancellable).
- **Mechanism:** at that point the last forward has completed and been sampled, `next` has not been
  forwarded, and `generated` holds exactly the tokens whose K/V (and, for a hybrid, whose recurrent
  state) the cache holds. The cache is as consistent as it is at the commit two branches later; the
  exit just does not say so. Agent harnesses cancel constantly — interrupts, timeouts, disconnects —
  so today each one costs the next turn a cold prefill.
- **Fix:** `if useGPU { m.residentCommitIDs(prompt, generated) }` on that exit only. The `err !=
  nil` exit after a forward stays a forget (a partial write is possible there). (Doc-review
  note, 2026-09-22: `residentCommitIDs` has since grown two more params for VL image blocks and
  LoRA — the live call sites pass `m.residentCommitIDs(prompt, generated, nil, lora)`; the
  two-arg form quoted here is the shape at the time this was written, not copy-pasteable today.)
- **Gate:** cancel mid-stream, then extend the prompt with what was emitted; equals cold.
- **Status (2026-09-03): fixed**, exactly as specified — the fix is the literal one-line addition
  quoted above, at the `ctx.Done()` exit only. Gated portably (no GPU/checkpoint needed) with the
  same fake-resident harness `resident_seam_test.go` already uses:
  `TestGenerateResident_cancelCommitsExactlyWhatWasEmitted` cancels mid-stream and asserts
  `resIDs == prompt+received` (the exact invariant, not a hardcoded token count, since the
  cancel-vs-send race can let one extra token through); `TestGenerateResident_forwardErrorStillForgets`
  pins the sibling negative case — the `err != nil` exit must keep forgetting. Mutation-checked
  (disabling the new call breaks the first test with `resIDs = []`, confirming it is load-bearing).
  `decoder/model.go` is a parity-manifest core file; `scripts/refresh_parity_hashes.sh` was run
  after this edit (31/31 forward goldens that ran stayed green, `deps_hash` refreshed for 28
  families, `validated_at` untouched). Full decoder suite green.
- **Correction (2026-09-04, docs/completed/review-2026-09-04.md V-04): the fix above was incomplete.** An
  iteration is `top-of-loop check → sample (µs) → send → Forward (ms)`; a cancel that arrives
  during Forward — the dominant interval by far — is not observed until the TOP-OF-LOOP select on
  the NEXT iteration, not the send-select this fix originally targeted. At that point the previous
  iteration's token had both been appended to `generated` and already forwarded (its K/V write
  happens synchronously before the loop ever reaches the top again), so the cache is exactly as
  consistent there as at the send-select exit — but the top-of-loop exit was never wired to commit,
  so most real cancels (the ones landing during Forward, not during the microsecond sample/send
  window) still cold-prefilled. Fixed: the same `if useGPU { m.residentCommitIDs(prompt, generated)
  }` added to the top-of-loop exit too (`decoder/model.go:1707`). Mutation-checked at
  `-count 100`: without this second commit the existing test fails intermittently (~35/100 runs,
  confirming V-10's "scheduling-dependent" characterization empirically); with it, 100/100 pass,
  and `-race -count 20` alongside the R-03 sibling test is clean.

### R-03 · Speculative generations forget the prefix — and could commit it

- **Where:** `decoder/spec_eagle.go`, `decoder/spec_ngram.go` (`residentForgetIDs` at entry, and
  again on the n-gram path), plus R-00's two.
- **Mechanism:** the resident cache is positional and attention reads `nKeys = pos+1`, so after a
  rejection the rows past the accepted position are junk that is never consulted and is overwritten
  by the next forward. For an attention-only family the cache after a speculative generation is
  therefore consistent with `prompt + accepted`, exactly as after a plain one. For a recurrent family
  it is not (the state ran ahead by the rejected width) — R-01 phase 1 would have restored it, but
  that track is closed per its own 2026-09-23 correction, so recurrent families stay on `forget`
  here for the foreseeable future, not pending a fix already in flight.
- **Fix:** at the end of each speculative loop, commit `prompt + accepted` when
  `!hasRecurrentState()`, forget otherwise. With `--drafter` this is the difference between an agent
  loop that gets prefix reuse and one that never does.
- **Gate:** the same token-identity-vs-cold test as R-01, run through each loop.
- **Status (2026-09-03): `spec_ngram.go` fixed; `spec_eagle.go` does not apply, corrected below.**
  `genNgramInto`'s resident branch (`decoder/spec_ngram.go`, `cache == nil && target.resident !=
  nil && target.DecodeRunnerEligible()`) now commits at every exit that follows a successful (or
  stop/cancelled) `emit` — as opposed to a `targetVerify`/prefill error, which can leave a partial
  write and correctly stays forgotten. The commit is `hist` (the loop's own name for "the prompt
  plus every token whose K/V is actually in the cache"), not always literally `prompt + accepted`:
  a round's own trailing token is streamed before it is forwarded (forwarding happens at the START
  of the next round, as that round's `targetVerify` `seq[0]`), so a return right after that
  specific emit is one token behind the stream — safe (never claims more than the cache holds),
  gated by `TestGenerateNgramSpeculative_residentCommitsAcceptedSequence`'s prefix check rather
  than exact equality. No conditional forget-for-recurrent was needed: `validateNgramSpec`'s
  `specRollbackSafe` check already rejects recurrent families before the goroutine starts, on
  every entry point (`GenerateNgramSpeculative` and `Session.genSpec` both call it), so
  `hasRecurrentState()` is always false inside this branch.
  **`spec_eagle.go`'s two functions (`GenerateEagleSpeculative`,
  `GenerateEagleSpeculativeTree`) never touch `m.resident` at all** — both use `m.embedToken`
  (not `embedResident`) and a local `tc := m.NewCache(...)` for every forward
  (`m.forwardNAttn(ctx, ids, tc, fastAttn)` routes to `m.forward`/`m.forwardLayersN` with that
  explicit CPU cache, never `m.resident.Forward`/`ForwardN`). Their `residentForgetIDs()` calls at
  entry are therefore not a partial-write guard against anything these functions themselves do —
  the resident device state, if any, is genuinely unmodified by an EAGLE generation, so
  R-03's "commit prompt+accepted" fix does not apply here: committing would claim the resident
  device holds tokens it never received, which is a worse bug than the one being fixed. Whether
  the forget itself is even necessary (the resident KV a prior turn committed remains valid and
  reusable after a CPU-only EAGLE call) is a real, separate recompute-avoidance opportunity in
  this document's own theme, but changing it needs the same care as everything else here and is
  left unstarted rather than risked on inference — filed as a candidate follow-up, not attempted.
  Mutation-checked (disabling `spec_ngram.go`'s commit call breaks the gate test). Full decoder
  suite green; `spec_ngram.go`/`spec_eagle.go` are not parity-manifest core files, no hash refresh
  needed.
- **Correction (2026-09-10, `docs/audit-2026-09-10.md` P-05): the "fixed" status above was true of
  the commit and false of the reuse.** `spec_ngram.go`'s and `blockspec.go`'s writers correctly
  wrote `resIDs`, but nothing on the read side consumed it: both speculative loops still
  cold-prefilled from position 0 on every call, so a `--spec`/`--drafter` agent loop got no prefix
  reuse at all — exactly the outcome this section's own "the difference between an agent loop that
  gets prefix reuse and one that never does" line claimed had been achieved. P-05 fixed both
  halves: `decoder/blockspec.go`'s `generate` now calls `m.residentReuseLen` before the forget and
  seeds/fuses only the unreused suffix (mirroring `generateInto`), and a new `Model.resDrafterSynced
  *BlockSpec` field (`decoder/model.go`) names which `BlockSpec` instance's own drafter context the
  current `resIDs` actually reflects, so a token-identical commit from a *different* writer (a plain
  turn, or another `BlockSpec`) correctly forces a cold drafter refuse rather than reusing a context
  it was never fused into. Confirmed present in the tree at review time
  (`decoder/blockspec.go:239,454,462`, `decoder/model.go:60-54`). No further action here; this note
  exists so a reader of this doc alone doesn't stop at "fixed" and miss that the fix needed a
  second pass elsewhere.

### R-04 · A second conversation, or a stop string, means a cold prefill

- **Where:** QUEUE §A "Resident prefix reuse is single-conversation" (the parked-KV fix: ~257 MiB
  and ~43 ms for a 2.3k-token conversation, against ~8.9 s to recompute); P-18
  (`internal/serveapp/sessions.go` whole-containment rule; `decoder/session.go` can reuse a partial
  prefix but is never asked; the stop-string tokens the client never sees are committed); L-15.
- **Fix shape:** per-conversation parking in host RAM (KV now; recurrent state with R-01 phase 2),
  and longest-common-prefix selection instead of whole containment. The shared Claude Code system
  prompt is the same prefix in every conversation on the box, which is what a prefix tree would
  compute once (FreeToken Lead 1). Measure P-18's cell first: TTFT of turn 3 with `"stop":["\n\n"]`
  vs without at 2k history.
- **Confirmed 2026-09-23 (scoping R-01 phase 2 before any build): the two "R-04" mechanisms this
  cell conflates have different failure modes, and one of them already degrades gracefully.**
  Losing the resident GPU slot (`resBusy` CAS, `decoder/model.go:1518`) is NOT the same event as
  a cold prefill: the comment at `decoder/model.go:1522` states it directly — "a loser falls back
  to the staged CPU path, which uses this call's own cache, so both still complete correctly" —
  and the code confirms it: `useGPU=false` (lines 1457-1500) falls straight into the existing
  `else` branch's `m.prefillLogits(ctx, prompt[prefillFrom:], cache)` (`decoder/model.go:1600`), i.e. that
  conversation's own `Session`/`KVCache`, not a fresh one. So a lost CAS costs GPU-vs-CPU decode
  speed for that turn, nothing more — **provided** the staged session that receives the fallback
  can find its own prefix. That second condition is exactly P-18/L-15: `sessions.go`'s
  `bestExtend` whole-containment rule (not `decoder/session.go`'s already-correct
  `rewindForReuse`/`commonPrefixLen`, which it never calls with a partial match) is what turns an
  otherwise-graceful CPU fallback into the measured 148× cold-prefill case. **This is why the
  existing sequencing below (P-18's cell first) is right, not just convenient**: fixing L-15
  makes the resident GPU's single-slot limit cost "CPU speed for one turn" instead of "cold
  prefill," which is the correct, cheap floor to have in place before spending any design effort
  on parking resident device state at all — Phase 2 only has a case to make once that floor
  exists, because it has to beat *that* baseline, not a cold prefill.
- **L-15/P-18 half FIXED 2026-09-23.** `bestExtend` (`internal/serveapp/sessions.go`) now scores
  every candidate by `commonPrefix(toks, prompt)` and picks the largest, but only when it beats
  the floor that same candidate shares with every OTHER known session (computed the same way,
  pairwise, inside the same pool) — a session that's nothing but the shared system-prompt preamble
  never clears its own floor, so two distinct conversations still can't hijack each other, exactly
  as `TestBestExtend`'s existing `fresh` case requires. An exact continuation's match is its own
  full length, which always clears that floor, so the old whole-containment cases are unchanged;
  the new part is that a session whose STORED tokens are no longer fully contained in the prompt —
  a stop-string hit's invisible tail, a `max_tokens` cut, or an edited last message — now still
  gets picked and handed to `decoder/session.go`'s `rewindForReuse`, which was already correct and
  needed no change (confirmed by tracing `internal/serveapp/openai.go:1324`'s
  `sess := lm.sessions.acquire(gr.promptIDs)` into the very next `sess.Generate(ctx, gr.promptIDs,
  ...)` call: same prompt both times, so `Generate`'s own `rewindForReuse` independently recomputes
  the true common prefix regardless of what `bestExtend` matched — `bestExtend` only decides WHICH
  session receives that treatment). `TestBestExtend_stopStringTokensForceColdPrefill` (P-18's own
  pinned-bug gate) is now `TestBestExtend_stopStringTokensReuse`, asserting the fix instead of the
  bug; `TestBestExtend_editedLastMessageReuse` added for P-18's other named case. `faultBack`'s
  cold-tier candidates get the same treatment for free (same `bestExtend`), though its floor is
  computed only within the cold pool, not against currently-resident sessions — a real but minor
  gap, not fixed here. Full `internal/serveapp` suite green (223 tests), `gofmt`/`go vet` clean,
  `staticcheck` (pinned 0.8.0) clean and confirmed live against a throwaway U1000 case.

### R-05 · Paged experts are unpacked on every use

- **Where:** `decoder/moepaging.go:108-113` — a kind-4 tensor registers row4 *or* canonical, never
  both, and a paged span is served as-is: there is no load-time repack for a read-only span, so a
  kind-3 paged expert runs the canonical kernel on every token it is routed to, and the M>1 tile
  (aikit S-01) never sees it.
- **Mechanism:** the row4 repack is a one-time precomputation resident tensors get and paged ones
  do not. The pager's fetch already copies into an owned buffer (the `pread` rewrite), so repacking
  during that copy memoises the unpack for the expert's whole residency.
- **Size:** row4 vs canonical is 1.33× on the M=1 GEMV (17 vs 12.75 instructions per 32 MACs);
  `docs/completed/task-zeno-compare.md` puts MoE at ~70% of a CPU-paged 35B token, "91.6% genuine GEMV" on
  the canonical kernel — ~1.2× on the token **where the pager is compute-bound**. Not on the
  SSD-bound Mac 35B at 2.19 tok/s, where the miss stream is the term. Also lets expert-major prefill
  (P18) take the tile.
- **Gate:** `TestMoEExpertMajor_bitIdentical` and the pager determinism tests; measure on the
  CPU-paged hybrid cell paired with `off`.
- **Status (2026-09-03): mechanism as described does not exist; the underlying premise is
  UNRESOLVED per this repo's own prior investigation — not implemented.**
  The "Mechanism" bullet's "the pager's fetch already copies into an owned buffer (the `pread`
  rewrite)" describes the **Metal** expert pool (`metal/expertpool.go`, `metal/gemma4_moe.go` —
  `preadIntoU32Buf`, GPU slot staging), not the CPU pager `decoder/moepaging.go:108-113` actually
  cites. That CPU pager (`expertPager.touch` → `mmap.SpanCache.Touch`) is zero-copy mmap +
  `MADV_WILLNEED`/`DONTNEED` — there is no copy step to repack during, so "repack during the copy
  that already happens" is not available as described; building it would mean adding an entirely
  new copy-and-own fetch path to a pager whose whole design is zero-copy residency bounding.
  **Update 2026-09-21**: that "entirely new copy-and-own fetch path" now exists —
  `decoder/moepool.go`'s `expertBufferPool` (Lever 1b, `docs/completed/task-moe-streaming.md`),
  opt-in via `GOINFER_MOE_PREAD_CPU=1`, default off. It does NOT close this gap, though: it was
  built and measured for a different reason entirely (a real darwin RAM cap, since
  `MADV_DONTNEED` is a no-op there — the CPU pager's zero-copy claim above still holds for the
  DEFAULT path), and its own `refill` re-fetches whichever representation was already selected
  at build time (row4-preferred, same choice `addExpert` already makes) — it does not repack
  canonical into row4 during the copy. Building THAT would be new scope on top of Lever 1b, not
  something it already does. The rest of this finding's conclusion is unaffected: the `.giw`
  kind-4 disk-baked row4 format below is still the reason this specific gap was already closed.
  That gap was already found and closed a different way, before this doc was scoped: the `.giw`
  kind-4 format (`SHIPPED 2026-08-24`, `docs/completed/task-w4a8-neon-bandwidth.md`'s "Format follow-on")
  bakes row4 onto DISK at prequant time (`cmd/prequant -row4`) instead of repacking at fetch
  time — "simpler than \[that doc\] anticipated: the owned-buffer `pread` architecture turned out
  NOT to be required" (its own words) — and `decoder/moepaging.go`'s `addExpert` (the exact
  function this item cites) already prefers the row4 span when present. So the mechanism R-05
  proposes is not just imprecise, it is a MORE complex reimplementation of a
  SIMPLER approach this repo already built and shipped for the identical goal.
  **That simpler, already-shipped approach's own performance case is explicitly unresolved.**
  `docs/completed/task-zeno-compare.md`'s "At-scale acceptance run" through "Supersession (2026-08-25)"
  is an unusually thorough saga on two real checkpoints (gemma4-26b, qwen3.5-35B-A3B): kind-4
  under paging first measured a REAL 25-34% regression (root-caused and fixed — the pager was
  registering both canonical AND row4 spans per tensor, doubling I/O per miss); post-fix it
  measured roughly at parity with kind-3 on a busy machine, then a **47-49% regression** on a
  quiet one (ruling out both I/O waste and hit-rate as causes via `AdvisedBytes` and a
  budget-invariance check); then a follow-on pass **withdrew** that regression entirely,
  measuring kind-4 **27-49% FASTER** on the identical configs — with the same untouched kind-3
  control drifting 29% between the two sessions with zero code change, meaning neither the
  "slower" nor the "faster" reading is distinguishable from noise. The doc's own conclusion:
  "reversed pending different-day confirmation... nothing here is a green light to dispatch
  row4 under `-stream-weights`," with `TestGemma4EndToEndThroughput`
  (`decoder/gemma4_endtoend_throughput_test.go`) kept specifically as that gate and two real
  `.giw` bundles kept on disk for it — a re-run nobody has executed since 2026-08-25 (not
  attempted here either: the bundles aren't on this machine, and at 46 GB free this box could
  not hold even one model's kind-3+kind-4 pair, ~35-75 GB, without risking the exact
  near-full-disk state suspected as the original swings' confound).
  **Also found and fixed as a byproduct:** `cmd/prequant -row4`'s own `-h` text still stated the
  withdrawn "69% SLOWER... regresses paged throughput 12-49%" finding as settled fact, with no
  mention of the 2026-08-25 retraction — corrected to describe the actual UNRESOLVED state
  (CLAUDE.md's own rule: a retraction must reach every place quoting the figure), while keeping
  the same "do NOT use with `-stream-weights`" caution the source doc itself argues for keeping
  until a confirmed reproduction lands either way.
  **Recommended next step, if this is pursued:** run `TestGemma4EndToEndThroughput` on a
  different machine/session/disk-fill state per the doc's own gate — not write new pager code.
  This repo's own measurement-discipline rule, written directly out of this saga
  ("any single-machine micro-benchmark result that will drive more than a day of downstream
  implementation work must reproduce on a different day... before any remedy gets built against
  it"), argues directly against building anything further on this premise before that happens.
  **Checked 2026-09-22: this box cannot be that different machine either.** `nobara-pc`'s NVMe
  is at 32 GB free of 1.8 TB (99% used) — less headroom than the 46 GB this doc's own prior
  session already found insufficient for one model's kind-3+kind-4 `.giw` pair (~35-75 GB), and
  neither bundle is present here to re-derive from a smaller footprint. The re-run this section
  calls for remains genuinely unattempted, not just unattempted-here — it needs either freed disk
  on an existing box or a third machine, not new pager code either way.

### R-06 · One activation, quantised seven times per layer

- **Where:** `decoder/attention.go:98-110` (`matmulInto` ×3 for q, k, v when they are W4A8; the
  W8A8 case already batches through `qkvOps` at `:74-77`), the gate/up pair in `decoder/mlp.go`.
  Each `MatmulBTW4A8Into` re-quantises its input row.
- **Fix:** aikit `MatmulBTW4A8Batch` mirroring `MatmulBTW8A8Batch` (task-simd-audit S-02/S-03),
  then the same `qkvOps` shape here for W4A8. Removes 3 of 7 quantisations and 3 of 5 fork/joins per
  layer; the W8A8 batch form measured 60 → 66–68 tok/s when it shipped.
- **Status (2026-09-03): wired, measured, PARKED behind `GOINFER_W4A8_BATCH` (default off).**
  aikit built `MatmulBTW4A8Batch` (commit `daeeff9`, shipped in v1.34.0) after measuring which of
  S-02's two remedies applied — goroutine-wake-stagger, not shard skew — and designed `W4A8Op` with
  OPTIONAL `Row4`/`Row4Scales` fields specifically so a canonical-only mirror wouldn't regress
  goinfer's row4-using decode path. goinfer added `isW4A8`/`wmW4A8Op` (`decoder/weightmat.go`) and
  `matmulW4A8Batch` (`decoder/backend.go`) helpers, new `qkvOpsW4`/`guOpsW4` scratch fields
  (`decoder/scratch.go`), and dispatch branches in `attention.go` (q/k/v) and `mlp.go` (gate/up),
  each gated behind `w4a8BatchEnabled` (`GOINFER_W4A8_BATCH=1`), mirroring the existing
  `w4a8SplitHalfRepackEnabled` precedent. First measured via a git-stash-built before/after A/B
  (11 runs, real `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`, Mac only): 1.08× mean, noisier
  per-pair spread (0.984–1.143). **Reproduced 2026-09-04 with a cleaner instrument on two
  independent architectures** — `scripts/bench_peer.py`'s own `goinfer`-only cell (no peer engine
  involved; the toggle is the sole variable), n=10 paired/interleaved runs, idle-gated between
  every cell, same `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`: MacBook (arm64, row4 kernel active)
  **1.071× mean** (stdev 0.0092, range 1.063–1.089); nobara (amd64, canonical/split-half — no
  row4, idles at load 0.00) **1.066× mean** (stdev 0.0008, range 1.065–1.067). Both land tightly
  inside S-02's pre-registered ambiguous zone (≥1.05 park / <1.15 ship) — not noise, a real,
  reproducible effect on two architectures. The near-identical ratio despite very different
  underlying kernels (row4 vs canonical) says the win is almost entirely the goroutine-wake-stagger
  amortization from fusing the fork/joins, not a row4-bandwidth effect. Stays off by default rather
  than shipped, per this repo's own measurement-discipline rule that the zone just below a
  threshold is where motivated reasoning lives — now with materially higher confidence than the
  original single-box reading. Correctness proven both toggle-off and toggle-on by
  `TestInt4_forwardParity` (mutation-checked: corrupting `group` in either batch call is caught by
  8+ of 22 fixtures), and re-confirmed against real aikit v1.34.0 by `TestForwardN_matchesSequential`
  (bit-identical) and `TestMoEExpertMajor_bitIdentical`, run explicitly. `docs/env-vars.md`
  documents the toggle. Honest next step if this is revisited: dynamic chunking, measured
  separately by aikit at 1.135× and complementary (it fixes the same stagger from the other
  direction) — not a looser reading of this gate.

### R-07 · The embeddings route prefills one token at a time

- **Where:** P-17's second half — `decoder/embed.go` runs one forward per token ("the sequential
  prefill the flags call ~9× slower"); `internal/serveapp/embeddings.go` tokenises each input twice.
- **Fix:** `forwardLayersN` over the whole input; tokenise once.
- **Status (2026-09-03): both halves FIXED.** `HiddenLast` now dispatches to `hiddenLastBatched`
  (one `runLayersFromEmbedN` call, `canBatchN`-gated, falling back to the original per-token
  `hiddenLastSequential` for `K==1` and the families that path excludes) — measured ~12-14× at
  K=64 on a real Qwen3-0.6B checkpoint, cosine 1.0000000000 against the old per-token path across
  four sequence lengths.
  `internal/serveapp/embeddings.go`'s double-tokenize traced to `decoderEmbedder.encodeLocked`
  (`internal/serveapp/decoder_embedder.go`): it tokenizes to get `ids`, uses them for the forward,
  and discards them; `CountTokens` then re-tokenized the SAME text from scratch purely to report
  `usage.prompt_tokens`. Fixed with a new optional capability, `embedBatchCounter`
  (`EncodeBatchCounted`, mirroring the existing `embedTokenCounter` pattern), that the
  decoder-backed embedder implements by returning `len(ids)` from the SAME `encodeLocked` call
  already made for the vector — no second tokenize. `handleEmbeddings` prefers it when the
  embedder implements it, falling back to the original `EncodeBatch` + `countEmbedTokens` for
  encoders that can't (the aikit `encoder.Encoder` interface — an external dependency —
  can't have `EncodeBatch` itself changed to return counts, unlike R-06's aikit blocker this one
  didn't need an aikit change at all, since the fix lives entirely in goinfer's own
  `decoderEmbedder` wrapper). Gated by
  `TestDecoderEmbedder_encodeBatchCountedMatchesSeparateCalls` (heavy: real Qwen3-0.6B checkpoint;
  vectors bit-identical to `EncodeBatch`, counts identical to `CountTokens`) and
  `TestHandleEmbeddings_prefersEncodeBatchCounted` (portable: a counting fake proves the handler
  actually dispatches to the byproduct path and never calls the two-pass fallback). Both
  mutation-checked. Full `internal/serveapp` suite green (133 pass / 0 fail / 27 skip).

### R-08 · Three O(n²)-per-token loops in serving

- P-17: `streamTokens` re-decodes the whole generated sequence and rescans it for stop strings on
  every token (`internal/serveapp/openai.go`, and the same pattern in `chatapp`, `gemmaapp`, the
  demo agent) — decode the new piece (`DecodePiece` exists) and scan the tail plus `max(len(stop))−1`
  bytes. P-15: presence/frequency penalties rebuild a `map[int]int` over the whole history every
  token — keep the counts. P-13: `computeLogprobs` full-sorts the vocabulary per token whenever
  `top_logprobs > 0` — `topKByLogit` exists. Each has its own measurement cell in the audit.
- **Status (2026-09-03): 3 of 3 pieces addressed; the serving hot path (`openai.go`) is fully
  fixed, the three demo CLIs remain open by deliberate scope choice.** P-13's `computeLogprobs`
  now uses `topKByLogit` plus a small deterministic tie-break sort instead of a full vocabulary
  sort (`decoder/sampler.go`), proven against a full-sort reference on random and tied-logit
  inputs. P-15's penalty path now keeps an incrementally-maintained `histCounts map[int]int`
  through a single `recordHistory` chokepoint (`decoder/sampler.go`), rebuilt from scratch only
  when `RepeatLastN>0` forces a windowed count; mutation-checked by bypassing the chokepoint with
  a raw `append` and confirming the new counts test catches the divergence.
  **P-17's `streamTokens` (`internal/serveapp/openai.go`) is now fixed.** Two independent
  quadratic sources, both removed: (1) `DecodeContinuation(ids)` re-decoded the whole generated
  sequence every token — replaced with `DecodePiece(id)` appended to a `strings.Builder`
  incrementally. Safe because `decode()` (`tokenizer/sentencepiece.go`) has no state that depends
  on chunk boundaries: a byte-fallback token's raw byte is written out whenever a flush lands, and
  concatenation is associative regardless of when that is, so decoding one token at a time and
  concatenating is byte-identical to decoding the whole sequence at once — proven directly by
  `TestDecodeContinuation_isIncrementallyAssociative` (`tokenizer/`) across a real multi-byte-emoji
  byte-fallback run, the exact case this used to be cautious about. (2) `firstStop`/`completeUTF8`
  themselves re-scanned the FULL accumulated text every token (`strings.Index` from byte 0,
  `completeUTF8`'s rune walk from byte 0) — fixing only the decode would have just moved the O(n²)
  cost, not removed it. Both are now bounded to `text[printed:]` (the not-yet-emitted tail, which
  does not grow with total output length): `text[:printed]` provably never contains a stop match,
  complete or in progress, because `stopTailHold`'s own invariant — hold back every suffix of the
  current text that could be a stop's prefix — means `printed` never advances past the start of a
  still-possible match. Checked empirically, not just argued: `TestStreamTokens_windowedScanMatchesFullRescan`
  (`internal/serveapp/`) runs 300 random token streams (including matches split across token
  boundaries — "EN"+"D" spelling "END" — near-misses, and a 3-token-split match) through both the
  new windowed implementation and a reference that mirrors the exact pre-fix full-rescan
  algorithm, comparing emitted text and stop-hit byte-for-byte. Both new tests mutation-checked
  (a corrupted window, a corrupted decode, and a broken offset translation were each confirmed
  caught). The existing static M-25 guard (`TestStreamTokens_decodesAsAContinuation`, an AST-based
  check that streamTokens never calls the leading-space-stripping `Decode`) needed updating to
  look for `DecodePiece` instead of `DecodeContinuation` — `DecodePiece` is equally non-stripping
  by its own documented contract, so the property it protects is unchanged; re-verified by
  mutation (swapping in `Decode` makes the guard fail again). Full `internal/serveapp` suite green
  (134 pass / 0 fail / 27 skip); `gofmt`/`go vet`/`staticcheck` clean.
  **Correction, 2026-09-23: the "NOT touched" claim above was already stale when this doc last
  said so.** `chatapp` and the demo agent (`demo/agent/agent/agent.go`) were fixed 2026-09-11
  (`c0ab6ed3`, "decode chat/agent CLI streams incrementally, not per-token whole-slice (P-17)") —
  eight days before this doc's most recent prior pass reasserted they were untouched. Both extract
  their own `streamGen` (`DecodePiece` appended to a `strings.Builder`), and both sidestep the
  strip question this section worried about entirely: they only ever decode the GENERATED
  continuation through `streamGen` — the prompt is never rendered through it, so there is no
  sequence-start strip to preserve incrementally in the first place. Found by grepping for
  `DecodePiece`/`streamGen` in these files directly rather than trusting the doc's own status line
  (this repo's own `doc-review` skill's rule 0) — a lesson from this doc's own recent R-01 phase 1
  retraction, applied here before writing anything, not after.
  **`gemmaapp` genuinely was untouched, and is now fixed too (2026-09-23, this pass).** Unlike
  chatapp/agent, `internal/gemmaapp/main.go`'s loop ALSO decodes the prompt (through the same call
  that renders the generation), specifically to make the SentencePiece leading-space strip land
  once at the true sequence start (`tokenizer/sentencepiece.go:800`'s own comment names this
  design). The fix keeps that one-time whole-sequence `Decode` call for the prompt exactly as
  before (it already ran once per request, not once per token, so it was never the O(n²) source)
  and only replaces the GENERATION loop's repeated whole-sequence re-decode with `DecodePiece`
  appended to a `strings.Builder` seeded from the prompt's own decoded text — the same
  concatenation-is-associative argument `streamTokens`/chatapp/agent already rely on, extended one
  step further to cover a one-time non-piece prefix. Extracted into its own `streamGen(tk,
  promptIDs, tokens, onChunk)` (`internal/gemmaapp/main.go`) so it's unit-testable without a model,
  mirroring chatapp's own test shape: `TestStreamGen_matchesWholeSequenceDecode` compares against
  `tk.Decode(append(promptIDs, genIDs...))` (the exact call the old code made, run once instead of
  per-token) across a real tokenizer fixture, chunk-by-chunk and on the final text; a second test
  covers the zero-generated-tokens edge case. `go build`/`go vet`/`gofmt`/`staticcheck` clean;
  `go test -v ./internal/gemmaapp/...` 2/2 pass. **R-08 is now fully closed — all four copies of
  this pattern (serving + three demo CLIs) are fixed**, not three of four as this section's prior
  text believed.

### R-09 / R-10 · Filed, not implemented

P-10 (the whole-bundle CRC at start; disposition "out of proportionate scope") and P-09 (Gemma-4's
per-layer re-gather; "contingent" on Gemma-4 CPU decode being a target) stay as the audit left them;
listed so the inventory is complete.

## 3. Sequencing — each independently droppable

0. **R-00** (the bug), then **R-02** and **R-03** for attention-only families: three small commits,
   all gated on token-identity vs cold, all worth having before any measurement.
1. **R-01 phase 0**: the exact-extension rule, the two-turn tests, L-05's TTFT rule on the 35B.
   (Shipped 2026-09-03; the `docs/completed/qwen3_5_moe.md:132` correction it called for landed
   2026-09-12, with the archival.)
2. ~~**R-01 phase 1**~~ — **CLOSED 2026-09-23**: `docs/spec/09-mtp-heads.md` already ran this
   measurement (2026-08-28) and killed it, 11.0% of a resident decode step against a 5% bar; this
   doc's own prior text here missed that verdict. Re-entry condition: a resident 27B+ trunk, not
   available on any CUDA box this repo currently has.
3. ~~**R-04, L-15/P-18 half**~~ — **FIXED 2026-09-23**: `bestExtend` longest-common-prefix
   selection, guarded against system-prompt-preamble hijacking. **R-01 phase 2** (resident-GPU
   state parking) stays open — its economics are NOT the ones phase 1 killed (it pays once per
   commit/conversation-switch, amortized over a whole turn's tokens, not once per speculative
   verify round), but the honest baseline it has to beat is now the L-15-fixed staged CPU path,
   not a cold prefill — see the pre-registered decision rule in R-01's own section above.
4. **R-06** (needs the aikit batch form, PARKED — measured, ambiguous zone, default off),
   **R-05** (blocked — needs disk/hardware this box doesn't have, checked 2026-09-22), **R-07**
   (fixed), ~~**R-08**~~ — **FIXED 2026-09-23** (`gemmaapp`, the last of its four copies).

## 4. Sources

`decoder/resident_reuse.go` (header rules 1–3), `decoder/model.go` (`generateInto`,
`residentPrefillSeed`), `decoder/session.go`, `decoder/kvcache.go` (`TruncateTo`,
`hasRecurrentState`, `resetRecurrent`), `decoder/forwardn.go` (`hasRecurrentState`,
`specRollbackSafe`), `decoder/blockspec.go`, `decoder/speculative.go`, `decoder/moepaging.go`,
`decoder/attention.go`, `internal/serveapp/openai.go`; `docs/spec/09-mtp-heads.md` (snapshot
pricing, 2026-08-28); `docs/completed/qwen3_5_moe.md` §"Hybrid cache"; `docs/completed/audit-2026-09-02.md` (C-12,
P-06, P-09, P-10, P-13, P-15, P-17, P-18, L-05, L-15); `docs/QUEUE.md` §A; aikit
`gpu/cuda_copy.go`, `gpu/metal_copy.go` (`CopyDevice`, `CopyDeviceBatch`), aikit
`docs/task-simd-audit.md` (S-01, S-02, S-03, S-09.1).

<!-- doc-reviewed: 2026-09-23 -->
