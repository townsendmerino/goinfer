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
> and says so. **Updated 2026-09-04** against `docs/audit-2026-09-02.md`'s full remediation, which
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
> as of that note) but not yet `MatmulBTW4A8Batch` — R-06 still needs the batch-matmul half.
> Cross-references: `docs/audit-2026-09-02.md` (P-06 and C-12 closed, as before), its L-05 and L-15,
> `docs/QUEUE.md` §A (the single-conversation limit), `docs/spec/09-mtp-heads.md` ("Pricing the
> narrow state snapshot"), `docs/task-freetoken-techniques.md` (Lead 1), aikit
> `docs/task-simd-audit.md` (S-02, S-03). **R-00 (the correctness bug, fixed 2026-09-03):**
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
| **R-00** | a plain turn reuses resident KV rows a block-spec or draft-model generation overwrote | `decoder/blockspec.go:195`, `decoder/speculative.go:125-130` — neither calls `residentForgetIDs` | wrong output, silently | forget (or commit) on both paths; a test that alternates the paths | **bug — fix first** |
| **R-01** | the whole conversation, every turn, on every recurrent family, resident path | `decoder/resident_reuse.go:50` refuses `hasRecurrentState()` outright | a full prefill per agent turn (8.85 s at 2.3k tokens on the 7B dense; the 35B-A3B is the model this hits) | phase 0: exact-extension reuse, no snapshot; phase 1: the narrow snapshot via `CopyDeviceBatch`; phase 2: parked checkpoints | **phase 0 fixed 2026-09-03** (exact-extension reuse; CUDA-hardware scenarios unrun); phase 1/2 open; L-05 |
| **R-02** | the prefix, after a cancelled generation | `decoder/model.go` `generateInto`'s `select` on `ctx.Done` returns without committing | the next turn cold-prefills after every interrupt | commit `prompt+generated` at that exit — the cache is consistent there | **fixed 2026-09-03** |
| **R-03** | the prefix, after any speculative generation | `decoder/spec_eagle.go`, `decoder/spec_ngram.go` forget; R-00's two never clear | a `--drafter`/`--spec` agent loop gets no prefix reuse at all | commit the accepted sequence for attention-only families; forget (or restore, R-01 phase 1) for recurrent ones | **`spec_ngram.go` fixed 2026-09-03**; `spec_eagle.go` never touches resident state — the fix doesn't apply there (see below) |
| **R-04** | the prefix, when a second conversation interleaves, or a stop string fires | QUEUE §A "single-conversation"; P-18 / L-15 (`internal/serveapp/sessions.go` whole-containment) | a cold prefill per switch; ~8.9 s vs 43 ms to park 257 MiB | park per-conversation KV (+ state, phase 2) in host RAM; ask `rewindForReuse` for the partial prefix | open; **P-18 confirmed and measured 2026-09-03** (148× TTFT at 2k tokens on the real prefill path, well past L-15's own funding bar) — the fix itself is still L-15's, not attempted |
| **R-05** | the int4 nibble unpack, per token, per paged expert | `decoder/moepaging.go:62-77` — a paged tensor is never repacked; the canonical kernel runs every use | row4 vs canonical is 1.33× on the M=1 GEMV; MoE is ~70% of a CPU-paged 35B token | repack into the slot on fetch (the owned-buffer fetch already copies) | **investigated 2026-09-03, not implemented**: the described mechanism belongs to the Metal pager, not this one; the CPU-paged equivalent (`.giw` kind-4 row4) already SHIPPED and its own performance case is UNRESOLVED per this repo's own measurement saga (swung between −49% and +49% across sessions) — see below |
| **R-06** | the same activation row quantised 7× per layer where 4 would do | `decoder/attention.go:79-81` (q, k, v as three `matmulInto`), the gate/up pair in `decoder/mlp.go` — W8A8 batches, W4A8 does not | ~509k elements/token on the 1.5B, plus 3 fork/joins per layer (fork/join measured 1.70× on decode, aikit S-09.1) | a `MatmulBTW4A8Batch` mirroring `MatmulBTW8A8Batch` (aikit S-02/S-03), wired where `qkvOps` already is | open; aikit-side first — **S-03's NEON quantiser shipped in aikit v1.33.0** (built 2026-09-03, unmeasured per aikit's own tracking; goinfer bumped to it 2026-09-04), but `MatmulBTW4A8Batch` itself does not exist yet — still the blocking half |
| **R-07** | one forward per token on the embeddings route; every input tokenised twice | P-17's second half (`decoder/embed.go`, `internal/serveapp/embeddings.go`) | "sequential prefill", ~9× slower than batched | batched prefill through `forwardLayersN`; tokenise once | **fully fixed 2026-09-03**: `decoder/embed.go`'s per-token forward (`hiddenLastBatched`, ~12-14× measured) and `embeddings.go`'s double-tokenize (`embedBatchCounter`) are both done |
| **R-08** | per token: the whole generated text re-decoded and rescanned for stops; a penalty map rebuilt over the whole history; a full vocabulary sort for `top_logprobs` | P-17 (`internal/serveapp/openai.go` `streamTokens` and three copies), P-15, P-13 | O(n²) in output length; ~1–2 ms/token late in a 64k reply; 10–20 ms/token with logprobs on | incremental: keep the decoded tail, keep the counts, keep a top-k | **2-of-3 done, 2026-09-03**: P-13's vocab sort and P-15's penalty-map rebuild are both FIXED (`topKByLogit`, incremental `histCounts`); only P-17's `streamTokens` re-decode remains open, deferred on genuine complexity (byte-fallback fusing + UTF-8 completeness + stop-string overlap, not lack of ROI) |
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

- **Where:** `decoder/blockspec.go:195` (`BlockSpec.generate`: `PrefillLastNArgmax(embs, 0)` /
  `PrefillSeedArgmax` prefill the prompt from position 0, then every verify round writes rows) and
  `decoder/speculative.go:125-130` (`GenerateSpeculative` claims `resBusy` and runs the target's
  verify on the resident KV). The five files that forget are `generate_vl.go`, `model.go`,
  `resident_reuse.go`, `spec_eagle.go`, `spec_ngram.go`. `blockspec.go` does not claim `resBusy`
  either.
- **Mechanism:** `resIDs` is written only by `residentCommitIDs` at the end of a completed plain
  generation (`decoder/model.go:1150-1152`) and read by `residentReuseLen` at the start of the next
  (`decoder/model.go:949-955`). `serve` routes a greedy request to `BlockSpec.GenerateStream` and a
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
  already-tested `decoder/model.go:949-950` pattern.

### R-01 · The hybrid families re-prefill the whole conversation every turn (resident path)

- **Where:** `decoder/resident_reuse.go:50` — `if m.hasRecurrentState() { return 0 }`, added
  2026-09-02 after repeated identical greedy prompts on qwen3.6-35B-A3B decoded from the previous
  generation's tail state. `decoder/forwardn.go:134-145` is the shared predicate;
  `cuda/resident.go:275` holds the per-layer `dnWin`/`dnState` that are mutated in place and
  re-zeroed only at pos 0.
- **What the staged path already does, and the resident path should copy:** the CPU `Session`
  reuses through `rewindForReuse` (`decoder/session.go:73-80`) → `KVCache.TruncateTo`
  (`decoder/kvcache.go:448-463`), whose rule for recurrent state is: `pos == 0` resets, `pos < c.pos`
  is **inexact** (cold prefill), and `pos == c.pos` is **exact**. An agent turn is `previous prompt +
  reply + tool result`, so `commonPrefixLen == c.pos` and the staged cache reuses it warm — the
  recurrent state after the committed sequence *is* the live state, nothing to rewind. The only
  hybrid-specific refusal on that path is `reconcile`'s reset after a mid-sweep rollback
  (`decoder/session.go:98-102`). So `docs/qwen3_5_moe.md:115` ("falls back to full recompute") is
  stale for the case that matters; correct it with the test below.
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
  blocker either way since `CopyDeviceBatch` predates both pins). The
  same record measured the composition cost the batch form is for: 36 separate copies on the 0.8B
  ran at 174 GB/s against 347 for one contiguous copy, ~446 → ~250 µs. Allocate each layer's
  `dnWin`+`dnState` adjacent (or all layers' in one arena) and the batch collapses further. Sizes
  from the same record: 62.8 MiB for the 35B-A3B, 149.6 MiB for the 27B — at DtoD rates, well under
  a millisecond against a 60–95 ms decode step on the 2070S. Consumers: Qwen3.8-27B's native MTP
  head (spec/09), DFlash pairings on hybrid targets, and R-03's commit-after-speculation for
  recurrent families. `specRollbackSafe` is `decoder/forwardn.go:146` exactly.

  Three design notes from spec/09's own pricing record, carried forward here so they aren't
  re-derived: (1) **reuse one buffer across rounds** — allocating fresh each time more than doubles
  the cost (5.2-6.2 ms at int8 vs 2.9 ms reused); (2) **copy `convWin`, not only `S`** — a width-4
  verify replaces the whole conv window, so skipping it is only 6.6% of the bytes but gives WRONG
  logits with no error, not a slower-but-correct path; (3) **make the windows contiguous** (per-layer
  or one arena) so `CopyDeviceBatch`'s adjacent-pair coalescing actually collapses them, which is
  where the 174 to 347 GB/s composition win comes from. CUDA's `DeltaNet` layer holds
  `dnWin`+`dnState` at `cuda/resident.go:275`; Metal has the same `CopyDeviceBatch` available.
  WebGPU is NOT covered by this plumbing at all -- its `dnState` lives in `gpu/decoderunner.go`
  (`*wgpu.Buffer`, transposed `[nv*hv*hk]` relative to the CPU's `[hk,hv]`) and would need its own
  copy path; not scoped here.
  - **Gate:** restore is bit-exact (`convWin` included — spec/09 shows a width-4 verify replaces the
    whole window), so speculative output on a hybrid equals greedy; `TestDFlashLoop_lossless`'s
    shape on a hybrid target. Measure snapshot+restore in situ per round as spec/09 did.
- **Phase 2 — parked checkpoints.** For non-extending reuse (an edited last message) and for
  R-04's multi-conversation case: at commit, `Download` the recurrent state beside the parked KV
  (~5 ms for 62.8 MiB at PCIe 3 ×16) keyed by conversation; restore = `Upload` both. Bytes are
  bounded by turns, not tokens — which is L-05's "semantic anchors" in the form the code already has.
- **Confidence:** high on phase 0 (the staged path is the existence proof, and the invariant is the
  one 3358e6b already relies on); high on phase 1's plumbing being present (read in aikit at
  438acfa); medium on the projected device-side numbers until measured.
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
  `out <- next` (`decoder/model.go:1096-1109`, the send that M8 made cancellable).
- **Mechanism:** at that point the last forward has completed and been sampled, `next` has not been
  forwarded, and `generated` holds exactly the tokens whose K/V (and, for a hybrid, whose recurrent
  state) the cache holds. The cache is as consistent as it is at the commit two branches later; the
  exit just does not say so. Agent harnesses cancel constantly — interrupts, timeouts, disconnects —
  so today each one costs the next turn a cold prefill.
- **Fix:** `if useGPU { m.residentCommitIDs(prompt, generated) }` on that exit only. The `err !=
  nil` exit after a forward stays a forget (a partial write is possible there).
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

### R-03 · Speculative generations forget the prefix — and could commit it

- **Where:** `decoder/spec_eagle.go`, `decoder/spec_ngram.go` (`residentForgetIDs` at entry, and
  again on the n-gram path), plus R-00's two.
- **Mechanism:** the resident cache is positional and attention reads `nKeys = pos+1`, so after a
  rejection the rows past the accepted position are junk that is never consulted and is overwritten
  by the next forward. For an attention-only family the cache after a speculative generation is
  therefore consistent with `prompt + accepted`, exactly as after a plain one. For a recurrent family
  it is not (the state ran ahead by the rejected width) until R-01 phase 1 restores it.
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

### R-05 · Paged experts are unpacked on every use

- **Where:** `decoder/moepaging.go:62-77` — a kind-4 tensor registers row4 *or* canonical, never
  both, and a paged span is served as-is: there is no load-time repack for a read-only span, so a
  kind-3 paged expert runs the canonical kernel on every token it is routed to, and the M>1 tile
  (aikit S-01) never sees it.
- **Mechanism:** the row4 repack is a one-time precomputation resident tensors get and paged ones
  do not. The pager's fetch already copies into an owned buffer (the `pread` rewrite), so repacking
  during that copy memoises the unpack for the expert's whole residency.
- **Size:** row4 vs canonical is 1.33× on the M=1 GEMV (17 vs 12.75 instructions per 32 MACs);
  `docs/task-zeno-compare.md` puts MoE at ~70% of a CPU-paged 35B token, "91.6% genuine GEMV" on
  the canonical kernel — ~1.2× on the token **where the pager is compute-bound**. Not on the
  SSD-bound Mac 35B at 2.19 tok/s, where the miss stream is the term. Also lets expert-major prefill
  (P18) take the tile.
- **Gate:** `TestMoEExpertMajor_bitIdentical` and the pager determinism tests; measure on the
  CPU-paged hybrid cell paired with `off`.
- **Status (2026-09-03): mechanism as described does not exist; the underlying premise is
  UNRESOLVED per this repo's own prior investigation — not implemented.**
  The "Mechanism" bullet's "the pager's fetch already copies into an owned buffer (the `pread`
  rewrite)" describes the **Metal** expert pool (`metal/expertpool.go`, `metal/gemma4_moe.go` —
  `preadIntoU32Buf`, GPU slot staging), not the CPU pager `decoder/moepaging.go:62-77` actually
  cites. That CPU pager (`expertPager.touch` → `mmap.SpanCache.Touch`) is zero-copy mmap +
  `MADV_WILLNEED`/`DONTNEED` — there is no copy step to repack during, so "repack during the copy
  that already happens" is not available as described; building it would mean adding an entirely
  new copy-and-own fetch path to a pager whose whole design is zero-copy residency bounding.
  That gap was already found and closed a different way, before this doc was scoped: the `.giw`
  kind-4 format (`SHIPPED 2026-08-24`, `docs/task-w4a8-neon-bandwidth.md`'s "Format follow-on")
  bakes row4 onto DISK at prequant time (`cmd/prequant -row4`) instead of repacking at fetch
  time — "simpler than \[that doc\] anticipated: the owned-buffer `pread` architecture turned out
  NOT to be required" (its own words) — and `decoder/moepaging.go`'s `addExpert` (the exact
  function this item cites) already prefers the row4 span when present. So the mechanism R-05
  proposes is not just imprecise, it is a MORE complex reimplementation of a
  SIMPLER approach this repo already built and shipped for the identical goal.
  **That simpler, already-shipped approach's own performance case is explicitly unresolved.**
  `docs/task-zeno-compare.md`'s "At-scale acceptance run" through "Supersession (2026-08-25)"
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

### R-06 · One activation, quantised seven times per layer

- **Where:** `decoder/attention.go:79-81` (`matmulInto` ×3 for q, k, v when they are W4A8; the
  W8A8 case already batches through `qkvOps` at `:74-77`), the gate/up pair in `decoder/mlp.go`.
  Each `MatmulBTW4A8Into` re-quantises its input row.
- **Fix:** aikit `MatmulBTW4A8Batch` mirroring `MatmulBTW8A8Batch` (task-simd-audit S-02/S-03),
  then the same `qkvOps` shape here for W4A8. Removes 3 of 7 quantisations and 3 of 5 fork/joins per
  layer; the W8A8 batch form measured 60 → 66–68 tok/s when it shipped.

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
- **Status (2026-09-03): 2 of 3 fixed.** P-13's `computeLogprobs` now uses `topKByLogit` plus a
  small deterministic tie-break sort instead of a full vocabulary sort (`decoder/sampler.go`),
  proven against a full-sort reference on random and tied-logit inputs. P-15's penalty path now
  keeps an incrementally-maintained `histCounts map[int]int` through a single `recordHistory`
  chokepoint (`decoder/sampler.go`), rebuilt from scratch only when `RepeatLastN>0` forces a
  windowed count; mutation-checked by bypassing the chokepoint with a raw `append` and confirming
  the new counts test catches the divergence. P-17's `streamTokens` re-decode/rescan is untouched —
  still open, deferred on genuine complexity (four call sites: `openai.go`, `chatapp`, `gemmaapp`,
  the demo agent).

### R-09 / R-10 · Filed, not implemented

P-10 (the whole-bundle CRC at start; disposition "out of proportionate scope") and P-09 (Gemma-4's
per-layer re-gather; "contingent" on Gemma-4 CPU decode being a target) stay as the audit left them;
listed so the inventory is complete.

## 3. Sequencing — each independently droppable

0. **R-00** (the bug), then **R-02** and **R-03** for attention-only families: three small commits,
   all gated on token-identity vs cold, all worth having before any measurement.
1. **R-01 phase 0**: the exact-extension rule, the two-turn tests, L-05's TTFT rule on the 35B.
   Correct `docs/qwen3_5_moe.md:115` in the same commit.
2. **R-01 phase 1**: `dnWin`/`dnState` snapshot via `CopyDeviceBatch`; lift `specRollbackSafe`'s
   refusal for the hybrid families on the resident path; measure per round as spec/09 did.
3. **R-04** (P-18's cell first) and **R-01 phase 2** together — one parking mechanism.
4. **R-06** (needs the aikit batch form), **R-05**, **R-07**, **R-08** — small, independent, each
   with its audit cell.

## 4. Sources

`decoder/resident_reuse.go` (header rules 1–3), `decoder/model.go` (`generateInto`,
`residentPrefillSeed`), `decoder/session.go`, `decoder/kvcache.go` (`TruncateTo`,
`hasRecurrentState`, `resetRecurrent`), `decoder/forwardn.go` (`hasRecurrentState`,
`specRollbackSafe`), `decoder/blockspec.go`, `decoder/speculative.go`, `decoder/moepaging.go`,
`decoder/attention.go`, `internal/serveapp/openai.go`; `docs/spec/09-mtp-heads.md` (snapshot
pricing, 2026-08-28); `docs/qwen3_5_moe.md` §"Hybrid cache"; `docs/audit-2026-09-02.md` (C-12,
P-06, P-09, P-10, P-13, P-15, P-17, P-18, L-05, L-15); `docs/QUEUE.md` §A; aikit
`gpu/cuda_copy.go`, `gpu/metal_copy.go` (`CopyDevice`, `CopyDeviceBatch`), aikit
`docs/task-simd-audit.md` (S-01, S-02, S-03, S-09.1).
