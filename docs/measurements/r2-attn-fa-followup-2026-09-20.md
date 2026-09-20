# R2 follow-up — a real (but irrelevant) race found and fixed, root cause still open

Continuation of `docs/measurements/r2-attn-fa-2026-09-19.md` (read that first — this record
assumes its background, kernel description, and the original `TestAttentionFA_endToEndReproduction`
table). That record parked R2 with two named-but-untested hypotheses: (1) whether Metal's automatic
hazard tracking correctly serializes `attention_fa` → `attention_fa_combine` under this kernel
pair's specific buffer-reuse pattern, and (2) whether the third-and-later command buffer
specifically behaves differently from the first two. This session picked R2 back up, tested both,
and closes neither — but narrows the search space in a way the next session shouldn't have to
redo.

**Status: STILL PARKED 2026-09-20. A genuine race was found, reproduced in isolation, and fixed —
but proven, empirically, to have zero effect on the actual reproduction test. Position-independence
of the divergence is now confirmed (rules out a data-dependent trigger). A full line-by-line audit
of the kernel source and every Go-side dispatch/uniform/buffer site found no defect. The root cause
remains unknown.**

## 1. The encode-ahead race: real, reproduced, fixed — then ruled out as R2's cause

### Hypothesis

`metal/model.go`'s production decode path (`execLoop`) pipelines command buffers: it encodes
buffer N+1 while buffer N is still executing on the GPU (`BeginNP`/`Commit`/`FinishEncoding`/
`WaitDone`, distinct from the synchronous `Begin`/`End` pattern used elsewhere). The
`attention_fa` dispatch site was, at the time, calling `r.uAttnFAG.SetU32(...)` /
`r.uAttnFANSplit.SetU32(...)` inline, once per layer, during encoding. Those are raw CPU writes to
Metal shared memory — not a tracked GPU command — so if that write happens while buffer N's
`attention_fa` dispatch is still reading the same buffer on the GPU, Metal's automatic hazard
tracking (which only orders GPU-encoded commands against each other) does not protect it. A classic
"safe on every machine until it isn't" pattern.

### Isolated repro

`metal/attn_fa_pipeline_race_test.go` (new): a minimal harness using the exact production
pipelining pattern (`cq.BeginNP()` / `enc.FinishEncoding()` / `cur.Commit()` / `cur.WaitDone()`),
with shared `uG`/`uNSplit` uniform buffers whose value is deliberately varied every iteration
(`splitVariants := []int{14, 8, 4, 28, 14, 8, 4, 28}`) so a race becomes observable instead of
silently landing on the same value every time. Re-run today for exact numbers:

```
$ go test -mod=readonly -tags goinfer_testhooks ./metal/ -run TestAttentionFA_pipelinedEncodeRace -v
iter 0: wantG=6 wantNSplit=14 cosine=1.0000000 maxAbs=0.0000
iter 1: wantG=6 wantNSplit=8  cosine=1.0000000 maxAbs=0.0000
iter 2: wantG=6 wantNSplit=4  cosine=NaN        maxAbs=1.3436   -- FAIL
iter 3: wantG=6 wantNSplit=28 cosine=1.0000000 maxAbs=0.0000
iter 4: wantG=6 wantNSplit=14 cosine=1.0000000 maxAbs=0.0000
iter 5: wantG=6 wantNSplit=8  cosine=1.0000000 maxAbs=0.0000
iter 6: wantG=6 wantNSplit=4  cosine=-0.0010009 maxAbs=1.5352   -- FAIL
iter 7: wantG=6 wantNSplit=28 cosine=0.1645985  maxAbs=0.8199   -- FAIL
--- FAIL: TestAttentionFA_pipelinedEncodeRace (0.10s)
```

This is a genuine race: iterations 0-1 correct, iteration 2 wrong (echoing R2's own "2 correct,
then wrong" shape closely enough to be the reason this was tried first) — but then iterations 3-5
correct again and 6-7 wrong, an irregular pattern with no fixed period. That irregularity is itself
diagnostic: a deterministic bug would fail the same iterations every run; this is what an actual
timing race looks like when it's reproduced at all.

### Fix applied

`metal/model.go`: moved the `SetU32` calls for `uAttnFAG`/`uAttnFANSplit` out of the per-layer
dispatch site (`encodeAttentionResidualWith`) and into `setPos`, which runs once per forward call,
before any layer's dispatch is encoded and before the command buffer that will use those values is
committed — the same timing guarantee `uPos`/`uNKeys`/`uRopePos` already relied on. This is safe
specifically because `nSplit`/`G` are provably depth-independent constants in the only regime
`attention_fa` ever engages in (`attnFASplitFor`'s `nKeys/32` cap never binds above
`attnFADepthFloor`), so a single write per forward call — instead of one that could race the
*next* call's encode-ahead — is correct, not just less racy. Added `r.attnFANKV` (cached at
`BuildResident`) so `setPos` can compute `G` without a `*decoder.Model` reference. Full comment
trail is at the `uAttnFAG, uAttnFANSplit Buffer` field declaration and at `setPos` itself.

### Result: zero effect on the actual bug

Re-running `TestAttentionFA_endToEndReproduction` after the fix produced **bit-for-bit identical
(still wrong) output** — same failure at step 2, same magnitude. This is because
`TestAttentionFA_endToEndReproduction` calls `forwardLogits`, which uses the synchronous
`Begin()`/`End()` pattern: one command buffer per forward call, encoded and then fully waited-on
before the function returns. There is no encode-ahead, no pipelining, no second command buffer
in flight while the first is executing — the exact condition the race requires never occurs on
this code path. The fix is real hardening (it closes a genuine defect that *would* bite the
pipelined `execLoop` production decode path under the right timing), but it does not touch, and
was never going to touch, this reproduction test.

**This was reported to the user directly as a walk-back rather than presented as a fix**, and the
user chose to keep investigating for the real, stateful cause rather than stop here.

## 2. Position-independence: the divergence is not tied to specific K/V content

### Hypothesis under test

Decode step 2 (absolute position 1602 in the original repro) is where the prefill's own content
first gets attended to *and* is one step further from the prefill boundary than steps 0-1. Is the
divergence caused by something specific to the data at that particular position (an accumulator
overflow, a NaN-producing edge case in a specific K or V value), or does it depend only on how many
decode calls have happened, independent of what the data at that call actually is?

### Test

Added `metal/attn_fa_e2e_depth_test.go` (`TestAttentionFA_endToEndReproductionDepth2200`), a copy
of `attn_fa_e2e_test.go` with `prefillLen` changed from 1600 to 2200. Because the harness's
Gaussian inputs are drawn from `rand.NewSource(7)` and prefill consumes `prefillLen` rows before
decode starts, this also changes every actual float value the decode steps see — not just the
position label.

```
$ GOINFER_HEAVY_TESTS=1 GOINFER_TEST_MODEL=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  go test -mod=readonly -tags goinfer_testhooks ./metal/ -run TestAttentionFA_endToEndReproductionDepth2200 -v -timeout=5m
decode step 0 (pos=2200): cosine=1.0000000 maxAbs=1.9073e-06
decode step 1 (pos=2201): cosine=1.0000000 maxAbs=1.9073e-06
decode step 2 (pos=2202): cosine=0.9994620 maxAbs=6.1551e-01   -- FAIL
decode step 3 (pos=2203): cosine=0.9992243 maxAbs=8.9019e-01   -- FAIL
decode step 4 (pos=2204): cosine=0.9997135 maxAbs=4.9738e-01   -- FAIL
--- FAIL (80.04s)
```

Side by side with the original depth-1600 result:

| step | depth-1600 pos | depth-1600 cosine | depth-1600 maxAbs | depth-2200 pos | depth-2200 cosine | depth-2200 maxAbs |
|---|---|---|---|---|---|---|
| 0 | 1600 | 1.0000000 | 2.86e-06 | 2200 | 1.0000000 | 1.91e-06 |
| 1 | 1601 | 1.0000000 | 2.86e-06 | 2201 | 1.0000000 | 1.91e-06 |
| 2 | 1602 | 0.9995727 | 0.588 | 2202 | 0.9994620 | 0.616 |
| 3 | 1603 | 0.9995052 | 0.623 | 2203 | 0.9992243 | 0.890 |
| 4 | 1604 | 0.9995974 | 0.569 | 2204 | 0.9997135 | 0.497 |

### Conclusion

Changing the prefill depth by 600 tokens — which changes every K/V value the decode steps
attend to, via a different RNG consumption offset — does **not** move the failure. It still starts
at relative decode step 2 (the third `ForwardEmb` call), at a comparable divergence magnitude
(cosine 0.9992-0.9997, maxAbs 0.5-0.9 in both runs). It would be an extraordinary coincidence for a
data-dependent numerical edge case (an overflow triggered by a specific K or V value at a specific
position) to land on the same *relative* call twice with completely different underlying data. This
rules the data-dependent-content hypothesis out and confirms the divergence is a **call-count or
carried-state mechanism**: something about being the third call to use `attention_fa` since the
resident was built, independent of what position that call is at or what the surrounding data is.

One structural fact this also establishes, worth recording since it wasn't explicit before: `Forward
Batch` (the batched prefill call) cannot itself be dispatching `attention_fa` at any point during
prefill, regardless of prefill length — `attention_fa`'s kernel addresses `q` as exactly one
query position per kv head (`qr := q + kvh*G*hd`), which cannot represent a multi-position batch.
If prefill silently invoked it per-token internally, the two prefill lengths (1600 vs 2200) would
put a very different number of prior `attention_fa` calls behind decode step 0 (roughly 64 vs 664,
counting positions past the depth floor within each prefill), and the failure would not land on the
same relative decode step. It does, which is only possible if `attention_fa` activity starts fresh
at decode step 0 in both cases — i.e., the "third call" is the third `forwardLogits`/`ForwardEmb`
invocation, full stop, not the third time `attention_fa` fires including some prefill-internal
usage.

## 3. Full kernel and dispatch source audit — clean

Per the user's explicit choice of next step (cheaper than building a new multi-layer harness): read
`attention_fa` and `attention_fa_combine`'s MSL source (`metal/kernels.go:953-1058`) and every
Go-side site that builds, sizes, or writes into the buffers they use (`metal/model.go`), looking
specifically for anything keyed on a value that could legitimately differ between the first two
calls and the third. Nothing was found. Specifically checked and ruled out:

- **`nSplit`/`G` uniform timing.** Written exactly once per `setPos` call (see §1's fix), before
  any layer's dispatch is encoded, under a fully synchronous `Begin()`/`End()` command buffer with
  no pipelining — there is no window in which a later write could land before an earlier read.
- **`nSplit`'s value itself.** `attnFASplitFor(nKeys, nKV) = clamp((2*14+nKV-1)/nKV, ..., nKeys/32,
  attnFAMaxSplit)`. For this model (nKV=2), `want=14`; the `nKeys/32` cap only binds below
  `nKeys≈448`, far below `attnFADepthFloor=1536` where the kernel can ever engage — so `nSplit=14`
  is provably identical across steps 0, 1, and 2 at both prefill depths tested. Confirmed by direct
  computation, not just by re-trusting the original investigation's debug-print claim.
- **`attnFAPartial` buffer sizing and addressing.** `pStride = G*(hd+2)` is computed identically on
  the Go allocation side (`maxAttnFAPartialElems = max over layers of nKV*G*(hd+2)`) and inside
  both kernels. The buffer is allocated for `attnFAMaxSplit=32` splits but only `nSplit=14` are
  ever written or read in a given call — same 14 on every call, so there is no leftover-from-a-
  different-nSplit staleness possible. `attention_fa`'s write region (`(kvh*nSplit+split)*pStride`)
  and `attention_fa_combine`'s read region use the same uniform-derived `nSplit`, so they cannot
  disagree with each other.
- **`uAttnFAG`/`uAttnFANSplit` aliasing.** Independently allocated (`NewBufferU32(d, 0)` twice at
  `BuildResident`, metal/model.go:1164) — no shared storage with any other uniform buffer that a
  copy-paste field-omission bug could explain.
- **Threadgroup memory (`shm`) staleness.** All 128 threads unconditionally write their own row
  (`shm + tid*stride`) before the barrier and before any thread reads a different row back — the
  combine step never depends on threadgroup memory being pre-zeroed or retained from a prior
  dispatch.
- **Online-softmax identity handling for empty key chunks.** Computed the actual chunk boundaries
  at prefillLen=1600, steps 0 and 2 by hand: `chunkLen = ceil(nWin/14)` is 115 in both cases, and
  the last split's chunk (`chunkStart=1495, chunkEnd=min(1610,nKeys)`) is non-empty at both steps —
  the `m == -INFINITY` empty-chunk path this kernel relies on for correctness is not newly
  triggered at step 2.
- **`hd` consistency.** Hardcoded `128u` inside `attention_fa`, passed as a uniform (`g.uHd`) into
  `attention_fa_combine` — both are 128 for this model at every layer, confirmed via the same `g :=
  L.geom` binding used earlier in the same function for the QK-norm dispatch.

No defect was found anywhere in this pass. Combined with §1 and §2, the three most concrete,
checkable hypotheses this investigation has produced are now all closed without finding the cause:
it is not the encode-ahead race (wrong code path entirely), not data-dependent K/V content
(position-independence proven), and not a dispatch-parameter, buffer-sizing, buffer-aliasing, or
uniform-timing bug anywhere in the code this session could read (full audit, clean).

## 4. What is and isn't established now

**Established, beyond what the original record had:**
- The `execLoop` pipelining race is real, reproduced in isolation, and fixed — genuine hardening,
  independently worth keeping, but not R2's cause.
- The divergence is call-count/state-based, not data-dependent — confirmed by changing the prefill
  depth (and thus all underlying K/V content) by 600 tokens without moving the failure off relative
  decode step 2.
- `ForwardBatch` cannot itself be invoking `attention_fa` mid-prefill (a structural fact inferred
  from the position-independence result, not previously stated explicitly).
- Every dispatch parameter, buffer size, buffer address computation, and uniform-write timing site
  reachable by reading the code is self-consistent and correct for step 2 exactly as for steps 0-1.

**Still not established:** what specifically differs, at the Metal/GPU level, about the third
`forwardLogits`/`ForwardEmb` call under the fully synchronous `Begin()`/`End()` pattern. Every
Go-level and kernel-source explanation this and the original session could think of has now been
checked and rejected. What remains is either something below the level a source read can see (GPU
driver/command-queue-internal state, e.g. Metal's typical triple-buffering depth — coincidentally
also 3 — for some internal resource neither `metal/model.go` nor the kernel source directly
controls), or a kernel-internal execution-order dependency that a static read cannot catch (though
the bug's exact, byte-for-byte reproducibility across independent runs argues against an ordinary
nondeterministic in-kernel race, which would typically vary run to run on real hardware).

## 5. File status and recommendation

All from this session, uncommitted as of this record:
- `metal/model.go` — the `setPos`-relocation fix (§1). Real hardening; recommend keeping and
  committing regardless of R2's outcome, with its comment trail already framed honestly (explains
  what it fixes and, via this record, what it doesn't).
- `metal/attn_fa_pipeline_race_test.go` — the isolated race repro. Recommend keeping as a permanent
  regression test for the `execLoop` pipelining pattern; its own doc comment should be checked
  against this record before commit so it doesn't imply (as an earlier draft of the reasoning did)
  that this race explains the real-generation bug.
- `metal/attn_fa_e2e_depth_test.go` — the depth-2200 variant. Recommend keeping as a second,
  permanent repro alongside the original: it's the only evidence for position-independence, and a
  future attempt at the root cause should re-run both together to make sure any fix closes both at
  once.
- `docs/tasks/red-october.md` — NOT touched this session; it is substantially uncommitted from
  another session's work (388 lines) and R2's row there already correctly points at "root cause not
  found — see the record." Whoever commits this record should add this file to that pointer chain
  without otherwise touching the other session's in-flight edits.

## 6. Next step, for whoever picks this up

The cheap, readable-from-source leads are exhausted. What's left needs either:
- A Metal GPU frame capture / `MTLCaptureManager` trace of the third command buffer specifically,
  to see what the GPU actually scheduled differently from the first two (a tool-access question,
  not a code-reading one).
- A minimal harness that chains multiple *separate*, fully-synchronous (`Begin`/`End`) command
  buffers back-to-back — not a full 28-layer real model, and not the pipelined pattern
  `attn_fa_pipeline_race_test.go` already covers — specifically to see whether the "third
  synchronous command buffer" boundary itself (as opposed to real-model state) is sufficient to
  reproduce the jump, which would point at something Metal-internal rather than anything in this
  repo's own buffer/uniform management.

`TestAttentionFA_endToEndReproduction` and `TestAttentionFA_endToEndReproductionDepth2200` remain
the two known-good, fully deterministic repros — start from them, not from rebuilding a new one.
