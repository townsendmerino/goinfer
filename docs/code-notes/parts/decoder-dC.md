# decoder: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `decoder`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## cpuFastAttention

Moved from `decoder/forwardn.go` (the comment above `cpuFastAttention`) on 2026-10-10.

```text
cpuFastAttention reports whether the operator opted into A3's f32 prefill
attention (G24). Read here only; every consumer receives it as an explicit
argument so no path can pick it up by accident — see runLayersFromEmbedN.

Enabling it gives up two of acc64's three guarantees for this model:
spec-decode verify == sequential greedy (structurally prevented from applying
there anyway), and decode == prefill. Measured divergence at dense 1.5B:
cosine ~0.9976, and a measured 2.28x on an 8k prefill
(docs/measurements/attention-a3-kernel-ratio-2026-08-26.md).
DEFAULT ON since 2026-08-31 (operator decision). GOINFER_CPU_FAST_ATTENTION=0 turns it off,
which is what --cpu-exact-prefill sets. The variable was previously opt-IN ("1" enabled it), so
the sense of an EXPLICIT "1" is unchanged and only the unset case moved.

What flipping the default costs, stated plainly because it is now the shipped behaviour: a long
prompt can produce a different response than the same prompt did before this change, at
temperature 0, on the CPU backend. Measured divergence is cosine ~0.9976 at dense 1.5B, stable
across 256/1024/2048-token prompts. Decode is untouched; this is prefill only. Speculative
verify is structurally excluded (it passes fastAttn=false, not a runtime check).

WHAT IT GIVES UP IS BIGGER THAN "prefill != decode", and the help text understated it until
2026-08-31: f32 attention is not SPLIT-INVARIANT. The f64 accumulator makes a prompt's KV
independent of how the prompt was chunked; f32 reassociation does not. So a SESSION — which
prefills a warm prefix and then a divergent suffix, two chunks — stops matching a one-shot
generate over the same tokens. Measured: TestSessionNgramSpecParity fails with the flag on and
passes with it off, and it failed that way BEFORE this became the default, so it is a latent
property of the flag rather than of defaulting it.

That divergence is ACCEPTED as of 2026-08-31 (operator decision) rather than excluded, because
excluding sessions does not restore the equality — it only moves the disagreement from the
split to the kernel — and it costs measurably: 1.43x on a cold 2048-token turn (+18.3s) and
1.32x on a warm 2048+128 suffix, since a suffix still attends over the whole prefix.
TestSessionFastAttnDivergence pins the new behaviour; the equality is still gated, under the
exact kernel, by TestSessionNgramSpecParity.

IT IS FLOORED BY PROMPT LENGTH — see fastAttnMinPrompt. The win scales with K; the divergence
does not, so below the floor the default would trade a different answer for nothing.

MoE IS *NOT* EXCLUDED, and that is deliberate: 66d0a05 removed the exclusion after measuring it
(1-cosine 2.126e-3 for MoE against 2.400e-3 for the dense case this already ships, depth-matched,
48/48 identical greedy continuation). --help still claims "REFUSED for MoE models at any
setting"; that text is STALE, not a description of a guard, and is corrected there.

N-34: the sentence that used to end this paragraph — "so it is now enforced in code" — was
itself false. cpuFastAttention() below reads one env var and has no arch check; 82dda2a
removed the guard when 66d0a05's measurement made the exclusion unnecessary. Two comments
naming the same list, and both wrong, in opposite directions.
cpuFastAttention is the live-environment reading (a hand-built caller with no Model); a Loaded model reads
its own snapshot through (*Model).cpuFastAttention.
```

## fastAttnMinPrompt

Moved from `decoder/forwardn.go` (the comment above `fastAttnMinPrompt`) on 2026-10-10.

```text
fastAttnMinPrompt is the prompt length below which f32 prefill attention is NOT used, even
when enabled. Attention is O(K·nKeys), so the win grows with K while the divergence does not:
a short prompt gets a different answer and buys almost nothing for it.

Measured 2026-08-31 (qwen2.5-coder-1.5b int4, M1 Pro, cold prefill, f32 vs f64-accum):

	K=512   8.59s vs  9.90s   1.15x
	K=1024 19.29s vs 23.66s   1.23x
	K=2048 42.83s vs 61.08s   1.43x
	K=8192 (from the flag's own record)  2.28x

Against that, an 8-TOKEN prompt diverged at the THIRD generated token of 24 and never
re-converged — full divergence, no measurable win. Without this floor, flipping the default
would have changed the output of every short request in exchange for nothing, which is not the
trade the flag documents or the one it was turned on for.

512 IS A JUDGEMENT, NOT A MEASUREMENT: it is the smallest K measured with a win over ~15%, and
nothing here identifies a crossover point. A sweep that found one should move this and say so.
It is deliberately NOT configurable — a knob here would be a third way for prefill numerics to
vary between two runs of the same build.
```

## attnHeadsParThreshold

Moved from `decoder/forwardn.go` (the comment above `attnHeadsParThreshold`) on 2026-10-10.

```text
attnHeadsParThreshold gates A1 move (a)'s head-parallel fan-out: below this
many MACs' worth of per-call work (K*nKeys, the QKᵀ/AV size driver), the
fork-join cost isn't worth it and attendBatchedHeads runs its heads serially
through pool[0] instead — the same "small work stays serial" discipline
int4ParThreshold and aikit's own parThreshold already apply one level down
(docs/task-attention-decode-cost.md's Gate A0 item 2 finding: a second
confirmed instance of that bug class, here avoided rather than repeated).
Measured, not guessed — see the campaign doc's move (a) writeup.
```

## attnGroupedNEONSize

Moved from `decoder/forwardn.go` (the comment above `attnGroupedNEONSize`) on 2026-10-10.

```text
attnGroupedNEONSize is the query-head group size aikit's grouped acc64
kernels (MatmulQKAcc64Group/MatmulAVAcc64Group) have a NEON port for — the
1.5B's real GQA ratio (NumHeads/NumKVHeads), R13's registered decision
shape (docs/tasks/red-october.md, aikit linalg commit af926e3). Any other
group ratio still runs correctly through the same Group functions (their
Go fallback is the oracle every G is gated against), but slower than G
separate per-head calls — so attendGroupedHeads is gated to EXACTLY this
size, never called for a model whose group ratio differs, rather than
trusting the kernel's own internal G-dispatch to save a mismatched caller.
```

## attnGroupedMinKeys

Moved from `decoder/forwardn.go` (the comment above `attnGroupedMinKeys`) on 2026-10-10.

```text
attnGroupedMinKeys is the nWin gate: below this many attended keys, the
per-head path runs unchanged. R13 step 0(iii)'s distinct-bytes probe
(docs/measurements/r13-distinct-bytes-probe-2026-09-19.md) found the
cache already dedups a GQA group's shared reads below roughly K=1024; the
aikit kernel A/B (docs/measurements/r13-neon-kernel-ab-2026-09-20.md)
nonetheless measured a real win from the smallest depth it tested (130),
so 128 — the smallest depth either R13 record measured, not the cache
probe's own crossover — is the conservative floor here, not the
aggressive one.
```

## knobSet.attnGrouped

Moved from `decoder/forwardn.go` (the comment above `knobSet.attnGrouped`) on 2026-10-10.

```text
knobSet.attnGrouped (knobs.go) reports whether R13's grouped-kernel decode path may
run. DEFAULT ON, matching moeExpertMajor's sense (mlp.go). Three earlier
wiring attempts (Arm A full-group ownership, Arm B split, Arm B serial)
all measured slower once profiled against a real checkpoint — traced,
via `go tool trace`'s per-goroutine breakdown (plain CPU pprof pointed at
the wrong cause first — see the record), to attendGroupedLayer having
accidentally SERIALIZED softmax, which the ungrouped path runs 6-way
parallel. Fixed by splitting softmax the same way QK/AV already are.
Real result: parity at depth 2048 (~1% overhead, noise-level), a genuine
1.32x served speedup at depth 8192 (docs/measurements/
r13-served-decode-2026-09-20.md) — matching the depth-dependence R13's
own step 0 predicted. GOINFER_ATTN_GROUPED=0 restores the per-head path.
```

## Model.canBatchN

Moved from `decoder/forwardn.go` (the comment above `Model.canBatchN`) on 2026-10-10.

```text
canBatchN reports whether the batched M=K path applies: the gated-MLP families
(Qwen / Llama / Gemma) AND standard sparse-MoE (Mellum / Mixtral) with K>1 —
their attention is plain GQA softmax, so the SIMD attendBatchedHeads applies
(the L² hotspot: a profile put scalar attendQuery at ~83% of MoE prefill). The
MoE FFN itself stays per-row (router picks different experts per token).
GPT-2 (non-gated + learned positions) and K≤1 take the sequential fallback.
```

## Model.canBatchN.ownForward

Moved from `decoder/forwardn.go` (the comment above `Model.canBatchN.ownForward`) on 2026-10-10.

```text
Every family with its own sequential forward is excluded, and the exclusion is DERIVED from
the dispatch table rather than restated here. It used to be restated, and the copy fell one
family behind: LFM2 dispatched to runLayersLFM2 in runLayers and was absent from this list,
so a 2-token prompt ran the dense attention stack over conv layers that load no q/k/v/o and
panicked in rmsNorm (audit-2026-09-02 C-01). A new family now gets this for free.
```

## Model.specRollbackSafe

Moved from `decoder/forwardn.go` (the comment above `Model.specRollbackSafe`) on 2026-10-10.

```text
specRollbackSafe reports whether speculative decode's rollback — KVCache.TruncateTo
after a partial accept — correctly restores this model's state. True for softmax /
GQA (truncate the appended K/V) and MLA (reslice the latent KV) — both live in the
cache, so a verified-then-rejected draft block leaves no residue. FALSE for every
family hasRecurrentState below marks Recurrent — not just the two most obvious
examples, Mamba-2 (granite / nemotron_h) and Gated DeltaNet (qwen3_5_moe), whose
rolling state mamba2Step / the delta scan mutate IN PLACE, but also LFM2's conv
window and KDA (bailing_hybrid) — and TruncateTo does NOT roll back any of them (it
only reslices KV layouts). Verifying a K-token block over-advances that state, and
the next round decodes from it: a silent distribution bug, not a crash (00-core §6;
N-10). Those families need the
checkpoint-at-block-start / restore path (not yet built); until then the n-gram
speculative entry points refuse them and the caller falls back to plain decode.
hasRecurrentState reports whether this model carries state that is mutated IN PLACE per token —
a conv window, an SSM state, a linear-attention state — which no positional rewind can restore.
It is the arch-side view of KVCache.hasRecurrentState(), read from the dispatch table's Recurrent
bit so it can be asked BEFORE any cache exists. That matters for the resident path, which has no
KVCache to interrogate.

ONE PREDICATE, READ EVERYWHERE, because the family list has now been missed once per consumer.
This expression used to be inlined in specRollbackSafe alone; decoder/resident_reuse.go needed
the same question and the same answer, and a second hand-written copy is exactly how LFM2's conv
window came to be absent from every site that should have named it (audit-2026-09-02 C-02). A new
state kind is added to the dispatch table once and every caller here follows.
```

## Model.specRollbackSafe.window

Moved from `decoder/forwardn.go` (the comment above `Model.specRollbackSafe.window`) on 2026-10-10.

```text
C1/C-04: a STAGED sliding-window cache stores local layers in physical rings. Once a ring
wraps (context > window), a rollback of >1 position can't restore the evicted positions, so
the verify reads stale history and diverges — the "lossless" guarantee broken for the families
rings serve (Gemma-3 local / Mistral / Phi-3). The earlier exemption keyed on m.resident==nil,
assuming a resident backend means the positional resident path is taken — but three of four
speculative loops at the time (EAGLE, since removed; grammar always; n-gram whenever a Session drives the staged cache)
use the staged ring EVEN when m.resident!=nil, and a resident CAS loss also falls back to
staged mid-flight. That misjudged path made the predicate return "safe" for a cache that wraps.
Refuse windowed models for speculation unconditionally: the resident positional path is itself
safe, but refusing costs little (→ plain decode) and closes the hole at the source. With windowed models refused here, the staged rollback sites
(KVCache.TruncateTo) only ever run on ring-free caches, where TruncateTo is always exact — so
no inexact case reaches them; re-enabling windowed speculation later must consume that exact
bool at each rollback site (audit C-04).
```

## Model.runLayersFromEmbedN.fastAttn

Moved from `decoder/forwardn.go` (the comment above `Model.runLayersFromEmbedN.fastAttn`) on 2026-10-10.

```text
A3 (G24): OPT-IN f32 attention for prefill. Off unless asked for — and as of
2026-08-29, available to MoE as well, which it was not.

The acc64 path is 8.14x slower than f32 at long-context shapes (measured,
docs/measurements/attention-a3-kernel-ratio-2026-08-26.md — note that is
more than double the "~3.7x" the kernel comment assumes), and attention is
~70% of an 8k dense prefill and 97.1% of an 8k MoE one ON A 4-LAYER SLICE —
which OVERSTATES the full model, see the speedup note below. It exists to hold three
guarantees, and enabling this gives up two of them for the model that enables
it: spec-decode verify == sequential greedy, and decode == prefill.

THE THIRD GUARANTEE — MoE router stability — USED TO EXCLUDE MoE OUTRIGHT, on
the argument that an f32 QK reassociation flips a top-k expert at a near-tie and
cascades. That argument was never measured; it is now, and it is half right.
The mechanism is REAL: at 28 layers, 14.5% of moeMLP calls select a different
expert set, and removing the routing term recovers 70.1% of the divergence. What
is NOT supported is the categorical refusal:

MATCHED on both depth and prompt length — 28 layers and K=2048 on each side,
which took three tries to get right; the two earlier pairings were matched on
one axis each and disagreed about the SIGN:

	1 - cosine   dense qwen2.5-coder-1.5b  2.352e-3
	             MoE   Mellum2             2.126e-3   (0.90x dense)
	greedy continuation, 48 tokens          IDENTICAL, 48/48

THAT 0.90x IS A K=2048 STATEMENT AND DOES NOT SURVIVE TO LONG CONTEXT. At
K=8192 on the full model MoE is 2.777e-3 against dense's recorded 2.400e-3 —
about 1.16x, i.e. marginally WORSE, and that dense figure is cross-session so
the direction is not load-bearing. Both remain ~4x inside the >= 0.99 bar
(1e-2), which is what the decision rests on; "MoE diverges less than dense" is
not something to repeat unqualified.

So the case the flag forbade diverges slightly LESS than the case it permits,
and never reaches the output at all. Both sit ~4x inside the >= 0.99 bar.
Refusing one while shipping the other was not a defensible line. Record:
docs/measurements/mellum2-moe-prefill-split-RESULT.md.

WHAT THE EVIDENCE COVERS, because it is one family: Mellum2, 28 layers, 21 of
them sliding-attention at window 1024 — which CAPS nKeys, and so caps how much
reassociation error a layer can accumulate. A full-attention MoE stresses this
harder and is unmeasured. Do not read the numbers above as "MoE in general".

WHAT IT ACTUALLY BUYS, measured on the FULL 28-layer model at K=8192 — both
earlier figures came from configurations nobody runs (3.11x on a 4-layer slice
at K=8192, 1.08x on the full model at K=2048):

	Ryzen 3700X  int8int8  8411.6s -> 5540.1s   1.52x
	M1 Pro       int4      3935.2s -> 2480.5s   1.59x  (paged; ratio corroborates)

~1.5x, not 3.11x. Two architectures, two quants and very different memory
conditions agreeing within 0.07x is better evidence than either alone. The
slice almost certainly overstated because its 1.6 GB of weights fit in cache,
so weight matmul was cheap and attention read as 97.1% of the work; the full
model's 6-12 GB is bandwidth-bound, attention's share falls, and an
attention-only swap buys less. That mechanism is UNVERIFIED — confirming it
needs a profile at K=8192 on the full model.

Same shape as --metal-fast-prefill: default off, divergence documented, and
the caller opts in knowingly.
fastAttn is the CALLER's statement that this sweep may diverge. It is a
parameter rather than a global read because the guard has to be structural:
spec-decode verify runs through forwardN and MUST keep acc64, or "verify ==
sequential greedy" silently stops holding. A runtime check could not tell
the two callers apart; a parameter cannot get it wrong.
Both exclusions are applied HERE, at the single point where the decision becomes
arithmetic, rather than at each caller: three call sites pass cpuFastAttention() and a
fourth (speculative verify) passes false, so a per-caller guard would be three chances to
forget and one already-correct site that looks the same.
N-35: THE FLOOR KEYS ON K, THE SUFFIX LENGTH — not on attention work, which is K·nKeys.
A warm 2048+128 suffix therefore does NOT take the fast path (K=128 < 512), even though
it attends over the whole 2176-key prefix and is precisely the shape the divergence note
above cites its 1.32x for. So that measurement describes a case this floor excludes.

Left keyed on K deliberately rather than moved to K·nKeys: the divergence was measured at
the shapes the floor admits, and re-keying would silently extend an accepted output
change to short suffixes whose divergence nobody has measured. Changing it means
measuring at that shape first, which is the audit's own first option.
```

## Model.runLayersFromEmbedN.pool

Moved from `decoder/forwardn.go` (the comment above `Model.runLayersFromEmbedN.pool`) on 2026-10-10.

```text
G16: prefill attention runs its heads in PARALLEL, budget permitting.

A1 deferred this ("no M>1-specific work here") and implemented the deferral
literally, as one pool slot — which forced attendBatchedHeads's serial
branch below. The deferral had a measured cost: CPU prefill sat at ~100% of
one core on a 6-P-core box while the weight matmuls beside it fanned out,
and since serial attention is O(K²) while those matmuls are O(K), attention
took a growing share as prompts got longer.

Nothing about the guarantee changes. A1's constraint permits exactly this —
"Parallelism may only split independent outputs across workers/registers —
heads, ..." — and attendBatchedHeads's own comment records that the nH query
heads are fully independent (disjoint ctx writes, no shared mutable state).
Each worker owns its own scratch slot. Bit-identity is gated by
TestPrefillAttnPoolInvariance, not assumed.

The pool is BUDGETED, not simply maxAttnWorkers: a slot's scores buffer is
K*nKeys floats, quadratic in prompt length, so the worker count falls back
toward serial on long prompts rather than the allocation growing without
bound (prefillAttnWorkers).
P-05 (audit-2026-09-02): useAcc64 and cache.treeMask are both fixed for this whole call (every
layer below reuses the same attnPool with the same useAcc64/cache), so fusedOK
(attendBatchedHeads: !useAcc64 && cache.treeMask == nil) is the same for every layer too —
exactly the promise newHeadWorkerPool's wantFused needs to safely skip vt/scores.
```

## Model.runLayersFromEmbedN.ws

Moved from `decoder/forwardn.go` (the comment above `Model.runLayersFromEmbedN.ws`) on 2026-10-10.

```text
Batch the qkv and gate/up projections (shared activation) so a GPU backend
runs each group as one submit (BatchTiled) instead of per-matmul syncs.

P-04: an unset Workspace keeps aikit's own conservative default (16.78M MACs), tuned for
prefill-scale work — but this same ws also carries small-K forwardN calls (spec verify,
K<=8), whose per-round matmuls sit well under that bar and so ran serial while decode's
identical-shape M=1 matmul, which sets DefaultDecodeParallelThreshold (300K) on its own
Workspace (scratch.go's newDecodeScratch), fans out. Large-K prefill matmuls clear both
thresholds comfortably, so this only changes behavior for the small-K case it was missing.
```

## Model.runLayersFromEmbedN.cancel

Moved from `decoder/forwardn.go` (the comment above `Model.runLayersFromEmbedN.cancel`) on 2026-10-10.

```text
G18: an abandoned client must not leave this loop running. Prefill is where
the time goes (a 3k-token prompt is minutes), and before this check a client
that gave up left a core burning to completion — measured at 47:38 of CPU
with nothing attached, with a retrying harness stacking one such prefill per
retry. Checked per LAYER, not per token: the check is free at this
granularity, but it is NOT instant — the bound is one layer's work, measured
at ~12s for a 3072-token prompt on an M1 Pro (cancel at 300ms, observed at
12.34s; TestPrefillCancelMidFlight logs both). That is the tail to tighten
if it ever matters — per-head inside attendBatchedHeads — not a claim that
cancellation is immediate here.
```

## Model.runLayersFromEmbedN.attn

Moved from `decoder/forwardn.go` (the comment above `Model.runLayersFromEmbedN.attn`) on 2026-10-10.

```text
QKᵀ and scores·V for all K positions, per head, on the SIMD A·Bᵀ kernel
(the L² terms) instead of the scalar per-position attendQuery. f64
accumulation (the `true` acc64 arg) is bit-identical to the sequential
reference decode also runs (causalAttention), so a batched verify reproduces
sequential greedy EXACTLY — required for same-model speculative decoding and
for the MoE top-k router never to cascade. (Was f32 for dense — only cosine
≥0.99, which broke spec parity since f32's reduction is M-dependent.)
Local layers read an assembled [base, startPos+K) window (ring history +
the K new rows in k/v); the ring write is deferred until after the read so
a K>W batch can't evict in-batch history. Global layers read append-forever.
```

## Model.runLayersFromEmbedN.moeFFN

Moved from `decoder/forwardn.go` (the comment above `Model.runLayersFromEmbedN.moeFFN`) on 2026-10-10.

```text
Sparse MoE (Mellum / Mixtral): the router selects different experts per
token, so the FFN isn't batchable across K — run the existing per-token
moeMLP for each row (bit-identical to the sequential path).

THIS RESIDUAL SHRINKS WITH PROMPT LENGTH, and the "~17%" that stood here
was a K≈1k-era figure quoted as if it were constant. It also over-attributed:
it named the expert matmuls, but the profile bucket it came from holds the
q/k/v/o projections too. Measured 2026-08-28 (Mellum2 4-layer slice,
int8int8, M1 Pro, real routing — a constant-id prompt collapses the top-k
and understates this), as a share of prefill work. NOTE THE SLICE: its
weights fit in cache, so these OVERSTATE attention's share on the full
model, where weight matmul is bandwidth-bound:

	K       attention   ALL weight matmul (an UPPER bound on the FFN)
	1024      77.3%       22.7%
	2048      88.8%       11.2%
	4096      93.9%        6.1%
	8192      97.1%        2.9%

So batching this FFN expert-major (docs/tasks/task-moe-streaming.md Lever 4) is not a
compute lever at agentic prompt lengths: an upper bound on what it could
return was measured at 4.6-5.1% at K=1-2k and was NOT RESOLVABLE above
run-to-run spread at K>=4096. Its case has to be made on streaming I/O,
where the same expert is re-fetched per row, and measured there.
Record: docs/measurements/mellum2-moe-prefill-split-RESULT.md.

GLM's dense prefix layers (Experts nil) fall through to the dense FFN below.
P18 (opt-in, GOINFER_MOE_EXPERT_MAJOR=1): run the routed experts
EXPERT-MAJOR in chunks instead of one row at a time, so each expert's
weights are read once per chunk rather than once per token. Refuses
and falls through for the order-dependent cases (test seams, a live
pager, a shared expert) -- see moeMLPBatch. Bit-identical when it
runs: TestMoEExpertMajor_bitIdentical.
```

## Model.attendTileForK

Moved from `decoder/forwardn.go` (the comment above `Model.attendTileForK`) on 2026-10-10.

```text
attendTileFor returns how many query rows attendOneHead may process at once with THIS slot: the
cache-sized tile from attnRowTile, clamped to what the slot's buffers actually hold.

THE TILE IS A PROPERTY OF THE POOL, NOT OF THE CALL, AND THAT WAS THE BUG. forwardLayersN sizes
the pool ONCE from maxKeys = startPos+K, on the premise its own comment states — "nKeys =
startPos+K is the same for every layer in this sweep". It is not. A local (sliding-window) layer
whose ring has wrapped assembles a SHORTER window, nKeys = W-1+K, and attnRowTile is INVERSE in
nKeys: fewer keys, more rows per tile. So the per-layer tile came out LARGER than the qh the slot
was allocated, and the Q gather sliced past its length — `panic: slice bounds out of range`, in a
worker goroutine on the fan-out arm and in the Generate goroutine on the serial one, neither
recovered (audit-2026-09-02 C-04).

Clamping is not a workaround for a sizing mistake; it is the invariant stated in the one place
that can enforce it. The tile is a memory-locality choice and the slot's capacity is the binding
constraint, so the slot is what gets to decide. Every other slot buffer follows from qh: with
kt <= t and nKeys <= maxKeys, scores needs kt*nKeys <= t*maxKeys, ch needs kt*hd <= t*hd, and
kh/vt need nKeys*hd <= maxKeys*hd. Clamp qh and they are all satisfied.

It also covers the hand-built scratch slices in the ring tests, which no pool constructor sizes.
```

## attnWorkerPool

Moved from `decoder/forwardn.go` (the comment above `attnWorkerPool`) on 2026-10-10.

```text
attnWorkerPool is a small set of LONG-LIVED goroutines, parked on a
channel between rounds, shared across every decode step's attention
fan-out (R13 Arm B).

MEASURED, NOT ASSUMED, why this replaced spawning a fresh goroutine per
split (go func(){...}()): a temporary time.Now() instrument around
attendBatchedHeads (GOINFER_ATTN_TIMING_DEBUG=1) showed attention's OWN
wall time — not just the wiring's net effect on total decode — going from
11.5ms/token (ungrouped) to 23.7ms/token (Arm B, spawn-per-split) on
qwen2.5-coder-1.5b at depth 2048: the fork-join overhead alone
outweighed nearly all of the grouped kernel's own savings. Each decoded
token does up to 28 layers x 2 kv heads x 2 phases (QK split, AV split) =
up to 112 fan-outs, each spawning up to 5 fresh goroutines — up to ~560
spawns/token. A persistent pool pays goroutine creation once, at
process start, and every later round is a channel send to an already-
running, already-parked goroutine (a runtime "goready", not a "newproc")
— see docs/measurements/r13-served-decode-2026-09-20.md for the
spawn-per-split numbers this replaced.
```

## runSplitAligned

Moved from `decoder/forwardn.go` (the comment above `runSplitAligned`) on 2026-10-10.

```text
runSplit fans work over the independent-output range [0,n) into up to
`workers` contiguous, non-overlapping slices, running fn(w, lo, hi) once
per non-empty slice — worker 0 inline (matching this file's other
fan-outs, e.g. attendBatchedHeads' own head-range loop), the rest via the
persistent globalAttnWorkerPool. fn must write only to worker w's own
scratch (pool[w]) and to its own disjoint [lo,hi) region of any shared
output — runSplit itself performs no combining, so callers must not
require one (see attendGroupedLayer's own doc for why R13 Arm B never
needs one).
runSplitAligned is runSplit with each slice's width rounded UP to a
multiple of `align`.

MEASURED, NOT ASSUMED, why this matters here specifically: aikit's NEON
grouped kernels (avAcc64GroupBlocks/qkAcc64GroupKeys) process their range
in 8-wide blocks, falling back to the plain Go grouped kernel (itself
6-10x slower than the per-head kernels — see linalg/matmul_group_acc64_
bench_test.go's own A/B, aikit) for whatever doesn't fit a whole block.
hd=128 (this model) has ZERO remainder taken as one call (128/8=16 exact
blocks) — but naive equal-width splitting into 6 pieces of ~22 gives
EVERY worker a ~6-dim remainder, so ~27% of each slice silently takes the
slow path instead of 0% of the whole in the unsplit case. Rounding each
slice up to a multiple of 8 keeps every slice block-aligned (hd=128, 6
workers, align=8 gives slices of 24,24,24,24,24,8 — all exact multiples
of 8) at the cost of very slightly uneven work distribution, which this
file's other block-tiled loops (e.g. attendOneHead's G20 row tile) already
accept for the same reason.
```

## attendBatchedHeads

Moved from `decoder/forwardn.go` (the comment above `attendBatchedHeads`) on 2026-10-10.

```text
attendBatchedHeads computes grouped-query causal attention for K query
positions at once, per head, via the SIMD A·Bᵀ matmul (linalg.MatmulBT)
instead of the scalar per-position attendQuery. The two O(L²) terms — QKᵀ and
scores·V — move off the scalar triple-loops onto the vector kernel, which an
end-to-end prefill profile showed were ~half the forward's CPU time.

Per KV head it gathers K_head [nKeys,hd] and V_headᵀ [hd,nKeys] once (reused
across the GQA group). Per query head: scores[K,nKeys] = Q_head·K_headᵀ; a
scaled, causal/window-masked softmax per row (row i attends to
[WindowStart(startPos+i), startPos+i], masked entries zeroed so they drop out
of the next matmul); then ctx_head[K,hd] = scores·V_head, expressed as
MatmulBT(scores, V_headᵀ); scattered into ctx[K,qDim].

NOT bit-identical to attendQuery: QKᵀ moves from float64 to f32 accumulation
and the matmul reassociates the reduction. Parity is argmax-exact + cosine —
the same standard the GPU residency attention already meets. The softmax exp
stays per-row in float64. Scratch slices (qh:[K*hd], kh:[maxKeys*hd],
vt:[maxKeys*hd], scores:[K*maxKeys], ch:[K*hd]) are caller-owned, reused across
layers.
keys/vals are the contiguous K/V the gather reads, with physical row 0 holding
absolute key position `base`: for a global layer that's cache.Keys(layer) at
base 0; for a local (sliding-window) layer it's an assembled [base, startPos+K)
window (the resident ring history + the K new rows) so the ring's wrap is
invisible here and the math is byte-identical to append-forever. Per-query
masking stays in absolute positions (WindowStart/attendHi) and maps to physical
columns s-base.
```

## attendBatchedHeads.acc64

Moved from `decoder/forwardn.go` (the comment above `attendBatchedHeads.acc64`) on 2026-10-10.

```text
MoE routing is discontinuous: the f32 QKᵀ reassociation (~4.6e-5) flips a
top-k expert at a near-tie and cascades, changing the output. The acc64
kernels accumulate each dot in f64 (bit-identical to the sequential f64
reference), killing that perturbation — slower than f32 but still ≫ the
scalar path. Dense MLPs tolerate the f32 error (cosine ≥0.99).

There is no shared `matmul` variable: the acc64 path calls MatmulQKAcc64 /
MatmulAVAcc64 directly (strided, no gather), and the f32 path is handed its
matmul per head — package-level MatmulBT when the head loop is serial, the
worker's serial Workspace when it is not. A single variable could not
express that, and the one that used to sit here was reachable only from the
f32 branches anyway.
```

## attendBatchedHeads.tile

Moved from `decoder/forwardn.go` (the comment above `attendBatchedHeads.tile`) on 2026-10-10.

```text
G20: walk the query rows in TILES. Every step below is row-wise — the Q
gather, QKᵀ (rows are the leading dimension), the per-row softmax, scores·V
(each row folds over keys independently) and the ctx scatter — so splitting
rows splits INDEPENDENT OUTPUTS, which is what A1's bit-identity constraint
permits. No key-dimension split happens here and none may: that would
re-associate the softmax denominator and the AV fold, the exact thing acc64
exists to prevent.

The point is memory, not speed: scores is tile*nKeys instead of K*nKeys, so
a worker slot stops growing with the square of the prompt and the G16 pool
can still fan out on a long prompt.

`i` indexes the TILE below; `gi` is the global row. Positions and masks must
use `gi` — startPos+gi, treeRowPos[gi], treeMask[gi] — while buffers use `i`.
attendTileFor, not attnRowTile: the slot's capacity binds, and recomputing the tile
from this layer's key count is what panicked a warm windowed session (C-04).
```

## attendBatchedHeads.fused

Moved from `decoder/forwardn.go` (the comment above `attendBatchedHeads.fused`) on 2026-10-10.

```text
P19: the FUSED schedule, when enabled and applicable. It replaces the
whole QKᵀ / softmax / scores·V sequence below for this tile, keeping
the score block resident instead of materializing kt x nKeys. Declines
(and falls through) for acc64, whose bit-identity it would break, and
for tree attention, whose per-(row,column) mask is not the contiguous
[lo,hi] bound this handles. Measured 1.69-1.73x causal over a whole
prefill — see fusedattn.go.
```

## attendBatchedHeads.qk

Moved from `decoder/forwardn.go` (the comment above `attendBatchedHeads.qk`) on 2026-10-10.

```text
QKᵀ: scores[K,nKeys] = Q_head[K,hd] · K_head[nKeys,hd]ᵀ. Acc64 reads
keys DIRECTLY — row stride kvDim (rows are nKeys apart), element
stride 1 (a head's hd floats are contiguous) — skipping a kh gather
entirely. Bit-identical by construction (P1; aikit v1.18.0
MatmulBTAcc64Strided runs the SAME sequential f64 reduction as
MatmulBTAcc64, only b's addressing differs), verified at goinfer's own
stride parameters by TestAttendStrided_matchesGatherReference.

A1 move (b): MatmulQKAcc64 interleaves 8 keys' dot products as 8
concurrent f64 accumulator chains, hiding FMA latency the single-chain
dotF32Acc64 leaves idle (each key's own d-order fold is unchanged, so
this is bit-identical, not just close — docs/task-attention-decode-cost.md).
Measured 4.4x in isolation (both depth 130 and 8192 — a pure latency
fix, not depth-dependent, unlike move (c)'s memory-order fix).
```

## attendBatchedHeads.av

Moved from `decoder/forwardn.go` (the comment above `attendBatchedHeads.av`) on 2026-10-10.

```text
scores·V: ctx_head[K,hd] = scores[K,nKeys] · V_head[nKeys,hd]
                         = MatmulBT(scores, V_headᵀ[hd,nKeys])
Acc64 reads vals DIRECTLY, "as if transposed" — row stride 1 (V's hd
floats are contiguous, and vt's row index IS that offset), element
stride kvDim (vt's column index steps by a whole KV row) — skipping
a vt gather+transpose. Same bit-identity argument as QKᵀ.

A1 move (c): MatmulAVAcc64 reads V rows contiguously (keys-outer,
dims-inner) into hd independent f64 accumulators, instead of
MatmulBTAcc64Strided's dims-outer/keys-inner walk (one cache line
per f64 MAC at kvDim stride). Bit-identical by construction — each
dim's accumulator sees the same key-ascending sequence of adds
either way (docs/task-attention-decode-cost.md, docs/task-decode-
splitkv-attention.md:36's "split the independent axis" principle).
Measured 1.81x at depth 130, 2.39x at depth 8192 (aikit
MatmulAVAcc64_ABBench).

R-17: a one-row tile (every K=1 decode) writes this head's context straight into ctx. MatmulAVAcc64 OVERWRITES its destination, and this head's hd
floats at ctx[t0*qDim+qhead*hd] are contiguous, so the scratch `ch` and the scatter copy are dead work there. A tile of several rows has a
[kt,hd] result that is NOT contiguous in ctx (row stride qDim), so it keeps the scatter.
```

## attendGroupedLayer

Moved from `decoder/forwardn.go` (the comment above `attendGroupedLayer`) on 2026-10-10.

```text
attendGroupedLayer is R13 Arm B: unlike Arm A above (runHeadRange /
attendGroupedHeads), which needs one worker to own a WHOLE kv group's
heads and so collapses worker count to nKV when it fires, this keeps
the full worker pool busy AND uses the efficient G=6 kernel by
splitting WITHIN each kv head's group instead of splitting ACROSS
heads: QK by key range, AV by dim range, softmax serial between them.

MEASURED, NOT ASSUMED, why Arm A alone was not enough: a served
BenchmarkDecodeAtDepth run on qwen2.5-coder-1.5b showed
GOINFER_ATTN_GROUPED on vs off as noise-level identical even AFTER
confirming (via attnGroupedRuns) that Arm A's grouped kernel really
was firing — because forcing one worker to own a whole 6-head group
collapsed the 1.5B's 6-way head parallelism down to nKV=2-way, and
that lost parallelism roughly cancelled the kernel's own per-call
efficiency gain (docs/measurements/r13-served-decode-2026-09-20.md).

BOTH splits here are INDEPENDENT-OUTPUT splits, not reduction splits:
QK's reduction is over d (head dim), so splitting by KEY range only
splits its output columns; AV's reduction is over keys, so splitting
by DIM range only splits ITS output columns. Neither needs any
floating-point combining — each worker's slice lands in a disjoint
region of the shared buffer via a plain byte copy, the exact same
"split the independent axis" argument this file already uses for
splitting by head or query row, one level finer. Bit-identical to the
ungrouped per-head path (and to Arm A) by construction;
TestAttendGroupedLayer_matchesPerHead is the check, not just the
argument.
attendGroupedLayer fans out ONE goroutine per kv head (via the
persistent pool, one join for the whole layer) and has each one call
attendGroupedHeads (Arm A's per-kv-head closure, above) exactly as
written — a single, unsplit MatmulQKAcc64Group/MatmulAVAcc64Group
call pair per kv head, no internal key/dim-range fan-out at all.

MEASURED, NOT ASSUMED, why the internal-split version (QK by key
range, AV by dim range, up to 6-way within EACH kv head) was
abandoned in favor of this simpler design: even after two real fixes
(a persistent worker pool, ruled out as not the cost; then NEON-
block-aligned split boundaries, which recovered a real chunk) it
still measured 486ms/30-steps attention-only at depth 2048 against a
325ms baseline — 1.5x slower, not faster, and fewer split-workers (3,
then 2) made it WORSE (511ms, 549ms), ruling out fork-join/wake-up
overhead scaling with worker count as the remaining story too. Every
internal-split fork-join has a synchronization cost that a plain
single-threaded kernel call inside one already-running goroutine does
not pay at all — this design pays that cost ONCE per kv head per
layer (nKV*NumLayers times) instead of twice per kv head per layer
PLUS the per-slice coordination inside each round. Full numbers:
docs/measurements/r13-served-decode-2026-09-20.md.
```

## attendGroupedLayer.softmax

Moved from `decoder/forwardn.go` (the comment above `attendGroupedLayer.softmax`) on 2026-10-10.

```text
MEASURED, NOT ASSUMED, why this is parallelized now and wasn't
before: attendOneHead's own softmax runs INSIDE whichever worker
owns that head, so the ungrouped path already spreads softmax
across up to 6 goroutines; the first version of this function ran
every kv head's every row's softmax SERIALLY on the one goroutine
that calls attendGroupedLayer. R13 step 0(ii)
(r13-attn-category-split-2026-09-19.md) already measured softmax
at ~21-26% of this model's attention time — serializing something
that used to run 6-way parallel is a real, direct, fully
mechanistic cost, found via `go tool trace`'s goroutine breakdown
(not pprof, which mis-set this investigation's first theory —
see docs/measurements/r13-served-decode-2026-09-20.md): the
calling goroutine's OWN execution time grew by ~755ms over 150
decode steps between the ungrouped and grouped runs, almost
exactly the wall-clock gap between them. Splitting by ROW (one
kv-head's one query-head's softmax) is another independent-
output split — no combining, same argument as the QK/AV splits
above, one level finer.
```

## attendBatchedHeads.f32

Moved from `decoder/forwardn.go` (the comment above `attendBatchedHeads.f32`) on 2026-10-10.

```text
f32 path — the DEFAULT for prefill above fastAttnMinPrompt since
2026-08-31, not the test-only fallback it was written as.

gatherKV fills ws's kh/vt for one kv head. Deterministic: a pure
function of (keys, vals, kvh), so two workers gathering the same kvh
into their own buffers produce identical bytes — which is why the
serial and fan-out arms below are BIT-IDENTICAL, not merely close
(TestAttendF32Fanout_bitIdentical).
```

## attendBatchedHeads.f32fanout

Moved from `decoder/forwardn.go` (the comment above `attendBatchedHeads.f32fanout`) on 2026-10-10.

```text
A3: fan out over QUERY heads, exactly as the acc64 path does. The old
code walked kv-major and reused one gather across a kv group, and its
comment read that sharing as a reason the path "stays single-threaded
(the gather itself is shared, mutable state a concurrent split would
race on)". The sharing is real; the conclusion did not follow. Every
pool slot ALREADY owns a full-size kh/vt pair (prefillAttnWorkers has
budgeted 2*nKeys*hd per slot all along), so a worker gathers into its
own buffers and nothing is shared at all.

MEASURED, and the reason this was worth doing: the claim that the f32
path was single-threaded was checked and came back 1.68x utilization,
not 1.0x — MatmulBT fans out internally over output columns, so the
matmuls were already parallel while the gather, the softmax and the
scatter were not. That left ~58% of the arm serial, which is what
head-level fan-out converts and column-level fan-out cannot reach.
See docs/measurements/a3-f32-attention-fanout-2026-09-01.md.

Heads are assigned in CONTIGUOUS runs so a worker walks whole kv
groups: it re-gathers only when kvh changes, so the total gather count
is at most nKV + workers rather than nH. That is the cost of dropping
the sharing, and it is bounded and small.
```

## attendBatchedHeads.acc64fanout

Moved from `decoder/forwardn.go` (the comment above `attendBatchedHeads.acc64fanout`) on 2026-10-10.

```text
A1 move (a): the acc64 path (the real one — every live caller). The nH
query heads are fully independent (disjoint ctx writes, no shared mutable
state — kh/vt aren't touched on this path at all). Below the measured
fan-out floor, or with only one pool slot / one head, run serially through
pool[0] instead: a fork-join here costs real time the (c)+(b) speedups
already shrank to ~13-14 µs/head at depth 130 — the same "small work stays
serial" discipline int4ParThreshold and aikit's parThreshold apply one
level down (Gate A0 item 2 — this is the SAME bug class, avoided here
rather than repeated a third time).
```

## Model.forwardNAttn

Moved from `decoder/forwardn.go` (the comment above `Model.forwardNAttn`) on 2026-10-10.

```text
forwardNAttn is forwardN with the attention-kernel choice made by the caller.

M-07: the two callers want DIFFERENT answers and shared one. Speculative verify must run the
exact kernel on both arms or its equality argument collapses — that is what `false` is for,
and it is unchanged. But EAGLE (removed 2026-09-24) also PREFILLED the prompt through here, while Generate's
prefillLogits prefills with cpuFastAttention() (default ON, floored at 512 tokens). So the
two produced different KV for the same prompt, and "token-identical to plain greedy" — which
EAGLE's whole contract rests on — stopped holding at temperature 0 for any prompt over the
floor. TestEagleSpecParity uses ~25 tokens, well under it, so nothing caught this.
```

## Model.prefillLogits

Moved from `decoder/forwardn.go` (the comment above `Model.prefillLogits`) on 2026-10-10.

```text
prefillLogits processes the whole prompt and returns the logits at its LAST
position (the seed for the first generated token). On the batched archs it
runs the layers at M=len(prompt) in one pass — each weight streamed once,
reused across all positions (~1.7–2× faster prompt prefill / time-to-first-
token than sequential M=1) — and runs the LM head on the last position ONLY
(the others' logits aren't needed). Falls back to sequential runLayers +
forward otherwise. Bit-identical to the sequential prefill (the seed token is
unchanged). The cache is filled with the whole prompt either way.
```

## causalAttention.acc64

Moved from `decoder/attention.go` (the comment above `causalAttention.acc64`) on 2026-10-10.

```text
4. Append this position's K/V, then attend over the stored history. Route
single-token decode through the SAME attendBatchedHeads kernel (at K=1) the
batched prefill/verify (forwardN) uses, with f64 accumulation (acc64) — so
decode is BIT-IDENTICAL to the batched forward for BOTH dense and MoE.
Same-model speculative decoding requires it: the target's batched verify must
reproduce sequential greedy exactly. The old dense path used the scalar
attendQuery + an f32 (MatmulBT) batched verify, only cosine ≥0.99 — that flipped
~11% of argmaxes (spec output diverged) and left ~7% of speculations rejected
(acceptance 0.93). f32's QKᵀ/AV reduction is M-dependent (K=1 decode ≠ M=K
verify) at every aikit version; f64 is order-independent ⇒ exact. MoE already
used acc64 for the same reason (top-k router stability); dense joins it — the
f64 attention cost buys bit-exact decode==prefill==verify (gate:
TestForwardN_matchesSequential / TestSpeculativeGreedyParity). The three cases
mirror forwardN's: ring window, int8-KV global (dequant to f32 scratch), f32
global (append-forever).
```

## causalAttention.ringDirect

Moved from `decoder/attention.go` (the comment above `causalAttention.ringDirect`) on 2026-10-10.

```text
Local ring layer, f32 (audit R-12): store this token's K/V first, then read the window [base, pos] in place as one contiguous slice. The write
before the read is safe because the slot it takes holds position pos-W, the one row just OUTSIDE the window [pos-W+1, pos] (and an empty slot
before the first wrap). The copy path below moved every resident row into scratch each token (about 32 ms of a 201 ms token at depth 4500 on
Gemma-2-2B, 2.5 ms of 39 on Gemma-3-1B at depth 900, measured here); the attention then sees the same rows in the same order, so it is bit-identical.
```

## applyAttnGate

Moved from `decoder/attention.go` (the comment above `applyAttnGate`) on 2026-10-10.

```text
applyAttnGate applies Laguna's softplus output gating to the attention context
in place, before the output projection:

	gate = softplus(g_proj · h)            // h = post-input_layernorm hidden state
	ctx *= gate                            // per-head (broadcast) or per-element

TWO PARITY DETAILS ARE LOAD-BEARING, both mirroring modeling_laguna.py:

 1. softplus is computed in FLOAT32 and the product taken there — the vendor
    writes F.softplus(self.g_proj(hidden_states).float()).to(attn_output.dtype),
    i.e. it deliberately upcasts before the nonlinearity. goinfer's activations are
    already f32, so this is the natural path rather than an extra cast; it is
    called out because a bf16 port of the same code would be wrong.

 2. the gate reads the layer INPUT (post-input_layernorm), NOT the attention
    output. Gating on the attention output is the natural-looking misread and
    would be a different model.

Granularity is read from the WEIGHT's row count, not from config.gating. The
released checkpoints make that necessary: Laguna-XS.2 declares `gating: true`,
which the XS-2.1/M.1 module resolves to per-ELEMENT, yet XS.2's own module
hardcodes nn.Linear(hidden, num_heads) and never reads the field — and its
shipped g_proj is [64, 2048], i.e. per-HEAD. The vendor's spelling→granularity
rule is generation-specific; the tensor shape is not. nH and nH*hd can never
collide (hd > 1), so the shape is unambiguous. arch.laguna.GatePerHead records
what the CONFIG declared and is used only to flag a mismatch at load.
```

## applyGateRow

Moved from `decoder/attention.go` (the comment above `applyGateRow`) on 2026-10-10.

```text
applyGateRow multiplies ONE position's attention context by its softplus gate,
in place. It is the single home for the gate math: causalAttention calls it at
K=1 and the batched forward calls it per row, so the two paths cannot drift.
Keeping them separate is exactly how the gate came to be applied on the decode
path but not in batched prefill — which reads as a plausible 0.957 cosine
rather than as a crash.

perHead ⇒ gates has nH entries, one per head, broadcast across that head's hd
channels; otherwise gates has nH*hd entries, one per channel.
```

## moeSelTrace

Moved from `decoder/mlp.go` (the comment above `moeSelTrace`) on 2026-10-10.

```text
moeSelTrace — when non-nil, records the top-k expert indices of every moeMLP call
in forward order. SPIKE instrumentation for the #2 (MoE expert demand-paging)
viability measurement (docs/ideas-weight-memory.md): de-interleave by NumLayers to
get per-(layer, token) selections, then simulate LRU hit rate. Off (nil) in
production — a single nil-check per MoE FFN, zero allocation. Set by the spike test.
```

## moeWtsTrace

Moved from `decoder/mlp.go` (the comment above `moeWtsTrace`) on 2026-10-10.

```text
moeWtsTrace mirrors moeSelTrace for the per-call routing WEIGHTS; together they
capture a forward's full routing decision. moeSelOverride/moeWtsOverride — when
non-nil — force each moeMLP call to REPLAY a recorded routing (idx+wts from a
higher-precision reference run) in forward order instead of routing on its own
hidden. The precision-localization experiment (E2, decoder/ssm_precision_localize_test.go):
does feeding the f32-SSM forward the f64 reference's routing recover quality
(→ a cheap router-selection island suffices) or not (→ the whole SSM needs precision)?
```

## moeMLP

Moved from `decoder/mlp.go` (the comment above `moeMLP`) on 2026-10-10.

```text
moeMLP runs a sparse mixture-of-experts FFN (Mixtral). The router scores all
experts; the top-k by softmax probability run as gated SwiGLU MLPs and their
outputs combine weighted by the (optionally renormalized) router weights:

	probs   = softmax(Router·h)              // over all NumExperts
	(w, e)  = topk(probs, TopK)              // weights + expert indices
	if NormTopKProb { w /= sum(w) }          // Mixtral renormalizes
	out     = Σ_j w[j] · expert_{e[j]}(h)    // expert = down(silu(gate(h)) ⊙ up(h))

Only the chosen experts are evaluated — the point of MoE.

scr, when non-nil, backs the router-logits/accumulator/expert-gate-up buffers with
per-stream scratch instead of allocating them fresh — the dominant share of MoE
decode's per-token allocation (P8). The single-token decode call sites always pass
their cache's scr; the batched-prefill call site (forwardn.go) has no cache.scr in
scope and passes nil, falling back to the original per-call allocation (amortized
over the K-token batch, not the flagged decode hot path). routeExperts/topK's own
small (NumExperts/TopK-sized) internal allocations are untouched — negligible next
to the hidden/intermediate-sized buffers below.
```

## activationFanoutThreshold

Moved from `decoder/mlp.go` (the comment above `activationFanoutThreshold`) on 2026-10-10.

```text
activationFanoutThreshold gates parallelElementwise's fan-out: below this element count, the
fork/join (goroutine-wake stagger) costs more than a short shard's work, so the range runs
serially instead. MEASURED on this Mac (arm64, 6 P-cores, `silubench` — a standalone serial-vs-
6-worker fan-out microbenchmark over the exact silu(gate)*up computation, arms alternating
rep-by-rep, median of 51 reps): parallel is a clear LOSS at 1024-2048 elements (0.79x-0.87x),
roughly break-even at 4096 (1.04x — the ambiguous zone), and a clean win from 8192 onward
(1.53x, climbing to ~5x by 1M elements). 8192 sits safely past the ambiguous band rather than
riding its edge. This is LOWER than S-06's own attention-fanout stagger estimate (~92us at six
workers) would suggest for a "decode-sized" (~8960-element) call — because this fan-out's
per-worker setup is a bare closure over a slice range, not headWorkerPool's per-worker
gather-then-matmul; a cheaper goroutine body has a cheaper stagger. Re-measure with silubench
before changing this constant; do not guess a new value from the attention pool's own number.
```

## moeExpertMajorChunk

Moved from `decoder/mlp.go` (the comment above `moeExpertMajorChunk`) on 2026-10-10.

```text
---------------------------------------------------------------------------
P18 — expert-major MoE prefill batching.

moeMLP runs one row at a time and swiGLUExpert issues its three matmuls at
M=1, so an expert's weights are re-read for every token that routes to it. At
K=8192 that is ~10^3 re-reads per expert per layer. Measured at real Mellum2
shapes with locality already perfect, converting M=1 -> M=N is worth 1.32x at
N=8 rising to 1.67x at N=256 and still falling
(docs/measurements/moe-expert-batching-m1-vs-mn-2026-09-01.md).

BIT-IDENTITY IS THE CONSTRAINT, AND IT IS ACHIEVABLE. Two things must hold:

 1. the matmuls must be M-invariant. linalg.MatmulBT documents this as a
    contract ("a row computed alone (M=1) equals the same row computed inside
    a batch"), and weightmat.go says the same of the int4 W4A8 kernel, so a
    row's expert output does not depend on how many rows shared the call.
 2. the per-row ACCUMULATION ORDER must be preserved. moeMLP folds the k
    experts as `for j, e := range idx { out += wts[j]*expOut }` -- in ROUTING
    RANK order. Float addition is not associative, so summing expert-major
    would change the result. This computes every (row, rank) expert output
    first and then folds each row in rank order, which keeps the sequence
    identical.

(2) is why this holds a [rows][k][hidden] buffer and therefore why it runs in
CHUNKS rather than over the whole prompt: at K=8192, k=8, hidden=2304 that
would be 604 MB.
```

## knobSet.moeExpertMajor

Moved from `decoder/mlp.go` (the comment above `knobSet.moeExpertMajor`) on 2026-10-10.

```text
moeExpertMajor reports whether the expert-major prefill path is enabled.

DEFAULT ON since 2026-09-01. Measured end to end on the full 28-layer Mellum2
at K=4096, paired and interleaved: 4.364x (1206.9s -> 276.6s), reproduced at
4.50x on the second pair. The pre-registered bar was 15%; this clears it by
more than twenty times.

It is BIT-IDENTICAL, which is why this default flip needs no golden change, no
documented divergence and no user-facing flag -- unlike --cpu-fast-attention,
this changes speed and nothing else. TestMoEExpertMajor_bitIdentical asserts
equality on every logit through the real forward at K=600 and K=4096, is
mutation-proven (reverse-order folding reddens 1381871/1382400 logits), and
asserts non-vacuity via a chunk counter so a silent refusal cannot pass as a
green.

GOINFER_MOE_EXPERT_MAJOR=0 restores the per-row path. Kept as an escape hatch
and an A/B handle, not as a user setting.

What the win is NOT: per-row allocation. moeMLP allocates ~5 slices per row
per layer here and the K=8192 profile recorded 339,293 GCs / 20.9 GB, so that
was the obvious explanation -- and reusing one scratch across the row loop,
measured as its own arm, is worth 0.99x and 1.02x. The mechanism is the
restructuring itself; the decomposition of why is measured as unexplained
rather than asserted. See docs/measurements/p18-expert-major-e2e-2026-09-01.md.
```

## moeMLPBatch

Moved from `decoder/mlp.go` (the comment above `moeMLPBatch`) on 2026-10-10.

```text
moeMLPBatch runs the MoE FFN over `rows` ([n, hidden]) expert-major, writing
n*hidden results into dst. Bit-identical to calling moeMLP per row.

It refuses (returns false) for the cases whose observable behaviour is
ORDER-dependent rather than value-dependent, because those cannot be made
identical by preserving the accumulation order alone:
  - moeSelOverride / moeSelTrace: test seams keyed on per-call forward order.
  - a live pager: `touch` order is the demand signal that drives eviction, so
    reordering it changes which experts are resident. Expert-major is very
    likely BETTER for paging, but "different" is not "better" until measured,
    and this item is about compute.
```

## moeMLPBatch.router

Moved from `decoder/mlp.go` (the comment above `moeMLPBatch.router`) on 2026-10-10.

```text
Router for the whole chunk in ONE matmul (M=n). This is separately worth
something: the profile put the per-row router matmul at 22.4 s and the
per-row `make([]float32, nE)` at 46.2 s of a 1443 s moeMLP.
```

## dequantHeads

Moved from `decoder/kvcache.go` (the comment above `dequantHeads`) on 2026-10-10.

```text
dequantHeads reconstructs nKV int8 head-rows (each headDim wide, scale per
head) back to f32 in dst — the inverse of quantizeHeads, for the batched
prefill's dequant-into-scratch (the f32 matmul kernels stay unchanged).
dequantHeads expands one position's int8 KV into f32, one scale per KV head. The arithmetic
lives in aikit — this is the argument order goinfer's nine call sites use, nothing more.

It WAS a duplicate of linalg.DequantizeRowsInt8Into (rows=nKV, cols=headDim, per-row scale), and
the duplication was removed rather than assumed: the two were gated RAW-BIT equal
(math.Float32bits, not a tolerance) over int8's full range including -128, per-head scales
spanning max-normal / min-normal / denormal / ±0 / Inf / NaN, and the tail shapes the cache
actually uses — then mutation-checked by perturbing one lane and watching the gate go red.
See aikit docs/task-goinfer-kernel-moves.md, M5.
```

## KVCache.rings

Moved from `decoder/kvcache.go` (the comment above `KVCache.rings`) on 2026-10-10.

```text
rings holds a fixed-W ring buffer for each sliding-window (local) layer;
nil entry = a global layer that append-forevers into keys/vals above. Set by
enableRings (from NewCache) for the families that take the attendQuery /
attendBatchedHeads paths — full-attention families (no local layers),
gemma4, and qwen3_5_moe keep append-forever (rings all nil). A local layer
stores only the W most recent positions (the only ones any future query can
read), so its KV is O(W) not O(context). See docs/completed/task-kv-ring-eviction.md.
```

## KVCache.mlaLatent

Moved from `decoder/kvcache.go` (the comment above `KVCache.mlaLatent`) on 2026-10-10.

```text
mlaLatent holds DeepSeek MLA's compressed-KV latent per layer, appended
[pos*latentDim] where latentDim = kv_lora_rank + qk_rope_head_dim. This is
the whole point of MLA: cache the low-rank latent (~576 floats/token), not a
reconstructed full K+V (~41k). forward_deepseek rebuilds per-head K/V from it
each step. nil slice on every other family. latentDim is the per-position
stride (learned on the first append). The standard keys/vals stay empty.
```

## KVCache.captureLayers

Moved from `decoder/kvcache.go` (the comment above `KVCache.captureLayers`) on 2026-10-10.

```text
captureLayers, when non-nil, requests that runLayersFromEmbed copy the residual
stream (the layer OUTPUT) after each listed layer index into captured[i] — the
read-only hidden-state seam a draft head reads (05's EAGLE-3 head fused low/mid/high
target states until its removal on 2026-09-24; block drafters read it now). nil = no capture, zero overhead. The
copies never feed back into the forward, so the token output is byte-identical.
```

## KVCache.treeRowPos

Moved from `decoder/kvcache.go` (the comment above `KVCache.treeRowPos`) on 2026-10-10.

```text
treeRowPos / treeMask, when non-nil, switch the batched verify (forwardN) from a
linear causal chain to TREE attention (05 EAGLE tree drafting; nothing sets these since that
head was removed on 2026-09-24 — kept as tested plumbing): row i takes its
RoPE position from treeRowPos[i] (its depth) instead of startPos+i, and among the
K new batch keys it attends only to columns j with treeMask[i][j] true (its
ancestor path, including itself) plus the whole committed prefix. nil = the
ordinary linear behavior, byte-identical (zero overhead on the common path).
```

## NewKVCache

Moved from `decoder/kvcache.go` (the comment above `NewKVCache`) on 2026-10-10.

```text
NewKVCache allocates an empty cache for a model with the given geometry.
capHint pre-sizes the per-layer slices to avoid reallocation during a
known-length generation; 0 is fine (grow on demand).

skipLayer (nil-safe: nil ⇒ no layer skipped) reports whether layer l never writes
c.keys[l]/c.vals[l] at all — a linear/mamba/conv mixer layer, or one of Nemotron's own
non-attention block kinds — in which case its capHint reservation is wasted capacity the
layer's forward pass never touches (P-02, docs/audit-2026-09-10.md): 100-200 MB+ per stream at
realistic context lengths on a hybrid family's non-attention layers, multiplied by however many
sessions the LRU keeps warm. This is independent of, and layered on top of, the caller's own
whole-cache decision to pass capHint=0 for an MLA family (model.go's own P-02 comment) — an MLA
layer DOES hold attention-shaped KV, just in the separate c.mlaLatent array skipLayer knows
nothing about, so that case is handled by the caller zeroing capHint itself, not by this
parameter.
```

## KVCache.hasRecurrentState

Moved from `decoder/kvcache.go` (the comment above `KVCache.hasRecurrentState`) on 2026-10-10.

```text
TruncateTo drops every stored position at index ≥ pos in all layers and resets
Pos to pos — the rollback speculative decoding needs after a partial accept
(rejected draft positions were appended but aren't real), and the seam prefix
reuse rides on to rewind to a shared prompt prefix (see Session). Cheap: a
reslice that keeps the backing arrays, so re-appending doesn't reallocate. pos
must be in [0, Pos()].

Per-position stride is derived per layer from what the layer actually holds
(len/Pos), not the cache's nominal kvDim, because two Gemma 4 facts break a
uniform stride: per-layer head_dim / KV-head counts make widths differ between
layers, and KV-shared tail layers Append nothing (length 0). Deriving the
stride handles both — a shared layer's stride is simply 0, so it stays empty.
TruncateTo rewinds the cache to hold exactly pos positions and returns whether the rewind was
EXACT. A wrapped sliding-window ring (count>w) rewound by more than one position cannot restore
the dropped positions — its window slots still physically hold them, and attention reads them as
history — so it returns exact=false. Callers reusing a rewound prefix (Session prefix reuse,
speculative rollback) MUST cold-prefill on an inexact rewind or produce silently wrong output
(C1). Global/int8/MLA layers store every position, so they always rewind exactly.
hasRecurrentState reports whether this cache carries state that is mutated IN PLACE per token
and has no per-position history — so TruncateTo cannot rewind it, Snapshot cannot persist it, and
a rolled-back session must go cold rather than warm-reuse it.

THREE KINDS, ONE PREDICATE, BECAUSE THE THIRD WAS INVISIBLE TO ALL OF THEM. Mamba-2 (c.mamba) and
Gated DeltaNet (c.delta) were hand-listed at each site; LFM2's short-conv window (c.conv) is the
same kind of state — mutated in place per token, exactly like the other two windows — and was
named at none of them. resetRecurrent() already cleared c.conv, and was unreachable for an
LFM2-only cache because the guard that calls it did not mention conv. So TruncateTo(0) left the
window intact: conversation B's first K-1 tokens convolved over conversation A's last Bx
vectors, at every conv layer — the cross-conversation leak audit C-01 closed for the other two
kinds — and a partial rewind reported exact=true, so rewindForReuse warm-reused a prefix whose
windows still held the dropped positions (audit-2026-09-02 C-02, audit §0 theme 1).

The fourth kind (KDA) WAS missed here, at every site (audit C-03); recurrent_census_test.go now asks the struct, so a fifth cannot be.
```

## KVCache.resetRecurrent

Moved from `decoder/kvcache.go` (the comment above `KVCache.resetRecurrent`) on 2026-10-10.

```text
resetRecurrent re-zeroes the Mamba-2 / Gated DeltaNet / KDA rolling state (conv window(s) +
SSM/linear-attn state) so a reused cache doesn't leak the prior sequence's recurrence
into a fresh one (audit C-01). No-op on non-recurrent families (nil slices).
```

## KVCache.resetMultimodal

Moved from `decoder/kvcache.go` (the comment above `KVCache.resetMultimodal`) on 2026-10-10.

```text
resetMultimodal clears the per-sequence multimodal state — image attention blocks and
Qwen2.5-VL m-RoPE positions/delta — so a reused cache doesn't leak a prior sequence's
image blocks or m-RoPE offsets into a fresh one (audit M-25). Like the recurrent state,
these have no per-position rewind, so they're only cleared on a full reset. No-op on
text-only caches (nil slices / zero delta). Latent today (every VL generation uses a
fresh cache and warm-KV sessions are text-only), so this makes the reset complete before
VL is ever wired through the warm-KV session path rather than fixing a live leak.
```

## headWorkerScratch

Moved from `decoder/scratch.go` (the comment above `headWorkerScratch`) on 2026-10-10.

```text
headWorkerScratch holds one worker's per-head attention scratch: qh/scores/ch
(+ avAcc, A1 move c) for the acc64 path's independent per-head compute, and
kh/vt for the f32 path's per-kvh K/V gather.

This comment used to say the f32 path was "test-only in practice — every live
caller passes useAcc64=true, so kh/vt sit unused". That stopped being true on
2026-08-31, when f32 prefill attention became the DEFAULT above
fastAttnMinPrompt: kh/vt are now touched on every prefill of a real prompt,
and each worker gathers into its own pair (A3 fan-out). The budget in
prefillAttnWorkers already charged 2*nKeys*hd per slot for them throughout,
so the fan-out needed no new allocation — it needed this sentence to stop
being believed.
```

## prefillAttnScratchBudget

Moved from `decoder/scratch.go` (the comment above `prefillAttnScratchBudget`) on 2026-10-10.

```text
prefillAttnScratchBudget caps the TOTAL per-head scratch a single batched
(prefill) sweep may hold across its worker pool. It exists because the
dominant slot buffer, `scores`, is K*nKeys floats — QUADRATIC in prompt
length. At K=nKeys=3020 one slot is ~42 MB, so an unbudgeted fan-out to
maxAttnWorkers would hold ~255 MB there and ~1.5 GB at an 8k prompt. Trading
quadratic memory for a constant-factor speedup is not a trade this makes
silently, so the worker count falls back toward serial as the prompt grows
rather than the allocation growing without bound.
```

## prefillAttnWorkersK.fused

Moved from `decoder/scratch.go` (the comment above `prefillAttnWorkersK.fused`) on 2026-10-10.

```text
P-05: newHeadWorkerPool allocates a fusedScratch (sBlk+tmp+acc+mRun+lRun+vBlk) ALONGSIDE the
materialized shape above whenever the fused schedule is enabled — "both exist while fusion is
a flag" (newHeadWorkerPool's own comment) — and this budget did not count it, undercounting
real per-slot use by ~25% at K=nKeys=8192 (mostly vBlk, hd*nKeys floats, the same order as
kh/vt). vt itself goes UNUSED once fusion is active (it gathers into fused.vBlk instead), but
removing that allocation needs the caller's useAcc64/treeMask state — unavailable here, and
getting it wrong risks a nil vt in the one case fusion doesn't apply — so this fixes the
measurable complaint (the budget undercounting, causing oversubscription) without touching
what newHeadWorkerPool allocates. Conservative: charged whenever fusion COULD apply, not only
when this specific call will use it, matching the budget's other worst-case assumptions.
```

## decodeScratch.headWorkerPool.fused

Moved from `decoder/scratch.go` (the comment above `decodeScratch.headWorkerPool.fused`) on 2026-10-10.

```text
M-03: ONLY when the caller will actually use it, and GROWN ONCE.

This sat in the per-call slot loop with no guard, so every decoded token allocated
min(nH,6) fused scratches per LAYER — each 4*hd*nKeys bytes plus change — on a path where
ws.fused is dead: decode runs with acc64=true, and attendBatchedHeads computes
fusedOK = !useAcc64 && treeMask == nil, so it never reads them. Since 84e0f13 made
GOINFER_FUSED_ATTENTION default-on, that was the shipped default.

Measured on qwen2.5-coder-0.5b int8int8, interleaved A/B, TotalAlloc per decoded token:

	context 256    11.9 MB/token on  vs  74 KB/token off   (160x)
	context 1024   42.6 MB/token on  vs  74 KB/token off   (573x)

linear in context, all of it zero-filled and churned through GC, and none of it read
(audit-2026-09-02 M-03, measured as P-01 asked).
```

## newHeadWorkerPool

Moved from `decoder/scratch.go` (the comment above `newHeadWorkerPool`) on 2026-10-10.

```text
newHeadWorkerPool builds n (capped at maxAttnWorkers) fresh headWorkerScratch
entries sized for one call's K/nKeys/hd — the batched (M=K>1) forward's
one-shot sibling of decodeScratch.headWorkerPool: that path already
allocates its per-call scratch fresh (no cache.scr to grow-and-reuse across
calls the way per-token decode does), so this mirrors that, not the
grow-once discipline.

wantFused is the caller's promise that EVERY call using this pool will have
fusedOK true (attendBatchedHeads: !useAcc64 && cache.treeMask == nil) — i.e.
useAcc64 and cache.treeMask are fixed for this pool's whole lifetime, not
just true at construction. Under that promise, vt and scores (P-05,
audit-2026-09-02) are skipped whenever fusion is also actually enabled
(knobSet.fusedAttention): vt is unused once fusion is active — gatherKV only
writes it in the non-fused branch — and scores (tile*nKeys, LARGER than
vt's nKeys*hd) is unused whenever the fused path is taken, regardless of
useAcc64, since attendTileFused writes ch directly. `fused`'s own
allocation is intentionally left gated on knobSet.fusedAttention alone, exactly
as before: changing that too would change which arm callers that pass
wantFused=false (because they legitimately mix useAcc64 states against one
pool, e.g. TestA3FanoutUtilization) actually exercise, which is a
correctness/measurement risk this fix does not need to take to close the
audit's claim. A caller that does not hold the promise (mixes useAcc64
states, or a treeMask, across calls to the same pool) must pass false —
getting this wrong risks a nil-slice access in the prefill hot path every
model goes through, exactly the risk the audit's own disposition flagged.
```

## Session.Snapshot.refused

Moved from `decoder/kvsnapshot.go` (the comment above `Session.Snapshot.refused`) on 2026-10-10.

```text
Some families carry recurrent / latent state this format does not persist: qwen3_5_moe's
DeltaNet (c.delta), Granite/Nemotron's Mamba-2 state (c.mamba), LFM2's short-conv window
(c.conv) and DeepSeek/Kimi's MLA compressed-KV latent (c.mlaLatent). Snapshotting any of them
would restore from zeroed/empty state and continue silently wrong — refuse (caller skips →
cold prefill). All other families serialize fully below, incl. ring (windowed) + int8 (C2).

The recurrent kinds come from hasRecurrentState() rather than being re-listed: LFM2 was
missing from this list too, so -session-dir and -kv-idle-demote restored an LFM2 session
"warm" with empty conv windows (audit-2026-09-02 C-02, the C-05 shape).

Since 2026-10-08 the refused kinds come from the cache-state grid (cachestate.go): every kind
whose snapshot cell is "refused" — the four recurrent kinds and MLA as before, plus the
multimodal image blocks and m-RoPE positions, which the format does not carry either.
```

## Model.LoadSession.dims

Moved from `decoder/kvsnapshot.go` (the comment above `Model.LoadSession.dims`) on 2026-10-10.

```text
M17: numLayers/kvDim/pos are blob-controlled and feed m.NewCache(pos), which allocates pos ×
the model's KV footprint — an inflated pos (up to 4B) would makeslice TBs BEFORE the geometry
guard below. Reject implausible header dims, and bound pos by what the body can hold (each of
the pos positions stores ≥1 byte across the numLayers·kvDim KV; division avoids overflow).
M-04: numLayers == 0 and kvDim == 0 USED TO PASS — only negatives and over-maxes were
rejected. They are not merely implausible, they disable the next check: perPos becomes 0,
the `perPos > 0` guard skips the pos bound entirely, and m.NewCache(pos) then allocates
with the MODEL's geometry and the blob's pos (up to 2^31-1). A 20-byte header in
-session-dir was a fatal `runtime: out of memory` at server boot.
```

## Model.LoadSession.geometry

Moved from `decoder/kvsnapshot.go` (the comment above `Model.LoadSession.geometry`) on 2026-10-10.

```text
GEOMETRY FIRST, THEN THE BOUND, THEN THE ALLOCATION (M-04).

This used to allocate m.NewCache(pos) and compare afterwards, so the OOM happened before
the check that would have rejected the blob. And the bound above divided by the BLOB's
numLayers·kvDim while the allocation multiplied by the MODEL's — two different geometries,
so a blob declaring a large per-position footprint could pass a bound it never had to meet.

NewCache(0) derives the same geometry from the arch and allocates no capacity, so the
comparison is free. Once it passes, blob and model geometry are equal by construction and
the bound below is expressed in the units the allocation actually uses.
```

## Model.LoadSession.tokens

Moved from `decoder/kvsnapshot.go` (the comment above `Model.LoadSession.tokens`) on 2026-10-10.

```text
TOKENS BEFORE THE ALLOCATION, AND A CEILING ON THE ALLOCATION ITSELF (2026-09-05).

This used to bound pos by `len(data)/(numLayers·kvDim)`, on the stated premise that "each of
the pos positions stores at least one byte across the numLayers·kvDim KV". THAT PREMISE IS
CONTRADICTED BY THIS FILE'S OWN WRITER: a never-written ring serialises as count/nLive/stride
and then `continue`s, and a KV-shared layer stores nothing, so a WELL-FORMED body can carry
ZERO KV bytes while pos > 0 — the writer says so in as many words ("Empty fields (KV-shared /
never-written rings) serialize as len 0"). The bound therefore rejected valid snapshots, and
not only the hand-built one in TestLoadSession_rejectsCorrupt: a ring layer stores only
min(count, W) rows, so on an ALL-sliding-window model any session longer than ~8·W failed it.

The guard's actual purpose (M17/M-04) is to stop a blob-controlled pos from driving a huge
m.NewCache(pos) — "a 20-byte header was a fatal runtime: out of memory at server boot". Two
checks serve that better, and neither assumes anything about payload sizes:

  1. pos == len(tokens), moved BEFORE the allocation. r.ints() already refuses to allocate
     more than the body holds (`!r.need(n*4)`), so a 20-byte header yields no tokens and is
     rejected here having allocated nothing. For the attack this was written for that is a
     TIGHTER bound than the old ratio, and unlike it, an invariant the format guarantees.
  2. an explicit ceiling on the BYTES the cache would occupy. The thing that OOM'd was the
     allocation, so bound the allocation rather than a proxy that legitimate blobs fail.
```

## availProbeTTL

Moved from `decoder/prefill_budget.go` (the comment above `availProbeTTL`) on 2026-10-10.

```text
availProbeTTL rate-limits the CURRENTLY-AVAILABLE memory probe (P-13, audit-2026-09-10):
hostRAMAvailable shells out (vm_stat on darwin, a /proc/meminfo read on linux) on every call,
and AdmitPrefillMemory runs it on every request that reaches prefill — a fork+exec per chat
request on darwin. Available memory does not need sub-250ms freshness for an admission check;
it changes on the timescale of other processes starting/exiting, not per-request.
```

## Model.AdmitPrefillMemory

Moved from `decoder/prefill_budget.go` (the comment above `Model.AdmitPrefillMemory`) on 2026-10-10.

```text
AdmitPrefillMemory is the request-time counterpart to fitguard.go's load-time guard (R13,
docs/measurements/cold-user-2026-09-07-macbook-arm64.md). The load-time guard prices the WORST
CASE a request could reach — the model's own maximum context — once, at load, and either caps
or refuses on that basis. But an unpinned load whose auto-pinned (or never-needed-a-pin) context
leaves real headroom can still be handed a request whose actual prompt is enormous: a real
agent's system prompt plus its full tool schema, tens of thousands of tokens. That is exactly
what happened on the run that found this — a 7B int4 model the load-time guard rated "79% of
budget" (KV priced at 0, since nothing was pinned) reached 14 GB RSS and swapped a 16 GB Mac
hard on its first opencode request, not on load.

PRICED AGAINST CURRENTLY-AVAILABLE MEMORY, NOT A FRACTION OF TOTAL RAM (R13-follow-on, the same
report's live re-run of this fix). The first version of this function repeated fitguard.go's
load-time shape — a fixed fraction of TOTAL RAM, minus resident weights — which implicitly
assumes nothing else running on the machine ever needs more than the remaining fraction. On the
live re-run, the load-time guard correctly auto-pinned a smaller context and this function
correctly reported every check as fitting — and `serve check`'s own requests still drove 9.7 GB
of swap, because "70% of 16 GB total" was never actually free: other processes on a real,
shared machine were already using more than the remaining 30% assumed available. Weights are
NOT subtracted here (unlike the load-time guard): at request time the model is already
resident, so `HostRAMAvailableBytes` already excludes its footprint by construction — subtracting
it again would double-count. Reads live available memory through cachedHostRAMAvailable (P-13,
audit-2026-09-10: a 250ms TTL cache, not read fresh on every call as this comment used to say —
the raw probe forks+execs on darwin, and every request that reaches prefill was paying for one).

This runs BEFORE prefill begins (internal/serveapp calls it right after `prepare` resolves the
prompt length and clamped max_tokens, for every endpoint that reaches prefill), and prices
KV(promptTokens+maxTokens) plus prefill scratch against what remains of currently-available
memory — never starting a prefill that would page. It runs for CPU and Metal-resident alike:
Metal's own residency guard (metal/backend.go's residentFitsMemory) prices weights only,
against host RAM (Metal's unified memory IS host RAM), and has no per-request check at all — this
is additive to it, not a replacement.
```

## Model.FitBudgetSummary

Moved from `decoder/prefill_budget.go` (the comment above `Model.FitBudgetSummary`) on 2026-10-10.

```text
FitBudgetSummary reports the numbers R13's banner line states at every load: the context KV is
priced at (whatever the load-time guard actually used — the pin, the auto-pinned cap, or the
model's own maximum), KV at that context, the resident weight bytes, and the memory budget.
known=false when availability or the model's config was not readable, matching the guard's own
"unknown ⇒ say nothing" rule — a banner line with half its numbers missing is worse than no
line.

budgetBytes is priced against CURRENTLY AVAILABLE memory (R13-follow-on), read at call time —
by the time this runs the model is already loaded, so weightBytes is ALREADY excluded from
availability by the OS's own accounting. The caller (internal/serveapp/banner.go) must NOT
subtract weightBytes from budgetBytes again when computing what remains — weightBytes is
returned for DISPLAY only, the same "no double-count" rule prefill_budget.go's
AdmitPrefillMemory applies at request time.
```

## Model.FitBudgetSummary.resident

Moved from `decoder/prefill_budget.go` (the comment above `Model.FitBudgetSummary.resident`) on 2026-10-10.

```text
A resident holds its own KV: its capacity, at the precision its backend allocates (Metal: f16 whatever -kv says).
Pricing it at the CPU's per-request ceiling over the model's whole window overstated E2B's Metal KV ~50x (A3,
docs/completed/task-audit-followups-2026-10-06.md).
```

## prefillScratchBytes

Moved from `decoder/prefill_budget.go` (the comment above `prefillScratchBytes`) on 2026-10-10.

```text
prefillScratchBytes is a stated approximation, not a full accounting — the honest scope cut
R13 makes rather than block the admission check on a complete scratch-byte model of every
backend's prefill path. Two terms:

 1. Attention scratch: decoder/scratch.go's prefillAttnScratchBudget (256 MiB) is a REAL,
    already-enforced hard cap — the batched-prefill worker pool throttles its slot count down
    to stay inside it (prefillAttnWorkers), so this term can never be more than what the prefill
    path already allows itself, on any prompt length.
 2. MLP/batched-matmul scratch: the dominant term prefillAttnScratchBudget does NOT cover —
    gate/up activations for a batched (M-token) prefill sweep, sized 2×IntermediateDim×M
    float32 (both buffers, f32 regardless of weight quant — activations are quantized into a
    separate, smaller int8 buffer the matmul path pools and reuses, decoder/weightmat.go's
    matmulWSPool, not counted here because it is pooled/reused rather than sized per-request).
    This is the term this function is honest about NOT having measured: it is a real, derivable
    upper bound from the model's own dimensions, not a number read off a profiler.
```

## deltaNetTiming

Moved from `decoder/deltanet.go` (the comment above `deltaNetTiming`) on 2026-10-10.

```text
deltaNetTiming env-gates the sixth outing of this repo's component-stub timing
method (GOINFER_DELTANET_TIMING=1): splits gatedDeltaNetStep's ~19%-of-decode-token
cost (docs/completed/task-zeno-compare.md's diagnostic) into the three dominant projections
(already W4A8/W8A8-quantized, presumably fast), the delta-rule recurrence proper
(section 3 below — plain scalar Go, the DeltaNet-CPU-recurrence brief's suspect),
and everything else (conv, gates, gated RMSNorm). Atomic accumulators, not
Generate-loop-locals, so concurrent decode streams don't race on them — added,
used to record the split, then reverted, per this repo's own discipline.
```

## deltanet.file

Moved from `decoder/deltanet.go` (the comment above `deltanet.file`) on 2026-10-10.

```text
Gated DeltaNet — the linear-attention primitive of Qwen3.5/3.6-MoE
(qwen3_5_moe). It replaces softmax attention on most layers with a gated
delta-rule recurrence over a fixed-size per-head matrix state, so its memory
is O(1) in sequence length rather than a growing KV cache. See
docs/qwen3_5_moe.md; the math mirrors HF's torch_recurrent_gated_delta_rule +
the surrounding conv / gates / gated-RMSNorm, validated op-for-op against a
traced golden (deltanet_test.go).

This is the parity-first reference implementation: plain f32, sequential over
positions. Perf (a chunked/parallel scan, quantized projections) is a later
track.
```

## deltaNetWeights

Moved from `decoder/deltanet.go` (the comment above `deltaNetWeights`) on 2026-10-10.

```text
THE THREE DOMINANT PROJECTIONS ARE QUANTIZABLE (2026-08-19). They were []float32 —
"parity-first", from the qwen3_5_moe bring-up — which meant a 27.8B Qwen3.8 at Quant:"int4"
still streamed them as f32: 22.1 GB per token across 48 DeltaNet layers, against ~9.5 GB for
the whole int4 FFN. Decode at this size is memory-bandwidth-bound, so that WAS the speed.
WeightMat keeps f32 when the caller asks for no quant (the tiny goldens still match HF
exactly), and carries int8/int4 when they do.
```

## deltaNetWeights.AB

Moved from `decoder/deltanet.go` (the comment above `deltaNetWeights.AB`) on 2026-10-10.

```text
A and B stay f32: [48, 5120] each is ~1 MB per layer (~94 MB total on the 27B) against the
22 GB above, and they feed the write/decay gates, where the recurrence is most sensitive to
precision. Quantizing them would buy ~0.2% of the bytes for real numerical risk.
```

## runLayersGemma4FromEmbed

Moved from `decoder/forward_gemma4.go` (the comment above `runLayersGemma4FromEmbed`) on 2026-10-10.

```text
runLayersGemma4FromEmbed is runLayersGemma4's shared body, parameterized over
an already-built (already-scaled) hidden-state embedding h — runLayersGemma4's
own case is a real token's embedding; a multimodal caller (P7) substitutes a
projected image/video/audio embedding here instead of a token-id lookup, the
"embed-by-vector" seam this family lacked (the June seams — runLayersFromEmbed
/ runLayersFromEmbedN in model.go/forwardn.go — only reach the GENERIC forward
path; gemma4's own-forward never went through them).

pleTokenID selects which token id's per-layer embedding feeds PLE's
token-identity term. For a real text position this is the same id h was
embedded from. For a multimodal position the real HF multimodal forward
substitutes the checkpoint's pad_token_id THERE, before computing PLE — not
the placeholder token's own id, and not a skipped/zeroed term (verified
against modeling_gemma4.py's real multimodal forward path, not assumed — see
docs/multimodal.md's P7 entry). A caller passes arch.gemma4.PadTokenID for an
image/video/audio position.
```

## gemma4AttendRange

Moved from `decoder/forward_gemma4_batched.go` (the comment above `gemma4AttendRange`) on 2026-10-10.

```text
gemma4AttendRange returns the inclusive absolute key range [lo,hi] a query at
pos (on a layer with the given global/window setting) may attend to, given at
most one bidirectional image/audio block [imgPos, imgPos+imgLen) (imgLen<=0
means no block: plain causal/windowed, identical to cache.WindowStart(pos,
global)..pos, the same range the sequential path implicitly uses).

v1 supported exactly one contiguous block per prefill; S11 takes several,
each its own block (below).

PROOF this is always a single interval, never two disjoint ones: for a query
at pos inside block [b0,b1), the causal/windowed interval is
[windowStart(pos), pos] and the block interval is [b0, b1-1]. Since
b0 <= pos <= b1-1 (the query is itself in the block) and windowStart(pos) <=
pos, `pos` is a member of BOTH intervals, so their union is connected:
[min(windowStart(pos), b0), max(pos, b1-1)]. A query not in the block gets
the plain range unchanged (HF's blockwise term requires block[q]>=0 too).
Verified against the real transformers masking_utils.py (create_causal_mask /
create_sliding_window_causal_mask, both apply
or_masks(windowed_causal, blockwise_overlay(block_ids)) unconditionally,
with NO layer-type gate — see docs/multimodal.md's P7 entry for the full
citation and the correction to this doc's own earlier, wrong claim that only
sliding layers get this treatment).

Several blocks (S11, docs/tasks/task-multimodal-support-2026-10.md): a query sees its OWN block only, so the same
proof holds block by block, and a query outside every block gets the plain range.
```

## gemma4MoEFFN

Moved from `decoder/forward_gemma4_moe.go` (the comment above `gemma4MoEFFN`) on 2026-10-10.

```text
gemma4MoEFFN applies the sub-block to one token's post-attention residual h
([hidden]) and returns the layer output ([hidden]). Position-independent, so the
decode loop calls it per token.

Lever 3 (task-moe-streaming.md, "overlap routed reads with the resident branch"):
the router only needs h (not the dense branch's output), and the dense branch only
needs h (not the router's chosen experts) — the two are independent until the join.
So the router runs FIRST (to learn which experts to fetch), the expert fills are
ISSUED, and the dense branch's matmuls run on the calling goroutine WHILE a second
goroutine drives the fills — hiding a cold miss's fault/pread latency behind the
dense branch's own compute time instead of paying both serially. Bit-identical
either way (dense and moe branches touch disjoint memory and are summed
order-independently at the join; TestGemma4MoEFFN_overlapBitIdentical), backend
calls untouched (the fill goroutine never touches `be`, only the pager).
```

## runLayersQwen35FromEmbed

Moved from `decoder/forward_qwen35.go` (the comment above `runLayersQwen35FromEmbed`) on 2026-10-10.

```text
runLayersQwen35FromEmbed is runLayersQwen35's body for a position whose residual-stream
embedding is supplied directly — the image-splice seam (P8a): an image row is a tower feature,
not a table lookup. Same shape as runLayersGemma4FromEmbed. It consumes h (the layers mutate it
in place) and returns it. One token per call, like its wrapper: there is no batched qwen3_5
prefill to mirror (prefill is this same per-token loop, so the DeltaNet recurrence sees every
token), and GDN layers take no position ids, so an image row needs nothing the text row doesn't
except the m-RoPE positions the full-attention layers read from cache.mropePos.
```

## runLayersLlama4

Moved from `decoder/forward_llama4.go` (the comment above `runLayersLlama4`) on 2026-10-10.

```text
Llama 4 (llama4_text) forward path — the iRoPE text decoder. Each block is a standard
Pre2 residual stack (input_layernorm → attention → +residual → post_attention_layernorm →
FFN → +residual), but two things vary PER LAYER:

  - Attention: RoPE layers (no_rope_layers[l]==1) apply interleaved RoPE over the full
    head_dim then a parameter-free L2 (RMS-over-head-dim) QK-norm; NoPE layers skip RoPE
    and instead scale the query by an attention "temperature"
    (log1p(floor((pos+1)/floor_scale))·attn_scale + 1) for length generalization.
  - FFN: dense layers vs MoE layers (moe_layers) — handled by the shared mlp() dispatch
    (dense ⇒ gatedMLP; MoE ⇒ llama4MoE, below: top-1 sigmoid routing + an ungated shared expert).

Chunked (local) attention on the RoPE layers: a query at position p attends only within its
own chunk, [(p/C)*C, p], where C is attention_chunk_size (8192 on Scout/Maverick). Below C
that is identical to full causal, which is what the parity gates exercise — and for a long
time it was ALL this did, because the config field was read and dropped, so from position C
on the RoPE layers saw keys HF masks out (M-05). The bound now comes from
Architecture.attnChunkStart, max()'d with the window start in attendQuery. NoPE layers stay
full-causal. Parity-first f32, one token per call; canBatchN excludes it.
```

## ssmStopLayer

Moved from `decoder/forward_granite.go` (the comment above `ssmStopLayer`) on 2026-10-10.

```text
ssmStopLayer truncates the forward after this layer (-1 = run every layer): the resident-SSM
bring-up's layer-sweep seam. It was the env var GOINFER_SSM_STOP_LAYER until 2026-09-24; only
tests set it now, through SetSSMStopLayerForTest (testhooks.go).
```

## qwen35BatchN

Moved from `decoder/forward_qwen35_n.go` (the comment above `qwen35BatchN`) on 2026-10-10.

```text
The batched Qwen3.5 forward: runLayersQwen35FromEmbed over K rows at once, for PromptHidden (Route B's hidden state,
docs/tasks/task-constrained-confidence.md D6b). The per-token forward reads every weight once per token, which on a
CPU is decode speed and memory-bandwidth bound: JEV-9B's 150-item D6b sample measured ~0.2 s/token at int4 and ~1.2
s/token at f32. Here every projection is one M=K matmul, so a weight is read once per prompt:

  - DeltaNet layers: in_proj_qkv, in_proj_z and out_proj are batched. The conv window, the gates, the recurrence and
    the gated norm stay sequential per token, through deltaNetCore, the same code the per-token step runs.
  - Full-attention layers: q/k/v/o are batched and attention is attendBatchedHeads over the cache, with f64
    accumulation (useAcc64), which its own comment records as bit-identical to the sequential attendQuery.
  - FFN: a dense SwiGLU is batched; an MoE runs moeMLP per row, as the generic batched path does.

It is not claimed bit-identical to the per-token forward: a batched f32 or quantized matmul may reduce in a
different order from its matvec. TestPromptHidden_batchedMatchesSequential bounds the difference.

qwen35BatchN reports whether it applies. Olmo Hybrid (PlainFullAttn, per-layer norm placement), a capture request
and a dense layer pager take the per-token path.
```

## kvDequantParallel

Moved from `decoder/kvdequant_parallel.go` (the comment above `kvDequantParallel`) on 2026-10-10.

```text
Audit R-15. An int8-KV decode token widens every stored K and V row of every global layer back to f32 (dequantGlobalLayer), and widens a ring layer's whole
window the same way. That is O(context) per token per layer, on ONE thread, directly in front of the attention that then fans out across heads: measured on
Qwen2.5 1.5B at depth 2000, 17.4 ms of an 80.5 ms int8-KV token (the f32 token is 59.5 ms), so the dequantization is essentially the whole int8 slowdown.

The widen is elementwise, float32(q) * scale per row with no shared state, so it splits by rows with no change to any bit. It is bandwidth-bound (1 byte read,
4 written per element), so the fan-out is the process's usual width (fusedWorkers: GOMAXPROCS capped by linalg's parallel width) and only kicks in once there is
enough to amortize a fork/join.
```

## prefillRefCache

Moved from `decoder/prefill_ref_cache_testhook.go` (the comment above `prefillRefCache`) on 2026-10-10.

```text
A content-keyed cache for the prefill fidelity gates' CPU f32-activation references (TE6(a) and TE8 of
docs/tasks/task-test-efficiency-2026-09.md). TestPrefillGateReference is the longest single run in the census (up to
168 min), all-or-nothing, and its output's staleness was judged by hand — a stale set had to be renamed aside once,
and set A's 2026-09-05 files were scored against logits for different text (prefill-ref-identity-2026-09-26.md).

Each (model, K, prompt) reference is keyed by everything it depends on:
  - the checkpoint's sha256 (cached by path + size + mtime, so a 4.7 GB file is hashed once);
  - the prompt's own token ids at that K (sha256, the same encoding as internal/fidelity.PromptSetHash);
  - the source the CPU reference path compiles from: every non-test .go file in decoder/, internal/giw/ and
    constrain/ (go list -deps ./decoder, 2026-09-28) plus go.mod, which pins aikit and golang.org/x;
  - runtime.GOARCH (the CPU reference is bit-identical within an arch, not across), the weight quant, the
    continuation length, and the forced exact attention.

A hit is a lookup, not a judgement; a miss is computed and stored atomically, so an interrupted generator resumes
where it stopped. The historical ~/goinfer-logs/prefill-ref[-<set>]/<model>-K<k>-p<i>.bin path is populated from the
cache (a hard link where the filesystem allows, else a copy), with a <file>.key.json sidecar beside it, so every
consumer keeps reading the path it reads today and can check the sidecar's prompt hash exactly.
```

## runLayersDeepseek

Moved from `decoder/forward_deepseek.go` (the comment above `runLayersDeepseek`) on 2026-10-10.

```text
DeepSeek-V2/V3 (deepseek_v2 / deepseek_v3) forward path — Multi-head Latent Attention
over a DeepSeekMoE FFN. MLA is the third efficient-attention coverage axis (latent-KV):
K/V are compressed to a shared low-rank latent (kv_lora_rank), and ONLY that latent (‖ a
per-position rope-carrying key) is cached — ~576 floats/token vs the ~41k a reconstructed
full K+V would need. Per-head K/V are rebuilt from the latent each step (the "naive" path,
bit-identical to HF; the "absorb" optimization that folds kv_b_proj into q/o is a perf
follow-up, not needed for parity). Decoupled RoPE rides on a separate qk_rope_head_dim
slice of Q and the shared latent key; the no-rope dims and the (different-width) V skip it.

The block is a standard Pre2 residual stack — input_layernorm → MLA → +residual →
post_attention_layernorm → MoE/dense → +residual — so the FFN reuses the generic mlp()
dispatch (dense prefix on l < first_k_dense_replace, DeepSeekMoE elsewhere). Parity-first
f32, one token per call (the latent append + causal attend mirror the other own-path
families); canBatchN excludes the MLA attention-kind.
```

## Model.runLayersFromEmbedN.capture

Moved from `decoder/forwardn.go` (the comment the comment above the MoE hidden-state capture in `runLayersFromEmbedN`) on 2026-10-10.

```text
Hidden-state seam (05), same as the dense path below: MoE layers must also
record captured[ci], or a capture against a sparse-MoE target (the since-removed EAGLE path)
(Mixtral/Mellum) leaves captured all-nil and fuseAt slices a nil slice → panic
(audit C-07). The `continue` used to skip this.
```

## attendBatchedHeads.attendOneHead

Moved from `decoder/forwardn.go` (the comment the comment above `attendOneHead` in `attendBatchedHeads`) on 2026-10-10.

```text
attendOneHead runs one query head's QKᵀ → softmax → scores·V → scatter into
ctx, using ws's scratch. A1 move (a): this is what runs concurrently across
heads below — bit-identical regardless of which pool slot or goroutine runs
it, or what order heads finish in, since every head's own math (moves b/c's
unchanged per-output reduction order) and its ctx write (a disjoint qhead*hd
slice — no two heads ever touch the same bytes) are exactly as before.
mm is the f32 matmul this head should use: the package-level MatmulBT
(column-parallel) on the serial arm, or the worker's own serial
Workspace on the head-parallel arm. Unused on the acc64 path, which
calls MatmulQKAcc64/MatmulAVAcc64 directly.
P19: the fused schedule is eligible only on the f32 path (it would break
acc64's bit-identity) and only without a tree mask. Each worker uses ITS OWN
ws.fused — never a shared one, since each gathers a different kv head's V.
```

## attendGroupedHeads

Moved from `decoder/forwardn.go` (the comment above `attendGroupedHeads`) on 2026-10-10.

```text
attendGroupedHeads is R13's grouped-kernel path: one MatmulQKAcc64Group/
MatmulAVAcc64Group call pair covers all attnGroupedNEONSize query heads
of ONE kv head, instead of attnGroupedNEONSize separate attendOneHead
calls each re-reading the same K/V rows. Callable only when the caller
has already checked K==1 (decode; M>1 is not wired — "wire decode
first" per the brief), no tree mask (its per-(row,column) mask needs
attendOneHead's own branch), and group == attnGroupedNEONSize exactly.

Q for these heads at this one position is CONTIGUOUS in q's [K,qDim]
layout (qDim = nH*hd, heads are laid out head-major within a row, and
a kv group's query heads are themselves contiguous head indices) — so
no gather is needed, unlike attendOneHead's per-tile copy. Same for
ctx on the scatter side. Softmax stays PER HEAD, row by row, identical
to attendOneHead's own non-tree branch (same masking, same max/exp/sum/
normalize sequence) — grouping only shares the QKᵀ and scores·V loads/
folds, never the reduction each head's own softmax performs, matching
```

## attendBatchedHeads.attnGroupedOK

Moved from `decoder/forwardn.go` (the comment the comment above `attnGroupedOK` in `attendBatchedHeads`) on 2026-10-10.

```text
attnGroupedOK is this call's eligibility for the grouped path — checked
once per attendBatchedHeads call, not per head, since none of these
depend on qhead. cache.treeMask != nil excludes speculative verify (its
per-(row,column) mask attendGroupedHeads does not implement); K != 1
excludes prefill/batched M>1 (not wired yet).
```

## attendBatchedHeads.runHeadRange

Moved from `decoder/forwardn.go` (the comment the comment above `runHeadRange` in `attendBatchedHeads`) on 2026-10-10.

```text
runHeadRange walks qhead across [h0,h1), taking the grouped path for
any run of attnGroupedNEONSize heads that (a) starts on a kv-group
boundary and (b) fits entirely inside [h0,h1) — i.e. exactly the
"same-KV-head run a worker already owns" the brief's Arm A wiring
names; a worker whose range splits a kv group falls back to
attendOneHead for that group's heads, unchanged from today.
```

## attendBatchedHeads.armB

Moved from `decoder/forwardn.go` (the comment the comment above the Arm B dispatch in `attendBatchedHeads`) on 2026-10-10.

```text
R13 Arm B takes priority over Arm A whenever it can actually help:
with more than one pool slot, splitting WITHIN each kv group (Arm B)
keeps every slot busy where Arm A's "one worker owns a whole group"
rule would strand the rest idle (see attendGroupedLayer's own doc for
the measured reason this matters). With only one slot there is
nothing for Arm B's internal fan-out to parallelize, and the plain
serial arm below already takes the grouped kernel path correctly
(runHeadRange's [0,nH) range always spans whole groups), so Arm B is
skipped rather than adding fork-join overhead for no benefit.
```

## moeMLP.gateUp

Moved from `decoder/mlp.go` (the comment the comment above the gate/up scratch in `moeMLP`) on 2026-10-10.

```text
One gate/up pair for the whole token. The experts run sequentially, so k pairs were never
simultaneously live — this was 2*k allocations per token where 2 suffice, and at top-k 8 with
a large moe_intermediate that is the bulk of moeMLP's per-token allocation.
```

## decodeScratch.ctxBuf

Moved from `decoder/scratch.go` (the comment above `decodeScratch.ctxBuf`) on 2026-10-10.

```text
ctxBuf returns the attention context buffer, length n, from the scratch (R-17): the own-forward families
(Qwen3.5, Granite, Llama 4, LFM2, Nemotron) used to make a fresh zeroed one per layer per token, and attendQuery
clears it before accumulating anyway. The caller consumes it (the o-projection) before the next layer asks again.
```

## newDecodeScratch

Moved from `decoder/scratch.go` (the comment the comment above the Workspace in `newDecodeScratch`) on 2026-10-10.

```text
Note: aikit's opt-in worker pool (Workspace.SetWorkers) is intentionally NOT
used — goinfer's end-to-end sweep showed it neutral-to-slightly-slower than
the spawn path (the batch=1 fork/join cost is a floor, not pool-fixable).

The W8A8 (int8) decode matmuls run through this Workspace (matmulInto), so its
PER-WORKSPACE threshold is what makes int8 decode parallelize — NOT the process
global. Setting it here means every decode stream (library Load, serve, tests)
gets it automatically and race-free, the same way the int4 path self-configures
```

## decodeScratch.headWorkerPool.khvt

Moved from `decoder/scratch.go` (the comment the comment above the kh/vt growth in `decodeScratch.headWorkerPool`) on 2026-10-10.

```text
P-03 (audit-2026-09-10): kh/vt are unused whenever useAcc64 is true — the acc64
kernels (MatmulQKAcc64/MatmulAVAcc64, forwardn.go) read keys/vals directly with
strided addressing, skipping the kh/vt gather entirely. headWorkerPool's only
caller (attention.go's decode path) hardcodes acc64 := true unconditionally, so
this is not a runtime toggle at the one call site that exists today — but the
check is on useAcc64 itself, not assumed, so a future non-acc64 caller still
grows them correctly on its own first call.
```

## forward_gptoss.file

Moved from `decoder/forward_gptoss.go` (the comment the file comment before `runLayersGptOss`) on 2026-10-10.

```text
gpt-oss forward (own path, CPU-only). gpt-oss is a sparse-MoE family whose two
ops diverge from the generic descriptor forward: attention adds a learned
per-head SINK to the softmax denominator (an escape valve that bleeds attention
mass, with no value), and the MoE experts use a clamped interleaved-SwiGLU with
an α-scaled sigmoid, a +1 on the linear branch, and per-expert biases. The layer
skeleton is otherwise plain pre-norm (NormPre2): everything else — embedding,
norms, residuals, final head — is the shared path. Isolating gpt-oss here keeps
the sink/clamped-activation out of the 20 other families' hot softmax/MLP kernels.
Parity-first and arch-neutral (no SIMD): §1 says the capability matters more than
speed on x86, and bench numbers are deferred (docs/completed/task-mxfp4-gptoss.md §6.6).
```

## forward_lfm2.file

Moved from `decoder/forward_lfm2.go` (the comment the file comment before `shortConvStep`) on 2026-10-10.

```text
LFM2 / LFM2.5 forward — one token per call. The caller (forward) applies the final norm
and the tied LM head.

Every layer is Pre2: operator_norm → mixer → residual, then ffn_norm → SwiGLU → residual.
The mixer is a gated short convolution on 22 of 30 layers and GQA softmax attention on the
other 8 (layer_types). Both halves write into the same hidden vector, so the only thing the
```

## lfm2Attention

Moved from `decoder/forward_lfm2.go` (the comment above `lfm2Attention`) on 2026-10-10.

```text
lfm2Attention is GQA + RoPE with per-head RMSNorm on Q and K.

The QK-norm is RMSNorm over head_dim, applied per head BEFORE RoPE — the ordering HF uses
and the one the existing hardcoded QK-norm path already implements, which is why declaring
QKNorm was enough and no new primitive was needed. (The original scoping brief said
LayerNorm; the released checkpoint carries q_layernorm.weight and no bias tensor anywhere,
```

## Model.LoadSession.globalLen

Moved from `decoder/kvsnapshot.go` (the comment the comment in the int8 arm of `Model.LoadSession`) on 2026-10-10.

```text
M-04: BLOB-CONTROLLED LENGTHS, NEVER COMPARED. The ring branch above checks its
stride, nLive and payload; the global branch checked nothing. The forward derives
nKeys from `keys` and then indexes `vals` at the same positions, so a vals array
one row short is an out-of-range read in the generation goroutine — a panic that
takes the process down. The CRC does not help: it covers the attacker's bytes.
```

## Model.LoadSession.tokenCount

Moved from `decoder/kvsnapshot.go` (the comment the comment above the token-count check in `Model.LoadSession`) on 2026-10-10.

```text
M-04: len(tokens) is blob-controlled and was never compared with pos. Too MANY tokens is
the quiet one — rewindForReuse computes matched > c.pos, TruncateTo treats an
out-of-range target as a no-op, and the reuse reports an exact match on a cache that was
never rewound. No panic, no error: a session that silently continues from the wrong KV.
```

## ring.truncate

Moved from `decoder/kvcache.go` (the comment above `ring.truncate`) on 2026-10-10.

```text
truncate drops logical positions ≥ p. Returns true iff the result is exact —
i.e. every position the post-truncation window [max(0,p-w), p) is still
physically resident. That holds whenever the ring never wrapped (count ≤ w);
a deeper rewind on a wrapped ring would need positions already evicted, so it
returns false and the caller must cold-prefill (the rewind rule, wired into
sessions in Increment 2). Spec-decode draft depths ≪ w on short contexts never
wrap, so they stay exact.
```

## KVCache.TruncateTo.recurrent

Moved from `decoder/kvcache.go` (the comment the comment above the recurrent reset in `KVCache.TruncateTo`) on 2026-10-10.

```text
Recurrent state is a single rolling state with no per-position history, so it cannot be
exactly rewound (audit C-01). Reset it on a full clear (Session.Reset → TruncateTo(0), and
sessionLRU.fresh), and report inexact on any rewind so rewindForReuse cold-prefills rather
than decoding a new sequence from the previous one's leaked state.
```

## runLayersQwen35

Moved from `decoder/forward_qwen35.go` (the comment above `runLayersQwen35`) on 2026-10-10.

```text
Qwen3.5/3.6-MoE (qwen3_5_moe) forward path — the hybrid: most layers are Gated
DeltaNet (linear attention, recurrent state in the cache), the rest gated
softmax attention (KV cache), every layer a routed+shared MoE. Parity-first
f32, allocate-per-call, mirroring runLayersGemma4 (perf is a later track). One
token per call; the caller (forward) applies the final norm + LM head, and
prefill drives this sequentially so the DeltaNet recurrence sees every token.
See docs/qwen3_5_moe.md.
```
