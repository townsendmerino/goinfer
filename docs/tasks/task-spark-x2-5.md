# Task — add Spark-X2.5 (`spark2_5`)

**Status:** SHIPPED 2026-09-21 (1.7B; the 4B was skipped — see Results). Filed 2026-09-21.
**Venue:** any — a 4B dense fits every rig and every backend.
**Scoped in:** `docs/audit-2026-09-10.md` L-06 (:463) and recommendation 7 (:595). This doc is
that item made actionable; it does not re-scope it.

---

## Results (2026-09-21)

**Two of this doc's own "we already have" claims, and L-06's, were wrong** — found by reading the
real `configuration_spark.py`/`modeling_spark.py` directly (fetched from `XHToken/Spark-X2.5-4B`,
trust_remote_code) before writing any code, exactly the verification step this doc itself asked
for ("verify each 'we already have' claim against current code — that audit is eleven days old and
the tree has moved"):

- **Not Cohere's parallel block.** Spark-X2.5's decoder layer is the standard sequential residual
  (norm→attn→residual, norm→mlp→residual), identical in shape to Llama/Qwen. No parallel-block
  wiring was needed at all — this removed what looked like the largest scoped item.
- **Not "non-gated GELU."** The MLP is gated (SwiGLU-shaped: `down(gelu(gate(x))*up(x))`); the real
  departure is the activation inside that gated shape — exact erf GELU (HF's `"gelu"`), not
  SiLU or GELU-tanh. `ActGelu` had previously only ever reached the NON-gated MLP path
  (Nemotron-H); `gatedMLP`'s switch had no case for it, and a new `gegluExact` kernel
  (`decoder/mlp.go`) was needed.

**What was real, and net-new:** the fused-QKV split (`buildSpark25Weights`, modeled on
`buildPhi3Weights`'s split, but with separate not-fused gate/up tensors and different tensor
names — `model.embedding.weight`, `self_attn.out_proj.weight`); a generic (non-MLA) sigmoid
attention-output gate (`Architecture.AttnGate`/`GateSigmoid`) — the sigmoid math
(`applySigmoidGateRow`) already existed for MLA/Bailing Hybrid but had no hook in the plain-GQA
forward path this family needs, so this was a real new wire in `attention.go`/`forwardn.go`, not
just an enum flip; and `gegluExact`. Per-layer-type partial RoPE and the 1:3 sliding:full
interleave were fully reusable as-is (Laguna's `RotaryDim`/`RotaryDimLocal`/`RoPEGlobalBase`/
`RoPELocalBase` mechanism, and the generic `layerIsGlobal`/`layer_types` machinery already shared
by Cohere2/Mellum/Gemma/etc).

**Gate 1** (tiny synthetic, `scripts/pin_spark2_5_tiny.py`, `decoder/spark25_test.go`): PASS,
cosine 1.00000000, maxAbs 1.79e-7, full greedy continuation match against the real
`modeling_spark.py`.

**A real gap none of the three gates caught, found by an unrelated CI check instead.**
`decoder/forwardn.go`'s batched-prefill path has its OWN copy of the gated-MLP activation
switch, separate from `decoder/mlp.go`'s (the same "decode path fixed, batched prefill wasn't"
class of bug this repo's own culture names explicitly). `gegluExact`/`ActGelu` was added to the
decode-path switch but not this one, so it fell to `default: return nil, errNotImplemented` —
meaning ANY multi-token prompt through the real `Generate()` entry point would have crashed
immediately. Gate 1/2/3's own tests never caught this because all three drive `m.forward()` in a
manual per-token loop, never `Generate()` itself — a real test-through-the-entry-point gap, the
same class this repo's own CLAUDE.md names ("a unit test that supplies its own calling convention
proves the unit works when called that way — not that anything calls it that way"). What caught
it: `decoder/serialize_census_test.go`'s `.giw` round-trip check, run generically over every
family in `censusList` (spark2-5-tiny was added there, not excluded, since the fused-QKV split
and the generic sigmoid gate are real per-layer state no other censused fixture covers) —
`greedyN` calls `m.Generate(...)` for real, an 8-token prompt long enough to route through batched
prefill. Fixed by mirroring the exact same `case ActGelu:` into `forwardn.go`'s switch. Re-ran all
three gates after the fix; unaffected (none of them exercise this path either way, which is
itself the point).

**Gate 2** (real oracle, `XHToken/Spark-X2.5-1.7B`, `scripts/pin_spark2_5_real.py`,
`decoder/spark25_real_test.go`): PASS — argmax exact, logit cosine **1.000000**, full 8-token
greedy continuation matches token-for-token (`"The capital of France is"` → `" Paris."`, then
continuing into coding-flavored text, consistent with this being a community coding-tuned
release). Needed a real transformers version pin (`~/.venv-spark25`, `transformers==4.57.1`
exactly) — the vendored modeling code breaks against newer transformers (a tied-weight-key
list-vs-dict API change, and `create_causal_mask()`'s signature dropping `input_embeds`). Also
needed a monitored fit-guard bypass on this specific machine (16 GB, ~8.6 GB available with the
usual desktop load; the 1.7B needs ~6.4 GB resident at f32, ~0.3 GB over the guard's 70%-margin
budget) — approved, watched (swap usage, before/after), no incident, 10 seconds total.

**Gate 3** (fit guard at long context, same real checkpoint): PASS. Confirmed the M-28 windowed-KV
fix (`decoder/arch.go`'s `kvPositionsAt`/`kvBytesForCtx`) correctly caps Spark-X2.5's 21 sliding
layers at their 512-token window instead of the naive full-context price — at a 131072-position
request, 3.80 GB (windowed) vs 15.03 GB (flat, pre-M-28), a 3.95× reduction, matched against a
hand-computed expectation from the real config. Then confirmed the real integration behavior: an
explicit `ResidentContext: 131072` pin on the actual checkpoint is cleanly refused
(`*FitDeclineError`), not silently loaded into swap — the exact failure mode an HF user hit
running this same family on llama.cpp with `--swa-full` (which disables that engine's own
sliding-window memory saving). `TestSpark25Real_gate` and `TestSpark25Real_fitGuardLongContext`
need OPPOSITE `GOINFER_NO_FIT_GUARD` states and must run as separate invocations — documented in
the test file's own doc comment.

**The 4B was skipped, not attempted and failed.** Real-oracle load needs ~16 GB resident at f32 —
this machine's entire RAM, not a close call like the 1.7B's ~0.3 GB shortfall — plus an 8.22 GB
download onto a disk already at 98% capacity (12 GB free at the time). Both axes are an
order-of-magnitude mismatch for this machine; left as a follow-up for a bigger box rather than
forced with `GOINFER_NO_FIT_GUARD` the way the 1.7B's small shortfall was.

CPU-only for now: `FeatAttnOutputGate` (no resident backend implements the sigmoid gate) forces
CUDA/Metal/WebGPU to decline, the same mechanism Laguna already uses — confirmed directly
(`spark2_5 → admitted by []`). Manifest promoted to `status: "validated"`, `method:
"full-forward-oracle"`; `validated_at` stays `null` until this lands on `main` (the field means a
real commit SHA, and nothing was committed until this push).

---

## Why now

The audit already costed this at 1–2 days including a tiny golden, and ranked it seventh. Two
things since move it up:

1. **Audience fit.** A community quant, SharpSpark (peculiar-ragdoll/Sharp-Spark-X2.5-4B-GGUF),
   re-packages Spark-X2.5-4B specifically for agentic coding on weak hardware — fixed chat
   template, a coding-oriented system prompt, a custom importance matrix weighted toward agentic
   coding and security, and a hand-tuned per-tensor bit allocation chosen against SWE-bench-Live
   rather than KL-divergence. That is our audience exactly. Nothing else we have been sent this
   month targets them this directly.
2. **An embedded-binary candidate.** Our model-embedded release builds ship qwen2.5-coder at 0.5B
   and 1.5B. A community Q4_K_M of Spark-X2.5-4B is ~2.6 GB. A 4B tuned for agentic coding, inside
   a single downloadable file, would be a real step up in what those binaries can do — and it is
   the kind of capability the landing page is built around.

Neither of those is a performance claim. They are reasons the family is worth its 1–2 days ahead
of the audit's ordering.

---

## What it is, per L-06

Fused QKV; a head-wise **sigmoid** output gate; 1:3 full/sliding interleave with a 512-token
window; layer-dependent RoPE dims and bases; GELU parallel FFN.

What we already have, per the audit:

- the full/sliding interleave
- per-layer-type rotary width and base (Laguna)
- the parallel block (Command-R)
- a softplus gate

**The gap is two things:** a sigmoid gate — `decoder/arch.go:288` already notes that another
family's gate is sigmoid — and non-gated GELU combined with the parallel block. The audit's
estimate: one descriptor plus a gate-activation enum.

Read L-06 in full before starting and verify each "we already have" claim against current code —
that audit is eleven days old and the tree has moved.

---

## Scope

1. The `spark2_5` architecture descriptor and the gate-activation enum.
2. A tiny golden, per the house pattern.
3. **Then promote straight to real-oracle.** Spark-X2.5-4B is ~8 GB bf16 and the 1.7B smaller —
   both fit the Linux box trivially, and the audit's own framing ("a 4B dense that fits every rig")
   means there is no hardware reason to leave it at tiny-oracle. Given what the promotion work has
   found — two real defects from four non-trivial promotions, one in goinfer and one in
   HuggingFace's own reference — do not ship a new family at the tier we just spent a week moving
   families *out* of. Use the pin template of the most recent clean promotion.
4. **Start with the 1.7B** for the real oracle if it is simpler to reach, then the 4B.

The reference forward needs `trust_remote_code=True` in transformers. That is the same situation
as internlm2, whose remote-code rotary embedding produced a corrupt reference through a
non-persistent buffer. Check `inv_freq` and any other non-persistent buffers in the Spark remote
code before trusting the golden.

---

## Two cautions to carry

**Sliding window means no speculative decoding.** `specRollbackSafe` refuses `SlidingWindow > 0`.
Correct and expected; note it in the capability-matrix row rather than letting it be rediscovered.

**The 1M-context claim is the hazard.** A user running Spark-X2.5-4B Q4_K_M on a 15 GB Linux box
with no NVIDIA GPU (HF discussion XHToken/Spark-X2.5-4B #3) found that the sliding window masks
long-context KV cost, and that `--swa-full -c 131072` pushed the machine into a kernel OOM.

That lands directly on the R13 work. The fit guard must price KV correctly for a sliding-window
model at long context — which is exactly the case where a naive calculation is most wrong. Verify
the guard's behaviour on `spark2_5` at 131072 explicitly, and confirm `MaxPositions` is read from
`context_length` for this family's GGUF builder: the R13 bonus bug was 16 of 18 builders not doing
so, and a new builder is the likeliest place for that to recur.

Same report, for context only: thinking mode by default consumed the whole generation budget, and
with thinking off it produced code but got a regex grouping wrong. That is a model-quality
observation, not ours to fix — but it argues for the chat-template handling of thinking being
explicit rather than inherited.

---

## The SharpSpark quant — check, don't assume

Mixed per-tensor quantization types in one GGUF are normal, and llama.cpp's own K-quant recipes
already do it. But SharpSpark's allocation is explicitly non-standard. We read 17 GGUF types,
including IQ4_NL and IQ4_XS but not the IQ2/IQ3 family.

Once `spark2_5` loads, try the SharpSpark file and report every tensor type it contains. If it
uses a type we do not read, record which — that is a finding about our quant coverage, not a
reason to add the type in this task.

Their custom chat template and system prompt are theirs; we should load and apply the template the
GGUF ships, as for any other model.

---

## Gates

**Gate 1 — loads and matches the tiny golden.**
**Gate 2 — real-oracle on released weights**, at the house bar. A failure is a finding; name the
differing term and stop.
**Gate 3 — fit guard correct at long context.** The guard refuses or auto-pins rather than letting
a 131072 request swap a 16 GB machine.

Embedding it in a release binary is a separate decision, after gate 2, and needs its own sign-off.

---

## Constraints

- Register the gate in the same change as the test — `TestRealckptGateIsListedOrExplicitlyNotRequired`
  caught three missed registrations already.
- Checkpoints from `~/models` on the box doing the run.
- No cgo.
- Do not use the words "honest" or "honesty".
- Leave uncommitted for review. (Reviewed and explicitly superseded 2026-09-21: committed and
  pushed once Gates 1–3 passed and the docs above were written.)

## Related

`docs/audit-2026-09-10.md` L-06 and rec 7; `decoder/arch.go:288` (the existing sigmoid note);
`docs/parity-coverage-policy.md` (promotion timings and the tier discipline); the R13 fit-guard
work and its `context_length` → `MaxPositions` regression gate; `docs/capability-matrix.md`.
