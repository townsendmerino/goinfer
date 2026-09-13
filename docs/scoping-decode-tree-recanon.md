# Scoping: re-canonicalising the decode attention reduction tree

> Scoping doc, 2026-09-13. **Recommendation: DO NOT START. Spike the thing it unblocks first —
> the prerequisite is one of the largest changes available in this tree and the payoff it exists to
> unlock has never been measured.** Kill criteria and the cheaper order are in §6.

## 1. What it is

Make the single-block kernel (`attn_batched` at M=1) and the split kernel (`splitkv_vsum`) compute
the SAME blocked reduction tree — S splits over keys as a pure function of `nKeys` — instead of the
strict sequential left fold both use today. Both kernels change; both produce different bytes than
they do now.

## 2. The reframe that decides most of this: it buys nothing on its own

The idea arrived as "then split-KV can be default wherever it wins with no fidelity dimension to the
gate at all." **Split-KV has no fidelity dimension today.** It is bit-identical to `attn_batched`
by construction — that constraint is the whole of Campaign A's design, it is stated four times in
`cuda/decode_splitkv.cu`'s own header, and it is gated by `TestSplitKV_bitIdentical` plus a second
gate on the hd=256/windowed path. The split-KV gate is *already* a pure performance decision.

So re-canonicalisation is **not a standalone improvement**. Its entire value is as a **prerequisite
for a non-bit-identical kernel** — specifically the flash-decode V-sum, which splits the V reduction
over keys and is the only identified lever on `splitkv_vsum`'s 75.5% long_scoreboard at 9%
occupancy. Scope it as that prerequisite or not at all.

## 3. What would actually change

- **`splitkv_vsum`** — from the whole per-dim sequential fold to S partials plus a fixed-order combine.
- **`attn_batched`** — must adopt the identical tree, or the two diverge. Note this kernel is ALSO
  the prefill exact path at M>1 and is what spec-decode verify and the parity gates run, so the
  change is not confined to decode.
- **The CPU reference** — goinfer's backends are parity-gated against pure-Go CPU, which is itself
  gated against HuggingFace. If CUDA's tree moves and CPU's does not, every CUDA-vs-CPU gate shifts.
- **Metal and WebGPU** — same argument. Either they adopt the tree or cross-backend identity, which
  `docs/positioning.md` describes as a within-machine property, degrades further.

## 4. Blast radius, measured not estimated

| | count |
|---|--:|
| test files asserting bit-identity / byte-identical | **79** |
| parity families in `testdata/parity_manifest.json` | **36** |
| goldens under `testdata/` and `decoder/testdata/` | **116** |
| backends that would have to move together | **4** (CPU, CUDA, Metal, WebGPU) |

Every one of those goldens is a numeric artifact that would need regenerating, and CLAUDE.md's
standing rule is that re-baselining a floor because a number moved is how a regression gets blessed.
A re-base of this size is defensible only with a stated mechanism and a way to tell an intended
shift from an unintended one — which does not exist today and would itself be part of the work.

## 5. Cross-M identity: four conditions, all checkable

Speculative decode requires the M>1 verify pass to agree bit-for-bit with M=1 decode. A shared tree
holds only if:

1. Tiles are anchored at key 0 and **S = f(nKeys) alone** — not of M, not of SM count.
2. The combine is a **fixed-order** pass, never atomics.
3. Per-row inner arithmetic is identical at M=1 and M>1 — a kernel that switches to MMA above some M
   breaks identity even with identical tiling.
4. S is **pinned per backend in a table**, not derived from an occupancy heuristic that differs
   between a 2070 and an M-series part.

**A refinement the original four missed:** under causal masking the rows of a batched verify attend
*different* key counts, so S must be evaluated **per row** from that row's own `nKeys`, not once per
launch from the maximum. A per-launch S would tile row *i* differently in verify than in decode.

The test already has a shape in this tree: row *i* of an M=K verify against the M=1 decode at the
same position, raw-bit, at every depth where S changes.

## 6. The cheaper order, and the kill criteria

**Do the spike before the prerequisite.** Build the flash-decode V-sum as an opt-in,
non-bit-identical kernel behind a flag and measure it. That is the repo's own shipping pattern —
`attn_fused` and split-KV both landed opt-in, gated, then default-on — and it costs one kernel
instead of 116 goldens.

Then decide with a number:

- **Spike gains < 5% on the attention kernel** → kill. Re-canonicalisation is unjustifiable at any
  price, and the V-sum's latency bound is simply where decode ends on this hardware.
- **5–15%** → park. Real but not worth re-basing four backends; revisit if a second consumer of the
  shared tree appears.
- **> 15%** → the prerequisite becomes arguable, and only then is this doc worth reopening.

**The reason to expect the low end, stated so it is not a surprise:** the q-staging change measured
on 2026-09-13 relieved a 93.5% stall and returned 5.5% on its kernel, because the stall underneath
took over. `splitkv_vsum` is 75.5% long_scoreboard; the same pattern predicts splitting it exposes
whatever is beneath, and nothing here says how much is left after that.

## 7. What is NOT in scope

The accuracy question is settled and is not a blocker either way: a blocked fold is measurably
**more** accurate than the sequential one — 1.76× to 4.93× closer to an f64 reference across 36
cells at nKeys ≥ 2048 (`docs/measurements/reduction-tree-accuracy-2026-09-12.md`). Re-canonicalising
would improve numerics. That is a real point in its favour and it is not sufficient on its own,
because nobody is asking for a more accurate V-sum.
