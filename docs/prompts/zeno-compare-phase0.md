# Task (goinfer): Zeno head-to-head, Phase 0 — feasibility only, no benchmarking yet

> **For:** Claude Code, in `~/tmcode/goinfer`, on the 16 GB M1 Pro MacBook. Written 2026-08-24.
> Context: Icosa's Zeno (r/ollama post, 2026-08) ships 4-bit Qwen3.5-35B-A3B on 16 GB Macs via
> disk offloading and posts: 10k prompt — 214 tok/s prefill / 8.7 tok/s decode; 541 prompt —
> 66 / 13.2; llama.cpp best-config at 8.7-12.6 prefill / 3.5 decode. Vendor-posted, one table
> image, treat as directional. goinfer's nearest measured point is gemma4 26B-A4B paged at
> 16.98 tok/s (`docs/completed/gemma4-resident-scope.md`). **Phase 0 answers "can a fair
> comparison be run at all," and nothing else.** The matched-depth bench (Phase 1) gets its own
> brief only if this clears. This is a side quest: do not run it concurrently with any W4A8
> plumbing benchmarking (quiet-box discipline cuts both ways), and it shares no files with that
> campaign.

## Deliverable

A short feasibility note — `docs/task-zeno-compare.md`, opening section — with a go/no-go for
Phase 1 and the "what's measurable" inventory that Phase 1's cell design depends on. A no-go
with reasons is a complete deliverable.

## Part A — can goinfer run the actual model?

The qwen3_5_moe family is parity-gated on the tiny/synthetic checkpoint only; **a real
35B-A3B has never been loaded, paged, or decoded.** That's the gap Phase 0 closes or fails on.

1. **Model:** Qwen3.5-35B-A3B, 4-bit GGUF (Q4_K_M) — 3.5, not 3.6, because same-model-family
   matching against what Zeno ships is the point; note in the doc that weight-identity with
   Zeno's copy cannot be verified (closed app), unlike the bench_peer/ollama discipline. Check
   disk first: ~18-20 GB GGUF plus the `.giw` prequant output — confirm the Mac has room for
   both before downloading.
2. **Prequant → paged load:** `cmd/prequant -quant int4` → `.giw`, then load on the paged path.
   `decoder/moepaging.go` covers generic `[NumExperts]expertWeights` families (not just
   gemma4's fused sub-block), and the DeltaNet layers are dense (unpaged) — but "covered by the
   code path" and "works on a real 35B checkpoint" are different claims; this step turns the
   first into the second. At 4-bit the experts alone exceed 16 GB unified memory, so paging is
   mandatory, not optional.
3. **Coherent decode, measured in passing:** a real prose completion (the
   gemma4-resident-scope bar: coherent output, slots, hit rate) with tok/s recorded at one
   short depth. This is a feasibility number, not a benchmark — one run, conditions labeled.
4. **Size the recorded handicap before Phase 1 can be honest:** `docs/queue-performance.md`
   (~line 97) records that this family runs f32 scratch regardless of `Options.Quant` —
   "parity-first, from the qwen3_5_moe bring-up... never revisited." Determine whether that
   sits on the decode hot path for this config and estimate its cost (a stub or a scratch-size
   accounting is enough — do NOT fix it in Phase 0). Decision rule for the note: if it plausibly
   costs more than ~20% of decode, Phase 1 either waits for that P-item or carries the caveat
   in its headline, because benchmarking a known-unoptimized path against a competitor without
   saying so is not this repo's standard.

## Part B — what can be measured on Zeno's side?

**Checkpoint with Francis before installing anything** — Zeno is a closed-source, week-one
binary from a startup's download page; he decides whether it goes on this Mac at all, and if
so, under a separate macOS user account or another machine. Phase 0's Part B does not proceed
past this checkpoint without his answer.

If cleared, the inventory to produce (this determines whether Phase 1 is API-grade or
stopwatch-grade, which determines what claims its doc can make):

- Does Zeno expose a local server or API (ollama-compatible or otherwise), or only its own UI?
- Does it report tok/s, TTFT, or token counts anywhere (UI, logs, files)?
- Can generation be pinned: greedy/temperature-0, fixed decode length, controlled prompt
  length? Can cold vs warm start be distinguished and forced?
- What model/quant does the shipped bundle actually declare, and does it match the post?
- RAM/disk footprint while decoding (Activity Monitor grade is fine) — their offload claim,
  observed.

## Go/no-go for Phase 1

**GO** = Part A produces coherent paged decode of the real model at any usable rate with the
handicap sized, AND Part B (if Francis clears the install) yields at least stopwatch-grade
measurement with controllable prompt length. The Phase 1 brief then designs matched-depth
cells (541 to mirror their short cell, 2k, 8k; prefill bounded, the 10k prefill cell only if
goinfer-side wall-clock is tolerable — batched MoE prefill is an unchecked roadmap item and
the prefill column will likely document that gap; expected outcomes stated before measuring,
per house convention).

**NO-GO** = the real checkpoint won't load/page/decode (that finding is itself the most
valuable possible output of Phase 0 — file it as the bring-up gap it reveals), or Zeno offers
no controllable measurement surface (then Phase 1 shrinks to goinfer-vs-llama.cpp on the same
model, with Zeno's posted table quoted as unverified context only).

## Not in scope

- Any Phase 1 benchmarking, any engine changes beyond what loading requires, fixing the
  f32-scratch P-item (size it only), batched MoE prefill work, Qwen3.6 (wrong match), and
  anything W4A8 — that campaign owns its own files and its own box time.
