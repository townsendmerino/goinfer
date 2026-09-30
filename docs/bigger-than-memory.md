<!-- Moved verbatim from the README on 2026-09-30 (links re-pointed, smoke markers dropped:
     the README keeps the runnable commands and the CI check on them). -->

# Running a model bigger than your RAM — or your GPU

A 20-35B-class MoE does not fit in 16 GB of RAM, or on an 8 GB GPU, and loading it anyway will
drive your machine into swap (or your CUDA allocator into an OOM) before anything says so.
The RAM-overflow and GPU-overflow examples below both use the checkpoint this project actually
validates at that size, rather than a size class with nothing behind it to download:

```bash
# a 20-35B-class MoE, real and resolvable — goinfer-chat models for the full entry
goinfer-chat pull gpt-oss-20b
```

```bash
# the flag to reach for whenever the checkpoint file is larger than about half your physical RAM
goinfer-serve -stream-weights -weight-cache 6 -model ~/models/gpt-oss-20b-MXFP4.gguf
```

Resident memory is then capped near `-weight-cache` rather than the model size, because only the
experts a token actually routes to are resident. Measured on an M1 Pro / 16 GB with a 21 GB
35B-A3B (a different checkpoint at the same size class — the mechanism is the same either way):
without the flag, **+7.8 GB of swap in five seconds**; with it, RSS peaked at **8.95 GB** and fell
back to 2.7 GB, with zero swapouts
([`docs/measurements/cold-user-2026-09-06.md`](measurements/cold-user-2026-09-06.md),
scenario D).

**Update on `gpt-oss-20b` specifically: its one-time sidecar transcode now streams (S2,
2026-09-23) and a real run showed swap-used flat throughout** —
[`docs/measurements/transcode-streaming-2026-09-23.md`](measurements/transcode-streaming-2026-09-23.md).
That run did not finish end-to-end (this machine's free disk ran out mid-write, an unrelated
capacity limit, not a memory one), so treat this as strong evidence rather than a completed proof
until a disk-headroom-permitting rerun confirms the finished bundle byte-identical to a resident
build. **The other five `needsResidentSerialize` families (gemma4, laguna, granite, nemotron,
llama4) still build resident before they can be transcoded at all** — the same historical risk
this whole section describes, for those families' one-time transcode specifically.
`GOINFER_SWAP_GUARD`'s load-time half (armed by default) will abort a resident build if swap grows
too far, but a real run on gpt-oss-20b (before its own S2 fix) found it does not always hold the
line unassisted under a fast enough burst
([`docs/measurements/swap-tripwire-2026-09-22.md`](measurements/swap-tripwire-2026-09-22.md)).
Treat the first `-stream-weights` run of any of those five families' checkpoints with the same
caution as a direct load of one — watch it, don't walk away from it.

**This is `goinfer-serve`'s job, not `goinfer-chat`'s.** The single-shot chat runtime holds all
weights resident by design; it has no `-stream-weights`. If your model is bigger than your RAM,
reach for the server.

**On cuda/metal, GPU means fully resident, full stop.** Neither backend has a partial/"staged"
GPU path (R9, [`docs/measurements/cold-user-2026-09-06-nobara-pc.md`](measurements/cold-user-2026-09-06-nobara-pc.md)):
a model or architecture that does not build the resident runner declines straight to CPU, at
whatever quant you asked for. A dense model bigger than your card has no partial-GPU story here —
only `-stream-weights` (above, RAM-side) or the CPU. **A MoE does**, and that is
`-moe-cache-experts`: the non-expert core stays resident while a slot cache of the experts a
token actually routes to streams host→VRAM per token on demand, so the whole model never needs
to fit VRAM. Measured on this project's own most-benchmarked checkpoint at this size,
`gemma-4-26b-a4b` (26B-A4B, 128 experts top-8; [`docs/benchmarks.md`](benchmarks.md) §B4/§B4.1),
on an RTX 2070 SUPER 8 GB: **40.2 tok/s** at ctx 2048, with every expert kept on the GPU (the DMA overlap;
[`docs/measurements/peer-claim-2026-09-25.md`](measurements/peer-claim-2026-09-25.md) cell c) — capacity-bound
(PCIe host→VRAM streaming), not a kernel or MoE deficiency:

```bash
# the checkpoint this project has the most measurements on at this size
goinfer-chat pull gemma-4-26b-a4b
```

```bash
# off by default; a model that does not fit then declines to the CPU path and says why
goinfer-serve -backend cuda -moe-cache-experts -model ~/models/gemma-4-26B_q4_0-it.gguf
```

`gpt-oss-20b` (`goinfer-chat pull gpt-oss-20b`; 12 GB at native MXFP4) is a second real option at
this size class — its own CUDA resident gate is measured on an 8 GB card too, see
[`capability-matrix.md`](capability-matrix.md)'s gpt-oss row — with `-moe-cache-experts` results not yet as
thoroughly measured as gemma-4-26b-a4b's.

Back to the [README](../README.md).
