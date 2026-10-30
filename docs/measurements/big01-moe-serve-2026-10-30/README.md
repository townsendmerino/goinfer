# BIG1 — a 35B MoE served on nobara's 8 GB card, experts host-streamed (2026-10-30)

Francis, 2026-10-30 10:35: the 7-9B models that fit nobara's card pass about 1 in 10 of tack's tiny coding tasks, so
try a bigger MoE today, Qwen3.6-35B-A3B first and gpt-oss-20b second, experts streamed from host RAM
(`-moe-cache-experts`). This record is the serve check that hands a command line to tack's E0.

## Why tack's E0 of 2026-10-18 was void (from the code and that run's own logs)

tack's C3 run (tack-runs `e0/c3-2026-10-18`, goinfer f20f0d93) loaded `qwen3.6-35b-a3b-q4_k_m.gguf` with
`-backend cuda -moe-cache-experts -thinking off -ctx 32768`. Its server log shows:

- The load did **not** re-quantize the `.gguf` into the heap. On Linux a `.gguf` routes to its sidecar
  (`decoder/sidecar.go:DefaultToSidecar`, `sidecarRoute`), and the v17 sidecar
  `qwen3.6-35b-a3b-q4_k_m.int4.e4h.cuda.v17.giw` (19.6 GB, built 12:51 that day) is what loaded: "reading its 19.6 GB
  once before the load".
- The anonymous term is the C′ expert cache's host copy. `cuda/pack_stream.go:cudaResident.packStackPinned` allocates
  the whole int4 expert stack as fresh `cuMemAllocHost` memory (`NewMappedHostBufferOn`, plan line `LinePinnedExperts`)
  and packs every expert into it from the mapped `.giw`. For this model that is 40 layers × 256 experts × 3 × 2048×512
  weights = 32.2 G weights: 16.1 GB of nibbles + 2.0 GB of f16 group scales ≈ 18.1 GB pinned, which is the fit line's
  "weights 18.2 GB". Pinned pages are anonymous and locked: they cannot be paged out, so the kernel finds them by
  reclaiming something else.
- The file-backed term is the `.giw` mapping itself: the pre-read and the pack touch all 19.6 GB, and those pages count
  in RSS while resident. So RSS ≈ 18 GB pinned + up to 19.6 GB file-backed; the 33.1 GB peak tack recorded is that sum
  with part of the file already reclaimed.
- The box was already 9.82 GB into system swap at the load ("swap guard: baseline 9.82 GB (system swap)"). Pinning
  18 GB on top pushed 0.90 GB of the server's own anonymous pages out ("TRIPPED — server swap grew 0.90 GB over
  baseline"), past the per-process guard's 512 MB, and every request after that got 503.

So the question for today is whether the same load stays out of swap on a box that is not already paging, and if not,
what to change. No transcode is needed: the v17 sidecar is current (`decoder/serialize.go:giwVersion` = 17).

## Registration (written before the run)

- **Tier 1**, one load, one pass, by day under the timing lock. Expected wall: load about 1.5 min, probe about 5 min.
- **Instrument:** `memwatch.py` samples the server's `/proc/<pid>/status` (VmRSS, RssAnon, RssFile, VmPin, VmLck, VmSwap,
  VmHWM) and system swap and MemAvailable each second from the server's start to its exit; `probe.py` sends, in order,
  a forced tool call (`tool_choice: "required"`, a seven-tool coding-agent list), a 256-token decode, a cold ~4k and a
  cold ~12k prompt, an append-only follow-up to the 12k conversation, then a harness turn pair at ~12k (a cold forced
  call over a 12k context, then the tool result appended and the next forced call).
- **Clean serve** = the model loads cuda-resident, the server's VmSwap stays under 64 MB and system swap stays within
  +512 MB of its value at the server's start for the whole run, and the forced call returns a call to a listed tool with
  arguments that parse as JSON.
- **Stopping rule:** memwatch sends SIGTERM at the first sample over either swap bound; the run is then void and the
  record says so. Otherwise one pass, no repeats.
- **Void at the start:** 1-minute load average above 8.0, or MemAvailable below the fit need plus 2 GB.
- The timings are single observations of one build, for tack's planning, not a peer or A/B claim.

## Results

(filled after the run)
