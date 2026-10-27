- **Changes behavior for library callers on macOS and Linux. `decoder.Load` of a `.gguf` or a safetensors directory opens its
  sidecar `.giw`**, transcoding it once beside the source, as `serve` and `chat` already did (MEM1,
  `docs/tasks/task-memory-campaign-2026-10.md`). The weights are a mapping of that file, not a heap copy: the 0.5B at int4 holds
  1 MiB of Go heap after a load, against 647 MiB before. A `.gguf` whose sidecar cannot be written is refused, naming the new
  `Options.DirectLoad`, which restores the direct load. Test binaries keep the direct load. The transcode and the sidecar cache
  moved from `internal/prequant` into `decoder` (`EnsureCachedGIW`, `TranscodeGIW`, `SidecarPath`, ...; Experimental tier).
