- **Changes behavior on macOS. The fit guard's available memory is the kernel's own free percentage** (`kern.memorystatus_level`, the
  figure `memory_pressure` prints) times physical memory, less 500 MiB (owner's decision, 2026-10-24). It counts app memory as
  available, since the kernel can compress it. On a 16 GB Mac with the Claude desktop app open it read 12.8 GiB where the free,
  file-backed and purgeable pages (B41's reading) read 5.9. A timed run's validity keeps B41's reading, `cmd/hostmem -reclaimable`:
  the night runner's start check, the benchmark harness's and the void rule, since compression under a timed run changes what it
  measures. Linux keeps `MemAvailable` for both.
