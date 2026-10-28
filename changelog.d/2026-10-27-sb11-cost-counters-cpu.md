- **Cost counters on the CPU** (SB11, step 1). `internal/cost` is the schema: per phase (prefill, decode) and per operation,
  calls, bytes read and written, launches, syncs, allocations and time, with X1's phase names and a field a backend does not
  count listed as unmeasured rather than reported as 0. The CPU's program walkers count each operation when a stream carries a
  counter. Off, the hook is a nil check (at most 0.0035% of a CPU decode token, measured). Nothing new is sent by the server.
