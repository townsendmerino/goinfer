- **Cost counters on CUDA** (SB11, step 2). The device counts its kernel launches and host waits (one atomic add each, always
  on), and the resident decode, when a counter is attached, records each launch against its kernel, a token's waits as
  `sync` and its new leases as `alloc`, and the token's wall time as the phase's. Bytes and time per kernel are listed
  unmeasured, since a launch is asynchronous.
  Off, the cost is at most 0.0847% of a CUDA decode token (measured). Nothing new is sent by the server.
