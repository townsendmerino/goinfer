- **Cost counters on WebGPU** (SB11, step 3). The `Context` counts the dispatches its decode runners record, the command
  buffers they submit and their blocking polls (always on, a few atomic adds a token). When a counter is attached, a
  runner records each submitted dispatch against its pipeline, and the token's waits, submits, new leases and wall time
  against the phase. Off, the cost is at most 0.1442% of a WebGPU decode token (measured). Nothing new is sent by the
  server.
