- **A server keeps serving when a background task fails.** The web UI load, the SSE heartbeat and the unload drain now
  run under the panic guard. A GPU allocation failure is now its own error type, so the broad recovers decline it cleanly
  while a programming bug still surfaces instead of being swallowed (#1120, #1130).
