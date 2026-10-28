- **The serving shell holds each loaded model through a handle interface** (SB7, H1), and its liveness and unload tests run
  on a fake model. Four more `internal/serveapp` test files no longer import `decoder`, so the count is now 33. No
  behavior changes: the wire and scheduling censuses reproduce byte for byte.
