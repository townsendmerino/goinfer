- **`GET /health` no longer says `"status": "ok"` while the server refuses work.** `status` is `halted` while the
  generation routes answer 503 `{"error":"halted"}`: after an admin halt, or while the swap guard is tripped. A tripped
  swap guard now shows in `/health` and `/admin/status` as `swap_guard_tripped`; before, only the log and the 503s said
  so. `/health` still answers 200 in every state, so a probe reads `status`, not the status code.
