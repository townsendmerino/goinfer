- **`serve`'s `-max-queue` bound and admission order moved to `internal/scheduler`** (SB6, M2). This covers claiming a queue slot,
  waiting for a turn, re-checking for a halt once the turn is granted, and giving both back. No behavior changes: the
  scheduling census reproduces byte for byte.
