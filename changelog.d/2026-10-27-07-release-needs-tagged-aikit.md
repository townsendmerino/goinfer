- **A release refuses an untagged aikit** (AX6). `go run ./cmd/gate release vX.Y.Z` now reports its module-pins row
  OWED when any module requires an aikit pseudo-version. Between weekly aikit tags `main` may require one; a goinfer
  release still requires a tagged aikit (owner decision SD3, RELEASING.md).
