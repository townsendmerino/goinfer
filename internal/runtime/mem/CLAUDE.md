# internal/mem — working rules
- Day loop: `go test ./internal/mem/...` (seconds), and `go test ./decoder -run TestDecisionCensus` (16 s) for any change to a decision. Never load a model or touch a device here.
- Import nothing from goinfer. Read the OS or a device only in `internal/mem/probe`; everything here takes its readings as a `Machine`.
- A decision change that moves the census is a red run: the PR says why, and only then rewrites the golden with `-census.update`.
- A guard threshold changes only with Francis's word (the root `CLAUDE.md`). The Policy's approval stamp (I7) is not built yet; only Francis writes one.
- A new invariant goes in `doc.go` with its test name.
