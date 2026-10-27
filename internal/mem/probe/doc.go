// Package probe is the one package in goinfer that reads the operating system's account of memory: physical and
// available RAM, swap in use (the machine's, and this process's own with its descendants), page faults, how much of a
// mapping is in the page cache, and another process's status. Everything else takes these readings as data, through a
// Reader, so a decision can be driven by a Fake instead of the machine it happens to run on.
//
// Contract. Every reading is a number or "unknown": 0 from a byte figure, ok false from the rest. Unknown is a real answer
// that every caller treats as "proceed", never as "no memory", and no reading is ever a guess. Physical is read once per
// process; every other reading is taken when asked. Nothing here decides anything except StartWatch, which samples a swap
// reader and reports growth over its first reading past a threshold the caller passes.
//
// Invariants, each with the test that holds it:
//
//	the only OS memory reader in the tree             TestOnlyProbeReadsOSMemory (source scan, all seven modules)
//	readings match the pre-move readers               TestHost_matchesFrozenReaders
//	a Fake drives every fit-guard outcome             decoder's TestFitGuard_fakeProbeReachesEveryOutcome
//	an unknown swap reading never trips the watch     TestWatch_unavailableReadingsAreSkipped
//
// Imports: the standard library, and golang.org/x/sys on darwin. No goinfer package. Device memory (CUDA's and Metal's)
// is read by the backends, which hold the device; it is not read here.
//
// Test it: go test ./internal/mem/probe/ (seconds; no model, no device).
//
// Design and plan: docs/tasks/task-subsystems-2026-10.md, Track MM (MM0).
package probe
