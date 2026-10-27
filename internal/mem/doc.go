// Package mem decides whether a load or a request fits the machine's memory, and says why in the lines the load banner and
// the refusals print. It is pure: every decision takes numbers (what the load will allocate, as the caller priced it from
// the checkpoint), a Machine (the readings, as data) and a Policy (the thresholds), and returns the same answer for the same
// inputs. It reads nothing: internal/mem/probe reads the OS, and the caller reads the checkpoint.
//
// Contract. SolveHost is the resident load guard, SolveGIW the guard for a .giw load (file-backed weights), AdmitRequest the
// per-request check, HostBudget and WeightCacheBudget the budgets the banner and the expert pager size by. SolvePlacement
// places a model on one backend's memory (context first, then an expert cache), and SolveMetalCtx picks an unpinned Metal
// load's context with it. Every unknown
// proceeds: a Machine whose reading is 0, or a load whose weights could not be priced, gets no verdict, never a refusal.
//
// Invariants, each with the test that holds it:
//
//	I4 every decision is pure                          TestSolve_pure
//	I4 the planner is pure                             TestSolvePlacement_pure
//	I5 more memory never gives a worse decision        TestSolve_monotone
//	I5 more memory never gives a worse placement       TestSolvePlacement_monotone (context first: slots may fall as it grows)
//	the decisions are today's, byte for byte           decoder's TestDecisionCensus
//
// Imports: the standard library only (TestImports).
//
// Test it: go test ./internal/mem/ (seconds; no model, no device). The census: go test ./decoder -run TestDecisionCensus.
//
// Plan and phases: docs/tasks/task-subsystems-2026-10.md, Track MM.
package mem
