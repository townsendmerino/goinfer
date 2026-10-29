// Package engine is what goinfer's serving shell asks of a model: a generation started from a prepared request and
// streamed as token ids, and its report once the stream closes (docs/tasks/task-subsystems-2026-10.md, SB7). It is
// Experimental (docs/api-tiers.md): it may change in any release.
//
// Contract. An Engine's Start begins one generation and returns its token stream, a report function the caller reads
// once the stream has closed (it may be nil, and so may its result, for an engine that keeps none), and a done function
// called once after the stream has been read to its end. Decoder is the Engine a loaded *decoder.Model generates on
// (the GPU-resident paths, or a CPU session from a SessionPool the caller keeps); Scripted is an Engine with no model,
// for tests: it streams fixed ids and keeps no report. The request's sampling, scope and the report's logprobs are
// decoder's own types.
//
// Invariants, each with the test that holds it:
//
//	the package exports only its contract                   TestSurface_exportsOnlyItsContract
//	a report's budget is the clamp when one was made        TestReport_BudgetFor
//	Scripted streams its ids within each request's budget   TestScripted_streamsItsIDsWithinTheBudget
//	the shell's wire output is unchanged by this package    serveapp's TestWireCensus
//
// Imports: decoder, constrain; never the serving shell (internal/layout's engine-below-the-shell rule).
//
// Test it: go test ./engine/ (no model), and the shell's census, go test ./internal/serveapp -run 'TestWireCensus'.
package engine
