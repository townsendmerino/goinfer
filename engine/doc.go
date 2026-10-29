// Package engine is what goinfer's serving shell asks of a model: a generation started from a prepared request and
// streamed as token ids, and its report once the stream closes (docs/tasks/task-subsystems-2026-10.md, SB7). It is
// Experimental (docs/api-tiers.md): it may change in any release.
//
// Contract. An Engine's Start begins one generation and returns its Stream: Tokens, read to its end; Report, the
// generation's account, read once Tokens has closed (nil for an engine that keeps none); and Close, called once the
// stream is drained. NewStream and GenerationStream build one, the second over a *decoder.Generation. Decoder is the
// Engine a loaded *decoder.Model generates on (the GPU-resident paths, or a CPU session from a SessionPool the caller
// keeps). The request's sampling, scope and the report's logprobs are decoder's own types. A fake engine for tests is
// in enginetest.
//
// Invariants, each with the test that holds it:
//
//	the package exports only its contract                   TestSurface_exportsOnlyItsContract
//	a report's budget is the clamp when one was made        TestReport_BudgetFor
//	a Stream's report is nil-safe and Close runs once       TestStream_reportIsNilSafeAndCloseRunsOnce
//	the shell's wire output is unchanged by this package    serveapp's TestWireCensus
//
// Imports: decoder, constrain; never the serving shell (internal/layout's engine-below-the-shell rule).
//
// Test it: go test ./engine/... (no model), and the shell's census, go test ./internal/serveapp -run 'TestWireCensus'.
package engine
