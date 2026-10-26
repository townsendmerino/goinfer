// Package kernelprint fingerprints the numerics of the aikit kernels goinfer's production code calls. Each kernel runs
// over a fixed, generated input suite that reaches its dispatch paths (M=1 and M>1, tiles and tails, the parallel
// thresholds goinfer sets, per-row and per-32 activation quantization, every int4 layout), and the SHA-256 of its output
// bits is that kernel's fingerprint.
//
// Contract. Compute is deterministic for one aikit build under one Config (GOARCH, linalg.ActiveKernels().Active,
// linalg.ForcedFallbacks()); fingerprints are comparable only under the same Config. It reads no file, environment
// variable or device, and changes no process-wide setting. A kernel's inputs never depend on another kernel's output,
// so a change to one kernel changes only the fingerprints of the kernels that call it.
//
// Gates (docs/tasks/task-subsystems-2026-10.md §5.4, "VK1's gates"):
//
//	K1 stable          TestCompute_stable
//	K2 mutation        TestCompute_mutation (runs with GOINFER_KERNELPRINT_MUTATION=1: it copies aikit and builds five times)
//	K3 census reach    TestCompute_reachesCensus
//
// Imports: the standard library and aikit's linalg and embed.
//
// Test it: go test ./internal/kernelprint/ (seconds, plus one coverage build for K3).
//
// Design: docs/code-notes/kernelprint.md.
package kernelprint
