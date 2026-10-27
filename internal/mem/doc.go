// Package mem decides whether a load or a request fits the machine's memory, and says why in the lines the load banner and
// the refusals print. It is pure: every decision takes numbers (what the load will allocate, as the caller priced it from
// the checkpoint), a Machine (the readings, as data) and a Policy (the thresholds), and returns the same answer for the same
// inputs. It reads nothing: internal/mem/probe reads the OS, and the caller reads the checkpoint.
//
// Contract. SolveHost is the resident load guard, SolveGIW the guard for a .giw load (file-backed weights), AdmitRequest the
// per-request check, HostBudget and WeightCacheBudget the budgets the banner and the expert pager size by. SolvePlacement
// places a model on one backend's memory (context first, then an expert cache), and SolveMetalCtx picks an unpinned Metal
// load's context with it. ResolveKV is the KV precision a backend's cache holds. In the guards (SolveHost, SolveGIW,
// AdmitRequest) every unknown proceeds: a Machine whose reading is 0, or a load whose weights could not be priced, gets no
// verdict, never a refusal. The planner is told the free bytes and places against them: a 0 reading declines (the CPU pages),
// and so does an expert cache whose experts could not be priced.
//
// Invariants, each with the test that holds it:
//
//	I4 every decision is pure                          TestSolve_pure
//	I4 the planner is pure                             TestSolvePlacement_pure
//	I5 more memory never gives a worse decision        TestSolve_monotone
//	I5 more memory never gives a worse placement       TestSolvePlacement_monotone, TestSolvePlacement_moeSawtooth
//	I6 no guard threshold is applied outside mem       TestNoGuardThresholdOutsideMem (the backends' own guards ratcheted)
//	the decisions are today's, byte for byte           decoder's TestDecisionCensus
//
// I5 holds MoE included because a placement's experts are reserved before the context grows (resident first, then the expert
// cache). The context never shrinks within a placement; moving up to resident can cost context.
//
// Relation to kvmgr. Today the load-time plan sets KV's share of memory: the resident context cap and the KV slot count,
// which size the backend's KV pools. internal/kvmgr then manages that many slots (decoder builds it with
// kvmgr.NewSlots(the resident's slot count)); its block manager runs with no block limit of its own, because the device pools
// bound the bytes, and mem is not consulted per block. The routed experts get what the plan leaves. Planned, not built: MM3's
// ledger leases the KV pools' allocations, and MM4 (KV11) makes an arbiter here the one budget for KV blocks, expert slots
// and the prefix cache. kvmgr then asks the arbiter for blocks, the arbiter decides which class sheds under pressure, and kvmgr
// still chooses which KV block goes (its LRU). See docs/tasks/task-subsystems-2026-10.md §6 and
// docs/tasks/task-kv-architecture-2026-10.md, KV11.
//
// Imports: the standard library only (TestImports).
//
// Test it: go test ./internal/mem/ (seconds; no model, no device). The census: go test ./decoder -run TestDecisionCensus.
//
// Plan and phases: docs/tasks/task-subsystems-2026-10.md, Track MM.
package mem
