// Package scheduler is serve's scheduling policy, as decisions over plain values: who gets a model's next turn, how
// many requests a model holds, and which resident KV session a prompt extends, which idle one gives it rows by copy and
// which is evicted. It holds no model, session, clock or I/O: the server computes the inputs (prompt tokens, each
// session's tokens, first prompt, busy flag and shared hit) and acts on the answers.
//
// Contract. Admission is a per-model FIFO turn granter, Capacity wide, that a waiter leaves the moment its context
// ends. Admit claims a slot of the model's queue (NewQueue: turns plus -max-queue, nil for unbounded), refusing at once
// when it is full, then waits for a turn and gives both back on a halt found after the grant; Exit gives them back in
// reverse. PickSession is the session a prompt extends (BestExtend's floor, the room rule, KeepsFirst's own-rollback
// exception), PickDonor the idle session with the longest shared hit (the least recent among equals), ColdestIdle the
// least recent idle session. Admission never reorders: it has no priority, class, deadline or fairness across sessions.
//
// Invariants, each with the test that holds it:
//
//	admission order, refusals and timing are today's   serveapp's TestSchedCensus_admission
//	placement decisions are today's                    serveapp's TestSchedCensus_placement
//	a cancelled waiter leaves without the holder       serveapp's TestAdmission_waiterLeavesOnContextCancelWithoutWaitingForHolder
//	any halt after the grant refuses and frees both    TestAdmit_haltWithNoReasonRefuses
//	-max-queue 0 is unbounded; turns count as 1 or more TestNewQueue_unboundedAtZero
//	a short or nil busy slice marks sessions idle       TestPickDonor_shortBusy
//	Capacity is race-free beside SetCap and Enter      TestAdmission_capacityUnderConcurrentSetCap (-race)
//	the standard library only                          layout's TestLayout_arrows (rule scheduler-standard-library)
//
// Imports: the standard library only.
//
// Test it: go test ./internal/scheduler/ (no model), and the census,
// go test ./internal/serveapp -run 'TestSchedCensus_admission|TestSchedCensus_placement' (about 4 s; the tiny CPU
// checkpoint under testdata/).
//
// Design: docs/code-notes/internal-scheduler.md. Plan: docs/tasks/task-subsystems-2026-10.md, SB6.
package scheduler
