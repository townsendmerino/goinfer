// Package kvsnap manages GPU-to-host prefix KV snapshots, LRU byte budgets
// (resident.KVHostBudget), candidate matching, and boundary/recurrent cut
// planning below decoder.
//
// Tests: TestKVSnapStore_reserveHoldsTheBudget,
// TestKVSnapStore_noEvictionForWhatCannotFit,
// TestKVSnapStore_removeLockedIdempotent, TestKVSnap_planAndRuns.
package kvsnap
