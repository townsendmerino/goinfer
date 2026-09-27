package decoder

// CPUConcurrentSafe reports whether several generations of this model may run at once on the CPU path, each on its own
// Session (MC3c, docs/tasks/task-concurrency-2026-09.md; TestConcurrentSessions_matchAlone pins it, under -race). A
// Model's weights are immutable after Load and each generation owns its KV cache and decode scratch, so distinct
// sessions are independent. False for a GPU-resident model — it has ONE claimed resident KV (resBusy) — and for a
// model that streams its weights through an expert or layer pager, whose residency state is shared.
func (m *Model) CPUConcurrentSafe() bool {
	return m.resident == nil && m.pager == nil && m.layerPager == nil
}
