package decoder

import "context"

// Forward is the model as an engine policy drives it (SB5, docs/tasks/task-subsystems-2026-10.md §8): a token prefix
// held in the model's KV, grown by Prefill and Verify and cut back by TruncateTo. Prefill returns the logits after its
// last id; Verify returns one row per id, row i being the logits after ids[i]. A policy written against Forward runs
// unchanged on the CPU model (cpuForward) and on a scripted fake.
type Forward interface {
	Prefill(ctx context.Context, ids []int) ([]float32, error)
	Verify(ctx context.Context, ids []int) ([][]float32, error)
	TruncateTo(pos int)
}

// cpuForward is Forward on the CPU path: m's forward over cache.
type cpuForward struct {
	m     *Model
	cache *KVCache
}

func (f cpuForward) Prefill(ctx context.Context, ids []int) ([]float32, error) {
	return f.m.prefillLogits(ctx, ids, f.cache)
}

func (f cpuForward) Verify(ctx context.Context, ids []int) ([][]float32, error) {
	return f.m.forwardN(ctx, ids, f.cache)
}

func (f cpuForward) TruncateTo(pos int) { f.cache.TruncateTo(pos) }
