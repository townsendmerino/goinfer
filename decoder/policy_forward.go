package decoder

import (
	"context"
	"fmt"
)

// Forward is the model as an engine policy drives it (SB5, docs/tasks/task-subsystems-2026-10.md §8): a token prefix
// held in the model's KV, grown by Prefill, Verify and Step and cut back by TruncateTo. Prefill returns the logits after
// its last id; Verify returns one row per id, row i being the logits after ids[i]; Step is a single token's forward,
// which on a model may differ in the last bits from a one-id Verify, so a policy uses the one its plain path uses. A
// policy written against Forward runs unchanged on the CPU model (cpuForward), a resident backend (residentForward) and
// a scripted fake.
type Forward interface {
	Prefill(ctx context.Context, ids []int) ([]float32, error)
	Verify(ctx context.Context, ids []int) ([][]float32, error)
	Step(ctx context.Context, id int) ([]float32, error)
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

func (f cpuForward) Step(_ context.Context, id int) ([]float32, error) {
	return f.m.forward(id, f.cache)
}

func (f cpuForward) TruncateTo(pos int) { f.cache.TruncateTo(pos) }

// residentForward is Forward on m's resident backend, which is positional: it tracks the position itself. Prefill
// seeds a fresh context only (position 0), the one way the policies on it prefill.
type residentForward struct {
	m   *Model
	pos int
}

func (f *residentForward) Prefill(ctx context.Context, ids []int) ([]float32, error) {
	if f.pos != 0 {
		return nil, fmt.Errorf("decoder: residentForward.Prefill at position %d: only a fresh context is seeded", f.pos)
	}
	l, err := f.m.residentPrefillSeed(ctx, ids, 0, false)
	if err == nil {
		f.pos = len(ids)
	}
	return l, err
}

func (f *residentForward) Verify(_ context.Context, ids []int) ([][]float32, error) {
	embs := make([][]float32, len(ids))
	for i, tok := range ids {
		embs[i] = f.m.embedResident(tok)
	}
	rows, err := f.m.resident.ForwardN(embs, f.pos)
	f.pos += len(ids)
	return rows, err
}

func (f *residentForward) Step(_ context.Context, id int) ([]float32, error) {
	l, err := f.m.resident.Forward(f.m.embedResident(id), f.pos)
	f.pos++
	return l, err
}

func (f *residentForward) TruncateTo(pos int) {
	f.pos = pos
	f.m.resident.TruncateTo(pos)
}
