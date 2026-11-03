//go:build cuda

package cuda

import (
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/decoder"
)

// An idle conversation's rows move to pinned host memory when the pool releases its blocks, and come back when it returns
// (docs/measurements/kv7-demote-2026-11/README.md). The copy is a prefix snapshot's, taken whole for that slot, charged to the host
// budget the prefix snapshots share. Not on windowed KV, and not on a recurrent model, which has one slot and so no release.

// SetKVHostBudget implements decoder.ResidentKVDemoter.
func (r *cudaResident) SetKVHostBudget(reserve func(int64) bool, release func(int64)) {
	_ = r.do(func() error {
		r.kvHostReserve, r.kvHostRelease = reserve, release
		return nil
	})
}

// kvDemote copies slot s's rows to host memory, just before kvEvictIdle releases them. A copy the budget or the driver refuses leaves
// the release as it always was: the conversation re-prefills on return. Executor thread only.
func (r *cudaResident) kvDemote(s int) {
	if r.kvHostReserve == nil || r.kvWin || r.dnet != nil || r.mamba != nil {
		return
	}
	end := r.kvHeld(s) * kvBlock
	bytes := r.KVSnapshotBytes(end)
	if bytes <= 0 || !r.kvHostReserve(bytes) {
		return
	}
	snap, err := r.snapshotKV(s, end)
	if err != nil {
		r.kvHostRelease(bytes)
		fmt.Fprintf(os.Stderr, "cuda: KV pool: slot %d's rows not kept in host memory: %v\n", s, err)
		return
	}
	r.kvDropDemoted(s)
	r.kvDemoted[s] = snap
	fmt.Fprintf(os.Stderr, "cuda: KV pool: kept slot %d's %d rows in host memory (%d MiB)\n", s, end, bytes>>20)
}

// kvDropDemoted frees slot s's demoted copy, if any, and returns its bytes to the budget. Executor thread only.
func (r *cudaResident) kvDropDemoted(s int) {
	if s < 0 || s >= len(r.kvDemoted) || r.kvDemoted[s] == nil {
		return
	}
	bytes := r.kvDemoted[s].Bytes()
	r.kvDemoted[s].Release()
	r.kvDemoted[s] = nil
	r.kvHostRelease(bytes)
}

// kvClearReleased ends slot s's released state: it holds a new sequence, so a demoted copy of the old one is stale. Executor thread only.
func (r *cudaResident) kvClearReleased(s int) {
	r.kvEvicted[s] = false
	r.kvDropDemoted(s)
}

// kvReleasedLost reports whether slot s's first prefix rows are gone: its blocks were released with no demoted copy that holds them.
// Executor thread only.
func (r *cudaResident) kvReleasedLost(s, prefix int) bool {
	if !r.kvReleased(s) {
		return false
	}
	return s >= len(r.kvDemoted) || r.kvDemoted[s] == nil || prefix > r.kvDemoted[s].rows
}

// kvPromote puts slot s's demoted rows back and frees the host copy; 0 blocks when it holds none. A failure leaves the slot released,
// and its rows gone. Executor thread only.
func (r *cudaResident) kvPromote(s int) (blocks int, err error) {
	if s < 0 || s >= len(r.kvDemoted) || r.kvDemoted[s] == nil {
		return 0, nil
	}
	snap := r.kvDemoted[s]
	r.kvDemoted[s] = nil // restoreKV's kvMap may release another idle slot, never this one (kvTake), and this one is no longer demoted
	defer func() {
		bytes := snap.Bytes()
		snap.Release()
		r.kvHostRelease(bytes)
	}()
	if err := r.restoreKV(snap, s, snap.rows); err != nil {
		return 0, err
	}
	return snap.rows / kvBlock, nil
}

// PromoteKV implements decoder.ResidentKVDemoter.
func (r *cudaResident) PromoteKV(slot int) (blocks int, err error) {
	if slot < 0 || slot >= r.KVSlots() {
		return 0, fmt.Errorf("cuda: PromoteKV slot %d of %d", slot, r.KVSlots())
	}
	err = r.do(func() error {
		var e error
		blocks, e = r.kvPromote(slot)
		return e
	})
	return blocks, err
}

var _ decoder.ResidentKVDemoter = (*cudaResident)(nil)
