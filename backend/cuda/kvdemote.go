//go:build cuda

package cuda

import (
	"fmt"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
)

// An idle conversation's rows move to pinned host memory when the pool releases its blocks, and come back when it returns
// (docs/measurements/kv7-demote-2026-11/README.md). The copy is a prefix snapshot's, taken whole for that slot, charged to the host
// budget the prefix snapshots share. Not on windowed KV, and not on a recurrent model, whose slot holds a state this copy does not carry.

// kvUploadFailForTest, when set, is the error UploadKV's row copy to the device fails with, for the layers it names. Only a test sets it.
var kvUploadFailForTest func(layer int) error

// kvTakeFailForTest, when set, says for which slot a block is refused, as a full pool refuses it. Only a test sets it.
var kvTakeFailForTest func(slot int) bool

// kvDemoteFailForTest, when set, is the error a demote's copy fails with. Only a test sets it.
var kvDemoteFailForTest error

// SetKVHostBudget implements decoder.ResidentKVDemoter.
func (r *cudaResident) SetKVHostBudget(b decoder.KVHostBudget) {
	_ = r.do(func() error {
		r.kvHost = b
		return nil
	})
}

// kvDemote copies slot s's valid rows to host memory, just before kvEvictIdle releases them for taker. A copy the budget or the driver
// refuses leaves the release as it always was: the conversation re-prefills on return. A slot a borrower's blocks were taken back from
// (kvStopped) keeps no copy: its rows past its share are gone. Nor does a slot holding a wrapped upload (kvFloor): it has no sliding-layer
// rows below its base, so a copy from position 0 would keep zeros. Executor thread only.
func (r *cudaResident) kvDemote(s, taker int) {
	if r.kvHost == nil || r.kvWin || r.dnet != nil || r.mamba != nil || r.kvStopped[s] {
		return
	}
	end := kvBlocksFor(r.kvValid[s]) * kvBlock // whole blocks: kvValid keeps which of their rows were written
	bytes := r.KVSnapshotBytes(end)
	if bytes <= 0 {
		return
	}
	if r.kvFloor[s] > 0 {
		logs.Warnf("cuda: KV pool: slot %d's rows not kept in host memory: its upload holds rows only for prefixes of at least %d", s, r.kvFloor[s])
		return
	}
	if !r.kvHostReserve(bytes, s, taker) {
		r.kvHost.Refused()
		return
	}
	snap, err := r.snapshotKV(s, end)
	if kvDemoteFailForTest != nil {
		if snap != nil {
			snap.Release()
		}
		snap, err = nil, kvDemoteFailForTest
	}
	if err != nil {
		r.kvHost.Release(bytes)
		logs.Warnf("cuda: KV pool: slot %d's rows not kept in host memory: %v", s, err)
		return
	}
	r.kvDropDemoted(s)
	r.kvDemoted[s] = snap
	r.kvHost.Kept()
	logs.Debugf("cuda: KV pool: kept slot %d's %d rows in host memory (%d MiB)", s, end, bytes>>20)
}

// kvHostReserve charges bytes for slot s's copy: the longest idle copies go first, never the bound slot's, taker's or a copy's source, and then
// held snapshots. When that could not make it fit it drops nothing. Executor thread only.
func (r *cudaResident) kvHostReserve(bytes int64, s, taker int) bool {
	var droppable int64
	for v, d := range r.kvDemoted {
		if d != nil && r.kvCopyDroppable(v, s, taker) {
			droppable += d.Bytes()
		}
	}
	if !r.kvHost.Fits(bytes, droppable) {
		return false
	}
	for !r.kvHost.Reserve(bytes, false) {
		v := r.kvOldestDemoted(s, taker)
		if v < 0 {
			return r.kvHost.Reserve(bytes, true)
		}
		r.kvDropDemoted(v)
		r.kvHost.Dropped()
	}
	return true
}

// kvCopyDroppable reports whether slot v's demoted copy may be dropped for room: not a's or b's, the slot the rows are for and the slot taking blocks,
// and not one a request is about to use or a copy is reading. Executor thread only.
func (r *cudaResident) kvCopyDroppable(v, a, b int) bool {
	return v != a && v != b && v != r.kvSlot && v != r.kvCopySrc
}

// kvOldestDemoted is the droppable slot (kvCopyDroppable) whose demoted copy has been idle longest; -1 when there is none. Executor thread only.
func (r *cudaResident) kvOldestDemoted(a, b int) int {
	best := -1
	for v, d := range r.kvDemoted {
		if d != nil && r.kvCopyDroppable(v, a, b) && (best < 0 || r.kvSlotUse[v] < r.kvSlotUse[best]) {
			best = v
		}
	}
	return best
}

// ReserveKVHost implements decoder.ResidentKVDemoter.
func (r *cudaResident) ReserveKVHost(n int64) bool {
	var ok bool
	_ = r.do(func() error {
		ok = r.kvHostReserve(n, -1, -1)
		return nil
	})
	return ok
}

// kvDropDemoted frees slot s's demoted copy, if any, and returns its bytes to the budget. Executor thread only.
func (r *cudaResident) kvDropDemoted(s int) {
	if s < 0 || s >= len(r.kvDemoted) || r.kvDemoted[s] == nil {
		return
	}
	bytes := r.kvDemoted[s].Bytes()
	r.kvDemoted[s].Release()
	r.kvDemoted[s] = nil
	r.kvHost.Release(bytes)
}

// kvClearReleased ends slot s's released state: it holds a new sequence, so a demoted copy of the old one is stale. Executor thread only.
func (r *cudaResident) kvClearReleased(s int) {
	r.kvEvicted[s] = false
	r.kvDropDemoted(s)
}

// kvReleasedLost reports whether slot s's first prefix rows are gone: past what its sequence wrote, short of the prefixes a wrapped upload covers (kvFloor),
// released with no demoted copy, or stopped, whose next write must start from position 0. kvValid holds through a release, so a copy answers for the rows
// that were written and not the rest of their block. Executor thread only.
func (r *cudaResident) kvReleasedLost(s, prefix int) bool {
	if prefix > r.kvValid[s] || prefix < r.kvFloor[s] || r.kvStopped[s] {
		return true
	}
	return r.kvReleased(s) && (s >= len(r.kvDemoted) || r.kvDemoted[s] == nil)
}

// kvPromote is PromoteKV's job. restoreKV frees the copy once its rows are on the device. A copy dropped for room after the caller was told
// the rows were there is an error, so the caller goes cold. Executor thread only.
func (r *cudaResident) kvPromote(s, rows int) (blocks int, err error) {
	snap := r.kvDemoted[s]
	if snap == nil {
		if r.kvReleased(s) {
			return 0, fmt.Errorf("cuda: KV pool: slot %d's rows are no longer held in host memory", s)
		}
		return 0, nil
	}
	end, valid, floor := min(kvBlocksFor(rows)*kvBlock, snap.rows), r.kvValid[s], r.kvFloor[s]
	if err := r.restoreKV(snap, s, 0, end); err != nil {
		return 0, err
	}
	r.kvValid[s], r.kvFloor[s] = min(valid, rows), floor
	return end / kvBlock, nil
}

// PromoteKV implements decoder.ResidentKVDemoter.
func (r *cudaResident) PromoteKV(slot, rows int) (blocks int, err error) {
	if slot < 0 || slot >= r.KVSlots() {
		return 0, fmt.Errorf("cuda: PromoteKV slot %d of %d", slot, r.KVSlots())
	}
	err = r.do(func() error {
		var e error
		blocks, e = r.kvPromote(slot, rows)
		return e
	})
	return blocks, err
}

// CopyDemotedKV implements decoder.ResidentKVDemoter.
func (r *cudaResident) CopyDemotedKV(src, dst, start, end int) (blocks int, held bool, err error) {
	if src < 0 || src >= r.KVSlots() || dst < 0 || dst >= r.KVSlots() || src == dst {
		return 0, false, fmt.Errorf("cuda: CopyDemotedKV slots %d -> %d of %d", src, dst, r.KVSlots())
	}
	if start < 0 || start > end || end <= 0 || start%kvBlock != 0 || end%kvBlock != 0 {
		return 0, false, fmt.Errorf("cuda: CopyDemotedKV rows [%d, %d) are not whole %d-row blocks", start, end, kvBlock)
	}
	err = r.do(func() error {
		if !r.kvReleased(src) {
			return nil
		}
		held = true
		if r.kvReleasedLost(src, end) {
			return &decoder.KVCopyDecline{Reason: fmt.Sprintf("slot %d's rows are no longer held in host memory", src)}
		}
		from := min(start, r.kvValid[dst]/kvBlock*kvBlock) // dst has no rows past those to keep
		if r.kvReleased(dst) || r.kvStopped[dst] {
			from = 0
		}
		r.kvCopySrc = src
		defer func() { r.kvCopySrc = -1 }()
		if e := r.restoreKV(r.kvDemoted[src], dst, from, end); e != nil {
			return e
		}
		r.kvFloor[dst] = max(r.kvFloor[dst], r.kvFloor[src])
		blocks = (end - from) / kvBlock
		return nil
	})
	return blocks, held, err
}

var _ decoder.ResidentKVDemoter = (*cudaResident)(nil)
