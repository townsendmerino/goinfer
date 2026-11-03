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
// (kvStopped) keeps no copy: its rows past its share are gone. Executor thread only.
func (r *cudaResident) kvDemote(s, taker int) {
	if r.kvHost == nil || r.kvWin || r.dnet != nil || r.mamba != nil || r.kvStopped[s] {
		return
	}
	end := kvBlocksFor(r.kvValid[s]) * kvBlock
	bytes := r.KVSnapshotBytes(end)
	if bytes <= 0 {
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
		fmt.Fprintf(os.Stderr, "cuda: KV pool: slot %d's rows not kept in host memory: %v\n", s, err)
		return
	}
	r.kvDropDemoted(s)
	r.kvDemoted[s] = snap
	r.kvHost.Kept()
	fmt.Fprintf(os.Stderr, "cuda: KV pool: kept slot %d's %d rows in host memory (%d MiB)\n", s, end, bytes>>20)
}

// kvHostReserve charges bytes for slot s's copy: the longest idle copies go first, never taker's, and then held snapshots. Executor thread only.
func (r *cudaResident) kvHostReserve(bytes int64, s, taker int) bool {
	if bytes > r.kvHost.Limit() {
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

// kvOldestDemoted is the slot, other than a and b, whose demoted copy has been idle longest; -1 when there is none. Executor thread only.
func (r *cudaResident) kvOldestDemoted(a, b int) int {
	best := -1
	for s, d := range r.kvDemoted {
		if d != nil && s != a && s != b && (best < 0 || r.kvSlotUse[s] < r.kvSlotUse[best]) {
			best = s
		}
	}
	return best
}

// DropDemotedKV implements decoder.ResidentKVDemoter.
func (r *cudaResident) DropDemotedKV(need int64) {
	_ = r.do(func() error {
		for need > 0 {
			v := r.kvOldestDemoted(-1, -1)
			if v < 0 {
				return nil
			}
			need -= r.kvDemoted[v].Bytes()
			r.kvDropDemoted(v)
			r.kvHost.Dropped()
		}
		return nil
	})
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

// kvReleasedLost reports whether slot s's first prefix rows are gone: its blocks were released with no demoted copy that holds them.
// Executor thread only.
func (r *cudaResident) kvReleasedLost(s, prefix int) bool {
	if !r.kvReleased(s) {
		return false
	}
	return s >= len(r.kvDemoted) || r.kvDemoted[s] == nil || prefix > r.kvDemoted[s].rows
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
	end := snap.rows
	if rows > 0 {
		end = min(kvBlocksFor(rows)*kvBlock, end)
	}
	if err := r.restoreKV(snap, s, end); err != nil {
		return 0, err
	}
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

var _ decoder.ResidentKVDemoter = (*cudaResident)(nil)
