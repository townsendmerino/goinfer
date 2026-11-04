//go:build cuda

package cuda

import "time"

// The tower profile on CUDA (docs/tasks/task-multimodal-support-2026-10.md, "Step 0's instrument on CUDA"): with
// profiling on, towerOps drains its queue at every change of kernel class and attributes the host time since the
// previous drain to the class that was running. aikit's Event has no elapsed-time call, so this is a wall-clock split
// with the queue drained at each class boundary, and the launch latency of the first kernel after a drain sits inside
// its class. With profiling off (towerOps.prof nil) every hook is a nil check and nothing else changes.

type towerClass int

const (
	clsGEMM towerClass = iota // the projections, with their bias and add epilogues
	clsAttn                   // attention
	clsNorm                   // LayerNorm and RMSNorm
	clsElem                   // RoPE, activations, adds, scales, position adds
	nTowerClass
)

var towerClassNames = [nTowerClass]string{"gemm", "attention", "norm", "elementwise"}

type towerProf struct {
	acc         [nTowerClass]time.Duration
	cur         towerClass
	last        time.Time
	open        bool
	gemmFLOPs   float64
	attnFLOPs   float64
	launchCount int
}

// cls marks the start of an op of class c: a change of class drains the queue and closes the running class's interval.
func (t *towerOps) cls(c towerClass) {
	p := t.prof
	if p == nil {
		return
	}
	p.launchCount++
	if !p.open {
		p.open, p.cur, p.last = true, c, time.Now()
		return
	}
	if p.cur != c {
		t.profFlush()
		p.cur = c
	}
}

// profFlush drains the queue and charges the time since the last drain to the running class.
func (t *towerOps) profFlush() {
	p := t.prof
	if p == nil || !p.open {
		return
	}
	_ = t.q.Sync() // a launch error is latched in t.err and reported by finish
	now := time.Now()
	p.acc[p.cur] += now.Sub(p.last)
	p.last = now
}

// profGEMM and profAttn add a call's FLOPs.
func (t *towerOps) profGEMM(M, N, K int) {
	if t.prof != nil {
		t.prof.gemmFLOPs += 2 * float64(M) * float64(N) * float64(K)
	}
}

func (t *towerOps) profAttn(np, nH, hd int) {
	if t.prof != nil {
		t.prof.attnFLOPs += 4 * float64(np) * float64(np) * float64(hd) * float64(nH)
	}
}

// startProf attaches a fresh profile; stopProf detaches and returns it. A forward ends in finish(), which closes the last interval on the executor thread (the CUDA context is bound to it,
// so stopProf does not drain from the caller's thread). Both run while no call is in flight.
func (t *towerOps) startProf() { t.prof = &towerProf{} }

func (t *towerOps) stopProf() *towerProf {
	p := t.prof
	t.prof = nil
	return p
}
