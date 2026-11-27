package cpuengine

import (
	"math"
	"runtime"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
)

// fusedPairScratch holds the per-worker serial Workspaces the fused path reuses across layers and tokens
// (each grows its own int8/f32 quant buffers once). The fan-out threshold is set above any reachable MAC
// count so every worker's call runs linalg's serial span — the parallelism here is this file's, not
// linalg's, and the two levels must not nest.
type fusedPairScratch struct {
	ws []*linalg.Workspace
}

func (s *fusedPairScratch) workspaces(n int) []*linalg.Workspace {
	for len(s.ws) < n {
		w := &linalg.Workspace{}
		w.SetThreshold(math.MaxInt / 4)
		s.ws = append(s.ws, w)
	}
	return s.ws[:n]
}

// fusedWorkers is the fan-out width: linalg's process-wide width cap (SetParallelWidth, 0 = GOMAXPROCS),
// bounded by GOMAXPROCS and by the column count — the same rule linalg's own resolveWidth applies, so a
// width sweep (TestR9_parWidthSweep) moves this path with the matmuls it replaces.
func fusedWorkers(n int) int {
	g := runtime.GOMAXPROCS(0)
	w := linalg.ParallelWidth()
	if w <= 0 || w > g {
		w = g
	}
	if n < 2*w {
		w = max(1, n/2)
	}
	return w
}

// GatedMLPFusedGateUp computes scr.gate = silu(gate·h) * (up·h) in one fork/join. It reports false,
// having touched nothing, when its preconditions do not hold — the caller then runs the normal path.
func GatedMLPFusedGateUp(h []float32, lw *darch.LayerWeights, arch *darch.Architecture, scr *DecodeScratch) bool {
	return gatedMLPFusedGateUpAct(h, lw, arch.Act, scr)
}

// gatedMLPFusedGateUpAct is GatedMLPFusedGateUp with the activation given: the program walker's form.
func gatedMLPFusedGateUpAct(h []float32, lw *darch.LayerWeights, act darch.ActKind, scr *DecodeScratch) bool {
	if activeHooks().Int4FloatAct {
		return false
	}
	if act != darch.ActSiLU || !IsW4A8(&lw.GateProj) || !IsW4A8(&lw.UpProj) {
		return false
	}
	if lw.GateProj.Int4Layout() != "canonical" || lw.UpProj.Int4Layout() != "canonical" {
		return false // repacked layouts (arm64 row4, amd64 split-half) need aikit's own per-arch dispatch
	}
	q4g, sg, group, _ := lw.GateProj.Int4F16()
	q4u, su, groupU, _ := lw.UpProj.Int4F16()
	N, K := lw.GateProj.Rows(), lw.GateProj.Cols()
	if group != groupU || lw.UpProj.Rows() != N || lw.UpProj.Cols() != K || group <= 0 || K%group != 0 ||
		len(h) < K || N > len(scr.gate) || N > len(scr.up) {
		return false
	}
	bpr, ng := (K+1)/2, K/group
	gate, up := scr.gate[:N], scr.up[:N]

	w := fusedWorkers(N)
	wss := scr.fusedPair.workspaces(w)
	// h is quantized once for every worker's gate and up chunk: each MatmulBTW4A8F16Into would otherwise re-quantize all of
	// h, 2w times a layer. The workers' workspaces carry the weight's activation group, or none (then aikit's process
	// default applies), so the block is quantized with the group they resolve to; the kernels only read it. activeHooks().W4A8PreOff
	// (tests only) restores a quantization per call.
	pre := !activeHooks().W4A8PreOff
	if pre {
		ag := lw.GateProj.ActQuantGroup()
		if ag == 0 {
			ag = linalg.ActQuantGroup()
		}
		linalg.QuantizeActQ(h, 1, K, ag, &scr.gateUpQ)
		if c := activeHooks().W4A8PreCalls; c != nil {
			c.Add(2)
		} // gate and up, every worker's chunk of each, on the one block
	}
	chunk := func(i int) {
		j0, j1 := N*i/w, N*(i+1)/w
		if j0 >= j1 {
			return
		}
		wss[i].SetActQuantGroup(lw.GateProj.ActQuantGroup())
		if pre {
			linalg.MatmulBTW4A8F16Pre(wss[i], &scr.gateUpQ, q4g[j0*bpr:j1*bpr], sg[j0*ng:j1*ng], gate[j0:j1], 1, K, j1-j0, group)
			linalg.MatmulBTW4A8F16Pre(wss[i], &scr.gateUpQ, q4u[j0*bpr:j1*bpr], su[j0*ng:j1*ng], up[j0:j1], 1, K, j1-j0, group)
		} else {
			linalg.MatmulBTW4A8F16Into(wss[i], h, q4g[j0*bpr:j1*bpr], sg[j0*ng:j1*ng], gate[j0:j1], 1, K, j1-j0, group)
			linalg.MatmulBTW4A8F16Into(wss[i], h, q4u[j0*bpr:j1*bpr], su[j0*ng:j1*ng], up[j0:j1], 1, K, j1-j0, group)
		}
		Swiglu(gate[j0:j1], up[j0:j1])
	}
	if w == 1 {
		chunk(0)
		return true
	}
	var wg sync.WaitGroup
	for i := 1; i < w; i++ {
		wg.Go(func() {
			chunk(i)
		})
	}
	chunk(0)
	wg.Wait()
	return true
}
