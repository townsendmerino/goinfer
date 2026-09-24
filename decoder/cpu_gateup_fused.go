package decoder

import (
	"math"
	"os"
	"runtime"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
)

// cpuFusedGateUp is the fused gate+up+SwiGLU decode path for W4A8 SiLU-gated MLPs: ONE fork/join per
// layer in which every worker computes gate AND up for its own column chunk and then applies the
// activation to that same chunk, instead of two fork/joins followed by a serial activation.
// docs/measurements/cpu-decode-roofline-2026-09-23.md.
//
// It removes two costs the per-component roofline attributes to the MLP block: a fork/join (~one
// goroutine-wake stagger, aikit S-02) and the SwiGLU's 3.65 ms/token of scalar float64 exp on the 1.5B,
// which ran serial because fanning it out separately loses (cpu-decode-attribution-2026-09-22-linux.md).
// Folded into the matmul's own barrier it costs nothing extra to run in parallel.
//
// Bit-identical to the unfused path, not merely close: each output column is a self-contained int4·int8
// dot whose value does not depend on which worker computes it or how the columns are chunked (the
// contract linalg's own width/fan-out already rests on), and swiglu is elementwise on the same values.
// Nothing here reorders any arithmetic. TestGatedMLPFusedGateUp_bitIdentical pins that; the paired A/B
// harness (cpu_roofline_ab_test.go) additionally fails on any greedy-token divergence.
//
// Default per architecture (fusedGateUpDefault, cpu_tuning_{arm64,other}.go): measured on the amd64
// Ryzen 7 3700X only, so arm64 stays off until the Mac measures it. GOINFER_CPU_FUSED_GATEUP=0 forces the
// unfused path (an A/B handle and an escape hatch), =1 forces it on.
var cpuFusedGateUp = envBoolDefault("GOINFER_CPU_FUSED_GATEUP", fusedGateUpDefault)

func envBoolDefault(name string, def bool) bool {
	switch os.Getenv(name) {
	case "":
		return def
	case "0", "false", "off":
		return false
	default:
		return true
	}
}

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

// gatedMLPFusedGateUp computes scr.gate = silu(gate·h) * (up·h) in one fork/join. It reports false,
// having touched nothing, when its preconditions do not hold — the caller then runs the normal path.
func gatedMLPFusedGateUp(h []float32, lw *LayerWeights, arch *Architecture, scr *decodeScratch) bool {
	if arch.Act != ActSiLU || !isW4A8(&lw.GateProj) || !isW4A8(&lw.UpProj) {
		return false
	}
	if lw.GateProj.Int4Layout() != "canonical" || lw.UpProj.Int4Layout() != "canonical" {
		return false // repacked layouts (arm64 row4, amd64 split-half) need aikit's own per-arch dispatch
	}
	q4g, sg, group, _ := lw.GateProj.Int4()
	q4u, su, groupU, _ := lw.UpProj.Int4()
	N, K := lw.GateProj.Rows(), lw.GateProj.Cols()
	if group != groupU || lw.UpProj.Rows() != N || lw.UpProj.Cols() != K || group <= 0 || K%group != 0 ||
		len(h) < K || N > len(scr.gate) || N > len(scr.up) {
		return false
	}
	bpr, ng := (K+1)/2, K/group
	gate, up := scr.gate[:N], scr.up[:N]

	w := fusedWorkers(N)
	wss := scr.fusedPair.workspaces(w)
	chunk := func(i int) {
		j0, j1 := N*i/w, N*(i+1)/w
		if j0 >= j1 {
			return
		}
		linalg.MatmulBTW4A8Into(wss[i], h, q4g[j0*bpr:j1*bpr], sg[j0*ng:j1*ng], gate[j0:j1], 1, K, j1-j0, group)
		linalg.MatmulBTW4A8Into(wss[i], h, q4u[j0*bpr:j1*bpr], su[j0*ng:j1*ng], up[j0:j1], 1, K, j1-j0, group)
		swiglu(gate[j0:j1], up[j0:j1])
	}
	if w == 1 {
		chunk(0)
		return true
	}
	var wg sync.WaitGroup
	wg.Add(w - 1)
	for i := 1; i < w; i++ {
		go func(i int) {
			defer wg.Done()
			chunk(i)
		}(i)
	}
	chunk(0)
	wg.Wait()
	return true
}
