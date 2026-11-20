package decoder

import (
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestInt4ParThreshold_qwen05bShapesFanOut pins int4ParThreshold's effect on qwen2.5 0.5B's decode projections: aikit runs
// an M=1 int4 call serial when M*totalN*K < the Workspace threshold, so the q/k/v batch and the o projection must clear
// it, and the fanned-out batch must equal the serial one bit for bit.
func TestInt4ParThreshold_qwen05bShapesFanOut(t *testing.T) {
	const hidden, nH, nKV, hd = 896, 14, 2, 64
	qN, kvN := nH*hd, nKV*hd
	for _, sh := range []struct {
		name string
		macs int
	}{
		{"q/k/v batch", (qN + 2*kvN) * hidden},
		{"o projection", qN * hidden},
	} {
		if sh.macs < int4ParThreshold {
			t.Errorf("qwen2.5 0.5B %s: %d MACs < int4ParThreshold %d, so it runs on one core", sh.name, sh.macs, int4ParThreshold)
		}
	}

	rng := rand.New(rand.NewPCG(5, 896))
	a := make([]float32, hidden)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	ops := func() []linalg.W4A8Op {
		var out []linalg.W4A8Op
		for _, n := range []int{qN, kvN, kvN} {
			wf := make([]float32, n*hidden)
			for i := range wf {
				wf[i] = float32(rng.NormFloat64()) * 0.05
			}
			wm := linalg.QuantizeInt4(wf, n, hidden, int4GroupSize)
			q4, q4s, _, ok := Int4F32(&wm)
			if !ok {
				t.Fatal("QuantizeInt4 did not yield an int4 WeightMat")
			}
			out = append(out, linalg.W4A8Op{W4: q4, Scales: q4s, Dst: make([]float32, n), N: n})
		}
		return out
	}()
	run := func(thr int) [][]float32 {
		var ws linalg.Workspace
		ws.SetThreshold(thr)
		ws.SetWorkers(8)
		for i := range ops {
			clear(ops[i].Dst)
		}
		linalg.MatmulBTW4A8Batch(&ws, a, 1, hidden, int4GroupSize, ops)
		res := make([][]float32, len(ops))
		for i := range ops {
			res[i] = append([]float32(nil), ops[i].Dst...)
		}
		return res
	}
	serial, fanned := run(1<<24), run(int4ParThreshold)
	for i := range serial {
		for j := range serial[i] {
			if serial[i][j] != fanned[i][j] {
				t.Fatalf("op %d col %d: fanned-out %v != serial %v", i, j, fanned[i][j], serial[i][j])
			}
		}
	}
}
