//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"math/rand"
	"os"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestGluQuantBlockSizeInvariant pins the claim glueQuantThreads' comment rests on: glu_quant is ONE block whose only reduction is a max
// (exact, order-independent) over per-element values, so its packed int8 output and its scale do not depend on the block size. It launches the
// kernel at 256 (the previous size) and at the production size on random gate/up rows, for both activations (SiLU, GELU-tanh) and several
// intermediate widths including ones not a multiple of the block size, and requires the outputs equal under Float32bits / int equality.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestGluQuantBlockSizeInvariant -v
func TestGluQuantBlockSizeInvariant(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a small model for a device context)")
	}
	path := modelPath("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Skip("resident path declined")
	}
	rng := rand.New(rand.NewSource(5))
	total := 0
	for _, I := range []int{4, 260, 1024, 4864, 8960, 18944, 14336} {
		g := make([]float32, I)
		u := make([]float32, I)
		for i := range g {
			g[i] = float32(rng.NormFloat64() * 3)
			u[i] = float32(rng.NormFloat64() * 3)
		}
		for _, act := range []int32{0, 1} { // ACT_GELU_TANH, ACT_SILU
			run := func(threads int) (q []int32, scale float32) {
				q = make([]int32, I/4)
				sc := make([]float32, 1)
				if e := rf.do(func() error {
					gb, ub, qb, sb, ds := rf.af(I), rf.af(I), rf.ai(I/4), rf.af(1), rf.af(I)
					if e := gpu.Upload(gb, g); e != nil {
						return e
					}
					if e := gpu.Upload(ub, u); e != nil {
						return e
					}
					if e := rf.launch(rf.fSw, onecfg(threads, threads*4), Arg(gb), Arg(ub), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)),
						gpu.ArgValue(int32(I)), gpu.ArgValue(act), Arg(qb), Arg(sb), Arg(ds)); e != nil {
						return e
					}
					if e := rf.stream.Sync(); e != nil {
						return e
					}
					if e := gpu.Download(qb, q); e != nil {
						return e
					}
					return gpu.Download(sb, sc)
				}); e != nil {
					t.Fatalf("launch(%d threads): %v", threads, e)
				}
				return q, sc[0]
			}
			qa, sa := run(256)
			qb, sb := run(glueQuantThreads)
			if math.Float32bits(sa) != math.Float32bits(sb) {
				t.Fatalf("I=%d act=%d: scale differs between 256 and %d threads: %v vs %v", I, act, glueQuantThreads, sa, sb)
			}
			for i := range qa {
				total++
				if qa[i] != qb[i] {
					t.Fatalf("I=%d act=%d: packed word %d differs between 256 and %d threads: %#x vs %#x", I, act, i, glueQuantThreads, qa[i], qb[i])
				}
			}
		}
	}
	t.Logf("glu_quant output identical at 256 and %d threads over %d packed words (7 widths x 2 activations)", glueQuantThreads, total)
}
