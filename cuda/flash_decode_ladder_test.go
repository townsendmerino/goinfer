//go:build cuda && goinfer_testhooks

package cuda

import (
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestFlashDecodeKernelLadder times ONE layer's decode attention, exact split-KV (3 launches) versus the flash-decode
// lane at each S, on the real geometry and a real-size KV, arms interleaved round by round with the start rotated,
// N launches per timing, medians over rounds. It is a KERNEL-level instrument (it informs the lane's S and its
// gate); the registered speed decision is served tok/s via bench_peer.py, not this.
//
// K/V/q are synthetic random f32 (attention cost does not depend on values; the softmax path is data-independent
// apart from exp).
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_CUDA_FLASH_DECODE=16 GOINFER_LADDER_MODEL=<gguf> GOINFER_LADDER_DEPTHS=2048,3900,8000 \
//	  go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestFlashDecodeKernelLadder -v
func TestFlashDecodeKernelLadder(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
	}
	path := os.Getenv("GOINFER_LADDER_MODEL")
	if path == "" {
		path = modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	depths := []int{2048, 3900, 8000}
	if v := os.Getenv("GOINFER_LADDER_DEPTHS"); v != "" {
		depths = depths[:0]
		for _, f := range splitInts(v) {
			depths = append(depths, f)
		}
	}
	maxD := 0
	for _, d := range depths {
		maxD = max(maxD, d)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: maxD + 64})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Skip("resident path declined")
	}
	if rf.faSplit < 1 || rf.skScores == (Pipeline{}) {
		t.Fatalf("need GOINFER_CUDA_FLASH_DECODE=<max S> and split-KV loaded (faSplit=%d)", rf.faSplit)
	}
	maxS := rf.faSplit
	Ly := &rf.layers[0]
	hd, nKV := Ly.hd, Ly.nKV
	kvDim := nKV * hd
	rng := rand.New(rand.NewSource(3))
	fill := func(b Buffer, n int) {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		if e := gpu.Upload(b, x); e != nil {
			t.Fatalf("upload: %v", e)
		}
	}
	err = rf.do(func() error {
		fill(rf.qB, rf.nH*hd)
		fill(rf.kc[0], (maxD+1)*kvDim)
		fill(rf.vc[0], (maxD+1)*kvDim)
		return nil
	})
	if err != nil {
		t.Fatalf("fill: %v", err)
	}
	rf.skMinKeys = 0
	type arm struct {
		name string
		s    int // 0 = exact
	}
	arms := []arm{{"exact", 0}}
	for s := 1; s <= maxS; s *= 2 {
		arms = append(arms, arm{"S=" + itoa(s), s})
	}
	N, rounds := envInt("GOINFER_LADDER_N", 200), envInt("GOINFER_LADDER_ROUNDS", 15)
	t.Logf("model %s: nH=%d nKV=%d hd=%d, N=%d launches/timing, %d rounds, arms interleaved+rotated", path, rf.nH, nKV, hd, N, rounds)
	for _, D := range depths {
		pos := D - 1
		times := make([][]float64, len(arms))
		timeArm := func(a arm) float64 {
			var el time.Duration
			e := rf.do(func() error {
				run := func() error {
					if a.s == 0 {
						return rf.splitKVAttnDecode(0, pos)
					}
					rf.faSplit = a.s
					return rf.flashDecodeAttn(0, pos)
				}
				for i := 0; i < 8; i++ { // warm
					if e := run(); e != nil {
						return e
					}
				}
				if e := rf.stream.Sync(); e != nil {
					return e
				}
				t0 := time.Now()
				for i := 0; i < N; i++ {
					if e := run(); e != nil {
						return e
					}
				}
				if e := rf.stream.Sync(); e != nil {
					return e
				}
				el = time.Since(t0)
				return nil
			})
			if e != nil {
				t.Fatalf("%s at %d: %v", a.name, D, e)
			}
			return float64(el.Microseconds()) / float64(N)
		}
		for r := 0; r < rounds; r++ {
			for k := range arms {
				i := (k + r) % len(arms)
				times[i] = append(times[i], timeArm(arms[i]))
			}
		}
		base := median(times[0])
		for i, a := range arms {
			mdn := median(times[i])
			t.Logf("depth %5d  %-6s %8.1f us/layer  (exact/this = %.2fx)", D, a.name, mdn, base/mdn)
		}
	}
	rf.faSplit = maxS
}

func median(x []float64) float64 {
	y := append([]float64(nil), x...)
	sort.Float64s(y)
	return y[len(y)/2]
}

func splitInts(s string) []int {
	var out []int
	cur, have := 0, false
	for _, c := range s + "," {
		if c >= '0' && c <= '9' {
			cur, have = cur*10+int(c-'0'), true
		} else if have {
			out = append(out, cur)
			cur, have = 0, false
		}
	}
	return out
}

func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		if n := splitInts(v); len(n) == 1 {
			return n[0]
		}
	}
	return def
}
