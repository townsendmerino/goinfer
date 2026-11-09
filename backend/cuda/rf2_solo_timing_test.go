//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestRF2SoloTiming times the greedy decode token (ForwardArgmax) on a real checkpoint through the one-sequence ragged step against
// today's launches (soloRaggedOffForTest), A B B A, at the same positions after a prefill of each depth, and requires both arms to
// pick the same ids. GOINFER_RF2_SOLO_MODEL names the checkpoint; it skips without one. It reports each arm's median ms per token
// and the ragged/today ratio of tok/s per block. Docs: docs/measurements/rf2-2026-11/g-rf2-s-final-2026-11-09/README.md, "Mechanism".
func TestRF2SoloTiming(t *testing.T) {
	path := os.Getenv("GOINFER_RF2_SOLO_MODEL")
	if path == "" {
		t.Skip("GOINFER_RF2_SOLO_MODEL is not set")
	}
	requireCUDADevice(t)
	t.Cleanup(func() { soloRaggedOffForTest = false })
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 2})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
	}
	_, _, _, _, _, _, vocab := m.Dims()
	emb := func(id int) []float32 { return m.EmbedResidentForTest(id % vocab) }
	const tokens, blocks = 64, 4
	for _, depth := range []int{128, 2048} {
		prompt := make([][]float32, depth)
		for i := range prompt {
			prompt[i] = emb(i*13 + 7)
		}
		if _, err := r.PrefillLast(context.Background(), prompt, 0); err != nil {
			t.Fatal(err)
		}
		// run decodes tokens at depth.. on one arm and returns the ids and the wall time; each call returns its id, so it is synchronous.
		run := func(today bool) ([]int, time.Duration) {
			soloRaggedOffForTest = today
			defer func() { soloRaggedOffForTest = false }()
			ids := make([]int, tokens)
			id := 1
			t0 := time.Now()
			for i := range tokens {
				var e error
				if id, e = r.ForwardArgmax(emb(id), depth+i); e != nil {
					t.Fatalf("depth %d token %d (today %v): %v", depth, i, today, e)
				}
				ids[i] = id
			}
			return ids, time.Since(t0)
		}
		run(false) // warm both arms once
		run(true)
		var ratios []float64
		var msRagged, msToday []float64
		for b := range blocks {
			order := []bool{false, true, true, false} // A B B A: A = ragged, B = today
			if b%2 == 1 {
				order = []bool{true, false, false, true}
			}
			var dr, dt time.Duration
			var idsR, idsT []int
			for _, today := range order {
				ids, d := run(today)
				if today {
					dt += d
					idsT = ids
				} else {
					dr += d
					idsR = ids
				}
			}
			for i := range idsR {
				if idsR[i] != idsT[i] {
					t.Fatalf("depth %d block %d: token %d is %d through the step and %d on today's call", depth, b, i, idsR[i], idsT[i])
				}
			}
			ratios = append(ratios, float64(dt)/float64(dr)) // tok/s ragged over tok/s today
			msRagged = append(msRagged, float64(dr.Microseconds())/1e3/(2*tokens))
			msToday = append(msToday, float64(dt.Microseconds())/1e3/(2*tokens))
		}
		t.Logf("RF2-SOLO depth %d: ragged %.3f ms/token, today %.3f ms/token, ragged/today tok/s per block %.3f (median %.3f)",
			depth, soloMedian(msRagged), soloMedian(msToday), ratios, soloMedian(ratios))
	}
}

func soloMedian(x []float64) float64 {
	s := append([]float64(nil), x...)
	sort.Float64s(s)
	if n := len(s); n%2 == 1 {
		return s[n/2]
	} else {
		return (s[n/2-1] + s[n/2]) / 2
	}
}
