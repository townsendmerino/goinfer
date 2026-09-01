//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillDecomp splits batched-prefill time into GEMV / attention / glue on the real dense
// qwen2.5-coder-1.5b, at 128/512/2048. It decides which lever matters BEFORE any attribution of the
// residual gap. Category boundaries are stream syncs (r.prof), so the category sum runs a bit over the
// pipelined wall time — that over-count is the price of per-kernel attribution and is reported too.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestPrefillDecomp -v
func TestPrefillDecomp(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest().(*cudaResident)
	_, _, _, _, _, _, vocab := mc.Dims()

	build := func(n int) [][]float32 {
		embs := make([][]float32, n)
		var s uint32 = uint32(n*2654435761 + 1)
		for i := range embs {
			s = s*1664525 + 1013904223
			embs[i] = append([]float32(nil), mc.EmbedResidentForTest(int(s>>8)%(vocab-1))...)
		}
		return embs
	}

	t.Logf("%-6s %10s %10s %10s %10s %8s", "N", "gemv", "attn", "glue", "catSum", "attn%")
	// Depths overridable (GOINFER_DECOMP_K=128,512,2048,3900): the fixed list stops
	// at 2048, but the question this instrument is now being asked -- what share of
	// CUDA prefill is attention, and therefore what a fused/FlashAttention schedule
	// could reach -- is a LONG-CONTEXT question. Attention is O(K^2) and the weight
	// matmuls are O(K), so a share read at 2048 systematically understates it at
	// 3900, which is where the 2026-09-01 re-anchor put goinfer furthest behind.
	depths := []int{128, 512, 2048}
	if v := os.Getenv("GOINFER_DECOMP_K"); v != "" {
		depths = depths[:0]
		for _, f := range strings.Split(v, ",") {
			if k, err := strconv.Atoi(strings.TrimSpace(f)); err == nil && k > 0 {
				depths = append(depths, k)
			}
		}
	}
	for _, n := range depths {
		embs := build(n)
		// warm once (JIT/caches), then take the best of 3 by category sum.
		if _, e := rf.PrefillLast(embs, 0); e != nil {
			t.Fatalf("warm n=%d: %v", n, e)
		}
		var best prefillProf
		bestSum := time.Duration(1<<62 - 1)
		for range 3 {
			rf.prof = &prefillProf{}
			if _, e := rf.PrefillLast(embs, 0); e != nil {
				t.Fatalf("prof n=%d: %v", n, e)
			}
			p := rf.prof
			rf.prof = nil
			sum := p.gemv + p.attn + p.glue
			if sum < bestSum {
				best, bestSum = *p, sum
			}
		}
		sum := best.gemv + best.attn + best.glue
		t.Logf("%-6d %10v %10v %10v %10v %7.1f%%", n, best.gemv, best.attn, best.glue, sum,
			100*float64(best.attn)/float64(sum))
	}
}
