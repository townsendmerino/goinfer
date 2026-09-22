//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoEStreamingDecodeProfile is the "measure before building" step for docs/tasks/task-moe-streaming.md's
// two open CUDA items (gocudrv async-H2D overlap of C′ miss DMAs; P20 redirected toward the expert-DMA
// cost). Both turned out to be already-closed elsewhere this session: item 1 by
// docs/completed/aikit-subrange-async-upload.md (2026-08-28 — declined as scoped, superseded by the
// already-shipped gpu.UploadBatch, which IS what cuda/resident.go's C′ path uses today, +9.3% tok/s
// measured on the 35B), item 2 by this session's own P20 expert-major build
// (docs/measurements/p20-expert-major-m26-2026-09-21.md, 2.26-2.66x on the real M26 PREFILL). Neither
// touches DECODE (M=1, no rows to bucket by expert), so this gets a FRESH, current-code reading of the
// decode-side C′ DMA share specifically, using GOINFER_MOE_CACHE_PROF's existing stall/host/dma split,
// to check whether the "genuine H2D/compute overlap" condition that decision doc's own §6 "Revisit if"
// names has now fired. Synthetic embeddings (EmbedResidentForTest), not a real prompt — a DMA-timing
// profile does not need real tokens, only real routing (the model's own trained router still decides
// which experts, from an in-vocab embedding).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestMoEStreamingDecodeProfile -v -timeout 20m
func TestMoEStreamingDecodeProfile(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset — real 26B decode")
	}
	path := os.Getenv("GOINFER_GEMMA4_26B_GIW")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "gemma4-26b-int4.giw")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 26B .giw at %s: %v", path, err)
	}
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	t.Setenv("GOINFER_MOE_CACHE_PROF", "1")

	t0 := time.Now()
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED the 26B with C′ on")
	}
	r := rf.(*cudaResident)
	if !r.cacheExperts {
		t.Fatal("C′ expert staging is OFF")
	}
	t.Logf("loaded 26B .giw in %s, cacheSlots=%d", time.Since(t0).Round(time.Second), r.cacheSlots)

	_, _, _, _, _, _, vocab := m.Dims()
	const nNew = 48
	var s uint32 = 24601
	next := func() int {
		s = s*1664525 + 1013904223
		return int(s>>8) % (vocab - 1)
	}
	pos := 0
	genStart := time.Now()
	id := next()
	for i := 0; i < nNew; i++ {
		emb := m.EmbedResidentForTest(id)
		nextID, err := r.ForwardArgmax(emb, pos)
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		id = nextID
		pos++
	}
	genDur := time.Since(genStart)

	rate := float64(nNew) / genDur.Seconds()
	hits, misses := r.CacheStatsForTest()
	hitRate := 0.0
	if hits+misses > 0 {
		hitRate = float64(hits) / float64(hits+misses)
	}
	t.Logf("generated %d tokens in %s (%.2f tok/s), C′ hit rate %.1f%% (%d hits / %d misses)",
		nNew, genDur.Round(time.Millisecond), rate, hitRate*100, hits, misses)
	if stall, host, dma, calls := r.CacheProfForTest(); calls > 0 {
		tot := stall + host + dma
		perTok := time.Duration(int64(tot) / int64(nNew))
		t.Logf("C′ round trip over %d layer-calls: stall %s (%.1f%%) | host %s (%.1f%%) | dma %s (%.1f%%) "+
			"= %s of %s generation (%.1f%%), %s/token",
			calls, stall.Round(time.Microsecond), 100*float64(stall)/float64(tot),
			host.Round(time.Microsecond), 100*float64(host)/float64(tot),
			dma.Round(time.Microsecond), 100*float64(dma)/float64(tot),
			tot.Round(time.Millisecond), genDur.Round(time.Millisecond), 100*float64(tot)/float64(genDur),
			perTok.Round(time.Microsecond))
		t.Logf("dma-of-total-token time: %.1f%%", 100*float64(dma)/float64(genDur))
	}
}
