//go:build cuda

package cuda

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// distinctTrigramRatio is the language-agnostic degeneracy metric from the CPU 26B gate
// (decoder/gemma4_26b_real_test.go): coherent prose cycles through many trigrams (high ratio);
// a looping forward reuses a few (ratio collapses). Floor 0.70 sits above both garbage flavors.
func distinctTrigramRatio(s string) float64 {
	r := []rune(s)
	if len(r) < 3 {
		return 1
	}
	seen := make(map[string]struct{})
	total := 0
	for i := 0; i+3 <= len(r); i++ {
		seen[string(r[i:i+3])] = struct{}{}
		total++
	}
	return float64(len(seen)) / float64(total)
}

// TestGemma4_26B_cache_B is B′: the real gemma4 26B-A4B, whose ~11.4 GB of int4 experts do NOT fit
// the 8 GB 2070, decoded RESIDENT via the C′ VRAM expert staging path (experts in pinned host
// memory, routed experts DMA'd into topK device slots per token). This is the first coherent
// continuation on this track from a model that does not fit the card. It is a CORRECTNESS +
// informative-latency run, NOT a benchmark: ~714 MB/token over PCIe + ~30 D2H idx readbacks/token
// make the tok/s poor by design (staging, not caching — that is C′ step 2). Routing capture
// (G4_CAPTURE) adds a second per-layer readback, so the latency here is a LOOSE upper bound.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda ./cuda/ -run TestGemma4_26B_cache_B -v -timeout 40m
func TestGemma4_26B_cache_B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset — real 26B decode")
	}
	dir := os.Getenv("GOINFER_GEMMA4_26B")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "gemma-4-26b-a4b-it")
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("no 26B checkpoint at %s: %v", dir, err)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	t.Setenv("GOINFER_G4_CAPTURE", "1") // capture routing through the gen (adds a per-layer readback)

	// The 11.4 GB pinned-host allocation happens here (cuMemAllocHost + copy from the packed
	// weights). Time it: a slow load is expected (non-pageable, large), not a hang.
	t0 := time.Now()
	m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("Load(26B, cuda int4): %v", err)
	}
	defer m.Close()
	loadDur := time.Since(t0)
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED the 26B with C′ env on — admission regressed or experts didn't stage")
	}
	t.Logf("loaded 26B resident + C′ staging in %s (11.4 GB experts host-pinned; ~0.7 GB slots + ~1.3 GB core in VRAM)", loadDur.Round(time.Second))

	tk, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	turns := []chat.Turn{{Role: "user", Content: "What is the capital of France, and what is it famous for?"}}
	ids, err := tk.EncodeSegments(chat.Gemma4().RenderSegments("", turns), false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	// Greedy continuation through the REAL chat template (the deliverable + the routing-through-
	// generation the tiny fixtures couldn't exercise).
	r := rf.(*cudaResident)
	r.g4capIdx = nil
	t1 := time.Now()
	out, _ := m.Generate(context.Background(), ids, 64, decoder.SamplingParams{})
	gen := make([]int, 0, 64)
	for id := range out {
		gen = append(gen, id)
	}
	genDur := time.Since(t1)
	if len(gen) == 0 {
		t.Fatal("no tokens generated")
	}
	text, err := tk.Decode(gen)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	// ROUTING at 128/top-8 (the question the fixtures couldn't answer). gemv_f32_f32 makes routing
	// bit-exact at any expert count, so this should CONFIRM: the selections captured through the gen
	// must be in range and varied — a router broken at 128 would show here. A full CPU idx
	// cross-check is infeasible (~52 GB / glacial at 26B); coherence below is the end-to-end proof.
	distinct := map[uint32]struct{}{}
	for _, dec := range r.g4capIdx {
		for _, e := range dec {
			if e >= 128 {
				t.Errorf("routed expert id %d out of range [0,128) — router broken at 128/top-8", e)
			}
			distinct[e] = struct{}{}
		}
	}
	if len(r.g4capIdx) == 0 {
		t.Error("no routing captured (G4_CAPTURE wiring)")
	} else if len(distinct) < 16 {
		t.Errorf("only %d distinct experts across %d decisions — routing looks degenerate at 128/top-8", len(distinct), len(r.g4capIdx))
	} else {
		t.Logf("routing at 128/top-8: %d decisions through the gen, %d distinct experts, all in [0,128) — sane", len(r.g4capIdx), len(distinct))
	}

	// COHERENCE: known answer + degeneracy floor (the hardened chat-template check).
	tr := distinctTrigramRatio(text)
	if !strings.Contains(text, "Paris") {
		t.Errorf("continuation lacks the known answer %q: %q", "Paris", text)
	}
	if tr < 0.70 {
		t.Errorf("continuation is degenerate (distinct-trigram %.3f < 0.70): %q", tr, text)
	}
	mspt := genDur.Seconds() * 1e3 / float64(len(gen))
	hits, misses := r.CacheStatsForTest()
	hitRate := 0.0
	if hits+misses > 0 {
		hitRate = float64(hits) / float64(hits+misses)
	}
	t.Logf("B′ OK — 26B resident via C′ (%d slots/layer): %d tok, distinct-trigram %.3f, %.0f ms/tok (%.2f tok/s — "+
		"capture readback inflates this; ~714 MB/tok PCIe at nSlots=topK; informative, NOT a benchmark)\n  cont: %q",
		r.cacheSlots, len(gen), tr, mspt, 1000/mspt, text)
	t.Logf("C′ cache: %d hits / %d misses = %.1f%% hit rate (each hit is one skipped expert DMA; nSlots=%d, "+
		"topK=%d, nE=128) — set GOINFER_MOE_CACHE_SLOTS>topK for cross-token reuse", hits, misses, hitRate*100, r.cacheSlots, r.topK)
}
