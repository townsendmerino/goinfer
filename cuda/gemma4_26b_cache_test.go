//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
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
	// Routing at 128/top-8 is already confirmed (the committed B′ run); leave G4_CAPTURE OFF by
	// default so the tok/s is HONEST (capture adds a second per-layer readback). Set GOINFER_G4_CAPTURE=1
	// externally to re-run the routing assertion.
	routingCheck := os.Getenv("GOINFER_G4_CAPTURE") != ""

	// The 11.4 GB pinned-host allocation happens here (cuMemAllocHost + copy from the packed
	// weights). Time it: a slow load is expected (non-pageable, large), not a hang.
	t0 := time.Now()
	opts := decoder.Options{Backend: "cuda", Quant: "int4"}
	// GOINFER_26B_CTX trades resident KV for expert-cache slots, which is the only knob that moves
	// the slot count on a card this size: KV is allocated BEFORE the cache is sized, so every
	// position costs 480 KB (30 layers x 8 kv heads x 256 head_dim x 2 x 4 B) that the cache then
	// cannot have. At the 4096 default that is 2.0 GB of the 8 GB card.
	if v, err := strconv.Atoi(os.Getenv("GOINFER_26B_CTX")); err == nil && v > 0 {
		opts.ResidentContext = v
		t.Logf("resident KV capped to %d positions (GOINFER_26B_CTX) — %.2f GB of KV",
			v, float64(v)*480*1024/1e9)
	}
	m, err := decoder.Load(dir, opts)
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
	r.g4capIdx, r.g4capLayer = nil, nil
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
	// G33 Tier 1: dump the routing trace so the cache replay can run offline. Only when asked —
	// the file is a few MB and the capture itself already inflates ms/tok (it syncs per layer and
	// disables CUDA graphs), so a run that writes this is NOT a timing run.
	if dst := os.Getenv("GOINFER_G4_TRACE_OUT"); dst != "" && len(r.g4capIdx) > 0 {
		type dec struct {
			Layer int      `json:"layer"`
			Idx   []uint32 `json:"idx"`
		}
		out := struct {
			TopK      int   `json:"topK"`
			NE        int   `json:"nE"`
			Slots     int   `json:"slots"`
			Tokens    int   `json:"tokens"`
			Decisions []dec `json:"decisions"`
		}{r.topK, 128, r.cacheSlots, len(gen), make([]dec, 0, len(r.g4capIdx))}
		for i := range r.g4capIdx {
			out.Decisions = append(out.Decisions, dec{r.g4capLayer[i], r.g4capIdx[i]})
		}
		b, e := json.Marshal(out)
		if e != nil {
			t.Fatalf("trace marshal: %v", e)
		}
		if e := os.WriteFile(dst, b, 0o644); e != nil {
			t.Fatalf("trace write: %v", e)
		}
		t.Logf("G33 trace: %d decisions over %d tokens, topK=%d slots=%d -> %s (%d bytes)",
			len(out.Decisions), len(gen), r.topK, r.cacheSlots, dst, len(b))
	}

	text, err := tk.Decode(gen)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	// ROUTING at 128/top-8 (opt-in via GOINFER_G4_CAPTURE — off by default so the tok/s stays honest).
	// gemv_f32_f32 makes routing bit-exact at any expert count, so this CONFIRMS: selections in range
	// and varied — a router broken at 128 would show here. A full CPU idx cross-check is infeasible
	// (~52 GB / glacial at 26B); coherence below is the end-to-end proof.
	if routingCheck {
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
			t.Errorf("only %d distinct experts across %d decisions — routing degenerate at 128/top-8", len(distinct), len(r.g4capIdx))
		} else {
			t.Logf("routing at 128/top-8: %d decisions, %d distinct experts, all in [0,128) — sane", len(r.g4capIdx), len(distinct))
		}
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
	capNote := "capture OFF"
	if routingCheck {
		capNote = "capture ON (readback inflates ms/tok)"
	}
	t.Logf("B′ OK — 26B resident via C′ (%d slots/layer): %d tok, distinct-trigram %.3f, %.0f ms/tok (%.2f tok/s — %s; "+
		"sync H2D; informative capability number, NOT a same-machine benchmark)\n  cont: %q",
		r.cacheSlots, len(gen), tr, mspt, 1000/mspt, capNote, text)
	t.Logf("C′ cache: %d hits / %d misses = %.1f%% hit rate (each hit is one skipped expert DMA; nSlots=%d, "+
		"topK=%d, nE=128) — set GOINFER_MOE_CACHE_SLOTS>topK for cross-token reuse", hits, misses, hitRate*100, r.cacheSlots, r.topK)

	// PRICE THE ROUTING ROUND TRIP. cacheProf has existed and been read by nothing; this wires it
	// up. It decomposes loadRoutedExperts into the three things it actually does per MoE layer per
	// token — the pipeline drain, the host-side slot bookkeeping, and the expert DMAs — which is the
	// number that decides whether speculative prefetch is worth its complexity (G30, spec/10's
	// standing verdict). Zero unless GOINFER_MOE_CACHE_PROF is set.
	if stall, host, dma, calls := r.CacheProfForTest(); calls > 0 {
		tot := stall + host + dma
		nsPerTok := float64(genDur) / float64(len(gen)) // ns per token
		pct := func(d time.Duration) float64 { return float64(d) / float64(genDur) * 100 }
		t.Logf("C′ ROUND TRIP over %d calls: stall %v (%.1f%% of decode) + host %v (%.1f%%) + dma %v (%.1f%%) "+
			"= %v (%.1f%%); per call %v; per token %.2f ms of a %.2f ms token",
			calls, stall.Round(time.Millisecond), pct(stall), host.Round(time.Millisecond), pct(host),
			dma.Round(time.Millisecond), pct(dma), tot.Round(time.Millisecond), pct(tot),
			(tot / time.Duration(calls)).Round(time.Microsecond),
			float64(tot)/float64(len(gen))/1e6, nsPerTok/1e6)

		// PHASE 0 ANSWERED IT, AND THE FIX IS IN. The question was whether the expert DMA is
		// bandwidth-bound or per-call-overhead bound: each miss used to issue FOUR blocking
		// null-stream uploads, and the tiny scale copy costing comparable per-call time to the big
		// weight copy would mean fixed per-call cost dominated. It did, so the copies are now
		// QUEUED and issued per layer by one gpu.UploadBatch. Copy count is therefore unchanged
		// and sync count is what moved, which is why the two are reported separately below.
		wB, sB, wC, sC := r.UploadProfForTest()
		batchTime, syncCalls := r.BatchProfForTest()
		rate := func(b uint64, d time.Duration) float64 {
			if d == 0 {
				return 0
			}
			return float64(b) / d.Seconds() / 1e9
		}
		per := func(d time.Duration, c uint64) time.Duration {
			if c == 0 {
				return 0
			}
			return (d / time.Duration(c)).Round(time.Microsecond)
		}
		t.Logf("C′ UPLOAD SPLIT: weights %d copies, %.1f MB | scales %d copies, %.1f MB",
			wC, float64(wB)/1e6, sC, float64(sB)/1e6)
		// THE GATE FOR THE BATCHING CHANGE. Copies unchanged; synchronizes down from one per copy
		// (wC+sC) to one per layer-with-a-miss. Printed as a RATIO so it reads the same whatever
		// the generation length, and so a regression to per-copy syncs is obvious rather than
		// buried in a duration.
		t.Logf("C′ UPLOAD BATCHING: %d copies -> %d synchronizes (%.1fx fewer), %v total (%.2f GB/s, %v/sync)",
			wC+sC, syncCalls, float64(wC+sC)/math.Max(float64(syncCalls), 1),
			batchTime.Round(time.Millisecond), rate(wB+sB, batchTime), per(batchTime, syncCalls))
	}
}
