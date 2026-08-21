//go:build cuda && goinfer_testhooks

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
	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestQwen36_35B_cache is the payoff run for the whole CUDA DeltaNet track: Qwen3.6-35B-A3B,
// whose ~20 GB of int4 experts do NOT fit the 8 GB 2070, decoded RESIDENT via C′ expert staging
// (experts in pinned host memory, the routed ones DMA'd into device slots per token).
//
// WHY THIS MODEL AND NOT THE DENSE ONE. Qwen3.8-27B is dense: C′ streams EXPERTS, so it does
// nothing for a model with none, and 15.3 GB of int4 dense weights simply do not fit. The MoE
// siblings are the only members of this family an 8 GB card can host at all, so this is the one
// combination where residency for this family is not merely faster but POSSIBLE.
//
// CORRECTION (2026-08-20): an earlier version of this comment said CUDA is "the only backend with
// the streaming path". That is wrong about Metal, which has its own shipped per-layer LRU expert
// pager (metal/expertpool.go, built for the same gemma4-26B problem) — today wired to the g4moe
// path rather than generic MoE, so it would need generalizing, but the mechanism is there. WebGPU
// is the one with no equivalent.
//
// It needed three things that did not exist a day ago: the DeltaNet mixer kernels, their wiring,
// and FeatMoEGatedShared (the sigmoid-gated shared expert this family carries). Any one missing
// and the model declines to the CPU path.
//
// CORRECTNESS + INFORMATIVE LATENCY, NOT A BENCHMARK. Per token the router picks 8 of 256 experts
// in each of 40 layers; at ~3.15 M int4 params per expert that is roughly 630 MB of PCIe traffic
// per token before any reuse, plus a D2H routing readback per layer. The tok/s here is a floor set
// by staging, and improving it is C′ step 2's LRU cache, not this test's business.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags "cuda goinfer_testhooks" ./cuda/ -run TestQwen36_35B_cache -v -timeout 90m
func TestQwen36_35B_cache(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset — real 35B decode")
	}
	// Default to the pre-quantized .giw: the bf16 safetensors is 67 GB and the loader holds the
	// mapping resident (P13), which on a 62 GB box is a swap experiment rather than a test.
	path := os.Getenv("GOINFER_QWEN36_35B")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen3.6-35b-a3b-int4.giw")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 35B checkpoint at %s: %v", path, err)
	}
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	// SLOT COUNT is the whole C′ step-2 lever, and its default defeats it: cacheSlots defaults to
	// topK, so each token's 8 routed experts evict the previous token's 8 and the LRU never hits.
	// Raising it trades VRAM for reuse — the allocator caps to what actually fits, so an
	// over-request degrades to fewer slots rather than an OOM.
	slots := os.Getenv("GOINFER_MOE_CACHE_SLOTS")
	if slots != "" {
		t.Setenv("GOINFER_MOE_CACHE_SLOTS", slots)
	}

	// The ~20 GB pinned-host allocation happens inside Load (cuMemAllocHost + copy). Time it: a
	// slow load is expected (non-pageable, large), not a hang.
	t0 := time.Now()
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("Load(35B, cuda int4): %v", err)
	}
	defer m.Close()
	loadDur := time.Since(t0)
	rf := m.ResidentForwardForTest()
	if rf == nil {
		// Non-vacuity: without this the test would quietly measure the CPU path and report it as
		// a streaming success — the exact failure the decode-path string exists to prevent.
		t.Fatalf("cuda resident DECLINED the 35B with C′ on — decode path %q; decline: %s",
			m.DecodePath(), m.ResidentDecline())
	}
	r := rf.(*cudaResident)
	if !r.cacheExperts {
		t.Fatal("resident built but C′ expert staging is OFF — this would be measuring a model " +
			"that fit VRAM after all, which is not what this test claims")
	}
	t.Logf("loaded 35B resident + C′ staging in %s (decode path %s)", loadDur.Round(time.Second), m.DecodePath())

	// THREE CONTAINERS, THREE LOADERS — and the .giw arm was missing while .giw is this test's
	// DEFAULT path, so the default invocation could not reach the decode it exists to measure. It
	// failed as `parse …int4.giw: invalid character 'G'`, i.e. tokenizer.Load reading the bundle
	// magic as JSON, which reads like a corrupt checkpoint rather than a missing case. The runs that
	// passed all set GOINFER_QWEN36_35B to the .gguf, which took the arm that existed.
	//
	//   directory  HF tokenizer.json on disk
	//   .gguf      tokenizer lives in the container's metadata
	//   .giw       the bundle carries the source GGUF's metadata half; hand those bytes to the
	//              same GGUF tokenizer parser
	var tk *tokenizer.Tokenizer
	switch {
	case strings.HasSuffix(path, ".giw"):
		var tokBytes []byte
		if tokBytes, err = giw.ReadTokFile(path); err == nil {
			tk, err = tokenizer.LoadGGUFBytes(tokBytes)
		}
	case strings.HasSuffix(path, ".gguf"):
		tk, err = tokenizer.LoadGGUF(path)
	default:
		tk, err = tokenizer.Load(path)
	}
	if err != nil {
		t.Fatalf("tokenizer (%s): %v", filepath.Ext(path), err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatalf("chat template: %v", err)
	}
	turns := []chat.Turn{{Role: "user", Content: "What is the capital of France, and what is it famous for?"}}
	ids, err := tk.EncodeSegments(tmpl.RenderSegments("", turns), false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	t.Logf("prompt: %d tokens via the %q template", len(ids), tmpl.Name())

	nNew := 48
	r.launchN = 0
	t1 := time.Now()
	ch, _ := m.Generate(context.Background(), ids, nNew, decoder.SamplingParams{Temperature: 0})
	var out []int
	for tok := range ch {
		out = append(out, tok)
	}
	genDur := time.Since(t1)
	txt, _ := tk.Decode(out)
	rate := float64(len(out)) / genDur.Seconds()
	// Cache accounting is the point of the slot knob: a miss is one expert's H2D DMA, so the hit
	// rate IS the fraction of per-token expert bytes reuse saves. Reporting tok/s without it would
	// leave the reader unable to tell "the DMA got faster" from "there was less DMA".
	hits, misses := r.CacheStatsForTest()
	hitRate := 0.0
	if hits+misses > 0 {
		hitRate = float64(hits) / float64(hits+misses)
	}
	t.Logf("generated %d tokens in %s (%.2f tok/s)", len(out), genDur.Round(time.Second), rate)
	// DISPATCH COST, from the counter the runner already keeps and the 11.8 µs/launch
	// TestCUDA_launchCost measures through the purego FFI. Reported next to the C′ split so the
	// token decomposes without a second instrument: whatever stall+host+dma+launch does not
	// account for is GPU-side.
	const usPerLaunch = 11.8
	lpt := float64(r.launchN) / float64(max(len(out), 1))
	launchMs := lpt * usPerLaunch / 1000 * float64(len(out))
	t.Logf("dispatch: %d launches / %d tokens = %.0f/token ≈ %.0f ms of host issue "+
		"at %.1f µs/launch (%.0f%% of generation)", r.launchN, len(out), lpt, launchMs,
		usPerLaunch, 100*launchMs/float64(genDur.Milliseconds()))
	if stall, host, dma, calls := r.CacheProfForTest(); calls > 0 {
		tot := stall + host + dma
		t.Logf("C′ round trip over %d layer-calls: stall %s (%.0f%%) | host %s (%.0f%%) | dma %s (%.0f%%) "+
			"= %s of %s generation (%.0f%%)", calls,
			stall.Round(time.Millisecond), 100*float64(stall)/float64(tot),
			host.Round(time.Millisecond), 100*float64(host)/float64(tot),
			dma.Round(time.Millisecond), 100*float64(dma)/float64(tot),
			tot.Round(time.Millisecond), genDur.Round(time.Millisecond), 100*float64(tot)/float64(genDur))
	}
	t.Logf("C′ cache: %d slots/layer of %d experts — %d hits / %d misses (%.1f%% hit rate, "+
		"~%.0f MB of expert DMA per token)", r.cacheSlots, r.nE, hits, misses, hitRate*100,
		float64(misses)/float64(max(len(out), 1))*1.97)
	t.Logf("continuation: %q", txt)

	if len(out) == 0 {
		t.Fatal("no tokens generated")
	}
	// COHERENCE, two ways, because each misses what the other catches. The trigram ratio catches a
	// looping/degenerate forward but passes fluent nonsense; the factual check catches a forward
	// that is fluent and wrong. A model streaming the wrong experts per token can be either.
	if ratio := distinctTrigramRatio(txt); ratio < 0.70 {
		t.Errorf("degenerate output: distinct-trigram ratio %.3f < 0.70 — the continuation is looping, "+
			"which is what a mis-DMA'd or mis-indexed expert slot looks like", ratio)
	}
	if !strings.Contains(strings.ToLower(txt), "paris") {
		t.Errorf("continuation does not name Paris — fluent but wrong is the OTHER failure mode of a "+
			"streaming bug, and the trigram ratio cannot see it: %q", txt)
	}
}
