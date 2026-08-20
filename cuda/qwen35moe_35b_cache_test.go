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
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestQwen36_35B_cache is the payoff run for the whole CUDA DeltaNet track: Qwen3.6-35B-A3B,
// whose ~20 GB of int4 experts do NOT fit the 8 GB 2070, decoded RESIDENT via C′ expert staging
// (experts in pinned host memory, the routed ones DMA'd into device slots per token).
//
// WHY THIS MODEL AND NOT THE DENSE ONE. Qwen3.8-27B is dense: C′ streams EXPERTS, so it does
// nothing for a model with none, and 15.3 GB of int4 dense weights simply do not fit. The MoE
// siblings are the only members of this family that an 8 GB card can host at all, and CUDA is the
// only backend with the streaming path — WebGPU has no equivalent. So this is the one combination
// where residency for this family is not merely faster but POSSIBLE.
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

	// tokenizer.Load takes a DIRECTORY; a .gguf carries its tokenizer inside the container.
	tk, err := tokenizer.Load(path)
	if strings.HasSuffix(path, ".gguf") {
		tk, err = tokenizer.LoadGGUF(path)
	}
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
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
	t1 := time.Now()
	ch, _ := m.Generate(context.Background(), ids, nNew, decoder.SamplingParams{Temperature: 0})
	var out []int
	for tok := range ch {
		out = append(out, tok)
	}
	genDur := time.Since(t1)
	txt, _ := tk.Decode(out)
	rate := float64(len(out)) / genDur.Seconds()
	t.Logf("generated %d tokens in %s (%.2f tok/s, staging-bound — see the header)", len(out), genDur.Round(time.Second), rate)
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
