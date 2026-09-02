package decoder

import (
	"encoding/json"
	"testing"
)

// M-05: attention_chunk_size was read from config and never applied. The RoPE layers attended
// [0, pos] on every position, so from position C on they saw keys HF's block-diagonal chunked
// mask removes. Nothing caught it because every parity gate uses a sequence shorter than C,
// where chunked and full-causal are the same function.
//
// WHAT IS PINNED HERE is the mask this build implements: a query at p attends [(p/C)*C, p] on a
// RoPE layer, and [0, p] on a NoPE layer. The audit rates the HF semantics medium-confidence
// (recalled, not read), so this states the rule explicitly rather than burying it — if HF turns
// out to chunk differently, this test names exactly what to change.
func TestLlama4_chunkedAttentionStart(t *testing.T) {
	const C = 8192
	arch := &Architecture{
		llama4: &llama4Params{
			chunkSize: C,
			useRope:   []bool{true, false}, // layer 0 chunked, layer 1 NoPE
		},
	}
	for name, tc := range map[string]struct {
		layer, pos, want int
	}{
		"first chunk, start":        {0, 0, 0},
		"first chunk, last":         {0, C - 1, 0},
		"second chunk, first token": {0, C, C},
		"second chunk, middle":      {0, C + 5, C},
		"third chunk":               {0, 2*C + 1, 2 * C},
		// A NoPE layer is full-causal at every position — HF applies the chunked mask only to
		// use_rope layers, and a fix that chunked everything would pass every case above.
		"NoPE layer stays full causal":      {1, 3 * C, 0},
		"NoPE layer stays full causal, mid": {1, C + 7, 0},
	} {
		t.Run(name, func(t *testing.T) {
			if got := arch.attnChunkStart(tc.layer, tc.pos); got != tc.want {
				t.Errorf("layer %d pos %d: start %d, want %d", tc.layer, tc.pos, got, tc.want)
			}
		})
	}

	// A checkpoint that does not set the field must be unaffected — 0 means no chunking, and
	// every non-llama4 family must be untouched by this change.
	t.Run("chunkSize 0 does not chunk", func(t *testing.T) {
		a := &Architecture{llama4: &llama4Params{chunkSize: 0, useRope: []bool{true}}}
		if got := a.attnChunkStart(0, 99999); got != 0 {
			t.Errorf("start %d with no chunk size, want 0", got)
		}
	})
	t.Run("a non-llama4 arch does not chunk", func(t *testing.T) {
		if got := (&Architecture{}).attnChunkStart(0, 99999); got != 0 {
			t.Errorf("start %d on a plain arch, want 0", got)
		}
	})
	// Layers past useRope's length must not panic — a config whose no_rope_layers list is
	// shorter than num_hidden_layers is malformed, but indexing it is still an out-of-range.
	t.Run("layer past the useRope list", func(t *testing.T) {
		if got := arch.attnChunkStart(99, C+1); got != 0 {
			t.Errorf("start %d for an out-of-range layer, want 0", got)
		}
	})
}

// THE CALL SITE, not just the predicate. attnChunkStart returning the right number proves
// nothing about whether attendQuery consults it — the same gap that made M-25's component test
// vouch for behaviour the system did not produce.
//
// Driven through the real attendQuery with a tiny chunk so the boundary is reachable: key 0
// carries a distinctive value, and a query at a position in the SECOND chunk must not see it.
func TestLlama4_attendQueryHonoursTheChunk(t *testing.T) {
	const (
		nH, nKV, hd = 1, 1, 2
		chunk       = 4
		nLayers     = 1
	)
	arch := &Architecture{
		NumHeads: nH, NumKVHeads: nKV, HeadDim: hd, NumLayers: nLayers,
		AttnScale: 1,
		llama4:    &llama4Params{chunkSize: chunk, useRope: []bool{true}},
	}
	// Positions 0..5. Key 0 is the one the chunk mask must exclude for a query at pos 4/5;
	// its VALUE is the marker, so if it is attended the context carries it.
	const nPos = 6
	c := NewKVCache(nLayers, nKV, hd, 0, nPos)
	for p := range nPos {
		k := []float32{1, 0}
		v := []float32{0, 0}
		if p == 0 {
			v = []float32{1000, 1000} // the out-of-chunk marker
		}
		c.Append(0, k, v)
	}
	c.scr = newDecodeScratch(arch)

	attend := func(pos int, a *Architecture) []float32 {
		ctx := make([]float32, nH*hd)
		q := []float32{1, 0}
		attendQuery(q, ctx, c.scr.scoresBuf(nPos), c, 0, pos, true, a)
		return ctx
	}

	// pos 4 is in the SECOND chunk ([4,7]), so key 0 is masked out.
	got := attend(4, arch)
	if got[0] > 1 {
		t.Errorf("query at pos 4 attended position 0 (ctx=%v): the chunk mask is not applied in "+
			"attendQuery, only computed (M-05)", got)
	}

	// The premise: WITHOUT chunking the same query does see it. If this stops holding the test
	// above passes for the wrong reason — the marker would be invisible either way.
	plain := &Architecture{NumHeads: nH, NumKVHeads: nKV, HeadDim: hd, NumLayers: nLayers, AttnScale: 1}
	if ref := attend(4, plain); ref[0] <= 1 {
		t.Fatalf("premise broke: an unchunked query at pos 4 does not see position 0 either "+
			"(ctx=%v), so this fixture cannot detect the mask", ref)
	}

	// And pos 3 is in the FIRST chunk, so it SHOULD still see position 0 even when chunked —
	// a fix that simply narrowed attention everywhere would pass the first check.
	if got := attend(3, arch); got[0] <= 1 {
		t.Errorf("query at pos 3 (same chunk as 0) does NOT see position 0 (ctx=%v): the mask "+
			"is too tight", got)
	}
}

// THE WIRING, config → Architecture. The two tests above build llama4Params by hand, so both
// pass with `chunkSize: cfg.AttentionChunkSize` deleted from the adapter — measured, and it is
// the same gap twice over: a component that is correct, a call site that is correct, and
// nothing checking that the VALUE reaches either. attention_chunk_size being read into Config
// and then dropped is precisely what M-05 was.
func TestLlama4_chunkSizeReachesTheArchitecture(t *testing.T) {
	cfg := &Config{
		ModelType: "llama4_text", HiddenDim: 64, NumLayers: 4, NumHeads: 8, NumKVHeads: 4, HeadDim: 8,
		VocabSize: 128, IntermediateDim: 128, IntermediateSizeMLP: 192, RMSNormEps: 1e-5,
		NumLocalExperts: 4, NumExpertsPerTok: 1, MoeLayers: []int{1, 3}, NoRopeLayers: []int{1, 1, 0, 1},
		UseQKNorm: true, AttnTemperatureTuning: true, FloorScale: 4, AttnScaleL4: 0.1,
		RopeParameters:     json.RawMessage(`{"rope_theta":10000.0,"rope_type":"default"}`),
		AttentionChunkSize: 8192,
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if arch.llama4 == nil {
		t.Fatal("no llama4 params on the resolved arch")
	}
	if arch.llama4.chunkSize != cfg.AttentionChunkSize {
		t.Errorf("chunkSize %d, config says %d — attention_chunk_size is read from JSON and "+
			"dropped on the way to the forward, which is M-05 exactly",
			arch.llama4.chunkSize, cfg.AttentionChunkSize)
	}
	// And it must actually bound a query past the chunk on a RoPE layer. Note the polarity of
	// no_rope_layers, which is the opposite of what its name suggests and which I got wrong
	// first time: the adapter reads `useRope[i] = NoRopeLayers[i] != 0`, so with {1,1,0,1}
	// layers 0/1/3 USE RoPE and layer 2 is the NoPE one.
	if got := arch.attnChunkStart(0, 8192+3); got != 8192 {
		t.Errorf("resolved arch: start %d at pos 8195 on RoPE layer 0, want 8192", got)
	}
	if got := arch.attnChunkStart(2, 8192+3); got != 0 {
		t.Errorf("resolved arch: NoPE layer 2 is chunked (start %d), want full causal", got)
	}
}
