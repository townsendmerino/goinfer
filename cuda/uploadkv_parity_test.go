//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math/rand"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestUploadKV_matchesSequentialForward is UploadKV's FIRST real correctness test (P6b / gap-0,
// docs/multimodal.md). Before this, UploadKV had zero non-test call sites and every fake stub
// (decoder/*_test.go) ignored its arguments and returned nil — the "no precision/layout
// mismatch" claim the hybrid resident-decode design leans on was asserted by doc comments, never
// exercised.
//
// Two arms on the SAME resident handle (Reset between them, not two loads — two resident 4B
// loads risk OOM on an 8 GB card):
//
//	ARM A (ground truth): sequential rf.Forward(emb, pos) for every prompt position, exactly the
//	  ordinary resident prefill path — this IS the reference definition, not a second thing to
//	  doubt.
//	ARM B (hybrid, the thing under test): a CPU prefill (PrefillLogitsForTest, the same
//	  bidirectional-capable CPU path GenerateVL's prefillLogitsVL uses) followed by UploadKV per
//	  layer using KVCache.LayerKVForTest's ring/quant-aware (k, v, base) — the exact bridge
//	  decoder/generate_vl_resident.go's residentUploadPrefill will use.
//
// Both arms then decode a few tokens from the same continuation point and must agree.
//
// Two cases: base=0 (short prompt, no sliding-window ring wrap) and base>0 (a prompt long
// enough to wrap at least one of gemma3's local layers — sliding_window=1024 per its own
// config.json, well under a real image-block-sized prefill) — the second is the exact case the
// UploadKV signature amendment (adding `base`) exists for; KVCache.Keys/Vals alone would have
// silently read empty for those layers, and base=0 would have silently written the live window
// at the wrong absolute position.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestUploadKV_matchesSequentialForward -v -timeout 20m
func TestUploadKV_matchesSequentialForward(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GEMMA3_4B")
	if path == "" {
		path = home + "/models/gemma-3-4b-it"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no gemma-3-4b-it at %s: %v", path, err)
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}

	for _, tc := range []struct {
		name string
		n    int // prompt length
	}{
		{"base0_unwrapped", 64},
		{"baseN_ringWrapped", 1200}, // > sliding_window=1024
	} {
		t.Run(tc.name, func(t *testing.T) {
			mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int8"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer mc.Close()
			rf := mc.ResidentForwardForTest()
			if rf == nil {
				t.Skip("gemma3 not resident-eligible on this build")
			}
			_, nLayers, _, _, _, _, vocab := mc.Dims()

			rng := rand.New(rand.NewSource(7))
			ids := make([]int, tc.n)
			for i := range ids {
				ids[i] = rng.Intn(vocab - 1)
			}
			embs := make([][]float32, tc.n)
			for i, id := range ids {
				embs[i] = mc.EmbedResidentForTest(id)
			}

			const nDecode = 4

			// ARM A: ground truth.
			var lastPrefillLogits []float32
			for i, e := range embs {
				l, err := rf.Forward(e, i)
				if err != nil {
					t.Fatalf("ground-truth prefill Forward pos %d: %v", i, err)
				}
				lastPrefillLogits = l
			}
			wantDecode := make([][]float32, nDecode)
			tok := argmaxF(lastPrefillLogits)
			for i := range nDecode {
				l, err := rf.Forward(mc.EmbedResidentForTest(tok), tc.n+i)
				if err != nil {
					t.Fatalf("ground-truth decode step %d: %v", i, err)
				}
				wantDecode[i] = l
				tok = argmaxF(l)
			}

			rf.Reset()

			// ARM B: hybrid — CPU prefill, then UploadKV per layer.
			cache := mc.NewCache(tc.n + nDecode)
			cpuLogits, err := mc.PrefillLogitsForTest(context.Background(), ids, cache)
			if err != nil {
				t.Fatalf("CPU prefill: %v", err)
			}
			for l := range nLayers {
				k, v, base := cache.LayerKVForTest(l)
				if len(k) == 0 {
					t.Fatalf("layer %d: LayerKVForTest returned empty K — ring/quant read is broken", l)
				}
				if err := rf.UploadKV(l, base, k, v); err != nil {
					t.Fatalf("UploadKV layer %d (base=%d, len(k)=%d): %v", l, base, len(k), err)
				}
			}
			gotDecode := make([][]float32, nDecode)
			tok = argmaxF(cpuLogits)
			for i := range nDecode {
				l, err := rf.Forward(mc.EmbedResidentForTest(tok), tc.n+i)
				if err != nil {
					t.Fatalf("hybrid decode step %d: %v", i, err)
				}
				gotDecode[i] = l
				tok = argmaxF(l)
			}

			for i := range nDecode {
				cos := cosine(wantDecode[i], gotDecode[i])
				t.Logf("decode step %d: cosine(ground-truth, hybrid) = %.8f", i, cos)
				if cos < 0.999999 {
					t.Errorf("decode step %d: cosine = %.8f, want ~1.0 (UploadKV bridge diverges from sequential Forward)", i, cos)
				}
			}
		})
	}
}
