//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestUploadKV_matchesSequentialForward is UploadKV's FIRST real correctness test on the WebGPU
// backend (P6b / gap-0, docs/multimodal.md) — the CUDA twin of this test lives in
// cuda/uploadkv_parity_test.go; see that file's doc comment for the full rationale. Before this,
// UploadKV had zero non-test call sites and every fake stub ignored its arguments and returned
// nil.
//
// Two arms on the SAME resident handle (Reset between them):
//
//	ARM A (ground truth): sequential rf.Forward(emb, pos) for every prompt position.
//	ARM B (hybrid, under test): CPU prefill (PrefillLogitsForTest) then UploadKV per layer using
//	  KVCache.LayerKVForTest's ring/quant-aware (k, v, base).
//
// Two cases: base=0 (no sliding-window ring wrap) and base>0 (prompt long enough to wrap the
// ring). Model: tinymistral-248m (Mistral, ALL layers sliding-window, sliding_window=32 per its
// own config.json) rather than Gemma 3 — confirmed by running this suite for real: gemma3
// declines WebGPU residency on this build ("arch needs unimplemented feature(s) [embed-scale
// gated-gelu sandwich-norm]", a pre-existing WebGPU feature gap, unrelated to this change), so it
// cannot exercise this test at all on WebGPU. tinymistral's tiny window (32, vs gemma3's 1024)
// also makes the ring-wrap case cheap to trigger — no need for a >1000-token prefill. The WebGPU
// UploadKV implementation additionally has real per-precision byte-offset arithmetic (f32/f16/
// int8, and int8's separate scale buffers) that the base=0 case alone cannot exercise, since an
// all-zero offset is degenerate for every precision.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'gpu goinfer_testhooks' ./gpu/ -run TestUploadKV_matchesSequentialForward -v -timeout 20m
func TestUploadKV_matchesSequentialForward(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_TINYMISTRAL_CKPT")
	if path == "" {
		path = home + "/models/tinymistral-248m"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no tinymistral-248m at %s: %v", path, err)
	}

	for _, tc := range []struct {
		name string
		n    int
	}{
		{"base0_unwrapped", 16},
		{"baseN_ringWrapped", 64}, // > sliding_window=32
	} {
		t.Run(tc.name, func(t *testing.T) {
			mc, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer mc.Close()
			rf := mc.ResidentForwardForTest()
			if rf == nil {
				t.Skip("tinymistral-248m not resident-eligible on this build")
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
				cos, maxAbs := cosSim(wantDecode[i], gotDecode[i])
				t.Logf("decode step %d: cosine(ground-truth, hybrid) = %.8f maxAbs=%.4g", i, cos, maxAbs)
				if cos < 0.999999 {
					t.Errorf("decode step %d: cosine = %.8f, want ~1.0 (UploadKV bridge diverges from sequential Forward)", i, cos)
				}
			}
		})
	}
}
