//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func resolveUploadKVModel(t *testing.T) string {
	t.Helper()
	candidates := []string{
		os.Getenv("GEMMA3_4B"),
		os.Getenv("GOINFER_QWEN15_CKPT"),
	}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		candidates = append(candidates, modelPath("gemma-3-4b-it"), modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"))
	}
	candidates = append(candidates, "../testdata/llama-tiny")
	for _, p := range candidates {
		if p == "" {
			continue
		}
		if f, err := os.Open(p); err == nil {
			f.Close()
			return p
		}
	}
	t.Skip("no accessible model fixture for UploadKV test")
	return ""
}

// TestUploadKV_matchesSequentialForward is UploadKV's FIRST real correctness test on the Metal backend.
// The CUDA twin lives in cuda/uploadkv_parity_test.go; the WebGPU twin lives in gpu/uploadkv_parity_test.go.
//
// Two arms on the SAME resident handle (Reset between them):
//
//	ARM A (ground truth): sequential rf.Forward(emb, pos) for every prompt position.
//	ARM B (hybrid, under test): CPU prefill (PrefillLogitsForTest) then UploadKV per layer using
//	  KVCache.LayerKVForTest's ring/quant-aware (k, v, base).
//
// Both arms then decode continuation tokens and must agree.
func TestUploadKV_matchesSequentialForward(t *testing.T) {
	path := resolveUploadKVModel(t)

	for _, tc := range []struct {
		name string
		n    int
	}{
		{"prompt_16", 16},
		{"prompt_32", 32},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mc, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int8int8"})
			if err != nil {
				t.Fatalf("load metal: %v", err)
			}
			defer mc.Close()
			rf := mc.ResidentForwardForTest()
			if rf == nil {
				t.Skip("model not resident-eligible on this build")
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

			// ARM A: ground truth sequential resident forward.
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

			// ARM B: hybrid — CPU prefill, then UploadKV per layer into resident cache.
			cache := mc.NewCache(tc.n + nDecode)
			cpuLogits, err := mc.PrefillLogitsForTest(context.Background(), ids, cache)
			if err != nil {
				t.Fatalf("CPU prefill: %v", err)
			}
			for l := range nLayers {
				k, v, base := cache.LayerKVForTest(l)
				if len(k) == 0 {
					t.Fatalf("layer %d: LayerKVForTest returned empty K", l)
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
				if cos < 0.9999 {
					t.Errorf("decode step %d: cosine = %.8f < 0.9999 (UploadKV bridge diverges)", i, cos)
				}
			}
		})
	}
}

func TestUploadKV_offsetBase(t *testing.T) {
	path := resolveUploadKVModel(t)
	mc, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Skip("model not resident-eligible on this build")
	}
	_, nLayers, _, _, _, _, vocab := mc.Dims()

	const prefixLen = 8
	const promptLen = 16
	const nDecode = 4

	rng := rand.New(rand.NewSource(19))
	prefixIDs := make([]int, prefixLen)
	for i := range prefixIDs {
		prefixIDs[i] = rng.Intn(vocab - 1)
	}
	promptIDs := make([]int, promptLen)
	for i := range promptIDs {
		promptIDs[i] = rng.Intn(vocab - 1)
	}

	// ARM A: Ground truth. Forward prefix, then prompt, then decode.
	for i, id := range prefixIDs {
		if _, err := rf.Forward(mc.EmbedResidentForTest(id), i); err != nil {
			t.Fatalf("arm A prefix step %d: %v", i, err)
		}
	}
	var lastPrefill []float32
	for i, id := range promptIDs {
		l, err := rf.Forward(mc.EmbedResidentForTest(id), prefixLen+i)
		if err != nil {
			t.Fatalf("arm A prompt step %d: %v", i, err)
		}
		lastPrefill = l
	}
	wantDecode := make([][]float32, nDecode)
	tok := argmaxF(lastPrefill)
	for i := range nDecode {
		l, err := rf.Forward(mc.EmbedResidentForTest(tok), prefixLen+promptLen+i)
		if err != nil {
			t.Fatalf("arm A decode step %d: %v", i, err)
		}
		wantDecode[i] = l
		tok = argmaxF(l)
	}

	rf.Reset()

	// ARM B: Forward prefix on resident GPU, prefill prompt on CPU with base=prefixLen,
	// UploadKV at base=prefixLen, then decode.
	for i, id := range prefixIDs {
		if _, err := rf.Forward(mc.EmbedResidentForTest(id), i); err != nil {
			t.Fatalf("arm B prefix step %d: %v", i, err)
		}
	}
	cache := mc.NewCache(prefixLen + promptLen + nDecode)
	if _, err := mc.PrefillLogitsForTest(context.Background(), prefixIDs, cache); err != nil {
		t.Fatalf("CPU prefix prefill: %v", err)
	}
	cpuLogits, err := mc.PrefillLogitsForTest(context.Background(), promptIDs, cache)
	if err != nil {
		t.Fatalf("CPU prefill: %v", err)
	}
	for l := range nLayers {
		k, v, base := cache.LayerKVForTest(l)
		if err := rf.UploadKV(l, base, k, v); err != nil {
			t.Fatalf("UploadKV layer %d (base=%d, len(k)=%d): %v", l, base, len(k), err)
		}
	}
	gotDecode := make([][]float32, nDecode)
	tok = argmaxF(cpuLogits)
	for i := range nDecode {
		l, err := rf.Forward(mc.EmbedResidentForTest(tok), prefixLen+promptLen+i)
		if err != nil {
			t.Fatalf("hybrid decode step %d: %v", i, err)
		}
		gotDecode[i] = l
		tok = argmaxF(l)
	}

	for i := range nDecode {
		cos, maxAbs := cosSim(wantDecode[i], gotDecode[i])
		t.Logf("offsetBase decode step %d: cosine(ground-truth, hybrid) = %.8f maxAbs=%.4g", i, cos, maxAbs)
		if cos < 0.9999 {
			t.Errorf("decode step %d: cosine = %.8f < 0.9999 (UploadKV offset base diverges)", i, cos)
		}
	}
}

// TestUploadKV_matchesSequentialForward_KVI8 validates UploadKV when KVPrecision is "i8".
func TestUploadKV_matchesSequentialForward_KVI8(t *testing.T) {
	path := resolveUploadKVModel(t)
	const n = 16
	mc, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int8int8", KVPrecision: "i8"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mc.Close()

	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Skip("model did not build resident")
	}

	_, nLayers, _, _, _, _, vocab := mc.Dims()
	if vocab <= 0 {
		t.Fatalf("vocab <= 0: %d", vocab)
	}

	rng := rand.New(rand.NewSource(1337))
	promptIDs := make([]int, n)
	for i := range promptIDs {
		promptIDs[i] = rng.Intn(vocab)
	}

	const nDecode = 4
	// ARM A: sequential forward
	wantDecode := make([][]float32, nDecode)
	var tok int
	for pos, id := range promptIDs {
		l, err := rf.Forward(mc.EmbedResidentForTest(id), pos)
		if err != nil {
			t.Fatalf("arm A prompt pos %d: %v", pos, err)
		}
		if pos == n-1 {
			tok = argmaxF(l)
		}
	}
	for i := range nDecode {
		l, err := rf.Forward(mc.EmbedResidentForTest(tok), n+i)
		if err != nil {
			t.Fatalf("arm A decode step %d: %v", i, err)
		}
		wantDecode[i] = l
		tok = argmaxF(l)
	}

	// Reset resident state
	rf.Reset()

	// ARM B: CPU prefill + UploadKV
	cache := mc.NewCache(n + nDecode)
	cpuLogits, err := mc.PrefillLogitsForTest(context.Background(), promptIDs, cache)
	if err != nil {
		t.Fatalf("CPU prefill: %v", err)
	}
	for l := range nLayers {
		k, v, base := cache.LayerKVForTest(l)
		if err := rf.UploadKV(l, base, k, v); err != nil {
			t.Fatalf("UploadKV layer %d: %v", l, err)
		}
	}

	gotDecode := make([][]float32, nDecode)
	tok = argmaxF(cpuLogits)
	for i := range nDecode {
		l, err := rf.Forward(mc.EmbedResidentForTest(tok), n+i)
		if err != nil {
			t.Fatalf("arm B decode step %d: %v", i, err)
		}
		gotDecode[i] = l
		tok = argmaxF(l)
	}

	for i := range nDecode {
		cos, maxAbs := cosSim(wantDecode[i], gotDecode[i])
		t.Logf("INT8 KV UploadKV decode step %d: cosine = %.7f maxAbs=%.4g", i, cos, maxAbs)
		if cos < 0.999 {
			t.Errorf("decode step %d: cosine = %.7f < 0.999", i, cos)
		}
	}
}
