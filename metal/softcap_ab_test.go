//go:build darwin

package metal

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestZZ_metalSoftcapTokenShare loads a real dense Gemma (final-logit softcap, 262k vocab) and times
// the full-logits sampling path (ForwardEmb → forwardLogits → finalizeLogits) per token, so the
// isolated softcap A/B (BenchmarkSoftcap_gemmaVocab_*: serial ~3.4ms, parallel ~0.86ms) can be
// expressed as a share of the token. Opt-in timing diagnostic, not a gate.
func TestZZ_metalSoftcapTokenShare(t *testing.T) {
	if os.Getenv("GOINFER_METAL_SOFTCAP_AB") == "" {
		t.Skip("Metal softcap token-share (timing diagnostic); set GOINFER_METAL_SOFTCAP_AB=1")
	}
	path := os.ExpandEnv("$HOME/models/gemma-4-12b-it-qat-q4_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present: %v", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Skipf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		// The gemma-4-12b QAT q4_0 checkpoint doesn't build int4/int8 resident on this Mac
		// (int4Concat weight-kind quirk); the E-series are MoE (paged, timing-misleading). This
		// diagnostic wants a DENSE softcapped gemma that builds resident — run it on the box with
		// one. The isolated win stands regardless (BenchmarkSoftcap_gemmaVocab_*).
		t.Skipf("resident (needs a dense softcapped gemma-4 that builds resident here): %v", err)
	}
	t.Cleanup(func() { r.Close() })
	if r.finalSoftcap <= 0 {
		t.Skipf("model has no final-logit softcap (gemma-3 removed it; needs gemma-2/gemma-4): got %v", r.finalSoftcap)
	}

	emb := make([]float32, r.H)
	pos := 0
	for ; pos < 8; pos++ { // warm
		r.ForwardEmb(emb, pos)
	}
	const K = 64
	best := time.Duration(1 << 62)
	for range 3 {
		start := time.Now()
		for range K {
			r.ForwardEmb(emb, pos) // full-logits sampling path (includes the parallel softcap)
			pos++
		}
		if d := time.Since(start) / K; d < best {
			best = d
		}
	}
	perTok := best
	// The isolated softcap A/B (this vocab): serial ~3.40ms, parallel ~0.86ms → saved ~2.54ms.
	const savedMs = 2.54
	fmt.Printf("\n=== Metal Gemma-3-4b full-logits (sampling) token time: %.2f ms (vocab=%d, softcap=%.0f) ===\n",
		float64(perTok.Microseconds())/1000, r.V, r.finalSoftcap)
	fmt.Printf("softcap serial->parallel saves ~%.2f ms/token → ~%.1f%% of a serial-softcap sampling token\n",
		savedMs, 100*savedMs/(float64(perTok.Microseconds())/1000+savedMs))
}
