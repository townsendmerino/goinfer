//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestSpecResidentPrefillRegression attributes the resident speculative slowdown to
// PROMPT LENGTH, which is the signature of per-token prefill.
//
// THE DEFECT. decoder/model.go's generateInto uses the optional Prefiller seam on a
// resident model — `pf.PrefillLast(embs, 0)`, one batched on-device pass — whenever
// len(prompt) >= 8. decoder/spec_ngram.go's genNgramInto does NOT: its resident
// branch loops `target.resident.Forward(embedResident(id), i)` once per prompt
// token. cudaResident implements PrefillLast (cuda/prefill.go), so the batched path
// exists and is simply not taken on the speculative path.
//
// WHY IT WAS NEVER SEEN. gpu/spec_ngram_resident_test.go's corpus prompts are 36-74
// tokens, where the penalty is a fraction of a second and hides inside generation.
// It scales with prompt length, and no harness had a long prompt until the realistic
// corpus (656-1039 tokens) in docs/spec/02.
//
// THE CONTROL. If the slowdown is prefill-driven it must be roughly CONSTANT in
// absolute terms and vanish as a ratio on a short prompt. If instead speculation
// were inherently slow here, the ratio would persist at both lengths. Same model,
// same session, interleaved.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_SPEC_PREFILL_REGRESSION=1 \
//	  go test -tags "cuda goinfer_testhooks" ./ -run TestSpecResidentPrefillRegression -v
func TestSpecResidentPrefillRegression(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("GOINFER_SPEC_PREFILL_REGRESSION") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 GOINFER_SPEC_PREFILL_REGRESSION=1")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	path := modelPath("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Skipf("not resident: %s", m.ResidentDecline())
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	src, err := os.ReadFile("../decoder/spec_adaptive.go")
	if err != nil {
		t.Skipf("corpus: %v", err)
	}
	full := string(src)
	ctx := context.Background()
	greedy := decoder.SamplingParams{Temperature: 0}
	const maxTok = 64

	t.Logf("%8s %7s | %9s %9s | %8s %10s", "prompt", "tokens", "off", "spec", "ratio", "abs delta")
	var toks, deltas []float64
	for _, nchars := range []int{150, 400, 1000, 2000, 3000} {
		p := full
		if len(p) > nchars {
			p = p[:nchars]
		}
		prompt, err := tk.Encode(p, true)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		var offMed, specMed []float64
		for r := 0; r < 3; r++ {
			t0 := time.Now()
			ch, g := m.Generate(ctx, prompt, maxTok, greedy)
			ref := collectToks(ch)
			offMed = append(offMed, float64(time.Since(t0).Microseconds())/1000)
			if g.Err() != nil {
				t.Fatalf("off: %v", g.Err())
			}

			t0 = time.Now()
			sch, sg, err := m.GenerateNgramSpeculativeAdaptive(ctx, prompt, maxTok, &decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: 8}, greedy)
			if err != nil {
				t.Fatalf("spec: %v", err)
			}
			got := collectToks(sch)
			specMed = append(specMed, float64(time.Since(t0).Microseconds())/1000)
			if sg.Err() != nil {
				t.Fatalf("spec: %v", sg.Err())
			}
			if !slices.Equal(got, ref) {
				t.Fatalf("LOSSLESS VIOLATION at %d chars", nchars)
			}
		}
		o, s := medianF(offMed), medianF(specMed)
		t.Logf("%8d %7d | %8.0fms %8.0fms | %7.2fx %+9.0fms", nchars, len(prompt), o, s, o/s, s-o)
		toks = append(toks, float64(len(prompt)))
		deltas = append(deltas, s-o)
	}

	// THE GATE. This is the assertion the original GPU speculative harness lacked: it
	// measured the right quantity and only LOGGED it ("Parity is hard-gated; speedup is
	// logged per workload"), so a 3-4.5x slowdown printed and failed nothing.
	//
	// It gates the DEFECT SIGNATURE, not "does speculation pay". Per-token prefill makes
	// the off-vs-spec gap grow LINEARLY in prompt length; whether speculation is a net win
	// at a given acceptance rate is a separate, noisy question that would make this flap.
	// Measured on this box: 2.66 ms/prompt-token with the bug, 0.12 ms/prompt-token after
	// wiring genNgramInto to residentPrefillSeed. The bar sits between them, nearer the
	// fixed value, so a regression has to be a real return of per-token prefill to trip it.
	slope := leastSquaresSlope(toks, deltas)
	t.Logf("gap-vs-prompt-length slope = %.3f ms/prompt-token (bar 0.50; 2.66 was the defect, 0.12 is fixed)", slope)
	if slope > 0.50 {
		t.Fatalf("REGRESSION: the speculative resident prefill cost is growing at %.2f ms per prompt token. "+
			"That is the signature of per-token prefill — check that genNgramInto still routes through "+
			"Model.residentPrefillSeed (the shared batched-prefill seam) rather than looping resident.Forward.", slope)
	}
	_ = strings.TrimSpace
}

func leastSquaresSlope(xs, ys []float64) float64 {
	n := float64(len(xs))
	if n < 2 {
		return 0
	}
	var sx, sy, sxx, sxy float64
	for i := range xs {
		sx, sy, sxx, sxy = sx+xs[i], sy+ys[i], sxx+xs[i]*xs[i], sxy+xs[i]*ys[i]
	}
	return (n*sxy - sx*sy) / (n*sxx - sx*sx)
}
