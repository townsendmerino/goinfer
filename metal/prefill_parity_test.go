//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillParity — the P4 gate. Ingest a prompt via PrefillLast (f16 MMA, one command buffer)
// and via the sequential Forward loop (the current correct path), and compare the LAST token's
// logits. Both populate the same positional KV cache and approximate the same model; the prefill
// path is f16 activations vs the decode path's int8 — so expect a high-but-not-exact cosine and
// a matching argmax. A prefill bug would show garbage (cosine ~0 / wrong argmax).
func TestPrefillParity(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_METAL_MODEL") // override to validate other families
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present: %v", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := BuildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}

	base := []int{785, 12095, 8948, 264, 6236, 1140, 13, 358, 3003, 264, 6236, 1140, 785, 12095}
	prompt := make([]int, 0, 140) // ~140-token prompt (repeat) so TTFT reflects a realistic length
	for len(prompt) < 140 {
		prompt = append(prompt, base...)
	}
	M := len(prompt)

	// sequential path (correct reference): Forward each token; keep the last logits.
	var seq []float32
	for i, id := range prompt {
		seq = r.Forward(id, i)
	}
	seqArg := argmaxF(seq)

	// prefill path: embeddings for the prompt, one PrefillLast call.
	embs := make([][]float32, M)
	for i, id := range prompt {
		e := make([]float32, r.H)
		r.embed.Row(id, e)
		embs[i] = e
	}
	pre := r.PrefillLast(embs, 0)
	preArg := argmaxF(pre)

	cos := cosF(seq, pre)
	t.Logf("prefill vs sequential last-token: argmax pre=%d seq=%d, cosine=%.5f", preArg, seqArg, cos)
	// NaN guard FIRST: this is the path that shipped NaN, and `NaN < 0.95` is false — the cosine
	// floor below cannot catch a degenerate prefill (parity-coverage-policy.md § Falsifiable).
	// TestPrefillNoNaN is the primary NaN gate; this keeps the parity gate itself non-vacuous.
	if math.IsNaN(float64(cos)) || math.IsInf(float64(cos), 0) {
		t.Fatalf("prefill parity FAIL: cosine is %v — degenerate (NaN/Inf) logits", cos)
	}
	if preArg != seqArg {
		t.Fatalf("prefill parity FAIL: last-token argmax %d != sequential %d (cosine %.4f)", preArg, seqArg, cos)
	}
	if cos < 0.95 {
		t.Fatalf("prefill parity FAIL: cosine %.4f too low (bug?)", cos)
	}
	t.Logf("prefill last-token argmax matches sequential ✓ (cosine %.4f)", cos)

	// TTFT: prefill vs the sequential loop for this M-token prompt.
	best := func(f func()) time.Duration {
		f()
		b := time.Hour
		for range 5 {
			t0 := time.Now()
			f()
			if dt := time.Since(t0); dt < b {
				b = dt
			}
		}
		return b
	}
	tSeq := best(func() {
		for i, id := range prompt {
			r.Forward(id, i)
		}
	})
	tPre := best(func() { r.PrefillLast(embs, 0) })
	t.Logf("prefill %d tokens: sequential %.2f ms vs PrefillLast %.2f ms => %.1fx faster TTFT",
		M, tSeq.Seconds()*1000, tPre.Seconds()*1000, float64(tSeq)/float64(tPre))
}
