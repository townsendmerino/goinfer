//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func hN(b Buffer, n int) []float32 {
	raw := b.U16s()[:n]
	out := make([]float32, n)
	for i, h := range raw {
		out[i] = f16ToF32(h)
	}
	return out
}
func l2(a []float32) float64 {
	var s float64
	for _, v := range a {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s)
}
func cosv(a, b []float32) float64 {
	var d, na, nb float64
	for i := range a {
		d += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return d / (math.Sqrt(na*nb) + 1e-12)
}

// Fable's hypothesis: the probe token is <bos> (id 2), Gemma's ATTENTION SINK. Sink V vectors
// are trained near-zero (sink K is a strong direction; sink V is a no-op). A cosine between two
// near-zero vectors is rounding noise — which would make "layer-1 V cos = -0.047 ⇒ x after
// layer 0 is orthogonal" a measurement artifact, not a bug. Every number in the debug report was
// a cosine; nobody measured a NORM. So measure norms — and probe a NON-sink token too.
func TestSink_NormsNotCosines(t *testing.T) {
	requireHeavyModel(t)
	path := os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint")
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer r.Close()
	mcpu, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("cpu: %v", err)
	}
	_, nL, _, nKV, hd, _, _ := mcpu.Dims()
	kvDim := nKV * hd

	for _, tok := range []int{2 /* <bos> — the sink */, 6037 /* an ordinary word token */} {
		cache := decoder.NewKVCache(nL, nKV, hd, 0, 64)
		if _, err := mcpu.ForwardForTest(tok, cache); err != nil {
			t.Fatalf("cpu fwd: %v", err)
		}
		r.ForwardEmb(m.EmbedResidentForTest(tok), 0)
		t.Logf("=== probe token %d %s ===", tok, map[bool]string{true: "(<bos> = attention sink)", false: "(ordinary token)"}[tok == 2])
		for _, l := range []int{0, 1, 2, 5} {
			if l >= nL {
				continue
			}
			cV, gV := cache.Vals(l)[:kvDim], hN(r.vc[l], kvDim)
			cK, gK := cache.Keys(l)[:kvDim], hN(r.kc[l], kvDim)
			t.Logf("  layer %d: |V|cpu=%.3e |V|gpu=%.3e  Vcos=%+.4f   |K|cpu=%.3e |K|gpu=%.3e  Kcos=%+.4f",
				l, l2(cV), l2(gV), cosv(cV, gV), l2(cK), l2(gK), cosv(cK, gK))
		}
	}
}
