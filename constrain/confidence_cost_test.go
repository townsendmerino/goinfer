package constrain_test

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestConfidenceCost_C1 prices CaptureConfidence against plain masking, per decode step, at three grammar states on
// a real vocabulary (V = 151,936, Qwen): an object key, an enum value, and inside a free string. C1 reads only
// outside free strings, where it keeps the legal tokens' entries too; the every-position C0 readout is in
// docs/measurements/confidence-c0-2026-09-27.md. Min of N, interleaved on/off.
//
//	GOINFER_HEAVY_TESTS=1 go test ./constrain/ -run TestConfidenceCost_C1 -v
func TestConfidenceCost_C1(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("needs a real tokenizer for a real vocab: set GOINFER_HEAVY_TESTS=1")
	}
	path := os.Getenv("GOINFER_MASKCOST_GGUF")
	if path == "" {
		path = filepath.Join("..", "testdata", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no tokenizer source at %s (set GOINFER_MASKCOST_GGUF): %v", path, err)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	const V = 151936
	toks := constrain.TokenBytes(V, tk.TokenText)
	eos := []int{tk.Special().EOS}
	schema := `{"type":"object","properties":{"category":{"enum":["billing","technical","account","shipping","other"]},` +
		`"note":{"type":"string"}},"required":["category","note"],"additionalProperties":false}`
	// Each state is reached by generating these ids through Process (the masker commits them on its next call).
	enc := func(s string) []int { ids, _ := tk.Encode(s, false); return ids }
	states := []struct {
		name   string
		prefix []int
	}{
		{"object key (after '{')", enc(`{`)},
		{"enum value (after '\"category\": \"')", enc(`{"category": "`)},
		{"free string (inside \"note\")", enc(`{"category": "billing", "note": "The customer`)},
	}
	logits := make([]float32, V)
	for i := range logits {
		logits[i] = float32(i%97) * 0.01
	}
	timeOne := func(capture bool, prefix []int) time.Duration {
		g, err := constrain.JSONSchema([]byte(schema))
		if err != nil {
			t.Fatal(err)
		}
		m := constrain.NewMasker(g, toks, eos)
		if capture {
			m.CaptureConfidence(constrain.ConfidenceOptions{})
		}
		buf := slices.Clone(logits)
		for i := range prefix { // walk the grammar to the state, one Process per emitted id
			copy(buf, logits)
			m.Process(prefix[:i], buf)
		}
		copy(buf, logits)
		t0 := time.Now()
		m.Process(prefix, buf) // the step being priced
		return time.Since(t0)
	}
	const N = 15
	for _, st := range states {
		best := map[bool]time.Duration{false: time.Hour, true: time.Hour}
		for i := range N {
			for _, on := range []bool{i%2 == 0, i%2 != 0} {
				if d := timeOne(on, st.prefix); d < best[on] {
					best[on] = d
				}
			}
		}
		extra := best[true] - best[false]
		t.Logf("%-40s mask %8.3f ms, mask+capture %8.3f ms, capture adds %7.3f ms (%.1f%% of the mask)",
			st.name, ms(best[false]), ms(best[true]), ms(extra), 100*float64(extra)/float64(best[false]))
	}
	fmt.Fprintln(os.Stderr, "priced: one Process call per state, min of", N)
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1e3 }
