package constrain_test

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestMaskCost_P20 measures what audit item P-20 only ESTIMATED.
//
// P-20 says `Masker.Process` is O(V) grammar walks per decode step and reasons: "Estimate
// 40–120 ns/token → 6–30 ms per step against ~2–5 ms per resident-GPU decode step —
// constrained decoding plausibly 3–10× slower per token on GPU", closing with the instruction
// this test carries out: "Measure one Process call at fsStr and at fsObjKeyOrClose for
// V=151,936 against the unconstrained step."
//
// It matters more than an optimisation note, which is why it is measured before anything is
// designed on top of it: constrained generation is the README's headline promise ("a Go struct
// the model cannot violate"), and whether it costs 1.2× or 10× decides whether that promise is
// usable on the fast backends or has to be documented as slow.
//
// Method: MaskAt is Process's hot loop without the commit, so driving a grammar to a chosen
// state by committing BYTES and then timing MaskAt isolates exactly the per-step masking cost
// at that state — no tokenizer round-trip, no decode, nothing else in the sample. Real vocab
// (V=151,936 Qwen tokens), min-of-N to trim scheduler noise.
//
//	GOINFER_HEAVY_TESTS=1 go test ./constrain/ -run TestMaskCost_P20 -v
func TestMaskCost_P20(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("needs a real tokenizer for a real vocab: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_MASKCOST_GGUF")
	if path == "" {
		path = filepath.Join(home, ".cache", "goinfer", "models", "Qwen",
			"Qwen2.5-Coder-0.5B-Instruct-GGUF", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no tokenizer source at %s (pull demo:0.5b, or set GOINFER_MASKCOST_GGUF)", path)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}

	const V = 151936 // Qwen2.5's padded model vocab — the size Process actually walks
	toks := constrain.TokenBytes(V, tk.TokenText)
	nonEmpty := 0
	for _, b := range toks {
		if len(b) > 0 {
			nonEmpty++
		}
	}
	sp := tk.Special()
	eos := []int{sp.EOS}

	type Person struct {
		Name string   `json:"name"`
		Age  int      `json:"age"`
		Tags []string `json:"tags"`
	}
	g, err := constrain.GrammarFromStruct(Person{})
	if err != nil {
		t.Fatalf("GrammarFromStruct: %v", err)
	}
	m := constrain.NewMasker(g, toks, eos)

	logits := make([]float32, V)
	timeAt := func(prefix string) time.Duration {
		best := time.Hour
		for range 7 {
			gs := m.GrammarClone()
			gs.Reset()
			if len(prefix) > 0 {
				gs.Commit([]byte(prefix))
			}
			for i := range logits {
				logits[i] = 0
			}
			t0 := time.Now()
			m.MaskAt(gs, logits)
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return best
	}

	t.Logf("vocab V=%d (%d ids with surface bytes), grammar = struct{Name string; Age int; Tags []string}", V, nonEmpty)
	t.Logf("%-22s %12s %14s", "grammar state", "per step", "ns/token")
	var worst time.Duration
	for _, c := range []struct{ name, prefix string }{
		{"fsObjKeyOrClose", `{`},
		{"fsStr (in a string)", `{"name":"Ada`},
		{"fsNum (in a number)", `{"name":"a","age":3`},
		{"complete document", `{"name":"a","age":3,"tags":[]}`},
	} {
		d := timeAt(c.prefix)
		if d > worst {
			worst = d
		}
		t.Logf("%-22s %9.3f ms %11.1f", c.name, float64(d.Microseconds())/1000, float64(d.Nanoseconds())/float64(V))
	}

	// The comparison P-20 asks for. Decode-step times are this box's measured resident-GPU
	// numbers for the 1.5B AFTER the G35/G36 kernel work (docs/QUEUE.md): the whole token is
	// ~6.2 ms at pos 64 and ~7.4 ms at pos 512, so the mask is compared against the cheaper
	// (harder) end. Quoting the pre-G36 figure would flatter the mask by ~3x.
	for _, step := range []struct {
		name string
		ms   float64
	}{{"resident GPU 1.5B, pos 64 (6.2 ms/token)", 6.2}, {"resident GPU 1.5B, pos 512 (7.4 ms/token)", 7.4}} {
		maskMs := float64(worst.Microseconds()) / 1000
		t.Logf("vs %-42s → constrained decode ≈ %.2fx unconstrained", step.name, (step.ms+maskMs)/step.ms)
	}
	fmt.Fprintf(os.Stderr, "[maskcost] worst-state mask = %.3f ms/step at V=%d\n", float64(worst.Microseconds())/1000, V)
}
