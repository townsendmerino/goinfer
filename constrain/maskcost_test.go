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

// TestMaskCost_P20 measures the per-step cost of Masker.Process that audit item P-20 only estimated:
// O(V) grammar walks per decode step, against a resident-GPU decode step. Whether that is 1.2x or 10x
// decides whether constrained generation (the README's headline promise) is usable on the fast backends
// or has to be documented as slow.
//
// Method: MaskAt is Process's hot loop without the commit, so driving a grammar to a chosen state by
// committing BYTES and then timing MaskAt isolates the per-step masking cost at that state: no tokenizer
// round-trip, no decode. Real vocab (V=151,936 Qwen tokens), min-of-N to trim scheduler noise.
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

	// Per-state costs are not what a caller pays: a real document spends most of its steps
	// inside strings and only a handful at a key boundary or in a number. So walk a
	// representative document token by token and time the mask AT EVERY STEP — the weighted
	// average is the number that actually shows up in a generation, and quoting the
	// worst state instead would overstate it several-fold now that strings are fast.
	doc := `{"name":"Ada Lovelace","tags":["mathematician","programmer"],"inner":{"note":"first published algorithm"}}`
	docIDs, _ := tk.Encode(doc, false)
	gd := m.GrammarClone()
	gd.Reset()
	var total time.Duration
	steps := 0
	for _, id := range docIDs {
		b := tk.TokenText(id)
		if len(b) == 0 || !gd.TryBytes(b) {
			break // the tokenizer's segmentation left the grammar; stop rather than mis-report
		}
		for i := range logits {
			logits[i] = 0
		}
		t0 := time.Now()
		m.MaskAt(gd, logits)
		total += time.Since(t0)
		gd.Commit(b)
		steps++
	}
	if steps > 0 {
		per := float64(total.Microseconds()) / 1000 / float64(steps)
		t.Logf("")
		t.Logf("REALISTIC: %d-token document, mask averaged over every step: %.3f ms/step", steps, per)
		for _, step := range []struct {
			name string
			ms   float64
		}{{"resident GPU 1.5B, pos 64 (6.2 ms/token)", 6.2}, {"resident GPU 1.5B, pos 512 (7.4 ms/token)", 7.4}} {
			t.Logf("  vs %-42s → constrained decode ≈ %.2fx unconstrained", step.name, (step.ms+per)/step.ms)
		}
	}

	// The comparison P-20 asks for, against the cheaper (harder) end of this box's resident-GPU decode
	// step for the 1.5B after the G35/G36 kernel work (docs/QUEUE.md): a pre-G36 figure would flatter the mask.
	t.Logf("")
	t.Logf("WORST STATE (an upper bound, not a typical step):")
	for _, step := range []struct {
		name string
		ms   float64
	}{{"resident GPU 1.5B, pos 64 (6.2 ms/token)", 6.2}, {"resident GPU 1.5B, pos 512 (7.4 ms/token)", 7.4}} {
		maskMs := float64(worst.Microseconds()) / 1000
		t.Logf("  vs %-42s → constrained decode ≤ %.2fx unconstrained", step.name, (step.ms+maskMs)/step.ms)
	}
	fmt.Fprintf(os.Stderr, "[maskcost] worst-state mask = %.3f ms/step at V=%d\n", float64(worst.Microseconds())/1000, V)
}
