package decoder

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// Options.EmbedInt4 must reach every safetensors loader: a loader that quantizes its embedding and LM head with
// quant.embedding() never sees the flag, so --embed-int4 (the default everywhere but Metal) does nothing on it and the
// head stays int8 whatever the flag says. Option D's registered HeadTable() assertion is what catches it
// (docs/tasks/task-multimodal-support-2026-10.md).

// A tiny fixture per family that has one. spark2_5, gpt_oss and internlm2 have none on disk; the source guard below covers their loaders structurally.
var embedInt4Fixtures = []struct{ name, dir string }{
	{"gpt2", "gpt2"}, {"granitemoehybrid", "granite-tiny"}, {"granite (dense)", "granite-dense-tiny"}, {"nemotron_h", "nemotron-tiny"},
	{"nemotron_h (nano)", "nemotron3nano-tiny"}, {"phi3", "phi3-tiny"}, {"glm_ocr", "glm-ocr-tiny"}, {"llama4_text", "llama4-tiny"},
}

func TestEmbedInt4_reachesEveryFamilyLoader(t *testing.T) {
	ids := []int{1, 5, 9, 3, 7}
	for _, fx := range embedInt4Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			dir := filepath.Join("..", "testdata", fx.dir)
			if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
				t.Skipf("no fixture at %s", dir)
			}
			logits := map[bool][]float32{}
			for _, on := range []bool{false, true} {
				m, err := Load(dir, Options{Backend: "cpu", Quant: "int4", EmbedInt4: on})
				if err != nil {
					t.Fatalf("EmbedInt4=%v: %v", on, err)
				}
				want := "int8"
				if on {
					want = "int4"
				}
				if got := m.HeadTable(); got != want {
					t.Errorf("EmbedInt4=%v: HeadTable() = %q, want %q: the flag did not reach this family's loader", on, got, want)
				}
				l, err := m.prefillLogits(context.Background(), ids, m.NewCache(len(ids)+1))
				if err != nil {
					t.Fatalf("EmbedInt4=%v: %v", on, err)
				}
				for _, v := range l {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatalf("EmbedInt4=%v: non-finite logit", on)
					}
				}
				logits[on] = append([]float32(nil), l...)
				m.Close()
			}
			// The two arms differ ONLY in the head table, so their logits are close but not equal; equal would mean the tables were the same kind.
			var dot, a2, b2, diff float64
			for i := range logits[false] {
				x, y := float64(logits[false][i]), float64(logits[true][i])
				dot += x * y
				a2 += x * x
				b2 += y * y
				diff += math.Abs(x - y)
			}
			// The bar is 0.5 for a stated reason, not a guess: a mis-laid int4 table reads near ZERO, while working tables read
			// 0.86 to 0.999 on these tiny fixtures (a ragged last group reconstructs at the same ~9% relative RMS as 64 columns).
			if cos := dot / math.Sqrt(a2*b2); cos < 0.5 {
				t.Errorf("logit cosine %.4f between the int8 and the int4 head is below 0.5: the int4 table is being misread", cos)
			}
			if diff == 0 {
				t.Errorf("the int8-head and int4-head logits are identical: the tables are the same kind")
			}
		})
	}
}

// The structural twin: no weight loader may quantize an embedding or a head with the pinned precision directly. A loader that does ignores Options.EmbedInt4, which
// is how the nine above went inert; spark2_5, gpt_oss and internlm2 have no fixture, so this is what covers them. It proves "no loader pins the head by itself",
// not "every head is int4": the behavioural test above is the one that reads HeadTable().
func TestEmbedInt4_noLoaderPinsTheHeadItself(t *testing.T) {
	pinned := regexp.MustCompile(`quantizeWM\([^,()]+(?:\([^)]*\))?, [\w.]*\.embedding\(\)\)`)
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	scanned := 0
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		scanned++
		for i, line := range strings.Split(string(b), "\n") {
			if pinned.MatchString(line) {
				t.Errorf("%s:%d quantizes with the pinned embedding precision and never sees Options.EmbedInt4: use quantizeEmbedWM(w, quant.embeddingWith(embedInt4), needCanonical): %s", f, i+1, strings.TrimSpace(line))
			}
		}
	}
	if scanned < 50 {
		t.Fatalf("scanned only %d source files: the guard is not looking at the package", scanned)
	}
}
