package decoder

import (
	"compress/gzip"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/sampling"
)

// inkling2TinyDir is a random-initialized transformers Inkling2LoopForCausalLM at toy sizes (scripts/inkling2/pin_tiny.py),
// converted to GGUF by llama.cpp's own convert_hf_to_gguf (scripts/inkling2/convert_tiny.py, F32): 6 physical layers, the
// core [2, 4) run three passes (10 visits), blocks of 2, a sliding window of 4 and a full-layer bias extent of 8 under a
// 12-token prompt, so the window, the extent, the log scaling (floor 4) and every conv tap are exercised.
const inkling2TinyDir = "testdata/inkling2-loop-tiny"

type inkling2TinyGolden struct {
	InputIDs []int         `json:"input_ids"`
	Visits   [][][]float32 `json:"visits"` // [visit][position][hidden]: the decoder layer's output
	Logits   [][]float32   `json:"logits"` // [position][vocab]
}

func cosine32(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / math.Sqrt(aa*bb)
}

// TestInkling2_noResidentBackend holds the family CPU-only: no resident implements opInkling2, and a backend that admits by
// feature set must not read it as a plain attention and MoE model.
func TestInkling2_noResidentBackend(t *testing.T) {
	m, err := Load(inkling2TinyDir+"/inkling2-loop-tiny.gguf", Options{Quant: "f32"})
	if err != nil {
		t.Fatal(err)
	}
	for _, be := range residentBackends {
		if ResidentEligible(m.w.Arch, be) {
			t.Errorf("%s admits inkling2_loop as resident", be)
		}
	}
}

// TestInkling2Tiny_matchesTransformers runs the tiny checkpoint token by token through the CPU walker and holds every visit's
// output and every position's logits to transformers' full-sequence forward of the same weights.
func TestInkling2Tiny_matchesTransformers(t *testing.T) {
	f, err := os.Open(inkling2TinyDir + "/golden.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var g inkling2TinyGolden
	if err := json.NewDecoder(gz).Decode(&g); err != nil {
		t.Fatal(err)
	}
	m, err := Load(inkling2TinyDir+"/inkling2-loop-tiny.gguf", Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	a := m.w.Arch
	if a.Name != "inkling2_loop" || a.NumLayers != len(g.Visits) {
		t.Fatalf("resolved %q with %d visits, want inkling2_loop with %d", a.Name, a.NumLayers, len(g.Visits))
	}
	visits := make([]int, a.NumLayers)
	for i := range visits {
		visits[i] = i
	}
	cache := m.NewCache(len(g.InputIDs))
	worstHidden, worstLogit := 1.0, 1.0
	for p, id := range g.InputIDs {
		logits, hidden, err := m.forwardCapture(id, cache, visits)
		if err != nil {
			t.Fatalf("position %d: %v", p, err)
		}
		for v := range visits {
			c := cosine32(hidden[v], g.Visits[v][p])
			worstHidden = min(worstHidden, c)
			if c < 0.99999 {
				t.Errorf("position %d visit %d: cosine %.7f to transformers", p, v, c)
			}
		}
		c := cosine32(logits, g.Logits[p])
		worstLogit = min(worstLogit, c)
		if c < 0.99999 {
			t.Errorf("position %d: logits cosine %.7f", p, c)
		}
		if got, want := sampling.Argmax(logits), sampling.Argmax(g.Logits[p]); got != want {
			t.Errorf("position %d: argmax %d, want %d", p, got, want)
		}
	}
	t.Logf("worst cosine: visits %.8f, logits %.8f", worstHidden, worstLogit)
}

// TestInkling2_kvPricedAsStored holds the KV plan to the cache: past the window, a local visit's cache still holds every
// position (Inkling 2 has no rings), so the plan prices every position on every visit.
func TestInkling2_kvPricedAsStored(t *testing.T) {
	m, err := Load(inkling2TinyDir+"/inkling2-loop-tiny.gguf", Options{Quant: "f32"})
	if err != nil {
		t.Fatal(err)
	}
	a := m.w.Arch
	const n = 10 // past the tiny window of 4
	c := m.NewCache(n)
	for p := range n {
		if _, err := m.forward(p+1, c); err != nil {
			t.Fatal(err)
		}
	}
	for l := range a.NumLayers {
		if got, want := c.storedRows(l, a.KVDimAt(l)), a.KVPositionsAt(l, n); got != want {
			t.Errorf("visit %d (global %v): the cache stores %d rows, the plan prices %d", l, a.IsGlobalLayer(l), got, want)
		}
	}
}
