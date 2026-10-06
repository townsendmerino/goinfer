package embeddinggemma2

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

const tinyDir = "../testdata/embeddinggemma2-tiny"

type tinyGolden struct {
	Transformers string `json:"transformers"`
	Cases        []struct {
		IDs        []int         `json:"ids"`
		Layers     [][][]float64 `json:"layers"`
		Last       [][]float64   `json:"last_hidden_state"`
		Pooled     []float64     `json:"pooled"`
		Normalized []float64     `json:"normalized"`
		F64        []float64     `json:"normalized_f64"`
	} `json:"cases"`
	SlidingMask [][]bool `json:"sliding_mask"`
}

func loadTiny(t *testing.T) (*Model, tinyGolden) {
	t.Helper()
	raw, err := os.ReadFile(tinyDir + "/golden.json")
	if err != nil {
		t.Fatalf("the committed tiny fixture is missing (scripts/pin_embeddinggemma2_tiny.py writes it): %v", err)
	}
	var g tinyGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	m, err := Load(tinyDir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return m, g
}

func flat(rows [][]float64) []float64 {
	var out []float64
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

func cosMax(got []float32, want []float64) (cos, maxAbs float64) {
	var dot, ng, nw float64
	for i, w := range want {
		g := float64(got[i])
		dot += g * w
		ng += g * g
		nw += w * w
		maxAbs = math.Max(maxAbs, math.Abs(g-w))
	}
	return dot / math.Sqrt(ng*nw), maxAbs
}

// TestTiny_parity is Gate 1 (docs/tasks/task-embeddinggemma2.md): on a random-weight fixture pinned with
// transformers' own embedding_gemma2 modeling code, every layer's hidden state, the projected last hidden state, the
// pooled vector before normalisation and the sentence embedding match the float32 reference to cosine 0.9999 (each on
// its own, so a pooling or normalisation bug cannot pass on a matching hidden state), and the embedding is as accurate
// as the reference's own float32 run: its max |diff| to the same weights run in float64 is at most 1.5x the float32
// reference's, plus 1e-7 (float32 rounding in the reference is the same size as goinfer's, so an absolute bar on the
// float32 output cannot tell them apart; the 1e-7 floor is the task doc's original band, about three float32 ulps at
// these values, because where the reference's own error is a couple of ulps, 1.5x of it is a rounding coin toss that
// differs by arch: linux/amd64, which does not fuse multiply-adds, read 1.06e-7 against arm64's 6.4e-8 on the one-token
// case). On a miss the first divergent layer is named.
// The fixture has 1-, 2-, 9- and 17-token cases, sliding and full layers of different head dims and KV-head counts,
// a sliding radius (3) the longer cases exceed, and layer_scalar and norm weights that are not 1.
func TestTiny_parity(t *testing.T) {
	m, g := loadTiny(t)
	if len(g.Cases) < 4 {
		t.Fatalf("golden has %d cases, want 4", len(g.Cases))
	}
	for ci, c := range g.Cases {
		last, layers, err := m.forward(c.IDs, true)
		if err != nil {
			t.Fatalf("case %d: %v", ci, err)
		}
		// HF's output_hidden_states holds each layer's input and, last, the projected output (it equals
		// last_hidden_state, checked below), so the per-layer comparison covers the layer inputs.
		if len(layers) != len(c.Layers) {
			t.Fatalf("case %d: %d layer states, reference has %d", ci, len(layers), len(c.Layers))
		}
		firstBad := -1
		for li := range layers[:len(layers)-1] {
			if cs, _ := cosMax(layers[li], flat(c.Layers[li])); cs < 0.9999 && firstBad < 0 {
				firstBad = li
				t.Errorf("case %d: layer %d's input (0 = the scaled embeddings) cosine %.6f", ci, li, cs)
			}
		}
		lc, lm := cosMax(last, flat(c.Last))
		pooled := make([]float32, m.Dim())
		T := len(c.IDs)
		for tt := range T {
			for j := range pooled {
				pooled[j] += last[tt*m.Dim()+j] / float32(T)
			}
		}
		pc, pm := cosMax(pooled, c.Pooled)
		emb, err := m.Embed(c.IDs)
		if err != nil {
			t.Fatal(err)
		}
		nc, nm := cosMax(emb, c.Normalized)
		_, gErr := cosMax(emb, c.F64)
		var rErr float64
		for j, w := range c.F64 {
			rErr = math.Max(rErr, math.Abs(c.Normalized[j]-w))
		}
		t.Logf("case %d (%d tokens): last hidden cos %.9f max|d| %.2e; pooled cos %.9f max|d| %.2e; embedding cos %.9f max|d| %.2e; vs float64 goinfer %.2e reference-f32 %.2e",
			ci, T, lc, lm, pc, pm, nc, nm, gErr, rErr)
		for _, chk := range []struct {
			name string
			cos  float64
		}{{"last hidden state", lc}, {"pooled", pc}, {"embedding", nc}} {
			if chk.cos < 0.9999 {
				t.Errorf("case %d %s: cosine %.9f (bar 0.9999); first divergent layer %d", ci, chk.name, chk.cos, firstBad)
			}
		}
		if len(c.F64) != len(emb) {
			t.Fatalf("case %d: golden has no float64 reference (re-pin with scripts/pin_embeddinggemma2_tiny.py)", ci)
		}
		if gErr > 1.5*rErr+1e-7 {
			t.Errorf("case %d: embedding is %.3e from the float64 reference, over 1.5x the float32 reference's own %.3e plus 1e-7", ci, gErr, rErr)
		}
	}
}

// TestTiny_slidingMaskIsTheReferences: the window the forward uses on a sliding layer, |q - k| <= sliding_window, is
// the mask transformers builds for the fixture's longest case, cell for cell.
func TestTiny_slidingMaskIsTheReferences(t *testing.T) {
	m, g := loadTiny(t)
	T := len(g.SlidingMask)
	if T < 2*m.cfg.SlidingWindow+2 {
		t.Fatalf("mask is %dx%d, too short to cut a radius-%d window", T, T, m.cfg.SlidingWindow)
	}
	for q := range T {
		for k := range T {
			want := g.SlidingMask[q][k]
			got := k >= max(0, q-m.cfg.SlidingWindow) && k <= min(T-1, q+m.cfg.SlidingWindow)
			if got != want {
				t.Fatalf("mask[%d][%d]: forward allows %v, reference %v", q, k, got, want)
			}
		}
	}
}

// TestLoadConfig_refusesOtherModels: a config.json that is not embedding_gemma2 is refused by name.
func TestLoadConfig_refusesOtherModels(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(dir+"/config.json", []byte(`{"model_type":"gemma3","text_config":{}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConfig(dir); err == nil {
		t.Fatal("a gemma3 config loaded as embedding_gemma2")
	}
}

// TestTiny_blockingIsInvisible: the attention runs over blocks of query rows against the key range each block can
// reach, and at the default block size the tiny fixture's inputs (17 tokens at most) fit in one block. With blocks of
// 4 and 5 rows, which cut through the sliding radius of 3 and leave a partial last block, every case still matches the
// reference (cosine 0.9999) and the one-block result to float32 rounding (cosine 0.9999999).
func TestTiny_blockingIsInvisible(t *testing.T) {
	m, g := loadTiny(t)
	defer func(b int) { attnBlock = b }(attnBlock)
	for ci, c := range g.Cases {
		attnBlock = 128
		ref, err := m.Embed(c.IDs)
		if err != nil {
			t.Fatal(err)
		}
		for _, b := range []int{4, 5} {
			attnBlock = b
			v, err := m.Embed(c.IDs)
			if err != nil {
				t.Fatal(err)
			}
			cr, _ := cosMax(v, c.Normalized)
			var dot, n1, n2 float64
			for j := range v {
				dot += float64(v[j]) * float64(ref[j])
				n1 += float64(v[j]) * float64(v[j])
				n2 += float64(ref[j]) * float64(ref[j])
			}
			cb := dot / math.Sqrt(n1*n2)
			if cr < 0.9999 || cb < 0.9999999 {
				t.Errorf("case %d (%d tokens), block %d: cosine %.9f to the reference, %.9f to the one-block run", ci, len(c.IDs), b, cr, cb)
			}
		}
	}
}
