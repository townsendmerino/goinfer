package vision

import (
	"compress/gzip"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Gate G0 of P8a (goinfer docs/measurements/p8a-qwen35-vl-2026-09/preregistration.md), pre-registered:
//
//	S1 patch_embed + interpolated pos_embed   per-row cosine >= 0.999999
//	S2 last block output (pre-merger)         per-row cosine >= 0.9999   (park band [0.999, bar))
//	S3 merged features                        per-row cosine >= 0.9999   (park band [0.999, bar))
//	bit-exact: bilinear tap INDICES equal HF's, weights within 1 ulp f32, rotary position ids equal.
//	A1 (pre-registration amendment, before any real-tower number): relative max|diff| =
//	max|got-golden|/max|golden| <= 5e-6 per stage on the tiny fixture (2e-5 real; see the doc) —
//	cosine alone passed a merger with GELU-tanh in place of erf.
//
// The bars are per ROW (one patch token pre-merge, one merged token post-merge), not one cosine over
// the whole tensor as the Qwen2.5-VL gate takes: a whole-tensor cosine lets one bad row hide behind
// hundreds of good ones.
type qwen3Golden struct {
	GridTHW       [][3]int  `json:"grid_thw"`
	NPatches      int       `json:"n_patches"`
	NMerged       int       `json:"n_merged"`
	Hidden        int       `json:"hidden"`
	OutHidden     int       `json:"out_hidden"`
	PixelValues   []float32 `json:"pixel_values"`
	S1            []float32 `json:"s1"`
	S2            []float32 `json:"s2"`
	S3            []float32 `json:"s3"`
	InterpIndices []int     `json:"interp_indices"`
	InterpWeights []float32 `json:"interp_weights"`
	PosIDs        []int     `json:"pos_ids"`
}

func readQwen3Golden(t *testing.T, path string) qwen3Golden {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("no golden %s (%v)", path, err)
	}
	defer f.Close()
	var r io.Reader = f
	if strings.HasSuffix(path, ".gz") {
		gz, err := gzip.NewReader(f)
		if err != nil {
			t.Fatalf("gunzip %s: %v", path, err)
		}
		defer gz.Close()
		r = gz
	}
	raw, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	var g qwen3Golden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	return g
}

// rowCosines returns the cosine of each of the len(a)/dim rows and the worst row's index.
func rowCosines(a, b []float32, dim int) (worst float64, worstRow int, maxAbs float64) {
	worst = 2
	for r := 0; r < len(a)/dim; r++ {
		var dot, na, nb float64
		for i := r * dim; i < (r+1)*dim; i++ {
			x, y := float64(a[i]), float64(b[i])
			dot += x * y
			na += x * x
			nb += y * y
			maxAbs = math.Max(maxAbs, math.Abs(x-y))
		}
		c := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		if c < worst {
			worst, worstRow = c, r
		}
	}
	return worst, worstRow, maxAbs
}

func gateStage(t *testing.T, name string, got, want []float32, dim int, bar, relBar float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, golden %d", name, len(got), len(want))
	}
	worst, row, maxAbs := rowCosines(got, want, dim)
	var scale float64
	for _, w := range want {
		scale = math.Max(scale, math.Abs(float64(w)))
	}
	rel := maxAbs / scale
	t.Logf("%s: worst-row cosine %.9f (row %d of %d), max|diff| %.3g, relative %.3g (bar %.3g)", name, worst, row, len(got)/dim, maxAbs, rel, relBar)
	switch {
	case rel <= relBar:
	case rel < relBar*5 && relBar > 5e-6: // real-tower park band [2e-5, 1e-4)
		t.Errorf("%s: PARKED — relative max|diff| %.3g in [%g, %g)", name, rel, relBar, relBar*5)
	default:
		t.Errorf("%s: FAIL — relative max|diff| %.3g > %g", name, rel, relBar)
	}
	switch {
	case worst >= bar:
	case worst >= 0.999:
		t.Errorf("%s: PARKED — worst-row cosine %.9f in [0.999, %g): neither pass nor defect; per-block diff before any change", name, worst, bar)
	default:
		t.Errorf("%s: FAIL — worst-row cosine %.9f < 0.999 (row %d)", name, worst, row)
	}
}

func checkQwen3Tower(t *testing.T, ckpt string, g qwen3Golden, relBar float64) {
	t.Helper()
	enc, err := LoadQwen3VisionEncoder(ckpt, false)
	if err != nil {
		t.Fatalf("LoadQwen3VisionEncoder: %v", err)
	}
	// Bit-exact sub-gates first: they need no forward pass and name their own stage.
	coords := patchCoords(g.GridTHW, enc.Cfg.SpatialMergeSize)
	if len(coords) != g.NPatches || len(g.PosIDs) != 2*g.NPatches || len(g.InterpIndices) != 4*g.NPatches {
		t.Fatalf("golden shape: %d coords, %d pos ids, %d tap indices for %d patches", len(coords), len(g.PosIDs), len(g.InterpIndices), g.NPatches)
	}
	side := enc.Cfg.gridSide()
	var idx [4]int
	var wt [4]float32
	for i, p := range coords {
		if p.row != g.PosIDs[2*i] || p.col != g.PosIDs[2*i+1] {
			t.Fatalf("rotary position ids differ at patch %d: got (%d,%d), HF (%d,%d)", i, p.row, p.col, g.PosIDs[2*i], g.PosIDs[2*i+1])
		}
		interpTaps(p.row, p.gh, side, p.col, p.gw, &idx, &wt)
		for k := range 4 {
			if idx[k] != g.InterpIndices[4*i+k] {
				t.Fatalf("bilinear tap index differs at patch %d tap %d: got %d, HF %d", i, k, idx[k], g.InterpIndices[4*i+k])
			}
			want := g.InterpWeights[4*i+k]
			if d := math.Abs(float64(wt[k] - want)); d > math.Abs(float64(want))*1.2e-7+1e-45 {
				t.Fatalf("bilinear weight differs by more than 1 ulp at patch %d tap %d: got %v, HF %v", i, k, wt[k], want)
			}
		}
	}

	s1, err := enc.Embed(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	gateStage(t, "S1 patch_embed+pos_embed", s1, g.S1, g.Hidden, 0.999999, relBar)
	s2, err := enc.ForwardViT(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatalf("ForwardViT: %v", err)
	}
	gateStage(t, "S2 last_hidden_state", s2, g.S2, g.Hidden, 0.9999, relBar)
	s3, err := enc.Forward(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	gateStage(t, "S3 merged features", s3, g.S3, g.OutHidden, 0.9999, relBar)
}

// TestQwen3VisionEncoder_parity is the small form of G0: a tiny random tower of the real structure
// (non-zero biases, non-unit norm weights, merger fc1 scaled x8, two images packed in one call). The
// tiny checkpoint is gitignored (*.safetensors, like qwen25vl-vision-tiny) and regenerated by the pin
// script, so on a box that never ran it this SKIPS — read -v for --- PASS, not `ok`.
func TestQwen3VisionEncoder_parity(t *testing.T) {
	const ckpt = "testdata/qwen35vl-vision-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no %s (%v); run scripts/oracle/pin_qwen35_vision.py", ckpt, err)
	}
	checkQwen3Tower(t, ckpt, readQwen3Golden(t, "testdata/qwen35vl_vision_golden.json"), 5e-6)
}

// TestQwen3VisionEncoder_realParity is the pre-registered G0 on the real Qwen3.5-0.8B tower: four
// grids (square, non-square, tall, tiny). Assets: the checkpoint and a golden written OUTSIDE the
// repo by `pin_qwen35_vision.py --real` (AIKIT_QWEN35_08B / AIKIT_QWEN35VL_TOWER_GOLDEN override).
func TestQwen3VisionEncoder_realParity(t *testing.T) {
	ckpt := os.Getenv("AIKIT_QWEN35_08B")
	if ckpt == "" {
		ckpt = filepath.Join(os.Getenv("HOME"), "models", "qwen3.5-0.8b")
	}
	gold := os.Getenv("AIKIT_QWEN35VL_TOWER_GOLDEN")
	if gold == "" {
		gold = filepath.Join(os.Getenv("HOME"), "models", "qwen35vl_tower_golden", "qwen35vl_tower_golden.json.gz")
	}
	if _, err := os.Stat(filepath.Join(ckpt, "config.json")); err != nil {
		t.Skipf("no real checkpoint at %s", ckpt)
	}
	checkQwen3Tower(t, ckpt, readQwen3Golden(t, gold), 2e-5)
}

// TestQwen3VisionEncoder_refusesWhatItCannotRun pins the loader's refusals: DeepStack indexes that are not distinct,
// increasing block indexes, and an activation the block MLP does not implement, must fail at load, loudly. A valid
// DeepStack list (Qwen3-VL proper) is accepted since goinfer's S10.
func TestQwen3VisionEncoder_refusesWhatItCannotRun(t *testing.T) {
	base := Qwen3EncoderConfig{Depth: 1, HiddenSize: 64, IntermediateSize: 96, NumHeads: 4, InChannels: 3,
		PatchSize: 4, SpatialMergeSize: 2, TemporalPatchSize: 2, OutHiddenSize: 48, NumPositionEmbeddings: 64,
		HiddenAct: "gelu_pytorch_tanh"}
	if err := base.validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	ds := base
	ds.Depth, ds.DeepstackVisualIndexes = 3, []int{0, 2}
	if err := ds.validate(); err != nil {
		t.Fatalf("a valid DeepStack config rejected: %v", err)
	}
	for name, mut := range map[string]func(*Qwen3EncoderConfig){
		"deepstack past depth": func(c *Qwen3EncoderConfig) { c.DeepstackVisualIndexes = []int{8, 16, 24} },
		"deepstack repeated":   func(c *Qwen3EncoderConfig) { c.Depth, c.DeepstackVisualIndexes = 3, []int{1, 1} },
		"deepstack decreasing": func(c *Qwen3EncoderConfig) { c.Depth, c.DeepstackVisualIndexes = 3, []int{2, 0} },
		"silu":                 func(c *Qwen3EncoderConfig) { c.HiddenAct = "silu" },
		"empty act":            func(c *Qwen3EncoderConfig) { c.HiddenAct = "" },
		"non-square pos":       func(c *Qwen3EncoderConfig) { c.NumPositionEmbeddings = 60 },
		"zero pos table":       func(c *Qwen3EncoderConfig) { c.NumPositionEmbeddings = 0 },
		"head_dim not /4":      func(c *Qwen3EncoderConfig) { c.HiddenSize = 24 },
	} {
		c := base
		mut(&c)
		if err := c.validate(); err == nil {
			t.Errorf("%s: validate accepted a config the tower cannot run", name)
		}
	}
}
