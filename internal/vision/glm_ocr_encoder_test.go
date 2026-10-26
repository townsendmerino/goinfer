package vision

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// Gate O2 of goinfer docs/tasks/task-glm-ocr-2026-10.md: the GLM-OCR vision tower against HF's
// GlmOcrVisionModel (transformers 5.12.0, f32, CPU), tower-only. Pinned by
// goinfer scripts/pin_glm_ocr_vision.py:
//
//	tiny  (committed): a random tower of the real structure — q/k norm, axial rotary over head_dim 64,
//	      RMSNorm, gated MLP, biased Conv2d downsample, unbiased merger — every norm weight and bias
//	      non-trivial, two non-square images in one call. Four stages, so a failure names its stage:
//	      s1 patch_embed, s2 post_layernorm (pre-downsample), s3 last_hidden_state (downsample out),
//	      s4 pooler_output (merged features). Relative max|diff| <= 5e-6 per stage (the Qwen3.5+ gate's
//	      A1 amendment: cosine alone passes a merger with GELU-tanh in place of erf).
//	real  (not committed; skip-guarded): the real checkpoint's tower on the REAL Glm46VImageProcessor's
//	      pixel_values for a small image and a page-sized one. Bar: per-merged-row cosine >= 0.9999.
type glmOcrGolden struct {
	GridTHW     [][3]int  `json:"grid_thw"`
	NPatches    int       `json:"n_patches"`
	NMerged     int       `json:"n_merged"`
	Hidden      int       `json:"hidden"`
	OutHidden   int       `json:"out_hidden"`
	PixelValues []float32 `json:"pixel_values"`
	S1          []float32 `json:"s1"`
	S2          []float32 `json:"s2"`
	S3          []float32 `json:"s3"`
	S4          []float32 `json:"s4"`
	PosIDs      []int     `json:"pos_ids"`
}

const glmTinyCkpt = "testdata/glm-ocr-vision-tiny"

func readGlmGolden(t *testing.T) glmOcrGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/glm_ocr_vision_golden.json")
	if err != nil {
		t.Fatalf("read golden: %v (run goinfer scripts/pin_glm_ocr_vision.py)", err)
	}
	var g glmOcrGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	return g
}

// TestGlmOcrVisionEncoder_parity is the small gate. The tiny checkpoint and golden are COMMITTED (the
// checkpoint with `git add -f`, since *.safetensors is otherwise ignored) so this cannot silently skip
// on a clean clone: a missing file is a Fatal, not a Skip.
func TestGlmOcrVisionEncoder_parity(t *testing.T) {
	g := readGlmGolden(t)
	enc, err := LoadGlmOcrVisionEncoder(glmTinyCkpt, false)
	if err != nil {
		t.Fatalf("LoadGlmOcrVisionEncoder: %v", err)
	}
	// Bit-exact rotary position ids first: they need no forward pass.
	coords := patchCoords(g.GridTHW, enc.Cfg.SpatialMergeSize)
	if len(coords) != g.NPatches || len(g.PosIDs) != 2*g.NPatches {
		t.Fatalf("golden shape: %d coords, %d pos ids for %d patches", len(coords), len(g.PosIDs), g.NPatches)
	}
	for i, p := range coords {
		if p.row != g.PosIDs[2*i] || p.col != g.PosIDs[2*i+1] {
			t.Fatalf("rotary position ids differ at patch %d: got (%d,%d), HF (%d,%d)", i, p.row, p.col, g.PosIDs[2*i], g.PosIDs[2*i+1])
		}
	}
	const relBar = 5e-6
	s1, err := enc.Embed(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	gateStage(t, "S1 patch_embed", s1, g.S1, g.Hidden, 0.999999, relBar)
	s2, err := enc.ForwardViT(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatalf("ForwardViT: %v", err)
	}
	gateStage(t, "S2 post_layernorm", s2, g.S2, g.Hidden, 0.9999, relBar)
	s3 := enc.downsample(s2)
	gateStage(t, "S3 downsample (last_hidden_state)", s3, g.S3, g.OutHidden, 0.9999, relBar)
	s4, err := enc.Forward(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	gateStage(t, "S4 merged features (pooler_output)", s4, g.S4, g.OutHidden, 0.9999, relBar)
	if len(s4) != g.NMerged*g.OutHidden {
		t.Fatalf("merged rows: got %d floats, want %d×%d", len(s4), g.NMerged, g.OutHidden)
	}
}

// TestGlmOcrVisionEncoder_quantSanity: the W8A8 path loads and stays near the f32 golden. Not a gate
// of the quantiser (random weights make a poor judge of that); it proves the quant wiring runs every
// projection, the downsample and the merger without a shape or aliasing fault.
func TestGlmOcrVisionEncoder_quantSanity(t *testing.T) {
	g := readGlmGolden(t)
	enc, err := LoadGlmOcrVisionEncoder(glmTinyCkpt, true)
	if err != nil {
		t.Fatalf("load quant: %v", err)
	}
	s4, err := enc.Forward(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatal(err)
	}
	worst, row, maxAbs := rowCosines(s4, g.S4, g.OutHidden)
	t.Logf("quant (W8A8) vs f32 golden: worst-row cosine %.5f (row %d), max|diff| %.3g", worst, row, maxAbs)
	if worst < 0.95 {
		t.Errorf("quantised tower diverged: worst-row cosine %.5f < 0.95", worst)
	}
}

// TestGlmOcrVisionEncoder_downsampleOrder pins the downsample flattening by itself, without HF: with a
// one-hot weight row that selects flat index c·m²+j, the output must be hidden[g·m²+j][c] — HF's
// view(-1, m, m, C).permute(0, 3, 1, 2) puts the channel OUTERMOST and the merge-block patch (dy·m+dx)
// innermost. A weight read patch-major (j·C+c) would pick a different element.
func TestGlmOcrVisionEncoder_downsampleOrder(t *testing.T) {
	const hidden, m, groups = 6, 2, 3
	const mu = m * m
	out := hidden * mu // one output row per flat input index
	w := make([]float32, out*hidden*mu)
	for o := range out {
		w[o*hidden*mu+o] = 1
	}
	e := &GlmOcrVisionEncoder{
		Cfg:   GlmOcrEncoderConfig{HiddenSize: hidden, OutHiddenSize: out, SpatialMergeSize: m},
		downW: linalg.WrapF32(w, out, hidden*mu),
		downB: make([]float32, out),
	}
	h := make([]float32, groups*mu*hidden)
	for i := range h {
		h[i] = float32(i + 1)
	}
	y := e.downsample(h)
	for g := range groups {
		for c := range hidden {
			for j := range mu {
				got := y[g*out+c*mu+j]
				want := h[(g*mu+j)*hidden+c]
				if got != want {
					t.Fatalf("group %d channel %d patch %d: got %v, want hidden[%d][%d] = %v", g, c, j, got, g*mu+j, c, want)
				}
			}
		}
	}
}

// TestGlmOcrVisionEncoder_refusesWhatItCannotRun pins the loader's refusals.
func TestGlmOcrVisionEncoder_refusesWhatItCannotRun(t *testing.T) {
	no := false
	base := GlmOcrEncoderConfig{Depth: 1, HiddenSize: 128, IntermediateSize: 256, NumHeads: 2, InChannels: 3,
		PatchSize: 4, SpatialMergeSize: 2, TemporalPatchSize: 2, OutHiddenSize: 192, RMSNormEps: 1e-5,
		HiddenAct: "silu", RopeParameters: &glmOcrRopeJSON{RopeTheta: 10000, RopeType: "axial"}}
	if err := base.validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	for name, mut := range map[string]func(*GlmOcrEncoderConfig){
		"gelu act":        func(c *GlmOcrEncoderConfig) { c.HiddenAct = "gelu" },
		"empty act":       func(c *GlmOcrEncoderConfig) { c.HiddenAct = "" },
		"no attn bias":    func(c *GlmOcrEncoderConfig) { c.AttentionBias = &no },
		"other rope type": func(c *GlmOcrEncoderConfig) { c.RopeParameters = &glmOcrRopeJSON{RopeTheta: 10000, RopeType: "yarn"} },
		"other theta":     func(c *GlmOcrEncoderConfig) { c.RopeParameters = &glmOcrRopeJSON{RopeTheta: 5000, RopeType: "axial"} },
		"head_dim not /4": func(c *GlmOcrEncoderConfig) { c.HiddenSize = 24; c.NumHeads = 8 },
		"heads not div":   func(c *GlmOcrEncoderConfig) { c.NumHeads = 3 },
		"zero out hidden": func(c *GlmOcrEncoderConfig) { c.OutHiddenSize = 0 },
		"negative eps":    func(c *GlmOcrEncoderConfig) { c.RMSNormEps = -1 },
	} {
		c := base
		mut(&c)
		if err := c.validate(); err == nil {
			t.Errorf("%s: validate accepted a config the tower cannot run", name)
		}
	}
	// A real file: the loader must reject a grid the tower cannot take, loudly.
	enc, err := LoadGlmOcrVisionEncoder(glmTinyCkpt, false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := enc.Forward(make([]float32, 3*enc.Cfg.patchDim()), [][3]int{{1, 3, 1}}); err == nil {
		t.Error("Forward accepted a grid whose h/w are not multiples of spatial_merge_size")
	}
	if _, err := enc.Forward(make([]float32, 5), [][3]int{{1, 2, 2}}); err == nil {
		t.Error("Forward accepted a pixel_values of the wrong length")
	}
}

// --- real checkpoint --------------------------------------------------------------------------

func glmRealPaths() (ckpt, gold string) {
	ckpt = os.Getenv("AIKIT_GLM_OCR")
	if ckpt == "" {
		ckpt = filepath.Join(os.Getenv("HOME"), "models", "glm-ocr")
	}
	gold = os.Getenv("AIKIT_GLM_OCR_TOWER_GOLDEN")
	if gold == "" {
		gold = filepath.Join(os.Getenv("HOME"), "models", "glm-ocr-tower-golden")
	}
	return ckpt, gold
}

type glmRealCase struct {
	Name      string   `json:"name"`
	GridTHW   [][3]int `json:"grid_thw"`
	NPatches  int      `json:"n_patches"`
	NMerged   int      `json:"n_merged"`
	PatchDim  int      `json:"patch_dim"`
	OutHidden int      `json:"out_hidden"`
	Hidden    int      `json:"hidden"`
}

func readF32(t *testing.T, path string, want int) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != want*4 {
		t.Fatalf("%s: %d bytes, want %d floats", path, len(b), want)
	}
	out := make([]float32, want)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

// TestGlmOcrVisionEncoder_realParity is the pre-registered O2 gate: per-merged-row cosine >= 0.9999 at
// f32 against HF on a small image and a page-sized one, both fed the pixel_values HF's own
// Glm46VImageProcessor produced (dumped by `pin_glm_ocr_vision.py real`). Assets live outside the repo
// (AIKIT_GLM_OCR, AIKIT_GLM_OCR_TOWER_GOLDEN override the defaults); absent -> SKIP, read -v for
// --- PASS. Estimated wall time: small ~2 s, page ~1-3 min on the 16-core Linux box.
func TestGlmOcrVisionEncoder_realParity(t *testing.T) {
	ckpt, gold := glmRealPaths()
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); err != nil {
		t.Skipf("no real checkpoint at %s", ckpt)
	}
	raw, err := os.ReadFile(filepath.Join(gold, "manifest.json"))
	if err != nil {
		t.Skipf("no real golden at %s (run goinfer scripts/pin_glm_ocr_vision.py real): %v", gold, err)
	}
	var man struct {
		Cases []glmRealCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &man); err != nil {
		t.Fatal(err)
	}
	enc, err := LoadGlmOcrVisionEncoder(ckpt, false)
	if err != nil {
		t.Fatalf("LoadGlmOcrVisionEncoder: %v", err)
	}
	if len(man.Cases) == 0 {
		t.Fatal("empty manifest")
	}
	deadline := time.Now().Add(20 * time.Minute)
	for _, c := range man.Cases {
		t.Run(c.Name, func(t *testing.T) {
			pv := readF32(t, filepath.Join(gold, c.Name+".pv.f32"), c.NPatches*c.PatchDim)
			wantPool := readF32(t, filepath.Join(gold, c.Name+".pooler.f32"), c.NMerged*c.OutHidden)
			wantLast := readF32(t, filepath.Join(gold, c.Name+".last.f32"), c.NMerged*c.OutHidden)
			t0 := time.Now()
			fmt.Fprintf(os.Stderr, "[glm-ocr real/%s] %d patches, forward start\n", c.Name, c.NPatches)
			last, err := enc.ForwardDownsampled(pv, c.GridTHW)
			if err != nil {
				t.Fatal(err)
			}
			pool := enc.merge(last)
			dt := time.Since(t0)
			fmt.Fprintf(os.Stderr, "[glm-ocr real/%s] forward done in %.1fs\n", c.Name, dt.Seconds())
			if time.Now().After(deadline) {
				t.Fatalf("deadline exceeded")
			}
			for _, st := range []struct {
				name      string
				got, want []float32
			}{{"last_hidden_state (downsample out)", last, wantLast}, {"pooler_output (merged)", pool, wantPool}} {
				worst, row, maxAbs := rowCosines(st.got, st.want, c.OutHidden)
				var scale float64
				for _, w := range st.want {
					scale = math.Max(scale, math.Abs(float64(w)))
				}
				t.Logf("%s: grid %v, %d rows: MIN per-row cosine %.9f (row %d), max|diff| %.4g (relative to max|ref| %.4g: %.3g); forward %.1fs",
					st.name, c.GridTHW, len(st.got)/c.OutHidden, worst, row, maxAbs, scale, maxAbs/scale, dt.Seconds())
				if worst < 0.9999 {
					t.Errorf("%s: worst-row cosine %.9f < 0.9999 (row %d)", st.name, worst, row)
				}
			}
		})
	}
}

// TestGlmOcrVisionEncoder_costSweep records tower wall time at three pixel counts (1, 2 and 4.8 MP — the
// processor's ceiling, 24,576 patches) on RANDOM pixel_values (cost is value-independent). It is a
// RECORD, not a gate, and runs only when AIKIT_GLM_OCR_COST is set to a comma-separated list of the
// points to run ("1", "2", "4.8"), so a plain `go test` never starts it. Per CLAUDE.md the sweep is a
// night job and a measurement belongs on a quiet box: one in-process run per point, load average
// printed beside each time. Heartbeat: a line at every point start and end (the 4.8 MP point is
// estimated at 10-20 min on this box, the 1 MP point ~1 min).
func TestGlmOcrVisionEncoder_costSweep(t *testing.T) {
	pts := os.Getenv("AIKIT_GLM_OCR_COST")
	if pts == "" {
		t.Skip("set AIKIT_GLM_OCR_COST=1,2,4.8 to record tower wall time (night job, quiet box)")
	}
	ckpt, _ := glmRealPaths()
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); err != nil {
		t.Skipf("no real checkpoint at %s", ckpt)
	}
	grids := map[string][3]int{"1": {1, 70, 72}, "2": {1, 100, 102}, "4.8": {1, 128, 192}}
	enc, err := LoadGlmOcrVisionEncoder(ckpt, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range splitComma(pts) {
		g, ok := grids[p]
		if !ok {
			t.Fatalf("unknown point %q (want 1, 2, 4.8)", p)
		}
		n := g[0] * g[1] * g[2]
		rng := rand.New(rand.NewSource(1))
		pv := make([]float32, n*enc.Cfg.patchDim())
		for i := range pv {
			pv[i] = float32(rng.NormFloat64())
		}
		load0 := loadAvg()
		fmt.Fprintf(os.Stderr, "[glm-ocr cost] point %s MP: grid %v = %d patches (%.2f MP), start, loadavg %s\n", p, g, n, float64(n)*196/1e6, load0)
		t0 := time.Now()
		out, err := enc.Forward(pv, [][3]int{g})
		if err != nil {
			t.Fatal(err)
		}
		dt := time.Since(t0)
		fmt.Fprintf(os.Stderr, "[glm-ocr cost] point %s MP: %d merged rows, tower wall %.2fs, loadavg before %s after %s\n", p, len(out)/enc.Cfg.OutHiddenSize, dt.Seconds(), load0, loadAvg())
		t.Logf("COST point %s MP (%d patches): tower wall %.2fs", p, n, dt.Seconds())
	}
}

func splitComma(s string) []string {
	var out []string
	cur := ""
	for _, r := range s {
		if r == ',' {
			out = append(out, cur)
			cur = ""
			continue
		}
		cur += string(r)
	}
	return append(out, cur)
}

func loadAvg() string {
	b, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return "n/a"
	}
	f := 0
	for i, c := range b {
		if c == ' ' {
			f++
			if f == 3 {
				return string(b[:i])
			}
		}
	}
	return string(b)
}
