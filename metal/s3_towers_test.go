//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// G-S3a of docs/tasks/task-multimodal-support-2026-10.md (S3), for the rebuilt towers (vl_towers.go): the Metal SigLIP
// and Qwen2.5-VL towers against aikit's CPU tower, both float32, every output token at cosine >= 0.9999; 0.999-0.9999 is
// ambiguous (parked). The tiny towers always, norms randomised first (through the exports' aliasing slices: the tiny
// checkpoints' norms are all ones), with each registered planted defect alone required to go red; the real ones
// (~/models/gemma-3-4b-it, ~/models/qwen25vl-3b-instruct, never the archive) under GOINFER_HEAVY_TESTS=1, on the four
// images F2a uses, attached the way serve attaches them (EnableResident). Times are exploratory.

func s3Grade(t *testing.T, what string, got, want []float32, width int) {
	t.Helper()
	w := gvWorst(t, got, want, width)
	fmt.Fprintf(os.Stderr, "[S3 G-S3a] %s: worst token cosine %.9f\n", what, w)
	switch {
	case w < 0.999:
		t.Errorf("%s: worst token cosine %.9f under 0.999 (G-S3a FAIL)", what, w)
	case w < 0.9999:
		t.Errorf("%s: worst token cosine %.9f in 0.999-0.9999 (G-S3a ambiguous, parked)", what, w)
	}
}

func s3Real(t *testing.T, name string) string {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1 (loads the real tower)")
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", name)
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive (CLAUDE.md)", dir)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s: %v", dir, err)
	}
	return dir
}

var s3Images = []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"}

// s3Defect is one of G-S3a's planted defects, by the registered number.
type s3Defect struct {
	n    int
	name string
	set  func(d *gvDefect)
}

var (
	s3SiglipDefects = []s3Defect{
		{1, "attention scale dropped", func(d *gvDefect) { d.noScale = true }},
		{5, "position table dropped", func(d *gvDefect) { d.noPosEmbed = true }},
		{6, "patch-embed bias dropped", func(d *gvDefect) { d.noPatchBias = true }},
	}
	s3QwenDefects = []s3Defect{
		{1, "attention scale dropped", func(d *gvDefect) { d.noScale = true }},
		{2, "RoPE halves swapped", func(d *gvDefect) { d.swapRope = true }},
		{3, "every block attends its whole frame", func(d *gvDefect) { d.noWindows = true }},
		{4, "window reordering skipped", func(d *gvDefect) { d.noWindowOrder = true }},
	}
)

// The tiny towers are degenerate along several of the registered defects' axes: their biases are all zero (a dropped patch bias
// cannot show) and Qwen2.5-VL's q/k weights are at init scale (0.02), so attention is nearly uniform and the scale, RoPE and
// windowing barely matter (each defect read 0.99999 or better). So, before comparing, the check sharpens the fixture through the
// exports' aliasing slices, which the CPU tower reads too: random biases (s3Sharpen) and q/k projections scaled up (s3Scale), so
// the attention logits are not near zero; for Qwen2.5-VL also v and the output projection, whose init scale left attention a
// negligible share of the residual stream (the tiny tower has one windowed block, then a full one).

func s3Sharpen(rng *rand.Rand, xs ...[]float32) {
	for _, x := range xs {
		for i := range x {
			x[i] = float32(0.5 * rng.NormFloat64())
		}
	}
}

func s3Scale(f float32, xs ...[]float32) {
	for _, x := range xs {
		for i := range x {
			x[i] *= f
		}
	}
}

func TestS3Towers_tiny(t *testing.T) {
	t.Run("siglip", func(t *testing.T) {
		enc, err := vision.LoadEncoder("../testdata/siglip-tiny", false)
		if err != nil {
			t.Fatal(err)
		}
		w, err := enc.Weights()
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewSource(9))
		for _, b := range w.Blocks {
			gvRandomise(rng, b.LN1W, b.LN1B, b.LN2W, b.LN2B)
		}
		s3Sharpen(rng, w.PatchB)
		for _, b := range w.Blocks {
			s3Sharpen(rng, b.Q.B, b.K.B)
			s3Scale(6, b.Q.W, b.K.W)
		}
		r, err := newSiglipVResident(enc)
		if err != nil {
			t.Fatal(err)
		}
		c := enc.Cfg
		px := make([]float32, c.NumChannels*c.ImageSize*c.ImageSize)
		for i := range px {
			px[i] = float32(rng.NormFloat64())
		}
		want, err := enc.Forward(px) // no resident attached: aikit's CPU tower
		if err != nil {
			t.Fatal(err)
		}
		patches, err := enc.GridPatches(px)
		if err != nil {
			t.Fatal(err)
		}
		worst := func() float64 {
			got, err := r.ForwardPatches(patches)
			if err != nil {
				t.Fatal(err)
			}
			return gvWorst(t, got, want, c.HiddenSize)
		}
		got, err := r.ForwardPatches(patches)
		if err != nil {
			t.Fatal(err)
		}
		s3Grade(t, "siglip-tiny", got, want, c.HiddenSize)
		for _, d := range s3SiglipDefects {
			r.a.planted = gvDefect{}
			d.set(&r.a.planted)
			wd := worst()
			r.a.planted = gvDefect{}
			fmt.Fprintf(os.Stderr, "[S3 G-S3a] siglip-tiny planted (%d) %s: worst %.6f\n", d.n, d.name, wd)
			if wd >= 0.9999 {
				t.Errorf("siglip-tiny: planted defect (%d) %s left the bar green (worst %.9f): the fixture cannot see it", d.n, d.name, wd)
			}
		}
	})
	t.Run("qwen2.5-vl", func(t *testing.T) {
		enc, err := vision.LoadQwenVisionEncoder("../testdata/qwen25vl-tiny", false)
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewSource(10))
		gw := enc.GPUWeights()
		gvRandomise(rng, gw.MergerLNw)
		H := gw.Hidden
		for _, b := range gw.Blocks {
			gvRandomise(rng, b.Norm1w, b.Norm2w)
			s3Sharpen(rng, b.QKVb[:2*H])
			s3Scale(12, b.QKVw.F32[:2*H*H])             // the q and k rows
			s3Scale(8, b.QKVw.F32[2*H*H:], b.Projw.F32) // v and the output projection: attention's share of the residual
		}
		r, err := newQwen25VResident(enc)
		if err != nil {
			t.Fatal(err)
		}
		c := enc.Cfg
		pd := c.InChans * c.TemporalPatchSize * c.PatchSize * c.PatchSize
		grids := [][][3]int{{{1, 8, 8}}, {{1, 12, 16}}, {{1, 16, 12}, {1, 8, 8}}}
		type tc struct {
			g    [][3]int
			px   []float32
			want []float32
		}
		var cases []tc
		for _, g := range grids {
			n := 0
			for _, x := range g {
				n += x[0] * x[1] * x[2]
			}
			px := make([]float32, n*pd)
			for i := range px {
				px[i] = float32(rng.NormFloat64())
			}
			want, err := enc.Forward(px, g)
			if err != nil {
				t.Fatal(err)
			}
			cases = append(cases, tc{g, px, want})
		}
		worst := func() float64 {
			w := 1.0
			for _, k := range cases {
				h, err := r.ForwardViT(k.px, k.g)
				if err != nil {
					t.Fatal(err)
				}
				w = math.Min(w, gvWorst(t, enc.MergeHidden(h, k.g), k.want, c.OutHiddenSize))
			}
			return w
		}
		w := worst()
		fmt.Fprintf(os.Stderr, "[S3 G-S3a] qwen25vl-tiny over grids %v: worst token cosine %.9f\n", grids, w)
		if w < 0.9999 {
			t.Errorf("qwen25vl-tiny: worst token cosine %.9f under 0.9999 (G-S3a)", w)
		}
		for _, d := range s3QwenDefects {
			r.a.planted = gvDefect{}
			d.set(&r.a.planted)
			wd := worst()
			r.a.planted = gvDefect{}
			fmt.Fprintf(os.Stderr, "[S3 G-S3a] qwen25vl-tiny planted (%d) %s: worst %.6f\n", d.n, d.name, wd)
			if wd >= 0.9999 {
				t.Errorf("qwen25vl-tiny: planted defect (%d) %s left the bar green (worst %.9f): the fixture cannot see it", d.n, d.name, wd)
			}
		}
	})
}

func TestS3Towers_real(t *testing.T) {
	t.Run("siglip", func(t *testing.T) {
		dir := s3Real(t, "gemma-3-4b-it")
		cpu, err := vision.LoadEncoder(dir, false)
		if err != nil {
			t.Fatal(err)
		}
		dev, err := vision.LoadEncoder(dir, false)
		if err != nil {
			t.Fatal(err)
		}
		if err := dev.EnableResident(); err != nil {
			t.Fatalf("the Metal SigLIP tower did not attach: %v", err)
		}
		t.Cleanup(dev.Close)
		for _, img := range s3Images {
			data, err := os.ReadFile(filepath.Join("../testdata", img))
			if err != nil {
				t.Fatal(err)
			}
			pv, err := vision.Preprocess(data, vision.Gemma3())
			if err != nil {
				t.Fatal(err)
			}
			t0 := time.Now()
			want, err := cpu.Forward(pv.Data)
			tc := time.Since(t0)
			if err != nil {
				t.Fatal(err)
			}
			t0 = time.Now()
			got, err := dev.Forward(pv.Data)
			tm := time.Since(t0)
			if err != nil {
				t.Fatal(err)
			}
			fmt.Fprintf(os.Stderr, "[S3 real siglip] %-30s tower Metal %s, CPU %s (exploratory)\n", img, tm.Round(time.Millisecond), tc.Round(time.Millisecond))
			s3Grade(t, "gemma-3-4b-it "+img, got, want, cpu.Cfg.HiddenSize)
		}
	})
	t.Run("qwen2.5-vl", func(t *testing.T) {
		dir := s3Real(t, "qwen25vl-3b-instruct")
		cpu, err := vision.LoadQwenVisionEncoder(dir, false)
		if err != nil {
			t.Fatal(err)
		}
		dev, err := vision.LoadQwenVisionEncoder(dir, false)
		if err != nil {
			t.Fatal(err)
		}
		if err := dev.EnableResident(); err != nil || !dev.ResidentEnabled() {
			t.Fatalf("the Metal Qwen2.5-VL tower did not attach: %v", err)
		}
		t.Cleanup(dev.Close)
		pp, err := multimodal.LoadQwenPreprocessConfig(dir)
		if err != nil {
			t.Fatal(err)
		}
		for _, img := range s3Images {
			data, err := os.ReadFile(filepath.Join("../testdata", img))
			if err != nil {
				t.Fatal(err)
			}
			px, grid, err := multimodal.QwenPreprocess(data, pp)
			if err != nil {
				t.Fatal(err)
			}
			g := [][3]int{grid}
			t0 := time.Now()
			want, err := cpu.Forward(px, g)
			tc := time.Since(t0)
			if err != nil {
				t.Fatal(err)
			}
			t0 = time.Now()
			got, err := dev.Forward(px, g)
			tm := time.Since(t0)
			if err != nil {
				t.Fatal(err)
			}
			fmt.Fprintf(os.Stderr, "[S3 real qwen2.5-vl] %-30s grid %v: tower Metal %s, CPU %s (exploratory)\n", img, grid, tm.Round(time.Millisecond), tc.Round(time.Millisecond))
			s3Grade(t, "qwen25vl-3b "+img, got, want, cpu.Cfg.OutHiddenSize)
		}
	})
}
