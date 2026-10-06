//go:build realckpt

package embeddinggemma2

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func readF32(t *testing.T, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (scripts/pin_embeddinggemma2_vision.py writes it)", err)
	}
	v := make([]float32, len(raw)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return v
}

func readPos(t *testing.T, path string) [][2]int {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	p := make([][2]int, len(raw)/8)
	for i := range p {
		p[i] = [2]int{int(int32(binary.LittleEndian.Uint32(raw[8*i:]))), int(int32(binary.LittleEndian.Uint32(raw[8*i+4:])))}
	}
	return p
}

func cos64(a []float32, b []float64) float64 {
	var dot, na, nb float64
	for i := range b {
		dot += float64(a[i]) * b[i]
		na += float64(a[i]) * float64(a[i])
		nb += b[i] * b[i]
	}
	return dot / math.Sqrt(na*nb)
}

// TestReal_vision is Phase V's gates (docs/tasks/task-embeddinggemma2.md, pre-registered before this ran), on the real
// google/embeddinggemma-2 (GOINFER_EG2_DIR, default ~/models/embeddinggemma-2) against sentence-transformers'
// references (testdata/embeddinggemma2-vision/golden.json, committed; and GOINFER_EG2_VISION_ARTIFACTS, default
// ~/goinfer-logs/embeddinggemma2-vision, HF's own patches, positions and image features per image):
//
//	V1: every case's ids equal the reference's.
//	V2: aikit's tower and projector, fed HF's own patches and positions, match HF's image features to cosine >= 0.9999
//	    on every soft token.
//	V3: aikit's preprocessing gives HF's patch grid and soft-token count; the patches' max |diff| is reported.
//	V4: every case's end-to-end embedding (aikit's preprocessing and tower, then the encoder) has cosine >= 0.999 with
//	    the reference, and the same embedding from HF's own pixels has cosine >= 0.9999. The pre-registered bar was
//	    0.9999 end to end; the read landed in its ambiguous band (0.99924-0.99986) with the whole gap in the resize
//	    (aikit bilinear, the reference bicubic; 1.000000000 from HF's pixels), and the owner accepted it on 2026-10-06
//	    ("since we understand the difference i'm ok with it"). The loose bar covers only the resize: everything after it
//	    is still held to 0.9999 through HF's pixels. V3 and V4 are read for bilinear.
//	R1: with the reference's bicubic (preprocess.go), V3's patch max |diff| is at most one uint8 step on every image.
//	R2: with bicubic, every case's end-to-end cosine is >= 0.9999. Bicubic's float64 variant is logged beside both.
//
// CPU, float32.
func TestReal_vision(t *testing.T) {
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GOINFER_EG2_DIR")
	if dir == "" {
		dir = filepath.Join(home, "models", "embeddinggemma-2")
	}
	art := os.Getenv("GOINFER_EG2_VISION_ARTIFACTS")
	if art == "" {
		art = filepath.Join(home, "goinfer-logs", "embeddinggemma2-vision")
	}
	for _, p := range []string{dir, art} {
		if strings.HasPrefix(p, "/Volumes/") || strings.HasPrefix(p, "/srv/models") {
			t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", p)
		}
	}
	raw, err := os.ReadFile("../testdata/embeddinggemma2-vision/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Items []struct {
			Image, Prompt, Text string
			IDs                 []int     `json:"ids"`
			NSoft               int       `json:"n_soft"`
			Embedding           []float64 `json:"embedding"`
		} `json:"items"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	e, err := LoadEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.EnableVision(); err != nil {
		t.Fatal(err)
	}
	H := e.Model().Config().Hidden
	// The modes read: aikit's bilinear (V3/V4), the reference's bicubic in torchvision's fixed point (R1/R2), and the
	// same bicubic in float64 with one rounding (reported beside it, the fixed point's own share).
	modes := []struct {
		name   string
		resize ImageResize
		float  bool
	}{{"bilinear", ResizeBilinear, false}, {"bicubic", ResizeBicubic, false}, {"bicubic-f64", ResizeBicubic, true}}
	type perImage struct {
		hfFeats []float32
		goFeats map[string][]float32
		n       int
	}
	images := map[string]*perImage{}
	for _, it := range g.Items {
		if _, ok := images[it.Image]; ok {
			continue
		}
		name := strings.TrimSuffix(filepath.Base(it.Image), filepath.Ext(it.Image))
		hfPatches := readF32(t, filepath.Join(art, name+".patches.f32"))
		hfPos := readPos(t, filepath.Join(art, name+".pos.i32"))
		hfFeats := readF32(t, filepath.Join(art, name+".feats.f32"))
		// V2: the tower on HF's own pixels.
		f, n, err := e.ImageFeaturesFrom(hfPatches, hfPos)
		if err != nil {
			t.Fatal(err)
		}
		if n*H != len(hfFeats) {
			t.Fatalf("%s: the tower gave %d soft tokens on HF's patches, HF %d", name, n, len(hfFeats)/H)
		}
		worst := 1.0
		for r := range n {
			var dot, na, nb float64
			for j := range H {
				x, y := float64(f[r*H+j]), float64(hfFeats[r*H+j])
				dot, na, nb = dot+x*y, na+x*x, nb+y*y
			}
			worst = math.Min(worst, dot/math.Sqrt(na*nb))
		}
		t.Logf("%s: V2 tower on HF's pixels, worst soft-token cosine %.9f over %d", name, worst, n)
		if worst < 0.9999 {
			t.Errorf("%s: V2 worst soft-token cosine %.9f, under 0.9999", name, worst)
		}
		img, err := os.ReadFile("../" + it.Image)
		if err != nil {
			t.Fatal(err)
		}
		hfAt := map[[2]int]int{}
		for i, p := range hfPos {
			hfAt[p] = i
		}
		pd := len(hfPatches) / len(hfPos)
		pi := &perImage{hfFeats: f, goFeats: map[string][]float32{}, n: n}
		for _, md := range modes {
			// V3 / R1: this mode's preprocessing against HF's pixels.
			if err := e.SetImageResize(md.resize); err != nil {
				t.Fatal(err)
			}
			eg2ResizeFloat = md.float
			t0 := time.Now()
			goPatches, goPos, err := e.preprocessImage(img)
			pre := time.Since(t0)
			if err != nil {
				t.Fatal(err)
			}
			maxDiff, missing, differ := 0.0, 0, 0
			for i, p := range goPos {
				j, ok := hfAt[p]
				if !ok {
					missing++
					continue
				}
				for k := range pd {
					d := math.Abs(float64(goPatches[i*pd+k] - hfPatches[j*pd+k]))
					maxDiff = math.Max(maxDiff, d)
					if d > 1e-6 {
						differ++
					}
				}
			}
			goFeats, goN, err := e.ImageFeatures(img)
			eg2ResizeFloat = false
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%s %-11s: V3 grid %d patches (HF %d, %d not at an HF position), soft tokens %d (HF %d), patch max|diff| %.3e (%.2f uint8 steps), %d of %d values differ; preprocess %s (exploratory)",
				name, md.name, len(goPos), len(hfPos), missing, goN, n, maxDiff, maxDiff*255, differ, len(goPos)*pd, pre.Round(time.Millisecond))
			if len(goPos) != len(hfPos) || missing != 0 || goN != n {
				t.Errorf("%s %s: V3 grid differs: %d patches (HF %d), %d off-grid, %d soft tokens (HF %d)", name, md.name, len(goPos), len(hfPos), missing, goN, n)
			}
			if md.name == "bicubic" && maxDiff > 1.0/255+1e-6 {
				t.Errorf("%s: R1 bicubic patch max|diff| %.3e, over one uint8 step", name, maxDiff)
			}
			pi.goFeats[md.name] = goFeats
		}
		images[it.Image] = pi
	}
	for i, it := range g.Items {
		im := images[it.Image]
		in := ImageInput{Text: it.Text, Prompt: it.Prompt}
		ids, _, err := e.TokenizeImage(in, im.n)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprint(ids) != fmt.Sprint(it.IDs) {
			t.Errorf("item %d (%s, prompt %q, text %q): V1 ids differ from the reference (%d against %d)", i, it.Image, it.Prompt, it.Text, len(ids), len(it.IDs))
			continue
		}
		vHF, _, err := e.EmbedImageFeatures(in, im.hfFeats, im.n)
		if err != nil {
			t.Fatal(err)
		}
		cHF := cos64(vHF, it.Embedding)
		line := fmt.Sprintf("item %2d %-38s prompt %-6q text %-5v: from HF's pixels %.9f", i, it.Image, it.Prompt, it.Text != "", cHF)
		if cHF < 0.9999 {
			t.Errorf("item %d: from HF's own pixels cosine %.9f, under 0.9999: something after the resize moved", i, cHF)
		}
		for _, md := range modes {
			v, _, err := e.EmbedImageFeatures(in, im.goFeats[md.name], im.n)
			if err != nil {
				t.Fatal(err)
			}
			c := cos64(v, it.Embedding)
			line += fmt.Sprintf(", %s %.9f", md.name, c)
			switch md.name {
			case "bilinear":
				if c < 0.999 {
					t.Errorf("item %d: V4 bilinear end-to-end cosine %.9f, under 0.999", i, c)
				}
			case "bicubic":
				if c < 0.9999 {
					t.Errorf("item %d: R2 bicubic end-to-end cosine %.9f, under 0.9999", i, c)
				}
			}
		}
		t.Log(line)
	}
}
