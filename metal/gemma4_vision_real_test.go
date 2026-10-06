//go:build darwin

package metal

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

	"github.com/townsendmerino/goinfer/embeddinggemma2"
)

func g4vReadF32(t *testing.T, path string) []float32 {
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

func g4vReadPos(t *testing.T, path string) [][2]int {
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

// TestG4VMetal_realParity is Phase VM's gates VM2 and VM3 (docs/tasks/task-embeddinggemma2.md) on the real
// google/embeddinggemma-2 (GOINFER_EG2_DIR, default ~/models/embeddinggemma-2, never the archive), with the encoder and
// its tower on Metal:
//
//	VM2: fed HF's own pixels and positions (GOINFER_EG2_VISION_ARTIFACTS, default ~/goinfer-logs/embeddinggemma2-vision),
//	     the Metal tower's soft tokens match HF's image features to cosine >= 0.9999 on every one, all four images.
//	VM3: end to end (the bicubic resize, the Metal tower, the Metal encoder), every one of the golden's 12 cases at
//	     cosine >= 0.9999 with sentence-transformers (testdata/embeddinggemma2-vision/golden.json).
//
// It also times the Metal and the CPU tower on each image, alternating, in this process: exploratory, not a result.
//
//	GOINFER_EG2_REAL=1 go test -count=1 -run '^TestG4VMetal_realParity$' -v ./metal/
func TestG4VMetal_realParity(t *testing.T) {
	if os.Getenv("GOINFER_EG2_REAL") != "1" {
		t.Skip("set GOINFER_EG2_REAL=1 (loads the real checkpoint)")
	}
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
			Embedding           []float64 `json:"embedding"`
		} `json:"items"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	e, err := embeddinggemma2.LoadEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := e.UseAccelerator("metal"); err != nil {
		t.Fatal(err)
	}
	if err := e.EnableVision(); err != nil {
		t.Fatal(err)
	}
	if e.VisionDevice() != "metal" {
		t.Fatalf("the tower runs on %q, want metal", e.VisionDevice())
	}
	t.Logf("tower on %s, encoder on %s, resize %s", e.VisionDevice(), e.Accelerator().Name(), e.ImageResizeMode())
	H := e.Model().Config().Hidden
	feats := map[string][]float32{}
	nSoft := map[string]int{}
	for _, it := range g.Items {
		if _, ok := feats[it.Image]; ok {
			continue
		}
		name := strings.TrimSuffix(filepath.Base(it.Image), filepath.Ext(it.Image))
		hfP := g4vReadF32(t, filepath.Join(art, name+".patches.f32"))
		hfPos := g4vReadPos(t, filepath.Join(art, name+".pos.i32"))
		hfF := g4vReadF32(t, filepath.Join(art, name+".feats.f32"))
		t0 := time.Now()
		mf, n, err := e.ImageFeaturesFrom(hfP, hfPos)
		tm := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		t0 = time.Now()
		cf, _, err := e.ImageFeaturesCPU(hfP, hfPos)
		tc := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		if len(mf) != len(hfF) {
			t.Fatalf("%s: %d soft-token values, HF %d", name, len(mf), len(hfF))
		}
		wHF, wCPU := g4vWorst(mf, hfF, H), g4vWorst(mf, cf, H)
		t.Logf("%-26s %4d patches: VM2 worst soft-token cosine %.9f against HF (%.9f against the CPU tower); tower Metal %s, CPU %s (exploratory)",
			name, len(hfPos), wHF, wCPU, tm.Round(time.Millisecond), tc.Round(time.Millisecond))
		if wHF < 0.9999 {
			t.Errorf("%s: VM2 worst soft-token cosine %.9f, under 0.9999", name, wHF)
		}
		img, err := os.ReadFile("../" + it.Image)
		if err != nil {
			t.Fatal(err)
		}
		f, n2, err := e.ImageFeatures(img)
		if err != nil {
			t.Fatal(err)
		}
		if n2 != n {
			t.Fatalf("%s: %d soft tokens from the image, %d from HF's pixels", name, n2, n)
		}
		feats[it.Image], nSoft[it.Image] = f, n
	}
	for i, it := range g.Items {
		in := embeddinggemma2.ImageInput{Text: it.Text, Prompt: it.Prompt}
		ids, _, err := e.TokenizeImage(in, nSoft[it.Image])
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprint(ids) != fmt.Sprint(it.IDs) {
			t.Errorf("item %d: ids differ from the reference", i)
			continue
		}
		v, _, err := e.EmbedImageFeatures(in, feats[it.Image], nSoft[it.Image])
		if err != nil {
			t.Fatal(err)
		}
		var dot, na, nb float64
		for j, r := range it.Embedding {
			dot += float64(v[j]) * r
			na += float64(v[j]) * float64(v[j])
			nb += r * r
		}
		c := dot / math.Sqrt(na*nb)
		t.Logf("item %2d %-38s prompt %-6q text %-5v: VM3 cosine %.9f", i, it.Image, it.Prompt, it.Text != "", c)
		if c < 0.9999 {
			t.Errorf("item %d: VM3 end-to-end cosine %.9f, under 0.9999", i, c)
		}
	}
}
