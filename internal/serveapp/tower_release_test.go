package serveapp

import (
	"errors"
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// oomGridTower is a device tower whose every call fails for want of device memory, so deviceFallback moves the tower to
// the CPU.
type oomGridTower struct{ closed bool }

func (f *oomGridTower) Name() string { return "fake" }
func (f *oomGridTower) Hidden([]float32, [][3]int) ([]float32, error) {
	return nil, errors.New("device allocation failed (fake)")
}
func (f *oomGridTower) Close() error { f.closed = true; return nil }

type fakeG4Tower struct{}

func (fakeG4Tower) Name() string                                  { return "fake" }
func (fakeG4Tower) Hidden([]float32, [][2]int) ([]float32, error) { return nil, nil }
func (fakeG4Tower) Close() error                                  { return nil }

type attachedQwenResident struct{}

func (attachedQwenResident) ForwardViT([]float32, [][3]int) ([]float32, error) { return nil, nil }
func (attachedQwenResident) Close()                                            {}

// countReleases wraps releaseTowerBlocks and counts the calls per family.
func countReleases(t *testing.T) map[string]int {
	got := map[string]int{}
	prev := releaseTowerBlocks
	releaseTowerBlocks = func(family string, enc any) { got[family]++; prev(family, enc) }
	t.Cleanup(func() { releaseTowerBlocks = prev })
	return got
}

func released(enc any) bool {
	r, ok := enc.(interface{ BlocksReleased() bool })
	return ok && r.BlocksReleased()
}

func pixelsFor(n int) []float32 {
	px := make([]float32, n)
	for i := range px {
		px[i] = float32(math.Sin(float64(i) * 0.37))
	}
	return px
}

func sameBits(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: differs at %d: %v against %v", what, i, got[i], want[i])
		}
	}
}

// TestTowerRelease_gridTowersFallBackOnReloadedBlocks is TWR-G2's grid half (docs/tasks/task-memory-campaign-2026-10.md,
// MEM4(b) gates): the Qwen3.5+ and GLM-OCR towers release their host blocks once the device tower starts, and when the
// device runs out of memory the CPU fallback reloads them and returns the never-released CPU tower's features, bit for bit.
func TestTowerRelease_gridTowersFallBackOnReloadedBlocks(t *testing.T) {
	got := countReleases(t)
	grid := [3]int{1, 4, 4}

	q3dev := &oomGridTower{}
	multimodal.RegisterQwen3Tower("fake-oom", func(*vision.Qwen3VisionEncoder) (multimodal.GridTowerAccelerator, error) { return q3dev, nil })
	t.Cleanup(func() { multimodal.UnregisterQwen3Tower("fake-oom") })
	q := &qwen3Tower{dir: "../../testdata/qwen35vl-tiny", plan: gridTowerPlan{device: "fake-oom", where: "fake"}}
	enc, err := q.encoder()
	if err != nil {
		t.Fatal(err)
	}
	if got["Qwen3.5"] != 1 || !released(enc) {
		t.Fatalf("Qwen3.5: %d releases, blocks released %v; want 1 and true", got["Qwen3.5"], released(enc))
	}
	c := enc.Cfg
	px := pixelsFor(16 * c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize)
	feats, err := q.features(px, grid)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := vision.LoadQwen3VisionEncoder(q.dir, false)
	if err != nil {
		t.Fatal(err)
	}
	want, err := multimodal.Qwen3TowerFeatures(ref, nil, px, [][3]int{grid})
	if err != nil {
		t.Fatal(err)
	}
	sameBits(t, "Qwen3.5 fallback features", feats, want)
	if !q3dev.closed || released(enc) {
		t.Errorf("Qwen3.5: device tower closed %v, blocks still released %v after the fallback; want true and false", q3dev.closed, released(enc))
	}

	gdev := &oomGridTower{}
	multimodal.RegisterGlmOcrTower("fake-oom", func(*vision.GlmOcrVisionEncoder) (multimodal.GridTowerAccelerator, error) { return gdev, nil })
	t.Cleanup(func() { multimodal.UnregisterGlmOcrTower("fake-oom") })
	g := &glmOcrTower{dir: "../../testdata/glm-ocr-vision-tiny", plan: gridTowerPlan{device: "fake-oom", where: "fake"}}
	genc, err := g.encoder()
	if err != nil {
		t.Fatal(err)
	}
	if got["GLM-OCR"] != 1 || !released(genc) {
		t.Fatalf("GLM-OCR: %d releases, blocks released %v; want 1 and true", got["GLM-OCR"], released(genc))
	}
	gc := genc.Cfg
	gpx := pixelsFor(16 * gc.InChannels * gc.TemporalPatchSize * gc.PatchSize * gc.PatchSize)
	gfeats, err := g.features(gpx, grid)
	if err != nil {
		t.Fatal(err)
	}
	gref, err := vision.LoadGlmOcrVisionEncoder(g.dir, false)
	if err != nil {
		t.Fatal(err)
	}
	gwant, err := multimodal.GlmOcrTowerFeatures(gref, nil, gpx, [][3]int{grid})
	if err != nil {
		t.Fatal(err)
	}
	sameBits(t, "GLM-OCR fallback features", gfeats, gwant)
}

// TestTowerRelease_startupTowers is TWR-G2's startup half: the Gemma 4 tower and Qwen2.5-VL's resident tower release
// their host blocks once each when a device tower attaches, and not at all when the tower stays on the CPU.
func TestTowerRelease_startupTowers(t *testing.T) {
	got := countReleases(t)
	multimodal.RegisterGemma4Tower("metal", func(*vision.Gemma4Encoder) (multimodal.Gemma4TowerAccelerator, error) { return fakeG4Tower{}, nil })
	t.Cleanup(func() { multimodal.UnregisterGemma4Tower("metal") })
	vision.RegisterQwenResident(func(*vision.QwenVisionEncoder) (vision.QwenResidentEncoder, error) {
		return attachedQwenResident{}, nil
	})
	t.Cleanup(func() { vision.RegisterQwenResident(nil) })

	s := &server{models: map[string]*loadedModel{}}
	for _, c := range []struct {
		backend string
		want    int
	}{{"metal", 1}, {"", 0}} {
		before := got["Gemma 4"]
		if err := s.loadGemma4VisionTower("../../testdata/gemma4-vision-tiny", false, c.backend, false, false); err != nil {
			t.Fatal(err)
		}
		if n := got["Gemma 4"] - before; n != c.want {
			t.Errorf("Gemma 4 on %q: %d releases, want %d", c.backend, n, c.want)
		}
		before = got["Qwen2.5-VL"]
		if err := s.loadQwenVisionTower("../../testdata/qwen25vl-tiny", false, c.backend, false, false); err != nil {
			t.Fatal(err)
		}
		if n := got["Qwen2.5-VL"] - before; n != c.want {
			t.Errorf("Qwen2.5-VL on %q: %d releases, want %d", c.backend, n, c.want)
		}
	}
}
