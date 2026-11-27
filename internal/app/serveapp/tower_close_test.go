package serveapp

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// fakeTower stands in for every device tower kind (Gemma 4 vision, the grid towers, Gemma 4 audio) and counts its Closes.
type fakeTower struct{ closes int }

func (f *fakeTower) Name() string                                  { return "fake" }
func (f *fakeTower) Hidden([]float32, [][2]int) ([]float32, error) { return nil, nil }
func (f *fakeTower) Blocks([]float32, int) ([]float32, error)      { return nil, nil }
func (f *fakeTower) Close() error                                  { f.closes++; return nil }

type closeCountGrid struct{ *fakeTower }

func (f closeCountGrid) Hidden([]float32, [][3]int) ([]float32, error) { return nil, nil }

// TestUnload_closesTheDeviceTowers is MEM2-G3: through the admin
// unload and the real drain, each device tower an entry built (the Gemma 4 vision tower, the Qwen3.5+ and GLM-OCR
// towers' accelerators, the Gemma 4 audio accelerator) is closed exactly once and the entry holds none afterwards; an entry
// that built none closes nothing. An entry holds one vision family, so each family is its own entry.
func TestUnload_closesTheDeviceTowers(t *testing.T) {
	for _, built := range []bool{true, false} {
		g4v, q3, glm, g4a := &fakeTower{}, &fakeTower{}, &fakeTower{}, &fakeTower{}
		g4 := &loadedModel{modelIdentity: modelIdentity{name: "gemma4"}, modelModalities: modelModalities{vision: &gemma4Vision{}, audio: &gemma4Audio{}}}
		qv := &qwen3Vision{qwen3Tower: &qwen3Tower{}}
		gv := &glmVision{glmOcrTower: &glmOcrTower{}}
		entries := []*loadedModel{g4, {modelIdentity: modelIdentity{name: "qwen3.5"}, modelModalities: modelModalities{vision: qv}}, {modelIdentity: modelIdentity{name: "glm-ocr"}, modelModalities: modelModalities{vision: gv}}}
		if built {
			g4.vision.(*gemma4Vision).tower = g4v
			g4.audio.(*gemma4Audio).acc = g4a
			qv.acc, gv.acc = closeCountGrid{q3}, closeCountGrid{glm}
		}
		s := &server{
			cfg:           config{allowAdmin: true, unloadDrainWait: 5 * time.Second},
			modelRegistry: modelRegistry{models: map[string]*loadedModel{}, liveness: map[io.Closer]*modelLiveness{}, draining: map[string]struct{}{}},
		}
		for _, lm := range entries {
			m := &fakeCloser{}
			lm.closer = m
			s.models[lm.name], s.liveness[m] = lm, &modelLiveness{refs: 1}
		}
		for _, lm := range entries {
			w := httptest.NewRecorder()
			s.handleAdminUnload(w, httptest.NewRequest("POST", "/admin/models/unload", strings.NewReader(`{"name":"`+lm.name+`"}`)))
			if w.Code != http.StatusOK {
				t.Fatalf("unload %s: %d %s", lm.name, w.Code, w.Body.String())
			}
		}
		want := 0
		if built {
			want = 1
		}
		for name, f := range map[string]*fakeTower{"gemma4 vision": g4v, "qwen3.5": q3, "glm-ocr": glm, "gemma4 audio": g4a} {
			if f.closes != want {
				t.Errorf("built %v: the %s tower closed %d times; want %d", built, name, f.closes, want)
			}
		}
		for _, lm := range entries {
			if g, ok := lm.audio.(*gemma4Audio); lm.vision != nil || (ok && g.acc != nil) {
				t.Errorf("built %v: %s still holds a tower after its unload", built, lm.name)
			}
		}
	}
}
