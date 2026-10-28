package serveapp

import (
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

// TestUnload_closesTheDeviceTowers is MEM2-G3 (docs/tasks/task-memory-campaign-2026-10.md, MEM2 gates): through the admin
// unload and the real drain, each device tower an entry built (the Gemma 4 vision tower, the Qwen3.5+ and GLM-OCR
// towers' accelerators, the Gemma 4 audio accelerator) is closed exactly once and the entry holds none afterwards; an entry
// that built none closes nothing.
func TestUnload_closesTheDeviceTowers(t *testing.T) {
	for _, built := range []bool{true, false} {
		g4v, q3, glm, g4a := &fakeTower{}, &fakeTower{}, &fakeTower{}, &fakeTower{}
		m := &fakeModel{}
		lm := &loadedModel{name: "t", model: m, qwen3: &qwen3Tower{}, glm: &glmOcrTower{}}
		if built {
			lm.gemma4Tower, lm.gemma4AudioAcc = g4v, g4a
			lm.qwen3.acc, lm.glm.acc = closeCountGrid{q3}, closeCountGrid{glm}
		}
		s := &server{models: map[string]*loadedModel{"t": lm}, liveness: map[modelHandle]*modelLiveness{m: {refs: 1}},
			draining: map[string]struct{}{}, cfg: config{allowAdmin: true, unloadDrainWait: 5 * time.Second}}
		w := httptest.NewRecorder()
		s.handleAdminUnload(w, httptest.NewRequest("POST", "/admin/models/unload", strings.NewReader(`{"name":"t"}`)))
		if w.Code != http.StatusOK {
			t.Fatalf("unload: %d %s", w.Code, w.Body.String())
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
		if lm.gemma4Tower != nil || lm.gemma4AudioAcc != nil || lm.qwen3 != nil || lm.glm != nil {
			t.Errorf("built %v: the entry still holds a tower after its unload", built)
		}
	}
}
