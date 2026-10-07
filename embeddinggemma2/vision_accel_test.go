package embeddinggemma2

import (
	"errors"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

type fakeText struct{ name string }

func (f fakeText) Name() string                                      { return f.name }
func (fakeText) Forward([]int, bool) ([]float32, [][]float32, error) { return nil, nil, nil }
func (fakeText) ForwardEmbeds([]float32, int, bool) ([]float32, [][]float32, error) {
	return nil, nil, nil
}
func (fakeText) Close() error { return nil }

type fakeTower struct{ closed *int }

func (fakeTower) Name() string                                  { return "fake-ok" }
func (fakeTower) Hidden([]float32, [][2]int) ([]float32, error) { return nil, nil }
func (f fakeTower) Close() error                                { *f.closed++; return nil }

// TestBindVisionAccel: the tower follows the encoder's accelerator by name, and VisionDevice says where it runs in
// each case: no accelerator, an accelerator with no tower of its name, a tower that declines, one that starts; binding
// again closes the tower it replaces.
func TestBindVisionAccel(t *testing.T) {
	closed := 0
	RegisterVisionAccelerator("fake-ok", func(*vision.Gemma4Encoder) (VisionAccelerator, error) { return fakeTower{&closed}, nil })
	RegisterVisionAccelerator("fake-no", func(*vision.Gemma4Encoder) (VisionAccelerator, error) { return nil, errors.New("no device") })
	defer func() {
		multimodal.UnregisterGemma4Tower("fake-ok")
		multimodal.UnregisterGemma4Tower("fake-no")
	}()
	e := &Encoder{vis: &visionTower{}}
	e.bindVisionAccel()
	if e.VisionDevice() != "CPU" || e.VisionAccelerator() != nil {
		t.Fatalf("no accelerator: %q", e.VisionDevice())
	}
	for _, c := range []struct{ accel, want string }{
		{"fake-none", "CPU (no fake-none tower in this binary)"},
		{"fake-no", "CPU (fake-no declined: no device)"},
		{"fake-ok", "fake-ok"},
	} {
		e.accel = fakeText{c.accel}
		e.bindVisionAccel()
		if got := e.VisionDevice(); got != c.want {
			t.Errorf("accelerator %s: VisionDevice %q, want %q", c.accel, got, c.want)
		}
		if (e.VisionAccelerator() != nil) != (c.accel == "fake-ok") {
			t.Errorf("accelerator %s: tower accelerator %v", c.accel, e.VisionAccelerator())
		}
	}
	e.bindVisionAccel()
	if closed != 1 {
		t.Errorf("rebinding closed %d towers, want 1", closed)
	}
	if !strings.Contains(strings.Join(VisionAccelerators(), ","), "fake-ok") {
		t.Errorf("VisionAccelerators %v", VisionAccelerators())
	}
	if (&Encoder{}).VisionDevice() != "" {
		t.Error("VisionDevice before EnableVision is not empty")
	}
}
