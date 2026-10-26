package vision

import "testing"

// TestQwenVision_zeroPatchGrid is the regression for AUDIT #4: a grid that yields
// zero patches (an empty gridTHW, or any grid with a zero dim) must return an
// error, not panic with "integer divide by zero" at len(freqs)/nPatches. These
// inputs are reachable from a request with zero images or a smart-resize that
// rounds a tiny image to a zero grid. The guards run before any weight is touched,
// so a minimal encoder exercises them.
func TestQwenVision_zeroPatchGrid(t *testing.T) {
	e := &QwenVisionEncoder{Cfg: QwenEncoderConfig{SpatialMergeSize: 2}}
	cases := []struct {
		name string
		grid [][3]int
	}{
		{"empty (no images)", nil},
		{"zero temporal", [][3]int{{0, 2, 2}}},
		{"zero height", [][3]int{{1, 0, 0}}},
		{"zero width", [][3]int{{1, 2, 0}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// A missing guard makes this panic; recover so the panic is reported as
			// a failure with context rather than crashing the whole test binary.
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("panicked instead of returning an error: %v", r)
				}
			}()
			if _, err := e.Forward(nil, tc.grid); err == nil {
				t.Errorf("Forward(%v) returned nil error, want a validation error", tc.grid)
			}
		})
	}
}
