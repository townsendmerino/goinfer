package decoder

import (
	"strings"
	"testing"
)

// R9 (docs/measurements/cold-user-2026-09-06-nobara-pc.md, corrected on review): "cuda-staged
// (int4)" on an 8 GB card sat at the idle VRAM baseline (464 MiB, unchanged) for a full request
// sampled at 1 Hz. Cuda's and metal's own Backend.MatmulBT implementations are bare CPU calls
// and neither implements QuantBackend at all, so their "staged" path never reaches the device at
// ANY quant — DecodePath() folds that into the same requested-vs-effective shape
// BackendSummary() already uses (see decoder/backend_report_test.go), rather than naming a
// "cuda-staged"/"metal-staged" path that was never real. declinedToCPUReason is the pure,
// Model-free piece of that string; DecodePath()'s own wiring is covered by R9's real-hardware
// verification (docs/task-first-hour.md), not a synthetic Model here.
func TestDeclinedToCPUReason(t *testing.T) {
	for _, tc := range []struct {
		backend, resDecline string
		wantAny             []string
	}{
		{"cuda", "arch is not eligible for the resident decode runner",
			[]string{"arch is not eligible for the resident decode runner", "cuda has no staged decode path"}},
		{"metal", "arch is not eligible for the resident decode runner",
			[]string{"arch is not eligible for the resident decode runner", "metal has no staged decode path"}},
		{"cuda", "", []string{"cuda has no staged decode path"}},
	} {
		t.Run(tc.backend, func(t *testing.T) {
			got := declinedToCPUReason(tc.backend, tc.resDecline)
			for _, want := range tc.wantAny {
				if !strings.Contains(got, want) {
					t.Errorf("declinedToCPUReason(%q, %q) = %q, want it to contain %q", tc.backend, tc.resDecline, got, want)
				}
			}
		})
	}
}

// stagedDeviceNote is webgpu-only now (DecodePath routes cuda/metal through
// declinedToCPUReason/BackendSummary instead, above) — a pure function of the quant string,
// needing no live device to test.
func TestStagedDeviceNote(t *testing.T) {
	for _, tc := range []struct {
		quant   string
		wantAny []string // note must contain at least one of these (nil = must be "")
	}{
		{"int4", []string{"no GPU dispatch", "resident runner"}},
		{"int4mix", []string{"FFN tensors (int4)", "attention (int8) reaches webgpu"}},
		{"int8", nil},
		{"int8int8", nil},
		{"", nil}, // native f32: webgpu's be.MatmulBT is called unconditionally
	} {
		t.Run(tc.quant, func(t *testing.T) {
			got := stagedDeviceNote(tc.quant)
			if tc.wantAny == nil {
				if got != "" {
					t.Errorf("stagedDeviceNote(%q) = %q, want empty — this quant DOES reach webgpu (MatmulW8A8/MatmulBT)", tc.quant, got)
				}
				return
			}
			found := false
			for _, want := range tc.wantAny {
				if strings.Contains(got, want) {
					found = true
				}
			}
			if !found {
				t.Errorf("stagedDeviceNote(%q) = %q, want a note containing one of %v", tc.quant, got, tc.wantAny)
			}
		})
	}
}
