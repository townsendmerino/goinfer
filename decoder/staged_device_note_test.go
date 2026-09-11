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

// TestResidentQuantLabel pins G10's reporting fix (docs/task-gpu-paths-2026-09.md): Metal has no
// int8 GEMV kernel, so an int8int8 resident load must not claim that precision back to the user —
// every other (backend, quant) pair, including int8int8 on cuda/webgpu (which DO have a real W8A8
// kernel) and every non-int8int8 quant on metal, must be a pure passthrough of the requested quant
// string.
func TestResidentQuantLabel(t *testing.T) {
	for _, tc := range []struct{ backend, quant, want string }{
		{"metal", "int8int8", "int8int8→int4, no Metal int8 GEMV kernel"},
		{"metal", "int4", "int4"},
		{"metal", "int8", "int8"},
		{"metal", "f32", "f32"},
		{"cuda", "int8int8", "int8int8"},
		{"webgpu", "int8int8", "int8int8"},
		{"cpu", "int8int8", "int8int8"},
	} {
		if got := residentQuantLabel(tc.backend, tc.quant); got != tc.want {
			t.Errorf("residentQuantLabel(%q, %q) = %q, want %q", tc.backend, tc.quant, got, tc.want)
		}
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
		{"int4", []string{"decode (one token) runs on webgpu"}},
		{"int4mix", []string{"decode (one token) runs on webgpu"}},
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
