package decoder

import (
	"strings"
	"testing"
)

// R9 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): "decode path: cuda-staged (int4)"
// on an 8 GB card sat at the idle VRAM baseline (464 MiB, unchanged) for a full request sampled
// at 1 Hz. matmul() (weightmat.go) is why: its int4 branch calls the CPU integer W4A8 kernel
// unconditionally, with no backend parameter at all — decoder.QuantBackend declares only
// MatmulW8A8, no int4 counterpart exists anywhere in this tree. stagedDeviceNote is a pure
// function of the quant string precisely so it needs no live device or a built backend to test:
// which quants reach the backend in the staged path is a static fact of matmul()'s dispatch,
// not something that varies at runtime.
func TestStagedDeviceNote(t *testing.T) {
	for _, tc := range []struct {
		quant   string
		wantAny []string // note must contain at least one of these (nil = must be "")
	}{
		{"int4", []string{"no GPU dispatch", "CPU-equivalent"}},
		{"int4mix", []string{"FFN tensors (int4)", "no GPU dispatch"}},
		{"int8", nil},
		{"int8int8", nil},
		{"", nil}, // native f32: be.MatmulBT is called unconditionally
	} {
		t.Run(tc.quant, func(t *testing.T) {
			got := stagedDeviceNote(tc.quant)
			if tc.wantAny == nil {
				if got != "" {
					t.Errorf("stagedDeviceNote(%q) = %q, want empty — this quant DOES reach the backend (MatmulW8A8/MatmulBT)", tc.quant, got)
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
