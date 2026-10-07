package serveapp

import (
	"errors"
	"strings"
	"testing"
)

type fakeResidentEnc struct {
	err   error
	calls int
}

func (f *fakeResidentEnc) EnableResident() error { f.calls++; return f.err }

// A resident GPU vision tower that cannot be attached (no VRAM left for it, a build without the backend) used to abort serve
// startup: loadVisionTower returned the EnableResident error and the model already loaded on the GPU was thrown away. The attach
// is now a warning, and the tower runs on the CPU path EnableResident leaves intact.
func TestEnableResidentTower(t *testing.T) {
	for _, backend := range []string{"cuda", "webgpu", "metal"} { // metal since S3 (aikit's visionmetal / qwenmetal)
		t.Run(backend+"/fails", func(t *testing.T) {
			var warn strings.Builder
			enc := &fakeResidentEnc{err: errors.New("out of memory")}
			if enableResidentTower(enc, backend, &warn) {
				t.Error("a failed attach was reported as resident: the banner would claim a GPU tower that is not there")
			}
			for _, want := range []string{"warning", "could not be enabled", "out of memory", "CPU tower"} {
				if !strings.Contains(warn.String(), want) {
					t.Errorf("the warning lacks %q: %q", want, warn.String())
				}
			}
		})
		t.Run(backend+"/succeeds", func(t *testing.T) {
			var warn strings.Builder
			enc := &fakeResidentEnc{}
			if !enableResidentTower(enc, backend, &warn) || warn.Len() != 0 {
				t.Errorf("a successful attach: resident reported false or warned: %q", warn.String())
			}
		})
	}
	// Any other backend never asks for a resident tower (and so never warns about not getting one).
	for _, backend := range []string{"cpu", "auto", ""} {
		var warn strings.Builder
		enc := &fakeResidentEnc{err: errors.New("must not be called")}
		if enableResidentTower(enc, backend, &warn) || enc.calls != 0 || warn.Len() != 0 {
			t.Errorf("backend %q: resident=true, or EnableResident called %d times, or warned %q", backend, enc.calls, warn.String())
		}
	}
}
