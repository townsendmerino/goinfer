//go:build cuda

package cuda

import (
	"errors"
	"testing"
)

// TestRecordUpload_capturesFirstError is the C-08 gate: BuildResident's load-time up* helpers must
// record a failed upload into setupErr (the setup job returns r.setupErr, which BuildResident turns
// into a decline). Before the fix they discarded gpu.Upload's error with `_ =`, so a failed upload left
// a zeroed buffer and the build returned ok=true — a resident that decodes garbage. Device-free: it
// exercises the recording contract directly (the seam the executor return-path and backend.go's
// `if setupErr != nil { … declined }` depend on), the same shape as the C-24 runJob gate.
func TestRecordUpload_capturesFirstError(t *testing.T) {
	r := &cudaResident{}

	r.recordUpload(nil) // a successful upload must not poison the build
	if r.setupErr != nil {
		t.Fatalf("recordUpload(nil) set setupErr = %v, want nil", r.setupErr)
	}

	first := errors.New("upload of layer 3 v_proj failed")
	r.recordUpload(first)
	r.recordUpload(errors.New("later upload also failed")) // first error wins; later noise ignored
	r.recordUpload(nil)                                    // a subsequent success does not clear it

	if r.setupErr == nil {
		t.Fatal("recordUpload(err) left setupErr nil — a failed upload would still yield ok=true (C-08)")
	}
	if !errors.Is(r.setupErr, first) {
		t.Fatalf("setupErr = %v, want the FIRST error %v (later errors must not overwrite it)", r.setupErr, first)
	}
}
