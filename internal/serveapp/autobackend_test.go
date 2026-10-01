package serveapp

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// TestRequireAutoBackend (R17): under -require-backend, a -backend auto that passed over a GPU backend this binary has
// refuses to start rather than run on the CPU, and names why and how to choose. Auto on a binary with no GPU backend,
// auto that found its GPU, and any named backend are left to the per-model checks (requireFastPaths).
func TestRequireAutoBackend(t *testing.T) {
	passedOver := &decoder.AutoChoice{Backend: "cpu", Reason: "auto: cuda is built in, but no CUDA device answered", Skipped: true}
	for _, tc := range []struct {
		name   string
		cfg    config
		refuse bool
	}{
		{"strict, auto passed over cuda", config{requireBE: true, load: loadflags.Flags{Backend: "cpu", Auto: passedOver}}, true},
		{"not strict", config{load: loadflags.Flags{Backend: "cpu", Auto: passedOver}}, false},
		{"strict, no GPU backend in the binary", config{requireBE: true, load: loadflags.Flags{Backend: "cpu",
			Auto: &decoder.AutoChoice{Backend: "cpu", Reason: "auto: this binary has no GPU backend"}}}, false},
		{"strict, auto found cuda", config{requireBE: true, load: loadflags.Flags{Backend: "cuda",
			Auto: &decoder.AutoChoice{Backend: "cuda", Reason: "auto: a CUDA device answered"}}}, false},
		{"strict, -backend cpu named", config{requireBE: true, load: loadflags.Flags{Backend: "cpu"}}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := requireAutoBackend(tc.cfg)
			if (err != nil) != tc.refuse {
				t.Fatalf("requireAutoBackend = %v, want refuse=%v", err, tc.refuse)
			}
			if err != nil && (!strings.Contains(err.Error(), passedOver.Reason) || !strings.Contains(err.Error(), "-backend cpu")) {
				t.Errorf("the refusal %q must name auto's reason and the -backend cpu way out", err)
			}
		})
	}
}
