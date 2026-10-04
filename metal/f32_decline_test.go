//go:build darwin

package metal

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestBuildResident_f32DeclinesClearly (int8 slice 4): an f32 load on -backend metal declines with the reason and the
// quant to use, not int4Concat's panic text, and the model runs on the CPU at the f32 it asked for.
func TestBuildResident_f32DeclinesClearly(t *testing.T) {
	m, err := decoder.Load("../testdata/llama-attnfa-tiny", decoder.Options{Backend: "metal", ResidentContext: 512})
	if err != nil {
		t.Skipf("load: %v", err)
	}
	defer m.Close()
	why := m.ResidentDecline()
	if !strings.Contains(why, "weights loaded at f32") || !strings.Contains(why, "-quant int4 or int8int8") || strings.Contains(why, "panicked") {
		t.Fatalf("decline %q: want the f32 reason and the quant to use", why)
	}
	if dp := m.DecodePath(); !strings.HasPrefix(dp, "cpu") {
		t.Fatalf("DecodePath %q: want the CPU", dp)
	}
}
