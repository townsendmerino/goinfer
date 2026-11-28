package decoder

import (
	"testing"
	"time"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
)

// The profile has to be safe to use unconditionally: paths that do not instrument (safetensors,
// .giw) hand back nil, and the banner calls Summary() on whatever it gets.
func TestLoadProfile_nilIsUsable(t *testing.T) {
	var p *darch.LoadProfile
	p.Record("x", time.Second) // must not panic
	p.SetBytes(1 << 30)
	if got := p.Phases(); got != nil {
		t.Errorf("nil profile returned %v phases", got)
	}
	if got := p.Total(); got != 0 {
		t.Errorf("nil profile total = %v", got)
	}
	if got := p.Summary(); got != "" {
		t.Errorf("nil profile summary = %q, want empty so an uninstrumented load prints nothing", got)
	}
	if _, _, share := p.Dominant(); share != 0 {
		t.Errorf("nil profile share = %v", share)
	}
	var m *Model
	if m.LoadProfile() != nil {
		t.Error("nil model returned a profile")
	}
}
