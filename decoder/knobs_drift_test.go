//go:build goinfer_testhooks

package decoder

import (
	"strings"
	"testing"
)

// The tripwire must fire on exactly the defect it exists for — t.Setenv after Load — and stay quiet for
// the two sanctioned routes (Options.Knobs, SetKnobForTest). A tripwire nobody has seen go red proves nothing.
func TestKnobDrift_firesOnPostLoadSetenv(t *testing.T) {
	t.Setenv(knobNoOptFwd, "")
	m, err := Load(tinyFixture(t), Options{Backend: "cpu", Knobs: map[string]string{knobFusedAttention: "0"}})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	t.Setenv(knobFusedAttention, "1") // pinned by Options.Knobs: exempt
	if m.knobs.fusedAttention() {
		t.Error("an Options.Knobs value must not follow a later environment change")
	}

	t.Setenv(knobNoOptFwd, "1") // the defect
	func() {
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("t.Setenv after Load did not trip the drift check — an A/B would compare a path with itself")
			}
			if msg, _ := r.(string); !strings.Contains(msg, "SetKnobForTest") || !strings.Contains(msg, knobNoOptFwd) {
				t.Errorf("panic should name the knob and the fix: %v", r)
			}
		}()
		m.knobs.get(knobNoOptFwd)
	}()

	restore := SetKnobForTest(m, knobNoOptFwd, "1") // the sanctioned route: quiet, and followed
	if m.knobs.get(knobNoOptFwd) != "1" {
		t.Error("SetKnobForTest value not visible")
	}
	restore()
}
