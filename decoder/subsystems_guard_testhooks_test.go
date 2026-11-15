//go:build goinfer_testhooks

package decoder

import (
	"strings"
	"testing"
)

// TestSubsystemsOf_panicsOnASubsystem: under goinfer_testhooks SubsystemsOf refuses a value that serves a subsystem contract
// without being a provider (the silent no-capability answer that declined every recurrent block drafter), and still answers a
// resident, a plain resident and nil.
func TestSubsystemsOf_panicsOnASubsystem(t *testing.T) {
	res := &splitResident{}
	for _, c := range []struct {
		name string
		arg  any
	}{
		{"a runtime subsystem", res.ResidentSubsystems().Runtime},
		{"a decode subsystem", NoopResidentDecode{}},
		{"a prefill subsystem", NoopResidentPrefill{}},
		{"a kv subsystem", NoopResidentKV{}},
	} {
		func() {
			defer func() {
				msg, _ := recover().(string)
				if !strings.Contains(msg, "not a ResidentSubsystemProvider") {
					t.Errorf("%s: SubsystemsOf panic = %q, want the provider message", c.name, msg)
				}
			}()
			SubsystemsOf(c.arg)
		}()
	}
	for _, arg := range []any{res, &stepPlainResident{}, nil} {
		SubsystemsOf(arg) // must not panic
	}
	if !SubsystemsOf(res).Runtime.DrafterHostAvailable() {
		t.Error("SubsystemsOf(resident) did not return the resident's own subsystems")
	}
}
