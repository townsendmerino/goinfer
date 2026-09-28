package decoder

import "testing"

// TE6(b): "identity-inherited (<method> @ <rev>)" is a T3 method exactly when its inner method is, and a known merge
// method on the same terms. A malformed string, or one wrapping a sub-T3 method, is neither.
func TestIdentityInheritedMethod(t *testing.T) {
	for _, c := range []struct {
		m          string
		t3, known  bool
		inner, rev string
	}{
		{"identity-inherited (full-forward-oracle @ 310a2e44)", true, true, "full-forward-oracle", "310a2e44"},
		{"identity-inherited (real-model-oracle @ abc1234)", true, true, "real-model-oracle", "abc1234"},
		{"identity-inherited (shared-path (via deepseek_v3) @ 0b74e671)", true, true, "shared-path (via deepseek_v3)", "0b74e671"},
		{"identity-inherited (tiny-golden @ 310a2e44)", false, true, "tiny-golden", "310a2e44"},
		{"identity-inherited (full-forward-oracle)", false, false, "", ""},
		{"identity-inherited (bogus-method @ 1)", false, false, "bogus-method", "1"},
		{"full-forward-oracle", true, true, "", ""},
	} {
		if got := isT3Method(c.m); got != c.t3 {
			t.Errorf("isT3Method(%q) = %v, want %v", c.m, got, c.t3)
		}
		if got := knownParityMethod(c.m); got != c.known {
			t.Errorf("knownParityMethod(%q) = %v, want %v", c.m, got, c.known)
		}
		if inner, rev, ok := identityInherited(c.m); ok && (inner != c.inner || rev != c.rev) {
			t.Errorf("identityInherited(%q) = %q, %q", c.m, inner, rev)
		}
	}
}
