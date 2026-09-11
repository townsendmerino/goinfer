//go:build goinfer_testhooks

package decoder

import "testing"

// TestTeacherForcedTop1Agreement_callerErrorIsNotAgreement: a firstDivergence of -1 means "every
// position agreed", so input that was never compared must not return it (audit-2026-09-10
// G-13(k)). Empty or length-mismatched slices returned 0, -1, and a caller gating only on
// firstDivergence == -1 passed on a comparison that never ran.
func TestTeacherForcedTop1Agreement_callerErrorIsNotAgreement(t *testing.T) {
	logits := [][]float32{{0, 1}, {1, 0}} // argmax 1, then 0
	for _, tc := range []struct {
		name string
		cand [][]float32
		ref  []int
	}{
		{"length mismatch", logits, []int{1}},
		{"no reference tokens", logits, nil},
		{"no candidate logits", nil, []int{1, 0}},
		{"both empty", nil, nil},
	} {
		rate, first := TeacherForcedTop1AgreementForTest(tc.cand, tc.ref)
		if first == -1 {
			t.Errorf("%s: firstDivergence = -1, the answer for \"no position disagreed\", on input that was never compared", tc.name)
		}
		if rate != 0 {
			t.Errorf("%s: agreementRate = %v, want 0", tc.name, rate)
		}
	}
	// The real answers are unchanged: full agreement is -1, a miss names its position.
	if rate, first := TeacherForcedTop1AgreementForTest(logits, []int{1, 0}); rate != 1 || first != -1 {
		t.Errorf("full agreement: got (%v, %d), want (1, -1)", rate, first)
	}
	if rate, first := TeacherForcedTop1AgreementForTest(logits, []int{1, 1}); rate != 0.5 || first != 1 {
		t.Errorf("miss at position 1: got (%v, %d), want (0.5, 1)", rate, first)
	}
}
