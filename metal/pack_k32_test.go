//go:build darwin

package metal

import (
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestInt4Pack_declinesNonMultipleOf32K is the gate for audit M-10: the W4A8 pack layout and its
// GEMV kernels hard-assume K (the contraction width) is a multiple of the group, 32 — a row is K/8
// packed words + K/32 group scales with no partial-group handling. A K%32 != 0 weight would pack a
// truncated last group (trailing nibbles decode as −8) with a per-row stride the kernel disagrees
// with (silently wrong), or panic outright at K<32. The pack entry points must reject it so
// BuildResident declines to the CPU path instead of building a corrupt resident.
func TestInt4Pack_declinesNonMultipleOf32K(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	t.Cleanup(func() { d.ReleaseObjects(); d.ReleaseAll() })

	const N = 4
	bad := linalg.QuantizeInt8(make([]float32, N*40), N, 40, true)  // K=40 → 40%32 = 8
	good := linalg.QuantizeInt8(make([]float32, N*64), N, 64, true) // K=64 → ok

	// int4Buf returns an error on the bad width (→ BuildResident CPU fallback) and accepts a valid one.
	if _, _, err := int4Buf(d, &bad); err == nil {
		t.Fatal("int4Buf accepted K=40 (K%32 != 0) — the W4A8 kernels hard-assume K%32==0 (M-10)")
	} else if !strings.Contains(err.Error(), "K=40") {
		t.Errorf("int4Buf decline reason %q should name the offending K", err)
	}
	if _, _, err := int4Buf(d, &good); err != nil {
		t.Fatalf("int4Buf wrongly declined a valid K=64: %v", err)
	}

	// int4Concat has no error return; it panics on the bad width, which buildResident's recover turns
	// into a clean decline. The panic itself is the gate.
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("int4Concat accepted K=40 — it must panic so BuildResident declines (M-10)")
			} else if !strings.Contains(strings.ToLower(err2str(r)), "k=40") {
				t.Errorf("int4Concat panic %v should name the offending K", r)
			}
		}()
		int4Concat(d, &bad)
	}()
}

func err2str(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	if e, ok := v.(error); ok {
		return e.Error()
	}
	return ""
}
