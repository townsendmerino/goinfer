//go:build cuda

package cuda

import "testing"

// TestLayerFusable_M23 gates the fuseQKV admission guard. Two failure modes it must prevent:
// a dense int4-QKV + int8-gate/up checkpoint reaching fGU with a nil ws16 (crash), and a MoE
// model losing its fQKV attention super-kernel because its unpacked g/u/d look non-int4.
// Pure logic — no device needed.
func TestLayerFusable_M23(t *testing.T) {
	for _, c := range []struct {
		name              string
		qkvInt4, moe, gu4 bool
		want              bool
	}{
		{"dense all int4", true, false, true, true},
		{"dense int8 gate/up → no fuse (crash guard)", true, false, false, false},
		{"dense int8 qkv → no fuse", false, false, true, false},
		{"MoE g/u/d unpacked → still fuses (regression fix)", true, true, false, true},
		{"MoE all int4", true, true, true, true},
		{"MoE int8 qkv → no fuse", false, true, true, false},
	} {
		if got := layerFusable(c.qkvInt4, c.moe, c.gu4); got != c.want {
			t.Errorf("%s: layerFusable(%v,%v,%v) = %v, want %v", c.name, c.qkvInt4, c.moe, c.gu4, got, c.want)
		}
	}
}
