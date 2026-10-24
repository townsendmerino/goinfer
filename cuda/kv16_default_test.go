//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// The suite runs on the exact f32 cache unless a test asks otherwise (kvpolicy.go:kvF16Auto). init runs before TestMain, and after kvpolicy.go's own.
func init() { kvF16Auto.Store(false) }

// TestKV16_theDefaultIsF16OnAClearedFamily loads with no -kv, and with "auto", the way serve's default and a library caller's zero Options do, with the default
// ON as shipped: a cleared family (qwen2, and gemma3 of a cleared KV4b group) must come up on the f16 cache, with "f32" always winning.
func TestKV16_theDefaultIsF16OnAClearedFamily(t *testing.T) {
	requireCUDADevice(t)
	kvF16Auto.Store(true)
	t.Cleanup(func() { kvF16Auto.Store(false) })
	path := modelPath(kv16Model)
	for _, tc := range []struct {
		kv   string
		want bool
	}{{"", true}, {"auto", true}, {"f16", true}, {"f32", false}} {
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: tc.kv, ResidentContext: 2048})
		if err != nil {
			t.Fatalf("-kv %q: %v", tc.kv, err)
		}
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			m.Close()
			t.Fatalf("-kv %q: no resident: %s", tc.kv, m.ResidentDecline())
		}
		if r.kvF16 != tc.want || r.layers[0].kvHalf != tc.want || m.ResidentKVPrecision() != map[bool]string{true: "f16", false: "f32"}[tc.want] {
			t.Errorf("-kv %q: kvF16=%v layer0 kvHalf=%v ResidentKVPrecision=%q, want f16=%v", tc.kv, r.kvF16, r.layers[0].kvHalf, m.ResidentKVPrecision(), tc.want)
		}
		m.Close()
	}
	// A family whose group's quality gate has cleared (gemma3, KV4b's sliding-window group) comes up on f16 under the default, and "f32" still
	// wins. Which delivered families auto keeps at f32 is decoder's TestCUDAKVF16AutoDecline_table; a family not delivered at all refuses an
	// explicit f16 by name (TestKVPrecisionDelivered).
	if g := modelPath("gemma3-1b-q4_k_m.gguf"); fileExists(g) {
		for _, tc := range []struct {
			kv   string
			want bool
		}{{"", true}, {"auto", true}, {"f16", true}, {"f32", false}} {
			m, err := decoder.Load(g, decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: tc.kv, ResidentContext: 2048})
			if err != nil {
				t.Fatalf("gemma3 -kv %q: %v", tc.kv, err)
			}
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok || r.kvF16 != tc.want || m.ResidentKVPrecision() != map[bool]string{true: "f16", false: "f32"}[tc.want] {
				t.Errorf("gemma3 -kv %q: resident %v, kvF16 %v, want f16=%v", tc.kv, ok, ok && r.kvF16, tc.want)
			}
			m.Close()
		}
	}
}

func fileExists(p string) bool { _, err := os.Stat(p); return err == nil }
