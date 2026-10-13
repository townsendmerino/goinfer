//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// The suite runs on the exact f32 cache unless a test asks otherwise: see decoder.SetCUDAKVF16DefaultForTest. This runs before TestMain.
func init() { decoder.SetCUDAKVF16DefaultForTest(false) }

// TestKV16_theDefaultIsF16OnAClearedFamily loads with no -kv, and with "auto", the way serve's default and a library caller's zero Options do, with the default
// ON as shipped: a cleared family (qwen2) must come up on the f16 cache and a family the f16 kernels do not cover must stay f32, with "f32" always winning.
func TestKV16_theDefaultIsF16OnAClearedFamily(t *testing.T) {
	requireCUDADevice(t)
	t.Cleanup(decoder.SetCUDAKVF16DefaultForTest(true))
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
	// A family the f16 kernels do not cover stays f32 under the default, and an explicit f16 for it is held at f32 with the reason.
	if g := modelPath("gemma3-1b-q4_k_m.gguf"); fileExists(g) {
		for _, kv := range []string{"", "auto", "f16"} {
			m, err := decoder.Load(g, decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: kv, ResidentContext: 2048})
			if err != nil {
				t.Fatalf("gemma3 -kv %q: %v", kv, err)
			}
			if r, ok := m.ResidentForwardForTest().(*cudaResident); !ok || r.kvF16 || m.ResidentKVPrecision() != "f32" {
				t.Errorf("gemma3 -kv %q: not an f32 resident (ok=%v)", kv, ok)
			}
			if kv == "f16" && m.KVF16DeclineReason() == "" {
				t.Error("gemma3 -kv f16 was held at f32 with no reason")
			}
			m.Close()
		}
	}
}

func fileExists(p string) bool { _, err := os.Stat(p); return err == nil }
