package serveapp

import "testing"

// Which towers load int8: the resident-capable Gemma 3 tower under a GPU backend (it needs int8), and any tower when asked for.
// The CPU-only towers never get int8 as a side effect of the backend (measured lossy and not faster: see towerInt8).
func TestTowerInt8(t *testing.T) {
	cpuOnly := []string{"qwen2_5_vl", "qwen3_5", "qwen3_5_moe", "gemma4", "glm_ocr"}
	for _, mt := range cpuOnly {
		for _, be := range []string{"cpu", "auto", "metal", "webgpu", "cuda"} {
			if towerInt8(mt, "", be) {
				t.Errorf("%s on %s: int8 without being asked for", mt, be)
			}
			if !towerInt8(mt, "int8", be) {
				t.Errorf("%s on %s: -vision-quant int8 was ignored", mt, be)
			}
		}
	}
	for _, mt := range []string{"gemma3", ""} {
		for be, want := range map[string]bool{"webgpu": true, "cuda": true, "cpu": false, "auto": false, "metal": false} {
			if got := towerInt8(mt, "", be); got != want {
				t.Errorf("gemma3 tower (%q) on %s: int8 = %v, want %v", mt, be, got, want)
			}
		}
		if !towerInt8(mt, "int8", "cpu") {
			t.Errorf("gemma3 tower: -vision-quant int8 was ignored")
		}
	}
}
