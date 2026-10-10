package serveapp

import "testing"

// Which towers load int8, for every vision family serve dispatches on (loadVisionTower: "" and "gemma3" reach the SigLIP path) on every backend, unset and with
// -vision-quant f32 or int8. Only Gemma 3's SigLIP tower gets int8 unasked, and only under webgpu, whose device tower needs it; on CUDA it is float32 (owner decision).
// Every other family stays float32 unless asked: its gates ran float32, and int8 measured lossy and not faster (see towerInt8). The table must name every family: a family
// missing from the float32 list silently takes the int8 default (qwen3_vl was the case).
func TestTowerInt8(t *testing.T) {
	backends := []string{"cpu", "auto", "metal", "webgpu", "cuda"}
	unasked := map[string]map[string]bool{ // family -> backend -> int8 without -vision-quant
		"":            {"webgpu": true, "cuda": true},
		"gemma3":      {"webgpu": true}, // cuda: float32 since 2026-10-08
		"gemma4":      {},
		"qwen2_5_vl":  {},
		"qwen3_5":     {},
		"qwen3_5_moe": {},
		"qwen3_vl":    {},
		"glm_ocr":     {},
	}
	for fam, want := range unasked {
		for _, be := range backends {
			if got := towerInt8(fam, "", be); got != want[be] {
				t.Errorf("family %q on %s, no -vision-quant: int8 = %v, want %v", fam, be, got, want[be])
			}
			if towerInt8(fam, "f32", be) { // explicit float32 (f37fe277): the way to ask for Gemma 3's float32 device tower
				t.Errorf("family %q on %s: -vision-quant f32 still loaded int8", fam, be)
			}
			if !towerInt8(fam, "int8", be) {
				t.Errorf("family %q on %s: -vision-quant int8 was ignored", fam, be)
			}
		}
	}
}
