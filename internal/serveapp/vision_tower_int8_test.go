package serveapp

import "testing"

// Which towers load int8, for every vision family serve dispatches on (loadVisionTower: "" and "gemma3" reach the SigLIP path)
// on every backend, with and without -vision-quant int8. Only Gemma 3's SigLIP tower gets int8 unasked, and only under cuda or
// webgpu, whose resident towers need it. Every other family stays float32 unless asked: its gates ran float32, and int8 measured
// lossy and not faster (see towerInt8). qwen3_vl is the case this table was written for: it was missing from the float32 list.
func TestTowerInt8(t *testing.T) {
	backends := []string{"cpu", "auto", "metal", "webgpu", "cuda"}
	unasked := map[string]map[string]bool{ // family -> backend -> int8 without -vision-quant
		"":            {"webgpu": true, "cuda": true},
		"gemma3":      {"webgpu": true, "cuda": true},
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
			if got := towerInt8(fam, "f32", be); got != want[be] {
				t.Errorf("family %q on %s, -vision-quant f32: int8 = %v, want %v", fam, be, got, want[be])
			}
			if !towerInt8(fam, "int8", be) {
				t.Errorf("family %q on %s: -vision-quant int8 was ignored", fam, be)
			}
		}
	}
}
