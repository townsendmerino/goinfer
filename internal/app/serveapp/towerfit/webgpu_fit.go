package towerfit

import "fmt"

// WebGPUTowerFits refuses a load whose WebGPU vision tower needs more host RAM
// than is currently available on a unified-memory machine.
func WebGPUTowerFits(cfg Config, dir string) error {
	if cfg.TowerBackend() != "webgpu" || !WebGPUTowerUnified() {
		return nil
	}
	need := Reserve(cfg, dir)
	if need <= 0 {
		return nil
	}
	avail := WebGPUTowerAvailable()
	if avail <= 0 || need <= avail {
		return nil
	}
	const gib = 1 << 30
	int8 := ""
	if VisionModelType(dir) == "gemma3" && cfg.VisionQuant != "int8" {
		int8 = ", use -vision-quant int8 (a smaller device tower, lossy against float32)"
	}
	return fmt.Errorf("the WebGPU vision tower needs about %.1f GiB beside the decoder and only %.1f GiB is available; "+
		"the CPU tower would hold the same RAM, so it is not a fallback. Free memory, lower -ctx or -kv-sessions%s, "+
		"or pass -vision-device cpu to run the CPU float32 tower anyway",
		float64(need)/gib, float64(avail)/gib, int8)
}
