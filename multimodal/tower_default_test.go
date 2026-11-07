package multimodal

import "testing"

// TestDefaultTowerOnDevice_isTheDeclaration: a backend runs every tower family it declares on the device when the user asks for
// nothing. WebGPU's towers were opt-in until their speed gates read (G-S12.3a-4, d-4, e-4, f-4: medians 0.355 to 0.488 of the CPU
// towers, 2026-10-12) and Francis applied the registered consequence (S12-defaults); -vision-device cpu is the way back, in serve.
func TestDefaultTowerOnDevice_isTheDeclaration(t *testing.T) {
	for backend, families := range declaredTowers {
		for _, f := range families {
			if !DefaultTowerOnDevice(backend, f) {
				t.Errorf("%s declares a %s tower but does not run it on the device by default", backend, f)
			}
		}
	}
	for _, f := range []string{TowerPixtral, TowerSiglip2NaFlex} { // CPU only: no backend declares them
		for _, backend := range []string{"cuda", "metal", "webgpu", "cpu"} {
			if DefaultTowerOnDevice(backend, f) {
				t.Errorf("%s: the CPU-only %s tower runs on the device by default", backend, f)
			}
		}
	}
	if DefaultTowerOnDevice("webgpu", TowerGemma4Audio) {
		t.Error("webgpu has no audio tower")
	}
}
