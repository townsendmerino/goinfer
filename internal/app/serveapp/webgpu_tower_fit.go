package serveapp

import (
	"fmt"
	"runtime"

	"github.com/townsendmerino/goinfer/decoder"
)

// A WebGPU vision tower placed on the device (docs/tasks/task-multimodal-remaining-2026-10.md, S12-defaults) puts a float32 tower beside the decoder in the same device memory. WebGPU has no free-device-memory query
// (backend/webgpu/kv_slots.go), so the one place a figure exists is a unified-memory machine (darwin), where the device's buffers are
// host RAM.
//
// There is ONE budget, and the backend applies it: towerReserve prices the tower into Options.ExtraResidentBytes before the decoder
// loads, and backend/webgpu counts it with the weights against the fit guard's share of RAM (and the memory available when the build
// started) when it sizes the context and the KV slots, so the plan leaves the tower its room. webgpuTowerFits is the backstop for what
// the plan could not know: once the decoder is resident it asks the physical question, whether the memory the tower needs is still
// available, and does not discount it again. When it is not, the load is REFUSED, not moved to the CPU: on unified memory a float32 CPU
// tower holds the same RAM, so a move would save nothing and only hide the pressure. On a discrete GPU there is no figure to compare,
// so the existing rules stay: an attach that fails warns and runs the CPU tower (or refuses under -require-backend), and a request that
// needs more device memory than the decoder left falls back or fails with a hint.

// webgpuTowerUnified says whether WebGPU's device memory is the host's RAM on this machine; a variable so a test can stand in for one.
var webgpuTowerUnified = func() bool { return runtime.GOOS == "darwin" }

// webgpuTowerAvailable is the host memory available now; a variable so a test can stand in for a machine.
var webgpuTowerAvailable = decoder.HostRAMAvailableBytes

// webgpuTowerFits refuses a load whose WebGPU vision tower, priced by towerReserve, needs more than the memory available now on a
// unified-memory machine, and names what to do instead. It proceeds off WebGPU, on a discrete GPU, with -vision-device cpu (the
// user's own choice of the CPU tower, which needs the same RAM), when no WebGPU tower is registered for the family, and when the
// available memory cannot be read: every unknown proceeds, as the fit guard does.
func webgpuTowerFits(cfg config, dir string) error {
	if cfg.towerBackend() != "webgpu" || !webgpuTowerUnified() {
		return nil
	}
	need := towerReserve(cfg, dir)
	if need <= 0 {
		return nil
	}
	avail := webgpuTowerAvailable()
	if avail <= 0 || need <= avail {
		return nil
	}
	const gib = 1 << 30
	// -vision-quant int8 is offered only where the int8 form is a gated device tower: Gemma 3's. For every other family it would move the
	// tower to the CPU's int8 path, which is lossy, ungated for GLM-OCR, and priced at zero.
	int8 := ""
	if visionModelType(dir) == "gemma3" && cfg.visionQuant != "int8" {
		int8 = ", use -vision-quant int8 (a smaller device tower, lossy against float32)"
	}
	return fmt.Errorf("the WebGPU vision tower needs about %.1f GiB beside the decoder and only %.1f GiB is available; "+
		"the CPU tower would hold the same RAM, so it is not a fallback. Free memory, lower -ctx or -kv-sessions%s, "+
		"or pass -vision-device cpu to run the CPU float32 tower anyway",
		float64(need)/gib, float64(avail)/gib, int8)
}
