package serveapp

import (
	"fmt"
	"runtime"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// WebGPU's vision towers are the default under --backend webgpu (docs/tasks/task-multimodal-remaining-2026-10.md, S12-defaults), so
// each default image load puts a float32 tower beside the decoder in the same device memory. WebGPU has no free-device-memory query
// (backend/webgpu/kv_slots.go), so the one place a figure exists is a unified-memory machine (darwin), where the device's buffers are
// host RAM and the fit guard's own budget applies. towerReserve prices the tower into Options.ExtraResidentBytes before the decoder
// loads, so the KV slots and the context are sized beside it (backend/webgpu counts it with the weights); webgpuTowerFits then checks,
// once the decoder is resident and the available memory is real, that the tower fits the budget. When it does not, the load is
// REFUSED, not moved to the CPU: on unified memory a float32 CPU tower holds the same RAM, so the move would save nothing and only
// hide the pressure. The way out is named in the error. On a discrete GPU there is no figure to compare, so the existing rules stay: an
// attach that fails warns and runs the CPU tower (or refuses under -require-backend), and a request that needs more device memory than
// the decoder left fails with a hint.

// webgpuTowerUnified says whether WebGPU's device memory is the host's RAM on this machine; a variable so a test can stand in for one.
var webgpuTowerUnified = func() bool { return runtime.GOOS == "darwin" }

// webgpuTowerAvailable is the host memory available now; a variable so a test can stand in for a machine.
var webgpuTowerAvailable = decoder.HostRAMAvailableBytes

// webgpuTowerFits refuses a load whose WebGPU vision tower, priced by towerReserve, exceeds the fit guard's budget of the memory
// available now on a unified-memory machine, and names what to do instead. It proceeds off WebGPU, on a discrete GPU, with
// -vision-device cpu (the user's own choice of the CPU tower, which needs the same RAM), when no WebGPU tower is registered for the
// family, and when the available memory cannot be read: every unknown proceeds, as the fit guard does.
func webgpuTowerFits(cfg config, dir string) error {
	if cfg.towerBackend() != "webgpu" || !webgpuTowerUnified() {
		return nil
	}
	need := towerReserve(cfg, dir)
	if need <= 0 {
		return nil
	}
	avail := webgpuTowerAvailable()
	if avail <= 0 {
		return nil
	}
	budget := mem.HostBudget(mem.HostMachine(avail), mem.DefaultPolicy())
	if need <= budget {
		return nil
	}
	const gb = 1 << 30
	return fmt.Errorf("the WebGPU vision tower needs about %.1f GB beside the decoder, and the fit guard's budget of the %.1f GB available is %.1f GB; "+
		"the CPU tower would hold the same RAM, so it is not a fallback. Free memory, lower -ctx or -kv-sessions, use -vision-quant int8 (a smaller, lossy tower), "+
		"or pass -vision-device cpu to run the CPU float32 tower anyway",
		float64(need)/gb, float64(avail)/gb, float64(budget)/gb)
}
