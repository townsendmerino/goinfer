package serveapp

import (
	"fmt"
	"io"
	"runtime"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// WebGPU's vision towers are the default under --backend webgpu (docs/tasks/task-multimodal-remaining-2026-10.md, S12-defaults), so
// each default image load puts a float32 tower beside the decoder in the same device memory. WebGPU has no free-device-memory query
// (backend/webgpu/kv_slots.go), so the one place a figure exists is a unified-memory machine (darwin), where the device's buffers are
// host RAM and the fit guard's own budget applies: towerReserve prices the tower, and webgpuTowerFits refuses to place it on the
// device when it exceeds that budget, with the CPU tower serving instead (or a refusal under -require-backend). On a discrete GPU
// there is no figure to compare, so the existing rules stay: an attach that fails warns and runs the CPU tower (or refuses under
// -require-backend), and a request that needs more device memory than the decoder left fails with a hint.

// webgpuTowerUnified says whether WebGPU's device memory is the host's RAM on this machine; a variable so a test can stand in for one.
var webgpuTowerUnified = func() bool { return runtime.GOOS == "darwin" }

// webgpuTowerAvailable is the host memory available now; a variable so a test can stand in for a machine.
var webgpuTowerAvailable = decoder.HostRAMAvailableBytes

// webgpuTowerFits returns cfg with the tower kept on the CPU (-vision-device cpu for this load) when the WebGPU tower, priced by
// towerReserve, does not fit the fit guard's budget of the memory available now, and says why on note. Under -require-backend that
// is an error instead. It changes nothing off WebGPU, on a discrete GPU, when no WebGPU tower is registered for the family, or when
// the available memory cannot be read: every unknown proceeds, as the fit guard does.
func webgpuTowerFits(cfg config, dir string, note io.Writer) (config, error) {
	if cfg.towerBackend() != "webgpu" || !webgpuTowerUnified() {
		return cfg, nil
	}
	need := towerReserve(cfg, dir)
	if need <= 0 {
		return cfg, nil
	}
	avail := webgpuTowerAvailable()
	if avail <= 0 {
		return cfg, nil
	}
	budget := mem.HostBudget(mem.HostMachine(avail), mem.DefaultPolicy())
	if need <= budget {
		return cfg, nil
	}
	const gb = 1 << 30
	why := fmt.Sprintf("the WebGPU vision tower needs about %.1f GB beside the decoder and the fit guard's budget of the %.1f GB available is %.1f GB",
		float64(need)/gb, float64(avail)/gb, float64(budget)/gb)
	if cfg.requireBE {
		return cfg, fmt.Errorf("-require-backend: %s; free memory, lower -ctx or -kv-sessions, or use -vision-device cpu", why)
	}
	fmt.Fprintf(note, "note: %s; the tower runs on the CPU (-vision-device cpu asks for that on purpose)\n", why)
	cfg.visionDevice = "cpu"
	return cfg, nil
}
