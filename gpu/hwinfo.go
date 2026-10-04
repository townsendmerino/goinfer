//go:build gpu

package gpu

import (
	"fmt"

	"github.com/townsendmerino/goinfer/decoder"
)

func init() { decoder.RegisterHardwareInfo("webgpu", webgpuHardwareInfo) }

// webgpuHardwareInfo is the adapter facts `check --hardware` prints for the WebGPU backend (docs/tasks/task-hardware-coverage-2026-10.md, H3). It opens an adapter only when asked, and says so
// plainly when there is none. WebGPU does not expose VRAM size or the driver version, so the report says it cannot report them rather than leaving the gap unexplained. The adapter's name is the
// GPU model, not a machine identifier.
func webgpuHardwareInfo() []string {
	c, err := New()
	if err != nil {
		return []string{"no usable adapter: " + err.Error()}
	}
	defer c.Close()
	info := c.adapter.GetInfo()
	name := info.Device
	if name == "" {
		name = info.Description
	}
	out := []string{"adapter: " + name}
	if info.Vendor != "" || info.Architecture != "" {
		out = append(out, fmt.Sprintf("vendor: %s, architecture: %s", orUnknown(info.Vendor), orUnknown(info.Architecture)))
	}
	out = append(out, fmt.Sprintf("graphics API: %s; adapter type: %s", c.Backend(), orUnknown(info.AdapterType.String())))
	if softwareAdapterInfo(info) {
		out = append(out, "SOFTWARE RENDERER: this adapter is a CPU implementation of the GPU API (lavapipe, llvmpipe, SwiftShader): slow, and its limits are lower than a real GPU's")
	}
	lim := c.adapter.GetLimits()
	out = append(out, fmt.Sprintf("limits: max buffer %.1f GiB, max storage binding %.1f GiB, workgroup memory %d KiB, %d invocations per workgroup",
		float64(lim.MaxBufferSize)/(1<<30), float64(lim.MaxStorageBufferBindingSize)/(1<<30), lim.MaxComputeWorkgroupStorageSize>>10, lim.MaxComputeInvocationsPerWorkgroup))
	out = append(out, fmt.Sprintf("integer dot-product (dot4I8Packed): %s", yesNo(c.hasDP4A)))
	out = append(out, "memory and driver version: not reported by WebGPU")
	return out
}

func orUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}

func yesNo(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

// SelfTestEligible is decoder's selfTestEligibility: the resident self-test's bars were measured on a real adapter, so a software renderer (lavapipe on a CI runner, a headless box falling back to
// llvmpipe) is not probed. It is reported as skipped, with this reason, in the report and /health, never as passed.
func (b *webgpuBackend) SelfTestEligible() (bool, string) {
	if softwareAdapterInfo(b.ctx.adapter.GetInfo()) {
		return false, "software adapter (lavapipe, llvmpipe, SwiftShader): its limits and math are not a GPU's, and the probe's bars were measured on a real one"
	}
	return true, ""
}
