//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// M-16: the single-block attention kernels size their scratch (nWin+128)*4 with NO ceiling, so
// past 12,160 attended keys the launch exceeds the 48 KB default and is refused by the driver.
// Decode fails at that position; batched prefill errors at layer 0 and falls back to the ~9x
// slower sequential path with nothing logged. The trigger is -ctx 16384+ on any geometry whose
// perf table says splitkvNever (nH >= 24: Qwen2.5-7B, Llama-3-8B, phi3-mini) — the -ctx 32768
// rows in benchmarks.md were on 0.5B/1.5B, whose split-KV engages at 3072/1024.
//
// The audit rated this medium confidence because it hinged on whether anything raises the
// kernel into the opt-in range. Nothing does, and this pins BOTH halves against the device.
func TestAttnShmemLimit_matchesDevice(t *testing.T) {
	// The arithmetic half runs anywhere.
	if got := attnShmemBytes(12160); got > singleBlockAttnShmemLimit {
		t.Errorf("12160 keys needs %d B, over the %d B limit — the constant and the sizing "+
			"formula disagree about where the boundary is", got, singleBlockAttnShmemLimit)
	}
	if got := attnShmemBytes(12161); got <= singleBlockAttnShmemLimit {
		t.Errorf("12161 keys needs only %d B — the boundary moved", got)
	}
	if !splitKVRequired(12161) {
		t.Error("splitKVRequired is false one key past the limit: the single-block launch would " +
			"be attempted and refused by the driver (M-16)")
	}
	if splitKVRequired(12160) {
		t.Error("splitKVRequired is true AT the limit, which would force split-KV one key early")
	}

	// The device half. MEASURED, not assumed: the constant must not exceed what this GPU allows,
	// or the guard passes launches the driver will refuse.
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	const attrMaxShmemPerBlock = 8 // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
	lim, err := dev.Attribute(gc.DeviceAttribute(attrMaxShmemPerBlock))
	if err != nil {
		t.Skipf("attribute query: %v", err)
	}
	t.Logf("device MAX_SHARED_MEMORY_PER_BLOCK = %d B (%.0f KB); guard is %d B => nWin <= %d keys",
		lim, float64(lim)/1024, singleBlockAttnShmemLimit, singleBlockAttnShmemLimit/4-128)
	if singleBlockAttnShmemLimit > lim {
		t.Errorf("the guard allows %d B but this device permits only %d B: launches inside the "+
			"guard would still be refused (M-16)", singleBlockAttnShmemLimit, lim)
	}
}
