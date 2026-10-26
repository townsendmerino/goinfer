//go:build darwin

package visionmetal

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
	gpu "github.com/townsendmerino/goinfer/metal/internal/gpudev"
)

// TestAttnThreadgroupBytes covers the audit C-06 guard's arithmetic without a checkpoint —
// visionmetal has no per-image variability the way qwenmetal does (np is fixed at build time), so
// this is the direct twin of qwenmetal/threadgroup_budget_test.go's TestAttnThreadgroupBytes for
// the SAME shared kernel (gpu.ViT's `attention`).
func TestAttnThreadgroupBytes(t *testing.T) {
	const apple = 32 << 10 // ~32 KiB, what Apple GPUs report

	// The static pair is always present, even at zero patches.
	if got, want := attnThreadgroupBytes(0), 2*gpu.ViTBlock*4; got != want {
		t.Errorf("attnThreadgroupBytes(0) = %d, want %d (the two static arrays)", got, want)
	}
	// Dynamic term is 4 bytes per patch on top of that.
	if got, want := attnThreadgroupBytes(1000), 1000*4+2*gpu.ViTBlock*4; got != want {
		t.Errorf("attnThreadgroupBytes(1000) = %d, want %d", got, want)
	}

	// The boundary against a 32 KiB budget. With ViTBlock=256 the static pair is 2048 B, leaving
	// 30720 B => the largest admissible patch count is 7680, NOT the 8192 a dynamic-only reading
	// gives. Both sides of the edge are pinned so a change to ViTBlock or the kernel's static
	// arrays fails here loudly.
	limit := (apple - 2*gpu.ViTBlock*4) / 4
	if attnThreadgroupBytes(limit) > apple {
		t.Errorf("np=%d should fit in %d B, needs %d", limit, apple, attnThreadgroupBytes(limit))
	}
	if attnThreadgroupBytes(limit+1) <= apple {
		t.Errorf("np=%d should NOT fit in %d B, needs %d", limit+1, apple, attnThreadgroupBytes(limit+1))
	}
	if gpu.ViTBlock == 256 && limit != 7680 {
		t.Errorf("expected the 32 KiB limit to be np=7680, got %d", limit)
	}
}

// TestNewEncoder_declinesOverBudgetPatchCount is the end-to-end half of the C-06 gate: a tower
// whose patch count exceeds the device's threadgroup-memory budget must be REFUSED by newEncoder,
// not silently built into an encoder that later runs `attention` into an aborted command buffer
// and returns a plausible, wrong hidden state (metal.go's own doc: an over-budget dispatch
// "silently tolerates ... status Completed"). No checkpoint needed — newEncoder's budget check
// runs before any layer weight is touched, so a synthetic vision.GPUWeights with only NumPatches/
// Hidden/Inter set is enough to reach it.
func TestNewEncoder_declinesOverBudgetPatchCount(t *testing.T) {
	if _, err := gpu.CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	const apple = 32 << 10
	overBudget := (apple-2*gpu.ViTBlock*4)/4 + 1 // one past the 7680 boundary

	_, err := newEncoder(vision.GPUWeights{NumPatches: overBudget, Hidden: 64, Inter: 64})
	if err == nil {
		t.Fatalf("newEncoder accepted np=%d (over the threadgroup-memory budget) — should decline", overBudget)
	}
	if !strings.Contains(err.Error(), "threadgroup memory") {
		t.Errorf("decline error %q does not name the reason", err)
	}
	t.Logf("declined as expected: %v", err)
}
