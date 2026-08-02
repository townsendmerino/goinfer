//go:build cuda

package cuda

import (
	"context"
	"fmt"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestAttention_HeadDimWidths guards that the shipped `attention` kernel is correct across the
// head-dim widths goinfer's arch set uses — including the ones NO other cuda test exercises. It drives the SHIPPED `attention` kernel at hd 16/64/128/256/512 through
// the known-good validateGlue oracle (cosine vs a CPU GQA online-softmax reference). 128 is the
// existing green control; 256 is gemma3's width (already resident); 512 is the gemma4 global-head
// question the Phase-9a spec gates on; 16 and 64 are the SMALL end — gemma4's local layer is
// hd=16, below every previously-tested width. The kernel decomposes each head over a fixed
// 128-thread block, so the large end (512 = 4 elems/thread) and the small end (16 = 112 of 128
// threads idle) stress different assumptions: any hd ≥ blockDim or blockDim % hd == 0 dependence
// would break at 16, not 512. Adding these rows keeps a red on the Split-A resident run
// attributable to the geometry seam, not to the tiny head. Single variable = hd.
func TestAttention_HeadDimWidths(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Skipf("no context: %v", err)
	}
	defer ctx.Close()
	bg := context.Background()
	glmod, err := ctx.LoadModule(gluePTX)
	if err != nil {
		t.Fatalf("glue module: %v", err)
	}
	fRms, _ := glmod.Function("rmsnorm_quant")
	fAttn, _ := glmod.Function("attention")
	fSwiglu, _ := glmod.Function("glu_quant")
	stream := mustStream(t, ctx)
	const nH, nKV, I, nKeys = 12, 2, 8960, 129
	const scale = float32(0.05)
	for _, hd := range []int{16, 64, 128, 256, 512} {
		t.Run(fmt.Sprintf("hd%d", hd), func(t *testing.T) {
			validateGlue(t, ctx, stream, bg, fRms, fSwiglu, fAttn, nH*hd, I, nH, nKV, hd, nKeys, scale)
		})
	}
}
