//go:build cuda

package cuda

import (
	"context"
	"fmt"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestAttention_HeadDimWidths guards that the shipped `attention` kernel is correct across the
// head-dim widths goinfer's arch set uses — including the ones NO other cuda test exercises. It drives the SHIPPED `attention` kernel at hd 128/256/512 through
// the known-good validateGlue oracle (cosine vs a CPU GQA online-softmax reference). 128 is the
// existing green control; 256 is gemma3's width (already resident); 512 is the gemma4 global-head
// question the Phase-9a spec gates on. Single variable = hd, so a red is attributable to head dim.
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
	for _, hd := range []int{128, 256, 512} {
		t.Run(fmt.Sprintf("hd%d", hd), func(t *testing.T) {
			validateGlue(t, ctx, stream, bg, fRms, fSwiglu, fAttn, nH*hd, I, nH, nKV, hd, nKeys, scale)
		})
	}
}
