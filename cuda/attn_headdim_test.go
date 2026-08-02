//go:build cuda

package cuda

import (
	"context"
	"fmt"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestAttention_TailPoison is the scratch-to-max gate for the per-layer geometry port
// (9a-P2). The runner allocates ONE Q/context scratch sized to the WIDEST layer (maxQDim =
// nH*512 for Gemma 4's global head), then runs the NARROW hd=16 local layer into it. If the
// attention kernel — or anything sizing off the allocation — reads past nH*hd, it picks up
// the residue a previous wide layer left in the tail.
//
// A zeroed scratch cannot catch that: an over-read folds in zeros, contributes nothing to
// the dot products / accumulations, and the hd=16 result stays correct — so
// TestAttention_HeadDimWidths passing at hd=16 proves nothing about tail-reads. The residue
// is non-zero, so the tail is memset to a sentinel and only the live hd=16 region written;
// the kernel then either stays byte-identical to the tight run (no over-read) or diverges
// (found the bug the zeroed test structurally could not see). It also closes the consumer-
// sizes-off-the-buffer gap the compiler removal alone cannot: anything deriving its extent
// from the wide allocation processes sentinel and diverges.
func TestAttention_TailPoison(t *testing.T) {
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
	fAttn, _ := glmod.Function("attention")
	stream := mustStream(t, ctx)

	const nH, nKV, hd, maxHd, nKeys = 12, 2, 16, 512, 129
	const scale = float32(0.05)
	const sentinel = float32(7.5) // the non-zero residue a wide layer would leave in the tail
	kvDim := nKV * hd

	// Live inputs (tight hd=16 layout), identical between the two runs.
	qh := make([]float32, nH*hd)
	for i := range qh {
		qh[i] = float32(math.Cos(float64(i) * 0.11))
	}
	kh := make([]float32, nKeys*kvDim)
	vh := make([]float32, nKeys*kvDim)
	for i := range kh {
		kh[i] = float32(math.Sin(float64(i) * 0.07))
		vh[i] = float32(math.Cos(float64(i) * 0.05))
	}

	// run drives the shipped attention kernel with the Q + context scratch sized to bufHd
	// (== hd is the tight baseline; == maxHd mirrors the runner's shared qB/cctx, live region
	// packed at [0,nH*hd) and the tail sentinel-poisoned). K/V stay tight, like the per-layer
	// KV caches. Returns the live [0,nH*hd) region of the output.
	run := func(bufHd int) []float32 {
		dqHost := make([]float32, nH*bufHd)
		for i := range dqHost {
			dqHost[i] = sentinel
		}
		copy(dqHost, qh) // live query at [0,nH*hd), sentinel beyond
		dq := mustAlloc[float32](t, ctx, nH*bufHd)
		dk := mustAlloc[float32](t, ctx, nKeys*kvDim)
		dv := mustAlloc[float32](t, ctx, nKeys*kvDim)
		dcHost := make([]float32, nH*bufHd)
		for i := range dcHost {
			dcHost[i] = sentinel
		}
		dc := mustAlloc[float32](t, ctx, nH*bufHd)
		_ = gc.CopyHtoD(bg, dq, dqHost)
		_ = gc.CopyHtoD(bg, dk, kh)
		_ = gc.CopyHtoD(bg, dv, vh)
		_ = gc.CopyHtoD(bg, dc, dcHost)
		_ = fAttn.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32(nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nKeys + 128) * 4)},
			gc.Arg(dq), gc.Arg(dk), gc.Arg(dv), gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(nKeys)), gc.ArgValue(scale), gc.ArgValue(int32(0)), gc.Arg(dc))
		_ = stream.Synchronize(bg)
		out := make([]float32, nH*bufHd)
		_ = gc.CopyDtoH(bg, out, dc)
		dq.Close()
		dk.Close()
		dv.Close()
		dc.Close()
		return out[:nH*hd]
	}

	tight := run(hd)   // tight nH*hd scratch (control)
	wide := run(maxHd) // 32x-wider scratch, non-zero sentinel tail — the residue case
	for i := range tight {
		if tight[i] != wide[i] {
			t.Fatalf("hd=16 attention in a maxHd=%d scratch with non-zero tail residue diverged at "+
				"element %d (tight %v, wide %v): the kernel reads past nH*hd into the tail. A zeroed "+
				"tail would have hidden this.", maxHd, i, tight[i], wide[i])
		}
	}
	t.Logf("tail poison clean: hd=16 attention byte-identical in tight vs %dx-wide sentinel-tailed scratch", maxHd/hd)
}

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
