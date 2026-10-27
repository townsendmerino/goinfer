//go:build gpu && goinfer_testhooks

package gpu

import (
	"github.com/townsendmerino/goinfer/gpu/internal/kernels"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestLocalize_BiasEpilogue is a localization step: does production's FUSED gemvBias kernel (dst
// = f32(acc)*aScale*bScale + bias, one dispatch, built in decoderunner.go for every biased W8A8
// projection) produce a BIT-DIFFERENT result than PrefillLastW8A8's UNFUSED approach (tiled GEMM
// → store → separate residual-kernel bias add), given the IDENTICAL quantized activation and the
// SAME real resident Q weight + bias from layer 0 of a real checkpoint? If the two disagree even
// in isolation (no attention, no multi-layer compounding), the epilogue fusion is the first
// differing stage (FMA contraction inside one shader vs a forced f32 round-trip through memory
// between two shaders). If they agree exactly, the difference is further downstream (rope, the KV
// cache write, or compounding over layers).
//
//	GOINFER_RESIDENT_GGUF=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
//	  GOINFER_HEAVY_TESTS=1 go test -tags 'gpu goinfer_testhooks' ./gpu/ -run TestLocalize_BiasEpilogue -v
func TestLocalize_BiasEpilogue(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_RESIDENT_GGUF")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no resident model at %s: %v", path, err)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Skip("model not GPU-resident")
	}
	rf := m.ResidentForwardForTest()
	rd, ok := rf.(*residentDecoder)
	if !ok {
		t.Fatalf("ResidentForwardForTest() is %T, want *residentDecoder", rf)
	}
	if len(rd.rm.Layers) == 0 {
		t.Fatal("no layers")
	}
	lw := &rd.rm.Layers[0]
	qw, ok := lw.Q.(*ResidentW8A8)
	if !ok {
		t.Fatalf("layer 0 q is %T, want *ResidentW8A8", lw.Q)
	}
	if lw.QBias == nil {
		t.Skip("this checkpoint's layer 0 has no q bias (not a Qwen2-bias model)")
	}
	hidden, _, _, _, _, _, _ := m.Dims()
	eps := float32(m.NormEps())
	addOne := m.RMSAddOne()

	c := rd.c
	for _, f := range []func() error{kernels.ContextEnsureFuseFunc(c), kernels.ContextEnsureGEMVBiasFunc(c), kernels.ContextEnsureTiledFunc(c), kernels.ContextEnsureLayerFunc(c)} {
		if err := f(); err != nil {
			t.Fatalf("ensure: %v", err)
		}
	}

	rng := rand.New(rand.NewSource(1))
	x := make([]float32, hidden)
	for i := range x {
		x[i] = float32(rng.NormFloat64()) * 0.5
	}
	xBuf, err := (*kernels.ContextDevice(c)).TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(x), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		t.Fatalf("upload x: %v", err)
	}
	defer xBuf.Release()

	// Shared activation: production's exact fused rmsnorm+quant kernel (both
	// paths below consume the SAME (aq, aScale) — this test controls out any
	// quantizer difference, isolating the GEMM+bias epilogue specifically).
	kp := kernels.PadK32(hidden)
	aq, err := (*kernels.ContextDevice(c)).TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(kp / 4 * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("alloc aq: %v", err)
	}
	defer aq.Release()
	as, err := (*kernels.ContextDevice(c)).TryCreateBuffer(&wgpu.BufferDescriptor{Size: 4, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("alloc as: %v", err)
	}
	defer as.Release()
	rq, err := (*kernels.ContextDevice(c)).TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{uint32(hidden), kernels.F32bits(eps), kernels.BoolU32(addOne), uint32(kp)}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		t.Fatalf("alloc rq uni: %v", err)
	}
	defer rq.Release()
	rqBG, err := (*kernels.ContextDevice(c)).TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: (*kernels.ContextRmsQuantLayout(c)), Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: xBuf, Size: xBuf.GetSize()}, {Binding: 1, Buffer: lw.AttnNorm, Size: lw.AttnNorm.GetSize()},
		{Binding: 2, Buffer: aq, Size: aq.GetSize()}, {Binding: 3, Buffer: as, Size: as.GetSize()}, {Binding: 4, Buffer: rq, Size: rq.GetSize()},
	}})
	if err != nil {
		t.Fatalf("rq bindgroup: %v", err)
	}
	defer rqBG.Release()

	// PATH A (production): fused gemvBias — dst = f32(acc)*aScale*bScale + bias, one kernel.
	fusedOut, err := (*kernels.ContextDevice(c)).TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64((*kernels.ResidentW8A8Rows(qw)) * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		t.Fatalf("alloc fusedOut: %v", err)
	}
	defer fusedOut.Release()
	gbDims, err := (*kernels.ContextDevice(c)).TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{1, uint32((*kernels.ResidentW8A8Kp(qw))), uint32((*kernels.ResidentW8A8Rows(qw))), 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		t.Fatalf("alloc gbDims: %v", err)
	}
	defer gbDims.Release()
	gbBG, err := (*kernels.ContextDevice(c)).TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: (*kernels.ContextGemvBiasLayout(c)), Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: aq, Size: aq.GetSize()}, {Binding: 1, Buffer: (*kernels.ResidentW8A8Bq(qw)), Size: (*kernels.ResidentW8A8Bq(qw)).GetSize()},
		{Binding: 2, Buffer: as, Size: as.GetSize()}, {Binding: 3, Buffer: (*kernels.ResidentW8A8BScales(qw)), Size: (*kernels.ResidentW8A8BScales(qw)).GetSize()},
		{Binding: 4, Buffer: fusedOut, Size: fusedOut.GetSize()}, {Binding: 5, Buffer: gbDims, Size: gbDims.GetSize()},
		{Binding: 6, Buffer: lw.QBias, Size: lw.QBias.GetSize()},
	}})
	if err != nil {
		t.Fatalf("gb bindgroup: %v", err)
	}
	defer gbBG.Release()
	gx, gy := kernels.GemvGrid((*kernels.ResidentW8A8Rows(qw)))

	// PATH B (PrefillLastW8A8-style): tiled GEMM (no bias) -> store -> separate
	// residual-kernel bias add, mirroring biasAdd/tiledProj in prefillrunner.go exactly.
	gemmOut, err := (*kernels.ContextDevice(c)).TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64((*kernels.ResidentW8A8Rows(qw)) * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("alloc gemmOut: %v", err)
	}
	defer gemmOut.Release()
	tiledDims, err := (*kernels.ContextDevice(c)).TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{1, uint32((*kernels.ResidentW8A8Kp(qw))), uint32((*kernels.ResidentW8A8Rows(qw))), 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		t.Fatalf("alloc tiledDims: %v", err)
	}
	defer tiledDims.Release()
	tiledBG, err := (*kernels.ContextDevice(c)).TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: (*kernels.ContextTiledLayout(c)), Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: aq, Size: aq.GetSize()}, {Binding: 1, Buffer: (*kernels.ResidentW8A8Bq(qw)), Size: (*kernels.ResidentW8A8Bq(qw)).GetSize()},
		{Binding: 2, Buffer: as, Size: as.GetSize()}, {Binding: 3, Buffer: (*kernels.ResidentW8A8BScales(qw)), Size: (*kernels.ResidentW8A8BScales(qw)).GetSize()},
		{Binding: 4, Buffer: gemmOut, Size: gemmOut.GetSize()}, {Binding: 5, Buffer: tiledDims, Size: tiledDims.GetSize()},
	}})
	if err != nil {
		t.Fatalf("tiled bindgroup: %v", err)
	}
	defer tiledBG.Release()
	tgx, tgy := uint32((*kernels.ResidentW8A8Rows(qw))+15)/16, uint32(1+15)/16

	residDims, err := (*kernels.ContextDevice(c)).TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{uint32((*kernels.ResidentW8A8Rows(qw))), 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		t.Fatalf("alloc residDims: %v", err)
	}
	defer residDims.Release()
	residBG, err := (*kernels.ContextDevice(c)).TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: (*kernels.ContextResidualLayout(c)), Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: gemmOut, Size: gemmOut.GetSize()}, {Binding: 1, Buffer: lw.QBias, Size: lw.QBias.GetSize()}, {Binding: 2, Buffer: residDims, Size: residDims.GetSize()},
	}})
	if err != nil {
		t.Fatalf("resid bindgroup: %v", err)
	}
	defer residBG.Release()

	enc, err := (*kernels.ContextDevice(c)).TryCreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("encoder: %v", err)
	}
	defer enc.Release()
	disp := func(pl *wgpu.ComputePipeline, bg *wgpu.BindGroup, gx, gy uint32) {
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(pl)
		pass.SetBindGroup(0, bg, nil)
		pass.DispatchWorkgroups(gx, gy, 1)
		pass.TryEnd()
		pass.Release()
	}
	disp((*kernels.ContextRmsQuantPipeline(c)), rqBG, 1, 1)
	disp((*kernels.ContextGemvBiasPipeline(c)), gbBG, gx, gy)                                              // PATH A
	disp((*kernels.ContextTiledPipeline(c)), tiledBG, tgx, tgy)                                            // PATH B step 1: GEMM only
	disp((*kernels.ContextResidualPipeline(c)), residBG, uint32((*kernels.ResidentW8A8Rows(qw))+63)/64, 1) // PATH B step 2: + bias

	stagA, _ := (*kernels.ContextDevice(c)).TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64((*kernels.ResidentW8A8Rows(qw)) * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer stagA.Release()
	stagB, _ := (*kernels.ContextDevice(c)).TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64((*kernels.ResidentW8A8Rows(qw)) * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer stagB.Release()
	if err := enc.TryCopyBufferToBuffer(fusedOut, 0, stagA, 0, uint64((*kernels.ResidentW8A8Rows(qw))*4)); err != nil {
		t.Fatalf("copy A: %v", err)
	}
	if err := enc.TryCopyBufferToBuffer(gemmOut, 0, stagB, 0, uint64((*kernels.ResidentW8A8Rows(qw))*4)); err != nil {
		t.Fatalf("copy B: %v", err)
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		t.Fatalf("finish: %v", err)
	}
	defer cmd.Release()
	(*kernels.ContextQueue(c)).Submit(cmd)

	var stA, stB wgpu.MapAsyncStatus
	if err := stagA.TryMapAsync(wgpu.MapModeRead, 0, uint64((*kernels.ResidentW8A8Rows(qw))*4), func(s wgpu.MapAsyncStatus) { stA = s }); err != nil {
		t.Fatalf("map A: %v", err)
	}
	if err := stagB.TryMapAsync(wgpu.MapModeRead, 0, uint64((*kernels.ResidentW8A8Rows(qw))*4), func(s wgpu.MapAsyncStatus) { stB = s }); err != nil {
		t.Fatalf("map B: %v", err)
	}
	(*kernels.ContextDevice(c)).Poll(true, nil)
	if stA != wgpu.MapAsyncStatusSuccess || stB != wgpu.MapAsyncStatusSuccess {
		t.Fatalf("map failed: A=%v B=%v", stA, stB)
	}
	fused := make([]float32, (*kernels.ResidentW8A8Rows(qw)))
	copy(fused, wgpu.FromBytes[float32](stagA.GetMappedRange(0, uint((*kernels.ResidentW8A8Rows(qw))*4))))
	unfused := make([]float32, (*kernels.ResidentW8A8Rows(qw)))
	copy(unfused, wgpu.FromBytes[float32](stagB.GetMappedRange(0, uint((*kernels.ResidentW8A8Rows(qw))*4))))
	stagA.TryUnmap()
	stagB.TryUnmap()

	cos, maxAbs := cosine(fused, unfused)
	nDiff := 0
	var maxRelBits uint32
	for i := range fused {
		if fused[i] != unfused[i] {
			nDiff++
			// ULP distance via bit pattern, for values of the same sign.
			bf, bu := f32bitsOf(fused[i]), f32bitsOf(unfused[i])
			d := bf - bu
			if bu > bf {
				d = bu - bf
			}
			if d > maxRelBits {
				maxRelBits = d
			}
		}
	}
	t.Logf("fused (production gemvBias) vs unfused (tiled GEMM + separate residual bias-add): cosine=%.9f maxAbs=%.3e, %d/%d elements differ, max ULP distance=%d",
		cos, maxAbs, nDiff, len(fused), maxRelBits)
	t.Logf("first 8 fused:   %v", fused[:8])
	t.Logf("first 8 unfused: %v", unfused[:8])
}

func f32bitsOf(f float32) uint32 { return math.Float32bits(f) }
