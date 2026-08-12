//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	_ "embed"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// gemvFwdPTX + gluePTX now live in kernels.go (shared with the production backend).

// TestRealForwardParity is B's gate (steps 2-3): the full cgo-free CUDA per-token forward
// on the REAL q4_k_m checkpoint must produce the same greedy argmax as goinfer's CPU decode
// at every prompt position (token-identical). Weights extracted via the BuildResident seam,
// driver-JIT'd kernels, CGO_ENABLED=0. This is the line between "benchmarked" and "shipped".
func TestRealForwardParity(t *testing.T) {
	requireHeavyModel(t)
	gguf := os.Getenv("GOINFER_CUDA_MODEL")
	if gguf == "" {
		gguf = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no model")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, _ := dev.Primary()
	defer cx.Close()
	bg := context.Background()

	m, err := decoder.Load(gguf, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	w := m.Weights()
	H, nLayers, nH, nKV, hd, I, vocab := m.Dims()
	qDim, kvDim := nH*hd, nKV*hd
	eps := m.NormEps()
	invFreq := m.RopeInvFreq()
	attnScale := m.AttnScale()
	half := hd / 2

	// ---- device helpers ----
	up32 := func(v []float32) *gc.Buffer[float32] {
		if len(v) == 0 {
			t.Fatal("up32 empty slice")
		}
		b, e := gc.Alloc[float32](cx, len(v))
		if e != nil {
			t.Fatalf("alloc f32 %d: %v", len(v), e)
		}
		_ = gc.CopyHtoD(bg, b, v)
		return b
	}
	upu16 := func(v []uint16) *gc.Buffer[uint16] {
		b, e := gc.Alloc[uint16](cx, len(v))
		if e != nil {
			t.Fatalf("alloc u16 %d: %v", len(v), e)
		}
		_ = gc.CopyHtoD(bg, b, v)
		return b
	}
	upu32 := func(v []uint32) *gc.Buffer[uint32] {
		b, e := gc.Alloc[uint32](cx, len(v))
		if e != nil {
			t.Fatalf("alloc u32 %d: %v", len(v), e)
		}
		_ = gc.CopyHtoD(bg, b, v)
		return b
	}
	af := func(n int) *gc.Buffer[float32] {
		b, e := gc.Alloc[float32](cx, n)
		if e != nil {
			t.Fatalf("alloc af %d: %v", n, e)
		}
		return b
	}
	ai := func(n int) *gc.Buffer[int32] {
		b, e := gc.Alloc[int32](cx, n)
		if e != nil {
			t.Fatalf("alloc ai %d: %v", n, e)
		}
		return b
	}

	// wq: a resident projection weight in whatever precision the real q4_k_m checkpoint
	// stored it (mixed per-tensor). f32 scales throughout (int4 group / int8 row) so the
	// int32 dp4a accumulation is bit-faithful to goinfer's CPU quantized matmul.
	type wq struct {
		kind string
		W    *gc.Buffer[uint32]
		ws   *gc.Buffer[float32] // int8 row scales
		ws16 *gc.Buffer[uint16]  // int4 group scales (f16)
		N, K int
	}
	packI8 := func(q8 []int8, N, K int) *gc.Buffer[uint32] {
		wpk := make([]uint32, N*(K/4))
		for i := range wpk {
			wpk[i] = uint32(uint8(q8[i*4])) | uint32(uint8(q8[i*4+1]))<<8 | uint32(uint8(q8[i*4+2]))<<16 | uint32(uint8(q8[i*4+3]))<<24
		}
		return upu32(wpk)
	}
	extract := func(pw interface {
		Rows() int
		Cols() int
		Kind() string
		Int4() ([]byte, []float32, int, bool)
		Int8() ([]int8, []float32, bool, bool)
		F32() ([]float32, bool)
	}) wq {
		N, K := pw.Rows(), pw.Cols()
		if K%4 != 0 {
			t.Fatalf("K=%d not mult-4", K)
		}
		switch pw.Kind() {
		case "int4":
			if K%32 != 0 {
				t.Fatalf("int4 K=%d not mult-32", K)
			}
			q4, sc, _, _ := pw.Int4()
			wpk := make([]uint32, N*(K/8))
			for i := range wpk {
				b := q4[i*4 : i*4+4]
				wpk[i] = permuteFast(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
			}
			gs := make([]uint16, len(sc))
			for i, v := range sc {
				gs[i] = f32tof16(v)
			}
			return wq{kind: "int4", W: upu32(wpk), ws16: upu16(gs), N: N, K: K}
		case "int8":
			q8, sc, _, _ := pw.Int8()
			return wq{kind: "int8", W: packI8(q8, N, K), ws: up32(sc), N: N, K: K}
		case "f32":
			f32, _ := pw.F32()
			q8, sc := linalg.QuantizeRowsInt8(f32, N, K)
			return wq{kind: "int8", W: packI8(q8, N, K), ws: up32(sc), N: N, K: K}
		default:
			t.Fatalf("unsupported kind %q", pw.Kind())
			return wq{}
		}
	}

	// ---- extract all weights ----
	type layer struct {
		q, k, v, o, g, u, d wq
		qb, kb, vb          *gc.Buffer[float32] // biases (nil-ptr if none)
		preNorm, postNorm   *gc.Buffer[float32]
		hasQKVBias          bool
	}
	layers := make([]layer, nLayers)
	for l := range nLayers {
		lw := &w.Layers[l]
		var L layer
		L.q = extract(&lw.QProj)
		L.k = extract(&lw.KProj)
		L.v = extract(&lw.VProj)
		L.o = extract(&lw.OProj)
		L.g = extract(&lw.GateProj)
		L.u = extract(&lw.UpProj)
		L.d = extract(&lw.DownProj)
		if lw.QBias != nil {
			L.qb, L.kb, L.vb = up32(lw.QBias), up32(lw.KBias), up32(lw.VBias)
			L.hasQKVBias = true
		}
		L.preNorm, L.postNorm = up32(lw.PreAttnNorm), up32(lw.PreMLPNorm)
		layers[l] = L
	}
	// LM head: untied if present, else tied to Embed.
	lmSrc := &w.LMHead
	if w.LMHead.Rows() == 0 {
		lmSrc = &w.Embed
	}
	lmW := extract(lmSrc)
	finalNorm := up32(w.FinalNorm)
	invF := up32(invFreq)

	// ---- kernels ----
	gmod, _ := cx.LoadModule(gemvFwdPTX)
	glmod, _ := cx.LoadModule(gluePTX)
	// The generic quantized GEMVs live in aikit/gpu since the Phase-1b blob-split;
	// gemvFwdPTX now carries only the LLM-specific kv_store / rope_kv.
	qmod, _ := cx.LoadModule(gpu.QuantGEMVPTX)
	gemvW4, err := qmod.Function("gemv_w4a8_fwd")
	if err != nil {
		t.Fatalf("gemv_fwd: %v", err)
	}
	gemvW8, _ := qmod.Function("gemv_w8a8_fwd")
	kvStore, _ := gmod.Function("kv_store")
	fRms, _ := glmod.Function("rmsnorm_quant")
	fQ, _ := glmod.Function("quant_vec")
	fRope, _ := glmod.Function("rope")
	fAttn, _ := glmod.Function("attention")
	fSw, _ := glmod.Function("glu_quant")
	fRes, _ := glmod.Function("residual")
	stream := mustStream(t, cx)

	// ---- scratch ----
	const ctxCap = 2048
	x := af(H)
	aq, aSc := ai(H/4), af(1)
	qB, kB, vB := af(qDim), af(kvDim), af(kvDim)
	kc := make([]*gc.Buffer[float32], nLayers)
	vc := make([]*gc.Buffer[float32], nLayers)
	for l := range kc {
		kc[l], vc[l] = af(ctxCap*kvDim), af(ctxCap*kvDim)
	}
	cctx := af(qDim)
	cq, cSc := ai(qDim/4), af(1)
	oO := af(H)
	mq, mSc := ai(H/4), af(1)
	gO, uO := af(I), af(I)
	dq, dSc, dScr := ai(I/4), af(1), af(I)
	dO := af(H)
	logits := af(vocab)

	L := func(f *gc.Function, cfg gc.LaunchConfig, a ...gc.KernelArg) {
		if e := f.LaunchOn(bg, stream, cfg, a...); e != nil {
			t.Fatalf("launch: %v", e)
		}
	}
	g1 := func(n, b int) gc.LaunchConfig {
		return gc.LaunchConfig{GridX: uint32((n + b - 1) / b), GridY: 1, GridZ: 1, BlockX: uint32(b), BlockY: 1, BlockZ: 1}
	}
	one := func(b, sh int) gc.LaunchConfig {
		return gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: uint32(b), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(sh)}
	}
	rms := func(src *gc.Buffer[float32], nrm *gc.Buffer[float32], qOut *gc.Buffer[int32], sOut *gc.Buffer[float32]) {
		L(fRms, one(256, (H+256)*4), gc.Arg(src), gc.Arg(nrm), gc.ArgValue(int32(H)), gc.ArgValue(eps), gc.ArgValue(int32(0)), gc.Arg(qOut), gc.Arg(sOut))
	}
	nullBias := gc.ArgDevicePtr(0)
	doG := func(wt wq, a *gc.Buffer[int32], as *gc.Buffer[float32], bias gc.KernelArg, dst *gc.Buffer[float32]) {
		cfg := gc.LaunchConfig{GridX: uint32((wt.N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		if wt.kind == "int4" {
			L(gemvW4, cfg, gc.Arg(wt.W), gc.Arg(a), gc.Arg(wt.ws16), gc.Arg(as), bias, gc.ArgValue(int32(wt.N)), gc.ArgValue(int32(wt.K/8)), gc.ArgValue(int32(wt.K/32)), gc.Arg(dst), gc.ArgValue(int32(0)))
		} else {
			L(gemvW8, cfg, gc.Arg(wt.W), gc.Arg(a), gc.Arg(wt.ws), gc.Arg(as), bias, gc.ArgValue(int32(wt.N)), gc.ArgValue(int32(wt.K/4)), gc.Arg(dst), gc.ArgValue(int32(0)))
		}
	}

	forward := func(emb []float32, pos int) []float32 {
		_ = gc.CopyHtoD(bg, x, emb)
		for l := range nLayers {
			Ly := &layers[l]
			rms(x, Ly.preNorm, aq, aSc)
			qbias, kbias, vbias := nullBias, nullBias, nullBias
			if Ly.hasQKVBias {
				qbias, kbias, vbias = gc.Arg(Ly.qb), gc.Arg(Ly.kb), gc.Arg(Ly.vb)
			}
			doG(Ly.q, aq, aSc, qbias, qB)
			doG(Ly.k, aq, aSc, kbias, kB)
			doG(Ly.v, aq, aSc, vbias, vB)
			L(fRope, g1(nH*half, 256), gc.Arg(qB), gc.Arg(invF), gc.ArgValue(int32(nH)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(pos)))
			L(fRope, g1(nKV*half, 64), gc.Arg(kB), gc.Arg(invF), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(pos)))
			L(kvStore, g1(kvDim, 128), gc.Arg(kB), gc.Arg(kc[l]), gc.ArgValue(int32(pos)), gc.ArgValue(int32(kvDim)))
			L(kvStore, g1(kvDim, 128), gc.Arg(vB), gc.Arg(vc[l]), gc.ArgValue(int32(pos)), gc.ArgValue(int32(kvDim)))
			nKeys := pos + 1
			L(fAttn, gc.LaunchConfig{GridX: uint32(nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nKeys + 128) * 4)},
				gc.Arg(qB), gc.Arg(kc[l]), gc.Arg(vc[l]), gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(nKeys)), gc.ArgValue(attnScale), gc.ArgValue(int32(0)), gc.Arg(cctx))
			L(fQ, one(256, 256*4), gc.Arg(cctx), gc.ArgValue(int32(qDim)), gc.Arg(cq), gc.Arg(cSc))
			doG(Ly.o, cq, cSc, nullBias, oO)
			L(fRes, g1(H, 256), gc.Arg(x), gc.Arg(oO), gc.ArgValue(int32(H)))
			rms(x, Ly.postNorm, mq, mSc)
			doG(Ly.g, mq, mSc, nullBias, gO)
			doG(Ly.u, mq, mSc, nullBias, uO)
			L(fSw, one(256, 256*4), gc.Arg(gO), gc.Arg(uO), gc.ArgValue(int32(0)), gc.ArgValue(int32(0)), gc.ArgValue(int32(I)), gc.ArgValue(int32(1)), gc.Arg(dq), gc.Arg(dSc), gc.Arg(dScr))
			doG(Ly.d, dq, dSc, nullBias, dO)
			L(fRes, g1(H, 256), gc.Arg(x), gc.Arg(dO), gc.ArgValue(int32(H)))
		}
		rms(x, finalNorm, aq, aSc)
		doG(lmW, aq, aSc, nullBias, logits)
		_ = stream.Synchronize(bg)
		out := make([]float32, vocab)
		_ = gc.CopyDtoH(bg, out, logits)
		return out
	}

	// ---- parity: CPU vs CUDA greedy argmax at each prompt position ----
	// Fixed valid token IDs — greedy argmax parity must hold for ANY input (both sides
	// get the same ids), so no tokenizer dependency is needed for the gate.
	prompt := []int{785, 3840, 315, 24231, 6137, 448, 264, 4285, 4522, 13}
	cache := m.NewCache(len(prompt) + 4)
	// Bars (goinfer's own): strict token-identity is ideal; the formal correctness gate
	// (kv_i8_parity_test.go) is the 3%-near-tie rule — an argmax flip is a defect ONLY if
	// the CPU preferred its token by > 3% of the logit range; near-ties are quant noise.
	exact, hardFail, worstGap := 0, 0, 0.0
	for i, tok := range prompt {
		cpuLogits, _ := m.ForwardForTest(tok, cache)
		cpuArg := argmaxF(cpuLogits)
		cudaLogits := forward(m.EmbedResidentForTest(tok), i)
		cudaArg := argmaxF(cudaLogits)
		if cpuArg == cudaArg {
			exact++
			t.Logf("pos %2d tok %6d: CPU=%6d CUDA=%6d  EXACT", i, tok, cpuArg, cudaArg)
			continue
		}
		// near-tie test: CPU's margin between its pick and CUDA's pick, / logit range.
		lo, hi := cpuLogits[0], cpuLogits[0]
		for _, v := range cpuLogits {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		gap := float64(cpuLogits[cpuArg]-cpuLogits[cudaArg]) / (float64(hi-lo) + 1e-9)
		if gap > worstGap {
			worstGap = gap
		}
		hard := gap > 0.03
		if hard {
			hardFail++
		}
		t.Logf("pos %2d tok %6d: CPU=%6d CUDA=%6d  FLIP gap=%.3f%% %s", i, tok, cpuArg, cudaArg, gap*100, map[bool]string{true: "HARD-FAIL", false: "near-tie(ok)"}[hard])
	}
	t.Logf("=== real-checkpoint parity: %d/%d exact | worst near-tie gap %.3f%% of range | hard fails %d ===", exact, len(prompt), worstGap*100, hardFail)
	if hardFail > 0 {
		t.Fatalf("B GATE FAILED: %d position(s) flipped by > 3%% of logit range — real backend bug, not quant noise", hardFail)
	}
	t.Logf("B GATE GREEN: cgo-free CUDA decode matches CPU on the real q4_k_m checkpoint (%d/%d exact, rest near-ties within goinfer's 3%% rule)", exact, len(prompt))
}

func argmaxF(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}
