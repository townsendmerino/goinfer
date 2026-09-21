//go:build darwin && goinfer_testhooks

package metal

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestR2_perLayerCtxDiff is the measurement R2's two prior investigations never took
// (docs/measurements/r2-attn-fa-2026-09-19.md, r2-attn-fa-followup-2026-09-20.md): they compared
// FINAL LOGITS of an attention_fa generation against a shipped-kernel generation, 28 layers and an
// int8-activation-quantized pipeline downstream of the kernel under test, on Gaussian-noise
// embeddings. The follow-up's own ULP control showed that pipeline is hypersensitive on that input
// (a 1-ULP scale nudge flips logits by ~0.9 immediately), so "two clean steps, then a stable
// plateau" is consistent with BOTH a real position-linked kernel defect AND a single int8 rounding
// flip whose timing happened to line up. This test separates them by comparing the kernels'
// OWN outputs (r.ctx) on IDENTICAL inputs, per layer, per decode step:
//
//   - trajectory 1: the shipped kernel end to end via ForwardEmb (the record's reference arm);
//   - trajectory 2: a manual per-layer harness with attention_fa OFF on both arms — must match
//     trajectory 1 bit-for-bit, or the harness itself is wrong (checked, fatal if not);
//   - trajectory 3: attention_fa's trajectory, where at every (step, layer) the shipped kernel is
//     run from the same pre-layer residual and KV state first, its ctx and layer output captured,
//     the residual restored, then attention_fa run — so |ctxFA - ctxShipped| is a pure kernel
//     comparison on identical (q, K, V, nKeys), and |xFA - xShipped| shows where any downstream
//     rounding flip lands. The trajectory then continues from attention_fa's own output, so its
//     per-step logits reproduce the record's arm exactly.
//
// The runtime toggle is r.decodeAttnFA (canUseAttnFA reads it on every dispatch; pipelines and the
// partial buffer are always built) — the same technique R1's root-cause tests used for its lane.
// One model load. Prefill length from GOINFER_R2_PREFILL (default 1600, the record's), steps from
// GOINFER_R2_STEPS (default 5). If a layer's ctx diverges locally (maxRel > 1e-3, three orders
// above f32 reduction noise), that layer's exact inputs and both outputs are dumped to
// GOINFER_R2_DUMP_DIR for TestR2_replayDump below.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags "darwin goinfer_testhooks" ./metal/ -run 'TestR2_perLayerCtxDiff$' -v -timeout 20m
func TestR2_perLayerCtxDiff(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint)")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.Getenv("GOINFER_TEST_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	prefillLen := 1600
	if v := os.Getenv("GOINFER_R2_PREFILL"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			t.Fatalf("GOINFER_R2_PREFILL: %v", err)
		}
		prefillLen = n
	}
	nSteps := 5
	if v := os.Getenv("GOINFER_R2_STEPS"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			t.Fatalf("GOINFER_R2_STEPS: %v", err)
		}
		nSteps = n
	}
	dumpDir := os.Getenv("GOINFER_R2_DUMP_DIR")

	logf := func(format string, args ...any) {
		t.Helper()
		t.Logf(format, args...)
		fmt.Fprintf(os.Stderr, "[r2-ctx] "+format+"\n", args...)
	}

	t.Setenv("GOINFER_METAL_ATTN_FA", "") // built with the kernel OFF; toggled at runtime below
	m, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	b := &metalBackend{}
	rf, ok, err := b.BuildResident(m)
	if err != nil || !ok {
		t.Fatalf("BuildResident: ok=%v err=%v", ok, err)
	}
	defer b.Close()
	r := rf.(*metalResident).r
	if r.decodeAttnFA {
		t.Fatalf("decodeAttnFA on at build despite GOINFER_METAL_ATTN_FA=\"\"")
	}
	if r.attnFAPartial == (Buffer{}) {
		t.Fatalf("attnFAPartial not allocated — attention_fa cannot engage on this model")
	}
	H, nL, nH := r.H, r.nL, r.nH
	g0 := r.layers[0].geom
	nKV, hd := g0.nKV, g0.hd
	kvDim := nKV * hd
	logf("model: H=%d nL=%d nH=%d nKV=%d hd=%d kvDim=%d prefillLen=%d steps=%d attnFADepthFloor=%d",
		H, nL, nH, nKV, hd, kvDim, prefillLen, nSteps, attnFADepthFloor)
	if prefillLen < attnFADepthFloor {
		t.Fatalf("prefillLen %d < attnFADepthFloor %d: attention_fa would never engage", prefillLen, attnFADepthFloor)
	}

	// Inputs: EXACTLY the e2e repro's construction (rand.NewSource(7); prefill rows drawn first,
	// then one row per decode step, in order) so the numbers are comparable to the records.
	genInputs := func() (embs [][]float32, stepEmbs [][]float32) {
		rng := rand.New(rand.NewSource(7))
		embs = make([][]float32, prefillLen)
		for i := range embs {
			row := make([]float32, H)
			for j := range row {
				row[j] = float32(rng.NormFloat64()) * 0.05
			}
			embs[i] = row
		}
		stepEmbs = make([][]float32, nSteps)
		for s := range stepEmbs {
			row := make([]float32, H)
			for j := range row {
				row[j] = float32(rng.NormFloat64()) * 0.05
			}
			stepEmbs[s] = row
		}
		return
	}
	embs, stepEmbs := genInputs()

	cosMax := func(a, bb []float32) (cos, maxAbs float64) {
		var dot, na, nb float64
		for i := range a {
			x, y := float64(a[i]), float64(bb[i])
			dot += x * y
			na += x * x
			nb += y * y
			if d := math.Abs(x - y); d > maxAbs {
				maxAbs = d
			}
		}
		return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-300), maxAbs
	}
	bitIdentical := func(a, bb []float32) bool {
		if len(a) != len(bb) {
			return false
		}
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(bb[i]) {
				return false
			}
		}
		return true
	}
	prefill := func() {
		r.decodeAttnFA = false
		if _, err := r.ForwardBatch(embs, 0); err != nil {
			t.Fatalf("ForwardBatch: %v", err)
		}
	}
	// The tail of forwardLogits (metal/model.go), verbatim: final norm -> int8 LM head -> host copy.
	logitsFromX := func() []float32 {
		e := r.q.Begin()
		r.encodeNorm(e, r.x, r.finalNorm, r.finalNormBias, r.aq, r.aSc)
		e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
		e.End()
		r.recordExecErr(e.Err())
		r.finalizeLogits()
		return append([]float32(nil), r.logitsHost...)
	}

	// ---- trajectory 1: shipped, end to end via ForwardEmb (the record's reference arm) ----
	prefill()
	shippedLogits := make([][]float32, nSteps)
	for s := 0; s < nSteps; s++ {
		l := r.ForwardEmb(append([]float32(nil), stepEmbs[s]...), prefillLen+s)
		shippedLogits[s] = append([]float32(nil), l...)
	}
	if err := r.takeExecErr(); err != nil {
		t.Fatalf("trajectory 1: %v", err)
	}

	// manual per-layer step. mode: modeControl = shipped kernels only (the pure shipped
	// trajectory, stepped per layer); modeFA = at every layer run the shipped arm from the
	// pre-layer state, capture, restore, run attention_fa, capture, and continue from
	// attention_fa's output; modePerturb = shipped kernels only, but after layer 0 add a tiny
	// fixed pseudo-random perturbation to the residual (GOINFER_R2_PERTURB, default 1e-6 — the
	// magnitude of attention_fa's own measured layer-output discrepancy) and continue: the
	// control the 2026-09-20 follow-up's ULP test should have been (same magnitude and
	// location as the real discrepancy, not a global scale change at every layer).
	const (
		modeControl = iota
		modeFA
		modePerturb
	)
	perturbMag := 1e-6
	if v := os.Getenv("GOINFER_R2_PERTURB"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			t.Fatalf("GOINFER_R2_PERTURB: %v", err)
		}
		perturbMag = f
	}
	type layerRec struct {
		ctxSh, ctxFA []float32
		xSh, xFA     []float32
		q            []float32 // attention_fa's q for this layer (r.qkv[:nH*hd] right after its run)
	}
	runStep := func(s int, mode int) (logits []float32, recs []layerRec) {
		pos := prefillLen + s
		copy(r.x.Floats(), stepEmbs[s])
		r.addLearnedPos(pos)
		r.setPos(pos)
		if mode == modeFA {
			r.decodeAttnFA = true
			for l := 0; l < nL; l++ {
				if !r.canUseAttnFA(l) {
					t.Fatalf("step %d layer %d: canUseAttnFA false with the toggle on (curNKeys=%d)", s, l, r.curNKeys)
				}
			}
			r.decodeAttnFA = false
		}
		prng := rand.New(rand.NewSource(int64(1000 + s)))
		recs = make([]layerRec, nL)
		for l := 0; l < nL; l++ {
			xPre := append([]float32(nil), r.x.Floats()[:H]...)
			// shipped arm from xPre
			r.decodeAttnFA = false
			e := r.q.Begin()
			r.encodeLayer(e, l)
			e.End()
			r.recordExecErr(e.Err())
			recs[l].ctxSh = append([]float32(nil), r.ctx.Floats()[:nH*hd]...)
			recs[l].xSh = append([]float32(nil), r.x.Floats()[:H]...)
			switch mode {
			case modeFA:
				// attention_fa arm from the SAME xPre and the SAME KV state (the shipped arm's
				// kv_store at (l,pos) wrote from xPre; this run rewrites identical values).
				copy(r.x.Floats(), xPre)
				r.decodeAttnFA = true
				e = r.q.Begin()
				r.encodeLayer(e, l)
				e.End()
				r.recordExecErr(e.Err())
				r.decodeAttnFA = false
				recs[l].ctxFA = append([]float32(nil), r.ctx.Floats()[:nH*hd]...)
				recs[l].xFA = append([]float32(nil), r.x.Floats()[:H]...)
				recs[l].q = append([]float32(nil), r.qkv.Floats()[:nH*hd]...)
				// trajectory continues from attention_fa's own layer output (r.x already holds it)
			case modePerturb:
				if l == 0 {
					xs := r.x.Floats()
					for i := 0; i < H; i++ {
						xs[i] += float32(prng.NormFloat64() * perturbMag)
					}
				}
				recs[l].xFA = append([]float32(nil), r.x.Floats()[:H]...)
			}
		}
		logits = logitsFromX()
		return
	}

	// ---- trajectory 2: manual harness, kernel OFF on both arms — must equal trajectory 1 exactly ----
	prefill()
	ctrl := make([][]layerRec, nSteps) // the pure shipped trajectory's per-layer residuals
	for s := 0; s < nSteps; s++ {
		lg, recs := runStep(s, modeControl)
		ctrl[s] = recs
		if !bitIdentical(lg, shippedLogits[s]) {
			cos, mx := cosMax(lg, shippedLogits[s])
			t.Fatalf("harness self-check FAILED at step %d: per-layer stepping (kernel off) is not bit-identical to ForwardEmb (cosine=%.9f maxAbs=%.3e) — fix the harness before trusting anything below", s, cos, mx)
		}
	}
	if err := r.takeExecErr(); err != nil {
		t.Fatalf("trajectory 2: %v", err)
	}
	logf("harness self-check: per-layer stepping with the kernel off reproduces ForwardEmb bit-for-bit at all %d steps", nSteps)

	countAbove := func(a, bb []float32, thr float64) int {
		n := 0
		for i := range a {
			if math.Abs(float64(a[i])-float64(bb[i])) > thr {
				n++
			}
		}
		return n
	}

	// ---- trajectory 3 (perturb mode, if requested): shipped kernels + a 1e-6 residual nudge after layer 0 ----
	if os.Getenv("GOINFER_R2_MODE") == "perturb" {
		prefill()
		logf("PERTURB CONTROL: shipped kernels only; after layer 0 of every step, x += N(0,1)*%.1e (seeded per step)", perturbMag)
		for s := 0; s < nSteps; s++ {
			pos := prefillLen + s
			lg, recs := runStep(s, modePerturb)
			cos, mx := cosMax(lg, shippedLogits[s])
			bad := cos < 0.9999 || mx > 1e-1
			logf("step %d pos=%d nKeys=%d: PERTURBED-trajectory logits vs shipped: cosine=%.7f maxAbs=%.4e bad=%v", s, pos, pos+1, cos, mx, bad)
			for l := 0; l < nL; l++ {
				_, acc := cosMax(recs[l].xFA, ctrl[s][l].xSh)
				n := countAbove(recs[l].xFA, ctrl[s][l].xSh, 1e-3)
				flag := ""
				if n > 0 {
					flag = "  <-- accumulated divergence crossed 1e-3"
				}
				logf("  step %d layer %2d: accumulated |perturbed - pure shipped| x maxAbs=%.3e elems>1e-3: %d%s", s, l, acc, n, flag)
			}
		}
		if err := r.takeExecErr(); err != nil {
			t.Fatalf("perturb trajectory: %v", err)
		}
		return
	}

	// ---- trajectory 3: attention_fa, instrumented per layer ----
	prefill()
	dumped := false
	for s := 0; s < nSteps; s++ {
		pos := prefillLen + s
		lg, recs := runStep(s, modeFA)
		cos, mx := cosMax(lg, shippedLogits[s])
		bad := cos < 0.9999 || mx > 1e-1
		logf("step %d pos=%d nKeys=%d (nKeys mod 4 = %d): FA-trajectory logits vs shipped: cosine=%.7f maxAbs=%.4e bad=%v",
			s, pos, pos+1, (pos+1)%4, cos, mx, bad)
		// per-layer local kernel comparison on identical inputs, plus the ACCUMULATED trajectory
		// divergence against the pure shipped trajectory (trajectory 2's per-layer residuals)
		worstLayer, worstRel := -1, 0.0
		for l := 0; l < nL; l++ {
			rc := recs[l]
			_, acc := cosMax(rc.xFA, ctrl[s][l].xSh)
			nAcc := countAbove(rc.xFA, ctrl[s][l].xSh, 1e-3)
			var amax, mxAbs, mxRel float64
			wi := -1
			for i := range rc.ctxSh {
				if a := math.Abs(float64(rc.ctxSh[i])); a > amax {
					amax = a
				}
			}
			for i := range rc.ctxSh {
				d := math.Abs(float64(rc.ctxFA[i]) - float64(rc.ctxSh[i]))
				rel := d / (math.Abs(float64(rc.ctxSh[i])) + 1e-3*amax)
				if d > mxAbs {
					mxAbs = d
				}
				if rel > mxRel {
					mxRel, wi = rel, i
				}
			}
			_, xMax := cosMax(rc.xFA, rc.xSh)
			flag := ""
			if mxRel > 1e-3 {
				flag = "  <-- LOCAL KERNEL DIVERGENCE"
			}
			if nAcc > 0 {
				flag += "  <-- accumulated divergence crossed 1e-3"
			}
			logf("  step %d layer %2d: ctx |FA-shipped| maxAbs=%.3e maxRel=%.3e (worst head %d dim %d, ctx amax %.3g) | local layer-out x maxAbs=%.3e | ACCUMULATED vs pure shipped: x maxAbs=%.3e elems>1e-3: %d%s",
				s, l, mxAbs, mxRel, wi/hd, wi%hd, amax, xMax, acc, nAcc, flag)
			if mxRel > worstRel {
				worstRel, worstLayer = mxRel, l
			}
			if !dumped && dumpDir != "" && mxRel > 1e-3 {
				// The exact inputs attention_fa saw for this dispatch: q = rc.q (r.qkv[:nH*hd], post-RoPE,
				// captured inside runStep right after this layer's attention_fa arm — r.qkv is per-layer
				// scratch, so reading it here after the whole step would give the LAST layer's q, a
				// mistake this test made once), K/V = layer l's own cache rows [0, nKeys) (per-layer
				// buffers, still valid here), nKeys = pos+1, scale = r.uScale.
				nKeys := pos + 1
				p := filepath.Join(dumpDir, fmt.Sprintf("r2-dump-prefill%d-step%d-layer%d.bin", prefillLen, s, l))
				if err := writeR2Dump(p, nH, nKV, hd, nKeys, r.uScale.Floats()[0],
					rc.q, r.kc[l].U16s()[:nKeys*kvDim], r.vc[l].U16s()[:nKeys*kvDim], rc.ctxSh, rc.ctxFA); err != nil {
					t.Fatalf("dump: %v", err)
				}
				logf("  dumped layer %d inputs+outputs to %s (replay with TestR2_replayDump, GOINFER_R2_DUMP=<path>)", l, p)
				dumped = true
			}
		}
		logf("step %d summary: worst local kernel divergence layer %d maxRel=%.3e", s, worstLayer, worstRel)
	}
	if err := r.takeExecErr(); err != nil {
		t.Fatalf("trajectory 3: %v", err)
	}
}

func writeR2Dump(path string, nH, nKV, hd, nKeys int, scale float32, q []float32, k, v []uint16, ctxSh, ctxFA []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := func(x any) error { return binary.Write(f, binary.LittleEndian, x) }
	for _, x := range []any{int32(nH), int32(nKV), int32(hd), int32(nKeys), scale, q, k, v, ctxSh, ctxFA} {
		if err := w(x); err != nil {
			return err
		}
	}
	return nil
}

func readR2Dump(path string) (nH, nKV, hd, nKeys int, scale float32, q []float32, k, v []uint16, ctxSh, ctxFA []float32, err error) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	var a, b2, c, d int32
	rd := func(x any) error { return binary.Read(f, binary.LittleEndian, x) }
	if err = rd(&a); err != nil {
		return
	}
	if err = rd(&b2); err != nil {
		return
	}
	if err = rd(&c); err != nil {
		return
	}
	if err = rd(&d); err != nil {
		return
	}
	if err = rd(&scale); err != nil {
		return
	}
	nH, nKV, hd, nKeys = int(a), int(b2), int(c), int(d)
	kvDim := nKV * hd
	q = make([]float32, nH*hd)
	k = make([]uint16, nKeys*kvDim)
	v = make([]uint16, nKeys*kvDim)
	ctxSh = make([]float32, nH*hd)
	ctxFA = make([]float32, nH*hd)
	for _, x := range []any{q, k, v, ctxSh, ctxFA} {
		if err = rd(x); err != nil {
			return
		}
	}
	return
}

// TestR2_replayDump replays a TestR2_perLayerCtxDiff dump in ISOLATION — the exact q/K/V/nKeys/scale
// attention_fa saw at the first locally-divergent layer — through attention_fa+combine (at the
// production nSplit and at S=1/S=4), the shipped attention kernel, and the f64 CPU reference
// (cpuAttention, attn_shape_test.go). If the isolated kernel reproduces the dumped ctxFA and the
// CPU reference sides with the shipped kernel, the defect is in the kernel's math on this data and
// is bisectable here without a model; if the isolated kernel is CORRECT on the same bytes, the
// production dispatch differs from the isolated one in something other than the data.
//
//	GOINFER_R2_DUMP=<path> go test -tags "darwin goinfer_testhooks" ./metal/ -run 'TestR2_replayDump$' -v
func TestR2_replayDump(t *testing.T) {
	p := os.Getenv("GOINFER_R2_DUMP")
	if p == "" {
		t.Skip("set GOINFER_R2_DUMP to a dump written by TestR2_perLayerCtxDiff")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	nH, nKV, hd, nKeys, scale, q, kh, vh, ctxSh, ctxFA, err := readR2Dump(p)
	if err != nil {
		t.Fatalf("read dump: %v", err)
	}
	G := nH / nKV
	t.Logf("dump: nH=%d nKV=%d hd=%d nKeys=%d scale=%.9g", nH, nKV, hd, nKeys, scale)

	kf, vf := make([]float32, len(kh)), make([]float32, len(vh))
	for i := range kh {
		kf[i], vf[i] = f16ToF32(kh[i]), f16ToF32(vh[i])
	}
	ref := cpuAttention(q, kf, vf, nH, nKV, hd, nKeys, 0, scale)

	report := func(name string, got []float32) {
		var mxRef, mxSh, mxFA, amax float64
		for i := range ref {
			if a := math.Abs(float64(ref[i])); a > amax {
				amax = a
			}
		}
		for i := range got {
			mxRef = math.Max(mxRef, math.Abs(float64(got[i])-float64(ref[i])))
			mxSh = math.Max(mxSh, math.Abs(float64(got[i])-float64(ctxSh[i])))
			mxFA = math.Max(mxFA, math.Abs(float64(got[i])-float64(ctxFA[i])))
		}
		t.Logf("%-28s maxAbs vs cpu-f64 ref=%.3e | vs dumped shipped ctx=%.3e | vs dumped FA ctx=%.3e  (ref amax %.3g)", name, mxRef, mxSh, mxFA, amax)
	}
	{
		var mxSF float64
		for i := range ctxSh {
			mxSF = math.Max(mxSF, math.Abs(float64(ctxSh[i])-float64(ctxFA[i])))
		}
		t.Logf("dumped production outputs: |shipped - FA| maxAbs=%.3e", mxSF)
	}
	report("dumped shipped ctx", ctxSh)
	report("dumped FA ctx", ctxFA)

	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pFA, err := d.NewComputePipeline(lib, "attention_fa")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	pCombine, err := d.NewComputePipeline(lib, "attention_fa_combine")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	pAttn, err := d.NewComputePipeline(lib, "attention")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	qB := NewBufferFloats(d, q)
	kc, vc := NewBufferU16s(d, kh), NewBufferU16s(d, vh)
	uNKV, uG := NewBufferU32(d, uint32(nKV)), NewBufferU32(d, uint32(G))
	uNH := NewBufferU32(d, uint32(nH))
	uHd := NewBufferU32(d, uint32(hd))
	uNKeys := NewBufferU32(d, uint32(nKeys))
	uScale := NewBufferFloats(d, []float32{scale})
	uWin := NewBufferU32(d, 0)
	uZero := NewBufferU32(d, 0)
	sinks := NewBufferFloats(d, make([]float32, nH))
	cq := d.NewCommandQueue()

	// shipped kernel in isolation
	{
		out := d.NewBufferLen(nH * hd)
		enc := cq.Begin()
		enc.Dispatch(pAttn, nH*tgReduceAttn, tgReduceAttn, qB, kc, vc, out, uNH, uNKV, uHd, uNKeys, uScale, uWin, sinks, uZero)
		enc.End()
		report("isolated shipped attention", append([]float32(nil), out.Floats()...))
	}
	for _, nSplit := range []int{14, 1, 4} {
		pStride := G * (hd + 2)
		partial := d.NewBufferLen(nKV * nSplit * pStride)
		out := d.NewBufferLen(nH * hd)
		uNSplit := NewBufferU32(d, uint32(nSplit))
		shmBytes := 128 * 6 * G * 4
		enc := cq.Begin()
		enc.DispatchTG(pFA, nKV*nSplit*128, 128, shmBytes, qB, kc, vc, partial, uNKV, uG, uNKeys, uScale, uWin, uNSplit)
		enc.Dispatch(pCombine, nH*hd, hd, partial, out, uG, uHd, uNSplit)
		enc.End()
		report(fmt.Sprintf("isolated attention_fa S=%d", nSplit), append([]float32(nil), out.Floats()...))
	}
}
