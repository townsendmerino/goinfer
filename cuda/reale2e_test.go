//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"runtime"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestRealE2EDecode is B step 4: the REAL end-to-end decode tok/s of the parity-green
// forward path (TestRealForwardParity), on the real q4_k_m checkpoint. Full per-token
// work — mixed int4/int8 GEMVs + RoPE + GQA attention + requant glue + on-device argmax +
// per-token sync — driven autoregressively at real advancing positions, through a
// LockOSThread-pinned CUDA executor fed by a channel (one round-trip per token), so the
// thread-safety executor cost is IN the number (guardrail #3). Wall-clock steady-state
// (the number a user feels), vs same-box pinned Ollama 149 / WebGPU 111.6. cgo-free.
// NOT A CORRECTNESS GATE, despite the name it used to carry. This drives a HAND-ROLLED sequence of
// kernel launches written inside the test — not the production resident path — so its tokens are not
// evidence about what ships. MEASURED, not assumed: driving the CPU reference identically (same
// prompt, same re-feed of the last prompt token, same argmax rule) yields a DIFFERENT sequence, so
// this bespoke pipeline does not reproduce decoder.forward. Making it faithful would just duplicate
// the production resident path, which is already gated — so it is labelled instead.
//
// CORRECTNESS FOR THIS PATH LIVES IN TestBackendResidentWired: it loads --backend cuda, takes the
// real *cudaResident, and compares argmax against mcpu.ForwardForTest position by position
// (measured 7/8 exact, worst near-tie 0.087%, hard fails 0). That is the token-identity evidence;
// this file is the throughput number.
func TestRealE2EDecodeThroughput(t *testing.T) {
	requireHeavyModel(t)
	gguf := os.Getenv("GOINFER_CUDA_MODEL")
	if gguf == "" {
		gguf = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no model")
	}

	// ---- host side (CPU): load + pack all weights (no CUDA yet) ----
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

	type hw struct {
		kind string
		wpk  []uint32
		ws   []float32
		ws16 []uint16
		N, K int
	}
	packI8 := func(q8 []int8, N, K int) []uint32 {
		p := make([]uint32, N*(K/4))
		for i := range p {
			p[i] = uint32(uint8(q8[i*4])) | uint32(uint8(q8[i*4+1]))<<8 | uint32(uint8(q8[i*4+2]))<<16 | uint32(uint8(q8[i*4+3]))<<24
		}
		return p
	}
	pack := func(pw interface {
		Rows() int
		Cols() int
		Kind() string
		Int4() ([]byte, []float32, int, bool)
		Int8() ([]int8, []float32, bool, bool)
		F32() ([]float32, bool)
	}) hw {
		N, K := pw.Rows(), pw.Cols()
		switch pw.Kind() {
		case "int4":
			q4, sc, _, _ := pw.Int4()
			p := make([]uint32, N*(K/8))
			for i := range p {
				b := q4[i*4 : i*4+4]
				p[i] = permuteFast(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
			}
			gs := make([]uint16, len(sc))
			for i, v := range sc {
				gs[i] = f32tof16(v)
			}
			return hw{kind: "int4", wpk: p, ws16: gs, N: N, K: K}
		case "int8":
			q8, sc, _, _ := pw.Int8()
			return hw{kind: "int8", wpk: packI8(q8, N, K), ws: sc, N: N, K: K}
		case "f32":
			f32, _ := pw.F32()
			q8, sc := linalg.QuantizeRowsInt8(f32, N, K)
			return hw{kind: "int8", wpk: packI8(q8, N, K), ws: sc, N: N, K: K}
		}
		t.Fatalf("kind %q", pw.Kind())
		return hw{}
	}
	type hlayer struct {
		q, k, v, o, g, u, d hw
		qb, kb, vb          []float32
		preNorm, postNorm   []float32
		hasBias             bool
	}
	hls := make([]hlayer, nLayers)
	for l := 0; l < nLayers; l++ {
		lw := &w.Layers[l]
		hls[l] = hlayer{
			q: pack(&lw.QProj), k: pack(&lw.KProj), v: pack(&lw.VProj), o: pack(&lw.OProj),
			g: pack(&lw.GateProj), u: pack(&lw.UpProj), d: pack(&lw.DownProj),
			preNorm: lw.PreAttnNorm, postNorm: lw.PreMLPNorm,
		}
		if lw.QBias != nil {
			hls[l].qb, hls[l].kb, hls[l].vb, hls[l].hasBias = lw.QBias, lw.KBias, lw.VBias, true
		}
	}
	lmSrc := &w.LMHead
	if w.LMHead.Rows() == 0 {
		lmSrc = &w.Embed
	}
	hlm := pack(lmSrc)
	hFinal := w.FinalNorm

	// Measured weight-byte stream per token: every resident projection is read exactly once
	// per token (decode is weight-streaming). Splits the GPU time into GEMV-bytes vs glue.
	var wBytes, lmBytes int64
	bytesOf := func(h hw) int64 { return int64(len(h.wpk))*4 + int64(len(h.ws))*4 + int64(len(h.ws16))*2 }
	for l := 0; l < nLayers; l++ {
		h := &hls[l]
		for _, x := range []hw{h.q, h.k, h.v, h.o, h.g, h.u, h.d} {
			wBytes += bytesOf(x)
		}
	}
	lmBytes = bytesOf(hlm)
	wBytes += lmBytes

	// ---- pinned CUDA executor: one OS thread owns the context; jobs arrive by channel ----
	reqCh := make(chan func() error)
	ackCh := make(chan error)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		for j := range reqCh {
			ackCh <- j()
		}
	}()
	do := func(j func() error) error { reqCh <- j; return <-ackCh }
	defer close(reqCh)

	bg := context.Background()
	var cx *gc.Context
	var stream *gc.Stream
	var gemvW4, gemvW8, ropeKV, fRms, fQ, fAttn, fSw, fArg *gc.Function

	// Release the primary context when the test ends. dev.Primary() RETAINS a refcounted,
	// per-device singleton: leaving it retained pins the shared context alive for the whole
	// test binary, so no OTHER test's Close can ever drop the count to zero — and nothing
	// anyone allocated is reclaimed until the process exits. That leak saturated the 8 GB card
	// mid-suite, after which every Alloc/NewStream returned nil and the resulting zero-filled
	// buffers surfaced as bogus "cosine 0.000000 — layout/unpack mismatch" parity failures.
	//
	// Registered AFTER `defer close(reqCh)` so LIFO runs it FIRST: the close has to execute on
	// the pinned executor thread that made the context current, while the channel is still live.
	defer func() {
		if cx != nil {
			_ = do(func() error { return cx.Close() })
		}
	}()

	type wq struct {
		kind string
		W    *gc.Buffer[uint32]
		ws   *gc.Buffer[float32]
		ws16 *gc.Buffer[uint16]
		N, K int
	}
	type layer struct {
		q, k, v, o, g, u, d wq
		qb, kb, vb          *gc.Buffer[float32]
		preNorm, postNorm   *gc.Buffer[float32]
		hasBias             bool
	}
	var layers []layer
	var lmW wq
	var finalNorm, invF *gc.Buffer[float32]
	var x, aSc, qB, kB, vB, cctx, cSc, mSc, gO, uO, dSc, dScr, logits, argVal *gc.Buffer[float32]
	var aq, cq, mq, dq *gc.Buffer[int32]
	var argIdx *gc.Buffer[int32]
	var kc, vc []*gc.Buffer[float32]
	const ctxCap = 4096

	// upload helpers (run only inside jobs, ctx current on the pinned thread)
	up32 := func(v []float32) *gc.Buffer[float32] {
		b := mustAlloc[float32](t, cx, len(v))
		_ = gc.CopyHtoD(bg, b, v)
		return b
	}
	upu32 := func(v []uint32) *gc.Buffer[uint32] {
		b := mustAlloc[uint32](t, cx, len(v))
		_ = gc.CopyHtoD(bg, b, v)
		return b
	}
	upu16 := func(v []uint16) *gc.Buffer[uint16] {
		b := mustAlloc[uint16](t, cx, len(v))
		_ = gc.CopyHtoD(bg, b, v)
		return b
	}
	af := func(n int) *gc.Buffer[float32] { b := mustAlloc[float32](t, cx, n); return b }
	ai := func(n int) *gc.Buffer[int32] { b := mustAlloc[int32](t, cx, n); return b }
	upW := func(h hw) wq {
		w := wq{kind: h.kind, W: upu32(h.wpk), N: h.N, K: h.K}
		if h.kind == "int4" {
			w.ws16 = upu16(h.ws16)
		} else {
			w.ws = up32(h.ws)
		}
		return w
	}

	// ---- setup job: create context on the pinned thread, upload everything ----
	if e := do(func() error {
		if err := gc.Init(); err != nil {
			return err
		}
		dev, err := gc.GetDevice(0)
		if err != nil {
			return err
		}
		cx, err = dev.Primary()
		if err != nil {
			return err
		}
		gmod, err := cx.LoadModule(gemvFwdPTX)
		if err != nil {
			return err
		}
		glmod, _ := cx.LoadModule(gluePTX)
		// Generic quantized GEMVs: aikit/gpu, since the Phase-1b blob-split.
		qmod, _ := cx.LoadModule(gpu.QuantGEMVPTX)
		gemvW4, _ = qmod.Function("gemv_w4a8_fwd")
		gemvW8, _ = qmod.Function("gemv_w8a8_fwd")
		ropeKV, _ = gmod.Function("rope_kv")
		fRms, _ = glmod.Function("rmsnorm_quant")
		fQ, _ = glmod.Function("quant_vec")
		fAttn, _ = glmod.Function("attention")
		fSw, _ = glmod.Function("glu_quant")
		// argmax_reduce lives in argmaxPTX, not gluePTX — see the note in e2e_decode_test.go (C-14).
		amod, _ := cx.LoadModule(argmaxPTX)
		fArg, _ = amod.Function("argmax_reduce")
		stream, _ = cx.NewStream()

		layers = make([]layer, nLayers)
		for l := 0; l < nLayers; l++ {
			h := &hls[l]
			L := layer{q: upW(h.q), k: upW(h.k), v: upW(h.v), o: upW(h.o), g: upW(h.g), u: upW(h.u), d: upW(h.d),
				preNorm: up32(h.preNorm), postNorm: up32(h.postNorm)}
			if h.hasBias {
				L.qb, L.kb, L.vb, L.hasBias = up32(h.qb), up32(h.kb), up32(h.vb), true
			}
			layers[l] = L
		}
		lmW = upW(hlm)
		finalNorm = up32(hFinal)
		invF = up32(invFreq)
		x, aSc = af(H), af(1)
		aq = ai(H / 4)
		qB, kB, vB = af(qDim), af(kvDim), af(kvDim)
		kc, vc = make([]*gc.Buffer[float32], nLayers), make([]*gc.Buffer[float32], nLayers)
		for l := range kc {
			kc[l], vc[l] = af(ctxCap*kvDim), af(ctxCap*kvDim)
		}
		cctx, cSc, cq = af(qDim), af(1), ai(qDim/4)
		mSc, mq = af(1), ai(H/4)
		gO, uO = af(I), af(I)
		dSc, dScr, dq = af(1), af(I), ai(I/4)
		logits = af(vocab)
		argIdx, argVal = ai(1), af(1)
		return nil
	}); e != nil {
		t.Fatalf("setup: %v", e)
	}

	// ---- per-token forward (runs on the pinned thread inside a job), returns argmax id ----
	L := func(f *gc.Function, cfg gc.LaunchConfig, a ...gc.KernelArg) { _ = f.LaunchOn(bg, stream, cfg, a...) }
	g1 := func(n, b int) gc.LaunchConfig {
		return gc.LaunchConfig{GridX: uint32((n + b - 1) / b), GridY: 1, GridZ: 1, BlockX: uint32(b), BlockY: 1, BlockZ: 1}
	}
	one := func(b, sh int) gc.LaunchConfig {
		return gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: uint32(b), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(sh)}
	}
	rms := func(src, nrm *gc.Buffer[float32], qOut *gc.Buffer[int32], sOut *gc.Buffer[float32]) {
		L(fRms, one(256, (H+256)*4), gc.Arg(src), gc.Arg(nrm), gc.ArgValue(int32(H)), gc.ArgValue(eps), gc.ArgValue(int32(0)), gc.Arg(qOut), gc.Arg(sOut))
	}
	nullBias := gc.ArgDevicePtr(0)
	doG := func(wt wq, a *gc.Buffer[int32], as *gc.Buffer[float32], bias gc.KernelArg, dst *gc.Buffer[float32], accum int32) {
		cfg := gc.LaunchConfig{GridX: uint32((wt.N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		if wt.kind == "int4" {
			L(gemvW4, cfg, gc.Arg(wt.W), gc.Arg(a), gc.Arg(wt.ws16), gc.Arg(as), bias, gc.ArgValue(int32(wt.N)), gc.ArgValue(int32(wt.K/8)), gc.ArgValue(int32(wt.K/32)), gc.Arg(dst), gc.ArgValue(accum))
		} else {
			L(gemvW8, cfg, gc.Arg(wt.W), gc.Arg(a), gc.Arg(wt.ws), gc.Arg(as), bias, gc.ArgValue(int32(wt.N)), gc.ArgValue(int32(wt.K/4)), gc.Arg(dst), gc.ArgValue(accum))
		}
	}
	launchToken := func(emb []float32, pos int) {
		_ = gc.CopyHtoD(bg, x, emb)
		for l := 0; l < nLayers; l++ {
			Ly := &layers[l]
			rms(x, Ly.preNorm, aq, aSc)
			qb, kb, vb := nullBias, nullBias, nullBias
			if Ly.hasBias {
				qb, kb, vb = gc.Arg(Ly.qb), gc.Arg(Ly.kb), gc.Arg(Ly.vb)
			}
			doG(Ly.q, aq, aSc, qb, qB, 0)
			doG(Ly.k, aq, aSc, kb, kB, 0)
			doG(Ly.v, aq, aSc, vb, vB, 0)
			L(ropeKV, g1(nH*half+nKV*half, 256), gc.Arg(qB), gc.Arg(kB), gc.Arg(vB), gc.Arg(invF), gc.Arg(kc[l]), gc.Arg(vc[l]),
				gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(pos)))
			nKeys := pos + 1
			L(fAttn, gc.LaunchConfig{GridX: uint32(nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nKeys + 128) * 4)},
				gc.Arg(qB), gc.Arg(kc[l]), gc.Arg(vc[l]), gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(nKeys)), gc.ArgValue(attnScale), gc.ArgValue(int32(0)), gc.Arg(cctx))
			L(fQ, one(256, 256*4), gc.Arg(cctx), gc.ArgValue(int32(qDim)), gc.Arg(cq), gc.Arg(cSc))
			doG(Ly.o, cq, cSc, nullBias, x, 1)
			rms(x, Ly.postNorm, mq, mSc)
			doG(Ly.g, mq, mSc, nullBias, gO, 0)
			doG(Ly.u, mq, mSc, nullBias, uO, 0)
			L(fSw, one(256, 256*4), gc.Arg(gO), gc.Arg(uO), gc.ArgValue(int32(0)), gc.ArgValue(int32(0)), gc.ArgValue(int32(I)), gc.ArgValue(int32(1)), gc.Arg(dq), gc.Arg(dSc), gc.Arg(dScr))
			doG(Ly.d, dq, dSc, nullBias, x, 1)
		}
		rms(x, finalNorm, aq, aSc)
		doG(lmW, aq, aSc, nullBias, logits, 0)
		L(fArg, one(256, 256*4+256*4), gc.Arg(logits), gc.ArgValue(int32(vocab)), gc.Arg(argIdx), gc.Arg(argVal))
	}
	// step = launch the token, then sync + read back the argmax id.
	step := func(emb []float32, pos int) int {
		launchToken(emb, pos)
		_ = stream.Synchronize(bg)
		id := make([]int32, 1)
		_ = gc.CopyDtoH(bg, id, argIdx)
		return int(id[0])
	}

	// ---- warm, then wall-clock steady-state autoregressive decode through the executor ----
	prompt := []int{785, 3840, 315, 24231, 6137}
	pos := 0
	for _, tok := range prompt {
		tk := tok
		p := pos
		_ = do(func() error { step(m.EmbedResidentForTest(tk), p); return nil })
		pos++
	}
	// warm decode
	tok := prompt[len(prompt)-1]
	for i := 0; i < 8; i++ {
		tk, p := tok, pos
		var nt int
		_ = do(func() error { nt = step(m.EmbedResidentForTest(tk), p); return nil })
		tok, pos = nt, pos+1
	}

	// timed: best per-token over several batches, real advancing positions, executor round-trip/token.
	best := 1e18
	for b := 0; b < 6; b++ {
		const N = 16
		startPos, startTok := pos, tok
		t0 := time.Now()
		for i := 0; i < N; i++ {
			tk, p := tok, pos
			var nt int
			_ = do(func() error { nt = step(m.EmbedResidentForTest(tk), p); return nil })
			tok, pos = nt, pos+1
		}
		dt := time.Since(t0).Seconds() / N
		if dt < best {
			best = dt
		}
		_ = startPos
		_ = startTok
	}
	tps := 1.0 / best

	// Directly measure the executor tax that the headline carries: the per-token round-trip
	// is exactly one empty do() (reqCh<-j; <-ackCh) to the pinned worker. Time many empty
	// round-trips → the channel-hop cost included in every token above.
	const hops = 20000
	th := time.Now()
	for i := 0; i < hops; i++ {
		_ = do(func() error { return nil })
	}
	hopUs := time.Since(th).Seconds() / hops * 1e6

	// ---- decomposition: CPU launch-issue cost vs GPU execution time ----
	// Issue N tokens' launches WITHOUT a per-token sync (async burst) and bracket the same
	// work with CUDA events. cpuIssue = what the CPU spends feeding the GPU (~launches ×
	// per-dispatch tax); gpu = actual device execution. If cpuIssue >= gpu the token is
	// LAUNCH-BOUND — the CPU can't feed the device fast enough and fusion/fewer launches is
	// the only lever (bytes are already at the GEMV roofline).
	var cpuIssueMs, gpuMs float64
	launchesPerTok := nLayers*13 + 2 // launch diet: fused rope_kv (-3) + accum epilogues (-2)
	_ = do(func() error {
		const N = 16
		emb := m.EmbedResidentForTest(tok)
		for i := 0; i < 4; i++ { // warm
			launchToken(emb, pos+i)
		}
		_ = stream.Synchronize(bg)
		evS, _ := cx.NewEvent()
		evE, _ := cx.NewEvent()
		t0 := time.Now()
		_ = evS.Record(stream)
		for i := 0; i < N; i++ {
			launchToken(emb, pos+i) // launches only — no sync, no readback
		}
		_ = evE.Record(stream)
		cpuIssueMs = time.Since(t0).Seconds() * 1000 / N // CPU-side issue cost (async)
		_ = stream.Synchronize(bg)
		el, _ := evS.Elapsed(evE)
		gpuMs = float64(el.Microseconds()) / 1000 / N // device execution
		return nil
	})

	const ollama, webgpu = 149.0, 111.6
	t.Logf("=== REAL e2e decode (parity-green path, real q4_k_m, pinned executor + channel/token, on-device argmax, real positions) ===")
	t.Logf("  HEADLINE (LockOSThread worker + channel round-trip/token — the multi-goroutine backend path): %.2f ms/token | %.1f tok/s", best*1000, tps)
	t.Logf("  measured executor tax = %.3f us/token (empty channel round-trip) = %.3f%% of the token — IN the headline, not on top of it", hopUs, hopUs/1e6/best*100)
	t.Logf("  vs Ollama %.0f (%.2fx) | vs our WebGPU %.1f (%.2fx) | cgo-free driver-only", ollama, tps/ollama, webgpu, tps/webgpu)
	t.Logf("  DECOMPOSITION: %d launches/token | CPU issue %.2f ms | GPU exec %.2f ms | wall %.2f ms",
		launchesPerTok, cpuIssueMs, gpuMs, best*1000)
	t.Logf("    per-dispatch CPU tax = %.2f us | verdict: %s",
		cpuIssueMs*1000/float64(launchesPerTok),
		verdictOf(cpuIssueMs, gpuMs))
	// GEMV-bytes vs glue: the chained-GEMV rate measured in-tree is ~374 GB/s.
	const chainGBs = 374.0
	gemvMs := float64(wBytes) / (chainGBs * 1e9) * 1000
	t.Logf("    weight stream %.0f MB/token (LM head %.0f MB = %.0f%%) → GEMV %.2f ms @ %.0f GB/s | GLUE = %.2f ms (%.0f%% of GPU)",
		float64(wBytes)/1e6, float64(lmBytes)/1e6, float64(lmBytes)/float64(wBytes)*100,
		gemvMs, chainGBs, gpuMs-gemvMs, (gpuMs-gemvMs)/gpuMs*100)
}

// verdictOf classifies the CPU-issue vs GPU-exec relationship. A near-tie is the crossover:
// the CPU launch stream exactly saturates the device, so the async-hiding margin is gone and
// BOTH fewer launches and less glue pay (the 0.5B regime).
func verdictOf(cpu, gpu float64) string {
	switch {
	case cpu > gpu*1.1:
		return "LAUNCH-BOUND (CPU can't feed the GPU — fusion/fewer launches is the only lever)"
	case cpu > gpu*0.9:
		return "AT THE CROSSOVER (CPU issue ~= GPU exec; async-hiding margin gone — fusion cuts both)"
	default:
		return "GPU-bound (device is the limit; CPU issue hides under it)"
	}
}
