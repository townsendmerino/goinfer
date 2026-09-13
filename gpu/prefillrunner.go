//go:build gpu

package gpu

import (
	"fmt"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// runModelToModelW narrows a resident runModel (the general polymorphic
// representation DecodeRunner uses — W8A8/W4A8, MoE, MLA, Mamba, DeltaNet,
// sliding window, QK-norm, bias, per-layer geometry overrides, …) down to the
// plain dense-W8A8 ModelW shape PrefillLastW8A8 accepts, or reports ok=false when
// this model uses ANY feature outside that shape. The same scope
// DecodeTokenFusedBatched declares ("no MoE/MLA/SSM/bias/QK-norm").
//
// q/k/v bias (Qwen2) is DELIBERATELY declined despite AttnWeights/biasAdd having
// working plumbing for it below (kept rather than deleted — see their own
// comments): measured on a real checkpoint (qwen2.5-coder-0.5b, GOINFER_HEAVY_TESTS=1,
// TestResidentPrefillLast_parity).
//
// UPDATE (2026-09-12, second investigation round): the original "diverges at
// nKeys>=3" symptom had TWO causes, not one. Cause #1, now FIXED: this function
// unconditionally dispatched c.attnPipeline (the plain f32 kernel), while
// decoderunner.go's sequential resident decode picks its kernel per-geometry via
// attnKernel (attention.go) — for this exact checkpoint's geometry (hd=64,
// kvDim=128, f32 KV) that's attnKeysEligible, so decode used the KEY-SPLIT kernel
// while this function always used the plain one. Two different kernels computing
// the same attention is not guaranteed bit-identical. Fixing this function to call
// the same c.attnKernel(hd, kvDim, false) decoderunner.go uses took nKeys=3
// bias-enabled parity from cosine 0.99/maxAbs 1.24 to EXACT (cosine 1.0, maxAbs
// ~2.9e-6, i.e. float32 noise) — confirmed correct up through nKeys=4.
//
// Cause #2, still OPEN: even with both paths on the identical kernel, nKeys past
// a threshold still diverges (cosine ~0.995-0.999, maxAbs ~0.4-1.2 depending on
// nKeys — not monotonic with nKeys past the threshold, e.g. nKeys=20 is BETTER
// than nKeys=5 on Metal). THE THRESHOLD ITSELF IS HARDWARE-DEPENDENT, measured on
// the two real GPUs available: Metal/M1 Pro breaks at nKeys>=5 (exact through
// nKeys=4); Vulkan/RTX 2070 SUPER breaks at nKeys>=8 (exact through nKeys=7,
// confirmed at every nKeys 1-7, maxAbs <=5.7e-6 = float32 noise the whole way).
// Same WGSL source, same Go dispatch code, two different backends, two different
// break points — this points AWAY from a pure logic/off-by-one bug (which would
// break at the same nKeys everywhere) and TOWARD something backend/precision-
// specific: how naga lowers attnKeysShaderWGSL's online-softmax reduction to
// SPIR-V (Vulkan) vs MSL (Metal), a driver-level fast-math/FMA-contraction
// difference, or a genuine numerical instability in that kernel that different
// backends' rounding happens to trigger at different key counts. Since both paths
// now dispatch the SAME c.attnKeysPipeline for the SAME cached K/V, either (a) the
// K/V cache contents themselves differ subtly between this function's rope()+cpy()
// writes and decoderunner.go's fused qkvFinalize writes (never diffed line-by-line
// — see attnKeysShaderWGSL vs qkvFinalizeShaderWGSL/ropeShaderWGSL), or (b)
// attnKeysShaderWGSL (attention.go, the multi-key tiled online-softmax kernel)
// itself has a real precision bug past some key count that a same-kernel-vs-itself
// comparison can still expose if the SOURCE data feeding it differs. Ruled out
// separately: the K/V bias-add width bug fixed alongside the kernel-mismatch fix
// (reproduces identically before/after that fix, in isolation); flushPasses/
// passesPerFlush (a huge passesPerFlush disables it with no change); bias being
// simply wrong (disabling it entirely makes the real-checkpoint case WORSE,
// ~0.2-0.5 cosine, at ANY nKeys). The M=20 synthetic-weight gate
// (TestPrefillLastW8A8_parity, no bias, always uses c.attnKernel now too) stays
// bit-exact throughout on BOTH backends — this is real-checkpoint-weight-
// magnitude-specific, or specific to a code path the synthetic test never
// exercises (e.g. real RoPE frequency values vs the test's synthetic invFreq).
//
// Re-enable bias only after cause #2 is found and re-gated; until then this
// decline keeps residentPrefillSeed's fallback to the (slower, proven-correct)
// sequential loop, rather than risk serving silently-wrong prefill logits for the
// most common dense-checkpoint family (Qwen2.5-*).
//
// It is a zero-copy view: every *wgpu.Buffer is wrapped, not duplicated, and the
// caller must not Close() the resulting ModelW (ModelW.Release would double-free
// buffers rd.rm still owns) — the wrapper DeviceBuffers exist only so ModelW's
// field types match; PrefillLastW8A8 never calls Close on them either.
// hd is the model's head dimension — needed only to tell genuine partial RoPE
// (ropeHalf set to something less than hd/2) apart from ropeHalf simply being SET
// to the full-rotation value instead of left at its 0 "use hd/2" sentinel; both
// PrefillLastW8A8's and DecodeTokenFusedBatched's rope() closures hardcode
// half:=hd/2 (full rotation, no partial-RoPE support), so the latter is fine and
// only the former must decline.
func runModelToModelW(rm *runModel, hd int) (ModelW, bool) {
	if rm.moe != nil || rm.mla != nil || rm.mamba != nil || rm.dnet != nil ||
		rm.kvF16 || rm.kvI8 || rm.slidingWindow != 0 || rm.gatedGELU ||
		(rm.ropeHalf != 0 && rm.ropeHalf != hd/2) ||
		hd > attnWG { // ensureAttnWide is never in PrefillLastW8A8's ensure-list (attnKernel
		// would otherwise dispatch a nil c.attnWidePipeline); dnet/mamba above already
		// exclude the one real hd=256 family (Gated-DeltaNet hybrids), so this only
		// guards a theoretical wide-head-dim dense model, not a real gap.
		return ModelW{}, false
	}
	view := func(b *wgpu.Buffer) *DeviceBuffer {
		if b == nil {
			return nil
		}
		return &DeviceBuffer{buf: b}
	}
	layers := make([]LayerW, len(rm.layers))
	for i := range rm.layers {
		l := &rm.layers[i]
		if l.qBias != nil || l.kBias != nil || l.vBias != nil ||
			l.qNorm != nil || l.kNorm != nil ||
			l.oBias != nil || l.postAttnNorm != nil || l.postMLPNorm != nil ||
			l.hasSink || l.isLocal || (l.ropeScale != 0 && l.ropeScale != 1) ||
			l.ghd != 0 || l.gnKV != 0 || l.ghalf != 0 || l.gKEqV ||
			l.isMoE || l.shGate != nil ||
			l.mlaQA != nil || l.mlaQB != nil || l.mlaQ != nil || l.mlaKVA != nil || l.mlaO != nil ||
			l.isMamba || l.nemoKind != nemoNone || l.isDeltaNet || l.qGate {
			return ModelW{}, false
		}
		q, qOK := l.q.(*ResidentW8A8)
		k, kOK := l.k.(*ResidentW8A8)
		v, vOK := l.v.(*ResidentW8A8)
		o, oOK := l.o.(*ResidentW8A8)
		gate, gOK := l.gate.(*ResidentW8A8)
		up, uOK := l.up.(*ResidentW8A8)
		down, dOK := l.down.(*ResidentW8A8)
		if !qOK || !kOK || !vOK || !oOK || !gOK || !uOK || !dOK {
			return ModelW{}, false // a W4A8 (or other non-W8A8) projection — no tiled GEMM for it
		}
		layers[i] = LayerW{
			Attn: AttnWeights{
				Norm: view(l.attnNorm), QProj: q, KProj: k, VProj: v, OProj: o,
				InvFreq: view(l.invFreq), KCache: view(l.kCache), VCache: view(l.vCache),
			},
			MLPNorm: view(l.mlpNorm),
			Gate:    gate, Up: up, Down: down,
		}
	}
	lmHead, lmOK := rm.lmHead.(*ResidentW8A8)
	if !lmOK {
		return ModelW{}, false
	}
	return ModelW{Layers: layers, FinalNorm: view(rm.finalNorm), LMHead: lmHead}, true
}

// PrefillLastW8A8 is the batched-prefill analogue of DecodeTokenFusedBatched: it
// runs M prompt-token rows through the dense W8A8 layers with weight-heavy
// PROJECTIONS as ONE tiled GEMM each (weight streamed once across all M rows, via
// the unbounded-M kernel in gemm.go — DP4A-accelerated when Context.hasDP4A), and
// the cheap per-row ops (rmsnorm+quant, RoPE, KV-store, attention, SwiGLU,
// residual) looped over the rows, mirroring DecodeTokenFusedBatched's exact
// write-all-KV-then-attend structure and Metal command-buffer-cap handling
// (flushPasses) — the difference is M is NOT capped at gemmRowMaxM (real prompts
// run to hundreds/thousands of tokens, past gemmRow's array<i32,16> accumulator
// limit), and only the LAST row's logits are computed (h[M-1] → norm → LM head),
// matching decoder/forwardn.go's prefillLogits CPU contract and the
// decoder.Prefiller.PrefillLast interface this feeds — the other M-1 rows' KV
// entries are still written (the whole prompt's cache is filled either way), just
// their logits are never needed for prefill and so are never computed.
//
// Bit-equivalent to M sequential DecodeRunner.Run calls at positions
// start..start+M-1: same int8 inputs, same int32 accumulation (the tiled GEMM
// equals the GEMV, gated bit-exact by TestTiledDP4A_parity), same per-row ops, same
// KV-then-attend ordering. Gated by TestPrefillLastW8A8_parity.
func (c *Context) PrefillLastW8A8(xs [][]float32, m ModelW, hidden, nH, nKV, hd, inter int, positions []int, start int, eps, scale float32, addOne bool) ([]float32, error) {
	M := len(xs)
	if M == 0 {
		return nil, fmt.Errorf("gpu: PrefillLastW8A8 M=0")
	}
	if len(positions) != M {
		return nil, fmt.Errorf("gpu: PrefillLastW8A8 %d rows but %d positions", M, len(positions))
	}
	for _, f := range []func() error{c.ensureGEMV, c.ensureQuantize, c.ensureLayer, c.ensureAttn, c.ensureTiled} {
		if err := f(); err != nil {
			return nil, err
		}
	}
	kvDim := nKV * hd

	var keep []func()
	defer func() {
		for _, f := range keep {
			f()
		}
	}()
	keepBuf := func(b *wgpu.Buffer) { keep = append(keep, b.Release) }
	keepBG := func(b *wgpu.BindGroup) { keep = append(keep, b.Release) }

	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	defer func() { enc.Release() }()

	// buildErr accumulates the FIRST device-allocation/bind failure (audit C-27, mirrored
	// from DecodeTokenFusedBatched): short-circuit once set, return it before Submit, so
	// VRAM exhaustion is an error the caller falls back on rather than a panic or a
	// downstream nil-buffer deref.
	var buildErr error
	storF := func(n int) *wgpu.Buffer {
		if buildErr != nil {
			return nil
		}
		b, e := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
		if e != nil {
			buildErr = e
			return nil
		}
		keepBuf(b)
		return b
	}
	uni := func(v []uint32) *wgpu.Buffer {
		if buildErr != nil {
			return nil
		}
		b, e := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(v), Usage: wgpu.BufferUsageUniform})
		if e != nil {
			buildErr = e
			return nil
		}
		keepBuf(b)
		return b
	}
	bind := func(layout *wgpu.BindGroupLayout, bufs ...*wgpu.Buffer) *wgpu.BindGroup {
		if buildErr != nil {
			return nil
		}
		es := make([]wgpu.BindGroupEntry, len(bufs))
		for i, b := range bufs {
			if b == nil {
				buildErr = fmt.Errorf("gpu: PrefillLastW8A8: nil buffer for binding %d (allocation failed)", i)
				return nil
			}
			es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
		}
		bg, e := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: es})
		if e != nil {
			buildErr = e
			return nil
		}
		keepBG(bg)
		return bg
	}
	// flushPasses: same Metal-uncommitted-command-buffer-cap workaround as
	// DecodeTokenFusedBatched (see its doc comment) — a real constraint here too, and
	// worse: real prefill M runs to the hundreds/thousands vs speculative verify's ≤16.
	flushPasses := func() {
		if buildErr != nil {
			return
		}
		cmd, e := enc.TryFinish(nil)
		if e != nil {
			buildErr = e
			return
		}
		c.queue.Submit(cmd)
		cmd.Release()
		enc.Release()
		enc, e = c.device.TryCreateCommandEncoder(nil)
		if e != nil {
			buildErr = e
		}
	}
	const passesPerFlush = 32
	dispCount := 0
	disp := func(pl *wgpu.ComputePipeline, bg *wgpu.BindGroup, gx, gy uint32) {
		if buildErr != nil || bg == nil {
			return
		}
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(pl)
		pass.SetBindGroup(0, bg, nil)
		pass.DispatchWorkgroups(gx, gy, 1)
		if err := pass.TryEnd(); err != nil {
			buildErr = err
		}
		pass.Release()
		dispCount++
		if dispCount >= passesPerFlush {
			flushPasses()
			dispCount = 0
		}
	}
	cpy := func(src *wgpu.Buffer, so uint64, dst *wgpu.Buffer, do, sz uint64) {
		if buildErr != nil || src == nil || dst == nil {
			return
		}
		if err := enc.TryCopyBufferToBuffer(src, so, dst, do, sz); err != nil {
			buildErr = err
		}
	}

	rms := func(in, w *wgpu.Buffer) *wgpu.Buffer {
		out := storF(hidden)
		p := uni([]uint32{uint32(hidden), f32bits(eps), boolU32(addOne), 0})
		disp(c.rmsnormPipeline, bind(c.rmsnormLayout, in, w, out, p), 1, 1)
		return out
	}
	quant1 := func(in *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		kp := padK(K)
		q := storF(kp / 4)
		s := storF(1)
		p := uni([]uint32{1, uint32(K), uint32(kp), 0})
		disp(c.quantizePipeline, bind(c.quantizeLayout, in, q, s, p), 1, 1)
		return q, s
	}
	rope := func(vec, invFreq *wgpu.Buffer, heads, pos int) {
		half := hd / 2
		p := uni([]uint32{uint32(heads), uint32(hd), uint32(half), uint32(pos), f32bits(1), 0, 0, 0})
		disp(c.ropePipeline, bind(c.ropeLayout, vec, invFreq, p), uint32(heads*half+63)/64, 1)
	}
	residual := func(x, y *wgpu.Buffer) {
		p := uni([]uint32{uint32(hidden), 0, 0, 0})
		disp(c.residualPipeline, bind(c.residualLayout, x, y, p), uint32(hidden+63)/64, 1)
	}
	// biasAdd is residual's shape (vec[i] += bias[i]) at an explicit width n, for the
	// q/k/v bias epilogue — q/k/v projections are qDim/kvDim/kvDim wide, NOT hidden
	// (GQA makes kvDim < hidden), so residual's hardcoded hidden width would read/
	// write past the buffer. Mirrors decoderunner.go's own biasAdd exactly.
	biasAdd := func(vec, bias *wgpu.Buffer, n int) {
		p := uni([]uint32{uint32(n), 0, 0, 0})
		disp(c.residualPipeline, bind(c.residualLayout, vec, bias, p), uint32(n+63)/64, 1)
	}

	// tiledProj batches a projection over the given rows: gather each row's quantized
	// activation into one packed buffer, run ONE unbounded-M tiled GEMM (weight streamed
	// once, DP4A-accelerated when available), scatter the output rows back to per-row
	// buffers. Passing a length-1 slice (used for the LM head below) runs the tiled
	// kernel at M=1 — correct, just not its optimal shape; a one-off per prefill call is
	// cheap enough that a special M=1 GEMV path isn't worth the extra code path.
	tiledProj := func(xnRows []*wgpu.Buffer, rm *ResidentW8A8) []*wgpu.Buffer {
		rowsM := len(xnRows)
		K, N := rm.cols, rm.rows
		kw := rm.kp / 4
		aqC := storF(rowsM * kw)
		asC := storF(rowsM)
		for r, xn := range xnRows {
			q, s := quant1(xn, K)
			cpy(q, 0, aqC, uint64(r*kw*4), uint64(kw*4))
			cpy(s, 0, asC, uint64(r*4), 4)
		}
		dstC := storF(rowsM * N)
		p := uni([]uint32{uint32(rowsM), uint32(rm.kp), uint32(N), 0})
		gx, gy := (uint32(N)+15)/16, (uint32(rowsM)+15)/16
		disp(c.tiledPipeline, bind(c.tiledLayout, aqC, rm.bq, asC, rm.bScales, dstC, p), gx, gy)
		outs := make([]*wgpu.Buffer, rowsM)
		for r := range outs {
			o := storF(N)
			cpy(dstC, uint64(r*N*4), o, 0, uint64(N*4))
			outs[r] = o
		}
		return outs
	}

	xd := make([]*wgpu.Buffer, M)
	for r := range xs {
		b, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(xs[r]), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
		if err != nil {
			return nil, err
		}
		keepBuf(b)
		xd[r] = b
	}

	// FeatAttnSink: always bound (WGSL bind groups can't bind a null storage buffer);
	// runModelToModelW already declined any model with a real sink, so this is always the
	// harmless dummy + hasSink=0, matching attnShaderWGSL's convention.
	noSinks := storF(1)
	noHasSink := uni([]uint32{0, 0, 0, 0})

	for i := range m.Layers {
		lw := &m.Layers[i]
		xn := make([]*wgpu.Buffer, M)
		for r := range xn {
			xn[r] = rms(xd[r], lw.Attn.Norm.buf)
		}
		q := tiledProj(xn, lw.Attn.QProj)
		k := tiledProj(xn, lw.Attn.KProj)
		v := tiledProj(xn, lw.Attn.VProj)
		if lw.Attn.QBias != nil { // Qwen2 q/k/v bias — applied before RoPE, matching decoderunner.go
			for r := range xd {
				biasAdd(q[r], lw.Attn.QBias.buf, nH*hd)
				biasAdd(k[r], lw.Attn.KBias.buf, kvDim)
				biasAdd(v[r], lw.Attn.VBias.buf, kvDim)
			}
		}
		for r := range xd {
			rope(q[r], lw.Attn.InvFreq.buf, nH, positions[r])
			rope(k[r], lw.Attn.InvFreq.buf, nKV, positions[r])
			cpy(k[r], 0, lw.Attn.KCache.buf, uint64(positions[r]*kvDim*4), uint64(kvDim*4))
			cpy(v[r], 0, lw.Attn.VCache.buf, uint64(positions[r]*kvDim*4), uint64(kvDim*4))
		}
		// All rows' K/V are now in the cache; each row attends to its causal prefix
		// (including earlier rows of this same prefill block) — the same ordering
		// DecodeTokenFusedBatched's parity gate already proves correct. A real fused
		// multi-query batched-attention kernel (task-gpu-batched-prefill.md's
		// Increment 1) is a perf follow-on once dispatch-count overhead at real M is
		// actually measured (Increment 3's TTFT gate) — this per-row loop into the
		// existing M=1 kernel is correctness-equivalent today.
		// attnKernel picks the SAME kernel decoderunner.go's sequential resident
		// decode would pick for this geometry (kvF16=false always: runModelToModelW
		// declines kvF16/kvI8 models). Dispatching a different kernel than the
		// production Forward() path for the same geometry is exactly what caused the
		// nKeys>=3 divergence this comment's neighbor (attnKernel, attention.go) documents.
		attnPl, attnLy := c.attnKernel(hd, kvDim, false)
		ctxv := make([]*wgpu.Buffer, M)
		for r := range xd {
			cv := storF(nH * hd)
			ap := uni([]uint32{uint32(nH), uint32(nKV), uint32(hd), uint32(positions[r] + 1), uint32(start), uint32(nH / nKV), f32bits(scale), 0})
			disp(attnPl, bind(attnLy, q[r], lw.Attn.KCache.buf, lw.Attn.VCache.buf, cv, noSinks, ap, noHasSink), uint32(nH), 1)
			ctxv[r] = cv
		}
		attnOut := tiledProj(ctxv, lw.Attn.OProj)
		for r := range xd {
			residual(xd[r], attnOut[r])
		}
		xn2 := make([]*wgpu.Buffer, M)
		for r := range xn2 {
			xn2[r] = rms(xd[r], lw.MLPNorm.buf)
		}
		gate := tiledProj(xn2, lw.Gate)
		up := tiledProj(xn2, lw.Up)
		mid := make([]*wgpu.Buffer, M)
		for r := range xd {
			md := storF(inter)
			sp := uni([]uint32{uint32(inter), 0, 0, 0})
			disp(c.swigluPipeline, bind(c.swigluLayout, gate[r], up[r], md, sp), uint32(inter+63)/64, 1)
			mid[r] = md
		}
		down := tiledProj(mid, lw.Down)
		for r := range xd {
			residual(xd[r], down[r])
		}
	}

	// Final norm + LM head on the LAST row ONLY (matches decoder/forwardn.go's
	// prefillLogits: the other M-1 rows' logits are never needed for prefill).
	xnLast := rms(xd[M-1], m.FinalNorm.buf)
	logitsBuf := tiledProj([]*wgpu.Buffer{xnLast}, m.LMHead)[0]

	if buildErr != nil {
		return nil, buildErr
	}

	vocab := m.LMHead.rows
	stag, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(vocab * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	keepBuf(stag)
	if err := enc.TryCopyBufferToBuffer(logitsBuf, 0, stag, 0, uint64(vocab*4)); err != nil {
		return nil, err
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	c.queue.Submit(cmd)

	status := wgpu.MapAsyncStatus(0)
	if err := stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(vocab*4), func(s wgpu.MapAsyncStatus) { status = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	if status != wgpu.MapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: PrefillLastW8A8 map failed: %v", status)
	}
	out := make([]float32, vocab)
	copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(vocab*4))))
	if err := stag.TryUnmap(); err != nil {
		return nil, err
	}
	return out, nil
}
