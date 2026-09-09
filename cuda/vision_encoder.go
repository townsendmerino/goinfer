//go:build cuda

// vision_encoder.go — resident CUDA SigLIP vision tower (P6, docs/multimodal.md's "P6's other
// half"). A genuinely separate, lightweight device — NOT a *cudaResident (no KV cache, no
// text-decoder state, no layer-fusion admission) — mirroring gpu/vision_encoder.go's shape
// (WebGPU's own resident tower) but composed almost entirely from kernels this session's own
// text-decoder prefill work already shipped:
//
//   - attention: attn_img_batched (cuda/attn_img_prefill.cu), called with imgStart=0, imgEnd=M —
//     every row then satisfies imgStart<=pos<imgEnd, so nKeys=M and winStart=0 for every row: full
//     bidirectional attention over all M patches, reachable via parameters alone, no new kernel.
//   - projections: bW8-shaped batched int8 GEMV (gemv_w8a8_batched.ptx), already M-batched and
//     already used for ordinary int8 text prefill — no new GEMM kernel.
//   - patch-embed / attention-output requantize: quant_vec_batched (prefill_batched.ptx) — the
//     SAME plain (no-norm) batched int8 quantize the text decoder's own segB ctx-quant step uses.
//
// What genuinely IS new (SigLIP uses LayerNorm, not RMSNorm, and a plain non-gated GELU-tanh MLP —
// neither exists anywhere else in this codebase): layernorm_quant.cu, gelu_quant.cu.
//
// Own device/context/goroutine — CUDA contexts are thread-affine, so this mirrors cudaResident's
// own reqCh/ackCh/LockOSThread pattern (cuda/backend.go's BuildResident) rather than reusing it.
package cuda

import (
	"fmt"
	"math"
	"runtime"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/vision"
)

// visionLayer holds one SigLIP transformer layer's device-resident weights.
type visionLayer struct {
	ln1w, ln1b     Buffer
	qw, kw, vw, ow cudaWQ
	qb, kb, vb, ob Buffer
	ln2w, ln2b     Buffer
	fc1w, fc2w     cudaWQ
	fc1b, fc2b     Buffer
}

// VisionEncoder is a device-resident SigLIP forward, uploaded once from vision.GPUWeights.
type VisionEncoder struct {
	dev   *Device
	reqCh chan func() error
	ackCh chan error

	bLayerNormQuant Pipeline // layernorm_quant_batched
	bLayerNormF32   Pipeline // layernorm_f32_batched (the tower's final post-LN)
	bGeluQuant      Pipeline // gelu_quant_batched
	bQuant          Pipeline // quant_vec_batched (prefill_batched.ptx) — plain int8 quantize
	bW8             Pipeline // gemv_w8a8_batched — every linear projection
	bAttnImg        Pipeline // attn_img_batched — full bidirectional attention, imgStart=0/imgEnd=M

	hidden, inter, numLayers, numHeads, headDim int
	numPatches, cpp                             int
	eps                                         float32

	patchW cudaWQ // [hidden, C*P*P], quantized on the Go side (GPUWeights keeps it f32)
	patchB Buffer
	posEmb Buffer
	layers []visionLayer

	postLNw, postLNb Buffer
}

// do runs j on the pinned executor goroutine — every device call in this file goes through this,
// matching cudaResident's own reqCh/ackCh discipline (CUDA contexts are thread-affine).
func (ve *VisionEncoder) do(j func() error) error {
	ve.reqCh <- j
	return <-ve.ackCh
}

// NewVisionEncoder uploads a loaded SigLIP tower (vision.LoadEncoder(dir, quant=true) +
// e.GPUWeights()) to a fresh CUDA device and JITs the kernels this file needs. Declines with an
// error (never crashes) on a missing/broken driver or an unsupported shape — the caller
// (vision_register.go's registered factory) leaves the CPU path intact on any error, exactly as
// decoder.BuildResident does for the text decoder.
func NewVisionEncoder(w vision.GPUWeights) (ve *VisionEncoder, err error) {
	defer func() {
		if p := recover(); p != nil {
			ve, err = nil, fmt.Errorf("cuda vision: NewVisionEncoder recovered from panic: %v", p)
		}
	}()

	r := &VisionEncoder{
		hidden: w.Hidden, inter: w.Inter, numLayers: w.NumLayers, numHeads: w.NumHeads,
		headDim: w.HeadDim, numPatches: w.NumPatches, cpp: w.NumChannels * w.PatchSize * w.PatchSize,
		eps: w.Eps,
	}
	r.reqCh = make(chan func() error)
	r.ackCh = make(chan error)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		for j := range r.reqCh {
			r.ackCh <- runJob(j)
		}
	}()

	setupErr := r.do(func() error {
		var e error
		if r.dev, e = CreateSystemDefaultDevice(); e != nil {
			return e
		}
		loadFrom := func(ptx []byte, name string, dst *Pipeline) error {
			mod, e := r.dev.CompileLibrary(ptx)
			if e != nil {
				return fmt.Errorf("cuda vision: compile %s: %w", name, e)
			}
			p, e := r.dev.NewComputePipeline(mod, name)
			if e != nil {
				return fmt.Errorf("cuda vision: pipeline %s: %w", name, e)
			}
			*dst = p
			return nil
		}
		for _, k := range []struct {
			ptx  []byte
			name string
			dst  *Pipeline
		}{
			{layernormQuantPTX, "layernorm_quant_batched", &r.bLayerNormQuant},
			{layernormQuantPTX, "layernorm_f32_batched", &r.bLayerNormF32},
			{geluQuantPTX, "gelu_quant_batched", &r.bGeluQuant},
			{prefillBatchedPTX, "quant_vec_batched", &r.bQuant},
			{gemvW8BatchedPTX, "gemv_w8a8_batched", &r.bW8},
			{attnImgPrefillPTX, "attn_img_batched", &r.bAttnImg},
		} {
			if e := loadFrom(k.ptx, k.name, k.dst); e != nil {
				return e
			}
		}

		q8, sc := linalg.QuantizeRowsInt8(w.PatchW, r.hidden, r.cpp)
		r.patchW = r.upW8(q8, sc, r.hidden, r.cpp)
		r.patchB = r.up32(w.PatchB)
		r.posEmb = r.up32(w.PosEmb)
		r.postLNw = r.up32(w.PostLNw)
		r.postLNb = r.up32(w.PostLNb)

		r.layers = make([]visionLayer, len(w.Layers))
		for i := range w.Layers {
			l := &w.Layers[i]
			r.layers[i] = visionLayer{
				ln1w: r.up32(l.LN1w), ln1b: r.up32(l.LN1b),
				qw: r.upGPUMat(l.Qw), kw: r.upGPUMat(l.Kw), vw: r.upGPUMat(l.Vw), ow: r.upGPUMat(l.Ow),
				qb: r.up32(l.Qb), kb: r.up32(l.Kb), vb: r.up32(l.Vb), ob: r.up32(l.Ob),
				ln2w: r.up32(l.LN2w), ln2b: r.up32(l.LN2b),
				fc1w: r.upGPUMat(l.FC1w), fc2w: r.upGPUMat(l.FC2w),
				fc1b: r.up32(l.FC1b), fc2b: r.up32(l.FC2b),
			}
		}
		return nil
	})
	if setupErr != nil {
		r.Close()
		return nil, setupErr
	}
	return r, nil
}

func (r *VisionEncoder) up32(v []float32) Buffer {
	b := gpu.NewBufferLenOf[float32](r.dev, len(v))
	if err := gpu.Upload(b, v); err != nil {
		panic(err) // caught by NewVisionEncoder's recover; setup-time only
	}
	return b
}

func (r *VisionEncoder) upW8(q8 []int8, scales []float32, N, K int) cudaWQ {
	wpk := packI8(q8, N, K)
	b := gpu.NewBufferLenOf[uint32](r.dev, len(wpk))
	if err := gpu.Upload(b, wpk); err != nil {
		panic(err)
	}
	return cudaWQ{kind: "int8", W: b, ws: r.up32(scales), N: N, K: K}
}

func (r *VisionEncoder) upGPUMat(m vision.GPUMat) cudaWQ {
	return r.upW8(m.Q, m.Scales, m.Rows, m.Cols)
}

// af allocates an M*n float32 device buffer (scratch — no upload).
func (r *VisionEncoder) af(n int) Buffer { return gpu.NewBufferLenOf[float32](r.dev, n) }
func (r *VisionEncoder) ai(n int) Buffer { return gpu.NewBufferLenOf[int32](r.dev, n) }

// bGemv projects a[M,K] (already int8-quantized, scale as) through wt[N,K] + bias into dst[M,N].
// The plain int8 half of cuda/prefill.go's bGemvB — this tower never needs int4.
func (r *VisionEncoder) bGemv(q *Queue, wt cudaWQ, a, as Buffer, bias Buffer, dst Buffer, M int) error {
	cfg := LaunchConfig{GridX: uint32((wt.N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
	return q.Launch(r.bW8, cfg, Arg(wt.W), Arg(a), Arg(wt.ws), Arg(as), Arg(bias),
		gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/4)), gpu.ArgValue(int32(M)),
		Arg(dst), gpu.ArgValue(int32(0)))
}

// quant batches x[M,N] (f32) into q/scale (int8, plain — no norm). Mirrors cuda/prefill.go's own
// bQuant (r.bQuant field there, same underlying kernel).
func (r *VisionEncoder) quant(q *Queue, x Buffer, N int, qOut, sOut Buffer, M int) error {
	return q.Launch(r.bQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
		Arg(x), gpu.ArgValue(int32(N)), Arg(qOut), Arg(sOut), gpu.ArgValue(int32(M)))
}

func (r *VisionEncoder) layerNormQuant(q *Queue, x Buffer, w, b Buffer, N int, qOut, sOut Buffer, M int) error {
	return q.Launch(r.bLayerNormQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((256 + N) * 4)},
		Arg(x), Arg(w), Arg(b), gpu.ArgValue(int32(N)), gpu.ArgValue(r.eps), Arg(qOut), Arg(sOut))
}

func (r *VisionEncoder) layerNormF32(q *Queue, x Buffer, w, b Buffer, N, M int) error {
	return q.Launch(r.bLayerNormF32, LaunchConfig{GridX: 1, GridY: uint32(M), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
		Arg(x), Arg(w), Arg(b), gpu.ArgValue(int32(N)), gpu.ArgValue(r.eps), gpu.ArgValue(int32(M)))
}

func (r *VisionEncoder) geluQuant(q *Queue, x Buffer, N int, qOut, sOut Buffer, M int) error {
	return q.Launch(r.bGeluQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((256 + N) * 4)},
		Arg(x), gpu.ArgValue(int32(N)), Arg(qOut), Arg(sOut))
}

// residual: x += y (elementwise), same shape as cuda/glue.cu's own `residual` kernel — but that
// kernel lives in glue.ptx, a module this tower does not load (it never touches the text decoder's
// glue kernels other than the shared quant_vec_batched). Reuses quant_vec_batched's own module
// (prefill_batched.ptx) is not possible for a plain add, so this is a tiny host-side loop over a
// download+add+upload instead — the residual add is O(hidden) per row, not the bottleneck.
//
// NOT a new kernel: see ForwardPatches's own comment at each residual site for why a device-side
// add was skipped for v1 (small, and correctness-first).

// ForwardPatches runs one image's patches through the full resident tower. patches is
// [NumPatches * (C*P*P)] (vision.Encoder.GridPatches' own output — the CPU im2col step stays on
// the host, matching gpu/vision_encoder.go's identical split).
func (r *VisionEncoder) ForwardPatches(patches []float32) ([]float32, error) {
	M, hidden, inter, hd, nH := r.numPatches, r.hidden, r.inter, r.headDim, r.numHeads
	qDim := nH * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	if len(patches) != M*r.cpp {
		return nil, fmt.Errorf("cuda vision: patches len %d, want %d (%d patches x %d)", len(patches), M*r.cpp, M, r.cpp)
	}

	var out []float32
	err := r.do(func() error {
		q := r.dev.NewCommandQueue()

		pd := r.af(M * r.cpp)
		if e := gpu.Upload(pd, patches); e != nil {
			return e
		}
		pq, ps := r.ai(M*r.cpp/4), r.af(M)
		if e := r.quant(&q, pd, r.cpp, pq, ps, M); e != nil {
			return e
		}
		h := r.af(M * hidden)
		if e := r.bGemv(&q, r.patchW, pq, ps, r.patchB, h, M); e != nil {
			return e
		}
		if e := q.Sync(); e != nil {
			return e
		}
		// posEmb add — PER-PATCH (posEmb is [NumPatches,Hidden], the same full size as h), NOT a
		// broadcast bias — addInPlaceHost (the plain full-array elementwise add), not addRowsHost
		// (which broadcasts one [hidden] row to every patch — right for patchB, wrong here: it
		// would silently add row 0's positional embedding to every patch, correct only for patch
		// 0 by coincidence). Small enough for a host round-trip at v1 (correctness first, matching
		// this file's residual-add note).
		if e := addInPlaceHost(&q, h, r.posEmb, M*hidden); e != nil {
			return e
		}

		aq, as := r.ai(M*hidden/4), r.af(M)
		qb, kb, vb := r.af(M*qDim), r.af(M*qDim), r.af(M*qDim)
		attnOut := r.af(M * qDim)
		aoq, aos := r.ai(M*qDim/4), r.af(M)
		oProj := r.af(M * hidden)
		fq, fs := r.ai(M*hidden/4), r.af(M)
		g := r.af(M * inter)
		gq, gs := r.ai(M*inter/4), r.af(M)
		fc2Out := r.af(M * hidden)

		for li := range r.layers {
			L := &r.layers[li]
			if e := r.layerNormQuant(&q, h, L.ln1w, L.ln1b, hidden, aq, as, M); e != nil {
				return e
			}
			if e := r.bGemv(&q, L.qw, aq, as, L.qb, qb, M); e != nil {
				return e
			}
			if e := r.bGemv(&q, L.kw, aq, as, L.kb, kb, M); e != nil {
				return e
			}
			if e := r.bGemv(&q, L.vw, aq, as, L.vb, vb, M); e != nil {
				return e
			}
			// attn_img_batched, imgStart=0/imgEnd=M: full bidirectional attention over all M
			// patches — see this file's header for the derivation. kb/vb ARE the "KV cache"
			// (row-major [M,qDim], position=row index=startPos+m with startPos=0) — no separate
			// store step, unlike the text decoder's rope kernel which writes q/k/v INTO r.kc/r.vc
			// as a side effect; this tower has no rope at all, so qb/kb/vb already sit where the
			// attention kernel expects to read them.
			maxNWin := M
			if e := q.Launch(r.bAttnImg, LaunchConfig{GridX: uint32(nH), GridY: uint32(M), GridZ: 1,
				BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((maxNWin + 128) * 4)},
				Arg(qb), Arg(kb), Arg(vb), gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nH)),
				gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(0)), gpu.ArgValue(scale),
				gpu.ArgValue(int32(0)), gpu.ArgValue(int32(M)), Arg(attnOut), ArgNull(),
				gpu.ArgValue(int32(0)), gpu.ArgValue(int32(M))); e != nil {
				return e
			}
			if e := r.quant(&q, attnOut, qDim, aoq, aos, M); e != nil {
				return e
			}
			if e := r.bGemv(&q, L.ow, aoq, aos, L.ob, oProj, M); e != nil {
				return e
			}
			if e := q.Sync(); e != nil {
				return e
			}
			if e := addInPlaceHost(&q, h, oProj, M*hidden); e != nil { // residual
				return e
			}
			if e := r.layerNormQuant(&q, h, L.ln2w, L.ln2b, hidden, fq, fs, M); e != nil {
				return e
			}
			if e := r.bGemv(&q, L.fc1w, fq, fs, L.fc1b, g, M); e != nil {
				return e
			}
			if e := r.geluQuant(&q, g, inter, gq, gs, M); e != nil {
				return e
			}
			if e := r.bGemv(&q, L.fc2w, gq, gs, L.fc2b, fc2Out, M); e != nil {
				return e
			}
			if e := q.Sync(); e != nil {
				return e
			}
			if e := addInPlaceHost(&q, h, fc2Out, M*hidden); e != nil { // residual
				return e
			}
		}
		if e := r.layerNormF32(&q, h, r.postLNw, r.postLNb, hidden, M); e != nil {
			return e
		}
		if e := q.Sync(); e != nil {
			return e
		}
		out = make([]float32, M*hidden)
		return gpu.Download(h, out)
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// addInPlaceHost: x += y (elementwise, full arrays of matched length n — NOT a broadcast; every
// caller here, including the posEmb add, needs a same-shape add). Correctness-first v1 via a host
// round-trip (small — O(M*hidden) per call, not the per-layer GEMV/attention bottleneck). A
// device-side elementwise-add kernel is a natural, cheap follow-on if profiling ever shows this
// matters; not attempted here (P6's own scope is the tower's GEMV/attention cost, untouched by
// this).
func addInPlaceHost(q *Queue, x, y Buffer, n int) error {
	if e := q.Sync(); e != nil {
		return e
	}
	xh := make([]float32, n)
	if e := gpu.Download(x, xh); e != nil {
		return e
	}
	yh := make([]float32, n)
	if e := gpu.Download(y, yh); e != nil {
		return e
	}
	for i := range xh {
		xh[i] += yh[i]
	}
	return gpu.Upload(x, xh)
}

// Close tears down the device and its executor goroutine.
func (r *VisionEncoder) Close() {
	if r.reqCh != nil {
		close(r.reqCh)
		r.reqCh = nil
	}
	if r.dev != nil {
		r.dev.ReleaseAll()
		r.dev = nil
	}
}
