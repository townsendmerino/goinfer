//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/linalg"
	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/backend/cuda/internal/launchgeom"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

func init() {
	decoder.RegisterBackend("cuda", func() (decoder.Backend, error) {
		return &cudaBackend{}, nil
	})
	decoder.RegisterMemoryProbe("cuda", freeVRAM)
}

// freeVRAM is the CUDA memory probe: the driver's free VRAM, in bytes. A live query, unlike Metal's registered probe: CUDA
// VRAM is a separate pool from host RAM, so the driver's free figure is reliable. It creates a throwaway device with no
// kernels loaded and releases it right after the query (the bare create-query-release shape of cuda/alloc_floor_test.go),
// not the persistent executor cudaResident needs.
func freeVRAM() (int64, bool) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		return 0, false
	}
	defer dev.ReleaseObjects()
	free, _, err := dev.Context().MemInfo()
	if err != nil {
		return 0, false
	}
	return int64(free), true
}

// Compile-time seam checks: cudaBackend must satisfy the decoder's backend +
// residency interfaces (catches signature drift against decoder/residency.go).
var (
	_ decoder.Backend          = (*cudaBackend)(nil)
	_ decoder.ResidencyBackend = (*cudaBackend)(nil)
	_ decoder.ResidentForward  = (*cudaResident)(nil)
	_ decoder.ResidentAdapter  = (*cudaResident)(nil)
)

// cudaBackend implements decoder.Backend + decoder.ResidencyBackend.
type cudaBackend struct {
	resident *cudaResident // set by BuildResident; shut down in Close
}

func (b *cudaBackend) Name() string { return "cuda" }

// MatmulBT is the CPU fallback (the shared SIMD linalg kernels): CUDA has no partial GPU path, so a model the resident
// runner does not admit, or a box with no driver, runs entirely on the CPU, and DecodePath says so.
func (b *cudaBackend) MatmulBT(a, bmat, dst []float32, M, K, N int) {
	linalg.MatmulBT(a, bmat, dst, M, K, N)
}

// layerFusable reports whether one layer permits the fused super-kernels. fQKV always reads Q/K/V, so those must be
// int4. fGU also reads gate/up/down as int4+f16-scales but runs only on dense-FFN layers: MoE layers take moeMLP and
// leave g/u/d unpacked, so they are exempt (requiring g/u/d there would strip fQKV from every MoE model), while skipping
// the check on a dense int8 gate/up layer would pass fGU a nil ws16 and crash the executor.
func layerFusable(qkvInt4, moe, guInt4 bool) bool {
	return qkvInt4 && (moe || guInt4)
}

// kquantKeepInt8ForTest makes BuildResident upload every native Q5_K / Q6_K weight's int8 copy beside it (cudaResident.kquantAlt),
// so one resident can run both arms of KS-O1's A/B. Set only by a test, before the load.
var kquantKeepInt8ForTest bool

// hlayer is one layer's host plan in BuildResident: its packed weights and side tables, and the layer kind.
type hlayer struct {
	q, k, v, o, g, u, d       hostW
	pleGate, pleProj          hostW     // Gemma 4 E-model per-layer embedding branch (S1 on CUDA); zero unless pleP > 0
	postPLENorm               []float32 // the PLE branch's post-norm weight [hidden]
	qb, kb, vb                []float32
	ob                        []float32 // attention output-projection bias (GPT-2 / gpt-oss)
	preNorm, postNorm         []float32
	qNorm, kNorm              []float32
	postAttnNorm, postMLPNorm []float32 // Gemma sandwich (nil unless the arch declares it)
	invFreq                   []float32 // per-layer RoPE table
	window                    int32
	hasBias                   bool
	hasOBias                  bool

	// MoE, per layer: GLM/DeepSeek run dense for the first FirstKDense layers and route
	// after, so this is keyed off the layer's own Experts (as decoder/mlp.go does), not
	// off the arch.
	isMoE            bool
	expGU, expDown   hostW
	router, routerBs []float32
	hasShared        bool
	shGU, shDown     hostW
	shGate           hostW // [1, hidden] sigmoid gate (Qwen-MoE); empty ⇒ ungated (GLM/DeepSeek)

	// Gemma-4 enable_moe_block (parallel dense‖MoE). g4moe routes the upload to gemma4MoeMLP's
	// fields: the dense branch reuses g/u/d (packed from mlpGate/up/down), the router reuses
	// router (RouterProjScaled, f32) + routerBs (zeros), the experts reuse expGU/expDown.
	g4moe                                               bool
	g4preFFN, g4postFFN1, g4preFFN2, g4postFFN2, g4post []float32
	perExpertScale                                      []float32
	layerScalar                                         float32

	// mixer and ffn are this layer's operations from the program (cudaMixerOf): its sequence mixer, then a feed-forward block when ffn.
	// Nothing else decides which kind of layer this is.
	mixer decoder.ResidentMixer
	ffn   bool
	// steps is the layer's fusion-table steps (cudaStepsOf); its sublayers are checked against the loaded blocks at build.
	steps []cudaStep

	// Gated-DeltaNet (qwen3_5_moe / qwen3_next / qwen3_5). MixerDeltaNet layers carry NO
	// q/k/v/o and no KV cache — the recurrence's fixed-size state is the whole history.
	// qGate marks the family's SOFTMAX layers, whose q_proj is double width.
	qGate                                 bool
	dnQKV, dnZ, dnOut, dnB, dnA           hostW
	dnConvW, dnDtBias, dnNegExpA, dnNormW []float32
	dnBetaScale, dnONormEps               float32 // deltaNetGateSettings

	// MLA (DeepSeek / Kimi, FeatMLA)
	mlaQA, mlaQB, mlaQ, mlaKVA hostW
	mlaQANorm, mlaKVANorm      []float32
	mlaWUK, mlaWUV             []float32

	// Mamba-2 (Nemotron-H, OP2b): the packed projections and small tables of a MixerMamba layer.
	mamba hostMamba
}

// BuildResident builds a resident CUDA decoder from a loaded Model: host-packs the int4/int8/f32 projections, spawns a
// LockOSThread-pinned CUDA executor, and uploads weights and KV scratch once, returning a *cudaResident. It declines
// (ok=false, no crash) when the driver is absent or a projection shape is not residency-compatible; the decoder then
// uses the staged/CPU path.
func (b *cudaBackend) BuildResident(m *decoder.Model) (rf decoder.ResidentForward, ok bool, err error) {
	return b.buildResident(m, false)
}

// moeActG32DeclineOffForTest lets a MoE model with per-32 activations past its decline (the planted defect of
// TestBackendAdmissionCensus_cuda_plant).
var moeActG32DeclineOffForTest bool

// planOnlySlack is buildResident's answer in plan-only mode: the allocation rounding over the buffers the build would hold for
// its weights, computed by the build's own planning before any allocation.
type planOnlySlack int64

func (planOnlySlack) Error() string { return "cuda: plan only" }

// buildResident is BuildResident; with planOnly it plans the weights, validating every expert instead of packing it, and returns
// planOnlySlack before it allocates anything.
func (b *cudaBackend) buildResident(m *decoder.Model, planOnly bool) (rf decoder.ResidentForward, ok bool, err error) {
	return b.buildResidentMode(m, planOnly, nil)
}

// buildResidentMode is buildResident, or with mp non-nil the memory plan (cudaBackend.planMemory): the build's planning past
// plan-only's stop, its memory decisions taken over mp's readings and written to mp, and a return before any device is opened.
func (b *cudaBackend) buildResidentMode(m *decoder.Model, planOnly bool, mp *memPlanRun) (rf decoder.ResidentForward, ok bool, err error) {
	plans := planOnly || mp != nil
	// Never crash the process on a missing/broken driver: recover → decline → fallback.
	defer func() {
		if p := recover(); p != nil {
			rf, ok, err = nil, false, fmt.Errorf("cuda: BuildResident recovered from panic: %v", p)
		}
	}()

	// The decline reason is printed unconditionally, not behind a debug flag: declining moves the whole forward to CPU, and
	// the runtime must name why it is off the fast path. It travels as a typed decline (decoder.DeclineResident): decoder's
	// load path prints it once and records it as the model's ResidentDecline, which DecodePath and `serve check` report.
	declined := func(e error) (decoder.ResidentForward, bool, error) {
		// The reason names the cause and what to try; a recovered panic's stack goes to stderr beside it.
		reason, detail := declineAdvice(e.Error(), m.MoECacheExperts())
		if detail != "" {
			logs.Infof("[cuda] resident build failure, detail:\n%s", detail)
		}
		return nil, false, decoder.DeclineResident("cuda: %s", reason)
	}

	// stopped turns a build phase's error into the build's return: a decline goes through declined, anything else is returned as it is.
	stopped := func(err error) (decoder.ResidentForward, bool, error) {
		var d buildDecline
		if errors.As(err, &d) {
			return declined(d.error)
		}
		return nil, false, err
	}

	s := &buildState{m: m, plans: plans}
	defer func() { _ = s.kq.Close() }()
	if err := s.admit(); err != nil {
		return stopped(err)
	}
	if err := s.planLayers(); err != nil {
		return stopped(err)
	}
	H, I, vocab := s.H, s.I, s.vocab
	nE, topK, moeInter, sharedInter := s.nE, s.topK, s.moeInter, s.sharedInter
	isMoE := s.isMoE
	postOnly, parallelBlock := s.postOnly, s.parallelBlock
	hls := s.hls

	// ---- resident + pinned executor ----
	r, err := s.newResident(planOnly, mp)
	if err != nil {
		return stopped(err)
	}

	// setup job: create the context on the pinned thread, JIT kernels, upload everything.
	setupErr := r.do(func() error {
		defer r.onLine(mem.LineScratch)()
		var e error
		if e = s.bindKernels(r); e != nil {
			return e
		}
		if !r.openedForPlan {
			r.stream = r.dev.NewCommandQueue()
		}

		if e = s.bindFamilyState(r); e != nil {
			return e
		}
		if e = s.uploadLayers(r); e != nil {
			return e
		}
		if e = s.checkSublayers(r); e != nil {
			return e
		}
		if e = s.sizeAttnScratch(r); e != nil {
			return e
		}
		maxQDim, maxHd := s.maxQDim, s.maxHd
		if e = s.allocKV(r); e != nil {
			return e
		}
		if r.actG32 && maxQDim%32 != 0 {
			return fmt.Errorf("per-32 activations need the attention width (%d) to be a multiple of 32", maxQDim)
		}
		r.cctx, r.cSc, r.cq = r.af(maxQDim), r.af(r.actScaleLen(maxQDim)), r.ai(maxQDim/4)
		// v_norm (Gemma 4: every K/V-owning layer; K=V layers also derive V = v_norm(k)) reuses qk_norm with a UNIT
		// weight [maxHd] (so x*inv*w = x*inv, scale-less; addOne=0). Allocate it only when needed.
		for l := range r.layers {
			if r.layers[l].vNorm {
				ones := make([]float32, maxHd)
				for i := range ones {
					ones[i] = 1.0
				}
				r.vNormUnit = r.upOn(mem.LineDense, ones)
				break
			}
		}
		r.oO = r.af(H)
		r.mSc, r.mq = r.af(r.actScaleLen(H)), r.ai(H/4)
		if r.layerNorm {
			r.zeroBias = r.af(H)
			if err := gpu.Upload(r.zeroBias, make([]float32, H)); err != nil {
				return err
			}
		}
		// Dense-FFN scratch, only if the model has a dense FFN: a model whose every layer is routed reports intermediate_size 0
		// (Qwen3.6-35B-A3B's config omits the key), and these would be 0-byte allocations, which this driver rejects as "invalid
		// length". The dense branch of segBFFN is unreachable for such a model. Scratch is sized to the widest dense layer: an
		// E-model's layers differ, and the launches take each layer's own width (cudaLayer.ffnI).
		maxI := I
		for l := range r.layers {
			maxI = max(maxI, r.layers[l].ffnI)
		}
		if maxI > 0 {
			r.gO, r.uO = r.af(maxI), r.af(maxI)
			r.dSc, r.dScr, r.dq = r.af(r.actScaleLen(maxI)), r.af(maxI), r.ai(maxI/4)
		} else if !isMoE {
			return fmt.Errorf("cuda: intermediate_size is 0 on a DENSE model — no FFN to run")
		}
		if r.moe {
			// Sized to the MoE expert width, not the dense one (Mellum's moe_intermediate_size
			// differs from intermediate_size).
			r.rLogits, r.rIdx, r.rWgt = r.af(nE), r.au32(topK), r.af(topK)
			if r.cacheExperts { // C′: per-token slot-id buffer (uploaded per layer) + readback scratch
				r.slotIdx = r.au32(topK)
				r.hostIdx = make([]uint32, topK)
				r.hostSlot = make([]uint32, topK)
				if hb, e := gpu.NewHostBufferOn[uint8](r.dev, mem.LineScratch, topK*4); e != nil {
					return e
				} else {
					r.slotIdxHost = hb
				}
			}
			r.moeGU = r.af(2 * moeInter)
			r.moeSc, r.moeScr, r.moeQ = r.af(1), r.af(moeInter), r.ai(moeInter/4)
			if sharedInter > 0 {
				r.shGUout = r.af(2 * sharedInter)
				r.shSc, r.shScr, r.shQ = r.af(1), r.af(sharedInter), r.ai(sharedInter/4)
				r.shDownOut = r.af(H)
				r.shGl = r.af(1) // the sigmoid gate logit; allocated unconditionally (one float)
			}
		}
		if r.gemma4Moe { // parallel dense‖MoE branch scratch
			r.g4x1, r.g4x2, r.g4rn = r.af(H), r.af(H), r.af(H)
		}
		if why := r.floatActDecline(); why == "" {
			fmod, e2 := r.dev.CompileLibrary(gemvW4F32PTX)
			if e2 != nil {
				return e2
			}
			for _, f := range []struct {
				dst  *Pipeline
				name string
			}{
				{&r.fW4F32, "gemv_w4f32"}, {&r.fMoEGemvF32, "gemv_w4f32_moe"}, {&r.fMoEWaccF32, "gemv_w4f32_moe_wacc"}, {&r.fGluF32, "glu_f32"},
			} {
				if *f.dst, e = r.dev.NewComputePipeline(fmod, f.name); e != nil {
					return e
				}
			}
			r.xF, r.dF, r.moeGLUF = r.af(H), r.af(max(maxI, 1)), r.af(moeInter)
			r.floatAct = true
			logs.Debugf("cuda: decode activations: f32 into every int4 projection (gemv_w4f32)")
		} else if r.gemma4Moe {
			logs.Debugf("cuda: decode activations: int8 (W4A8): %s", why)
		}
		r.dO, r.logits = r.af(H), r.af(vocab)
		r.argIdx, r.argVal = r.ai(1), r.af(1) // greedy fast-path readback (4 B vs 594 KB)
		r.topkOut = r.ai(2*topkMaxK + 2)      // sampled fast-path readback (~2 KB vs 594 KB)
		r.topkCV, r.topkCI, r.topkMx, r.topkPart = r.af(topkCap), r.ai(topkCap), r.af(1), r.af(topkThreads)
		r.topkCnt, r.topkDone = r.ai(1), r.ai(1)
		if e := gpu.Upload(r.topkCnt, []int32{0}); e != nil {
			return e
		}
		if e := gpu.Upload(r.topkDone, []int32{0}); e != nil {
			return e
		}
		nGb := launchgeom.GumbelBlocks(vocab)
		r.gbKey, r.gbIdx, r.gbOut = r.af(nGb), r.ai(nGb), r.ai(1) // Gumbel-max sampling scratch (R7b)
		if hb, e := gpu.NewHostBufferOn[float32](r.dev, mem.LineScratch, vocab); e != nil {
			return e
		} else {
			r.logitsPinned, r.logitsHost = hb, hb.Slice()
		}
		// KV8a: the recurrent checkpoint comes out of what is left before the expert cache and the windowed grant are sized from it.
		if postPoolDoneForTest != nil {
			postPoolDoneForTest()
		}
		r.allocRecurrentCheckpoint()
		// C′: the core + KV + scratch are now up, so free VRAM reflects them — size the expert slot
		// cache to what actually fits (cap-and-log, never OOM) and allocate it.
		if r.cacheExperts {
			r.kvGrantRollback(r.expertCacheFloorBytes()) // before the cache is sized: the grant comes out of it, never below its minimum
		}
		if e := r.allocSlots(); e != nil {
			return e
		}
		if !r.cacheExperts {
			r.kvGrantRollback(0) // last: everything it must not cost is allocated
		}
		r.pk = nil // the packing staging blocks: the garbage the build leaves behind is a few blocks, not a packed copy
		return r.setupErr
	})
	if setupErr != nil {
		r.Close()
		// A silent decline suits a shape this backend does not implement (the staged path serves it, slower). It is wrong when
		// the operator explicitly asked for a resident context that does not fit: degrading quietly turns that into a latency
		// mystery under load, so errKVWontFit is a hard startup error naming the GB. Everything else keeps the decline.
		if errors.Is(setupErr, errKVWontFit) || errors.Is(setupErr, decoder.ErrLoadAborted) {
			return nil, false, setupErr // the tripwire stopping the load is the load's error, not a decline that carries on on the CPU
		}
		return declined(setupErr)
	}
	// The fused path needs int4 projections on the layers it runs on (layerFusable). Anything else falls back to the unfused
	// chain, which handles mixed quant correctly.
	r.fuseQKV = true
	for l := range hls {
		h := &hls[l]
		if !layerFusable(
			h.q.kind == "int4" && h.k.kind == "int4" && h.v.kind == "int4",
			h.isMoE,
			h.g.kind == "int4" && h.u.kind == "int4" && h.d.kind == "int4") {
			r.fuseQKV = false
			break
		}
	}
	if r.knobValue("GOINFER_CUDA_NO_FUSE") != "" {
		r.fuseQKV = false
	}
	if postOnly {
		// fQKV bakes a real pre-norm weight into its rmsnorm+quant dispatch, and a postOnly arch has none (Ly.preNorm is never
		// uploaded), so it takes the unfused segA chain, which quantizes the raw residual.
		r.fuseQKV = false
	}
	if parallelBlock {
		// fQKV goes straight from x to Q/K/V and never materializes the normed, quantized activation that segBFFN's MLP branch
		// reuses as its own input (FeatParallelBlock: segA's r.aq/r.aSc); the fused path has nowhere to hand it back.
		r.fuseQKV = false
	}
	if r.isMLA {
		r.fuseQKV = false
	}
	if r.actG32 {
		// fused_rms_qkv / fused_rms_gu quantize per vector internally.
		r.fuseQKV = false
		// Their per-32 twins (fused_rms_qkv_g32 / fused_rms_gu_g32) take any per-32 weight kind, per projection, with fuseQKV's
		// structural exclusions (no pre-norm, parallel block, MLA, gated q, MoE layers). Size-gated (fusedG32MaxHidden): every
		// block redoes the rmsnorm and per-32 quant, which grows with H while the launches saved matter less as the GEMVs grow.
		// Fusion paid off at H=1536 and lost at 3072 and 3584.
		r.fuseG32 = r.knobValue("GOINFER_CUDA_NO_FUSE") == "" && !postOnly && !parallelBlock && !r.isMLA &&
			H <= fusedG32MaxHidden
		for l := range hls {
			h := &hls[l]
			if h.isMoE || h.qGate || !g32Fusable(h.q.kind, h.k.kind, h.v.kind, h.g.kind, h.u.kind) {
				r.fuseG32 = false
				break
			}
		}
	}
	// CUDA graphs (GOINFER_CUDA_GRAPHS): capture each layer's static launch segments now that fuseQKV is final (segA/segB
	// branch on it), so decode replays them instead of re-issuing the launches. Off, launchToken is unchanged. Incompatible
	// with the g4cap diagnostic, which syncs inside a segment. admitGraphs gates it: replay is bit-exact only under
	// EXCLUSIVE_PROCESS tenancy or MPS, so a shared-GPU box under DEFAULT declines to the live path
	// (docs/cuda-graphs-investigation.md).
	// A windowed layer's K/V pointer moves with its base, so a captured graph would replay a stale one: graphs decline under windowed KV.
	r.graphs = os.Getenv("GOINFER_CUDA_GRAPHS") != "" && !r.g4cap && !r.isMLA && !r.kvWin && r.mamba == nil &&
		!(r.dnet != nil && r.KVSlots() > 1) // a captured segment holds one slot's window and state
	r.graphsSync = os.Getenv("GOINFER_CUDA_GRAPHS_SYNC") != "" // debug: serialize replays (bisect ordering hazards)
	r.graphMask = os.Getenv("GOINFER_CUDA_GRAPHS_ONLY")        // debug: replay only these segments (A/B/C), rest live
	if e := r.admitGraphs(); e != nil {
		r.Close()
		return declined(fmt.Errorf("cuda: %w", e))
	}
	// C′ compute/DMA overlap (see the overlap field). Not under graphs: a captured segB cannot record
	// the router event mid-segment and a captured segC cannot wait per miss.
	r.overlap = r.cacheExperts && !r.graphs && r.knobValue("GOINFER_MOE_DMA_OVERLAP") != "0"
	if r.overlap {
		if e := r.initOverlap(); e != nil {
			r.Close()
			return declined(fmt.Errorf("cuda: overlap: %w", e))
		}
	}
	// A model the batched prefill's guards batch has steps on every layer; a layer without would send it to the sequential
	// path with nothing said, so the build declines naming the layer.
	if !prefillStepsBuildChecksOffForTest && r.prefillFlagsDecline() == nil {
		for l := range r.layers {
			if len(r.layers[l].prefillSteps) == 0 {
				r.Close()
				return declined(fmt.Errorf("cuda: layer %d: the batched prefill's guards admit the model and the program gives the layer no batched steps", l))
			}
		}
	}
	// KS-B8: the default pass width, from what the build left free.
	if r.prefillReady {
		if e := r.do(func() error { r.choosePrefillWidth(); return nil }); e != nil {
			r.Close()
			return declined(fmt.Errorf("cuda: prefill width: %w", e))
		}
	}
	b.resident = r
	return r, true, nil
}

// kvPoolsDoneForTest and postPoolDoneForTest, when set (tests only), are called after the KV pools are allocated and before the
// self-sizing grants (the recurrent checkpoint, the expert slots, the windowed grant), so a test can hold postPoolBytes to what the
// build allocates between them (TestKSH6C_marginPriced).
var kvPoolsDoneForTest, postPoolDoneForTest func()

// routePoolOffForTest, when set (tests only), skips payRoutePool: H6C-S5's plant.
var routePoolOffForTest bool

// payRoutePool pays moe_route's deferred first-launch local-memory reservation before checkKVFits and allocSlots read free VRAM,
// so the context and the expert cache are sized net of it by construction rather than covered by a margin. moe_route declares
// per-thread local scratch, and the driver backs it for the device's occupancy the first time the kernel runs, not at module
// load: on the RTX 2070 SUPER the launch demands about 289 MB and retains about 138 MB. A MoE build whose context is planned pays
// it before the plan (openForPlan); every MoE build pays it here before the fit check, where, once the pool exists, it costs one
// launch and no memory. The startup self-test's own pool goes with its context, so it does not count
// (docs/tasks/parked/task-kernel-survey-2026-10.md, "KS-H6 build, review amendment").
//
// Naming moe_route is safe because the backing store is shared and sized by the largest kernel launched, and
// TestKernelLocalMemoryCensus enumerates every entry point in every embedded module and fails, naming this site, if another kernel
// declares more. route_gptoss does (4608 B/thread against 4416), so it is forced too wherever it is bound
// (TestRouteGptOssGrowsPoolPastMoERoute). This was measured with sequential single-stream launches, which is what goinfer does;
// concurrent streams would reopen whether the bound is the max or a sum. A dense build binds neither and pays nothing.
func (r *cudaResident) payRoutePool() error {
	defer r.onLine(mem.LineScratch)()
	if r.fRoute == (Pipeline{}) || routePoolOffForTest {
		return nil
	}
	// nE=1, k=1, nGroup=1 does the least work the kernel can do, on scratch released below: only the allocation side effect is
	// wanted (logits and bias are read uninitialised, the index and weight outputs discarded).
	lg, idx, wgt := gpu.NewBufferLenOfOn[float32](r.dev, r.line, 1), gpu.NewBufferLenOfOn[uint32](r.dev, r.line, 1), gpu.NewBufferLenOfOn[float32](r.dev, r.line, 1)
	defer func() { r.dev.ReleaseBuf(lg); r.dev.ReleaseBuf(idx); r.dev.ReleaseBuf(wgt) }()
	if e := r.stream.Launch(r.fRoute, onecfg(1, 0),
		Arg(lg), Arg(lg), Arg(idx), Arg(wgt),
		gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)),
		gpu.ArgValue(int32(0)), gpu.ArgValue(float32(1)),
		gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1))); e != nil {
		return fmt.Errorf("pre-sizing warm-up of moe_route failed: %w", e)
	}
	if r.gptOssRoute != (Pipeline{}) {
		// nE=1, k=1, with the same discarded outputs and the logits as the bias.
		if e := r.stream.Launch(r.gptOssRoute, onecfg(1, 0),
			Arg(lg), Arg(lg), Arg(idx), Arg(wgt),
			gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1))); e != nil {
			return fmt.Errorf("pre-sizing warm-up of route_gptoss failed: %w", e)
		}
	}
	// Synchronise: the reservation must be a fact before free VRAM is read, and an async launch would let the fit check read the
	// pre-launch figure.
	if e := r.stream.Sync(); e != nil {
		return fmt.Errorf("pre-sizing warm-up of moe_route failed to complete: %w", e)
	}
	return nil
}

// openForPlan is a MoE build's step before its plan, on the executor thread: it creates the build's device and stream, checks the
// weights fit (checkWeightsFit, against the margin set before it), binds moe_route (and route_gptoss on gpt-oss) from their own
// modules, and pays the local-memory pool (payRoutePool), so the plan's free reading is net of it. The setup job compiles those
// modules again with the rest of their kernels.
func (r *cudaResident) openForPlan(m *decoder.Model) error {
	var e error
	if r.dev, e = CreateSystemDefaultDevice(); e != nil {
		return e
	}
	if e := r.dev.BindCurrentThread(); e != nil {
		return e
	}
	r.stream, r.openedForPlan = r.dev.NewCommandQueue(), true
	if e := r.checkWeightsFit(m); e != nil {
		return e
	}
	if prePlanPoolOffForTest {
		return nil
	}
	mod, e := r.dev.CompileLibrary(moePTX)
	if e != nil {
		return e
	}
	if r.fRoute, e = r.dev.NewComputePipeline(mod, "moe_route"); e != nil {
		return e
	}
	if _, _, isGptOss := m.GptOssActResident(); isGptOss {
		gmod, ge := r.dev.CompileLibrary(gptOssActPTX)
		if ge != nil {
			return ge
		}
		if r.gptOssRoute, e = r.dev.NewComputePipeline(gmod, "route_gptoss"); e != nil {
			return e
		}
	}
	return r.payRoutePool()
}

// prePlanPoolOffForTest, when set (tests only), makes openForPlan skip the warm-up: H6C-S6's plant, the pool paid only before the
// fit check, after the plan has read free VRAM.
var prePlanPoolOffForTest bool

// cudaKVSlotsRequest is how many resident KV slots a build asks checkKVFits for: the model's request on this backend
// (decoder.Model.ResidentKVSlotsRequestFor), forced to 1 on expert streaming (MoECacheExperts), which takes whatever VRAM is
// left, so every extra KV slot is an expert-cache slot lost (docs/tasks/parked/task-concurrency-2026-09.md).
func cudaKVSlotsRequest(m *decoder.Model) int {
	if m.MoECacheExperts() {
		return 1
	}
	return m.ResidentKVSlotsRequestFor("cuda")
}

// cudaBackend and cudaResident implement Close() error and io.Closer, the same spelling of "free this GPU resource" as
// gpu and metal, so callers can write generic cleanup; a signature drift back to a no-error Close breaks the build here.
var (
	_ io.Closer = (*cudaBackend)(nil)
	_ io.Closer = (*cudaResident)(nil)
)

// Close releases the resident backend's GPU resources. Propagates the resident's Close error (best-
// effort teardown returns nil today, but the contract is honored so a future failing release surfaces).
func (b *cudaBackend) Close() error {
	if b.resident != nil {
		return b.resident.Close()
	}
	return nil
}

// cudaMixerOf is a layer's mixer and whether an FFN follows, from its operations (program.MixerOf), as the decoder's
// ResidentMixer the build and the layer walk switch on. ok is false for a mixer no resident runs, which the build declines,
// naming it.
func cudaMixerOf(ops []program.Op) (mixer decoder.ResidentMixer, ffn, ok bool) {
	x, ffn := program.MixerOf(ops)
	switch x {
	case program.MixerAttention:
		return decoder.MixerAttention, ffn, true
	case program.MixerDeltaNet:
		return decoder.MixerDeltaNet, ffn, true
	case program.MixerMamba:
		return decoder.MixerMamba, ffn, true
	case program.MixerNone:
		return decoder.MixerNone, ffn, true
	case program.MixerMLA:
		return decoder.MixerMLA, ffn, true
	}
	return 0, ffn, false
}

// modelDir is where the packed-experts file goes for a model loaded from path: a checkpoint directory itself, a file's directory, or "" when
// the model has no path, so the file is never written to the working directory by default.
func modelDir(path string) string {
	if path == "" {
		return ""
	}
	if fi, err := os.Stat(path); err == nil && fi.IsDir() {
		return path
	}
	return filepath.Dir(path)
}

// ctxExplicitFor reports whether m's resident context is the caller's own -ctx, which the build never trims and whose VRAM miss
// is errKVWontFit. A context the load-time fit guard pinned for an unrequested load is a ceiling (decoder.Model's
// ResidentContextPinned), so the build trims it to what is free, never below cudaCtxCapDefault, as it trims an unpinned one.
func ctxExplicitFor(m *decoder.Model) bool {
	return m.ResidentContextRequest() > 0 && m.ResidentContextPinned()
}
