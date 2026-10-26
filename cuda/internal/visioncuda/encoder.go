//go:build linux

// Package visioncuda is the native-CUDA vision.ResidentEncoder — a whole SigLIP
// tower running on the GPU, cgo-free (docs/task-native-gpu.md, Phase 3). It is the
// NVIDIA counterpart of goinfer's WebGPU resident encoder, and the vision analogue of
// what gpu/anncuda is for ANN: aikit's device substrate plus one real consumer.
//
// Importing this package is the opt-in. aikit's core `vision` never imports `gpu`, so
// the default build stays pure-Go CPU; this package's init() plugs the factory into
// vision.RegisterResident, and Encoder.EnableResident then routes Forward here.
//
// The tower is uploaded once (int8 matmul weights + f32 norms/biases) and every op runs
// on-device: the [np, hidden] residual stream never leaves the GPU between the patch
// embed and the final post-LayerNorm. Only the im2col patches go up and the last hidden
// state comes back.
//
// The forward mirrors vision/encoder.go's Forward op for op — same order, same
// formulations — because the gate is cosine ≈ 1.0 against that CPU tower. The kernels
// it dispatches (gpu/vit.cu) carry the arithmetic-matching detail.
package visioncuda

import (
	"fmt"
	"math"
	"sync"

	gpu "github.com/townsendmerino/goinfer/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/vision"
)

func init() {
	vision.RegisterResident(func(e *vision.Encoder) (vision.ResidentEncoder, error) {
		w, err := e.GPUWeights()
		if err != nil {
			return nil, err
		}
		return newEncoder(w)
	})
}

// mat is one resident W8A8 weight: int8 rows [Rows,Cols] plus per-row scales.
type mat struct {
	q, s       gpu.Buffer
	rows, cols int
}

// layer is one transformer block's resident weights.
type layer struct {
	ln1w, ln1b, ln2w, ln2b gpu.Buffer
	qb, kb, vb, ob         gpu.Buffer
	fc1b, fc2b             gpu.Buffer
	qw, kw, vw, ow         mat
	fc1w, fc2w             mat
}

// encoder is the device-resident SigLIP tower.
type encoder struct {
	dev *gpu.Device
	q   gpu.Queue
	k   gpu.ViT
	w   vision.GPUWeights

	patchW, patchB, posEmb gpu.Buffer
	postLNw, postLNb       gpu.Buffer
	layers                 []layer

	// Per-forward scratch, allocated once and reused across layers — every layer
	// shares one shape, so re-allocating per layer would churn hundreds of MB on a
	// real tower (the same finding vision's encScratch records for the CPU path).
	patches, h, n1, n2 gpu.Buffer
	qa, ka, va, att, o gpu.Buffer
	mid, mlp, out      gpu.Buffer
	qi8, qs            gpu.Buffer // quantized activation + per-row scale
	cpp                int
	graph              *gpu.Graph
	mu                 sync.Mutex
}

// newEncoder uploads the tower and allocates scratch. A device OOM surfaces from the
// gpu layer as a panic (MustBuf's loud-failure contract); recover it into an error so
// EnableResident declines and the caller keeps the CPU path.
func newEncoder(w vision.GPUWeights) (enc *encoder, err error) {
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		return nil, err
	}
	defer func() {
		if r := recover(); r != nil {
			dev.ReleaseObjects()
			enc, err = nil, fmt.Errorf("visioncuda: device upload failed: %v", r)
		}
	}()
	kern, err := dev.NewViT()
	if err != nil {
		dev.ReleaseObjects()
		return nil, err
	}
	np, hidden, inter := w.NumPatches, w.Hidden, w.Inter
	if np <= 0 || hidden <= 0 || inter <= 0 {
		dev.ReleaseObjects()
		return nil, fmt.Errorf("visioncuda: degenerate tower (np=%d hidden=%d inter=%d)", np, hidden, inter)
	}
	cpp := w.NumChannels * w.PatchSize * w.PatchSize

	e := &encoder{dev: dev, q: dev.NewCommandQueue(), k: kern, w: w, cpp: cpp}
	up := func(x []float32) gpu.Buffer { return gpu.NewBufferOf(dev, x) }
	upMat := func(m vision.GPUMat) mat {
		return mat{q: gpu.NewBufferOf(dev, m.Q), s: gpu.NewBufferOf(dev, m.Scales), rows: m.Rows, cols: m.Cols}
	}
	e.patchW, e.patchB, e.posEmb = up(w.PatchW), up(w.PatchB), up(w.PosEmb)
	e.postLNw, e.postLNb = up(w.PostLNw), up(w.PostLNb)
	e.layers = make([]layer, len(w.Layers))
	for i, L := range w.Layers {
		e.layers[i] = layer{
			ln1w: up(L.LN1w), ln1b: up(L.LN1b), ln2w: up(L.LN2w), ln2b: up(L.LN2b),
			qb: up(L.Qb), kb: up(L.Kb), vb: up(L.Vb), ob: up(L.Ob),
			fc1b: up(L.FC1b), fc2b: up(L.FC2b),
			qw: upMat(L.Qw), kw: upMat(L.Kw), vw: upMat(L.Vw), ow: upMat(L.Ow),
			fc1w: upMat(L.FC1w), fc2w: upMat(L.FC2w),
		}
	}
	// scratch
	f32 := func(n int) gpu.Buffer { return gpu.NewBufferLenOf[float32](dev, n) }
	e.patches = f32(np * cpp)
	e.h, e.n1, e.n2 = f32(np*hidden), f32(np*hidden), f32(np*hidden)
	e.qa, e.ka, e.va = f32(np*hidden), f32(np*hidden), f32(np*hidden)
	e.att, e.o, e.mlp, e.out = f32(np*hidden), f32(np*hidden), f32(np*hidden), f32(np*hidden)
	e.mid = f32(np * inter)
	wide := max(inter, hidden)
	e.qi8 = gpu.NewBufferLenOf[int8](dev, np*wide)
	e.qs = f32(np)

	// Capture the entire static forward tower into a CUDA Graph.
	// If graph capture succeeds, ForwardPatches replays all launches in one driver call.
	if g, err := e.q.Capture(e.forwardTower); err == nil {
		e.graph = g
	}
	return e, nil
}

// launch is the single dispatch point, so every kernel error is caught the same way.
func (e *encoder) launch(p gpu.Pipeline, cfg gpu.LaunchConfig, args ...gpu.KernelArg) error {
	return e.q.Launch(p, cfg, args...)
}

// quantRows quantizes src[M,K] into the shared qi8/qs scratch.
func (e *encoder) quantRows(src gpu.Buffer, M, K int) error {
	return e.launch(e.k.QuantRows, gpu.RowGrid(M),
		gpu.Arg(src), gpu.Arg(e.qi8), gpu.Arg(e.qs),
		gpu.ArgValue(int32(M)), gpu.ArgValue(int32(K)))
}

// projQuantizedBias runs gemm_w8a8_bias on already-quantized activation in qi8/qs.
// Falls back to gemm_w8a8_reg/tiled + add_bias if fused plan is unavailable.
func (e *encoder) projQuantizedBias(m mat, bias, dst gpu.Buffer, M int) error {
	K, N := m.cols, m.rows
	bp, bc := e.k.GEMMW8A8BiasPlan(M, N, K)
	if bp != (gpu.Pipeline{}) {
		return e.launch(bp, bc,
			gpu.Arg(e.qi8), gpu.Arg(e.qs), gpu.Arg(m.q), gpu.Arg(m.s), gpu.Arg(bias), gpu.Arg(dst),
			gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N)), gpu.ArgValue(int32(K)))
	}
	gp, gc := e.k.GEMMW8A8Plan(M, N, K)
	if err := e.launch(gp, gc,
		gpu.Arg(e.qi8), gpu.Arg(e.qs), gpu.Arg(m.q), gpu.Arg(m.s), gpu.Arg(dst),
		gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N)), gpu.ArgValue(int32(K))); err != nil {
		return err
	}
	return e.launch(e.k.AddBias, gpu.Grid1D(M*N, 256),
		gpu.Arg(dst), gpu.Arg(bias), gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N)))
}

// projBias is quantRows + projQuantizedBias, for projections with their own input.
func (e *encoder) projBias(src gpu.Buffer, m mat, bias, dst gpu.Buffer, M int) error {
	if err := e.quantRows(src, M, m.cols); err != nil {
		return err
	}
	return e.projQuantizedBias(m, bias, dst, M)
}

// projBiasAdd quantizes src, runs GEMM + bias + residual add accumulating directly into residual.
// Eliminates both add_bias and add_vec launches and avoids intermediate buffer round-trips.
func (e *encoder) projBiasAdd(src gpu.Buffer, m mat, bias, residual gpu.Buffer, M int) error {
	if err := e.quantRows(src, M, m.cols); err != nil {
		return err
	}
	K, N := m.cols, m.rows
	bap, bac := e.k.GEMMW8A8BiasAddPlan(M, N, K)
	if bap != (gpu.Pipeline{}) {
		return e.launch(bap, bac,
			gpu.Arg(e.qi8), gpu.Arg(e.qs), gpu.Arg(m.q), gpu.Arg(m.s), gpu.Arg(bias), gpu.Arg(residual),
			gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N)), gpu.ArgValue(int32(K)))
	}
	// Fallback to unfused
	gp, gc := e.k.GEMMW8A8Plan(M, N, K)
	if err := e.launch(gp, gc,
		gpu.Arg(e.qi8), gpu.Arg(e.qs), gpu.Arg(m.q), gpu.Arg(m.s), gpu.Arg(e.o),
		gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N)), gpu.ArgValue(int32(K))); err != nil {
		return err
	}
	if err := e.launch(e.k.AddBias, gpu.Grid1D(M*N, 256),
		gpu.Arg(e.o), gpu.Arg(bias), gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N))); err != nil {
		return err
	}
	return e.addVec(residual, e.o, M*N)
}

func (e *encoder) norm(src, w, b, dst gpu.Buffer, rows, dim int) error {
	return e.launch(e.k.LayerNorm, gpu.RowGrid(rows),
		gpu.Arg(src), gpu.Arg(w), gpu.Arg(b), gpu.Arg(dst),
		gpu.ArgValue(int32(rows)), gpu.ArgValue(int32(dim)), gpu.ArgValue(e.w.Eps))
}

func (e *encoder) addVec(dst, v gpu.Buffer, n int) error {
	return e.launch(e.k.AddVec, gpu.Grid1D(n, 256), gpu.Arg(dst), gpu.Arg(v), gpu.ArgValue(int32(n)))
}

// forwardTower enqueues the complete SigLIP forward sequence onto the command stream.
func (e *encoder) forwardTower() error {
	np, hidden, inter := e.w.NumPatches, e.w.Hidden, e.w.Inter
	nH, hd := e.w.NumHeads, e.w.HeadDim

	// patch embed: h = patches[np,cpp] · patchW[hidden,cpp]ᵀ + patchB + posEmb
	pep, pecfg := e.k.GEMMF32Plan(np, hidden, e.cpp)
	if err := e.launch(pep, pecfg,
		gpu.Arg(e.patches), gpu.Arg(e.patchW), gpu.Arg(e.h),
		gpu.ArgValue(int32(np)), gpu.ArgValue(int32(hidden)), gpu.ArgValue(int32(e.cpp))); err != nil {
		return err
	}
	if err := e.launch(e.k.AddBias, gpu.Grid1D(np*hidden, 256),
		gpu.Arg(e.h), gpu.Arg(e.patchB), gpu.ArgValue(int32(np)), gpu.ArgValue(int32(hidden))); err != nil {
		return err
	}
	if err := e.addVec(e.h, e.posEmb, np*hidden); err != nil {
		return err
	}

	scale := float32(1.0 / math.Sqrt(float64(hd)))
	for i := range e.layers {
		L := &e.layers[i]
		// --- attention block (pre-LN, residual) ---
		if err := e.norm(e.h, L.ln1w, L.ln1b, e.n1, np, hidden); err != nil {
			return err
		}
		// Quantize n1 once for Q, K, and V projections (shared activation)
		if err := e.quantRows(e.n1, np, hidden); err != nil {
			return err
		}
		if err := e.projQuantizedBias(L.qw, L.qb, e.qa, np); err != nil {
			return err
		}
		if err := e.projQuantizedBias(L.kw, L.kb, e.ka, np); err != nil {
			return err
		}
		if err := e.projQuantizedBias(L.vw, L.vb, e.va, np); err != nil {
			return err
		}
		ap, acfg := e.k.AttentionPlan(np, nH, hd)
		if err := e.launch(ap, acfg,
			gpu.Arg(e.qa), gpu.Arg(e.ka), gpu.Arg(e.va), gpu.Arg(e.att),
			gpu.ArgValue(int32(np)), gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(hd)),
			gpu.ArgValue(scale)); err != nil {
			return err
		}
		// Fused O projection + bias + residual add accumulating directly into h
		if err := e.projBiasAdd(e.att, L.ow, L.ob, e.h, np); err != nil {
			return err
		}
		// --- MLP block (pre-LN, residual): fc2(geluTanh(fc1(x))) ---
		if err := e.norm(e.h, L.ln2w, L.ln2b, e.n2, np, hidden); err != nil {
			return err
		}
		if err := e.projBias(e.n2, L.fc1w, L.fc1b, e.mid, np); err != nil {
			return err
		}
		if err := e.launch(e.k.GELUTanh, gpu.Grid1D(np*inter, 256),
			gpu.Arg(e.mid), gpu.ArgValue(int32(np*inter))); err != nil {
			return err
		}
		// Fused FC2 projection + bias + residual add accumulating directly into h
		if err := e.projBiasAdd(e.mid, L.fc2w, L.fc2b, e.h, np); err != nil {
			return err
		}
	}
	return e.norm(e.h, e.postLNw, e.postLNb, e.out, np, hidden)
}

// ForwardPatches runs the resident SigLIP forward on im2col patches
// [np * (C*P*P)] (vision.Encoder.GridPatches) and returns last_hidden_state
// [np * hidden]. Op for op the same sequence as vision/encoder.go's Forward.
func (e *encoder) ForwardPatches(patches []float32) ([]float32, error) {
	np, hidden := e.w.NumPatches, e.w.Hidden
	if len(patches) != np*e.cpp {
		return nil, fmt.Errorf("visioncuda: patches len %d, want %d (np=%d cpp=%d)", len(patches), np*e.cpp, np, e.cpp)
	}
	e.mu.Lock()
	defer e.mu.Unlock()

	if err := gpu.Upload(e.patches, patches); err != nil {
		return nil, err
	}
	if e.graph != nil {
		if err := e.graph.Replay(); err != nil {
			return nil, err
		}
	} else {
		if err := e.forwardTower(); err != nil {
			return nil, err
		}
	}
	// One sync for the whole tower — every launch above is async on one stream, so
	// they run in issue order and each sees the prior one's writes.
	if err := e.q.Sync(); err != nil {
		return nil, err
	}
	dst := make([]float32, np*hidden)
	if err := gpu.Download(e.out, dst); err != nil {
		return nil, err
	}
	return dst, nil
}

// Close releases the device context and everything allocated through it.
func (e *encoder) Close() {
	if e.graph != nil {
		_ = e.graph.Close()
		e.graph = nil
	}
	e.dev.ReleaseObjects()
}
