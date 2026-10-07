//go:build cuda

package cuda

import (
	"fmt"
	"runtime"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// The CUDA vision-tower base (S4 of docs/tasks/task-multimodal-support-2026-10.md, Gate 0: docs/measurements/multimodal-support-2026-10/s4-gate0-cuda-towers.md).
// It is aikit's gpu.ViT (f32 GEMMs with bias and bias+residual epilogues, LayerNorm, RMSNorm, GELU, scaled attention, a NeoX RoPE) plus the six kernels of
// tower_base.cu, behind one executor goroutine that owns the device (CUDA contexts are thread-affine, as in vision_encoder.go) and a launch helper that latches
// the first error. Metal's base is eg2Ops; this is the CUDA counterpart, not a port: CUDA kernels take their scalars as launch arguments, and attention is aikit's
// own kernel (np <= 12288 by its shared-memory row), so Metal's matmul-blocked attention is not needed for a correct baseline.
//
// Every method except new/close/do must run inside do(): they touch the device. A failed allocation panics inside the executor (gpu.NewBufferLenOf's contract)
// and do() turns it into an error, so a tower's factory and Hidden return errors, never panics, and serve can name the CPU fallback.

type towerOps struct {
	dev *Device
	q   Queue
	vit gpu.ViT

	posAdd, clamp, clampCopy, ropeAxial, mul, scale Pipeline

	zeroBias Buffer // zeros, as long as the widest bias-free projection: lets gemm run the register-blocked bias kernel at any M

	scratch []Buffer // per-call buffers, released by releaseScratch
	err     error    // the first launch error of the current call

	reqCh chan func() error
	ackCh chan error
}

// newTowerOps creates the device and loads the kernels on a pinned executor goroutine. maxN is the widest output of any projection run without a bias.
func newTowerOps(maxN int) (t *towerOps, err error) {
	t = &towerOps{reqCh: make(chan func() error), ackCh: make(chan error)}
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		for j := range t.reqCh {
			t.ackCh <- runJob(j)
		}
	}()
	err = t.do(func() error {
		var e error
		if t.dev, e = CreateSystemDefaultDevice(); e != nil {
			return e
		}
		t.q = t.dev.NewCommandQueue()
		if t.vit, e = t.dev.NewViT(); e != nil {
			return fmt.Errorf("cuda tower: aikit's ViT kernels: %w", e)
		}
		lib, e := t.dev.CompileLibrary(towerBasePTX)
		if e != nil {
			return fmt.Errorf("cuda tower: compile tower_base: %w", e)
		}
		for _, b := range []struct {
			name string
			dst  *Pipeline
		}{{"tower_pos_add", &t.posAdd}, {"tower_clamp", &t.clamp}, {"tower_clamp_copy", &t.clampCopy},
			{"tower_rope_axial", &t.ropeAxial}, {"tower_mul", &t.mul}, {"tower_scale", &t.scale}} {
			if *b.dst, e = t.dev.NewComputePipeline(lib, b.name); e != nil {
				return fmt.Errorf("cuda tower: pipeline %s: %w", b.name, e)
			}
		}
		t.zeroBias = gpu.NewBufferOf(t.dev, make([]float32, max(maxN, 1)))
		return nil
	})
	if err != nil {
		t.close()
		return nil, err
	}
	return t, nil
}

// do runs j on the executor goroutine; a panic inside it (an allocation failure) comes back as an error.
func (t *towerOps) do(j func() error) error {
	t.reqCh <- j
	return <-t.ackCh
}

// close releases everything the device holds and stops the executor. Safe to call once.
func (t *towerOps) close() {
	if t.reqCh == nil {
		return
	}
	_ = t.do(func() error {
		if t.dev != nil {
			t.releaseScratch()
			t.dev.ReleaseObjects()
		}
		return nil
	})
	close(t.reqCh)
	t.reqCh = nil
}

// up uploads a float32 slice as a persistent buffer (weights, tables).
func (t *towerOps) up(x []float32) Buffer {
	if len(x) == 0 {
		return Buffer{}
	}
	return gpu.NewBufferOf(t.dev, x)
}

// af allocates a float32 scratch buffer of n elements, released by releaseScratch.
func (t *towerOps) af(n int) Buffer {
	b := gpu.NewBufferLenOf[float32](t.dev, max(n, 1))
	t.scratch = append(t.scratch, b)
	return b
}

// ai allocates an int32 scratch buffer.
func (t *towerOps) ai(n int) Buffer {
	b := gpu.NewBufferLenOf[int32](t.dev, max(n, 1))
	t.scratch = append(t.scratch, b)
	return b
}

func (t *towerOps) releaseScratch() {
	for _, b := range t.scratch {
		t.dev.ReleaseBuf(b)
	}
	t.scratch = t.scratch[:0]
}

// launch enqueues one kernel and latches the first error.
func (t *towerOps) launch(p Pipeline, cfg LaunchConfig, args ...KernelArg) {
	if t.err != nil {
		return
	}
	t.err = t.q.Launch(p, cfg, args...)
}

// finish waits for the queue and returns the call's first error, clearing it.
func (t *towerOps) finish() error {
	if t.err == nil {
		t.err = t.q.Sync()
	}
	err := t.err
	t.err = nil
	return err
}

var i32 = gpu.ArgValue[int32]
var f32v = gpu.ArgValue[float32]
var i64 = gpu.ArgValue[int64]

// gemm is out[M,N] = a[M,K] · w[N,K]ᵀ. With K a multiple of 16 it runs the bias kernel with a zero bias, which takes any M and N at the register-blocked
// speed; otherwise aikit's plan (the register kernel when M and N are multiples of 64, else the tiled one).
func (t *towerOps) gemm(a, w, out Buffer, M, N, K int) {
	if K%16 == 0 && N <= t.zeroBias.Len() {
		p, cfg := t.vit.GEMMF32BiasPlan(M, N, K)
		t.launch(p, cfg, Arg(a), Arg(w), Arg(t.zeroBias), Arg(out), i32(int32(M)), i32(int32(N)), i32(int32(K)))
		return
	}
	p, cfg := t.vit.GEMMF32Plan(M, N, K)
	t.launch(p, cfg, Arg(a), Arg(w), Arg(out), i32(int32(M)), i32(int32(N)), i32(int32(K)))
}

// gemmBias is out[M,N] = a[M,K] · w[N,K]ᵀ + bias[N]. A K that is not a multiple of 16 (real GLM-OCR's patch embed, K = 1176) takes the tiled GEMM plus add_bias.
func (t *towerOps) gemmBias(a, w, bias, out Buffer, M, N, K int) {
	if K%16 == 0 {
		p, cfg := t.vit.GEMMF32BiasPlan(M, N, K)
		t.launch(p, cfg, Arg(a), Arg(w), Arg(bias), Arg(out), i32(int32(M)), i32(int32(N)), i32(int32(K)))
		return
	}
	p, cfg := t.vit.GEMMF32Plan(M, N, K)
	t.launch(p, cfg, Arg(a), Arg(w), Arg(out), i32(int32(M)), i32(int32(N)), i32(int32(K)))
	t.launch(t.vit.AddBias, gpu.Grid1D(M*N, 256), Arg(out), Arg(bias), i32(int32(M)), i32(int32(N)))
}

// gemmBiasAdd is resid[M,N] += a[M,K] · w[N,K]ᵀ + bias[N]. A K that is not a multiple of 16 goes through tmp[M*N] and add_vec.
func (t *towerOps) gemmBiasAdd(a, w, bias, resid, tmp Buffer, M, N, K int) {
	if K%16 == 0 {
		p, cfg := t.vit.GEMMF32BiasAddPlan(M, N, K)
		t.launch(p, cfg, Arg(a), Arg(w), Arg(bias), Arg(resid), i32(int32(M)), i32(int32(N)), i32(int32(K)))
		return
	}
	t.gemmBias(a, w, bias, tmp, M, N, K)
	t.addVec(resid, tmp, M*N)
}

// rms is out = rmsnorm(x) * w over rows of dim (in place when out is x), eps as given.
func (t *towerOps) rms(x, w, out Buffer, rows, dim int, eps float32) {
	t.launch(t.vit.RMSNorm, gpu.RowGrid(rows), Arg(x), Arg(w), Arg(out), i32(int32(rows)), i32(int32(dim)), f32v(eps))
}

// layerNorm is out = layernorm(x) * w + b over rows of dim.
func (t *towerOps) layerNorm(x, w, b, out Buffer, rows, dim int, eps float32) {
	t.launch(t.vit.LayerNorm, gpu.RowGrid(rows), Arg(x), Arg(w), Arg(b), Arg(out), i32(int32(rows)), i32(int32(dim)), f32v(eps))
}

func (t *towerOps) addVec(x, y Buffer, n int) {
	t.launch(t.vit.AddVec, gpu.Grid1D(n, 256), Arg(x), Arg(y), i32(int32(n)))
}

func (t *towerOps) geluTanh(x Buffer, n int) {
	t.launch(t.vit.GELUTanh, gpu.Grid1D(n, 256), Arg(x), i32(int32(n)))
}

// mul is x *= u, elementwise.
func (t *towerOps) mulVec(x, u Buffer, n int) {
	t.launch(t.mul, gpu.Grid1D(n, 256), Arg(x), Arg(u), i64(int64(n)))
}

// scaleVec is x *= s.
func (t *towerOps) scaleVec(x Buffer, n int, s float32) {
	t.launch(t.scale, gpu.Grid1D(n, 256), Arg(x), i64(int64(n)), f32v(s))
}

// clampInPlace clamps x[:n] to [lo, hi]; clampCopy writes the clamped x into out.
func (t *towerOps) clampInPlace(x Buffer, n int, lo, hi float32) {
	t.launch(t.clamp, gpu.Grid1D(n, 256), Arg(x), i64(int64(n)), f32v(lo), f32v(hi))
}

func (t *towerOps) clampCopyTo(x, out Buffer, n int, lo, hi float32) {
	t.launch(t.clampCopy, gpu.Grid1D(n, 256), Arg(x), Arg(out), i64(int64(n)), f32v(lo), f32v(hi))
}

// posAddTo is x[i, d] += X[pos[i].x, d] + Y[pos[i].y, d] over rows of width H (pos is [rows, 2] int32).
func (t *towerOps) posAddTo(x, X, Y, pos Buffer, rows, H int) {
	n := rows * H
	t.launch(t.posAdd, gpu.Grid1D(n, 256), Arg(x), Arg(X), Arg(Y), Arg(pos), i32(int32(H)), i64(int64(n)))
}

// ropeAxialTo is Gemma 4's axial 2-D RoPE in place over x [T, heads, hd] from cos/sin [T, hd].
func (t *towerOps) ropeAxialTo(x, cs, sn Buffer, T, heads, hd int) {
	n := T * heads * hd / 2
	t.launch(t.ropeAxial, gpu.Grid1D(n, 256), Arg(x), Arg(cs), Arg(sn), i32(int32(heads)), i32(int32(hd)), i64(int64(n)))
}

// attention is aikit's bidirectional multi-head self-attention over np patches, q/k/v/out all [np, nH*hd] with the heads in place, at the given scale. The
// score row lives in shared memory (np*4 bytes), so np is capped (48 KB by default, np <= 12288); a larger image is refused by name, not truncated.
func (t *towerOps) attention(q, k, v, out Buffer, np, nH, hd int, scale float32) error {
	if np > 12288 {
		return fmt.Errorf("cuda tower: %d patches exceed the attention kernel's shared-memory row (12288)", np)
	}
	p, cfg := t.vit.AttentionPlan(np, nH, hd)
	t.launch(p, cfg, Arg(q), Arg(k), Arg(v), Arg(out), i32(int32(np)), i32(int32(nH)), i32(int32(hd)), f32v(scale))
	return nil
}
