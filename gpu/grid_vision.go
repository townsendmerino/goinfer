//go:build gpu

package gpu

import (
	"fmt"
	"math"
	"sync"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/internal/vision"

	"github.com/townsendmerino/goinfer/multimodal"
)

// The Qwen3.5+ and GLM-OCR vision towers on WebGPU: aikit's ForwardViT up to the last block, in float32, from aikit's
// export (Weights, RopeTables, PositionEmbeds, VisionSegments). The tail (Qwen3: the merger; GLM-OCR: post-norm,
// downsample, merger) stays aikit's FinishHidden, on the host.
//
// Both towers' blocks: biased fused qkv (split into three projections at upload), 2-D RoPE (rotate-half over the full
// head, aikit's own tables), attention at 1/sqrt(head_dim) within each image frame, biased proj and residual, then the
// MLP and residual. Qwen3: LayerNorm with bias, MLP fc1 -> GELU-tanh -> fc2, and the interpolated position rows added
// after the patch embed. GLM-OCR: RMSNorm, per-head RMSNorm of q and k before RoPE, MLP down(silu(gate)·up), no
// position table.

func init() {
	multimodal.RegisterQwen3Tower("webgpu", func(enc *vision.Qwen3VisionEncoder) (multimodal.GridTowerAccelerator, error) {
		return newQwen3VTower(enc)
	})
	multimodal.RegisterGlmOcrTower("webgpu", func(enc *vision.GlmOcrVisionEncoder) (multimodal.GridTowerAccelerator, error) {
		return newGlmOcrVTower(enc)
	})
}

// gvRopeWGSL: rotate-half RoPE in place over x [T, heads, hd] from tables cs/sn [T, hd]: the pair (i, i + hd/2) of
// every head is rotated by the row's angle i. One invocation owns one pair; n = T * heads * hd/2.
const gvRopeWGSL = `
struct P { n: u32, heads: u32, hd: u32, _p: u32 };
@group(0) @binding(0) var<storage, read_write> x:  array<f32>;
@group(0) @binding(1) var<storage, read>       cs: array<f32>;
@group(0) @binding(2) var<storage, read>       sn: array<f32>;
@group(0) @binding(3) var<uniform>             p:  P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = ng.x * 64u;
    let hh = p.hd / 2u;
    for (var g: u32 = gid.x; g < p.n; g = g + total) {
        let i = g % hh;
        let th = g / hh;
        let t = th / p.heads;
        let xi = th * p.hd + i;
        let ci = t * p.hd + i;
        let a = x[xi];
        let b = x[xi + hh];
        x[xi] = a * cs[ci] - b * sn[ci];
        x[xi + hh] = b * cs[ci + hh] + a * sn[ci + hh];
    }
}
`

// gvSiluMulWGSL: the gated MLP's activation, g = silu(g) * u in place. The exponent is capped so a large negative
// input gives a finite denominator.
const gvSiluMulWGSL = `
struct P { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> g: array<f32>;
@group(0) @binding(1) var<storage, read>       u: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = ng.x * 64u;
    for (var i: u32 = gid.x; i < p.n; i = i + total) {
        let v = g[i];
        g[i] = v / (1.0 + exp(min(-v, 80.0))) * u[i];
    }
}
`

// gvWinScoresWGSL: windowed attention's scores. Every (row, head) is scored against the keys of the row's own segment
// seg[row] = [start, end): sc[row, head, j] = q[row, head] · k[start + j, head] for the segment's keys and a mask value
// past its end. dbg plants a defect for the towers' gate (0 outside it): 1 reads each key one row late.
const gvWinScoresWGSL = `
struct P { n: u32, heads: u32, hd: u32, maxT: u32, rows: u32, dbg: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       q:   array<f32>; // [rows, heads*hd]
@group(0) @binding(1) var<storage, read>       k:   array<f32>;
@group(0) @binding(2) var<storage, read>       seg: array<u32>; // [rows, 2]
@group(0) @binding(3) var<storage, read_write> sc:  array<f32>; // [rows, heads, maxT]
@group(0) @binding(4) var<uniform>             p:   P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = ng.x * 64u;
    let H = p.heads * p.hd;
    for (var g: u32 = gid.x; g < p.n; g = g + total) { // n = rows * heads * maxT
        let j = g % p.maxT;
        let rh = g / p.maxT;
        let h = rh % p.heads;
        let i = rh / p.heads;
        let s0 = seg[2u * i];
        var s: f32 = -3.0e38;
        if (s0 + j < seg[2u * i + 1u]) {
            let key = min(s0 + j + p.dbg, p.rows - 1u);
            let qb = i * H + h * p.hd;
            let kb = key * H + h * p.hd;
            var acc: f32 = 0.0;
            for (var d: u32 = 0u; d < p.hd; d = d + 1u) { acc = acc + q[qb + d] * k[kb + d]; }
            s = acc;
        }
        sc[g] = s;
    }
}
`

// gvWinSoftmaxWGSL: the softmax of each (row, head)'s scores [maxT] after multiplying by scale, in place; masked
// entries become 0.
const gvWinSoftmaxWGSL = `
struct P { rows: u32, maxT: u32, scale: f32, _p: u32 };
@group(0) @binding(0) var<storage, read_write> sc: array<f32>; // [rows, maxT]
@group(0) @binding(1) var<uniform>             p:  P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = ng.x * 64u;
    for (var r: u32 = gid.x; r < p.rows; r = r + total) {
        let base = r * p.maxT;
        var m: f32 = -3.0e38;
        for (var j: u32 = 0u; j < p.maxT; j = j + 1u) { m = max(m, sc[base + j]); }
        var sum: f32 = 0.0;
        for (var j: u32 = 0u; j < p.maxT; j = j + 1u) {
            let v = sc[base + j];
            var e: f32 = 0.0;
            if (v > -1.0e38) { e = exp((v - m) * p.scale); }
            sc[base + j] = e;
            sum = sum + e;
        }
        let inv = 1.0 / sum;
        for (var j: u32 = 0u; j < p.maxT; j = j + 1u) { sc[base + j] = sc[base + j] * inv; }
    }
}
`

// gvWinValuesWGSL: windowed attention's output, ctx[row, head, d] = Σ_j sc[row, head, j] · v[start + j, head, d] over
// the row's segment.
const gvWinValuesWGSL = `
struct P { n: u32, heads: u32, hd: u32, maxT: u32 };
@group(0) @binding(0) var<storage, read>       sc:  array<f32>; // [rows, heads, maxT]
@group(0) @binding(1) var<storage, read>       v:   array<f32>; // [rows, heads*hd]
@group(0) @binding(2) var<storage, read>       seg: array<u32>; // [rows, 2]
@group(0) @binding(3) var<storage, read_write> ctx: array<f32>; // [rows, heads*hd]
@group(0) @binding(4) var<uniform>             p:   P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = ng.x * 64u;
    let H = p.heads * p.hd;
    for (var g: u32 = gid.x; g < p.n; g = g + total) { // n = rows * heads * hd
        let d = g % p.hd;
        let rh = g / p.hd;
        let h = rh % p.heads;
        let i = rh / p.heads;
        let s0 = seg[2u * i];
        let T = seg[2u * i + 1u] - s0;
        let pb = rh * p.maxT;
        let vb = s0 * H + h * p.hd + d;
        var acc: f32 = 0.0;
        for (var j: u32 = 0u; j < T; j = j + 1u) { acc = acc + sc[pb + j] * v[vb + j * H]; }
        ctx[g] = acc;
    }
}
`

// gvKind is the tower family; it decides the norm, the MLP and the position term.
type gvKind uint8

const (
	gvkQwen3  gvKind = iota // LayerNorm, GELU-tanh MLP, interpolated position rows
	gvkGlmOcr               // RMSNorm, per-head q/k RMSNorm, SiLU-gated MLP
	gvkQwen25               // RMSNorm, SiLU-gated MLP, no patch bias, windows except in the full-attention blocks
)

func (k gvKind) rmsNorm() bool { return k == gvkGlmOcr || k == gvkQwen25 }
func (k gvKind) gated() bool   { return k == gvkGlmOcr || k == gvkQwen25 }

// gvWindowMax is the longest segment the windowed attention kernels take, and gvWindowBytes the largest score buffer
// [rows, heads, longest segment] they are given; a block past either runs attention as matmuls, a segment at a time.
const (
	gvWindowMax   = 256
	gvWindowBytes = 128 << 20
)

// gvNoWindowKernelsForTest sends every block down the matmul path, so the towers' gate can compare the two.
var gvNoWindowKernelsForTest bool

// gvProj is one projection on the device: [out, in] weights and, when the projection has one, a bias [out].
type gvProj struct {
	w, b    *DeviceBuffer
	out, in int
}

type gvBlock struct {
	norm1w, norm1b, norm2w, norm2b *DeviceBuffer // LayerNorm (Qwen3), or the RMSNorm weight alone (GLM-OCR)
	q, k, v, proj                  gvProj
	qNorm, kNorm                   *DeviceBuffer // GLM-OCR's per-head q/k RMSNorm
	fc1, fc2                       gvProj        // Qwen3
	gate, up, down                 gvProj        // GLM-OCR
}

// gvDefect is a planted defect for the towers' gates; zero outside them.
type gvDefect struct {
	noScale, swapRope, transposePos, noQKNorm, noPatchBias bool
	noWindows, noWindowOrder, lateKey                      bool // Qwen2.5-VL's: no windows, no window order, the windowed kernels' keys one row late
	tapShift                                               int  // DeepStack taps read this many blocks late
}

// gridVTower is a grid-based tower on WebGPU, on a Context of its own. One image batch at a time (mu).
type gridVTower struct {
	mu sync.Mutex
	*towerOps
	kind           gvKind
	hidden, inter  int
	heads, hd      int
	patchDim       int
	lnEps          float32
	rope, siluMul  towerPipe
	winScores      towerPipe // windowed attention: the scores, the softmax, the weighted values
	winSoftmax     towerPipe
	winValues      towerPipe
	patchW, patchB *DeviceBuffer
	blocks         []gvBlock
	owned          []*DeviceBuffer                     // every uploaded weight, for Close
	posEmbeds      func([][3]int) []float32            // Qwen3's interpolated position rows; nil for GLM-OCR
	ropeTables     func([][3]int) (cos, sin []float32) // [n_patches, head_dim] each
	planted        gvDefect
	winBlocks      int   // blocks run on the windowed kernels so far (the towers' gate reads it)
	uerr           error // the first failed upload (up)
}

func (a *gridVTower) Name() string { return "webgpu" }

// Close releases the tower's device buffers and its Context.
func (a *gridVTower) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	bufs := a.owned
	a.owned, a.blocks = nil, nil
	return a.towerOps.close(bufs)
}

func newGridVTower(kind gvKind, hidden, inter, heads, patchDim int, eps float64) (*gridVTower, error) {
	if heads <= 0 || hidden%heads != 0 || (hidden/heads)%2 != 0 {
		return nil, fmt.Errorf("gpu: vision tower hidden %d over %d heads: the kernels need an even head_dim dividing hidden", hidden, heads)
	}
	ops, err := newTowerOps(float32(eps))
	if err != nil {
		return nil, err
	}
	a := &gridVTower{towerOps: ops, kind: kind, hidden: hidden, inter: inter, heads: heads, hd: hidden / heads, patchDim: patchDim, lnEps: float32(eps)}
	ops.pipe(&a.rope, "gvRope", gvRopeWGSL)
	ops.pipe(&a.siluMul, "gvSiluMul", gvSiluMulWGSL)
	ops.pipe(&a.winScores, "gvWinScores", gvWinScoresWGSL)
	ops.pipe(&a.winSoftmax, "gvWinSoftmax", gvWinSoftmaxWGSL)
	ops.pipe(&a.winValues, "gvWinValues", gvWinValuesWGSL)
	if ops.err != nil {
		a.Close()
		return nil, fmt.Errorf("gpu: vision tower: %w", ops.err)
	}
	return a, nil
}

// up uploads one weight vector (nil for an empty one); the first failure is kept in a.uerr.
func (a *gridVTower) up(v []float32) *DeviceBuffer {
	if a.uerr != nil || len(v) == 0 {
		return nil
	}
	d, err := a.c.UploadF32(v)
	if err != nil {
		a.uerr = err
		return nil
	}
	a.owned = append(a.owned, d)
	return d
}

func (a *gridVTower) proj(name string, p vision.VisionProj) gvProj {
	if a.uerr == nil && (len(p.W) != p.Out*p.In || (len(p.B) != 0 && len(p.B) != p.Out)) {
		a.uerr = fmt.Errorf("%s is [%d, %d] with %d weights and %d biases", name, p.Out, p.In, len(p.W), len(p.B))
	}
	return gvProj{w: a.up(p.W), b: a.up(p.B), out: p.Out, in: p.In}
}

// splitQKV uploads a fused [3·hidden, hidden] qkv projection (rows q, then k, then v) as three projections.
func (a *gridVTower) splitQKV(p vision.VisionProj) (q, k, v gvProj) {
	H := a.hidden
	if a.uerr == nil && (p.Out != 3*H || p.In != H || len(p.W) != 3*H*H || (len(p.B) != 0 && len(p.B) != 3*H)) {
		a.uerr = fmt.Errorf("qkv is %dx%d with %d biases, want %dx%d", p.Out, p.In, len(p.B), 3*H, H)
		return q, k, v
	}
	part := func(i int) gvProj {
		r := gvProj{w: a.up(p.W[i*H*H : (i+1)*H*H]), out: H, in: H}
		if len(p.B) == 3*H {
			r.b = a.up(p.B[i*H : (i+1)*H])
		}
		return r
	}
	return part(0), part(1), part(2)
}

// built ends a tower's construction: the tower, or its upload error with everything released.
func (a *gridVTower) built(what string) (*gridVTower, error) {
	if a.uerr != nil {
		err := a.uerr
		a.Close()
		return nil, fmt.Errorf("gpu: upload %s vision tower: %w", what, err)
	}
	return a, nil
}

func newQwen3VTower(enc *vision.Qwen3VisionEncoder) (*gridVTower, error) {
	w, err := enc.Weights()
	if err != nil {
		return nil, fmt.Errorf("gpu: %w", err)
	}
	c := w.Cfg
	a, err := newGridVTower(gvkQwen3, c.HiddenSize, c.IntermediateSize, c.NumHeads,
		c.InChannels*c.TemporalPatchSize*c.PatchSize*c.PatchSize, w.LNEps)
	if err != nil {
		return nil, err
	}
	a.patchW, a.patchB = a.up(w.PatchW), a.up(w.PatchB)
	a.posEmbeds, a.ropeTables = enc.PositionEmbeds, enc.RopeTables
	for _, b := range w.Blocks {
		q, k, v := a.splitQKV(b.QKV)
		a.blocks = append(a.blocks, gvBlock{norm1w: a.up(b.Norm1W), norm1b: a.up(b.Norm1B), norm2w: a.up(b.Norm2W), norm2b: a.up(b.Norm2B),
			q: q, k: k, v: v, proj: a.proj("proj", b.Proj), fc1: a.proj("fc1", b.FC1), fc2: a.proj("fc2", b.FC2)})
	}
	return a.built("Qwen3")
}

func newGlmOcrVTower(enc *vision.GlmOcrVisionEncoder) (*gridVTower, error) {
	w, err := enc.Weights()
	if err != nil {
		return nil, fmt.Errorf("gpu: %w", err)
	}
	c := w.Cfg
	a, err := newGridVTower(gvkGlmOcr, c.HiddenSize, c.IntermediateSize, c.NumHeads,
		c.InChannels*c.TemporalPatchSize*c.PatchSize*c.PatchSize, w.RMSEps)
	if err != nil {
		return nil, err
	}
	a.patchW, a.patchB = a.up(w.PatchW), a.up(w.PatchB)
	a.ropeTables = enc.RopeTables
	for _, b := range w.Blocks {
		if a.uerr == nil && (len(b.QNorm) != a.hd || len(b.KNorm) != a.hd) {
			a.uerr = fmt.Errorf("q/k norm widths %d/%d, want head_dim %d", len(b.QNorm), len(b.KNorm), a.hd)
		}
		q, k, v := a.splitQKV(b.QKV)
		a.blocks = append(a.blocks, gvBlock{norm1w: a.up(b.Norm1W), norm2w: a.up(b.Norm2W), q: q, k: k, v: v, proj: a.proj("proj", b.Proj),
			qNorm: a.up(b.QNorm), kNorm: a.up(b.KNorm), gate: a.proj("gate", b.Gate), up: a.proj("up", b.Up), down: a.proj("down", b.Down)})
	}
	return a.built("GLM-OCR")
}

// linear runs out = x·Wᵀ (+ bias) over rows into a new buffer.
func (a *gridVTower) linear(s *towerScope, x *DeviceBuffer, p *gvProj, rows int) (*DeviceBuffer, error) {
	out, err := s.add(a.c.matmulF32Device(x.buf, p.w.buf, rows, p.in, p.out))
	if err != nil {
		return nil, err
	}
	if p.b != nil {
		if err := a.c.addRowsDevice(out.buf, p.b.buf, rows*p.out, p.out, 0); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// norm is the block's norm over x [rows, hidden] into a new buffer: LayerNorm with bias (Qwen3) or the weight-only
// RMSNorm (GLM-OCR).
func (a *gridVTower) norm(s *towerScope, x, w, b *DeviceBuffer, rows int) (*DeviceBuffer, error) {
	if a.kind.rmsNorm() {
		return a.rmsRows(s, x, w, rows, a.hidden)
	}
	dst, err := s.add(a.c.newDevBuf(rows * a.hidden))
	if err != nil {
		return nil, err
	}
	return dst, a.c.layerNormRowsDevice(x.buf, w.buf, b.buf, dst.buf, rows, a.hidden, a.lnEps)
}

// Hidden runs the patch embed and every block (multimodal.GridTowerAccelerator); the attention segments are the image
// frames.
func (a *gridVTower) Hidden(pixels []float32, gridTHW [][3]int) ([]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	in, err := a.gridInput(pixels, gridTHW)
	if err != nil {
		return nil, err
	}
	return a.run(in, nil)
}

// HiddenTaps is Hidden plus the outputs of the given blocks, in that order (multimodal.GridTowerTapper): Qwen3-VL's
// DeepStack taps. Each tap is the residual right after its block.
func (a *gridVTower) HiddenTaps(pixels []float32, gridTHW [][3]int, blocks []int) ([]float32, [][]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	in, err := a.gridInput(pixels, gridTHW)
	if err != nil {
		return nil, nil, err
	}
	taps := make([][]float32, len(blocks))
	h, err := a.run(in, func(bi int, x *DeviceBuffer) error {
		for k, b := range blocks {
			if b+a.planted.tapShift == bi {
				t, err := a.c.Readback(x)
				if err != nil {
					return err
				}
				taps[k] = t
			}
		}
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	for k, t := range taps {
		if t == nil {
			return nil, nil, fmt.Errorf("gpu: DeepStack tap %d (block %d) was not reached in %d blocks", k, blocks[k], len(a.blocks))
		}
	}
	return h, taps, nil
}

// gvInput is one forward's host-side input: the patch rows in the order the blocks run them, the RoPE tables and
// position rows in that order, and each block's attention segments as cumulative row offsets.
type gvInput struct {
	pixels   []float32 // [np, patchDim]
	np       int
	cos, sin []float32 // [np, head_dim]
	pos      []float32 // [np, hidden]; nil for a tower with no position rows
	segs     func(bi int) []int
}

// gridInput is the input of a tower whose segments are the image frames and whose tables come from aikit by grid.
func (a *gridVTower) gridInput(pixels []float32, gridTHW [][3]int) (gvInput, error) {
	segs := vision.VisionSegments(gridTHW)
	in := gvInput{pixels: pixels, np: segs[len(segs)-1], segs: func(int) []int { return segs }}
	if in.np == 0 || len(pixels) != in.np*a.patchDim {
		return in, fmt.Errorf("gpu: %d pixel values for %d patches of %d", len(pixels), in.np, a.patchDim)
	}
	in.cos, in.sin = a.ropeTables(gridTHW)
	if a.posEmbeds != nil {
		g := gridTHW
		if a.planted.transposePos { // the position rows for the transposed grid
			g = make([][3]int, len(gridTHW))
			for i, t := range gridTHW {
				g[i] = [3]int{t[0], t[2], t[1]}
			}
		}
		in.pos = a.posEmbeds(g)
	}
	return in, nil
}

// run is the forward; tap, when not nil, is called after each block with the residual. The caller holds a.mu.
func (a *gridVTower) run(in gvInput, tap func(bi int, x *DeviceBuffer) error) ([]float32, error) {
	if a.c == nil {
		return nil, fmt.Errorf("gpu: the vision tower is closed")
	}
	c, H, hd, np := a.c, a.hidden, a.hd, in.np
	if np == 0 || len(in.pixels) != np*a.patchDim {
		return nil, fmt.Errorf("gpu: %d pixel values for %d patches of %d", len(in.pixels), np, a.patchDim)
	}
	if np > 65535 {
		return nil, fmt.Errorf("gpu: %d patches; the tower's row kernels take at most 65535", np)
	}
	cos, sin := in.cos, in.sin
	if len(cos) != np*hd || len(sin) != np*hd {
		return nil, fmt.Errorf("gpu: rope tables %d/%d for %d patches of head_dim %d", len(cos), len(sin), np, hd)
	}
	if in.pos != nil && len(in.pos) != np*H {
		return nil, fmt.Errorf("gpu: %d position values for %d patches of %d", len(in.pos), np, H)
	}
	if a.planted.swapRope { // the row and column halves of each rotation swapped
		cos, sin = swapRopeHalves(cos, hd), swapRopeHalves(sin, hd)
	}
	a.pending = 0
	var whole towerScope // the buffers that live for the whole forward
	defer whole.release()
	px, err := whole.add(c.UploadF32(in.pixels))
	if err != nil {
		return nil, err
	}
	cs, err := whole.add(c.UploadF32(cos))
	if err != nil {
		return nil, err
	}
	sn, err := whole.add(c.UploadF32(sin))
	if err != nil {
		return nil, err
	}
	pe := gvProj{w: a.patchW, b: a.patchB, out: H, in: a.patchDim}
	if a.planted.noPatchBias {
		pe.b = nil
	}
	x, err := a.linear(&whole, px, &pe, np)
	if err != nil {
		return nil, err
	}
	if in.pos != nil {
		pd, err := whole.add(c.UploadF32(in.pos))
		if err != nil {
			return nil, err
		}
		if err := c.addRowsDevice(x.buf, pd.buf, np*H, H, 1); err != nil {
			return nil, err
		}
	}
	bounds := map[*int]*DeviceBuffer{} // a segment list's per-row bounds on the device, by the list's first element
	for bi := range a.blocks {
		segs := in.segs(bi)
		if len(segs) < 2 || segs[0] != 0 || segs[len(segs)-1] != np {
			return nil, fmt.Errorf("gpu: block %d's segments do not cover %d patches", bi, np)
		}
		var seg *DeviceBuffer // nil: attention as matmuls
		if longest := maxSegment(segs); !gvNoWindowKernelsForTest && len(segs) > 2 && longest <= gvWindowMax && 4*np*a.heads*longest <= gvWindowBytes {
			if seg = bounds[&segs[0]]; seg == nil {
				if seg, err = whole.add(a.uploadBounds(segs)); err != nil {
					return nil, err
				}
				bounds[&segs[0]] = seg
			}
		}
		if err := a.block(&a.blocks[bi], x, cs, sn, np, segs, seg); err != nil {
			return nil, fmt.Errorf("gpu: vision tower block %d: %w", bi, err)
		}
		if tap != nil {
			if err := tap(bi, x); err != nil {
				return nil, err
			}
		}
	}
	return c.Readback(x)
}

// maxSegment is the longest segment of a cumulative-offset list.
func maxSegment(segs []int) int {
	m := 0
	for i := 1; i < len(segs); i++ {
		m = max(m, segs[i]-segs[i-1])
	}
	return m
}

// uploadBounds puts each row's segment [start, end) on the device, [rows, 2] of u32, for the windowed kernels.
func (a *gridVTower) uploadBounds(segs []int) (*DeviceBuffer, error) {
	b := make([]uint32, 2*segs[len(segs)-1])
	for si := 1; si < len(segs); si++ {
		for r := segs[si-1]; r < segs[si]; r++ {
			b[2*r], b[2*r+1] = uint32(segs[si-1]), uint32(segs[si])
		}
	}
	buf, err := a.c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Label: "seg-bounds", Contents: wgpu.ToBytes(b), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	return newDeviceBuffer(buf, len(b)), nil
}

// windowAttention is attention within each row's own segment for every row and head at once: the scores, their
// softmax, the weighted values. seg holds the per-row bounds and longest the longest segment.
func (a *gridVTower) windowAttention(s *towerScope, q, k, v, ctx, seg *DeviceBuffer, np, longest int, scale float32) error {
	c := a.c
	nH, hd := a.heads, a.hd
	n := np * nH * longest
	sc, err := s.add(c.newDevBuf(n))
	if err != nil {
		return err
	}
	dbg := uint32(0)
	if a.planted.lateKey {
		dbg = 1
	}
	if err := c.kernelN(a.winScores, []*wgpu.Buffer{q.buf, k.buf, seg.buf, sc.buf},
		[]uint32{uint32(n), uint32(nH), uint32(hd), uint32(longest), uint32(np), dbg, 0, 0}, maxWG((n+63)/64), 1); err != nil {
		return err
	}
	if err := c.kernelN(a.winSoftmax, []*wgpu.Buffer{sc.buf}, []uint32{uint32(np * nH), uint32(longest), math.Float32bits(scale), 0}, maxWG((np*nH+63)/64), 1); err != nil {
		return err
	}
	m := np * nH * hd
	return c.kernelN(a.winValues, []*wgpu.Buffer{sc.buf, v.buf, seg.buf, ctx.buf}, []uint32{uint32(m), uint32(nH), uint32(hd), uint32(longest)}, maxWG((m+63)/64), 1)
}

// swapRopeHalves returns t [n, hd] with the two quarters of each half exchanged.
func swapRopeHalves(t []float32, hd int) []float32 {
	out := make([]float32, len(t))
	half, q := hd/2, hd/4
	for r := 0; r+hd <= len(t); r += hd {
		for h := 0; h < hd; h += half {
			copy(out[r+h:], t[r+h+q:r+h+half])
			copy(out[r+h+half-q:r+h+half], t[r+h:r+h+q])
		}
	}
	return out
}

// block runs one block over the residual stream x [np, hidden] in place, and waits for the device before its scratch
// is released. segs are the attention segments as cumulative row offsets; seg, when not nil, is their per-row bounds on
// the device and the block's attention runs on the windowed kernels.
func (a *gridVTower) block(b *gvBlock, x, cs, sn *DeviceBuffer, np int, segs []int, seg *DeviceBuffer) error {
	c := a.c
	H, I, nH, hd := a.hidden, a.inter, a.heads, a.hd
	var s towerScope
	defer s.release()
	xn, err := a.norm(&s, x, b.norm1w, b.norm1b, np)
	if err != nil {
		return err
	}
	q, err := a.linear(&s, xn, &b.q, np)
	if err != nil {
		return err
	}
	k, err := a.linear(&s, xn, &b.k, np)
	if err != nil {
		return err
	}
	v, err := a.linear(&s, xn, &b.v, np)
	if err != nil {
		return err
	}
	if a.kind == gvkGlmOcr && !a.planted.noQKNorm { // per-head q/k RMSNorm, before RoPE
		if q, err = a.rmsRows(&s, q, b.qNorm, np*nH, hd); err != nil {
			return err
		}
		if k, err = a.rmsRows(&s, k, b.kNorm, np*nH, hd); err != nil {
			return err
		}
	}
	nr := np * nH * hd / 2
	for _, t := range []*DeviceBuffer{q, k} {
		if err := c.kernelN(a.rope, []*wgpu.Buffer{t.buf, cs.buf, sn.buf}, []uint32{uint32(nr), uint32(nH), uint32(hd), 0}, maxWG((nr+63)/64), 1); err != nil {
			return err
		}
	}
	scale := float32(1 / math.Sqrt(float64(hd)))
	if a.planted.noScale {
		scale = 1
	}
	ctx, err := s.add(c.newDevBuf(np * H))
	if err != nil {
		return err
	}
	if seg != nil {
		if err := a.windowAttention(&s, q, k, v, ctx, seg, np, maxSegment(segs), scale); err != nil {
			return err
		}
		a.winBlocks++
	} else {
		for si := 1; si < len(segs); si++ {
			if err := a.attention(q, k, v, ctx, segs[si-1], segs[si]-segs[si-1], H, nH, hd, scale); err != nil {
				return err
			}
		}
	}
	o, err := a.linear(&s, ctx, &b.proj, np)
	if err != nil {
		return err
	}
	if err := c.addRowsDevice(x.buf, o.buf, np*H, H, 1); err != nil {
		return err
	}
	if xn, err = a.norm(&s, x, b.norm2w, b.norm2b, np); err != nil {
		return err
	}
	var f *DeviceBuffer
	if a.kind.gated() {
		g, err := a.linear(&s, xn, &b.gate, np)
		if err != nil {
			return err
		}
		u, err := a.linear(&s, xn, &b.up, np)
		if err != nil {
			return err
		}
		if err := c.kernelN(a.siluMul, []*wgpu.Buffer{g.buf, u.buf}, []uint32{uint32(np * I), 0, 0, 0}, maxWG((np*I+63)/64), 1); err != nil {
			return err
		}
		if f, err = a.linear(&s, g, &b.down, np); err != nil {
			return err
		}
	} else {
		m, err := a.linear(&s, xn, &b.fc1, np)
		if err != nil {
			return err
		}
		if err := c.inplaceKernel(c.geluPipeline, c.geluLayout, m.buf, []uint32{uint32(np * I), 0, 0, 0}, maxWG((np*I+63)/64)); err != nil {
			return err
		}
		if f, err = a.linear(&s, m, &b.fc2, np); err != nil {
			return err
		}
	}
	if err := c.addRowsDevice(x.buf, f.buf, np*H, H, 1); err != nil {
		return err
	}
	c.device.Poll(true, nil)
	return nil
}
