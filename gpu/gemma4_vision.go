//go:build gpu

package gpu

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/internal/vision"

	"github.com/townsendmerino/goinfer/multimodal"
)

// The Gemma 4 vision tower on WebGPU: aikit's Gemma4Encoder.Forward up to its pool, in float32, from aikit's export
// (Gemma4Encoder.Weights). The patch embed and every layer run here on the float32 matmul the SigLIP tower uses
// (matmulF32Device), with the row RMSNorm, the ClippableLinear clamps, the axial 2-D RoPE and the gated GELU below. The
// pool and the projection after them stay aikit's (FinishHidden), on the host. Registered as multimodal's "webgpu"
// Gemma 4 tower.

func init() {
	multimodal.RegisterGemma4Tower("webgpu", func(enc *vision.Gemma4Encoder) (multimodal.Gemma4TowerAccelerator, error) {
		return newG4VTower(enc)
	})
}

// g4vClampWGSL: a ClippableLinear's output clamp, in place.
const g4vClampWGSL = `
struct P { n: u32, lo: f32, hi: f32, _p: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform>             p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = ng.x * 64u;
    for (var i: u32 = gid.x; i < p.n; i = i + total) { x[i] = min(max(x[i], p.lo), p.hi); }
}
`

// g4vClampCopyWGSL: its input clamp, into a copy (q, k and v read the same normed input under different bounds).
const g4vClampCopyWGSL = `
struct P { n: u32, lo: f32, hi: f32, _p: u32 };
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>             p:   P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = ng.x * 64u;
    for (var i: u32 = gid.x; i < p.n; i = i + total) { dst[i] = min(max(src[i], p.lo), p.hi); }
}
`

// g4vRopeWGSL: axial 2-D RoPE in place over x [T, heads, hd] from aikit's tables cs/sn [T, hd]. Each half of a head is
// rotated rotate-half within itself (pairs d, d + hd/4), the first half by the patch's x and the second by its y. One
// invocation owns one pair; n = T * heads * hd/2.
const g4vRopeWGSL = `
struct P { n: u32, heads: u32, hd: u32, _p: u32 };
@group(0) @binding(0) var<storage, read_write> x:  array<f32>;
@group(0) @binding(1) var<storage, read>       cs: array<f32>;
@group(0) @binding(2) var<storage, read>       sn: array<f32>;
@group(0) @binding(3) var<uniform>             p:  P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = ng.x * 64u;
    let hh = p.hd / 2u;
    let qd = p.hd / 4u;
    for (var g: u32 = gid.x; g < p.n; g = g + total) {
        let pr = g % hh;
        let th = g / hh;
        let t = th / p.heads;
        let side = (pr / qd) * hh;
        let d = pr % qd;
        let xi = th * p.hd + side + d;
        let ci = t * p.hd + side + d;
        let a = x[xi];
        let b = x[xi + qd];
        x[xi] = a * cs[ci] - b * sn[ci];
        x[xi + qd] = b * cs[ci + qd] + a * sn[ci + qd];
    }
}
`

// g4vGeluMulWGSL: the gated MLP's activation, g = gelu_tanh(g) * u in place. The tanh argument is clamped to ±15, where
// tanh is already ±1 in float32: a device tanh may return NaN for a large argument.
const g4vGeluMulWGSL = `
struct P { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> g: array<f32>;
@group(0) @binding(1) var<storage, read>       u: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = ng.x * 64u;
    let c = 0.7978845608028654; // sqrt(2/pi)
    for (var i: u32 = gid.x; i < p.n; i = i + total) {
        let v = g[i];
        g[i] = 0.5 * v * (1.0 + tanh(clamp(c * (v + 0.044715 * v * v * v), -15.0, 15.0))) * u[i];
    }
}
`

// g4vProj is one ClippableLinear on the device: its [out, in] weight and its bounds. An infinite bound is held as the
// largest float32, which clamps nothing.
type g4vProj struct {
	w                            *DeviceBuffer
	out, in                      int
	inMin, inMax, outMin, outMax float32
	clipIn, clipOut              bool
}

type g4vLayer struct {
	inNorm, postAttnNorm, preFFNorm, postFFNorm, qNorm, kNorm *DeviceBuffer
	q, k, v, o, gate, up, down                                g4vProj
}

// g4vTower is the tower on WebGPU, on a Context of its own. One image at a time (mu).
type g4vTower struct {
	mu sync.Mutex
	*towerOps
	cfg                             vision.Gemma4EncoderConfig
	theta                           float64
	clamp, clampCopy, rope, geluMul towerPipe
	patchW, ones                    *DeviceBuffer // ones: v's unweighted norm
	posX, posY                      []float32     // the position tables, added on the host side of the upload
	layers                          []g4vLayer
}

// g4vDefectForTest plants one defect in the tower, for its gate's planted-defect arm (gpu/gemma4_vision_test.go): 1 the
// clamps skipped, 2 the x and y position tables swapped, 3 RoPE not applied to k, 4 the v norm skipped, 5 k normed with
// q's weight. 0, the only value outside that test, plants nothing.
var g4vDefectForTest int

func newG4VTower(enc *vision.Gemma4Encoder) (*g4vTower, error) {
	w, err := enc.Weights()
	if err != nil {
		return nil, fmt.Errorf("gpu: %w", err)
	}
	cfg := w.Cfg
	if cfg.HeadDim%4 != 0 || cfg.NumAttentionHeads*cfg.HeadDim != cfg.HiddenSize {
		return nil, fmt.Errorf("gpu: Gemma 4 tower %d heads of %d over hidden %d: the kernels need heads*hd = hidden, hd a multiple of 4",
			cfg.NumAttentionHeads, cfg.HeadDim, cfg.HiddenSize)
	}
	ops, err := newTowerOps(float32(cfg.RMSNormEps))
	if err != nil {
		return nil, err
	}
	c := ops.c
	a := &g4vTower{towerOps: ops, cfg: cfg, theta: w.RopeTheta, posX: w.PosEmbX, posY: w.PosEmbY}
	fail := func(e error) (*g4vTower, error) {
		a.Close()
		return nil, fmt.Errorf("gpu: Gemma 4 vision tower: %w", e)
	}
	ops.pipe(&a.clamp, "g4vClamp", g4vClampWGSL)
	ops.pipe(&a.clampCopy, "g4vClampCopy", g4vClampCopyWGSL)
	ops.pipe(&a.rope, "g4vRope", g4vRopeWGSL)
	ops.pipe(&a.geluMul, "g4vGeluMul", g4vGeluMulWGSL)
	if ops.err != nil {
		return fail(ops.err)
	}
	up := func(x []float32) *DeviceBuffer {
		if err != nil {
			return nil
		}
		var d *DeviceBuffer
		d, err = c.UploadF32(x)
		return d
	}
	bound := func(v float32) (float32, bool) {
		if math.IsInf(float64(v), 0) {
			return float32(math.Copysign(math.MaxFloat32, float64(v))), false
		}
		return v, true
	}
	proj := func(name string, p vision.Gemma4Proj) g4vProj {
		if err == nil && len(p.W) != p.Out*p.In {
			err = fmt.Errorf("%s is [%d, %d] with %d weights", name, p.Out, p.In, len(p.W))
		}
		g := g4vProj{w: up(p.W), out: p.Out, in: p.In}
		var a0, a1, b0, b1 bool
		g.inMin, a0 = bound(p.InMin)
		g.inMax, a1 = bound(p.InMax)
		g.outMin, b0 = bound(p.OutMin)
		g.outMax, b1 = bound(p.OutMax)
		g.clipIn, g.clipOut = a0 || a1, b0 || b1
		return g
	}
	a.patchW = up(w.PatchEmbed)
	ones := make([]float32, cfg.HeadDim)
	for i := range ones {
		ones[i] = 1
	}
	a.ones = up(ones)
	a.layers = make([]g4vLayer, len(w.Layers))
	for i, l := range w.Layers {
		a.layers[i] = g4vLayer{inNorm: up(l.InputNorm), postAttnNorm: up(l.PostAttnNorm), preFFNorm: up(l.PreFFNNorm),
			postFFNorm: up(l.PostFFNNorm), qNorm: up(l.QNorm), kNorm: up(l.KNorm), q: proj("q", l.Q), k: proj("k", l.K),
			v: proj("v", l.V), o: proj("o", l.O), gate: proj("gate", l.Gate), up: proj("up", l.Up), down: proj("down", l.Down)}
	}
	if err != nil {
		return fail(err)
	}
	return a, nil
}

func (a *g4vTower) Name() string { return "webgpu" }

// Close releases the tower's device buffers and its Context.
func (a *g4vTower) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	bufs := []*DeviceBuffer{a.patchW, a.ones}
	for i := range a.layers {
		l := &a.layers[i]
		bufs = append(bufs, l.inNorm, l.postAttnNorm, l.preFFNorm, l.postFFNorm, l.qNorm, l.kNorm,
			l.q.w, l.k.w, l.v.w, l.o.w, l.gate.w, l.up.w, l.down.w)
	}
	a.layers = nil
	return a.towerOps.close(bufs)
}

// proj runs one ClippableLinear over rows of x into a new buffer: the input clamp into a copy when a bound is finite,
// the matmul, then the output clamp.
func (a *g4vTower) proj(s *towerScope, x *DeviceBuffer, p *g4vProj, rows int) (*DeviceBuffer, error) {
	c := a.c
	clamps := g4vDefectForTest != 1
	in := x
	if clamps && p.clipIn {
		n := rows * p.in
		cp, err := s.add(c.newDevBuf(n))
		if err != nil {
			return nil, err
		}
		par := []uint32{uint32(n), math.Float32bits(p.inMin), math.Float32bits(p.inMax), 0}
		if err := c.kernelN(a.clampCopy, []*wgpu.Buffer{x.buf, cp.buf}, par, maxWG((n+63)/64), 1); err != nil {
			return nil, err
		}
		in = cp
	}
	out, err := s.add(c.matmulF32Device(in.buf, p.w.buf, rows, p.in, p.out))
	if err != nil {
		return nil, err
	}
	if clamps && p.clipOut {
		n := rows * p.out
		par := []uint32{uint32(n), math.Float32bits(p.outMin), math.Float32bits(p.outMax), 0}
		if err := c.kernelN(a.clamp, []*wgpu.Buffer{out.buf}, par, maxWG((n+63)/64), 1); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// ropeRows rotates x [np, heads, hd] in place.
func (a *g4vTower) ropeRows(x, cs, sn *DeviceBuffer, np int) error {
	nH, hd := a.cfg.NumAttentionHeads, a.cfg.HeadDim
	n := np * nH * hd / 2
	return a.c.kernelN(a.rope, []*wgpu.Buffer{x.buf, cs.buf, sn.buf}, []uint32{uint32(n), uint32(nH), uint32(hd), 0}, maxWG((n+63)/64), 1)
}

// Hidden runs the patch embed and every layer (multimodal.Gemma4TowerAccelerator): the last layer's output, [np, hidden],
// for aikit's FinishHidden. The device is waited for once per layer.
func (a *g4vTower) Hidden(patches []float32, pos [][2]int) ([]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.c == nil {
		return nil, fmt.Errorf("gpu: the Gemma 4 vision tower is closed")
	}
	a.pending = 0
	c, cfg := a.c, a.cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	np, pd := len(pos), 3*cfg.PatchSize*cfg.PatchSize
	if np == 0 || len(patches) != np*pd {
		return nil, fmt.Errorf("gpu: %d patch values for %d patches of %d", len(patches), np, pd)
	}
	last := time.Now()
	mark := func(step string) {
		if visionProfileForTest == nil {
			return
		}
		c.device.Poll(true, nil)
		now := time.Now()
		visionProfileForTest[step] += now.Sub(last)
		last = now
	}
	pt := make([]float32, len(patches))
	for i, v := range patches {
		pt[i] = 2 * (v - 0.5) // the patch embedder's own remap, as Forward does it
	}
	posX, posY := a.posX, a.posY
	if g4vDefectForTest == 2 {
		posX, posY = posY, posX
	}
	pe := make([]float32, np*H) // each patch's row of the x table plus its row of the y table
	for i, p := range pos {
		if p[0] < 0 || p[0] >= cfg.PosEmbTableSize || p[1] < 0 || p[1] >= cfg.PosEmbTableSize {
			return nil, fmt.Errorf("gpu: position id (%d,%d) out of range [0,%d)", p[0], p[1], cfg.PosEmbTableSize)
		}
		xr, yr, dst := posX[p[0]*H:p[0]*H+H], posY[p[1]*H:p[1]*H+H], pe[i*H:i*H+H]
		for d := range dst {
			dst[d] = xr[d] + yr[d]
		}
	}
	cosT, sinT := vision.Gemma4RopeTables(pos, hd, a.theta)

	var whole towerScope // the buffers that live for the whole forward
	defer whole.release()
	ptD, err := whole.add(c.UploadF32(pt))
	if err != nil {
		return nil, err
	}
	peD, err := whole.add(c.UploadF32(pe))
	if err != nil {
		return nil, err
	}
	cs, err := whole.add(c.UploadF32(cosT))
	if err != nil {
		return nil, err
	}
	sn, err := whole.add(c.UploadF32(sinT))
	if err != nil {
		return nil, err
	}
	x, err := whole.add(c.matmulF32Device(ptD.buf, a.patchW.buf, np, pd, H))
	if err != nil {
		return nil, err
	}
	if err := c.addRowsDevice(x.buf, peD.buf, np*H, H, 1); err != nil {
		return nil, err
	}
	mark("patch embed")
	for li := range a.layers {
		if err := a.layer(&a.layers[li], x, cs, sn, np, mark); err != nil {
			return nil, fmt.Errorf("gpu: Gemma 4 vision tower layer %d: %w", li, err)
		}
	}
	return c.Readback(x)
}

// layer runs one encoder layer over the residual stream x [np, hidden] in place, and waits for the device before its
// scratch is released.
func (a *g4vTower) layer(ly *g4vLayer, x, cs, sn *DeviceBuffer, np int, mark func(string)) error {
	c, cfg := a.c, a.cfg
	H, I, nH, hd := cfg.HiddenSize, cfg.IntermediateSize, cfg.NumAttentionHeads, cfg.HeadDim
	var s towerScope
	defer s.release()
	xn, err := a.rmsRows(&s, x, ly.inNorm, np, H)
	if err != nil {
		return err
	}
	mark("norms")
	q, err := a.proj(&s, xn, &ly.q, np)
	if err != nil {
		return err
	}
	k, err := a.proj(&s, xn, &ly.k, np)
	if err != nil {
		return err
	}
	v, err := a.proj(&s, xn, &ly.v, np)
	if err != nil {
		return err
	}
	mark("q, k, v projections")
	if q, err = a.rmsRows(&s, q, ly.qNorm, np*nH, hd); err != nil {
		return err
	}
	kNorm := ly.kNorm
	if g4vDefectForTest == 5 {
		kNorm = ly.qNorm
	}
	if k, err = a.rmsRows(&s, k, kNorm, np*nH, hd); err != nil {
		return err
	}
	if g4vDefectForTest != 4 {
		if v, err = a.rmsRows(&s, v, a.ones, np*nH, hd); err != nil {
			return err
		}
	}
	if err := a.ropeRows(q, cs, sn, np); err != nil {
		return err
	}
	if g4vDefectForTest != 3 {
		if err := a.ropeRows(k, cs, sn, np); err != nil {
			return err
		}
	}
	mark("head norms and RoPE")
	ctx, err := s.add(c.newDevBuf(np * H))
	if err != nil {
		return err
	}
	if err := a.attention(q, k, v, ctx, 0, np, H, nH, hd, 1); err != nil {
		return err
	}
	mark("attention")
	o, err := a.proj(&s, ctx, &ly.o, np)
	if err != nil {
		return err
	}
	mark("o projection")
	if o, err = a.rmsRows(&s, o, ly.postAttnNorm, np, H); err != nil {
		return err
	}
	if err := c.addRowsDevice(x.buf, o.buf, np*H, H, 1); err != nil {
		return err
	}
	if xn, err = a.rmsRows(&s, x, ly.preFFNorm, np, H); err != nil {
		return err
	}
	mark("norms")
	g, err := a.proj(&s, xn, &ly.gate, np)
	if err != nil {
		return err
	}
	u, err := a.proj(&s, xn, &ly.up, np)
	if err != nil {
		return err
	}
	mark("gate and up projections")
	if err := c.kernelN(a.geluMul, []*wgpu.Buffer{g.buf, u.buf}, []uint32{uint32(np * I), 0, 0, 0}, maxWG((np*I+63)/64), 1); err != nil {
		return err
	}
	mark("gated GELU")
	f, err := a.proj(&s, g, &ly.down, np)
	if err != nil {
		return err
	}
	mark("down projection")
	if f, err = a.rmsRows(&s, f, ly.postFFNorm, np, H); err != nil {
		return err
	}
	if err := c.addRowsDevice(x.buf, f.buf, np*H, H, 1); err != nil {
		return err
	}
	c.device.Poll(true, nil)
	mark("norms")
	return nil
}
