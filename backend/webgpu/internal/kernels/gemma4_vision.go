//go:build gpu

package kernels

import (
	"fmt"
	"github.com/townsendmerino/goinfer/internal/mem"
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
		d, err = c.UploadF32On(mem.LineCompanion, x)
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
	ptD, err := whole.add(c.UploadF32On(mem.LineCompanion, pt))
	if err != nil {
		return nil, err
	}
	peD, err := whole.add(c.UploadF32On(mem.LineCompanion, pe))
	if err != nil {
		return nil, err
	}
	cs, err := whole.add(c.UploadF32On(mem.LineCompanion, cosT))
	if err != nil {
		return nil, err
	}
	sn, err := whole.add(c.UploadF32On(mem.LineCompanion, sinT))
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
	prog := vision.Gemma4Program(cfg)
	for li := range a.layers {
		if err := a.walkBlock(prog, &a.layers[li], x, cs, sn, np, mark); err != nil {
			return nil, fmt.Errorf("gpu: Gemma 4 vision tower layer %d: %w", li, err)
		}
	}
	return c.Readback(x)
}

// walkBlock runs one block's program over the residual stream x [np, hidden] in place, each op through the tower's
// existing operation, and waits for the device before its scratch is released: a norm and a per-head norm through
// rmsRowsEps at the op's epsilon (the ones buffer for v's unscaled form), a linear through proj with its clamps, axial
// RoPE through ropeRows on q or k, attention over every row at the op's scale, the GELU product through its kernel, a
// residual add through addRowsDevice. Every register but x is a buffer the op that writes it allocates; one read before
// it is written is a new zeroed buffer. An op that does not fit its kind (vision.TowerOp.CheckRegs), its weight
// (vision.TowerOp.CheckShape) or this tower is an error before anything is allocated for it. g4vDefectForTest's
// defects act at their ops.
func (a *g4vTower) walkBlock(prog vision.TowerBlock, ly *g4vLayer, x, cs, sn *DeviceBuffer, np int, mark func(string)) error {
	c, cfg := a.c, a.cfg
	H, I, nH, hd := cfg.HiddenSize, cfg.IntermediateSize, cfg.NumAttentionHeads, cfg.HeadDim
	var s towerScope
	defer s.release()
	regs := map[vision.TowerReg]*DeviceBuffer{vision.TowerX: x}
	reg := func(n vision.TowerReg) (*DeviceBuffer, error) {
		if d := regs[n]; d != nil {
			return d, nil
		}
		w := vision.TowerRegWidth(n, H, I)
		if w == 0 {
			return nil, fmt.Errorf("register %d unknown", n)
		}
		d, err := s.add(c.newDevBuf(np * w))
		if err != nil {
			return nil, err
		}
		regs[n] = d
		return d, nil
	}
	for i, op := range prog {
		if err := op.CheckRegs(); err != nil {
			return fmt.Errorf("op %d: %w", i, err)
		}
		if err := a.checkOp(op, ly); err != nil {
			return fmt.Errorf("op %d: %w", i, err)
		}
		var in, out *DeviceBuffer
		var err error
		if op.ReadsIn() {
			if in, err = reg(op.In); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		}
		if op.Kind != vision.TowerNorm && op.Kind != vision.TowerLinear && op.Kind != vision.TowerAttention {
			if out, err = reg(op.Out); err != nil { // the ops that update their output in place
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		}
		eps := float32(op.Eps)
		switch op.Kind {
		case vision.TowerNorm:
			w, _ := ly.towerNorm(op.W)
			d, err := a.rmsRowsEps(&s, in, w, np, H, eps)
			if err != nil {
				return err
			}
			regs[op.Out] = d
			mark("norms")
		case vision.TowerHeadNorm:
			w := a.ones
			if op.Form == vision.TowerRMSNorm {
				w, _ = ly.towerNorm(op.W)
				if op.W == vision.TowerKNorm && g4vDefectForTest == 5 {
					w = ly.qNorm
				}
			} else if g4vDefectForTest == 4 { // v's unscaled norm skipped
				continue
			}
			d, err := a.rmsRowsEps(&s, out, w, np*nH, hd, eps)
			if err != nil {
				return err
			}
			regs[op.Out] = d
		case vision.TowerLinear:
			p, _ := ly.towerLinear(op.W)
			d, err := a.proj(&s, in, p, np)
			if err != nil {
				return err
			}
			regs[op.Out] = d
			mark("projections")
		case vision.TowerRoPEAxial:
			if op.Out == vision.TowerK && g4vDefectForTest == 3 {
				continue
			}
			if err := a.ropeRows(out, cs, sn, np); err != nil {
				return err
			}
			mark("head norms and RoPE")
		case vision.TowerAttention:
			var qkv [3]*DeviceBuffer
			for j, r := range []vision.TowerReg{vision.TowerQ, vision.TowerK, vision.TowerV} {
				if qkv[j], err = reg(r); err != nil {
					return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
				}
			}
			ctx, err := s.add(c.newDevBuf(np * H))
			if err != nil {
				return err
			}
			if err := a.attention(qkv[0], qkv[1], qkv[2], ctx, 0, np, H, nH, hd, float32(op.Scale)); err != nil {
				return err
			}
			regs[op.Out] = ctx
			mark("attention")
		case vision.TowerGELUMul:
			n := np * I
			if err := c.kernelN(a.geluMul, []*wgpu.Buffer{out.buf, in.buf}, []uint32{uint32(n), 0, 0, 0}, maxWG((n+63)/64), 1); err != nil {
				return err
			}
			mark("gated GELU")
		case vision.TowerAdd:
			if err := c.addRowsDevice(out.buf, in.buf, np*H, H, 1); err != nil {
				return err
			}
		}
	}
	c.device.Poll(true, nil)
	mark("norms")
	return nil
}

// checkOp refuses an op this tower does not take, before anything is allocated for it: a kind it does not implement; a
// norm that is not RMSNorm or names a weight the layer lacks; a per-head norm off q, k and v, or RoPE off q and k; a
// linear whose weight is missing or does not fit its registers (vision.TowerOp.CheckShape).
func (a *g4vTower) checkOp(op vision.TowerOp, ly *g4vLayer) error {
	switch op.Kind {
	case vision.TowerNorm:
		if _, ok := ly.towerNorm(op.W); !ok || op.Form != vision.TowerRMSNorm {
			return fmt.Errorf("norm of weight %d, form %d", op.W, op.Form)
		}
	case vision.TowerHeadNorm:
		if op.Out != vision.TowerQ && op.Out != vision.TowerK && op.Out != vision.TowerV {
			return fmt.Errorf("the per-head norm runs on q, k or v, not register %d", op.Out)
		}
		switch op.Form {
		case vision.TowerRMSNormUnscaled:
		case vision.TowerRMSNorm:
			if _, ok := ly.towerNorm(op.W); !ok {
				return fmt.Errorf("per-head norm of weight %d", op.W)
			}
		default:
			return fmt.Errorf("per-head norm form %d", op.Form)
		}
	case vision.TowerLinear:
		p, ok := ly.towerLinear(op.W)
		if !ok {
			return fmt.Errorf("linear of weight %d", op.W)
		}
		return op.CheckShape(p.out, p.in, a.cfg.HiddenSize, a.cfg.IntermediateSize)
	case vision.TowerRoPEAxial:
		if op.Out != vision.TowerQ && op.Out != vision.TowerK {
			return fmt.Errorf("RoPE runs on q or k, not register %d", op.Out)
		}
	case vision.TowerAttention, vision.TowerGELUMul, vision.TowerAdd:
	default:
		return fmt.Errorf("kind %s not implemented on the WebGPU Gemma 4 tower", op.Kind)
	}
	return nil
}

func (ly *g4vLayer) towerNorm(w vision.TowerWeight) (*DeviceBuffer, bool) {
	switch w {
	case vision.TowerNorm1:
		return ly.inNorm, true
	case vision.TowerNormPostAttn:
		return ly.postAttnNorm, true
	case vision.TowerNorm2:
		return ly.preFFNorm, true
	case vision.TowerNormPostMLP:
		return ly.postFFNorm, true
	case vision.TowerQNorm:
		return ly.qNorm, true
	case vision.TowerKNorm:
		return ly.kNorm, true
	}
	return nil, false
}

func (ly *g4vLayer) towerLinear(w vision.TowerWeight) (*g4vProj, bool) {
	switch w {
	case vision.TowerWQ:
		return &ly.q, true
	case vision.TowerWK:
		return &ly.k, true
	case vision.TowerWV:
		return &ly.v, true
	case vision.TowerWO:
		return &ly.o, true
	case vision.TowerWGate:
		return &ly.gate, true
	case vision.TowerWUp:
		return &ly.up, true
	case vision.TowerWDown:
		return &ly.down, true
	}
	return nil, false
}
