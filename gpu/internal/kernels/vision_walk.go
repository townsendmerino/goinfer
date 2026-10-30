//go:build gpu

package kernels

import (
	"fmt"
	"math"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// visProj is one projection in whichever form the tower was uploaded (VisionEncoder.ForwardPatches' proj).
type visProj func(src *DeviceBuffer, w8 *ResidentW8A8, wf, bias *DeviceBuffer, M, K, N int) (*DeviceBuffer, error)

// walkBlock queues one block's program over the residual stream h [np, hidden], each op through the tower's existing
// operation (VisionEncoder.ln, proj, attend, the GELU kernel, addRowsDevice). Every register but h is a buffer the op
// that writes it allocates; walkBlock returns them all, on success and on error, for the caller to release after the
// layer's sync. defect is visionDefectForTest's planted defect for this layer, 0 for none.
func (ve *VisionEncoder) walkBlock(prog vision.TowerBlock, L *visLayer, h *DeviceBuffer, defect int, proj visProj, mark func(string)) ([]*DeviceBuffer, error) {
	c, w := ve.c, ve.w
	np, hidden, inter := w.NumPatches, w.Hidden, w.Inter
	regs := map[vision.TowerReg]*DeviceBuffer{vision.TowerX: h}
	var keep []*DeviceBuffer
	width := func(r vision.TowerReg) int {
		if r == vision.TowerMid {
			return inter
		}
		return hidden
	}
	// A register read before this block writes it is a new zeroed buffer, as the CPU and CUDA read their scratch: a program
	// in the wrong order gives a wrong output, not an error.
	reg := func(r vision.TowerReg) (*DeviceBuffer, error) {
		if b := regs[r]; b != nil {
			return b, nil
		}
		if r < vision.TowerX || r > vision.TowerMid {
			return nil, fmt.Errorf("register %d unknown", r)
		}
		b, err := c.newDevBuf(np * width(r))
		if err != nil {
			return nil, err
		}
		keep = append(keep, b)
		regs[r] = b
		return b, nil
	}
	for i, op := range prog {
		if err := op.CheckRegs(); err != nil {
			return keep, fmt.Errorf("op %d: %w", i, err)
		}
		if err := ve.checkShape(op, L); err != nil {
			return keep, fmt.Errorf("op %d: %w", i, err)
		}
		var in *DeviceBuffer
		if op.Kind != vision.TowerAttention && op.Kind != vision.TowerGELUTanh { // they read no In register
			var err error
			if in, err = reg(op.In); err != nil {
				return keep, fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		}
		switch op.Kind {
		case vision.TowerNorm:
			nw, nb, ok := L.towerNorm(op.W)
			if !ok || op.Form != vision.TowerLayerNorm {
				return keep, fmt.Errorf("op %d: norm of weight %d, form %d", i, op.W, op.Form)
			}
			dst, err := ve.c.newDevBuf(np * hidden)
			if err != nil {
				return keep, err
			}
			keep = append(keep, dst)
			if err := c.layerNormRowsDevice(in.buf, nw.buf, nb.buf, dst.buf, np, hidden, float32(op.Eps)); err != nil {
				return keep, err
			}
			regs[op.Out] = dst
			mark("layernorm")
		case vision.TowerLinear, vision.TowerLinearAdd:
			w8, wf, bias, ok := L.towerLinear(op.W)
			if !ok {
				return keep, fmt.Errorf("op %d: linear of weight %d", i, op.W)
			}
			switch {
			case defect == 2 && op.W == vision.TowerWQ:
				bias = nil
			case defect == 3 && op.W == vision.TowerWQ:
				wf = L.kf
			case defect == 3 && op.W == vision.TowerWK:
				wf = L.qf
			}
			K, N := width(op.In), width(op.Out)
			d, err := proj(in, w8, wf, bias, np, K, N)
			if err != nil {
				return keep, err
			}
			keep = append(keep, d)
			if op.Kind == vision.TowerLinear {
				regs[op.Out] = d
				mark(visStep(op.W))
				continue
			}
			out, err := reg(op.Out)
			if err != nil {
				return keep, fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
			if !(defect == 5 && op.W == vision.TowerWFC2) {
				if err := c.addRowsDevice(out.buf, d.buf, np*N, N, 1); err != nil { // out += d
					return keep, err
				}
			}
			mark(visStep(op.W))
		case vision.TowerAttention:
			var qkv [3]*DeviceBuffer
			for j, r := range []vision.TowerReg{vision.TowerQ, vision.TowerK, vision.TowerV} {
				var err error
				if qkv[j], err = reg(r); err != nil {
					return keep, fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
				}
			}
			att, err := ve.attend(qkv[0], qkv[1], qkv[2], float32(op.Scale), mark)
			if att != nil {
				keep = append(keep, att)
			}
			if err != nil {
				return keep, err
			}
			regs[op.Out] = att
		case vision.TowerGELUTanh:
			out, err := reg(op.Out)
			if err != nil {
				return keep, fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
			if defect != 4 {
				n := np * width(op.Out)
				if err := c.inplaceKernel(c.geluPipeline, c.geluLayout, out.buf, []uint32{uint32(n), 0, 0, 0}, maxWG((n+63)/64)); err != nil {
					return keep, err
				}
			}
			mark("gelu")
		default:
			return keep, fmt.Errorf("op %d: kind %s not implemented on WebGPU", i, op.Kind)
		}
	}
	return keep, nil
}

// checkShape refuses, before anything is allocated or queued, a kind walkBlock does not implement and a linear whose
// weight, in whichever form the tower was uploaded, does not fit its registers (op.CheckShape). The float32 form's buffers
// carry only their length, so its [out, in] is the shape NewVisionEncoderF32 required of that weight at upload.
func (ve *VisionEncoder) checkShape(op vision.TowerOp, L *visLayer) error {
	switch op.Kind {
	case vision.TowerNorm, vision.TowerAttention, vision.TowerGELUTanh:
		return nil
	case vision.TowerLinear, vision.TowerLinearAdd:
	default:
		return fmt.Errorf("kind %s not implemented on WebGPU's SigLIP tower", op.Kind)
	}
	w8, wf, _, ok := L.towerLinear(op.W)
	if !ok {
		return fmt.Errorf("linear of weight %d", op.W)
	}
	if w8 != nil {
		return op.CheckShape(w8.rows, w8.cols, ve.w.Hidden, ve.w.Inter)
	}
	if wf == nil {
		return fmt.Errorf("linear of weight %d has no matrix", op.W)
	}
	out, in := ve.w.Hidden, ve.w.Hidden // q, k, v and o
	switch op.W {
	case vision.TowerWFC1:
		out = ve.w.Inter
	case vision.TowerWFC2:
		in = ve.w.Inter
	}
	if wf.n != out*in {
		return fmt.Errorf("linear of weight %d holds %d values, not [%d, %d]", op.W, wf.n, out, in)
	}
	return op.CheckShape(out, in, ve.w.Hidden, ve.w.Inter)
}

// visStep is a projection's name in visionProfileForTest's profile.
func visStep(w vision.TowerWeight) string {
	switch w {
	case vision.TowerWQ, vision.TowerWK, vision.TowerWV:
		return "q, k, v projections"
	case vision.TowerWO:
		return "o projection and residual"
	case vision.TowerWFC1:
		return "fc1"
	}
	return "fc2"
}

// attend is bidirectional attention of q, k and v [np, hidden] at scale, head by head: a new [np, hidden] buffer, which
// the caller releases (also when attend returns an error with it).
func (ve *VisionEncoder) attend(q, k, v *DeviceBuffer, scale float32, mark func(string)) (*DeviceBuffer, error) {
	c, w := ve.c, ve.w
	np, hidden, hd := w.NumPatches, w.Hidden, w.HeadDim
	att, err := c.newDevBuf(np * hidden)
	if err != nil {
		return nil, err
	}
	for head := range w.NumHeads {
		off := head * hd
		// Each head's buffers are released at the end of its iteration; hrel releases whatever it has allocated so far,
		// on the success path and every error return.
		var hbufs []*DeviceBuffer
		hrel := func() {
			for _, b := range hbufs {
				b.Release()
			}
		}
		htrack := func(d *DeviceBuffer, e error) (*DeviceBuffer, error) {
			if e != nil {
				return nil, e
			}
			hbufs = append(hbufs, d)
			return d, nil
		}
		qh, e1 := htrack(ve.gatherHead(q, np, hd, hidden, off, 0))
		kh, e2 := htrack(ve.gatherHead(k, np, hd, hidden, off, 0))
		vt, e3 := htrack(ve.gatherHead(v, np, hd, hidden, off, 2)) // [hd, np]
		if e1 != nil || e2 != nil || e3 != nil {
			hrel()
			return att, fmt.Errorf("gather: %v %v %v", e1, e2, e3)
		}
		mark("per head: gather q, k, v")
		scores, e := htrack(c.matmulF32Device(qh.buf, kh.buf, np, hd, np)) // [np,np]
		if e != nil {
			hrel()
			return att, e
		}
		mark("per head: scores = q·kᵀ")
		if e := c.inplaceKernel(c.softmaxPipeline, c.softmaxLayout, scores.buf, []uint32{uint32(np), uint32(np), math.Float32bits(scale), 0}, np); e != nil {
			hrel()
			return att, e
		}
		mark("per head: softmax")
		oh, e := htrack(c.matmulF32Device(scores.buf, vt.buf, np, np, hd)) // [np,hd]
		if e != nil {
			hrel()
			return att, e
		}
		mark("per head: scores·v")
		if e := c.copyHeadDevice(oh.buf, att.buf, np, hd, hidden, off, 1); e != nil { // scatter
			hrel()
			return att, e
		}
		mark("per head: scatter")
		hrel()
	}
	return att, nil
}

func (L *visLayer) towerNorm(w vision.TowerWeight) (weight, bias *DeviceBuffer, ok bool) {
	switch w {
	case vision.TowerNorm1:
		return L.ln1w, L.ln1b, true
	case vision.TowerNorm2:
		return L.ln2w, L.ln2b, true
	}
	return nil, nil, false
}

// towerLinear returns a projection in both forms (one of w8 and wf is nil) and its bias.
func (L *visLayer) towerLinear(w vision.TowerWeight) (w8 *ResidentW8A8, wf, bias *DeviceBuffer, ok bool) {
	switch w {
	case vision.TowerWQ:
		return L.qw, L.qf, L.qb, true
	case vision.TowerWK:
		return L.kw, L.kf, L.kb, true
	case vision.TowerWV:
		return L.vw, L.vf, L.vb, true
	case vision.TowerWO:
		return L.ow, L.of, L.ob, true
	case vision.TowerWFC1:
		return L.fc1w, L.fc1f, L.fc1b, true
	case vision.TowerWFC2:
		return L.fc2w, L.fc2f, L.fc2b, true
	}
	return nil, nil, nil, false
}
