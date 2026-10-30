//go:build cuda

package cuda

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// int8Regs are the int8 SigLIP tower's registers. Each is held as float32 (f), as int8 rows and their scales (q8), or both:
// the int8 form is the walker's storage for a register only linears read, not part of the program (docs/tasks/
// task-subsystems-2026-10.md §8, "CUDA's int8 SigLIP tower"). A register starts with its float buffer current.
type int8Regs struct {
	f     map[vision.TowerReg]Buffer
	q8    map[vision.TowerReg][2]Buffer
	int8  map[vision.TowerReg]bool // the int8 form is current
	stale map[vision.TowerReg]bool // the float form is not current
	tmp   Buffer                   // a linear-add's product, [rows, hidden]
	alloc func(width int) [2]Buffer
}

func (s *int8Regs) init() {
	if s.q8 == nil {
		s.q8, s.int8, s.stale = map[vision.TowerReg][2]Buffer{}, map[vision.TowerReg]bool{}, map[vision.TowerReg]bool{}
	}
}

func (s *int8Regs) quantized(r vision.TowerReg, width int) [2]Buffer {
	p, ok := s.q8[r]
	if !ok {
		p = s.alloc(width)
		s.q8[r] = p
	}
	return p
}

// linearsOnly reports whether r, written by prog[i], is read by at least one linear as its input before its next write
// (or the block's end), and by no other op: the condition for prog[i] to write r in int8 only.
func linearsOnly(prog vision.TowerBlock, i int, r vision.TowerReg) bool {
	read := false
	for _, op := range prog[i+1:] {
		switch {
		case op.Kind == vision.TowerAttention && (r == vision.TowerQ || r == vision.TowerK || r == vision.TowerV):
			return false
		case op.ReadsIn() && op.In == r:
			if op.Kind != vision.TowerLinear && op.Kind != vision.TowerLinearAdd {
				return false
			}
			read = true
			if op.Out == r { // a linear writing its own input: r's next value starts here
				return read
			}
		case op.Out == r && op.Kind != vision.TowerLinear && op.Kind != vision.TowerNorm:
			return false // it reads r in place (a linear-add's residual, the GELU)
		case op.Out == r:
			return read
		}
	}
	return read
}

// int8Step is one op of a block as planned: its weights, and whether a linear quantizes its float input first.
type int8Step struct {
	op         vision.TowerOp
	nw, nb     Buffer // a norm's
	wt         cudaWQ // a linear's
	bias       Buffer
	quantizeIn bool
}

// planBlock checks a whole block against the tower and the registers' current forms in s, without queueing anything,
// and returns its steps and the forms the block leaves: int8 maps the registers it leaves int8-current, stale those whose
// float form it leaves out of date. It refuses an op that does not fit its kind (vision.TowerOp.CheckRegs), its weight
// (vision.TowerOp.CheckShape) or this tower: a norm or GELU whose output is not read only by linears (they are written in
// int8 only), x written in int8, x left without a current float form, a float read of a register held in int8 only or
// with no float buffer, a norm that is not hidden-wide, and a kind the tower does not run.
func (r *VisionEncoder) planBlock(prog vision.TowerBlock, L *visionLayer, s *int8Regs) (steps []int8Step, int8, stale map[vision.TowerReg]bool, err error) {
	hidden, inter := r.hidden, r.inter
	width := func(n vision.TowerReg) int { return vision.TowerRegWidth(n, hidden, inter) }
	int8, stale = map[vision.TowerReg]bool{}, map[vision.TowerReg]bool{}
	for k, v := range s.int8 {
		int8[k] = v
	}
	for k, v := range s.stale {
		stale[k] = v
	}
	floatOK := func(n vision.TowerReg) error {
		if _, ok := s.f[n]; !ok {
			return fmt.Errorf("register %d has no float buffer on this tower", n)
		}
		if stale[n] {
			return fmt.Errorf("register %d is held int8 only here", n)
		}
		return nil
	}
	writeInt8 := func(i int, n vision.TowerReg) error {
		if n == vision.TowerX {
			return fmt.Errorf("x is never held int8 only on this tower")
		}
		if !linearsOnly(prog, i, n) {
			return fmt.Errorf("register %d would be written int8 only, and a linear alone must read it", n)
		}
		int8[n], stale[n] = true, true
		return nil
	}
	writeFloat := func(n vision.TowerReg) error {
		if _, ok := s.f[n]; !ok {
			return fmt.Errorf("register %d has no float buffer on this tower", n)
		}
		int8[n], stale[n] = false, false
		return nil
	}
	for i, op := range prog {
		if err := op.CheckRegs(); err != nil {
			return nil, nil, nil, fmt.Errorf("op %d: %w", i, err)
		}
		st := int8Step{op: op}
		var err error
		switch op.Kind {
		case vision.TowerNorm:
			var ok bool
			if st.nw, st.nb, ok = L.towerNorm(op.W); !ok || op.Form != vision.TowerLayerNorm {
				err = fmt.Errorf("norm of weight %d, form %d", op.W, op.Form)
			} else if width(op.In) != hidden || width(op.Out) != hidden {
				err = fmt.Errorf("a norm runs hidden-wide, not %d -> %d", width(op.In), width(op.Out))
			} else if err = floatOK(op.In); err == nil {
				err = writeInt8(i, op.Out)
			}
		case vision.TowerLinear, vision.TowerLinearAdd:
			var ok bool
			if st.wt, st.bias, ok = L.towerLinear(op.W); !ok {
				err = fmt.Errorf("linear of weight %d", op.W)
				break
			}
			if err = op.CheckShape(st.wt.N, st.wt.K, hidden, inter); err != nil {
				break
			}
			if !int8[op.In] {
				if err = floatOK(op.In); err != nil {
					break
				}
				st.quantizeIn, int8[op.In] = true, true
			}
			if op.Kind == vision.TowerLinearAdd {
				if err = floatOK(op.Out); err != nil {
					break
				}
			}
			err = writeFloat(op.Out)
		case vision.TowerAttention:
			for _, n := range []vision.TowerReg{vision.TowerQ, vision.TowerK, vision.TowerV} {
				if err = floatOK(n); err != nil {
					break
				}
			}
			if err == nil {
				err = writeFloat(op.Out)
			}
		case vision.TowerGELUTanh:
			if err = floatOK(op.Out); err == nil {
				err = writeInt8(i, op.Out)
			}
		default:
			err = fmt.Errorf("kind %s not implemented on the CUDA int8 SigLIP tower", op.Kind)
		}
		if err != nil {
			return nil, nil, nil, fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
		}
		steps = append(steps, st)
	}
	if stale[vision.TowerX] {
		return nil, nil, nil, fmt.Errorf("the block leaves x without a current float form")
	}
	return steps, int8, stale, nil
}

// walkBlock queues one block's program over rows, after planBlock has checked the whole of it: a norm through
// layernorm_quant into int8, a linear through the W8A8 GEMV (quantizing a float input first with quant_vec), attention
// through attend, the GELU through gelu_quant into int8, a linear-add's residual through the host round trip.
func (r *VisionEncoder) walkBlock(q *Queue, prog vision.TowerBlock, L *visionLayer, s *int8Regs, rows int,
	attend func(qb, kb, vb, out Buffer, scale float32) error) error {
	s.init()
	steps, int8, stale, err := r.planBlock(prog, L, s)
	if err != nil {
		return err
	}
	width := func(n vision.TowerReg) int { return vision.TowerRegWidth(n, r.hidden, r.inter) }
	for _, st := range steps {
		op := st.op
		switch op.Kind {
		case vision.TowerNorm:
			p := s.quantized(op.Out, width(op.Out))
			if err := r.layerNormQuantEps(q, s.f[op.In], st.nw, st.nb, width(op.In), float32(op.Eps), p[0], p[1], rows); err != nil {
				return err
			}
		case vision.TowerLinear, vision.TowerLinearAdd:
			p := s.quantized(op.In, width(op.In))
			if st.quantizeIn {
				if err := r.quant(q, s.f[op.In], width(op.In), p[0], p[1], rows); err != nil {
					return err
				}
			}
			dst := s.f[op.Out]
			if op.Kind == vision.TowerLinearAdd {
				dst = s.tmp
			}
			if err := r.bGemv(q, st.wt, p[0], p[1], st.bias, dst, rows); err != nil {
				return err
			}
			if op.Kind == vision.TowerLinearAdd {
				if err := q.Sync(); err != nil {
					return err
				}
				if err := addInPlaceHost(q, s.f[op.Out], dst, rows*width(op.Out)); err != nil { // residual
					return err
				}
			}
		case vision.TowerAttention:
			if err := attend(s.f[vision.TowerQ], s.f[vision.TowerK], s.f[vision.TowerV], s.f[op.Out], float32(op.Scale)); err != nil {
				return err
			}
		case vision.TowerGELUTanh:
			p := s.quantized(op.Out, width(op.Out))
			if err := r.geluQuant(q, s.f[op.Out], width(op.Out), p[0], p[1], rows); err != nil {
				return err
			}
		}
	}
	s.int8, s.stale = int8, stale
	return nil
}

func (L *visionLayer) towerNorm(w vision.TowerWeight) (weight, bias Buffer, ok bool) {
	switch w {
	case vision.TowerNorm1:
		return L.ln1w, L.ln1b, true
	case vision.TowerNorm2:
		return L.ln2w, L.ln2b, true
	}
	return Buffer{}, Buffer{}, false
}

func (L *visionLayer) towerLinear(w vision.TowerWeight) (cudaWQ, Buffer, bool) {
	switch w {
	case vision.TowerWQ:
		return L.qw, L.qb, true
	case vision.TowerWK:
		return L.kw, L.kb, true
	case vision.TowerWV:
		return L.vw, L.vb, true
	case vision.TowerWO:
		return L.ow, L.ob, true
	case vision.TowerWFC1:
		return L.fc1w, L.fc1b, true
	case vision.TowerWFC2:
		return L.fc2w, L.fc2b, true
	}
	return cudaWQ{}, Buffer{}, false
}
