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

func (s *int8Regs) setFloat(r vision.TowerReg) { s.int8[r], s.stale[r] = false, false }

func (s *int8Regs) setInt8(r vision.TowerReg) { s.int8[r], s.stale[r] = true, true }

// onlyLinearsRead reports whether every op after prog[i] that reads r, up to r's next write, is a linear reading it as
// its input: the condition for prog[i] to write r in int8 only.
func onlyLinearsRead(prog vision.TowerBlock, i int, r vision.TowerReg) bool {
	for _, op := range prog[i+1:] {
		switch {
		case op.Kind == vision.TowerAttention && (r == vision.TowerQ || r == vision.TowerK || r == vision.TowerV):
			return false
		case op.ReadsIn() && op.In == r && op.Kind != vision.TowerLinear && op.Kind != vision.TowerLinearAdd:
			return false
		case op.Out == r && op.Kind != vision.TowerLinear && op.Kind != vision.TowerNorm:
			return false // it reads r in place (a linear-add's residual, the GELU)
		case op.Out == r:
			return true
		}
	}
	return true
}

// walkBlock queues one block's program over rows, each op through the tower's own kernels: a norm through
// layernorm_quant, written int8 for the linears that read it; a linear through the W8A8 GEMV, quantizing a float input
// first (quant_vec); attention through attend; the GELU through gelu_quant, written int8 for its linear; a linear-add's
// residual through the host round trip. An op that does not fit its kind (vision.TowerOp.CheckRegs), its weight
// (vision.TowerOp.CheckShape) or this tower (a norm or GELU whose output a non-linear reads, a float read of a register
// held int8 only, a kind it does not run) is an error before it is queued.
func (r *VisionEncoder) walkBlock(q *Queue, prog vision.TowerBlock, L *visionLayer, s *int8Regs, rows int,
	attend func(qb, kb, vb, out Buffer, scale float32) error) error {
	hidden, inter := r.hidden, r.inter
	s.init()
	width := func(n vision.TowerReg) int { return vision.TowerRegWidth(n, hidden, inter) }
	floatOf := func(n vision.TowerReg) (Buffer, error) {
		b, ok := s.f[n]
		if !ok || s.stale[n] {
			return Buffer{}, fmt.Errorf("register %d is held int8 only on this tower", n)
		}
		return b, nil
	}
	for i, op := range prog {
		if err := op.CheckRegs(); err != nil {
			return fmt.Errorf("op %d: %w", i, err)
		}
		switch op.Kind {
		case vision.TowerNorm:
			w, b, ok := L.towerNorm(op.W)
			if !ok || op.Form != vision.TowerLayerNorm {
				return fmt.Errorf("op %d: norm of weight %d, form %d", i, op.W, op.Form)
			}
			if !onlyLinearsRead(prog, i, op.Out) {
				return fmt.Errorf("op %d: this tower writes a norm in int8 only, and register %d has a reader that is not a linear", i, op.Out)
			}
			in, err := floatOf(op.In)
			if err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
			p := s.quantized(op.Out, hidden)
			if err := r.layerNormQuantEps(q, in, w, b, hidden, float32(op.Eps), p[0], p[1], rows); err != nil {
				return err
			}
			s.setInt8(op.Out)
		case vision.TowerLinear, vision.TowerLinearAdd:
			wt, bias, ok := L.towerLinear(op.W)
			if !ok {
				return fmt.Errorf("op %d: linear of weight %d", i, op.W)
			}
			if err := op.CheckShape(wt.N, wt.K, hidden, inter); err != nil {
				return fmt.Errorf("op %d: %w", i, err)
			}
			var dst, res Buffer
			if op.Kind == vision.TowerLinear {
				var ok bool
				if dst, ok = s.f[op.Out]; !ok {
					return fmt.Errorf("op %d (%s): register %d has no buffer", i, op.Kind, op.Out)
				}
			} else {
				var err error
				if res, err = floatOf(op.Out); err != nil {
					return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
				}
				dst = s.tmp
			}
			var p [2]Buffer
			if s.int8[op.In] {
				p = s.q8[op.In]
			} else {
				in, err := floatOf(op.In)
				if err != nil {
					return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
				}
				p = s.quantized(op.In, width(op.In))
				if err := r.quant(q, in, width(op.In), p[0], p[1], rows); err != nil {
					return err
				}
				s.int8[op.In] = true
			}
			if err := r.bGemv(q, wt, p[0], p[1], bias, dst, rows); err != nil {
				return err
			}
			if op.Kind == vision.TowerLinearAdd {
				if err := q.Sync(); err != nil {
					return err
				}
				if err := addInPlaceHost(q, res, dst, rows*width(op.Out)); err != nil { // residual
					return err
				}
			}
			s.setFloat(op.Out)
		case vision.TowerAttention:
			var qkv [3]Buffer
			for j, n := range []vision.TowerReg{vision.TowerQ, vision.TowerK, vision.TowerV} {
				var err error
				if qkv[j], err = floatOf(n); err != nil {
					return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
				}
			}
			out, ok := s.f[op.Out]
			if !ok {
				return fmt.Errorf("op %d (%s): register %d has no buffer", i, op.Kind, op.Out)
			}
			if err := attend(qkv[0], qkv[1], qkv[2], out, float32(op.Scale)); err != nil {
				return err
			}
			s.setFloat(op.Out)
		case vision.TowerGELUTanh:
			if !onlyLinearsRead(prog, i, op.Out) {
				return fmt.Errorf("op %d: this tower writes the GELU in int8 only, and register %d has a reader that is not a linear", i, op.Out)
			}
			in, err := floatOf(op.Out)
			if err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
			p := s.quantized(op.Out, width(op.Out))
			if err := r.geluQuant(q, in, width(op.Out), p[0], p[1], rows); err != nil {
				return err
			}
			s.setInt8(op.Out)
		default:
			return fmt.Errorf("op %d: kind %s not implemented on the CUDA int8 SigLIP tower", i, op.Kind)
		}
	}
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
