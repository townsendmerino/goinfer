//go:build gpu

package kernels

import (
	"fmt"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/internal/vision"
)

// walkBlock runs one block's program over the residual stream x [np, hidden] in place, each op through the tower's
// existing operation (layerNormRowsDevice, rmsRowsEps, linear, the fused qkv as the three split linears, the RoPE kernel,
// windowAttention or attention per segment, the GELU and SiLU-product kernels, addRowsDevice), and waits for the device
// before its scratch is released. Every register but x is a buffer the op that writes it allocates; one read before it
// is written is a new zeroed buffer, so a program in the wrong order gives a wrong output, not an error. segs and seg are
// block's.
func (a *gridVTower) walkBlock(prog vision.TowerBlock, b *gvBlock, x, cs, sn *DeviceBuffer, np int, segs []int, seg *DeviceBuffer) error {
	c := a.c
	H, nH, hd := a.hidden, a.heads, a.hd
	var s towerScope
	defer s.release()
	width := func(n vision.TowerReg) int {
		if n == vision.TowerMid || n == vision.TowerMid2 {
			return a.inter
		}
		return H
	}
	regs := map[vision.TowerReg]*DeviceBuffer{vision.TowerX: x}
	reg := func(n vision.TowerReg) (*DeviceBuffer, error) {
		if d := regs[n]; d != nil {
			return d, nil
		}
		if n < vision.TowerX || n > vision.TowerMid2 {
			return nil, fmt.Errorf("register %d unknown", n)
		}
		d, err := s.add(c.newDevBuf(np * width(n)))
		if err != nil {
			return nil, err
		}
		regs[n] = d
		return d, nil
	}
	for i, op := range prog {
		var in, out *DeviceBuffer
		var err error
		if op.ReadsIn() {
			if in, err = reg(op.In); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		}
		if op.Kind != vision.TowerLinearQKV && op.Kind != vision.TowerNorm && op.Kind != vision.TowerLinear && op.Kind != vision.TowerAttention {
			if out, err = reg(op.Out); err != nil { // the ops that update their output in place
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		}
		switch op.Kind {
		case vision.TowerNorm, vision.TowerHeadNorm:
			w, bias, ok := b.towerNorm(op.W)
			if !ok {
				return fmt.Errorf("op %d: norm of weight %d", i, op.W)
			}
			eps := float32(op.Eps)
			var d *DeviceBuffer
			switch {
			case op.Kind == vision.TowerNorm && op.Form == vision.TowerLayerNorm:
				if d, err = s.add(c.newDevBuf(np * H)); err == nil {
					err = c.layerNormRowsDevice(in.buf, w.buf, bias.buf, d.buf, np, H, eps)
				}
			case op.Kind == vision.TowerNorm && op.Form == vision.TowerRMSNorm:
				d, err = a.rmsRowsEps(&s, in, w, np, H, eps)
			case op.Kind == vision.TowerHeadNorm && op.Form == vision.TowerRMSNorm:
				if a.planted.noQKNorm {
					continue
				}
				d, err = a.rmsRowsEps(&s, out, w, np*nH, hd, eps)
			default:
				return fmt.Errorf("op %d (%s): norm form %d not implemented on WebGPU", i, op.Kind, op.Form)
			}
			if err != nil {
				return err
			}
			regs[op.Out] = d
		case vision.TowerLinearQKV:
			if op.W != vision.TowerWQKV {
				return fmt.Errorf("op %d: fused qkv of weight %d", i, op.W)
			}
			for _, t := range []struct {
				r vision.TowerReg
				p *gvProj
			}{{vision.TowerQ, &b.q}, {vision.TowerK, &b.k}, {vision.TowerV, &b.v}} {
				d, err := a.linear(&s, in, t.p, np)
				if err != nil {
					return err
				}
				regs[t.r] = d
			}
		case vision.TowerLinear, vision.TowerLinearAdd:
			p, ok := b.towerLinear(op.W)
			if !ok {
				return fmt.Errorf("op %d: linear of weight %d", i, op.W)
			}
			d, err := a.linear(&s, in, p, np)
			if err != nil {
				return err
			}
			if op.Kind == vision.TowerLinear {
				regs[op.Out] = d
				continue
			}
			if err := c.addRowsDevice(out.buf, d.buf, np*p.out, p.out, 1); err != nil { // out += d
				return err
			}
		case vision.TowerRoPE:
			nr := np * nH * hd / 2
			if err := c.kernelN(a.rope, []*wgpu.Buffer{out.buf, cs.buf, sn.buf}, []uint32{uint32(nr), uint32(nH), uint32(hd), 0}, maxWG((nr+63)/64), 1); err != nil {
				return err
			}
		case vision.TowerAttention:
			var qkv [3]*DeviceBuffer
			for j, r := range []vision.TowerReg{vision.TowerQ, vision.TowerK, vision.TowerV} {
				if qkv[j], err = reg(r); err != nil {
					return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
				}
			}
			scale := float32(op.Scale)
			if a.planted.noScale {
				scale = 1
			}
			ctx, err := s.add(c.newDevBuf(np * H))
			if err != nil {
				return err
			}
			if seg != nil {
				if err := a.windowAttention(&s, qkv[0], qkv[1], qkv[2], ctx, seg, np, maxSegment(segs), scale); err != nil {
					return err
				}
				a.winBlocks++
			} else {
				for si := 1; si < len(segs); si++ {
					if err := a.attention(qkv[0], qkv[1], qkv[2], ctx, segs[si-1], segs[si]-segs[si-1], H, nH, hd, scale); err != nil {
						return err
					}
				}
			}
			regs[op.Out] = ctx
		case vision.TowerGELUTanh:
			n := np * width(op.Out)
			if err := c.inplaceKernel(c.geluPipeline, c.geluLayout, out.buf, []uint32{uint32(n), 0, 0, 0}, maxWG((n+63)/64)); err != nil {
				return err
			}
		case vision.TowerSiLUMul:
			n := np * width(op.Out)
			if err := c.kernelN(a.siluMul, []*wgpu.Buffer{out.buf, in.buf}, []uint32{uint32(n), 0, 0, 0}, maxWG((n+63)/64), 1); err != nil {
				return err
			}
		default:
			return fmt.Errorf("op %d: kind %s not implemented on WebGPU", i, op.Kind)
		}
	}
	c.device.Poll(true, nil)
	return nil
}

func (b *gvBlock) towerNorm(w vision.TowerWeight) (weight, bias *DeviceBuffer, ok bool) {
	switch w {
	case vision.TowerNorm1:
		return b.norm1w, b.norm1b, true
	case vision.TowerNorm2:
		return b.norm2w, b.norm2b, true
	case vision.TowerQNorm:
		return b.qNorm, nil, true
	case vision.TowerKNorm:
		return b.kNorm, nil, true
	}
	return nil, nil, false
}

func (b *gvBlock) towerLinear(w vision.TowerWeight) (*gvProj, bool) {
	switch w {
	case vision.TowerWO:
		return &b.proj, true
	case vision.TowerWFC1:
		return &b.fc1, true
	case vision.TowerWFC2:
		return &b.fc2, true
	case vision.TowerWGate:
		return &b.gate, true
	case vision.TowerWUp:
		return &b.up, true
	case vision.TowerWDown:
		return &b.down, true
	}
	return nil, false
}
