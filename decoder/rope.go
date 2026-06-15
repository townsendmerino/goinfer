package decoder

import (
	"fmt"
	"math"
)

// applyRoPE rotates a single position's projected heads in place. vec is
// [heads, headDim] flattened row-major (a query or key vector for ONE token at
// absolute position pos). invFreq is the precomputed inverse-frequency table
// (arch.ropeInvFreq, with base, scaling, and rotary dim already resolved), so
// the rotated span is rotaryDim = 2*len(invFreq).
//
// NeoX / HF convention (rotate_half, non-interleaved), matching encoder/rope.go
// and HF's apply_rotary_pos_emb:
//
//	x1 = x[:half]; x2 = x[half:rot]
//	out[:half]    = x1*cos - x2*sin
//	out[half:rot] = x2*cos + x1*sin
//
// where θ_d = pos * invFreq[d] for d ∈ [0, rotaryDim/2). When rotaryDim <
// headDim (Phi's partial_rotary_factor), the trailing headDim-rotaryDim dims
// pass through unrotated.
//
// scale is YaRN's attention_factor (mscale), folded into cos/sin so the rotated
// q/k are scaled by it — matching HF, which multiplies cos/sin by
// attention_scaling. Pass 1.0 for non-YaRN families (no scaling).
func applyRoPE(vec []float32, heads, headDim, pos int, invFreq []float64, scale float64) {
	half := len(invFreq) // == rotaryDim/2
	posF := float64(pos)
	for d := range half {
		theta := posF * invFreq[d]
		c := math.Cos(theta) * scale
		s := math.Sin(theta) * scale
		for h := range heads {
			off := h * headDim
			x1 := float64(vec[off+d])
			x2 := float64(vec[off+half+d])
			vec[off+d] = float32(x1*c - x2*s)
			vec[off+half+d] = float32(x2*c + x1*s)
		}
	}
}

// ropeAt rotates the token at absolute sequence position seqPos, choosing m-RoPE
// (Qwen2.5-VL image prefill — section non-nil and mropePos carries this position)
// or plain scalar RoPE otherwise. The single seam both the batched (forwardN) and
// sequential (causalAttention) RoPE sites call, so adding m-RoPE didn't fork them.
func ropeAt(vec []float32, heads, headDim, seqPos int, invFreq []float64, scale float64, section []int, mropePos [][3]int) {
	if section != nil && seqPos < len(mropePos) {
		applyMRoPE(vec, heads, headDim, mropePos[seqPos], section, invFreq, scale)
		return
	}
	applyRoPE(vec, heads, headDim, seqPos, invFreq, scale)
}

// applyMRoPE is Qwen2.5-VL's multimodal RoPE: same rotate_half rotation as
// applyRoPE, but the len(invFreq) frequencies are partitioned into section[]
// chunks assigned to the (temporal, height, width) position components, so each
// frequency d rotates by pos[comp(d)]·invFreq[d]. section sums to len(invFreq).
// For a TEXT token the three positions are equal, so this reduces EXACTLY to
// applyRoPE(pos[0]) — the basis for keeping the text path bit-identical. (P5)
//
// Matches HF apply_multimodal_rotary_pos_emb: the cos/sin head_dim-wide tables are
// split over mrope_section*2 = [t,h,w,t,h,w]; frequency d and its pair half+d fall
// in the same component, so one component index per d suffices.
func applyMRoPE(vec []float32, heads, headDim int, pos [3]int, section []int, invFreq []float64, scale float64) {
	half := len(invFreq) // == rotaryDim/2; section sums to this
	for d := range half {
		theta := float64(pos[mropeComponent(d, section)]) * invFreq[d]
		c := math.Cos(theta) * scale
		s := math.Sin(theta) * scale
		for h := range heads {
			off := h * headDim
			x1 := float64(vec[off+d])
			x2 := float64(vec[off+half+d])
			vec[off+d] = float32(x1*c - x2*s)
			vec[off+half+d] = float32(x2*c + x1*s)
		}
	}
}

// mropeComponent returns which position component (0=temporal, 1=height, 2=width)
// frequency index d belongs to, given the cumulative section boundaries.
func mropeComponent(d int, section []int) int {
	acc := 0
	for i, n := range section {
		acc += n
		if d < acc {
			return i
		}
	}
	return len(section) - 1 // d == sum(section) shouldn't occur; clamp to width
}

// mropePositions computes Qwen2.5-VL m-RoPE 3D positions [seq][3] = (temporal,
// height, width) for the token sequence — a port of HF Qwen2_5_VLModel.get_rope_index.
// Text tokens get scalar positions (all three equal); each run of imageToken (one
// image per grid in order, grid = [t,h,w] in patch units) gets the spatial_merge'd
// grid coordinates offset by the running scalar position, after which scalar
// position resumes from that block's max + 1. mismatched image-token / grid counts
// return an error (the caller passes grids that match the placeholder runs).
func mropePositions(ids []int, imageToken int, grids [][3]int, merge int) ([][3]int, error) {
	pos := make([][3]int, len(ids))
	st, img, i := 0, 0, 0
	for i < len(ids) {
		if ids[i] != imageToken {
			pos[i] = [3]int{st, st, st}
			st++
			i++
			continue
		}
		if img >= len(grids) {
			return nil, fmt.Errorf("decoder(mrope): image token at %d but only %d grid(s) given", i, len(grids))
		}
		t, hm, wm := grids[img][0], grids[img][1]/merge, grids[img][2]/merge
		img++
		n := t * hm * wm
		if i+n > len(ids) {
			return nil, fmt.Errorf("decoder(mrope): grid %v needs %d image tokens at %d, sequence too short", grids[img-1], n, i)
		}
		base := st
		for tt := range t {
			for hh := range hm {
				for ww := range wm {
					pos[i] = [3]int{base + tt, base + hh, base + ww}
					i++
				}
			}
		}
		st = base + max(t, max(hm, wm)) // resume scalar from the block's max position + 1
	}
	return pos, nil
}
