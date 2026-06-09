package decoder

import (
	"reflect"
	"testing"
)

// STEP 1 (de-risk the crux, zero download): prove the inverse V-head reorder is
// bit-clean. llama.cpp's converter tiles HF's grouped-by-K V heads when
// num_v_heads > num_k_heads; the GGUF loader must undo it exactly or it silently
// corrupts every linear-attn layer. We replicate the converter's FORWARD with an
// INDEPENDENT permutation-index reference (gather, not the production copy loop),
// then require untileVHeads to recover the original bit-for-bit across every
// tensor block geometry the loader will touch — with the real ratio num_k=16,
// num_v=32 (num_v_per_k=2), head_dim=128.

// fwdReorderRef is an independent reference for the converter's forward
// (HF grouped → GGUF tiled): it builds the permutation index the way llama.cpp's
// reorder_rows does (row_perm = _reorder_v_heads(arange) ; index_select) and
// gathers, rather than reusing the production reorderVHeads copy loop.
func fwdReorderRef(src []float32, lead, numK, numVPerK, hd, trail int) []float32 {
	mid := numK * numVPerK * hd
	// perm[destRow] = srcRow, matching grouped[k,v]→tiled[v,k].
	perm := make([]int, mid)
	for k := 0; k < numK; k++ {
		for v := 0; v < numVPerK; v++ {
			for c := 0; c < hd; c++ {
				srcRow := (k*numVPerK+v)*hd + c // HF grouped: [numK, numVPerK, hd]
				dstRow := (v*numK+k)*hd + c     // GGUF tiled: [numVPerK, numK, hd]
				perm[dstRow] = srcRow
			}
		}
	}
	dst := make([]float32, len(src))
	for l := 0; l < lead; l++ {
		for r := 0; r < mid; r++ {
			s := (l*mid + perm[r]) * trail
			d := (l*mid + r) * trail
			copy(dst[d:d+trail], src[s:s+trail])
		}
	}
	return dst
}

func iota32(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i + 1) // 1-based so 0 can't mask an indexing bug
	}
	return x
}

// TestVReorder_tinyHandComputed is a fully independent check: a 2×2 case whose
// tiled order is computed by hand. grouped (a,b): [0,1,2,3] with a=heads, b=v_per_k
// → tiled (b-major): [0,2,1,3].
func TestVReorder_tinyHandComputed(t *testing.T) {
	grouped := []float32{10, 11, 12, 13} // indices 0,1,2,3
	wantTiled := []float32{10, 12, 11, 13}
	gotTiled := fwdReorderRef(grouped, 1, 2, 2, 1, 1)
	if !reflect.DeepEqual(gotTiled, wantTiled) {
		t.Fatalf("forward ref tiled = %v, want %v", gotTiled, wantTiled)
	}
	// production helper must produce the same forward...
	if got := reorderVHeads(grouped, 1, 2, 2, 1, 1); !reflect.DeepEqual(got, wantTiled) {
		t.Fatalf("reorderVHeads forward = %v, want %v", got, wantTiled)
	}
	// ...and untile must invert it.
	if got := untileVHeads(wantTiled, 1, 2, 2, 1, 1); !reflect.DeepEqual(got, grouped) {
		t.Fatalf("untileVHeads = %v, want %v", got, grouped)
	}
}

func TestVReorder_roundTripAllBlocks(t *testing.T) {
	const (
		numK = 16
		numV = 32
		vpk  = numV / numK // 2
		hkd  = 128         // linear_key_head_dim
		hvd  = 128         // linear_value_head_dim
	)
	qDim := hkd * numK       // 2048
	kDim := hkd * numK       // 2048
	vDim := hvd * numV       // 4096
	qkChan := 2 * hkd * numK // 4096 (conv1d q+k channel block)

	// Each case: a buffer + the (lead, hd, trail) geometry of the V block to
	// reorder, plus the offset/size of that block inside the buffer (the q/k
	// rows of attn_qkv and the q/k channels of conv1d are NOT reordered).
	cases := []struct {
		name            string
		total           int // full tensor length
		off, blk        int // V-block offset and length within total
		lead, hd, trail int
	}{
		// attn_qkv [qDim+kDim+vDim, hidden]: only the V row-block, trail=hidden.
		{"attn_qkv(V-block)", (qDim + kDim + vDim) * 2, (qDim + kDim) * 2, vDim * 2, 1, hvd, 2},
		// attn_gate / in_proj_z [vDim, hidden]: full reorder, trail=hidden.
		{"attn_gate", vDim * 2, 0, vDim * 2, 1, hvd, 2},
		// ssm_alpha/beta / in_proj_a/b [numV, hidden]: rows hd=1, trail=hidden.
		{"ssm_alpha/beta", numV * 2, 0, numV * 2, 1, 1, 2},
		// ssm_a (A_log) / ssm_dt (dt_bias) [numV]: 1-D, hd=1, trail=1.
		{"ssm_a/dt(1-D)", numV, 0, numV, 1, 1, 1},
		// ssm_conv1d [convDim, K]: only the V channel-block, trail=K(=4).
		{"ssm_conv1d(V-block)", (qkChan + vDim) * 4, qkChan * 4, vDim * 4, 1, hvd, 4},
		// ssm_out / out_proj [hidden, vDim]: COLUMNS reordered → lead=hidden,
		// the whole row is the reordered axis, trail=1.
		{"ssm_out(cols)", 2 * vDim, 0, 2 * vDim, 2, hvd, 1},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			orig := iota32(c.total)
			// Simulate the converter forward on just the V block.
			tiled := append([]float32(nil), orig...)
			fwd := fwdReorderRef(orig[c.off:c.off+c.blk], c.lead, numK, vpk, c.hd, c.trail)
			copy(tiled[c.off:c.off+c.blk], fwd)

			// Non-vacuous: the forward must actually permute the V block.
			if reflect.DeepEqual(tiled, orig) {
				t.Fatalf("%s: forward did not permute (vacuous test)", c.name)
			}
			// q/k region (everything before off) must be untouched.
			if c.off > 0 && !reflect.DeepEqual(tiled[:c.off], orig[:c.off]) {
				t.Fatalf("%s: forward disturbed the non-V (q/k) region", c.name)
			}

			// Production inverse on the V block must recover the original exactly.
			got := append([]float32(nil), tiled...)
			inv := untileVHeads(got[c.off:c.off+c.blk], c.lead, numK, vpk, c.hd, c.trail)
			copy(got[c.off:c.off+c.blk], inv)

			if !reflect.DeepEqual(got, orig) {
				t.Fatalf("%s: round trip not bit-exact", c.name)
			}
		})
	}
}
