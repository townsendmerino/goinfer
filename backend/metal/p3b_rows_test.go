//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
)

// p3bKVs are the cache precisions P3b-1 adds to the batched step.
var p3bKVs = []string{"i8", "f32"}

// withMC3KV loads the MC3 fixture at --kv kv for the rest of the test.
func withMC3KV(t *testing.T, kv string) {
	prev := mc3TestKV
	mc3TestKV = kv
	t.Cleanup(func() { mc3TestKV = prev })
}

// checkMC3KV fails unless r holds the cache kv asked for.
func checkMC3KV(t *testing.T, r *resident, kv string) {
	if r.kvI8 != (kv == "i8") || r.kvF32 != (kv == "f32") {
		t.Fatalf("--kv %s built kvI8 %v, kvF32 %v: this would check the wrong path", kv, r.kvI8, r.kvF32)
	}
}

// TestP3b_stepBitIdentical is P3B-G1 (docs/tasks/task-kv-precision-everywhere-2026-10.md, "P3b-1 gates"): at --kv i8 and f32, a
// batched step of four sequences at depths 5, 23, 40 and 300, each on its own slot, gives every row's logits bit-identical to
// production's single-token step on a twin slot; and past the attention kernel's 4096-key tile, the same at depths 4200 and 100.
func TestP3b_stepBitIdentical(t *testing.T) {
	for _, kv := range p3bKVs {
		t.Run(kv, func(t *testing.T) {
			withMC3KV(t, kv)
			mc3IdentityWith(t, []int{5, 23, 40, 300}, 1024, func(r *resident) { checkMC3KV(t, r, kv) })
		})
		t.Run(kv+"/deep", func(t *testing.T) {
			withMC3KV(t, kv)
			mc3IdentityWith(t, []int{4200, 100}, 4400, func(r *resident) { checkMC3KV(t, r, kv) })
		})
	}
}

// p3bSlotKV is slot s's K/V cache bytes for positions 0..n-1 on every layer, with an int8 cache's scales: what the cache holds,
// compared byte for byte.
func p3bSlotKV(t *testing.T, r *resident, s, n int) [][]byte {
	if err := r.useKVSlot(s); err != nil {
		t.Fatal(err)
	}
	var out [][]byte
	eb := r.kvElemBytes()
	for l := range r.layers {
		d := r.layers[l].geom.kvDim
		o := r.kvHostOff(l, 1) // bytes
		for _, b := range []Buffer{r.kc[l], r.vc[l]} {
			raw := b.Int8s()[o : o+n*d*eb]
			bs := make([]byte, len(raw))
			for i, x := range raw {
				bs[i] = byte(x)
			}
			out = append(out, bs)
		}
		if r.kvI8 {
			so, nKV := r.kvScaleOff(s, l), r.layers[l].geom.nKV
			for _, b := range []Buffer{r.ks[l], r.vs[l]} {
				f := b.Floats()[so : so+n*nKV]
				bs := make([]byte, 0, 4*len(f))
				for _, x := range f {
					u := math.Float32bits(x)
					bs = append(bs, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
				}
				out = append(out, bs)
			}
		}
	}
	return out
}

// TestP3b_promptInRowsBitIdentical is P3B-G1's prompt half: a fresh prompt fed to the batched step in 8-row pieces from position 0
// writes the same K/V bytes (and int8 scales) and gives the same logits, every row, as the sequential single-token loop.
func TestP3b_promptInRowsBitIdentical(t *testing.T) {
	for _, kv := range p3bKVs {
		t.Run(kv, func(t *testing.T) {
			withMC3KV(t, kv)
			_, r := mc3Resident(t, 2, 1024)
			checkMC3KV(t, r, kv)
			seed := uint32(10101)
			rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
			for _, K := range []int{8, 32, 64} {
				embs := make([][]float32, K)
				for i := range embs {
					embs[i] = mc3Emb(r, rnd())
				}
				if err := r.useKVSlot(0); err != nil {
					t.Fatal(err)
				}
				var ref [][]float32
				for i, e := range embs {
					ref = append(ref, append([]float32(nil), r.ForwardEmb(e, i)...))
				}
				refKV := p3bSlotKV(t, r, 0, K)
				var got [][]float32
				for c := 0; c < K; c += 8 {
					var seqs []batchSeq
					for i := c; i < min(c+8, K); i++ {
						seqs = append(seqs, batchSeq{slot: 1, pos: i, emb: embs[i]})
					}
					for _, row := range mc3Step(t, r, seqs) {
						got = append(got, append([]float32(nil), row...))
					}
				}
				gotKV := p3bSlotKV(t, r, 1, K)
				lg, kd := 0, 0
				for i := range got {
					for j := range got[i] {
						if math.Float32bits(got[i][j]) != math.Float32bits(ref[i][j]) {
							lg++
						}
					}
				}
				for i := range refKV {
					for j := range refKV[i] {
						if refKV[i][j] != gotKV[i][j] {
							kd++
						}
					}
				}
				fmt.Fprintf(os.Stderr, "[P3B-G1] --kv %s, K = %d in 8-row steps: %d of %d logits differ, %d cache bytes differ\n", kv, K, lg, K*r.V, kd)
				if len(got) != K || lg != 0 || kd != 0 {
					t.Errorf("K = %d: %d rows back, %d logits and %d cache bytes differ from the sequential loop", K, len(got), lg, kd)
				}
			}
		})
	}
}

// TestP3b_stepPrefillTakesInt8AndF32 is P3B-G2: at --kv i8 and f32 the batched step is built, PrefillPath names it, and PrefillLast
// of a 20-token prompt (below the step's ceiling, where an int8 cache takes the step too since P3b-2 gave it the MMA pass above
// it) returns logits bit-identical to the sequential loop's.
func TestP3b_stepPrefillTakesInt8AndF32(t *testing.T) {
	for _, kv := range p3bKVs {
		t.Run(kv, func(t *testing.T) {
			withMC3KV(t, kv)
			_, r := mc3Resident(t, 2, 1024)
			checkMC3KV(t, r, kv)
			if why := r.batchIneligible(); why != "" {
				t.Fatalf("batched step ineligible: %s", why)
			}
			a := &metalResident{r: r, hidden: r.H}
			if !a.stepPrefill() {
				t.Fatal("stepPrefill false")
			}
			if ok, why := a.PrefillPath(); !ok || !strings.Contains(why, "step") {
				t.Errorf("PrefillPath = %v, %q; want the step kernels", ok, why)
			}
			seed := uint32(4242)
			rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
			embs := make([][]float32, 20)
			for i := range embs {
				embs[i] = mc3Emb(r, rnd())
			}
			if err := r.useKVSlot(1); err != nil {
				t.Fatal(err)
			}
			var ref []float32
			for i, e := range embs {
				ref = append([]float32(nil), r.ForwardEmb(e, i)...)
			}
			if err := r.useKVSlot(0); err != nil {
				t.Fatal(err)
			}
			got, err := a.PrefillLast(context.Background(), embs, 0)
			if err != nil {
				t.Fatalf("PrefillLast: %v", err)
			}
			diff := 0
			for j := range got {
				if math.Float32bits(got[j]) != math.Float32bits(ref[j]) {
					diff++
				}
			}
			fmt.Fprintf(os.Stderr, "[P3B-G2] --kv %s: PrefillPath step; a 20-token prompt's last logits: %d of %d differ from sequential\n", kv, diff, len(ref))
			if diff != 0 {
				t.Errorf("%d of %d last-row logits differ from the sequential loop", diff, len(ref))
			}
		})
	}
}
