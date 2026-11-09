//go:build gpu && goinfer_testhooks

package kernels

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestMoEExpertW4A8_parity gates the indexed stacked-int4 expert GEMV: for several
// router slots it runs the GPU W4A8 kernel against refMatmulW4A8 (the CPU twin that
// dequantizes the exact f16 group scales the GPU reads). It exercises the stacking
// index (e = idx[slot], row = e*N+n), the nibble unpack, and the group scales.
func TestMoEExpertW4A8_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const nE, N, K = 6, 48, 64 // K is a multiple of the 32-wide W4A8 group
	const group = W4a8GroupSize
	ng := K / group
	rng := rand.New(rand.NewSource(7))

	nib := make([][]uint8, nE)
	sc := make([][]float32, nE)
	for e := range nE {
		ne := make([]uint8, N*K)
		for i := range ne {
			ne[i] = uint8(rng.Intn(16)) // nibble 0..15 (value−8)
		}
		se := make([]float32, N*ng)
		for i := range se {
			se[i] = float32(rng.Float64()*0.05 + 0.005)
		}
		nib[e], sc[e] = ne, se
	}
	stack, err := ctx.UploadStackedExpertsInt4(nib, sc, nE, N, K)
	if err != nil {
		t.Fatalf("UploadStackedExpertsInt4: %v", err)
	}
	defer stack.Release()
	if !stack.w4 {
		t.Fatal("stack.w4 not set")
	}

	act := make([]int8, K)
	for i := range act {
		act[i] = int8(rng.Intn(255) - 127)
	}
	aScale := float32(0.02)

	idx := []int{5, 0, 3} // chosen experts per slot
	for slot := range idx {
		got, err := ctx.IndexedGEMVForTestInt4(stack, act, aScale, idx, slot)
		if err != nil {
			t.Fatalf("IndexedGEMVForTestInt4 slot %d: %v", slot, err)
		}
		e := idx[slot]
		want := refMatmulW4A8(act, aScale, nib[e], sc[e], N, K, group)
		var dot, na, nb, maxAbs float64
		for n := range N {
			dot += float64(got[n]) * float64(want[n])
			na += float64(got[n]) * float64(got[n])
			nb += float64(want[n]) * float64(want[n])
			if d := math.Abs(float64(got[n] - want[n])); d > maxAbs {
				maxAbs = d
			}
		}
		cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
		t.Logf("slot %d → expert %d: cosine %.6f maxAbs %.3e", slot, e, cos, maxAbs)
		if cos < 0.99999 {
			t.Errorf("slot %d expert %d: cosine %.6f < 0.99999", slot, e, cos)
		}
	}
}

// TestUploadStackedExpertsInt4Packed_matchesUnpacked holds the packed upload to the unpacked one: the same experts, given to
// UploadStackedExpertsInt4 as one nibble per element and to UploadStackedExpertsInt4Packed as the decoder's int4 bytes (two nibbles a
// byte, element k's in the low nibble when k is even: decoder.Int4F32's layout, what the resident's caller passes), must give
// bit-identical expert GEMV outputs for every expert and two activations; the GEMV reads every weight word and group scale of an
// expert, and the experts' buffers are storage-only, so it is the comparison the device allows. A planted copy of the bytes with one
// byte's nibbles swapped must change its expert's output, so the comparison can go red.
func TestUploadStackedExpertsInt4Packed_matchesUnpacked(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const nE, N, K = 5, 40, 96 // K a multiple of the 32-wide group, as the packed path requires
	ng := K / W4a8GroupSize
	rng := rand.New(rand.NewSource(11))
	nib, q4, sc := make([][]uint8, nE), make([][]byte, nE), make([][]float32, nE)
	for e := range nE {
		nib[e], q4[e], sc[e] = make([]uint8, N*K), make([]byte, N*K/2), make([]float32, N*ng)
		for i := range nib[e] {
			v := uint8(rng.Intn(16))
			nib[e][i] = v
			if i&1 == 0 {
				q4[e][i>>1] |= v
			} else {
				q4[e][i>>1] |= v << 4
			}
		}
		for i := range sc[e] {
			sc[e][i] = float32(rng.Float64()*0.05 + 0.005)
		}
	}
	upload := func(packedBytes [][]byte) *ResidentStackedW8A8 {
		t.Helper()
		st, err := ctx.UploadStackedExpertsInt4Packed(packedBytes, sc, nE, N, K)
		if err != nil {
			t.Fatalf("UploadStackedExpertsInt4Packed: %v", err)
		}
		return st // the caller releases it before ctx.Close (a t.Cleanup would run after it)
	}
	unpacked, err := ctx.UploadStackedExpertsInt4(nib, sc, nE, N, K)
	if err != nil {
		t.Fatalf("UploadStackedExpertsInt4: %v", err)
	}
	defer unpacked.Release()
	packed := upload(q4)
	defer packed.Release()
	if !packed.w4 || packed.kp != unpacked.kp || packed.rows != unpacked.rows || packed.cols != unpacked.cols {
		t.Fatalf("packed geometry w4=%v kp=%d rows=%d cols=%d, unpacked kp=%d rows=%d cols=%d", packed.w4, packed.kp, packed.rows, packed.cols,
			unpacked.kp, unpacked.rows, unpacked.cols)
	}
	acts := make([][]int8, 2)
	for a := range acts {
		acts[a] = make([]int8, K)
		for i := range acts[a] {
			acts[a][i] = int8(rng.Intn(255) - 127)
		}
	}
	idx := make([]int, nE) // every expert, one router slot each
	for e := range idx {
		idx[e] = nE - 1 - e
	}
	// differing returns the first (activation, slot, row) where st's expert output differs from the unpacked upload's, or ok.
	differing := func(st *ResidentStackedW8A8) (string, bool) {
		for a, act := range acts {
			for slot := range idx {
				x, err := ctx.IndexedGEMVForTestInt4(st, act, 0.02, idx, slot)
				if err != nil {
					t.Fatal(err)
				}
				y, err := ctx.IndexedGEMVForTestInt4(unpacked, act, 0.02, idx, slot)
				if err != nil {
					t.Fatal(err)
				}
				if len(x) != N || len(y) != N {
					t.Fatalf("GEMV returned %d and %d rows, want %d", len(x), len(y), N)
				}
				for n := range x {
					if math.Float32bits(x[n]) != math.Float32bits(y[n]) {
						return fmt.Sprintf("activation %d, expert %d, row %d: %v against the unpacked upload's %v", a, idx[slot], n, x[n], y[n]), false
					}
				}
			}
		}
		return "", true
	}
	if where, ok := differing(packed); !ok {
		t.Fatalf("the packed upload differs from the unpacked one at %s", where)
	}
	// The plant: one byte's two nibbles swapped (or, if they are equal, one changed).
	planted := make([][]byte, nE)
	for e := range nE {
		planted[e] = append([]byte(nil), q4[e]...)
	}
	if b := planted[2][7]; b>>4 == b&0xF {
		planted[2][7] ^= 0x01
	} else {
		planted[2][7] = b<<4 | b>>4
	}
	bad := upload(planted)
	defer bad.Release()
	if _, ok := differing(bad); ok {
		t.Fatal("plant: a swapped nibble pair left every output equal; the comparison cannot see a layout error")
	}
	t.Log("plant: the swapped nibble pair is caught (red, as it must be)")
}
