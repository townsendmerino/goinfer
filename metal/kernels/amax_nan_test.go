//go:build darwin

package kernels

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/goinfer/internal/sampling"
)

// TestAmaxHead_nanRowsMatchTheHost is G-MA1 (docs/tasks/task-subsystems-2026-10.md §8, "The device argmax on Metal: NaN rows"):
// each fused head kernel (gemv_w8a8_amax, and gemv_w8f32_amax on f32 activations) then argmax_finish gives sampling.Argmax of the
// same logits on an all -inf row, a row whose NaN leads the tile holding the maximum, a random row and a random row with ties at
// the maximum; an all-NaN row gives an id at or past V, which the host maps to 0. The activations are small integers, so the host
// computes the logits exactly as the kernels do.
func TestAmaxHead_nanRowsMatchTheHost(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatal(err)
	}
	finish, err := d.NewComputePipeline(lib, "argmax_finish")
	if err != nil {
		t.Fatal(err)
	}
	q := d.NewCommandQueue()
	const V, K = 1024, 64 // 128 tiles of 8 rows
	nTiles := V / 8
	rng := rand.New(rand.NewSource(3))
	nan, inf := float32(math.NaN()), float32(math.Inf(1))

	type head struct {
		kernel string
		f32    bool // f32 activations (gemv_w8f32_amax); else int8 with one scale (gemv_w8a8_amax)
	}
	for _, h := range []head{{"gemv_w8a8_amax", false}, {"gemv_w8f32_amax", true}} {
		pipe, err := d.NewComputePipeline(lib, h.kernel)
		if err != nil {
			t.Fatal(err)
		}
		// run returns the device id for activations a (scaled by asc on the int8 head), weights bq and row scales bsc, and
		// the host's logits.
		run := func(a []float32, asc float32, bq []int8, bsc []float32) (uint32, []float32) {
			logits := make([]float32, V)
			for r := range V {
				acc := float32(0)
				for k := range K {
					acc += float32(bq[r*K+k]) * a[k]
				}
				if h.f32 {
					logits[r] = acc * bsc[r]
				} else {
					logits[r] = acc * asc * bsc[r]
				}
			}
			var act Buffer
			if h.f32 {
				act = NewBufferFloats(d, a)
			} else {
				aq := make([]int8, K)
				for k := range aq {
					aq[k] = int8(a[k])
				}
				act = NewBufferInt8(d, aq)
			}
			part, tok := d.NewBufferLen(nTiles*2), d.NewBufferLen(1)
			e := q.Begin()
			e.Dispatch(pipe, V*32, 256, act, NewBufferFloats(d, []float32{asc}), NewBufferInt8(d, bq), NewBufferFloats(d, bsc),
				part, NewBufferU32(d, uint32(K)))
			e.DispatchTG(finish, 256, 256, 0, part, tok, NewBufferU32(d, uint32(nTiles)))
			e.End()
			return tok.U32s()[0], logits
		}
		a := make([]float32, K)
		for k := range a {
			a[k] = float32(rng.Intn(15) - 7)
		}
		bq := make([]int8, V*K)
		for i := range bq {
			bq[i] = int8(rng.Intn(255) - 127)
		}
		scales := func() []float32 {
			s := make([]float32, V)
			for i := range s {
				s[i] = 0.01
			}
			return s
		}
		ones := make([]float32, K)
		for k := range ones {
			ones[k] = 1
		}
		negs := make([]int8, V*K)
		for i := range negs {
			negs[i] = -1
		}
		allInf := scales()
		for i := range allInf {
			allInf[i] = inf
		}
		// big makes row r's dot large and positive: its weights take the activations' signs.
		big := func(w []int8, r int) {
			for k := range K {
				w[r*K+k] = 0
				if a[k] > 0 {
					w[r*K+k] = 100
				} else if a[k] < 0 {
					w[r*K+k] = -100
				}
			}
		}
		tileBq := append([]int8(nil), bq...)
		big(tileBq, 43)
		tiles := scales()
		tiles[40] = nan // NaN leads tile 5 (rows 40-47), whose row 43 holds the maximum
		// nanTile is every row NaN but rows 41-47, +inf (with ones and negs, their logits are -inf).
		nanTile := func() []float32 {
			s := make([]float32, V)
			for i := range s {
				s[i] = nan
			}
			for i := 41; i <= 47; i++ {
				s[i] = inf
			}
			return s
		}
		tieBq := append([]int8(nil), bq...)
		for _, r := range []int{300, 301, 700} {
			big(tieBq, r) // the same weights and scale, so the three logits tie at the maximum
		}
		for _, c := range []struct {
			name string
			a    []float32
			asc  float32
			bq   []int8
			bsc  []float32
		}{
			{"all -inf", ones, 1, negs, allInf},
			{"NaN leads the max's tile", a, 1, tileBq, tiles},
			{"random", a, 1, bq, scales()},
			{"ties at the maximum", a, 1, tieBq, scales()},
		} {
			got, logits := run(c.a, c.asc, c.bq, c.bsc)
			if want := sampling.Argmax(logits); int(got) != want {
				t.Errorf("%s, %s: device %d, host %d", h.kernel, c.name, got, want)
			}
		}
		// Every logit NaN: a NaN in the activations (a NaN scale on the int8 head, a NaN element on the f32 one).
		nanA := append([]float32(nil), a...)
		ascNaN := float32(1)
		if h.f32 {
			nanA[0] = nan
		} else {
			ascNaN = nan
		}
		if got, _ := run(nanA, ascNaN, bq, scales()); int(got) < V {
			t.Errorf("%s, all NaN: device %d, want an id at or past V (%d), which the host maps to 0", h.kernel, got, V)
		}
		// A tile of NaN then -inf (rows 40-47: NaN at 40, -inf at 41-47) and every other row NaN. The old reduction seeded the tile
		// with row 40's NaN, so every partial was NaN and argmax_finish wrote 0xFFFFFFFF; the fixed one skips the NaN, and the
		// row's id is 41, the first -inf. Row 0 is NaN, the case left as it is (§8), so the id is checked against 41, not the host.
		if got, _ := run(ones, 1, negs, nanTile()); got != 41 {
			t.Errorf("%s, a -inf tile among NaN: device %d, want 41", h.kernel, got)
		}
	}

	// gemv_w4a8_sa_amax (not built in production) reduces its tiles the same way; the same row, through its int4 layout: every
	// weight -1 (nibble 7) and every activation 1, so each dot is negative, and the f16 group scales are NaN (0x7E00) except
	// rows 41-47's, which are +inf (0x7C00).
	sa, err := d.NewComputePipeline(lib, "gemv_w4a8_sa_amax")
	if err != nil {
		t.Fatal(err)
	}
	words := make([]uint32, V*(K/8))
	for i := range words {
		words[i] = 0x77777777
	}
	sct := make([]uint16, V*(K/32))
	for r := range V {
		for g := range K / 32 {
			sct[r*(K/32)+g] = 0x7E00
			if r >= 41 && r <= 47 {
				sct[r*(K/32)+g] = 0x7C00
			}
		}
	}
	aq := make([]int8, K)
	for k := range aq {
		aq[k] = 1
	}
	part, tok := d.NewBufferLen(nTiles*2), d.NewBufferLen(1)
	e := q.Begin()
	e.DispatchTG(sa, V*32, 256, K*2, NewBufferUint32s(d, words), NewBufferU16s(d, sct), NewBufferInt8(d, aq),
		NewBufferFloats(d, []float32{1}), part, NewBufferU32(d, uint32(K)))
	e.DispatchTG(finish, 256, 256, 0, part, tok, NewBufferU32(d, uint32(nTiles)))
	e.End()
	if got := tok.U32s()[0]; got != 41 {
		t.Errorf("gemv_w4a8_sa_amax, a -inf tile among NaN: device %d, want 41", got)
	}
}
