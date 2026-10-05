//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"os"
	"slices"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMC3Step_deviceArgmaxMatchesHost is E-P08's step gate on the MC3 fixture: the batched step's argmax-only ids, and a
// greedy draw row's id (Temperature 0, so invT is +Inf: ForwardSample's argmax shortcut), taken on the GPU (mc3DeviceArgmaxOn), equal
// argmaxF32 of the same step's full logits rows, at B = 2, 3 and 4 over 6 steps; and the device's own id buffer holds them.
func TestMC3Step_deviceArgmaxMatchesHost(t *testing.T) {
	if !mc3DeviceArgmaxOn {
		t.Skip("mc3DeviceArgmaxOn is off")
	}
	depths := []int{5, 23, 40, 300}
	_, r := mc3Resident(t, len(depths), 1024)
	seed := uint32(424242)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	for m, D := range depths {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, m, ids)
	}
	greedy := &decoder.ResidentBatchDraw{Temperature: 0} // invT = +Inf: ForwardSample's greedy shortcut
	checked := 0
	for _, B := range []int{2, 3, 4} {
		for st := range 6 {
			seqs := make([]batchSeq, B)
			for m := range B {
				seqs[m] = batchSeq{slot: m, pos: depths[m] + st, emb: mc3Emb(r, rnd())}
			}
			full, _, err := r.forwardMultiInto(seqs, false) // the same positions re-run below write the same K/V
			if err != nil {
				t.Fatal(err)
			}
			_, got, err := r.forwardMultiInto(seqs, true)
			if err != nil {
				t.Fatal(err)
			}
			draw := append([]batchSeq(nil), seqs...)
			draw[B-1].draw = greedy
			_, gotDraw, err := r.forwardMultiInto(draw, false)
			if err != nil {
				t.Fatal(err)
			}
			for m := range B {
				want := argmaxF32(full[m])
				if got[m] != want || int(r.batch.amaxTok.U32s()[m]) != want && m < B-1 {
					t.Fatalf("B=%d step %d row %d: device argmax %d (buffer %d), host %d", B, st, m, got[m], r.batch.amaxTok.U32s()[m], want)
				}
				checked++
			}
			if want := argmaxF32(full[B-1]); gotDraw[B-1] != want {
				t.Fatalf("B=%d step %d: greedy-draw row id %d, host argmax %d", B, st, gotDraw[B-1], want)
			}
		}
	}
	t.Logf("%d rows: the step's device argmax equals argmaxF32 of its full logits rows; greedy-draw rows too", checked)
}

// TestEP08_stepWallAB is E-P08's whole-step read: an argmax-only batched step (the prompt step's and verify's form), or
// with GOINFER_EP08_ROUTE=serve serve's greedy rows (Greedy draws against full-logits rows argmaxed on the host, the
// decoder's path before E-P08), with the argmax on the GPU and on the host, wall time per step (the host scan runs after the command buffer, so GPU time cannot
// see it), at B = 2, 4 and 8, depth 128, arms alternated rep by rep in one process, ids compared every step. Exploratory.
//
//	GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestEP08_stepWallAB$' -v ./metal/
func TestEP08_stepWallAB(t *testing.T) {
	const maxB, D, reps, tokens = 8, 128, 7, 12
	_, r := mc3RealResident(t, maxB, D+64)
	prev := mc3DeviceArgmaxOn
	defer func() { mc3DeviceArgmaxOn = prev }()
	seed := uint32(13579)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	for sl := range maxB {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, sl, ids)
	}
	med := func(v []float64) float64 {
		s := append([]float64(nil), v...)
		sort.Float64s(s)
		return s[len(s)/2]
	}
	serveRoute := os.Getenv("GOINFER_EP08_ROUTE") == "serve"
	for _, B := range []int{2, 4, maxB} {
		var ratios, onMs, offMs []float64
		for rep := range reps + 1 {
			arm := map[bool]float64{}
			var first [2][]int
			for k, on := range [][2]bool{{true, false}, {false, true}}[rep%2] {
				mc3DeviceArgmaxOn = on
				var ms []float64
				seedArm := seed
				for tok := range tokens {
					seqs := make([]batchSeq, B)
					for m := range B {
						seqs[m] = batchSeq{slot: m, pos: D, emb: mc3Emb(r, int(seedArm)%20000+tok+m)}
					}
					st := time.Now()
					var ids []int
					if serveRoute {
						// serve's route: with the switch on, Greedy draws (the decoder's, since E-P08); off, the
						// full-logits rows the decoder then argmaxed on the host, as it did before
						if on {
							for m := range seqs {
								seqs[m].draw = &decoder.ResidentBatchDraw{Greedy: true}
							}
							_, got, err := r.forwardMultiInto(seqs, false)
							if err != nil {
								t.Fatal(err)
							}
							ids = got
						} else {
							rows, _, err := r.forwardMultiInto(seqs, false)
							if err != nil {
								t.Fatal(err)
							}
							for _, row := range rows {
								ids = append(ids, argmaxF32(row))
							}
						}
					} else {
						var err error
						if _, ids, err = r.forwardMultiInto(seqs, true); err != nil {
							t.Fatal(err)
						}
					}
					ms = append(ms, float64(time.Since(st).Microseconds())/1e3)
					if tok == 0 {
						first[k] = append([]int(nil), ids...)
					}
				}
				arm[on] = med(ms)
			}
			if !slices.Equal(first[0], first[1]) {
				t.Fatalf("B=%d rep %d: ids differ between the arms: %v / %v", B, rep, first[0], first[1])
			}
			if rep > 0 {
				ratios = append(ratios, arm[false]/arm[true])
				onMs, offMs = append(onMs, arm[true]), append(offMs, arm[false])
			}
		}
		above := 0
		for _, q := range ratios {
			if q > 1 {
				above++
			}
		}
		fmt.Fprintf(os.Stderr, "[e-p08] RESULT (route %q) B=%d depth %d: host argmax %.3f ms, device argmax %.3f ms a step (wall); host / device median %.3f, %d of %d reps above 1 (per rep %v)\n",
			map[bool]string{false: "argmax-only", true: "serve"}[serveRoute], B, D, med(offMs), med(onMs), med(ratios), above, reps, auditFmt3(ratios))
	}
}
