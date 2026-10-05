//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/prequant"
)

// TestG4LayerMajor_matchesSequential is 4b's gate (docs/tasks/task-m26-mac-2026-10.md): the layer-major paged prefill
// against the sequential loop on the tiny Gemma 4 MoE, transcoded to a metal-target .giw as M26's bundle is and paged
// at 2 and 3 slots (pread staging, as M26 runs). Prompts of 1, 5 and 8 tokens from position 0, and 5 tokens continuing
// a 3-token prefix the sequential loop wrote. Every K/V element and the last row's logits must be equal bit for bit,
// then 3 decode steps after it, and the layer-major arm must stage no more experts than the sequential one.
func TestG4LayerMajor_matchesSequential(t *testing.T) {
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no fixture (%s): %v", ckpt, err)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	giw := filepath.Join(t.TempDir(), "gemma4-moe-tiny.metal.giw")
	if err := prequant.Transcode(context.Background(), ckpt, giw, "int4", false, decoder.GIWTargetMetal); err != nil {
		t.Fatalf("transcode: %v", err)
	}
	toks := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 64, 9}
	for _, slots := range []int{2, 3} {
		t.Run("slots="+strconv.Itoa(slots), func(t *testing.T) {
			t.Setenv("GOINFER_METAL_MOE_SLOTS", strconv.Itoa(slots))
			m, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			mr, ok := m.ResidentForwardForTest().(*metalResident)
			if !ok || mr.r.g4moe == nil || !mr.r.g4moe.paged {
				t.Fatalf("expected a paged Gemma 4 MoE build (%s)", m.ResidentDecline())
			}
			r := mr.r
			stages := func() (n int) {
				for l := range r.layers {
					if gl := r.layers[l].g4moe; gl != nil && gl.pool != nil {
						n += gl.pool.stages + gl.pool.prefetched // every expert read, a prefetch included (g4PrefetchOn)
					}
				}
				return n
			}
			kv := func(n int) [][]uint16 {
				var out [][]uint16
				for l := range r.layers {
					d := r.layers[l].geom.kvDim
					o := r.kvHostOff(l, 2)
					out = append(out, append([]uint16(nil), r.kc[l].U16s()[o:o+n*d]...), append([]uint16(nil), r.vc[l].U16s()[o:o+n*d]...))
				}
				return out
			}
			emb := func(i int) []float32 { return m.EmbedResidentForTest(toks[i]) }
			for _, c := range []struct{ prefix, n int }{{0, 1}, {0, 5}, {0, 8}, {3, 5}} {
				total := c.prefix + c.n
				// run returns the last prompt row's logits, the K/V, and 3 decode steps' logits; layerMajor picks how the
				// prompt rows after the prefix run.
				run := func(layerMajor bool) (last []float32, kvs [][]uint16, dec [][]float32, staged int) {
					mr.Reset()
					for l := range r.layers { // every arm starts from an empty pool, so the stage counts compare
						if gl := r.layers[l].g4moe; gl != nil && gl.pool != nil {
							p := gl.pool
							for s := range p.slotExpert {
								p.slotExpert[s] = -1
							}
							p.where, p.lru = map[int]int{}, p.lru[:0]
						}
					}
					for i := range c.prefix {
						if err := mr.ForwardNoLogits(emb(i), i); err != nil {
							t.Fatal(err)
						}
					}
					s0 := stages()
					if layerMajor {
						embs := make([][]float32, c.n)
						for i := range embs {
							embs[i] = emb(c.prefix + i)
						}
						last = append([]float32(nil), r.prefillG4Paged(embs, c.prefix, true)...)
						if err := r.takeExecErr(); err != nil {
							t.Fatal(err)
						}
					} else {
						for i := c.prefix; i < total; i++ {
							if i < total-1 {
								if err := mr.ForwardNoLogits(emb(i), i); err != nil {
									t.Fatal(err)
								}
								continue
							}
							lg, err := mr.Forward(emb(i), i)
							if err != nil {
								t.Fatal(err)
							}
							last = append([]float32(nil), lg...)
						}
					}
					staged = stages() - s0
					kvs = kv(total)
					for s := range 3 {
						lg, err := mr.Forward(emb(total+s), total+s)
						if err != nil {
							t.Fatal(err)
						}
						dec = append(dec, append([]float32(nil), lg...))
					}
					return last, kvs, dec, staged
				}
				sLast, sKV, sDec, sStaged := run(false)
				lLast, lKV, lDec, lStaged := run(true)
				label := "prefix " + strconv.Itoa(c.prefix) + ", " + strconv.Itoa(c.n) + " rows"
				if len(lLast) != len(sLast) {
					t.Fatalf("%s: layer-major returned %d logits, sequential %d", label, len(lLast), len(sLast))
				}
				for j := range sLast {
					if math.Float32bits(sLast[j]) != math.Float32bits(lLast[j]) {
						t.Fatalf("%s: last-row logit %d: layer-major %v, sequential %v", label, j, lLast[j], sLast[j])
					}
				}
				diff := 0
				for i := range sKV {
					for j := range sKV[i] {
						if sKV[i][j] != lKV[i][j] {
							diff++
						}
					}
				}
				if diff != 0 {
					t.Fatalf("%s: %d K/V elements differ", label, diff)
				}
				for s := range sDec {
					for j := range sDec[s] {
						if math.Float32bits(sDec[s][j]) != math.Float32bits(lDec[s][j]) {
							t.Fatalf("%s: decode step %d logit %d differs", label, s, j)
						}
					}
				}
				if lStaged > sStaged {
					t.Fatalf("%s: layer-major staged %d experts, sequential %d", label, lStaged, sStaged)
				}
				t.Logf("%s: logits, every K/V element and 3 decode steps bit-identical; staged %d (sequential %d)", label, lStaged, sStaged)
			}
		})
	}
}

// TestG4LayerMajor_generateThroughDecoder tests 4b through its caller: decoder.Generate on the paged tiny Gemma 4 MoE,
// with g4LayerMajorOn and without, a 10-token prompt and 12 greedy tokens, then a second prompt extending the first
// (the reused prefix). The tokens must be equal, and the layer-major arm must have taken the route (PrefillLast is
// offered suffixes of 8 or more, residentPrefillSeed).
func TestG4LayerMajor_generateThroughDecoder(t *testing.T) {
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no fixture (%s): %v", ckpt, err)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	t.Setenv("GOINFER_METAL_MOE_SLOTS", "3")
	giw := filepath.Join(t.TempDir(), "gemma4-moe-tiny.metal.giw")
	if err := prequant.Transcode(context.Background(), ckpt, giw, "int4", false, decoder.GIWTargetMetal); err != nil {
		t.Fatalf("transcode: %v", err)
	}
	p1 := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 64}
	p2 := append(append([]int(nil), p1...), 9, 77, 31, 2, 150, 6, 44, 18, 120, 11)
	run := func(on bool) (toks [][]int, runs int) {
		prev := g4LayerMajorOn
		g4LayerMajorOn = on
		defer func() { g4LayerMajorOn = prev }()
		m, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		mr, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok || mr.r.g4moe == nil || !mr.r.g4moe.paged {
			t.Fatalf("expected a paged Gemma 4 MoE build (%s)", m.ResidentDecline())
		}
		for _, p := range [][]int{p1, p2} {
			ch, g := m.Generate(context.Background(), p, 12, decoder.SamplingParams{})
			var got []int
			for id := range ch {
				got = append(got, id)
			}
			if err := g.Err(); err != nil {
				t.Fatalf("generate (layer-major %v): %v", on, err)
			}
			toks = append(toks, got)
		}
		return toks, mr.r.g4LayerMajorRuns
	}
	off, offRuns := run(false)
	on, onRuns := run(true)
	if offRuns != 0 || onRuns == 0 {
		t.Fatalf("layer-major runs: %d with the switch off, %d with it on", offRuns, onRuns)
	}
	for i := range off {
		if !slices.Equal(off[i], on[i]) {
			t.Fatalf("prompt %d: layer-major %v, sequential %v", i, on[i], off[i])
		}
	}
	t.Logf("2 prompts (10 tokens, then 20 reusing them), 12 tokens each, equal with and without layer-major (%d layer-major prefills)", onRuns)
}
