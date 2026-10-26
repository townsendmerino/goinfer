//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/embeddinggemma2"
	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/internal/vision"
)

// mem2Inst is one built tower: its device (for the ledger), a forward over input i, and Close.
type mem2Inst struct {
	d     *Device
	fwd   func(i int) ([]float32, error)
	close func() error
}

// mem2Tower builds a fresh accelerator over one tiny fixture; inputs 0..n-1 grow in size.
type mem2Tower struct {
	name  string
	n     int
	build func(t *testing.T) mem2Inst
}

func mem2Towers(t *testing.T) []mem2Tower {
	rng := rand.New(rand.NewSource(21))
	var towers []mem2Tower

	g4v := g4vTinyTower(t)
	var g4vIn [][]float32
	var g4vPos [][][2]int
	for k := 1; k <= 3; k++ {
		p, pos := g4vCase(g4v, k, k, false, uint64(k))
		g4vIn, g4vPos = append(g4vIn, p), append(g4vPos, pos)
	}
	towers = append(towers, mem2Tower{"gemma4-vision", 3, func(t *testing.T) mem2Inst {
		a, err := newG4VAccel(g4v)
		if err != nil {
			t.Fatal(err)
		}
		return mem2Inst{a.d, func(i int) ([]float32, error) { return a.Hidden(g4vIn[i], g4vPos[i]) }, a.Close}
	}})

	g4a, err := audio.LoadGemma4AudioEncoder("../testdata/gemma4-audio-tiny")
	if err != nil {
		t.Fatal(err)
	}
	var g4aIn [][]float32
	for _, n := range []int{8, 20, 40} {
		h := make([]float32, n*g4a.Cfg.HiddenSize)
		for i := range h {
			h[i] = float32(rng.NormFloat64())
		}
		g4aIn = append(g4aIn, h)
	}
	towers = append(towers, mem2Tower{"gemma4-audio", 3, func(t *testing.T) mem2Inst {
		a, err := newG4AAccel(g4a)
		if err != nil {
			t.Fatal(err)
		}
		return mem2Inst{a.d, func(i int) ([]float32, error) {
			return a.Blocks(g4aIn[i], len(g4aIn[i])/g4a.Cfg.HiddenSize)
		}, a.Close}
	}})

	eg2, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	var eg2In [][]int
	for _, n := range []int{5, 40, 300} { // 300: two attention blocks
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*7919 + 13) % eg2.Config().VocabSize
		}
		eg2In = append(eg2In, ids)
	}
	towers = append(towers, mem2Tower{"embeddinggemma2", 3, func(t *testing.T) mem2Inst {
		a, err := newEG2Accel(eg2)
		if err != nil {
			t.Fatal(err)
		}
		return mem2Inst{a.d, func(i int) ([]float32, error) {
			last, _, err := a.Forward(eg2In[i], false)
			return last, err
		}, a.Close}
	}})

	grids := gvGrids[:3] // 16, 48 and 60 patches
	for _, gc := range []struct {
		name string
		mk   func(t *testing.T) gvCase
	}{{"qwen3.5", gvQwen3}, {"glm-ocr", gvGlm}} {
		pd := gc.mk(t).patchDim
		var px [][]float32
		for _, g := range grids {
			px = append(px, gvPixels(rng, g, pd))
		}
		towers = append(towers, mem2Tower{gc.name, 3, func(t *testing.T) mem2Inst {
			a := gc.mk(t).acc
			return mem2Inst{a.d, func(i int) ([]float32, error) { return a.Hidden(px[i], grids[i]) }, a.Close}
		}})
	}

	sig, err := vision.LoadEncoder("../testdata/siglip-tiny", false)
	if err != nil {
		t.Fatal(err)
	}
	towers = append(towers, mem2Tower{"siglip", 3, func(t *testing.T) mem2Inst {
		r, err := newSiglipVResident(sig)
		if err != nil {
			t.Fatal(err)
		}
		// SigLIP takes one patch count, so its three inputs are the same size: the check is that repeated forwards add
		// nothing.
		in := make([][]float32, 3)
		for i := range in {
			in[i] = make([]float32, r.np*r.a.patchDim)
			for j := range in[i] {
				in[i][j] = float32(math.Sin(float64(i*7 + j)))
			}
		}
		return mem2Inst{r.a.d, func(i int) ([]float32, error) { return r.ForwardPatches(in[i]) },
			func() error { r.Close(); return nil }}
	}})

	q25, err := vision.LoadQwenVisionEncoder("../testdata/qwen25vl-tiny", false)
	if err != nil {
		t.Fatal(err)
	}
	q25Grids := [][][3]int{{{1, 8, 8}}, {{1, 12, 16}}, {{1, 16, 12}, {1, 8, 8}}}
	c := q25.Cfg
	q25PD := c.InChans * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	var q25In [][]float32
	for _, g := range q25Grids {
		q25In = append(q25In, gvPixels(rng, g, q25PD))
	}
	towers = append(towers, mem2Tower{"qwen2.5-vl", 3, func(t *testing.T) mem2Inst {
		r, err := newQwen25VResident(q25)
		if err != nil {
			t.Fatal(err)
		}
		return mem2Inst{r.a.d, func(i int) ([]float32, error) { return r.ForwardViT(q25In[i], q25Grids[i]) },
			func() error { r.Close(); return nil }}
	}})
	return towers
}

// TestMEM2_towerScratchFreed is MEM2-G1 and MEM2-G2 (docs/tasks/task-memory-campaign-2026-10.md, MEM2 gates). G1: a
// tower that grows three times (A, inputs ascending) holds the same ledger buffers as one that grew once (B, the same
// inputs largest first), Metal's allocated size agrees within 256 KiB, and every output is byte-equal. G2: after Close the
// ledger is empty and Metal's allocated size is back within 64 KiB of its reading before the build; a second Close
// returns nil and a forward after Close is an error.
func TestMEM2_towerScratchFreed(t *testing.T) {
	probe, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer probe.ReleaseObjects()
	// slack bounds Metal's reading after Close; growSlack bounds it between the two orders, where Metal's suballocation
	// alone moves it by 128 KiB with the same buffers (the gate's amendment in the task doc).
	const slack, growSlack = 64 << 10, 256 << 10
	for _, tw := range mem2Towers(t) {
		t.Run(tw.name, func(t *testing.T) {
			// A tower's first build in the process keeps 256 KiB past its Close that later builds do not; build one first.
			warm := tw.build(t)
			if err := warm.close(); err != nil {
				t.Fatal(err)
			}
			type reading struct {
				allocs int
				bytes  int64
				outs   [][]float32
			}
			run := func(order []int) reading {
				base := int64(probe.CurrentAllocatedSize())
				in := tw.build(t)
				r := reading{outs: make([][]float32, tw.n)}
				for _, i := range order {
					out, err := in.fwd(i)
					if err != nil {
						t.Fatalf("forward %d: %v", i, err)
					}
					r.outs[i] = out
				}
				r.allocs, _ = in.d.LedgerLen()
				r.bytes = int64(probe.CurrentAllocatedSize()) - base
				if err := in.close(); err != nil {
					t.Fatalf("Close: %v", err)
				}
				allocs, objs := in.d.LedgerLen()
				after := int64(probe.CurrentAllocatedSize()) - base
				fmt.Fprintf(os.Stderr, "[MEM2] %s order %v: %d buffers, %+d bytes live; after Close ledger (%d, %d), %+d bytes\n",
					tw.name, order, r.allocs, r.bytes, allocs, objs, after)
				if allocs != 0 || objs != 0 {
					t.Errorf("G2: after Close the ledger holds %d buffers and %d objects", allocs, objs)
				}
				if after > slack {
					t.Errorf("G2: after Close Metal still holds %+d bytes over its reading before the build", after)
				}
				if err := in.close(); err != nil {
					t.Errorf("G2: a second Close: %v", err)
				}
				if _, err := in.fwd(0); err == nil {
					t.Error("G2: a forward after Close returned no error")
				}
				return r
			}
			asc := make([]int, tw.n)
			for i := range asc {
				asc[i] = i
			}
			desc := slices.Clone(asc)
			slices.Reverse(desc)
			a, b := run(asc), run(desc)
			if a.allocs != b.allocs {
				t.Errorf("G1: grown stepwise the ledger holds %d buffers, grown once %d", a.allocs, b.allocs)
			}
			if d := a.bytes - b.bytes; d > growSlack || d < -growSlack {
				t.Errorf("G1: grown stepwise Metal holds %d bytes, grown once %d (%+d)", a.bytes, b.bytes, d)
			}
			for i := range a.outs {
				if len(a.outs[i]) != len(b.outs[i]) {
					t.Fatalf("G1: input %d: %d outputs against %d", i, len(a.outs[i]), len(b.outs[i]))
				}
				for j := range a.outs[i] {
					if math.Float32bits(a.outs[i][j]) != math.Float32bits(b.outs[i][j]) {
						t.Errorf("G1: input %d differs at %d: %v against %v", i, j, a.outs[i][j], b.outs[i][j])
						break
					}
				}
			}
		})
	}
}
