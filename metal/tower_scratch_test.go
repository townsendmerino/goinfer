//go:build darwin

package metal

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/audio"
	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/embeddinggemma2"
)

// The tower accelerators' scratch gates (MEM2, docs/tasks/task-memory-campaign-2026-10.md): a grow releases the
// capacity it replaces, Close returns the device to what it held before the accelerator was built, and a call after a
// regrow is bit-identical to the same call on a fresh accelerator.

// mem2Accel is one tower accelerator as the gates drive it.
type mem2Accel struct {
	ops   *eg2Ops
	grow  func(rows int)                    // the grow a call over rows makes
	run   func(step int) ([]float32, error) // a call at size step (1 < 2 < 3), its input fixed by step
	close func() error
}

type mem2Case struct {
	name string
	new  func(t *testing.T) mem2Accel
}

func mem2Cases() []mem2Case {
	return []mem2Case{
		{"gemma4-vision", func(t *testing.T) mem2Accel {
			enc := g4vTinyTower(t)
			a, err := newG4VAccel(enc)
			if err != nil {
				t.Fatal(err)
			}
			cells := [][2]int{{1, 1}, {1, 1}, {2, 1}, {2, 2}}
			return mem2Accel{ops: &a.eg2Ops, grow: a.grow, close: a.Close, run: func(step int) ([]float32, error) {
				p, pos := g4vCase(enc, cells[step][0], cells[step][1], false, uint64(step))
				return a.Hidden(p, pos)
			}}
		}},
		{"gemma4-audio", func(t *testing.T) mem2Accel {
			enc, err := audio.LoadGemma4AudioEncoder("../testdata/gemma4-audio-tiny")
			if err != nil {
				t.Fatal(err)
			}
			a, err := newG4AAccel(enc)
			if err != nil {
				t.Fatal(err)
			}
			return mem2Accel{ops: &a.eg2Ops, grow: a.grow, close: a.Close, run: func(step int) ([]float32, error) {
				n := 8 << step
				return a.Blocks(mem2Input(step, n*enc.Cfg.HiddenSize), n)
			}}
		}},
		{"embeddinggemma2", func(t *testing.T) mem2Accel {
			m, err := embeddinggemma2.Load(eg2TinyDir)
			if err != nil {
				t.Fatal(err)
			}
			a, err := newEG2Accel(m)
			if err != nil {
				t.Fatal(err)
			}
			return mem2Accel{ops: &a.eg2Ops, grow: a.grow, close: a.Close, run: func(step int) ([]float32, error) {
				T := 4 << step
				out, _, err := a.ForwardEmbeds(mem2Input(step, T*a.cfg.Hidden), T, false)
				return out, err
			}}
		}},
		{"grid (GLM-OCR)", func(t *testing.T) mem2Accel {
			enc, err := vision.LoadGlmOcrVisionEncoder("../testdata/glm-ocr-vision-tiny", false)
			if err != nil {
				t.Fatal(err)
			}
			a, err := newGlmOcrVAccel(enc)
			if err != nil {
				t.Fatal(err)
			}
			grids := [][][3]int{nil, {{1, 4, 4}}, {{1, 6, 8}}, {{1, 10, 6}}}
			return mem2Accel{ops: &a.eg2Ops, grow: func(rows int) { a.grow(rows, rows) }, close: a.Close,
				run: func(step int) ([]float32, error) {
					return a.Hidden(gvPixels(rand.New(rand.NewSource(int64(step))), grids[step], a.patchDim), grids[step])
				}}
		}},
	}
}

func mem2Input(seed, n int) []float32 {
	r := rand.New(rand.NewSource(int64(seed)))
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	return x
}

func mem2Probe(t *testing.T) *Device {
	t.Helper()
	probe, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	t.Cleanup(probe.ReleaseObjects)
	return probe
}

// TestTowerScratch_growKeepsOneCapacity: after grows to 64, 512 and 4096 rows, the device holds the buffers and the
// bytes of one grow to 4096 on a fresh accelerator, by the ledger and by MTLDevice's own allocation total.
func TestTowerScratch_growKeepsOneCapacity(t *testing.T) {
	probe := mem2Probe(t)
	for _, c := range mem2Cases() {
		t.Run(c.name, func(t *testing.T) {
			once := c.new(t)
			defer once.close()
			l0, _ := once.ops.d.LedgerLen()
			b0 := probe.CurrentAllocatedSize()
			once.grow(4096)
			l1, _ := once.ops.d.LedgerLen()
			wantBufs, wantBytes := l1-l0, int64(probe.CurrentAllocatedSize())-int64(b0)
			if wantBufs == 0 || wantBytes <= 0 {
				t.Fatalf("one grow allocated %d buffers, %d bytes: the gate cannot see a leak", wantBufs, wantBytes)
			}

			chain := c.new(t)
			defer chain.close()
			l0, _ = chain.ops.d.LedgerLen()
			b0 = probe.CurrentAllocatedSize()
			for _, rows := range []int{64, 512, 4096} {
				chain.grow(rows)
			}
			l1, _ = chain.ops.d.LedgerLen()
			gotBufs, gotBytes := l1-l0, int64(probe.CurrentAllocatedSize())-int64(b0)
			t.Logf("one grow to 4096 rows: %d buffers, %d bytes; three grows: %d buffers, %d bytes", wantBufs, wantBytes, gotBufs, gotBytes)
			if gotBufs != wantBufs || gotBytes != wantBytes {
				t.Errorf("after three grows the device holds %d buffers and %d bytes of scratch, want one capacity's %d and %d",
					gotBufs, gotBytes, wantBufs, wantBytes)
			}
		})
	}
}

// TestTowerScratch_closeReleasesEverything: an accelerator built, grown and run, then closed, leaves MTLDevice's
// allocation total where it was before the build and its ledger empty; a call after Close is refused, and a second
// Close is a no-op.
func TestTowerScratch_closeReleasesEverything(t *testing.T) {
	probe := mem2Probe(t)
	for _, c := range mem2Cases() {
		t.Run(c.name, func(t *testing.T) {
			before := probe.CurrentAllocatedSize()
			a := c.new(t)
			a.grow(4096)
			if _, err := a.run(1); err != nil {
				t.Fatal(err)
			}
			held := probe.CurrentAllocatedSize()
			d := a.ops.d
			if err := a.close(); err != nil {
				t.Fatal(err)
			}
			after := probe.CurrentAllocatedSize()
			t.Logf("device allocation: %d before the build, %d built and run, %d after Close", before, held, after)
			if held <= before {
				t.Fatalf("the accelerator allocated nothing the probe can see (%d → %d): the gate is blind", before, held)
			}
			if after != before {
				t.Errorf("after Close the device holds %+d bytes more than before the build (it held %+d)",
					int64(after)-int64(before), int64(held)-int64(before))
			}
			if bufs, objs := d.LedgerLen(); bufs != 0 || objs != 0 {
				t.Errorf("after Close the ledger holds %d buffers and %d objects", bufs, objs)
			}
			if _, err := a.run(1); !errors.Is(err, errAccelClosed) {
				t.Errorf("a call after Close returned %v, want errAccelClosed", err)
			}
			if err := a.close(); err != nil {
				t.Errorf("a second Close: %v", err)
			}
		})
	}
}

// TestTowerScratch_regrowIsBitIdentical: calls at three growing sizes, each regrowing the scratch, give the same bits
// at the largest as one call on a fresh accelerator, so a grow never releases a buffer a later call still reads.
func TestTowerScratch_regrowIsBitIdentical(t *testing.T) {
	for _, c := range mem2Cases() {
		t.Run(c.name, func(t *testing.T) {
			a, b := c.new(t), c.new(t)
			defer a.close()
			defer b.close()
			var got []float32
			for step := 1; step <= 3; step++ {
				var err error
				if got, err = a.run(step); err != nil {
					t.Fatal(err)
				}
			}
			want, err := b.run(3)
			if err != nil {
				t.Fatal(err)
			}
			mem2Identical(t, got, want)
		})
	}
}

func mem2Identical(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%d values, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("value %d is %v, want %v (bit for bit)", i, got[i], want[i])
		}
	}
}

// TestEG2RopeCache_releasesWhatItDrops: across forwards at 2*eg2RopeCap+1 lengths (longest first, so the scratch
// grows once), the ledger holds exactly the scratch and the two tables of every cached rope entry, the cache stays
// bounded, and the last forward is bit-identical to a fresh accelerator's.
func TestEG2RopeCache_releasesWhatItDrops(t *testing.T) {
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	a, err := newEG2Accel(m)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	keys := map[[2]float64]bool{} // the rope entries one forward makes: one per head dim and theta
	for _, ly := range a.layers {
		keys[[2]float64{float64(ly.w.HeadDim), ly.w.RopeTheta}] = true
	}
	n := 2*eg2RopeCap + 1
	a.grow(n)
	base, _ := a.d.LedgerLen()
	H := a.cfg.Hidden
	var last []float32
	for T := n; T >= 1; T-- {
		if last, _, err = a.ForwardEmbeds(mem2Input(T, T*H), T, false); err != nil {
			t.Fatal(err)
		}
		bufs, _ := a.d.LedgerLen()
		if bufs != base+2*len(a.ropes) || len(a.ropes) >= eg2RopeCap+len(keys) {
			t.Fatalf("after the forward at %d rows: %d buffers on the ledger for %d cached rope entries (want %d), cap %d",
				T, bufs, len(a.ropes), base+2*len(a.ropes), eg2RopeCap+len(keys))
		}
	}
	fresh, err := newEG2Accel(m)
	if err != nil {
		t.Fatal(err)
	}
	defer fresh.Close()
	want, _, err := fresh.ForwardEmbeds(mem2Input(1, H), 1, false)
	if err != nil {
		t.Fatal(err)
	}
	mem2Identical(t, last, want)
}
