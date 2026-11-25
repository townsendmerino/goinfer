//go:build linux

package gpudev

import (
	"math/rand"
	"testing"
)

// TestCUDA_graphDependentChain stresses whether a captured MULTI-kernel segment preserves its
// intra-graph data dependencies. It captures N in-place increments of one buffer (vadd(x,one,x) — each
// launch reads AND writes x, so kernel k+1 has a strict read-after-write dependency on kernel k). If
// single-stream capture yields the linear chain it must, replay is serial and x == seed + N every
// time. If the instantiated graph drops those edges, the nodes can run concurrently and lose updates —
// invisible in isolation (the tiny kernels finish in issue order), but exposed by concurrent GPU load
// from another context. Run it with a background GPU churn to reproduce a graph-vs-serial divergence
// at the PRIMITIVE level (independent of any goinfer forward). N chosen large enough that a dropped
// edge reliably loses at least one update under contention.
func TestCUDA_graphDependentChain(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer d.ReleaseObjects()
	lib, err := d.CompileLibrary(smokePTX)
	if err != nil {
		t.Fatalf("CompileLibrary: %v", err)
	}
	p, err := d.NewComputePipeline(lib, "vadd")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	q := d.NewCommandQueue()
	const n = 5
	const N = 64 // dependent increments per replay
	x := NewBufferOf(d, make([]float32, n))
	one := NewBufferOf(d, []float32{1, 1, 1, 1, 1})
	nb := NewBufferOf(d, []uint32{n})

	chain := func() error {
		for range N {
			if e := q.launch(p, n, n, []Buffer{x, one, x, nb}); e != nil { // x = x + one, in place (RAW chain)
				return e
			}
		}
		return nil
	}
	g, err := q.Capture(chain)
	if err != nil {
		t.Fatalf("Capture: %v", err)
	}
	defer func() { _ = g.Close() }()

	const reps = 200
	bad := 0
	for r := range reps {
		if e := Upload(x, make([]float32, n)); e != nil { // reset x = 0
			t.Fatalf("Upload: %v", e)
		}
		if e := g.Replay(); e != nil {
			t.Fatalf("Replay: %v", e)
		}
		if e := q.Sync(); e != nil {
			t.Fatalf("Sync: %v", e)
		}
		got := make([]float32, n)
		if e := Download(x, got); e != nil {
			t.Fatalf("Download: %v", e)
		}
		for i := range got {
			if got[i] != float32(N) {
				bad++
				if bad <= 3 {
					t.Errorf("rep %d x[%d] = %v, want %v — a dropped intra-graph dependency lost %v updates",
						r, i, got[i], float32(N), float32(N)-got[i])
				}
				break
			}
		}
	}
	if bad == 0 {
		t.Logf("dependent chain (N=%d ×%d reps): every replay serial and exact — intra-graph edges preserved", N, reps)
	} else {
		t.Fatalf("%d/%d replays lost updates — captured graph does NOT serialize its dependent nodes", bad, reps)
	}
}

// TestCUDA_graphReplay verifies the Graph primitive: a captured segment replays bit-identically to
// direct launches, AND it reads the CURRENT buffer contents on each replay (the property the MoE
// expert cache relies on — the graph topology is fixed, only the DMA'd slot contents change). If
// replay froze the inputs at capture time, the second replay would return the first result.
func TestCUDA_graphReplay(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer d.ReleaseObjects()
	lib, err := d.CompileLibrary(smokePTX)
	if err != nil {
		t.Fatalf("CompileLibrary: %v", err)
	}
	p, err := d.NewComputePipeline(lib, "vadd")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	q := d.NewCommandQueue()
	a := NewBufferOf(d, []float32{1, 2, 3, 4, 5})
	b := NewBufferOf(d, []float32{10, 20, 30, 40, 50})
	out := d.NewBufferLen(5)
	nb := NewBufferOf(d, []uint32{5})

	// Capture vadd(a,b,out,n) — a pure async launch, no sync inside the closure.
	g, err := q.Capture(func() error { return q.launch(p, 5, 5, []Buffer{a, b, out, nb}) })
	if err != nil {
		t.Fatalf("Capture: %v", err)
	}
	defer func() { _ = g.Close() }()

	read := func() []float32 {
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		got := make([]float32, 5)
		if err := Download(out, got); err != nil {
			t.Fatalf("Download: %v", err)
		}
		return got
	}

	// Replay 1: out = a + b.
	if err := g.Replay(); err != nil {
		t.Fatalf("Replay: %v", err)
	}
	got := read()
	want := []float32{11, 22, 33, 44, 55}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("replay1 out[%d]=%v want %v", i, got[i], want[i])
		}
	}

	// Change a's CONTENTS (same buffer pointer) and replay: the graph must read the new bytes.
	if err := Upload(a, []float32{100, 200, 300, 400, 500}); err != nil {
		t.Fatalf("Upload: %v", err)
	}
	if err := g.Replay(); err != nil {
		t.Fatalf("Replay2: %v", err)
	}
	got = read()
	want = []float32{110, 220, 330, 440, 550}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("replay2 out[%d]=%v want %v — graph froze inputs at capture (contents-vary broken)", i, got[i], want[i])
		}
	}
	t.Logf("Graph: captured vadd replays bit-identically and reads current buffer contents across replays")
}

// TestCUDA_vitGraphReplayMatchesDirect captures a short chain of ViT kernels (LayerNorm, then the tanh GELU in place) and
// requires the replay to be bit-identical to dispatching the same launches directly, with fresh input bytes in the same
// buffer on the second replay: the property a whole-tower graph depends on, at kernel level and without a checkpoint.
func TestCUDA_vitGraphReplayMatchesDirect(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(9))
	const rows, dim = 16, 64
	n := rows * dim
	w, b := randF32(rng, dim, 1), randF32(rng, dim, 1)
	dx := NewBufferLenOf[float32](d, n)
	dw, db := NewBufferOf(d, w), NewBufferOf(d, b)
	dn := NewBufferLenOf[float32](d, n)
	chain := func() error {
		if err := q.Launch(v.LayerNorm, RowGrid(rows),
			Arg(dx), Arg(dw), Arg(db), Arg(dn),
			ArgValue(int32(rows)), ArgValue(int32(dim)), ArgValue(float32(1e-6))); err != nil {
			return err
		}
		return q.Launch(v.GELUTanh, Grid1D(n, 256), Arg(dn), ArgValue(int32(n)))
	}
	run := func(x []float32, replay func() error) []float32 {
		if err := Upload(dx, x); err != nil {
			t.Fatalf("Upload: %v", err)
		}
		if err := replay(); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		out := make([]float32, n)
		if err := Download(dn, out); err != nil {
			t.Fatalf("Download: %v", err)
		}
		return out
	}

	g, err := q.Capture(chain)
	if err != nil {
		t.Fatalf("Capture: %v", err)
	}
	defer func() { _ = g.Close() }()

	for i := range 2 {
		x := randF32(rng, n, 2)
		direct := run(x, chain)
		replayed := run(x, g.Replay)
		for j := range direct {
			if direct[j] != replayed[j] {
				t.Fatalf("input %d: element %d diverged: graph %v != direct %v", i, j, replayed[j], direct[j])
			}
		}
	}
}
