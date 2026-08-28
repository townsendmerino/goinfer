//go:build realckpt

// MEASUREMENT ONLY — prices the narrow DeltaNet state snapshot for MTP speculation.
// Builds nothing: no snapshot/restore is wired into any decode path, specRollbackSafe is
// untouched, and nothing here is called from non-test code.
//
// WHAT IS BEING PRICED, and it is not the thing docs/qwen3_5_moe.md deferred. That entry scoped
// "state checkpoints" for cross-call PREFIX REUSE — restore to an arbitrary earlier position,
// later, possibly across requests. Speculation needs something much weaker: snapshot immediately
// before a verify, restore on rejection, discard. One buffer, one round deep, lifetime of
// milliseconds. The two were bundled because they share a root cause (deltanet.go:150-153: the
// state is fixed-size and NOT position-truncatable), not because they are the same size of
// problem.
//
//	GOINFER_QWEN35_08B=~/models/qwen3.5-0.8b \
//	  go test -tags realckpt ./decoder/ -run TestDeltaNetSnapshotCost -v -timeout 30m
package decoder

import (
	"fmt"
	"os"
	"sort"
	"testing"
	"time"
)

// snapshotBuf is a preallocated destination for one round's snapshot. The narrow scheme reuses
// ONE buffer across rounds, so steady-state cost excludes allocation; the allocating variant is
// measured separately and labelled, rather than being folded in.
type snapshotBuf struct {
	layers []int         // linear-layer indices, in order
	s      [][]float32   // per entry: copy of deltaState.s
	conv   [][][]float32 // per entry: copy of deltaState.convWin
}

func newSnapshotBuf(c *KVCache) *snapshotBuf {
	b := &snapshotBuf{}
	for l, d := range c.delta {
		if d == nil {
			continue
		}
		b.layers = append(b.layers, l)
		b.s = append(b.s, make([]float32, len(d.s)))
		w := make([][]float32, len(d.convWin))
		for i := range d.convWin {
			w[i] = make([]float32, len(d.convWin[i]))
		}
		b.conv = append(b.conv, w)
	}
	return b
}

// take copies live state -> buffer. restore writes buffer -> live state.
func (b *snapshotBuf) take(c *KVCache) {
	for i, l := range b.layers {
		d := c.delta[l]
		copy(b.s[i], d.s)
		for j := range d.convWin {
			copy(b.conv[i][j], d.convWin[j])
		}
	}
}

func (b *snapshotBuf) restore(c *KVCache) {
	for i, l := range b.layers {
		d := c.delta[l]
		copy(d.s, b.s[i])
		for j := range d.convWin {
			copy(d.convWin[j], b.conv[i][j])
		}
	}
}

func (b *snapshotBuf) bytes() int {
	n := 0
	for i := range b.s {
		n += len(b.s[i]) * 4
		for _, w := range b.conv[i] {
			n += len(w) * 4
		}
	}
	return n
}

func stats(d []time.Duration) (lo, med, hi time.Duration) {
	s := append([]time.Duration(nil), d...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	return s[0], s[len(s)/2], s[len(s)-1]
}

func TestDeltaNetSnapshotCost(t *testing.T) {
	ckpt := os.Getenv("GOINFER_QWEN35_08B")
	if ckpt == "" {
		ckpt = os.ExpandEnv("$HOME/models/qwen3.5-0.8b")
	}
	if _, err := os.Stat(ckpt + "/config.json"); err != nil {
		t.Skipf("no checkpoint at %s (%v)", ckpt, err)
	}

	// The DENOMINATOR is regime-dependent and that is the point: a faster decode step makes the
	// same copy a larger fraction. f32 CPU is the SLOWEST denominator available here, so the
	// percentages it produces are a LOWER bound on the cost fraction, not a typical value.
	quant := os.Getenv("GOINFER_SNAPCOST_QUANT")
	m, err := Load(ckpt, Options{Quant: quant})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	a := m.w.arch
	if a.qwen35 == nil {
		t.Fatalf("%s is not a qwen3_5 hybrid", a.Name)
	}

	// Prime past K-1 tokens so convWin is at STEADY STATE. A snapshot taken in the first few
	// tokens of a sequence is smaller, and pricing that would understate the real cost.
	cache := m.NewCache(64)
	const prime = 16
	for i := 0; i < prime; i++ {
		if _, err := m.forward(1+i, cache); err != nil {
			t.Fatalf("prime forward %d: %v", i, err)
		}
	}

	// Cross-check the live cache against what the config predicts. The size table in the report
	// is computed from config; this is what stops it being a paper number.
	p := *a.qwen35
	wantS := p.NumValueHeads * p.KeyHeadDim * p.ValueHeadDim
	wantConvDim := 2*(p.KeyHeadDim*p.NumKeyHeads) + p.ValueHeadDim*p.NumValueHeads
	nLinear := 0
	for l, d := range cache.delta {
		if d == nil {
			if a.isLinearLayer(l) {
				t.Errorf("layer %d is linear but has no deltaState", l)
			}
			continue
		}
		nLinear++
		if len(d.s) != wantS {
			t.Errorf("layer %d: len(s)=%d want %d", l, len(d.s), wantS)
		}
		if len(d.convWin) != p.ConvKernel-1 {
			t.Errorf("layer %d: convWin has %d vectors, want K-1=%d", l, len(d.convWin), p.ConvKernel-1)
		}
		for j, w := range d.convWin {
			if len(w) != wantConvDim {
				t.Errorf("layer %d convWin[%d]: %d want %d", l, j, len(w), wantConvDim)
			}
		}
	}

	buf := newSnapshotBuf(cache)
	total := buf.bytes()
	qLabel := quant
	if qLabel == "" {
		qLabel = "f32"
	}
	t.Logf("REGIME: qwen3_5 0.8B, quant=%s, CPU backend, single sequence. NOT transferable to a 27B/35B/80B trunk.", qLabel)
	t.Logf("geometry: %d layers, %d linear / %d full · nv=%d hk=%d hv=%d nk=%d K=%d",
		a.NumLayers, nLinear, a.NumLayers-nLinear, p.NumValueHeads, p.KeyHeadDim, p.ValueHeadDim, p.NumKeyHeads, p.ConvKernel)
	t.Logf("snapshot: %d B total (%.2f MiB) over %d linear layers = %d B/layer",
		total, float64(total)/(1<<20), nLinear, total/nLinear)

	// INTERLEAVED, paired per round: the decode step and the snapshot+restore are measured in the
	// same loop on the same cache, so between-run machine drift moves both together instead of
	// landing on one of them. Pooling separately-run batches is what this repo's rule 7 forbids.
	const rounds = 30
	var dec, snap, rest []time.Duration
	for r := 0; r < rounds; r++ {
		t0 := time.Now()
		if _, err := m.forward(1+(r%64), cache); err != nil {
			t.Fatalf("decode forward: %v", err)
		}
		dec = append(dec, time.Since(t0))

		t1 := time.Now()
		buf.take(cache)
		snap = append(snap, time.Since(t1))

		t2 := time.Now()
		buf.restore(cache)
		rest = append(rest, time.Since(t2))
	}

	dLo, dMed, dHi := stats(dec)
	sLo, sMed, sHi := stats(snap)
	rLo, rMed, rHi := stats(rest)
	sr := sMed + rMed
	gbs := float64(total) * 2 / float64(sr.Nanoseconds()) // snapshot reads+writes `total`; same for restore
	t.Logf("IN SITU, %d paired rounds (min / median / max):", rounds)
	t.Logf("  decode step      %8v %8v %8v", dLo, dMed, dHi)
	t.Logf("  snapshot         %8v %8v %8v", sLo, sMed, sHi)
	t.Logf("  restore          %8v %8v %8v", rLo, rMed, rHi)
	t.Logf("  snapshot+restore %v (median sum) = %.1f GB/s effective", sr, gbs)
	t.Logf("RATIOS (median): %% of one decode step = %.3f%%", 100*float64(sr)/float64(dMed))
	for _, K := range []int{4, 7} {
		t.Logf("   %% of a K=%d round (round ~= %d decode steps) = %.3f%%", K, K, 100*float64(sr)/float64(int64(K)*int64(dMed)))
	}

	// Tight-loop control. If this disagrees with the in-situ figure above, the IN-SITU one is the
	// answer and the disagreement is the finding — G26 had two microbenchmarks give wrong answers,
	// one under-bounding by 7-10x and one inverting a sign.
	ctl := newSnapshotBuf(cache)
	var tight []time.Duration
	for r := 0; r < rounds; r++ {
		t0 := time.Now()
		ctl.take(cache)
		ctl.restore(cache)
		tight = append(tight, time.Since(t0))
	}
	tLo, tMed, tHi := stats(tight)
	t.Logf("TIGHT-LOOP control (no decode between): %v / %v / %v — %.1f GB/s",
		tLo, tMed, tHi, float64(total)*2/float64(tMed.Nanoseconds()))
	t.Logf("  in-situ/tight ratio = %.2fx", float64(sr)/float64(tMed))

	// Allocating variant: what the FIRST round costs if the buffer is not reused.
	var alloc []time.Duration
	for r := 0; r < 10; r++ {
		t0 := time.Now()
		b := newSnapshotBuf(cache)
		b.take(cache)
		alloc = append(alloc, time.Since(t0))
	}
	aLo, aMed, aHi := stats(alloc)
	t.Logf("ALLOCATING snapshot (buffer not reused): %v / %v / %v", aLo, aMed, aHi)

	fmt.Fprintf(os.Stderr, "SNAPCOST_ROWS quant=%s total_bytes=%d linear=%d dec_med_ns=%d snap_med_ns=%d rest_med_ns=%d tight_med_ns=%d\n",
		qLabel, total, nLinear, dMed.Nanoseconds(), sMed.Nanoseconds(), rMed.Nanoseconds(), tMed.Nanoseconds())
}
