//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestDFlashDispatchAmortization settles the ONE assumption gate 3's draft term still rests on.
//
// The 6.6 ms draft in docs/spec/08 is `5.33 × per-layer(M=16)`, where per-layer comes from
// dividing the 36-layer target's batched verify by 36. That silently assumes **a 5-layer model
// costs 5/36ths of a 36-layer one** — i.e. that per-layer cost is independent of how many layers
// are in the stack. It need not be: a 36-layer forward has 36 dispatches to hide launch latency
// behind, and five have far less. If per-layer cost RISES as the stack shortens, the drafter is
// more expensive than 6.6 ms and every projected speedup drops.
//
// This is not hypothetical in this repo. It is the mechanism that landed the CUDA-graphs
// projection (1.4–1.7×) at a measured 1.01× — CPU dispatch overlaps GPU compute differently at
// different scales — and the mechanism behind Lever 2's "the draft was the wall, not the verify".
//
// METHOD: run the resident forward with the layer loop truncated (`r.nLayers`), at both M=1
// (`launchToken`) and M=16 (`PrefillLast`, the regime the block draft actually runs in), and
// compare per-layer cost across stack depths. The outputs are numerically meaningless — a
// truncated stack is not a model — but the TIMING is exactly the quantity in question, and the
// weights for every layer are already resident so no reload is involved.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_CUDA_MODEL=$HOME/models/qwen3-4b \
//	  go test -tags 'cuda goinfer_testhooks' -run TestDFlashDispatchAmortization -v
func TestDFlashDispatchAmortization(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident did not engage")
	}
	_, _, _, _, _, _, vocab := mc.Dims()
	emb := mc.EmbedResidentForTest(1234 % (vocab - 1))

	const depth = 1024
	warm := make([][]float32, depth)
	for i := range warm {
		warm[i] = mc.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1))
	}
	if _, e := r.PrefillLast(warm, 0); e != nil {
		t.Fatalf("warm: %v", e)
	}
	full := r.nLayers
	defer func() { r.nLayers = full }() // restore, whatever happens

	block := make([][]float32, 16)
	for i := range block {
		block[i] = mc.EmbedResidentForTest((i*7919 + 13) % (vocab - 1))
	}

	best := func(f func() error) time.Duration {
		b := time.Hour
		for range 7 {
			t0 := time.Now()
			if err := f(); err != nil {
				t.Fatalf("run: %v", err)
			}
			if d := time.Since(t0); d < b {
				b = d
			}
		}
		return b
	}
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

	t.Logf("target has %d layers; probing truncated stacks at depth %d", full, depth)
	t.Logf("%7s | %10s %12s | %10s %12s", "layers", "M=1 ms", "per-layer", "M=16 ms", "per-layer")
	type row struct {
		n         int
		pl1, pl16 float64
	}
	var rows []row
	for _, n := range []int{5, 8, 12, 18, 36} {
		if n > full {
			continue
		}
		r.nLayers = n
		d1 := best(func() error {
			return r.do(func() error {
				if e := r.launchToken(emb, depth, false); e != nil { // no head: the drafter has none
					return e
				}
				return r.stream.Sync()
			})
		})
		d16 := best(func() error {
			_, e := r.PrefillLast(block, depth)
			return e
		})
		pl1, pl16 := ms(d1)/float64(n), ms(d16)/float64(n)
		rows = append(rows, row{n, pl1, pl16})
		t.Logf("%7d | %10.3f %12.4f | %10.3f %12.4f", n, ms(d1), pl1, ms(d16), pl16)
	}
	r.nLayers = full

	// The verdict: per-layer cost at 5 layers against per-layer cost at the full stack. The
	// 6.6 ms draft assumes this ratio is 1.0.
	var short, long row
	for _, x := range rows {
		if x.n == 5 {
			short = x
		}
		if x.n == full {
			long = x
		}
	}
	if short.n == 0 || long.n == 0 {
		t.Skip("need both a 5-layer and a full-stack point")
	}
	t.Logf("")
	t.Logf("per-layer INFLATION at 5 layers vs %d:  M=1 %.2fx   M=16 %.2fx",
		full, short.pl1/long.pl1, short.pl16/long.pl16)
	t.Logf("=> drafter (5 layers + fc) at M=16: %.2f ms  (the 6.6 ms projection assumes %.2f)",
		5.33*short.pl16, 5.33*long.pl16)
}

// TestDFlashCaptureSeamCost measures a composition cost the gate-3 arithmetic omits entirely.
//
// The projection composes draft + verify, where verify is the measured `W + C*k` curve. But that
// curve was measured with the HIDDEN-STATE SEAM OFF. In the real loop the drafter needs the
// target's residual at 5 tap layers for every token the target commits, and `capVec` implements
// that as a full `r.stream.Sync()` followed by a device->host `Download` — **per tap**. Five taps
// is five pipeline stalls per token, mid-forward.
//
// That is not a hypothetical cost: it is the difference between the verify the projection prices
// and the verify the loop would actually run. If it is large, every speedup figure is optimistic
// by that margin, and the fix (capture into a device buffer, download once, or overlap on a
// second stream) becomes a prerequisite rather than an optimization.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_CUDA_MODEL=$HOME/models/qwen3-4b \
//	  go test -tags 'cuda goinfer_testhooks' -run TestDFlashCaptureSeamCost -v
func TestDFlashCaptureSeamCost(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident did not engage")
	}
	_, _, _, _, _, _, vocab := mc.Dims()
	emb := mc.EmbedResidentForTest(1234 % (vocab - 1))
	const depth = 1024
	warm := make([][]float32, depth)
	for i := range warm {
		warm[i] = mc.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1))
	}
	if _, e := r.PrefillLast(warm, 0); e != nil {
		t.Fatalf("warm: %v", e)
	}
	defer func() { _ = r.SetHiddenCapture(nil) }()

	best := func() float64 {
		b := time.Hour
		for range 7 {
			t0 := time.Now()
			err := r.do(func() error {
				if e := r.launchToken(emb, depth, true); e != nil {
					return e
				}
				return r.stream.Sync()
			})
			if err != nil {
				t.Fatalf("launchToken: %v", err)
			}
			if d := time.Since(t0); d < b {
				b = d
			}
		}
		return float64(b.Microseconds()) / 1000
	}

	if e := r.SetHiddenCapture(nil); e != nil {
		t.Fatalf("capture off: %v", e)
	}
	off := best()
	// The 4B DFlash drafter's real taps.
	taps := []int{1, 9, 17, 25, 33}
	if e := r.SetHiddenCapture(taps); e != nil {
		t.Fatalf("capture on: %v", e)
	}
	on := best()
	_ = r.SetHiddenCapture(nil)

	t.Logf("M=1 decode @depth %d: capture OFF %.3f ms | ON (%d taps) %.3f ms | overhead %.3f ms (%.0f%%)",
		depth, off, len(taps), on, on-off, 100*(on-off)/off)
	t.Logf("per tap: %.3f ms — capVec does stream.Sync() + Download for EACH tap", (on-off)/float64(len(taps)))
	// Cost per verify round: the seam runs on every token the target commits (anchor + accepted).
	for _, acc := range []float64{3.97, 4.24} {
		t.Logf("  at %.2f accepted/round: +%.2f ms per round of seam overhead", acc, (on-off)*(acc+1))
	}
}

// TestDFlashRoundComposition measures the LOOP, not its parts.
//
// Every term in gate 3's projection is now measured — acceptance, the verify curve, decode, the
// draft (8.82 ms), the capture seam (0.465 ms/token). What is still arithmetic is the
// COMPOSITION: that a round costs draft + verify + seam and nothing else. Real loops have costs
// between their operations — host round-trips, stream syncs at the boundaries, the argmax and
// accept comparison, cache rollback — that a sum of independently-timed parts cannot show.
//
// The real drafter kernel does not exist yet, so this substitutes the TARGET's stack truncated to
// 5 layers as a timing stand-in. Its OUTPUT is meaningless — a truncated stack is not the
// drafter — but its COST is the right shape: 5 layers at M=16 over the same geometry, which is
// exactly what TestDFlashDispatchAmortization measured at 8.273 ms. What this adds is the
// sequencing: draft, then verify at M=k with capture live, then the host-side accept, per round.
//
// Reading: if measured/round ≈ predicted/round, the arithmetic composes and the projection's only
// remaining risk is the drafter kernel's own efficiency. If it exceeds the prediction, there is
// per-round overhead the projection never priced.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_CUDA_MODEL=$HOME/models/qwen3-4b \
//	  go test -tags 'cuda goinfer_testhooks' -run TestDFlashRoundComposition -v
func TestDFlashRoundComposition(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident did not engage")
	}
	_, _, _, _, _, _, vocab := mc.Dims()
	const depth = 1024
	warm := make([][]float32, depth)
	for i := range warm {
		warm[i] = mc.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1))
	}
	if _, e := r.PrefillLast(warm, 0); e != nil {
		t.Fatalf("warm: %v", e)
	}
	full := r.nLayers
	defer func() { r.nLayers = full; _ = r.SetHiddenCapture(nil) }()

	rows := func(n int) [][]float32 {
		out := make([][]float32, n)
		for i := range out {
			out[i] = mc.EmbedResidentForTest((i*7919 + 13) % (vocab - 1))
		}
		return out
	}
	const kVerify = 7 // code's optimum
	block16, verifyRows := rows(16), rows(kVerify)
	taps := []int{1, 9, 17, 25, 33}

	const rounds = 12
	best := time.Hour
	for range 5 {
		t0 := time.Now()
		for range rounds {
			// --- draft: 5 layers at M=16, capture OFF (the drafter has no seam) ---
			r.nLayers = 5
			_ = r.SetHiddenCapture(nil)
			if _, e := r.PrefillLast(block16, depth); e != nil {
				t.Fatalf("draft: %v", e)
			}
			// --- verify: full stack at M=k, capture LIVE (the drafter needs the taps) ---
			r.nLayers = full
			if e := r.SetHiddenCapture(taps); e != nil {
				t.Fatalf("capture on: %v", e)
			}
			outs, e := r.PrefillLastN(verifyRows, depth)
			if e != nil {
				t.Fatalf("verify: %v", e)
			}
			// --- host-side accept: argmax per row, the comparison the loop actually does ---
			for _, lg := range outs {
				bi, bv := 0, lg[0]
				for i, v := range lg {
					if v > bv {
						bi, bv = i, v
					}
				}
				_ = bi
			}
		}
		if d := time.Since(t0); d < best {
			best = d
		}
	}
	r.nLayers = full
	_ = r.SetHiddenCapture(nil)

	perRound := float64(best.Microseconds()) / 1000 / rounds
	// The arithmetic the projection uses, with every term as measured elsewhere.
	const draftMs, W, C, seamPerTok = 8.273, 8.77, 2.35, 0.465
	predicted := draftMs + (W + C*kVerify) + seamPerTok*kVerify
	t.Logf("k=%d, %d rounds x5, best-of:", kVerify, rounds)
	t.Logf("  predicted/round = draft %.2f + verify %.2f + seam %.2f = %.2f ms",
		draftMs, W+C*kVerify, seamPerTok*kVerify, predicted)
	t.Logf("  MEASURED/round  = %.2f ms   (%.2fx the prediction, %+.2f ms unaccounted)",
		perRound, perRound/predicted, perRound-predicted)
	t.Logf("  => composition overhead is %.0f%% of the round", 100*(perRound-predicted)/perRound)
}
