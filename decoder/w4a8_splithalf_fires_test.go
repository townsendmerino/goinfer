package decoder

import (
	"os"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestW4A8SplitHalfFires_onBenchModel reports how many of the bench model's
// int4 tensors the split-half repack actually accepted. It is the guard the
// end-to-end A/B reads BEFORE trusting a decode number: if this says 0
// repacked, the A/B's two arms are the same binary doing the same work and any
// "flat" it returns is measuring nothing.
//
// Not an assertion on a specific count — the count is a property of the model
// and the arch, not of the wiring — but it FAILS on zero-repacked, which is the
// only outcome that would silently invalidate the comparison.
func TestW4A8SplitHalfFires_onBenchModel(t *testing.T) {
	if os.Getenv("GOINFER_BENCH_QUANT") != "int4" {
		t.Skip("set GOINFER_BENCH_QUANT=int4 (and GOINFER_PREQUANT_GGUF) — this checks the int4 load path the A/B measures")
	}
	// The repack is default-OFF (see w4a8SplitHalfRepackEnabled), so enable it
	// here — otherwise this guard would report zero and "fail" on the shipped
	// configuration rather than on a broken one.
	defer func(prev bool) { w4a8SplitHalfRepackEnabled = prev }(w4a8SplitHalfRepackEnabled)
	w4a8SplitHalfRepackEnabled = true
	w4a8SplitHalfRepacked.Store(0)
	w4a8SplitHalfSkipped.Store(0)
	w4a8SplitHalfBytes.Store(0)

	if _, err := loadBenchModel(); err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	got, skipped := w4a8SplitHalfRepacked.Load(), w4a8SplitHalfSkipped.Load()
	t.Logf("split-half repack: %d tensors repacked, %d skipped (ineligible shape/arch/type), %.1f MiB of second-copy nibbles",
		got, skipped, float64(w4a8SplitHalfBytes.Load())/(1<<20))
	if got == 0 {
		// Zero is legitimate on a host aikit declines: non-amd64, no AVX2, or —
		// the non-obvious one — a host WITH AVX-512 VNNI, where canonical uses
		// the VNNI tier and split-half (AVX2-only) would be a downgrade, so the
		// repack refuses on purpose. Distinguish that from broken wiring by
		// asking aikit directly on a tensor of known-good shape: if it declines
		// that too, the host is ineligible and there is nothing to measure; if
		// it accepts, the wiring is what failed.
		probe := linalg.QuantizeInt4(make([]float32, 128*512), 128, 512, int4GroupSize)
		if !probe.RepackInt4SplitHalf() {
			t.Skipf("split-half repack declined by this host (%d tensors skipped) — amd64+AVX2 and NO AVX-512 VNNI required; there is no A/B to run here", skipped)
		}
		t.Fatalf("split-half repack accepted ZERO of this model's tensors (%d skipped) on a host where it ACCEPTS a known-good shape: the A/B arms would be identical and any decode result from them is void — check quant (int4) and that this load path is streamQuantized/quantizeWM rather than the deliberately-excluded .giw loader", skipped)
	}
}
