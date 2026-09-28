//go:build cuda

package cuda

import (
	"errors"
	"strings"
	"testing"
)

// The load-time prefill report (decoder.PrefillPathReporter).
//
// WHY THIS EXISTS. `--backend cuda --quant int8int8` on a dense model builds a full resident decode
// path — ResidentActive is true and decode runs at ~0.7× int4 — but the batched prefill GEMV is
// int4-only (gemv_w4a8_batched / _rn read group-scaled int4 words), so prefillCore declines and every
// prompt falls back to one forward per token. Measured on a 300-token prompt, real 0.5B, RTX 2070
// SUPER: 1.73 s vs 0.19 s (9×), 4.56 vs 0.22 CPU-seconds (20×), with no compute hotspot — the CPU is
// the executor spin-waiting through 300 sequential launches. The fallback is silent by design, so the
// only defence is reporting it at load, and the only way that report stays true is sharing the guard
// with prefillCore (prefillStaticDecline).
//
// These tests need NO DEVICE: the guard reads struct state only.

// declineFixture builds the minimal cudaResident the static guard inspects: n uniform dense layers,
// batched kernels loaded, all seven projections of the given kind.
func declineFixture(n int, kind string) *cudaResident {
	r := &cudaResident{prefillReady: true, nLayers: n}
	for range n {
		w := cudaWQ{kind: kind, N: 8, K: 32}
		r.layers = append(r.layers, cudaLayer{
			q: w, k: w, v: w, o: w, g: w, u: w, d: w,
			hd: 4, nKV: 2, qDim: 8, kvDim: 8, rhalf: 2,
		})
	}
	return r
}

// TestPrefillPath_int8Declines is the gate for the shipped defect: int8 weights must report the
// sequential path, name int4 as the requirement, and state the cost.
// TestPrefillPath_int8Batched: int8 bundles now get batched prefill (§C6 — the batched W8A8 GEMV is
// exact-int32, bit-identical to gemv_w8a8_fwd by construction). This was TestPrefillPath_int8Declines
// before int8 batched prefill landed; it now asserts the OPPOSITE, so the 9× TTFT trap is gone.
func TestPrefillPath_int8Batched(t *testing.T) {
	batched, why := declineFixture(4, "int8").PrefillPath()
	if !batched {
		t.Fatalf("int8 projections reported as NOT batched (%q) — int8 batched prefill regressed", why)
	}
	if !strings.Contains(why, "batched") {
		t.Errorf("reason %q does not say batched", why)
	}
	t.Logf("reason: %s", why)
}

// TestPrefillPath_nativeDeclinesNamed: a non-batchable projection kind (native/f32) still declines, and
// the reason names int4/int8 as the batchable kinds so an operator can act on it.
func TestPrefillPath_nativeDeclinesNamed(t *testing.T) {
	batched, why := declineFixture(4, "f32").PrefillPath()
	if batched {
		t.Fatalf("native (f32) projections reported as batched — %q", why)
	}
	for _, want := range []string{"sequential", "int4", "int8", "TTFT"} {
		if !strings.Contains(why, want) {
			t.Errorf("reason %q does not mention %q", why, want)
		}
	}
	t.Logf("reason: %s", why)
}

// TestPrefillPath_int4Batched: the shipped fast lane must not be reported as a decline
// (-require-backend would refuse to start an entirely healthy server).
func TestPrefillPath_int4Batched(t *testing.T) {
	batched, why := declineFixture(4, "int4").PrefillPath()
	if !batched {
		t.Fatalf("all-int4 dense model reported as declined: %s", why)
	}
}

// TestPrefillPath_matchesPrefillCore is the anti-drift gate, and the reason the guard was extracted
// rather than duplicated: whatever prefillCore refuses, the startup line must call sequential. A
// future guard added to one and not the other would make the report a lie — silently, which is
// exactly the failure mode this whole change exists to end.
func TestPrefillPath_matchesPrefillCore(t *testing.T) {
	cases := []struct {
		name string
		r    *cudaResident
	}{
		{"int4", declineFixture(2, "int4")},
		{"int8", declineFixture(2, "int8")},
		{"kernels-unavailable", func() *cudaResident { r := declineFixture(2, "int4"); r.prefillReady = false; return r }()},
		// moe / gemma4moe / non-uniform NO LONGER DECLINE — the MoE FFN runs per row off the batched
		// residual and geometry is bound per layer. They stay in this list because the property under
		// test is that the GUARD and the REPORT agree, whichever way they answer; a case that flipped
		// from declining to batching still exercises that. The names say what the model is, not what
		// the verdict is, so they do not go stale a second time.
		{"moe", func() *cudaResident { r := declineFixture(2, "int4"); r.moe = true; return r }()},
		{"gemma4moe", func() *cudaResident { r := declineFixture(2, "int4"); r.gemma4Moe = true; return r }()},
		{"non-uniform", func() *cudaResident { r := declineFixture(2, "int4"); r.layers[1].nKV = 1; return r }()},
		// k-eq-v still declines, but for a NARROWER reason than before: K=V itself is handled now, and
		// what is refused is a K=V layer with no v_norm unit weight to normalise with.
		{"k-eq-v-no-vnorm", func() *cudaResident { r := declineFixture(2, "int4"); r.layers[1].kEqV = true; return r }()},
		// A recurrent (Gated-DeltaNet) model whose layers ALSO carry valid int4 q/k/o. This is the
		// case the weight-kind check cannot catch: qwen3_5_moe declines today only because its
		// DeltaNet layers load no q/k/o, so the real guard was never exercised by any fixture. Here
		// the projections are present and valid, so the ONLY thing that can refuse it is the
		// recurrent-state check itself — remove that check and this case goes batched and wrong.
		{"deltanet-with-projections", func() *cudaResident {
			r := declineFixture(2, "int4")
			r.dnet = &dnetParams{}
			return r
		}()},
		{"mixed-quant-layer1", func() *cudaResident {
			r := declineFixture(2, "int4")
			r.layers[1].d.kind = "int8"
			return r
		}()},
		{"postOnly", func() *cudaResident {
			r := declineFixture(2, "int4")
			r.postOnly = true
			return r
		}()},
		{"qkNormWhole", func() *cudaResident {
			r := declineFixture(2, "int4")
			r.qkNorm = true
			r.qkNormWhole = true
			return r
		}()},
		{"layerNorm-parallelBlock", func() *cudaResident {
			r := declineFixture(2, "int4")
			r.layerNorm = true
			r.parallelBlock = true
			return r
		}()},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := c.r.prefillStaticDecline()
			batched, why := c.r.PrefillPath()
			if (err == nil) != batched {
				t.Fatalf("guard says err=%v but PrefillPath says batched=%v (%s)", err, batched, why)
			}
			if err != nil && !errors.Is(err, errPrefillDeclined) {
				t.Errorf("static decline must wrap errPrefillDeclined so callers can fall back: %v", err)
			}
			if !batched && !strings.HasPrefix(why, "sequential") {
				t.Errorf("declined reason should start with the resolved path: %q", why)
			}
		})
	}
}

// TestPrefillPath_mixedQuantNamesTheKind: a bundle that is int4 at layer 0 but int8 deeper (int4mix)
// still declines. The message falls back to the generic form naming the layer — worse than the
// int8int8 message, but it must not claim the batched path.
// TestPrefillPath_mixedInt4Int8Batches: an int4mix-style bundle (int4 in most projections, int8 in
// one) now gets batched prefill — dispatch is per projection, so a mix of the two batchable kinds
// "falls out for free" (§C6). A genuinely non-batchable kind (native/f32) at a specific layer still
// declines with the layer located.
func TestPrefillPath_mixedInt4Int8Batches(t *testing.T) {
	r := declineFixture(3, "int4")
	r.layers[2].g.kind = "int8"
	if batched, why := r.PrefillPath(); !batched {
		t.Fatalf("a mixed int4/int8 (int4mix) bundle did not batch: %q", why)
	}
	// A native/f32 projection is not batchable and must decline, naming the layer.
	r.layers[2].g.kind = "f32"
	batched, why := r.PrefillPath()
	if batched {
		t.Fatalf("a bundle with an f32 projection reported batched: %q", why)
	}
	if !strings.Contains(why, "layer 2") {
		t.Errorf("reason should locate the declining layer: %q", why)
	}
}

// TestPrefillPath_deltaNetBindsItsOwnProjections pins what replaced the blanket recurrent decline
// (docs/tasks/task-cuda-deltanet-prefill-2026-09.md): a Gated-DeltaNet model IS admitted to batched prefill, and the
// admission is decided by the projections prefillDeltaNetRows actually binds — dnQKV/dnB/dnA/dnZ/dnOut — not by q/k/o.
//
// The fixture's layers carry valid int4 q/k/o, which a DeltaNet layer never binds. If nonBatchableKind checked those
// instead of the DeltaNet five, the first half below would be admitted with absent DeltaNet weights and run the
// batched GEMVs over nil buffers; that is the LFM2 bug class (audit-2026-09-02 C-01) the kind dispatch exists to stop.
func TestPrefillPath_deltaNetBindsItsOwnProjections(t *testing.T) {
	r := declineFixture(2, "int4")
	r.dnet = &dnetParams{}
	for i := range r.layers {
		r.layers[i].isDeltaNet = true
	}
	err := r.prefillStaticDecline()
	if err == nil {
		t.Fatal("a DeltaNet layer with NO DeltaNet projections was admitted — nonBatchableKind is checking q/k/o, " +
			"which a DeltaNet layer never binds")
	}
	if !errors.Is(err, errPrefillDeclined) {
		t.Errorf("decline must wrap errPrefillDeclined so the caller falls back: %v", err)
	}
	w := cudaWQ{kind: "int4", N: 8, K: 32}
	for i := range r.layers {
		L := &r.layers[i]
		L.dnQKV, L.dnB, L.dnA, L.dnZ, L.dnOut = w, w, w, w, w
	}
	if e := r.prefillStaticDecline(); e != nil {
		t.Fatalf("a Gated-DeltaNet model with int4 DeltaNet projections must batch, got %v", e)
	}
	r.layers[1].dnOut.kind = "f32"
	if e := r.prefillStaticDecline(); e == nil {
		t.Fatal("an f32 dnOut was admitted — the M-wide GEMV takes int4/int8 only")
	}
}

// TestPrefillPath_seamGuardsAreMoEOnly pins the SCOPE of the per-token debug-seam declines, which
// is a different property from whether they exist.
//
// They exist because prefill calls layerTail from exactly one site — inside the per-row MoE FFN
// loop — where a per-token seam would fire M times per layer and hand its consumer M rows where it
// expects one. On a DENSE model that site is never reached, so refusing there gains nothing and
// costs a real feature: DFlash's block drafter arms hidCapTaps and verifies through the batched
// path on a dense model.
//
// The first version of the guard was not scoped, and it broke exactly that —
// TestDFlashRoundComposition and TestDFlashCompositionResidual both failed with "per-token
// hidden-state taps are armed". They are heavy, GPU-only and 40+ minutes into the suite; this
// costs microseconds and fails for the same reason, which is the point of writing it down here.
func TestPrefillPath_seamGuardsAreMoEOnly(t *testing.T) {
	armed := func(moe bool) *cudaResident {
		r := declineFixture(2, "int4")
		r.hidCapTaps = []int{0}
		r.moe = moe
		return r
	}
	if err := armed(false).prefillStaticDecline(); err != nil {
		t.Errorf("DENSE model with hidden-state taps armed was refused (%v) — prefill never reaches "+
			"layerTail on a dense model, so the taps are untouched and this refusal only breaks the "+
			"block drafter's batched verify", err)
	}
	if err := armed(true).prefillStaticDecline(); err == nil {
		t.Error("MoE model with hidden-state taps armed was ADMITTED — the per-row FFN loop calls " +
			"layerTail once per row, so each tap would record M residuals instead of one")
	}
	// Same shape for the layerCap probe, which rides the same site.
	r := declineFixture(2, "int4")
	r.layerCap, r.moe = true, false
	if err := r.prefillStaticDecline(); err != nil {
		t.Errorf("DENSE model with layerCap armed was refused: %v", err)
	}
	r.moe = true
	if err := r.prefillStaticDecline(); err == nil {
		t.Error("MoE model with layerCap armed was ADMITTED — it would append M snapshots per layer")
	}
}

// TestPrefillPath_normPlacementsBatch verifies that Olmo 3 (postOnly, qkNormWhole) and
// Cohere/Command-R (layerNorm, parallelBlock) report batched prefill and are admitted by
// prefillStaticDecline without falling back to sequential prefill.
func TestPrefillPath_normPlacementsBatch(t *testing.T) {
	// Olmo 3: postOnly + qkNormWhole
	rOlmo := declineFixture(2, "int4")
	rOlmo.postOnly = true
	rOlmo.qkNorm = true
	rOlmo.qkNormWhole = true
	if err := rOlmo.prefillStaticDecline(); err != nil {
		t.Errorf("Olmo 3 (postOnly + qkNormWhole) was refused by prefillStaticDecline: %v", err)
	}
	if batched, why := rOlmo.PrefillPath(); !batched {
		t.Errorf("Olmo 3 reported as not batched: %s", why)
	}

	// Cohere / Command-R: layerNorm + parallelBlock
	rCohere := declineFixture(2, "int4")
	rCohere.layerNorm = true
	rCohere.parallelBlock = true
	if err := rCohere.prefillStaticDecline(); err != nil {
		t.Errorf("Cohere (layerNorm + parallelBlock) was refused by prefillStaticDecline: %v", err)
	}
	if batched, why := rCohere.PrefillPath(); !batched {
		t.Errorf("Cohere reported as not batched: %s", why)
	}
}
