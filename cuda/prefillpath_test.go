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
	for i := 0; i < n; i++ {
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
func TestPrefillPath_int8Declines(t *testing.T) {
	batched, why := declineFixture(4, "int8").PrefillPath()
	if batched {
		t.Fatal("int8 projections reported as batched prefill — this is the 9× TTFT regression being " +
			"reported as the fast path")
	}
	for _, want := range []string{"sequential", "int4", "int8", "TTFT"} {
		if !strings.Contains(why, want) {
			t.Errorf("reason %q does not mention %q — an operator can't act on it", why, want)
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
		{"moe", func() *cudaResident { r := declineFixture(2, "int4"); r.moe = true; return r }()},
		{"gemma4moe", func() *cudaResident { r := declineFixture(2, "int4"); r.gemma4Moe = true; return r }()},
		{"k-eq-v", func() *cudaResident { r := declineFixture(2, "int4"); r.layers[1].kEqV = true; return r }()},
		{"non-uniform", func() *cudaResident { r := declineFixture(2, "int4"); r.layers[1].nKV = 1; return r }()},
		{"mixed-quant-layer1", func() *cudaResident {
			r := declineFixture(2, "int4")
			r.layers[1].d.kind = "int8"
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
func TestPrefillPath_mixedQuantNamesTheKind(t *testing.T) {
	r := declineFixture(3, "int4")
	r.layers[2].g.kind = "int8"
	batched, why := r.PrefillPath()
	if batched {
		t.Fatal("a mixed-quant bundle reported batched prefill")
	}
	if !strings.Contains(why, "layer 2") {
		t.Errorf("reason should locate the declining layer: %q", why)
	}
}
