package decoder

import (
	"context"
	"strings"
	"testing"
)

// The prefill half of the seam gate (resident_seam_test.go).
//
// WHY THIS EXISTS. The resident DECODE path being silently CPU-only was one bug class; the batched
// PREFILL declining silently is the same class one layer down, and it shipped. `--backend cuda
// --quant int8int8` builds a full resident decode path (ResidentActive is true, decode runs at 0.7×
// int4 — everything looks healthy), but the batched prefill GEMV is int4-only, so every prompt takes
// the sequential per-token loop instead of one weight-stationary pass: measured 1.73 s vs 0.19 s on a
// 300-token prompt (9×), 4.56 vs 0.22 CPU-seconds (20×). generateInto's fallback discards the decline
// error by design, so nothing — no log, no field, no error — said so.
//
// These tests need no GPU: they fake residents that decline the way the CUDA backend declines.

// prefillingResident is a fakeResident that also implements Prefiller AND PrefillPathReporter — the
// shape a reporting backend (cuda) presents. prefillerOnly below is the non-reporting shape.
type prefillingResident struct {
	fakeResident
	batched bool
	reason  string
}

func (p *prefillingResident) PrefillLast(_ context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return p.Forward(embeddings[len(embeddings)-1], startPos+len(embeddings)-1)
}

func (p *prefillingResident) PrefillPath() (bool, string) { return p.batched, p.reason }

// residentModel returns a Model whose resident field is rf, without a backend build — PrefillPath
// reads only m.resident (and m.be for the name), so this is the whole seam under test.
func residentModel(rf ResidentForward) *Model {
	return &Model{resident: rf}
}

// TestPrefillPath_declineIsReported is the gate: a backend that declines the batched prefill must
// surface WHY at load time. Before this, the decline existed only as an error value thrown away
// inside generateInto.
func TestPrefillPath_declineIsReported(t *testing.T) {
	const want = "sequential — batched prefill requires int4 projections (int8 weights) — ~9× slower TTFT"
	rf := &prefillingResident{batched: false, reason: want}
	batched, why := residentModel(rf).PrefillPath()
	if batched {
		t.Fatal("PrefillPath reported batched=true for a resident that declines — a 9× TTFT regression " +
			"would be reported as the fast path")
	}
	if why != want {
		t.Fatalf("reason not propagated verbatim:\n got: %s\nwant: %s", why, want)
	}
	// The reason must name the CONDITION, not just the outcome: "declined" tells an operator nothing.
	if !strings.Contains(why, "int4") {
		t.Errorf("reason does not name the actual condition: %q", why)
	}
}

// TestPrefillPath_batchedWhenAccepted: the accepting case must not be reported as a decline, or
// -require-backend would refuse to start a perfectly good server.
func TestPrefillPath_batchedWhenAccepted(t *testing.T) {
	rf := &prefillingResident{batched: true, reason: "batched (one weight-stationary CUDA pass)"}
	if batched, why := residentModel(rf).PrefillPath(); !batched {
		t.Fatalf("accepting backend reported as declined: %s", why)
	}
}

// TestPrefillPath_noPrefillerIsSequential: a resident backend with no batched prefill at all runs one
// forward per prompt token. That is a decline too — it just has a different cause.
func TestPrefillPath_noPrefillerIsSequential(t *testing.T) {
	batched, why := residentModel(&fakeResident{vocab: 8}).PrefillPath()
	if batched {
		t.Fatal("a resident with no Prefiller reported batched prefill")
	}
	if !strings.Contains(why, "sequential") {
		t.Errorf("reason should say sequential: %q", why)
	}
}

// TestPrefillPath_prefillerWithoutReporter: a Prefiller that doesn't report (metal today) must not be
// claimed as an unconditional batched path — it can still decline per prompt. The message has to say
// so rather than overclaim.
func TestPrefillPath_prefillerWithoutReporter(t *testing.T) {
	rf := &prefillingResident{batched: true}
	var res ResidentForward = &prefillerOnly{rf}
	batched, why := residentModel(res).PrefillPath()
	if !batched {
		t.Fatalf("unreporting Prefiller treated as a decline: %s", why)
	}
	if !strings.Contains(why, "fallback is still possible") {
		t.Errorf("unreporting backend overclaims a guaranteed batched path: %q", why)
	}
}

// prefillerOnly wraps a resident to expose Prefiller WITHOUT PrefillPathReporter (embedding the
// struct directly would promote its PrefillPath method).
type prefillerOnly struct{ p *prefillingResident }

func (w *prefillerOnly) Forward(e []float32, pos int) ([]float32, error) { return w.p.Forward(e, pos) }
func (w *prefillerOnly) ForwardN(e [][]float32, s int) ([][]float32, error) {
	return w.p.ForwardN(e, s)
}
func (w *prefillerOnly) PrefillLast(_ context.Context, e [][]float32, s int) ([]float32, error) {
	return w.p.PrefillLast(context.Background(), e, s)
}
func (w *prefillerOnly) UploadKV(layer int, keys, vals []float32) error { return nil }
func (w *prefillerOnly) TruncateTo(pos int)                             {}
func (w *prefillerOnly) Reset()                                         {}
func (w *prefillerOnly) Close() error                                   { return nil }

// TestPrefillPath_envForceSequential: GOINFER_BATCHED_PREFILL=0 is an A/B escape hatch that turns
// every resident model sequential. Left set in an environment, it is exactly the invisible 9× this
// whole seam exists to expose — so it must be reported as the cause.
func TestPrefillPath_envForceSequential(t *testing.T) {
	t.Setenv("GOINFER_BATCHED_PREFILL", "0")
	rf := &prefillingResident{batched: true, reason: "batched"}
	batched, why := residentModel(rf).PrefillPath()
	if batched {
		t.Fatal("GOINFER_BATCHED_PREFILL=0 forces the per-token loop, but PrefillPath reported batched")
	}
	if !strings.Contains(why, "GOINFER_BATCHED_PREFILL") {
		t.Errorf("reason should name the env override: %q", why)
	}
}

// TestPrefillPath_stagedModelReportsCPUBatching covers the non-resident half: the staged/CPU path has
// its own batched prefill (forwardLayersN) with its own decline (canBatchN excludes the families with
// a sequential-only forward), and PrefillPath must answer for it too rather than claiming residency.
func TestPrefillPath_stagedModelReportsCPUBatching(t *testing.T) {
	m, err := Load(tinyFixture(t), Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load tiny fixture: %v", err)
	}
	defer m.Close()
	batched, why := m.PrefillPath()
	if batched != m.canBatchN(2) {
		t.Fatalf("PrefillPath (%v) disagrees with canBatchN (%v) — the staged report is not reading the "+
			"gate that actually decides: %s", batched, m.canBatchN(2), why)
	}
	if why == "" {
		t.Fatal("empty prefill reason")
	}
	if dp := m.DecodePath(); !strings.HasPrefix(dp, "cpu (") {
		t.Errorf("DecodePath for a cpu-backend model = %q, want cpu (<quant>)", dp)
	}
}
