package decoder

import "testing"

// Theta's domain and the per-backend wiring.
//
// The bug these pin: AdaptiveDepth documented and enforced Theta in [0,1) and
// reset anything outside it to 0.5. Metal measures 1.006-1.048 (two models, two
// depths, 2026-09-01), so EVERY value Metal actually has was rejected and
// replaced by 0.5 — the most over-drafting setting on the dial, chosen
// automatically at the one moment the measurement said "do not draft at all".

// TestTheta_geOneIsLegalAndDisablesDrafting is the domain fix. A Theta >= 1 must
// survive ensure() and must produce depth 0 for every acceptance rate, because
// alpha < 1 always and a node costing a full step can never be repaid.
func TestTheta_geOneIsLegalAndDisablesDrafting(t *testing.T) {
	for _, theta := range []float64{1.0, 1.02, 1.048, 2.0} {
		a := &AdaptiveDepth{Theta: theta}
		a.ensure()
		if a.Theta != theta {
			t.Fatalf("Theta %v was rewritten to %v — the domain still rejects the measured value", theta, a.Theta)
		}
		// Every alpha, including one that would otherwise draft the maximum.
		for _, alpha := range []float64{0.1, 0.5, 0.9, 0.99, 0.9999} {
			a.alpha = alpha
			if d := a.Depth(8); d != 0 {
				t.Fatalf("Theta=%v alpha=%v: drafted %d, want 0 (a node costs a whole step)", theta, alpha, d)
			}
		}
	}
}

// TestTheta_geOneSkipsTheProbe. The periodic probe forces D>=1 after ProbeEvery
// idle rounds to refresh a stale alpha. When Theta >= 1 the decision does not
// depend on alpha at all, so the probe cannot change any future answer and is
// pure wasted draft work — a token drafted and thrown away every 16 rounds,
// forever, on Metal.
//
// This is a separate assertion from the one above on purpose: the probe check
// sits BEFORE the alpha<=Theta test in Depth(), so a fix that only corrected the
// comparison would leave this path drafting and the test above would still pass.
func TestTheta_geOneSkipsTheProbe(t *testing.T) {
	a := &AdaptiveDepth{Theta: 1.02, ProbeEvery: 4}
	a.ensure()
	for range 20 { // well past ProbeEvery
		if d := a.Depth(8); d != 0 {
			t.Fatalf("probe fired under Theta>=1 (drafted %d); it can never change the answer", d)
		}
		a.Observe(0, 0) // the D=0 round that advances idle
	}
	if a.idle < 4 {
		t.Fatalf("idle=%d — the probe path was never actually reached, so this test proved nothing", a.idle)
	}
}

// TestTheta_belowOneStillDrafts guards the fix from over-reaching: the whole
// point is that Theta < 1 behaves exactly as before.
func TestTheta_belowOneStillDrafts(t *testing.T) {
	a := &AdaptiveDepth{Theta: 0.25}
	a.ensure()
	a.alpha = 0.9
	d := a.Depth(8)
	if d <= 0 {
		t.Fatalf("Theta=0.25 alpha=0.9 drafted %d, want > 0 — the disable branch is too greedy", d)
	}
	// And the CUDA constant must draft DEEPER than the CPU one at equal alpha,
	// which is the entire reason for wiring it per backend.
	shallow := &AdaptiveDepth{Theta: thetaFor("cpu")}
	shallow.ensure()
	shallow.alpha = 0.9
	deep := &AdaptiveDepth{Theta: thetaFor("cuda")}
	deep.ensure()
	deep.alpha = 0.9
	if !(deep.Depth(8) > shallow.Depth(8)) {
		t.Fatalf("cuda depth %d not deeper than cpu depth %d at alpha=0.9 — the table is not doing anything",
			deep.Depth(8), shallow.Depth(8))
	}
}

// TestThetaFor_measuredValues pins the table to the measurements, so a future
// edit that "tidies" a constant has to change a test that says where it came
// from. Bounds, not exact equality: the point is the regime each backend is in.
func TestThetaFor_measuredValues(t *testing.T) {
	if got := thetaFor("metal"); got < 1.0 {
		t.Fatalf("metal Theta %v < 1.0 — measured 1.006-1.048; below 1 re-enables drafting on a loop ForwardN", got)
	}
	if got := thetaFor("cuda"); got <= 0 || got >= 0.5 {
		t.Fatalf("cuda Theta %v outside the measured 0.155-0.251 regime", got)
	}
	if got := thetaFor("cpu"); got != defaultTheta {
		t.Fatalf("cpu Theta %v != %v (re-measured 0.506/0.532 on 2026-09-01)", got, defaultTheta)
	}
	if got := thetaFor("nonesuch-backend"); got != defaultTheta {
		t.Fatalf("unknown backend got %v, want the %v fallback", got, defaultTheta)
	}
}

// TestVerifyTheta_stagedGetsCPUValue is the subtle one. Theta describes the path
// the VERIFY runs on. genNgramInto uses resident.ForwardN only when residency
// built and otherwise falls back to the CPU batched forwardN — so a model whose
// residency DECLINED must get the CPU constant even though its backend is a GPU.
// Keying off the requested backend alone would hand a declined model the GPU
// value, mis-tuning exactly the case where the decline already costs the user
// the whole forward.
func TestVerifyTheta_stagedGetsCPUValue(t *testing.T) {
	staged := &Model{be: cpuBackendForTest(t), resident: nil}
	if got := staged.verifyTheta(); got != defaultTheta {
		t.Fatalf("staged model got Theta %v, want the CPU %v — it verifies on the CPU path", got, defaultTheta)
	}
	var nilModel *Model
	if got := nilModel.verifyTheta(); got != defaultTheta {
		t.Fatalf("nil model got %v, want %v", got, defaultTheta)
	}
}

func cpuBackendForTest(t *testing.T) Backend {
	t.Helper()
	be, err := NewBackend("cpu")
	if err != nil {
		t.Fatalf("NewBackend(cpu): %v", err)
	}
	return be
}
