//go:build gpu

package gpu

import (
	"errors"
	"strings"
	"testing"
)

// Audit C-17 — batched verify must DECLINE on a recurrent model.
//
// WHY THIS EXISTS. residentDecoder.TruncateTo is a no-op, documented on the premise that "the
// resident cache is positional" — true of the KV, false of the Mamba-2 {win,ssm} state the same
// runner owns. mamba2Step mutates that state in place, so ForwardN(K) advances it K times and a
// partial accept cannot undo the rejected rows: the next round decodes from over-advanced state.
// Silently wrong output, not a crash.
//
// WHY IT WAS UNREACHABLE AND STILL NEEDS A GATE. decoder.specRollbackSafe refuses the recurrent
// families at the four speculative entry points, so nothing calls this today with mamba set. That
// protection lives in ANOTHER PACKAGE, one indirection away from the state it protects — a future
// "re-enable recurrent speculation" change relaxes that check and never reads this function. The
// guard therefore belongs on the runner that owns the state. This test pins it there.
//
// NO DEVICE NEEDED: the guard runs before any runner/queue is touched, and newRunner is a func
// field, so the control case is stubbable.

// recurrentDecoder builds the minimal residentDecoder the guard inspects: a context cap big enough
// that checkCap passes, and mamba set or not. Nothing else is initialised — reaching past the guard
// with a nil runner is itself the failure this test detects.
func recurrentDecoder(mamba bool) *residentDecoder {
	rd := &residentDecoder{ctxCap: 4096}
	if mamba {
		rd.rm.mamba = &mambaRunParams{dConv: 4, convDim: 8, nHeads: 2, hp: 2, dn: 4}
	}
	return rd
}

// TestForwardN_declinesRecurrent is the gate: K>1 on a Mamba-2 model must return an error rather
// than advance unrollbackable state.
func TestForwardN_declinesRecurrent(t *testing.T) {
	rd := recurrentDecoder(true)
	embs := [][]float32{{0, 0}, {0, 0}}

	out, err := rd.ForwardN(embs, 0)
	if err == nil {
		t.Fatal("ForwardN(K=2) accepted a recurrent model — it would advance {win,ssm} for rows a " +
			"partial accept then rejects, and the next round decodes from over-advanced state")
	}
	if out != nil {
		t.Errorf("declining ForwardN returned logits %v, want nil", out)
	}
	// The reason must name the actual condition; "declined" alone sends the reader to the wrong layer.
	for _, want := range []string{"recurrent", "roll"} {
		if !strings.Contains(strings.ToLower(err.Error()), want) {
			t.Errorf("decline reason %q does not mention %q", err.Error(), want)
		}
	}
}

// TestForwardN_doesNotOverFire: the guard must not refuse a NON-recurrent model, or every
// speculative verify on the shipped families turns into plain decode — a silent throughput
// regression traded for the silent correctness one. The stubbed newRunner proves control reached
// the build path, i.e. the guard let it through.
func TestForwardN_doesNotOverFire(t *testing.T) {
	sentinel := errors.New("newRunner reached")
	rd := recurrentDecoder(false)
	rd.newRunner = func() (*DecodeRunner, error) { return nil, sentinel }

	_, err := rd.ForwardN([][]float32{{0, 0}, {0, 0}}, 1)
	if !errors.Is(err, sentinel) {
		t.Fatalf("ForwardN on a non-recurrent model did not reach the runner build (got %v) — the "+
			"recurrent guard is over-firing and would demote every speculative verify to plain decode", err)
	}
}
