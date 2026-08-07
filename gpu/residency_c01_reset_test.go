//go:build gpu && goinfer_testhooks

package gpu

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestResident_C01_pos0ResetsRecurrent is the resident-half gate for audit C-01: the compounding
// Mamba-2 {win,ssm} state must be re-zeroed at the start of a fresh sequence, or a second Generate
// on the same *Model decodes its first token from the PRIOR sequence's state (cross-conversation
// leak, silently wrong output). residentDecoder.Reset() existed but "was called by nothing"; the fix
// wired it into Forward at pos==0. This reproduces the leak directly: run token T at pos 0 from a
// fresh resident, COMPOUND the state by decoding several more tokens, then run token T at pos 0
// AGAIN — with the fix the second pos-0 logits are identical to the first; without it they diverge.
//
// Requires an SSM (Mamba-2 hybrid) resident, which only exists as a real model — this Mac carries no
// such checkpoint, so it runs on the CUDA/webgpu box (set GOINFER_HEAVY_TESTS=1; override the path
// with GOINFER_SSM_MODEL). The CPU rewind half is covered by decoder.TestTruncateTo_resetsRecurrent.
func TestResident_C01_pos0ResetsRecurrent(t *testing.T) {
	requireHeavyModel(t)
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	path := os.Getenv("GOINFER_SSM_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/granite/granite-4.0-h-tiny-Q8_0.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no SSM model (%s) — set GOINFER_SSM_MODEL to a Mamba-2 hybrid gguf", path)
	}

	os.Setenv("GOINFER_SSM_RESIDENT", "1")
	defer os.Unsetenv("GOINFER_SSM_RESIDENT")
	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load resident: %v", err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Fatal("model did not go resident")
	}
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Fatalf("resident forward is %T, not *residentDecoder", m.ResidentForwardForTest())
	}
	if rd.rm.mamba == nil {
		t.Skip("model has no Mamba-2 state — C-01 resident leak does not apply")
	}

	tok := 1
	emb0 := append([]float32(nil), m.EmbedResidentForTest(tok)...)

	// (1) First sequence, token `tok` at pos 0 from a fresh resident.
	l0, err := rd.Forward(emb0, 0)
	if err != nil {
		t.Fatalf("Forward pos 0 (first): %v", err)
	}
	first := append([]float32(nil), l0...)

	// (2) Compound the recurrent state: decode several more tokens so {win,ssm} evolves away from
	// zero. mamba2Step mutates the state in place, so after this it is NOT the fresh-sequence state.
	next := argmaxF(first)
	moved := false
	for pos := 1; pos <= 8; pos++ {
		lg, err := rd.Forward(m.EmbedResidentForTest(next), pos)
		if err != nil {
			t.Fatalf("Forward pos %d (compounding): %v", pos, err)
		}
		next = argmaxF(lg)
		moved = true
	}
	if !moved {
		t.Fatal("did not advance the state — test is vacuous")
	}

	// (3) Start a fresh sequence: the SAME token at pos 0. The fix re-zeroes {win,ssm} here, so the
	// logits must match the first pos-0 run. A mismatch means the prior sequence's state leaked in.
	l0b, err := rd.Forward(emb0, 0)
	if err != nil {
		t.Fatalf("Forward pos 0 (second): %v", err)
	}
	second := append([]float32(nil), l0b...)

	if len(first) != len(second) {
		t.Fatalf("logit length changed: %d vs %d", len(first), len(second))
	}
	var maxAbs float32
	for i := range first {
		d := first[i] - second[i]
		if d < 0 {
			d = -d
		}
		if d > maxAbs {
			maxAbs = d
		}
	}
	// Same token, same pos, re-zeroed state ⇒ identical up to nondeterministic GPU reduction order.
	const tol = 1e-3
	if maxAbs > tol {
		t.Fatalf("pos-0 logits differ by %.4g after an intervening sequence — recurrent {win,ssm} "+
			"state leaked across sequences (C-01: pos==0 must Reset). argmax first=%d second=%d",
			maxAbs, argmaxF(first), argmaxF(second))
	}
	t.Logf("C-01 resident OK: pos-0 logits reproducible across sequences (maxAbs=%.4g), state re-zeroed", maxAbs)
}
