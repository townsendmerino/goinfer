//go:build darwin

package metal

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMetalResident_C09_execErrSurfaces is the goinfer-side proof for audit C-09: a command-buffer
// abort recorded by any forward path must surface through the metalResident adapter as an error
// instead of the stale logits the aborted buffer left behind — and must clear so the next token is
// unaffected. It runs a REAL GPU decode on a tiny dense resident, so it also proves the happy path
// never false-positives (recordExecErr(nil) is a no-op, execErr stays nil across normal decode).
//
// Why the abort is injected via recordExecErr rather than a real GPU fault: this machine's GPU
// silently tolerates every abort trigger (see aikit gpu.TestCmdBufStatus_C09) — status Error can't
// be provoked here — so the aikit layer verifies Encoder.Err()'s objc status read + NSError
// formatting on device, and this test verifies the goinfer propagation/clearing logic on device.
func TestMetalResident_C09_execErrSurfaces(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	w := genTinyWeights(rand.New(rand.NewSource(909)))
	dir := t.TempDir()
	writeDense(t, dir, w)
	m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load dense: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	a := &metalResident{r: r, hidden: r.H}
	t.Cleanup(func() { a.Close() })

	emb := make([]float32, a.hidden) // a zero embedding is a fine decode input for the mechanism test

	// (1) Happy path: a real decode returns logits and NO error, and leaves execErr clear.
	logits, err := a.Forward(emb, 0)
	if err != nil {
		t.Fatalf("clean Forward returned error %v — C-09 false positive on normal decode", err)
	}
	if len(logits) != r.V {
		t.Fatalf("clean Forward: got %d logits, want V=%d", len(logits), r.V)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("clean Forward: logit[%d] = %v (non-finite)", i, v)
		}
	}
	if r.execErr != nil {
		t.Fatalf("execErr = %v after a clean decode, want nil", r.execErr)
	}

	// (2) Inject an abort the way a faulted command buffer would (recordExecErr latches the first
	// non-nil error; the clean decode's recordExecErr(nil) inside this next Forward is a no-op, so
	// the injected error survives to takeExecErr). The adapter must return THAT error, not logits.
	sentinel := errors.New("metal: command buffer aborted: simulated GPU fault (C-09 test)")
	r.recordExecErr(sentinel)
	got, err := a.Forward(emb, 1)
	if err == nil {
		t.Fatal("Forward returned nil error after an abort was recorded — C-09: caller would trust stale logits")
	}
	if err != sentinel {
		t.Fatalf("Forward surfaced %v, want the recorded abort %v", err, sentinel)
	}
	if got != nil {
		t.Fatalf("Forward returned %d logits alongside the error — must return nil logits on abort", len(got))
	}

	// (3) The error must have been consumed: the next decode is clean again.
	if r.execErr != nil {
		t.Fatalf("execErr = %v after takeExecErr, want cleared", r.execErr)
	}
	if _, err := a.Forward(emb, 2); err != nil {
		t.Fatalf("Forward after a consumed abort returned %v, want nil (error must not stick)", err)
	}
}

// TestMetalResident_C11_argmaxEqualsFullLogits is the device gate for audit C-11: the fused
// block-argmax LM head (ForwardArgmax → gemv_w8a8_amax tile partials → argmax_finish reduce) must
// pick the SAME token as argmax over the full-logits Forward. C-11 sized r.part/r.uP with floor
// V/8, but the amax kernel dispatches ceil(V/8) tiles and writes part[tgid] unconditionally, so a
// V%8 != 0 both wrote past r.part and left the last tile out of uP's reduce — the greedy token
// could diverge from argmax(Forward). The fix is nTiles := (V+7)/8 (ceil), correct by construction:
// r.part holds every tile the dispatch writes and uP counts them all.
//
// This exercises the ceil-sized reduction on device for a %8 vocab (tmVocab=64) — a committed,
// always-runnable fixture, so it needs no heavy model.
//
// N-33: this used to say "C-10's buildResident guard declines any non-%8 vocab to the CPU". It
// does not, and model.go says so explicitly — vocab is NOT checked there, because the LM head is
// pinned int8 and dispatches gemv_w8a8_coal (no hazard) or gemv_w8a8_amax, and ForwardArgmax
// ROUTES a non-%8 vocab around the hazardous kernel (full logits + host argmax) rather than
// declining the family. The code is right; this comment described a guard that was considered
// and not built.
func TestMetalResident_C11_argmaxEqualsFullLogits(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	w := genTinyWeights(rand.New(rand.NewSource(1111)))
	dir := t.TempDir()
	writeDense(t, dir, w)
	m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load dense: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	t.Cleanup(func() { r.Close() })

	// Greedy walk: at each position the fused argmax must equal argmax over the full logits Forward
	// produces at the SAME (tok, pos). Both entry points apply the identical embed path (loadEmbedRow)
	// and write the same KV at pos, so they are directly comparable.
	tok := 1
	for pos := 0; pos < 12; pos++ {
		want := argmaxF(r.Forward(tok, pos))  // full lm head → host argmax
		got := int(r.ForwardArgmax(tok, pos)) // fused block-argmax (ceil-tiled reduce)
		if got != want {
			t.Fatalf("pos %d: ForwardArgmax=%d != argmax(Forward)=%d — fused block-argmax diverged (C-11)", pos, got, want)
		}
		tok = want
	}
}
