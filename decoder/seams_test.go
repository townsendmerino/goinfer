package decoder

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// Gates H, C and S of docs/tasks/task-forward-architecture-2026-10.md §4.8 ("OP3's seams PR"), on every tiny fixture whose family
// walks a program, each with its planted defect (seamPlantForTest). A gate returns the fixtures it found wrong; one it cannot
// reach is reported, never passed.

func sameBits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

// seamLogits runs the plain per-token forward over ids on a fresh cache and returns the last logits.
func seamLogits(t *testing.T, m *Model, ids []int) []float32 {
	t.Helper()
	cache := m.NewCache(len(ids))
	var lg []float32
	for _, id := range ids {
		out, err := m.forward(id, cache)
		if err != nil {
			t.Fatal(err)
		}
		lg = append(lg[:0], out...)
	}
	return lg
}

// gateH: HiddenLast's row is the forward's final hidden state (the residual after the last token, head-normed), and the head
// over it is the forward's last logits.
func gateH(t *testing.T, m *Model, ids []int) error {
	cache := m.NewCache(len(ids))
	var h []float32
	for _, id := range ids {
		var err error
		if h, err = m.runLayers(id, cache); err != nil {
			return err
		}
	}
	want := append([]float32(nil), h...)
	hp := &m.program().head
	b := m.w.FinalNormBias
	if !hp.Norm.Bias {
		b = nil
	}
	applyNorm(hp.Norm, want, want, m.w.FinalNorm, b, m.w.arch.HiddenDim) // the head's norm, not the seam's own path
	got, err := m.HiddenLast(ids)
	if err != nil {
		return err
	}
	if !sameBits(got, want) {
		return fmt.Errorf("HiddenLast's row is not the forward's final hidden state")
	}
	if !sameBits(m.logitsFromNormed(append([]float32(nil), got...), m.NewCache(1)), seamLogits(t, m, ids)) {
		return fmt.Errorf("the head over HiddenLast's row is not the forward's last logits")
	}
	return nil
}

// captureAll feeds ids[:n-1] to a fresh cache, then captures every layer's residual at the last token.
func captureAll(m *Model, ids []int) ([]float32, [][]float32, error) {
	n := len(ids)
	cache := m.NewCache(n)
	for _, id := range ids[:n-1] {
		if _, err := m.forward(id, cache); err != nil {
			return nil, nil, err
		}
	}
	layers := make([]int, m.w.arch.NumLayers)
	for l := range layers {
		layers[l] = l
	}
	lg, hidden, err := m.ForwardCapture(ids[n-1], cache, layers)
	return append([]float32(nil), lg...), hidden, err
}

// gateC: the residual captured after layer l is the forward over the same prompt with the program cut to layers 0..l; the
// capture leaves the logits as they were.
func gateC(t *testing.T, m *Model, ids []int) error {
	lg, hidden, err := captureAll(m, ids)
	if err != nil {
		return err
	}
	if !sameBits(lg, seamLogits(t, m, ids)) {
		return fmt.Errorf("ForwardCapture changed the logits")
	}
	for l := range m.w.arch.NumLayers {
		cut := *m.program()
		cut.layers = cut.layers[:l+1]
		mm := *m
		mm.prog = &cut
		cache := mm.NewCache(len(ids))
		cache.manualPos = true // the cut program never reaches the last layer, whose append would step the position
		var h []float32
		for _, id := range ids {
			if h, err = mm.runLayers(id, cache); err != nil {
				return err
			}
		}
		if !sameBits(hidden[l], h) {
			return fmt.Errorf("layer %d: the captured residual is not the forward cut after that layer", l)
		}
	}
	return nil
}

// gateS: every layer from 1 on is rebuilt bit for bit from the residual before it and the captured contributions, applied as
// the layer's own residual operations apply them. A layer with an in-place operation the contributions do not cover is skipped.
func gateS(t *testing.T, m *Model, ids []int) (checked, skipped int, err error) {
	_, resid, err := captureAll(m, ids)
	if err != nil {
		return 0, 0, err
	}
	n := len(ids)
	cache := m.NewCache(n + 1)
	for _, id := range ids[:n-1] {
		if _, err := m.forward(id, cache); err != nil {
			return 0, 0, err
		}
	}
	attn, mlp, _, _, err := m.ForwardSubCapture(ids[n-1], cache)
	if err != nil {
		return 0, 0, err
	}
	prog := m.program()
layers:
	for l := 1; l < len(prog.layers); l++ {
		h := append([]float32(nil), resid[l-1]...)
		lastFFN := false
		for _, o := range prog.layers[l] {
			switch o.kind {
			case opFFN:
				lastFFN = true
			case opProjO, opMLA, opKDA, opShortConv, opMamba2, opDeltaNet, opGatedAttn, opGemma4Attn:
				lastFFN = false
			case opGemma4MoE, opPLE:
				skipped++
				continue layers
			case opLayerScale:
				if s := m.w.Layers[l].LayerScalar; s != 0 {
					for i := range h {
						h[i] *= s
					}
				}
			case opResidual:
				switch {
				case o.res.Two:
					addResidual2(h, attn[l], mlp[l])
				default:
					c := attn[l]
					if lastFFN {
						c = mlp[l]
					}
					if o.res.Scale != 0 {
						addScaled(h, c, float32(o.res.Scale))
					} else {
						addResidual(h, c)
					}
				}
			}
		}
		if !sameBits(h, resid[l]) {
			return checked, skipped, fmt.Errorf("layer %d: the residual is not rebuilt from the one before it and the captured contributions", l)
		}
		checked++
	}
	// The capture leaves the stream as it was: the next token's logits after it equal those after a plain step.
	plain := m.NewCache(n + 1)
	for _, id := range ids {
		if _, err := m.forward(id, plain); err != nil {
			return checked, skipped, err
		}
	}
	lp, err := m.forward(ids[0], plain)
	if err != nil {
		return checked, skipped, err
	}
	lp = append([]float32(nil), lp...)
	lc, err := m.forward(ids[0], cache)
	if err != nil {
		return checked, skipped, err
	}
	if !sameBits(lp, lc) {
		return checked, skipped, fmt.Errorf("ForwardSubCapture changed the stream: the next token's logits differ")
	}
	return checked, skipped, nil
}

func TestSeams_gates(t *testing.T) {
	fx := programFixtures(t)
	for _, f := range fx {
		ids := programPrompt(f.m)
		if err := gateH(t, f.m, ids); err != nil {
			t.Errorf("H %s: %v", f.name, err)
		}
		if err := gateC(t, f.m, ids); err != nil {
			t.Errorf("C %s: %v", f.name, err)
		}
		if f.m.program().subCaptureUncovered() {
			t.Logf("S %s NOT RUN: ForwardSubCapture refuses it (a PLE branch or MoE block; TestSeams_subCaptureRefusedWhereGateSCannotRun)", f.name)
			continue
		}
		checked, skipped, err := gateS(t, f.m, ids)
		switch {
		case err != nil:
			t.Errorf("S %s: %v", f.name, err)
		case checked == 0:
			t.Errorf("S %s: no layer rebuilt (%d skipped), but ForwardSubCapture serves it", f.name, skipped)
		default:
			t.Logf("S %s: %d layers rebuilt, %d skipped", f.name, checked, skipped)
		}
	}
	t.Logf("H, C, S over %d fixtures", len(fx))
}

// TestSeams_subCaptureRefusedWhereGateSCannotRun pins the owner's condition (2026-10-16): ForwardSubCapture, and its bisect hook,
// refuse exactly the fixtures where gate S cannot rebuild a layer (Gemma 4's E-models and 26B MoE: the PLE branch and the MoE
// block), and serve every other one. HiddenLast and ForwardCapture serve all of them (TestSeams_gates).
func TestSeams_subCaptureRefusedWhereGateSCannotRun(t *testing.T) {
	refused, present := map[string]bool{}, map[string]bool{}
	for _, f := range programFixtures(t) {
		present[f.name] = true
		ids := programPrompt(f.m)
		cache := f.m.NewCache(len(ids))
		_, _, _, _, err := f.m.ForwardSubCapture(ids[0], cache)
		if f.m.program().subCaptureUncovered() {
			if err == nil || !strings.Contains(err.Error(), "do not cover") {
				t.Errorf("%s: ForwardSubCapture did not refuse (%v)", f.name, err)
			}
			refused[f.name] = true
			continue
		}
		if err != nil {
			t.Errorf("%s: ForwardSubCapture refused a fixture gate S covers: %v", f.name, err)
		}
		// Every served fixture rebuilds at least one layer: the refusal is exactly where gate S cannot run.
		if checked, _, err := gateS(t, f.m, ids); err != nil || checked == 0 {
			t.Errorf("%s: served, but gate S rebuilt %d layers (%v)", f.name, checked, err)
		}
	}
	// The four the owner's condition names; three are gitignored, so a box without them (CI) reports them NOT RUN. gemma4-moe-kv-tiny32 is
	// gemma4-moe-kv-tiny at CUDA's int4 widths (the windowed-KV expert-cache test's fixture): the same shape, refused for the same reason.
	want := []string{"gemma4-emodel-tiny", "gemma4-moe-kv-tiny", "gemma4-moe-kv-tiny32", "gemma4-moe-tiny", "gemma4-moe-unified-tiny"}
	wantPresent := 0
	for _, w := range want {
		switch {
		case !present[w]:
			t.Logf("%s NOT RUN: the fixture is not present (a skip is not a pass)", w)
		case !refused[w]:
			t.Errorf("%s is not refused: the owner's condition names it", w)
		default:
			wantPresent++
		}
	}
	if len(refused) != wantPresent {
		t.Errorf("refused %d fixtures, want exactly the %d present ones gate S could not run on: %v", len(refused), wantPresent, refused)
	}
}

// TestSeams_plantsGoRed: each planted defect turns its gate red on at least one fixture.
func TestSeams_plantsGoRed(t *testing.T) {
	fx := programFixtures(t)
	t.Cleanup(func() { seamPlantForTest = plantNone })
	for _, pl := range []struct {
		name  string
		plant seamPlant
		gate  func(f programFixture, ids []int) error
	}{
		{"H, the final norm skipped", plantHiddenNoNorm, func(f programFixture, ids []int) error { return gateH(t, f.m, ids) }},
		{"C, the capture one operation early", plantCaptureEarly, func(f programFixture, ids []int) error { return gateC(t, f.m, ids) }},
		{"S, the attention contribution before its post-norm", plantSubPrePostNorm, func(f programFixture, ids []int) error {
			_, _, err := gateS(t, f.m, ids)
			return err
		}},
	} {
		var red []string
		ran := 0
		for _, f := range fx {
			if pl.plant == plantSubPrePostNorm && f.m.program().subCaptureUncovered() {
				continue // refused, not run: its error is not the plant being seen
			}
			ran++
			seamPlantForTest = pl.plant
			err := pl.gate(f, programPrompt(f.m))
			seamPlantForTest = plantNone
			if err != nil {
				red = append(red, f.name)
			}
		}
		t.Logf("plant %q: red on %d of %d fixtures it runs on (%s)", pl.name, len(red), ran, strings.Join(red, " "))
		if len(red) == 0 {
			t.Errorf("plant %q turned no fixture red: its gate does not see the defect", pl.name)
		}
	}
}
