//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"os"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestAttentionFA_positionSweep answers a sharp objection to the "call-count, not position"
// conclusion in docs/measurements/r2-attn-fa-followup-2026-09-20.md: the two prefill depths
// already tried, 1600 and 2200, are BOTH multiples of 8 — so "third decode call" and "key count ≡
// 3 (mod 4)" were the same event in both runs, and the earlier experiment cannot tell them apart.
// attention_fa's own kernel groups work by 4 consecutive vocabulary-adjacent... no, by 4-key tiles
// (its own doc comment: "cooperative load, 32 lanes x half4"), so a boundary condition tied to
// nKeys mod 4 is a live, structurally-motivated alternative to "call count" that the prior
// experiment could not rule out.
//
// This sweeps prefillLen over 1601, 1602, 1603 — three depths NOT sharing 1600/2200's residue —
// so the failing step's mod-4 alignment and its call-count alignment come apart. For each depth,
// steps 0-4 land at positions (prefillLen+step); if the true trigger is "3rd decode call"
// regardless of position, the first-divergent STEP stays at 2 for all three. If it is really
// "nKeys ≡ 3 (mod 4)" (or any other position-linked residue), the first-divergent step moves
// depth-to-depth (since prefillLen+step's residue mod 4 shifts as prefillLen shifts by 1).
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_TEST_MODEL=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf go test -tags goinfer_testhooks ./metal/ -run TestAttentionFA_positionSweep -v -timeout 20m
func TestAttentionFA_positionSweep(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint twice per depth)")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.Getenv("GOINFER_TEST_MODEL")
	if path == "" {
		t.Skip("set GOINFER_TEST_MODEL to a real hd=128 GQA checkpoint (e.g. qwen2.5-coder-1.5b)")
	}
	const nSteps = 5

	runOne := func(prefillLen int, enableFA bool) (H int, logitsPerStep [][]float32) {
		t.Helper()
		if enableFA {
			t.Setenv("GOINFER_METAL_ATTN_FA", "1")
		} else {
			t.Setenv("GOINFER_METAL_ATTN_FA", "")
		}
		m, err := decoder.Load(path, decoder.Options{Quant: "int4"})
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		b := &metalBackend{}
		rf, ok, err := b.BuildResident(m)
		if err != nil || !ok {
			t.Fatalf("BuildResident: ok=%v err=%v", ok, err)
		}
		mr, ok := rf.(*metalResident)
		if !ok {
			t.Fatalf("expected *metalResident")
		}
		r := mr.r
		if r.decodeAttnFA != enableFA {
			t.Fatalf("decodeAttnFA=%v, want %v", r.decodeAttnFA, enableFA)
		}
		H = r.H
		rng := rand.New(rand.NewSource(7))
		embs := make([][]float32, prefillLen)
		for i := range embs {
			row := make([]float32, H)
			for j := range row {
				row[j] = float32(rng.NormFloat64()) * 0.05
			}
			embs[i] = row
		}
		if _, err := r.ForwardBatch(embs, 0); err != nil {
			t.Fatalf("ForwardBatch: %v", err)
		}
		for step := 0; step < nSteps; step++ {
			pos := prefillLen + step
			emb := make([]float32, H)
			for j := range emb {
				emb[j] = float32(rng.NormFloat64()) * 0.05
			}
			l := r.ForwardEmb(append([]float32(nil), emb...), pos)
			logitsPerStep = append(logitsPerStep, append([]float32(nil), l...))
		}
		b.Close()
		m.Close()
		return H, logitsPerStep
	}

	// Subtests, not a plain loop: each subtest is invoked as its OWN process
	// (`-run 'TestAttentionFA_positionSweep/1601'`), so darwin gets a real process-exit reclaim
	// point between depths instead of six back-to-back ~2GB loads inside one process. Measured
	// necessary on this machine: two back-to-back subtests inside one process both refused to
	// load ("this machine currently has ~4.0 GB of memory available" against the ~4.9 GB the
	// checkpoint needs), even though a single depth alone loads fine.
	for _, prefillLen := range []int{1601, 1602, 1603} {
		prefillLen := prefillLen
		t.Run(strconv.Itoa(prefillLen), func(t *testing.T) {
			_, shippedLogits := runOne(prefillLen, false)
			_, faLogits := runOne(prefillLen, true)

			firstBad := -1
			for step := 0; step < nSteps; step++ {
				pos := prefillLen + step
				lShipped, lFA := shippedLogits[step], faLogits[step]
				var dot, na, nb, maxabs float64
				for i := range lShipped {
					a, bb := float64(lShipped[i]), float64(lFA[i])
					dot += a * bb
					na += a * a
					nb += bb * bb
					if d := math.Abs(a - bb); d > maxabs {
						maxabs = d
					}
				}
				cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
				bad := cos < 0.9999 || maxabs > 1e-1
				t.Logf("prefillLen=%d step=%d pos=%d (pos mod 4 = %d): cosine=%.7f maxAbs=%.4e bad=%v",
					prefillLen, step, pos, pos%4, cos, maxabs, bad)
				if bad && firstBad < 0 {
					firstBad = step
				}
			}
			if firstBad < 0 {
				t.Logf("RESULT prefillLen=%d: no divergence in %d steps", prefillLen, nSteps)
			} else {
				t.Logf("RESULT prefillLen=%d: first divergence at step=%d pos=%d (pos mod 4=%d)",
					prefillLen, firstBad, prefillLen+firstBad, (prefillLen+firstBad)%4)
			}
			// No assertion: this is a diagnostic sweep, not a gate. The interpretation (does the
			// step index stay fixed, or does it track a position residue) belongs in the
			// write-up, not a pass/fail threshold here — CONFIRMED_CALL_COUNT vs
			// CONFIRMED_POSITION_RESIDUE is a judgment call over the whole table (all three
			// subtests' RESULT lines), not any one row.
		})
	}
}

// TestAttentionFA_ulpPerturbationControl tests the OTHER sharp alternative to "attention_fa has a
// state-carrying bug": that "two clean decode steps, then a stable wrong plateau" is simply what
// ANY non-bit-identical kernel eventually looks like once accumulated f32 error crosses a
// downstream int8 activation-quantization rounding boundary — a discrete, not gradual, jump, by
// construction of rounding. If that is right, deliberately perturbing the REFERENCE (shipped)
// kernel's own attention scale by a single float32 ULP — nothing to do with attention_fa at all —
// should reproduce the same qualitative signature: some early steps identical, then a stable
// divergence once the rounding boundary is crossed.
//
// Both arms here use the SAME (shipped) kernel; only r.uScale's bit pattern differs between them
// by exactly one ULP. If this control shows the same "clean, then stable jump" shape, the shape
// itself carries no information about a bug in attention_fa specifically — it would be expected
// under R2's OWN registered fidelity gate (attention_fa is deliberately not bit-identical), and R2
// can proceed to that gate directly rather than keep hunting for a mechanism. If the control stays
// clean throughout (or diverges gradually, not in a sudden plateau), the signature IS diagnostic
// and the search for a state-carrying mechanism in attention_fa specifically should continue.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_TEST_MODEL=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf go test -tags goinfer_testhooks ./metal/ -run TestAttentionFA_ulpPerturbationControl -v -timeout 10m
func TestAttentionFA_ulpPerturbationControl(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint twice)")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.Getenv("GOINFER_TEST_MODEL")
	if path == "" {
		t.Skip("set GOINFER_TEST_MODEL to a real hd=128 GQA checkpoint (e.g. qwen2.5-coder-1.5b)")
	}
	const prefillLen = 1600
	const nSteps = 5

	runOne := func(perturbScale bool) (H int, logitsPerStep [][]float32) {
		t.Helper()
		t.Setenv("GOINFER_METAL_ATTN_FA", "") // attention_fa OFF in BOTH arms -- shipped kernel only
		m, err := decoder.Load(path, decoder.Options{Quant: "int4"})
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		b := &metalBackend{}
		rf, ok, err := b.BuildResident(m)
		if err != nil || !ok {
			t.Fatalf("BuildResident: ok=%v err=%v", ok, err)
		}
		mr, ok := rf.(*metalResident)
		if !ok {
			t.Fatalf("expected *metalResident")
		}
		r := mr.r
		if perturbScale {
			orig := r.uScale.Floats()[0]
			nudged := math.Nextafter32(orig, float32(math.Inf(1)))
			if nudged == orig {
				t.Fatalf("ULP nudge produced no change (orig=%v) -- something is wrong with this probe", orig)
			}
			r.uScale.Floats()[0] = nudged
			t.Logf("perturbed uScale: %.9g -> %.9g (delta %.3e)", orig, nudged, float64(nudged-orig))
		}
		H = r.H
		rng := rand.New(rand.NewSource(7))
		embs := make([][]float32, prefillLen)
		for i := range embs {
			row := make([]float32, H)
			for j := range row {
				row[j] = float32(rng.NormFloat64()) * 0.05
			}
			embs[i] = row
		}
		if _, err := r.ForwardBatch(embs, 0); err != nil {
			t.Fatalf("ForwardBatch: %v", err)
		}
		for step := 0; step < nSteps; step++ {
			pos := prefillLen + step
			emb := make([]float32, H)
			for j := range emb {
				emb[j] = float32(rng.NormFloat64()) * 0.05
			}
			l := r.ForwardEmb(append([]float32(nil), emb...), pos)
			logitsPerStep = append(logitsPerStep, append([]float32(nil), l...))
		}
		b.Close()
		m.Close()
		return H, logitsPerStep
	}

	_, unperturbed := runOne(false)
	_, perturbed := runOne(true)

	firstBad := -1
	for step := 0; step < nSteps; step++ {
		pos := prefillLen + step
		a, bb := unperturbed[step], perturbed[step]
		var dot, na, nb, maxabs float64
		for i := range a {
			x, y := float64(a[i]), float64(bb[i])
			dot += x * y
			na += x * x
			nb += y * y
			if d := math.Abs(x - y); d > maxabs {
				maxabs = d
			}
		}
		cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		bad := cos < 0.9999 || maxabs > 1e-1
		t.Logf("step=%d pos=%d: cosine=%.7f maxAbs=%.4e bad=%v", step, pos, cos, maxabs, bad)
		if bad && firstBad < 0 {
			firstBad = step
		}
	}
	if firstBad < 0 {
		t.Logf("CONTROL RESULT: 1-ULP scale perturbation produced no meaningful divergence in %d steps -- "+
			"a 'clean then stable jump' shape is NOT simply what any tiny perturbation looks like on this "+
			"path; the attention_fa signature remains unexplained by this hypothesis", nSteps)
	} else {
		t.Logf("CONTROL RESULT: 1-ULP scale perturbation FIRST DIVERGES at step=%d -- compare this shape "+
			"(clean-steps-before-divergence, stable-or-not after) against attention_fa's own step=2 divergence", firstBad)
	}
	// No assertion: this is a diagnostic control, not a gate. Divergence here is not itself a
	// failure of anything -- a 1-ULP scale change is EXPECTED to eventually matter for some input;
	// the question is only what SHAPE that mattering takes, read from the log above.
}
