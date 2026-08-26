package decoder

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestPerRowScalePhase0b is §7's OUTPUT-quality gate — the follow-up Phase 0 (weight-space, 1.24×
// rel-error for per-row symmse) opened: does that 1.24× translate to real forward degradation, or is
// it benign like nemotron int4? It teacher-forces a real prose+code sample through three builds of the
// SAME model and compares each to the f32 oracle: (1) f32 (oracle), (2) per-group sym int4 (≈ shipped),
// (3) per-row symmse int4 (the §7 fork, no rotation). Metrics per build vs oracle: top-1 agreement,
// mean KL(oracle‖build), and perplexity. The decision: if per-row's degradation over the oracle is
// close to per-group's, the fork is output-benign and the tensor-core/decode-stream payoff (§7) is
// reachable with a scale search + parity refresh — no rotation. If per-row is markedly worse, rotation
// (last-resort) is back on the table. No kernel; CPU forwards via forwardN.
//
//	GOINFER_PERROW_PHASE0B=1 go test -run TestPerRowScalePhase0b -v -timeout 30m ./decoder/
func TestPerRowScalePhase0b(t *testing.T) {
	if os.Getenv("GOINFER_PERROW_PHASE0B") == "" {
		t.Skip("§7 Phase 0b (loads a 1.7B model three ways); set GOINFER_PERROW_PHASE0B=1")
	}
	// Safetensors (f32/bf16) source — the GGUF loader quantizes rows directly and BYPASSES
	// quantizeWM/fakeInt4WM (gguf.go buildWeightsFromGGUF), so the fakequant knob only fires on the
	// safetensors path. Same model as Phase 0's weight-space measurement, for an apples-to-apples read.
	path := os.ExpandEnv("$HOME/models/qwen3-1.7b-bf16")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	tk, err := tokenizer.Load(path)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	// A real prose+code sample so the measurement runs on the outlier-bearing distribution (the 5th
	// axis), not a low-entropy repeat that would trivially inflate agreement.
	sample := "The transformer architecture processes tokens in parallel using self-attention. " +
		"Each layer refines the representation. In Go, a slice header is a pointer, length, and capacity: " +
		"func sum(xs []float64) float64 { total := 0.0; for _, x := range xs { total += x }; return total }. " +
		"Quantization trades numerical precision for memory bandwidth, which on modern GPUs is the binding constraint for decode."
	ids, _ := tk.Encode(sample, true)
	if len(ids) < 32 {
		t.Skipf("sample too short (%d ids)", len(ids))
	}
	N := len(ids) - 1 // teacher-forced positions (predict ids[i+1] from position i)
	t.Logf("§7 Phase 0b — teacher-forced quality on %d tokens (%s)", len(ids), path)

	// perPositionDist runs the model over ids and returns, per position, the softmax dist and argmax,
	// plus the teacher-forced NLL against ids[i+1]. Loads the model with the current fakequant vars.
	type run struct {
		dist [][]float64
		arg  []int
		nll  float64
	}
	capture := func(quant string) (*run, bool) {
		opts := Options{Backend: "cpu"}
		if quant != "" {
			opts.Quant = quant
		}
		m, e := Load(path, opts)
		if e != nil {
			t.Fatalf("load(%q): %v", quant, e)
		}
		defer m.Close()
		cache := m.NewCache(len(ids) + 4)
		logits, e := m.forwardN(context.Background(), ids, cache)
		if e != nil {
			t.Skipf("forwardN not supported here: %v", e)
			return nil, false
		}
		r := &run{dist: make([][]float64, N), arg: make([]int, N)}
		for i := range N {
			d := softmax64(logits[i])
			r.dist[i] = d
			r.arg[i] = argmax64(d)
			p := d[ids[i+1]]
			r.nll += -math.Log(math.Max(p, 1e-12))
		}
		return r, true
	}

	// (1) f32 oracle — fakequant off.
	fakeQuantScheme, fakeQuantPerRow = "", false
	oracle, ok := capture("")
	if !ok {
		return
	}
	t.Logf("%-22s perplexity=%.3f", "f32 oracle", math.Exp(oracle.nll/float64(N)))

	compare := func(name string, r *run) (agreePct, meanKL, ppl float64) {
		agree, klSum := 0, 0.0
		for i := range N {
			if r.arg[i] == oracle.arg[i] {
				agree++
			}
			for v, p := range oracle.dist[i] {
				if p > 1e-9 {
					klSum += p * (math.Log(p) - math.Log(math.Max(r.dist[i][v], 1e-12)))
				}
			}
		}
		agreePct, meanKL, ppl = 100*float64(agree)/float64(N), klSum/float64(N), math.Exp(r.nll/float64(N))
		t.Logf("%-22s agreement=%.1f%%  meanKL(f32‖x)=%.4f  perplexity=%.3f (oracle %.3f)",
			name, agreePct, meanKL, ppl, math.Exp(oracle.nll/float64(N)))
		return
	}

	// (2) per-group sym int4 — the shipped-equivalent control.
	fakeQuantScheme, fakeQuantPerRow = "sym", false
	pg, okG := capture("int4")
	fakeQuantScheme, fakeQuantPerRow = "", false // reset before any early return
	var pgKL, prKL float64
	if okG {
		_, pgKL, _ = compare("per-group sym (shipped)", pg)
	}

	// (3) per-row symmse int4 — the §7 fork (no rotation).
	fakeQuantScheme, fakeQuantPerRow = "symmse", true
	pr, okR := capture("int4")
	fakeQuantScheme, fakeQuantPerRow = "", false
	if okR {
		_, prKL, _ = compare("per-row symmse (§7 fork)", pr)
	}

	// Sanity: the two configs MUST differ. Bit-identical means the fakequant knob didn't switch
	// (e.g. a loader that bypasses quantizeWM) — which would masquerade as a false "benign" verdict.
	if okG && okR && pgKL == prKL {
		t.Fatalf("per-group and per-row KL are bit-identical (%.6f) — the fakequant config did not switch; "+
			"the load path is bypassing fakeInt4WM. Result is INVALID.", pgKL)
	}
	t.Logf("VERDICT: read the two rows above — if per-row agreement/KL/ppl ≈ per-group, the fork is")
	t.Logf("  output-benign (scale search + parity refresh, NO rotation). If markedly worse, rotation is back.")
}

func softmax64(logits []float32) []float64 {
	mx := float32(math.Inf(-1))
	for _, v := range logits {
		if v > mx {
			mx = v
		}
	}
	out := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v - mx))
		out[i] = e
		sum += e
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func argmax64(d []float64) int {
	best, bi := math.Inf(-1), 0
	for i, v := range d {
		if v > best {
			best, bi = v, i
		}
	}
	return bi
}
