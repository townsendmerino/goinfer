package decoder

import (
	"context"
	"sort"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestNgramAlphaPredictor is the §06 gating measurement: is the n-gram source's
// per-position acceptance PREDICTABLE from its only draft-time feature, the suffix
// match length? It runs n-gram-speculative decode over the copy-heavy workloads with
// the trace collector, buckets the per-position SpecTraces by match_len, and reports
// the empirical accept_prob per bucket (the calibration curve = α̂_ngram(match_len))
// plus the AUC of match_len predicting the realized accept. The doc's rule: establish
// the minimal predictor's AUC before shipping anything heavier. If match_len carries
// the signal, the runtime needs only this tiny 1-D table — no offline ML, no Python.
// Run: GINFER_PREQUANT_GGUF=... go test ./decoder -run TestNgramAlphaPredictor -v
func TestNgramAlphaPredictor(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
	}
	tk, err := tokenizer.LoadGGUF(benchGGUFPath())
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}
	const maxTok, K = 96, 8

	col := NewTraceCollector(nil)
	for _, w := range specWorkloads {
		prompt, err := tk.Encode(w.prompt, true)
		if err != nil {
			t.Fatalf("encode %s: %v", w.name, err)
		}
		ch, _, err := m.genNgram(ctx, prompt, maxTok, &NgramDrafter{}, K, greedy, col.Record, nil)
		if err != nil {
			t.Fatalf("genNgram %s: %v", w.name, err)
		}
		for range ch {
		}
	}
	rows := col.Rows
	if len(rows) < 30 {
		t.Skipf("only %d traced positions — too few for a stable curve", len(rows))
	}

	// Calibration curve: bucket by match_len, mean accept_prob (the exact per-position
	// acceptance, lower variance) + realized accept rate + count.
	type bucket struct {
		n              int
		sumAcc, sumHit float64
	}
	buckets := map[int]*bucket{}
	for _, r := range rows {
		b := buckets[r.NgramMatch]
		if b == nil {
			b = &bucket{}
			buckets[r.NgramMatch] = b
		}
		b.n++
		b.sumAcc += r.AcceptProb
		if r.Accepted {
			b.sumHit++
		}
	}
	keys := make([]int, 0, len(buckets))
	for k := range buckets {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	t.Logf("α̂_ngram(match_len) calibration over %d positions:", len(rows))
	t.Logf("%8s %6s %12s %12s", "matchLen", "n", "accept_prob", "realized")
	for _, k := range keys {
		b := buckets[k]
		t.Logf("%8d %6d %12.3f %12.3f", k, b.n, b.sumAcc/float64(b.n), b.sumHit/float64(b.n))
	}

	// AUC of match_len predicting the realized accept (Mann–Whitney): P(matchLen of a
	// random accepted position > that of a random rejected one). 0.5 = no signal.
	auc, nPos, nNeg := aucByFeature(rows)
	t.Logf("match_len → accepted: AUC=%.3f  (accepted=%d rejected=%d)", auc, nPos, nNeg)
	t.Logf("mean accept_prob (α̅) = %.3f", col.MeanAccept())

	// §06 verdict: match_len must carry real signal for the calibrated α̂_ngram to be
	// worth consulting (else the router should just use the base rate). Loose bound —
	// the measured value is ~0.82; guard against a regression to noise.
	if nNeg >= 5 && auc < 0.65 {
		t.Errorf("match_len no longer predicts acceptance (AUC %.3f < 0.65) — re-fit ngramAlpha or drop the predictor", auc)
	}
}

// TestNgramAlphaTable gates the calibrated α̂_ngram lookup (no model needed): it must be
// a valid probability (∈[0,1]) and monotone non-decreasing in match_len — longer copies
// are never less reliable. These are the structural invariants the §06 fit must keep.
func TestNgramAlphaTable(t *testing.T) {
	prev := -1.0
	for ml := 0; ml <= 20; ml++ {
		a := ngramAlpha(ml)
		if a < 0 || a > 1 {
			t.Fatalf("ngramAlpha(%d)=%g out of [0,1]", ml, a)
		}
		if a < prev-1e-9 {
			t.Fatalf("ngramAlpha not monotone: α(%d)=%g < α(prev)=%g", ml, a, prev)
		}
		prev = a
	}
	// A long verbatim copy must beat grammar's forced proposal; a short (len-2) one must
	// not — the calibration's whole point (the old raw-match-len heuristic got len-2
	// wrong by ranking it under a fixed grammar constant on an incomparable scale).
	if ngramAlpha(16) <= grammarConf {
		t.Errorf("α̂_ngram(16)=%.3f should exceed α̂_grammar=%.3f (long copy wins)", ngramAlpha(16), grammarConf)
	}
	if ngramAlpha(2) >= grammarConf {
		t.Errorf("α̂_ngram(2)=%.3f should fall below α̂_grammar=%.3f (short copy yields)", ngramAlpha(2), grammarConf)
	}
}

// aucByFeature computes the rank-AUC of NgramMatch predicting Accepted.
func aucByFeature(rows []SpecTrace) (auc float64, nPos, nNeg int) {
	type pt struct {
		x   float64
		pos bool
	}
	pts := make([]pt, len(rows))
	for i, r := range rows {
		pts[i] = pt{x: float64(r.NgramMatch), pos: r.Accepted}
		if r.Accepted {
			nPos++
		} else {
			nNeg++
		}
	}
	if nPos == 0 || nNeg == 0 {
		return 0.5, nPos, nNeg
	}
	// rank-sum: average ranks (ties → mid-rank), AUC = (R_pos - nPos(nPos+1)/2) / (nPos·nNeg)
	sort.Slice(pts, func(i, j int) bool { return pts[i].x < pts[j].x })
	ranks := make([]float64, len(pts))
	for i := 0; i < len(pts); {
		j := i
		for j < len(pts) && pts[j].x == pts[i].x {
			j++
		}
		avg := float64(i+j+1) / 2 // mid-rank (1-based) for ties [i, j)
		for k := i; k < j; k++ {
			ranks[k] = avg
		}
		i = j
	}
	var rPos float64
	for i, p := range pts {
		if p.pos {
			rPos += ranks[i]
		}
	}
	auc = (rPos - float64(nPos)*float64(nPos+1)/2) / (float64(nPos) * float64(nNeg))
	return auc, nPos, nNeg
}
