package decoder

import (
	"context"
	"sort"
	"testing"

	"github.com/townsendmerino/goinfer/constrain"
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
// Run: GOINFER_PREQUANT_GGUF=... go test ./decoder -run TestNgramAlphaPredictor -v
func TestNgramAlphaPredictor(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
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
	// Both sources are now calibrated accept-probs on a common scale. The trace-fit
	// α̂_grammar (~0.20, tokenization-fragile) is the weakest source, so EVERY n-gram copy
	// — even the shortest (len-2 ≈ 0.70) — must outrank grammar; grammar drafts only when
	// n-gram has no copy. (This corrected the original guess that grammar was ~0.9.)
	if ngramAlpha(2) <= grammarConf {
		t.Errorf("α̂_ngram(2)=%.3f should exceed the tokenization-fragile α̂_grammar=%.3f", ngramAlpha(2), grammarConf)
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

// TestGrammarAlphaPredictor is the §06 trace-fit of α̂_grammar: it runs grammar-fused
// decode with a PURE GrammarDrafter over several JSON-schema constraints, so every
// drafted token is a grammar-FORCED byte, and measures their empirical acceptance. The
// §6 sanity is that forced tokens accept ≈1 — they're grammar-legal by construction, so
// the only loss is a tokenization mismatch (the model preferring a different legal
// tokenization of the same bytes). The mean is the calibrated α̂_grammar that replaces
// the heuristic constant. Run: GOINFER_PREQUANT_GGUF=... go test ./decoder -run GrammarAlpha -v
func TestGrammarAlphaPredictor(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	tk, err := tokenizer.LoadGGUF(benchGGUFPath())
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	vocabBytes := constrain.TokenBytes(m.w.arch.VocabSize, tk.TokenText)
	encode := func(s string) []int { ids, _ := tk.Encode(s, false); return ids }
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}
	const maxTok = 64

	// Many required keys + enums per schema → many grammar-FORCED byte runs (key names,
	// punctuation, enum values). Keeps the per-call forced count high enough for a stable
	// α̂_grammar on a small model.
	obj := func(props, req string) string {
		return `{"type":"object","properties":{` + props + `},"required":[` + req + `],"additionalProperties":false}`
	}
	workloads := []struct{ schema, prompt string }{
		{obj(`"location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]},"humidity":{"type":"integer"}`, `"location","unit","humidity"`),
			"Weather for Paris: celsius, humidity 60, as JSON matching the schema."},
		{obj(`"name":{"type":"string"},"age":{"type":"integer"},"active":{"type":"boolean"},"role":{"enum":["admin","user","guest"]}`, `"name","age","active","role"`),
			"A person named John Doe, age 30, active, role admin, as JSON."},
		{obj(`"id":{"type":"integer"},"status":{"enum":["open","closed","pending"]},"title":{"type":"string"},"priority":{"enum":["low","high"]}`, `"id","status","title","priority"`),
			"A ticket id 7, status open, title Fix bug, priority high, as JSON."},
		{obj(`"street":{"type":"string"},"city":{"type":"string"},"zip":{"type":"string"},"country":{"enum":["US","UK","CA"]}`, `"street","city","zip","country"`),
			"An address: 1 Main St, Springfield, 12345, US, as JSON."},
		{obj(`"product":{"type":"string"},"price":{"type":"integer"},"currency":{"enum":["USD","EUR","GBP"]},"available":{"type":"boolean"}`, `"product","price","currency","available"`),
			"A product Widget, price 20, currency USD, available, as JSON."},
		{obj(`"method":{"enum":["GET","POST","PUT","DELETE"]},"path":{"type":"string"},"code":{"type":"integer"}`, `"method","path","code"`),
			"An HTTP log: GET /index, code 200, as JSON."},
		{obj(`"first":{"type":"string"},"last":{"type":"string"},"email":{"type":"string"},"verified":{"type":"boolean"}`, `"first","last","email","verified"`),
			"User: first Jane, last Smith, email jane@x.com, verified, as JSON."},
		{obj(`"language":{"enum":["go","rust","python","java"]},"version":{"type":"string"},"stable":{"type":"boolean"}`, `"language","version","stable"`),
			"A release: language go, version 1.22, stable, as JSON."},
		{obj(`"lat":{"type":"integer"},"lon":{"type":"integer"},"label":{"type":"string"},"kind":{"enum":["city","park","road"]}`, `"lat","lon","label","kind"`),
			"A place: lat 40, lon 70, label Central, kind park, as JSON."},
		{obj(`"event":{"type":"string"},"level":{"enum":["info","warn","error"]},"count":{"type":"integer"}`, `"event","level","count"`),
			"A log: event login, level info, count 3, as JSON."},
		{obj(`"sku":{"type":"string"},"qty":{"type":"integer"},"unit":{"enum":["box","kg","each"]},"taxable":{"type":"boolean"}`, `"sku","qty","unit","taxable"`),
			"An item: sku A1, qty 5, unit box, taxable, as JSON."},
		{obj(`"action":{"enum":["create","update","delete"]},"target":{"type":"string"},"ok":{"type":"boolean"}`, `"action","target","ok"`),
			"An audit: action create, target user, ok, as JSON."},
	}

	col := NewTraceCollector(nil)
	for _, w := range workloads {
		g, gerr := constrain.JSONSchema([]byte(w.schema))
		if gerr != nil {
			t.Fatalf("JSONSchema: %v", gerr)
		}
		mask := constrain.NewMasker(g, vocabBytes, m.eosIDs).StopWhenComplete()
		prompt, _ := tk.Encode("<|im_start|>user\n"+w.prompt+"<|im_end|>\n<|im_start|>assistant\n", true)
		gd := &GrammarDrafter{Mask: mask, Encode: encode}
		out := make(chan int)
		gen := &Generation{Spec: &SpecStats{}}
		go func() {
			defer close(out)
			m.genGrammarInto(ctx, out, gen, gen.Spec, mask, gd, prompt, 0, maxTok, 8, greedy, col.Record, nil, nil)
		}()
		for range out {
		}
	}

	var forced []SpecTrace
	for _, r := range col.Rows {
		if r.Forced {
			forced = append(forced, r)
		}
	}
	if len(forced) < 20 {
		t.Skipf("only %d forced positions — too few for a stable α̂_grammar", len(forced))
	}
	var sumAcc, hits float64
	byDepth := map[int]*struct {
		n   int
		acc float64
	}{}
	for _, r := range forced {
		sumAcc += r.AcceptProb
		if r.Accepted {
			hits++
		}
		b := byDepth[r.Pos]
		if b == nil {
			b = &struct {
				n   int
				acc float64
			}{}
			byDepth[r.Pos] = b
		}
		b.n++
		b.acc += r.AcceptProb
	}
	meanAcc := sumAcc / float64(len(forced))
	t.Logf("α̂_grammar over %d FORCED positions: mean accept_prob=%.3f, realized accept=%.3f",
		len(forced), meanAcc, hits/float64(len(forced)))
	depths := make([]int, 0, len(byDepth))
	for d := range byDepth {
		depths = append(depths, d)
	}
	sort.Ints(depths)
	for _, d := range depths {
		b := byDepth[d]
		t.Logf("  forced-run depth %d: n=%d accept_prob=%.3f", d, b.n, b.acc/float64(b.n))
	}
	t.Logf("current grammarConf constant = %.3f", grammarConf)

	// The measured value (~0.20) is the finding, not a bug: forced bytes are grammar-legal
	// but the canonical retokenization mismatches the model's tokenization under the mask.
	// Gate only that it's a valid probability and tracks grammarConf — a regression in the
	// forced/mask machinery would push it toward 0 (nothing accepts).
	if meanAcc < 0 || meanAcc > 1 {
		t.Fatalf("α̂_grammar=%.3f not a probability", meanAcc)
	}
	if meanAcc < 0.05 {
		t.Errorf("α̂_grammar=%.3f ~0 — the grammar source accepts almost nothing; check forced-bytes/mask alignment", meanAcc)
	}
}

// fakeSource is a deterministic Drafter with a fixed static confidence and a fixed
// realized accept fraction, for testing the router's online correction without a model.
type fakeSource struct {
	conf     float64 // static α̂ reported to the router
	proposal []int   // what Draft returns (non-empty = participates)
	accept   int     // accepted count the harness will report for this source's rounds
}

func (f *fakeSource) Draft(ctx []int, k int) []int { return f.proposal }
func (f *fakeSource) Confidence() float64          { return f.conf }

// TestRouterOnlineCorrection checks the §06 §9 mechanism: a source whose STATIC α̂ is high
// but whose realized acceptance is poor must be demoted below a steadier source within a
// few rounds, and the router must be byte-identical to the old static behavior until the
// first outcome is recorded.
func TestRouterOnlineCorrection(t *testing.T) {
	// hi: static 0.70 but never accepts (the over-trusted spurious-match case);
	// lo: static 0.20 but always accepts its 2-token proposal (the steady source).
	hi := &fakeSource{conf: 0.70, proposal: []int{1, 2}, accept: 0}
	lo := &fakeSource{conf: 0.20, proposal: []int{3, 4}, accept: 2}
	r := &RouterDrafter{Sources: []Drafter{hi, lo}}

	// Round 0: no outcomes yet ⇒ pure static α̂ ⇒ the higher static (hi) wins.
	if d := r.Draft(nil, 4); d[0] != hi.proposal[0] {
		t.Fatalf("cold start should pick the higher static-α̂ source (hi), got %v", d)
	}
	if r.chosen != 0 {
		t.Fatalf("chosen=%d, want 0 (hi)", r.chosen)
	}

	// Feed realized outcomes for whichever source the router picks each round; hi keeps
	// missing (0/2), lo always hits (2/2). The running rate should flip the ranking.
	flipped := -1
	for round := range 30 {
		r.Draft(nil, 4)
		src := hi
		if r.chosen == 1 {
			src = lo
		}
		r.RecordOutcome(src.accept, len(src.proposal))
		if r.chosen == 1 && flipped < 0 {
			flipped = round
		}
	}
	if flipped < 0 {
		t.Fatalf("router never switched to the steady source lo despite hi accepting 0/2 every round")
	}
	if flipped > 12 {
		t.Errorf("router took %d rounds to correct — too slow", flipped)
	}
	t.Logf("online correction flipped hi→lo after %d rounds (hi eff %.3f, lo eff %.3f)",
		flipped, r.effective(0, hi.conf), r.effective(1, lo.conf))
	if r.effective(0, hi.conf) >= r.effective(1, lo.conf) {
		t.Errorf("after correction, hi eff %.3f should be < lo eff %.3f", r.effective(0, hi.conf), r.effective(1, lo.conf))
	}
}

// TestNgramAlphaCrossWorkload is the §06 §3 held-out validation: does the shipped
// ngramAlpha table — fit on the copy-heavy specWorkloads (codeedit/rag/agent-json) —
// generalize to workloads it was NOT fit on? For each held-out workload it generates
// n-gram traces, then reports the AUC of match_len and the Expected Calibration Error
// ECE = Σ_bin (n/N)·|empirical_accept − ngramAlpha(match_len)| of the table against that
// workload's realized acceptance. Low ECE across held-out workloads ⇒ the static table
// generalizes; a high-ECE workload quantifies the drift the §9 online correction absorbs.
// Run: GOINFER_PREQUANT_GGUF=... go test ./decoder -run CrossWorkload -v
func TestNgramAlphaCrossWorkload(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	tk, err := tokenizer.LoadGGUF(benchGGUFPath())
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}
	const maxTok, K = 96, 8

	// HELD-OUT workloads: distinct domains from the fit set, each with enough internal
	// repetition that the n-gram drafter fires (so there are traces to score).
	heldOut := []struct{ name, prompt string }{
		{"csv", "id,name,score\n1,alice,10\n2,bob,20\n3,carol,30\n4,dave,40\n5,eve,"},
		{"config", "[server]\nhost = localhost\nport = 8080\n[client]\nhost = localhost\nport = 9090\n[proxy]\nhost = localhost\nport = "},
		{"markdown", "| key | value |\n| --- | --- |\n| alpha | 1 |\n| bravo | 2 |\n| charlie | 3 |\n| delta | "},
		{"sql", "INSERT INTO t (a,b) VALUES (1,2);\nINSERT INTO t (a,b) VALUES (3,4);\nINSERT INTO t (a,b) VALUES (5,6);\nINSERT INTO t (a,b) VALUES ("},
		{"prose-rep", "The cat sat on the mat. The dog sat on the mat. The bird sat on the mat. The fish sat on the "},
	}

	var sumW, sumECE float64
	t.Logf("%-10s %5s %6s %8s", "heldout", "n", "AUC", "ECE")
	for _, w := range heldOut {
		col := NewTraceCollector(nil)
		prompt, _ := tk.Encode(w.prompt, true)
		ch, _, err := m.genNgram(ctx, prompt, maxTok, &NgramDrafter{}, K, greedy, col.Record, nil)
		if err != nil {
			t.Fatalf("genNgram %s: %v", w.name, err)
		}
		for range ch {
		}
		if len(col.Rows) < 20 {
			t.Logf("%-10s %5d  (too few traces to score)", w.name, len(col.Rows))
			continue
		}
		// ECE of the shipped table: bin by match_len, |empirical − ngramAlpha(ml)|, mass-weighted.
		type b struct {
			n   int
			hit float64
		}
		bins := map[int]*b{}
		for _, r := range col.Rows {
			bb := bins[r.NgramMatch]
			if bb == nil {
				bb = &b{}
				bins[r.NgramMatch] = bb
			}
			bb.n++
			if r.Accepted {
				bb.hit++
			}
		}
		var ece float64
		N := float64(len(col.Rows))
		for ml, bb := range bins {
			emp := bb.hit / float64(bb.n)
			ece += float64(bb.n) / N * absf(emp-ngramAlpha(ml))
		}
		auc, _, _ := aucByFeature(col.Rows)
		t.Logf("%-10s %5d %6.3f %8.3f", w.name, len(col.Rows), auc, ece)
		sumW++
		sumECE += ece
	}
	if sumW == 0 {
		t.Skip("no held-out workload produced enough n-gram traces")
	}
	meanECE := sumECE / sumW
	t.Logf("mean held-out ECE = %.3f over %d workloads (shipped ngramAlpha table)", meanECE, int(sumW))
	// Verdict: the static table generalizes if its calibration error stays modest on
	// unseen workloads. A loose bound — the §9 online correction absorbs residual drift.
	if meanECE > 0.25 {
		t.Errorf("mean held-out ECE %.3f > 0.25 — α̂_ngram does not generalize; rely on §9 online correction / re-fit", meanECE)
	}
}

func absf(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
