//go:build goinfer_testhooks

// Code added for docs/task-prefill-gap.md §3's fidelity gate (a backend's fast/batched prefill
// vs its own exact/sequential path) and docs/task-peer-benchmarks.md §4's fidelity column
// (goinfer vs a peer engine) -- both want the same teacher-forced top-1 agreement and KL
// divergence scorer, so it is written once here rather than twice. Test-only hook (B-08): these
// gate correctness/quality, not production inference, so they stay off the public API surface.

package decoder

import "math"

// NearTieHardFailPct is the bar NearTieArgmaxForTest hard-fails at -- the same 3% every existing
// near-tie gate in this tree already uses inline (cuda/realforward_test.go's argmaxF comparison,
// gpu/kv_i8_parity_test.go), named here so a new gate cites the rule instead of retyping the
// literal.
const NearTieHardFailPct = 0.03

// NearTieArgmaxForTest reproduces the 3%-near-tie rule cuda/realforward_test.go's argmaxF
// comparison established: comparing two logit vectors' argmax, a flip is a defect only if the
// REFERENCE's own margin between its pick and the candidate's pick exceeds NearTieHardFailPct of
// the reference's logit range -- smaller gaps are quant/reassociation noise, not a real
// preference change. gapPct is always computed (0 when they agree), so a caller can report the
// worst gap seen across a run even on ticks that don't hard-fail.
func NearTieArgmaxForTest(refLogits, candLogits []float32) (agree bool, gapPct float64, hardFail bool) {
	refArg, candArg := argmax(refLogits), argmax(candLogits)
	if refArg == candArg {
		return true, 0, false
	}
	lo, hi := refLogits[0], refLogits[0]
	for _, v := range refLogits {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	gap := float64(refLogits[refArg]-refLogits[candArg]) / (float64(hi-lo) + 1e-9)
	return false, gap, gap > NearTieHardFailPct
}

// TeacherForcedTop1AgreementForTest measures how faithfully an engine reproduces a reference
// continuation WITHOUT the cascade a free-running greedy comparison carries, where one early
// near-tie flip makes every later token diverge and the score collapses to "how long before the
// first flip" instead of "how good is the engine at each position on its own". candLogits[i] is
// the engine's output at continuation position i when fed the reference's own tokens as context
// through position i-1 (teacher-forced, not autoregressive on the engine's own output);
// refTokens[i] is the token the reference continuation actually placed at position i. Reports
// the fraction of positions where the engine's argmax equals the reference token, and the first
// position that disagrees (-1 if none). Returns 0, -1 if the slices are empty or mismatched in
// length -- a caller error, not a measurement of zero agreement.
func TeacherForcedTop1AgreementForTest(candLogits [][]float32, refTokens []int) (agreementRate float64, firstDivergence int) {
	firstDivergence = -1
	n := len(candLogits)
	if n == 0 || n != len(refTokens) {
		return 0, firstDivergence
	}
	agree := 0
	for i, lg := range candLogits {
		if argmax(lg) == refTokens[i] {
			agree++
		} else if firstDivergence == -1 {
			firstDivergence = i
		}
	}
	return float64(agree) / float64(n), firstDivergence
}

// KLDivergenceForTest computes KL(p || q) in nats between two logit vectors, after converting
// each to a probability distribution the same way sampling does (softmaxStable, temperature 1).
// p is the reference/exact distribution and q the candidate/approximate one, so the result reads
// as "how much information is lost approximating p with q" -- the §3 gate's "reported, not
// gating" KL-vs-exact figure. Terms where p is ~0 are skipped rather than evaluated: the limit of
// p*log(p/q) as p->0 is 0 regardless of q, and evaluating it risks NaN from log(0).
func KLDivergenceForTest(pLogits, qLogits []float32) float64 {
	p := softmaxStable(pLogits, 1)
	q := softmaxStable(qLogits, 1)
	var kl float64
	for i, pi := range p {
		if pi < 1e-12 {
			continue
		}
		kl += pi * math.Log(pi/(q[i]+1e-300))
	}
	return kl
}
