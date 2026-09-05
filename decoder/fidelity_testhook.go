//go:build goinfer_testhooks

// Code added for docs/task-prefill-gap.md §3's fidelity gate (a backend's fast/batched prefill
// vs its own exact/sequential path) and docs/task-peer-benchmarks.md §4's fidelity column
// (goinfer vs a peer engine) -- both want the same teacher-forced top-1 agreement and KL
// divergence scorer, so it is written once here rather than twice. Test-only hook (B-08): these
// gate correctness/quality, not production inference, so they stay off the public API surface.

package decoder

import (
	"bufio"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

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

// PrefillGateProseFiles are real prose read at run time — not scripts/prompts.json's word-
// repetition filler, which docs/task-prefill-gap.md §0 rules out for anything content-dependent
// ("the fidelity gate (§3) uses prose"). Ten distinct real technical documents from this repo,
// chosen only for being real, sizeable (each encodes to well over 3900 tokens on its own, so no
// prompt needs repeating to reach the deepest K), and stable — not for their content, the same
// reasoning metal/spec_prefill_regression_test.go's readRepoCorpus gives for reading real
// repository source instead of a short hand-written corpus.
//
// Paths are relative to a package directory one level under the repo root (as metal/'s and
// decoder/'s own test packages both are), so the same list resolves identically from either —
// this is shared between metal/prefill_gate_test.go (Metal arms) and
// decoder/prefill_ref_gen_test.go (the CPU f32-activation reference, §3.1) precisely so the two
// runs score the SAME ten prompts.
var PrefillGateProseFiles = []string{
	"../docs/audit-2026-09-02.md",
	"../docs/QUEUE.md",
	"../docs/queue-engineering.md",
	"../docs/ollama-chase.md",
	"../docs/benchmarks.md",
	"../docs/parity-coverage-policy.md",
	"../docs/task-w4a8-neon-bandwidth.md",
	"../docs/legacy-benchmarks.md",
	"../docs/task-zeno-compare.md",
	"../docs/queue-release.md",
}

// PrefillGateProseIDsForTest reads f, encodes it with tk, and returns at least minTokens ids
// (repeating the same real content if one file is somehow too short for a future larger K, rather
// than padding with filler — a repeated real paragraph is still content-dependent, unlike
// scripts/prompts.json's "the the the").
func PrefillGateProseIDsForTest(t *testing.T, tk *tokenizer.Tokenizer, f string, minTokens int) []int {
	t.Helper()
	raw, err := os.ReadFile(f)
	if err != nil {
		t.Fatalf("read prose seed %s: %v", f, err)
	}
	text := string(raw)
	ids, err := tk.Encode(text, true)
	if err != nil {
		t.Fatalf("encode prose seed %s: %v", f, err)
	}
	for len(ids) < minTokens {
		more, err := tk.Encode(text, false)
		if err != nil {
			t.Fatalf("encode prose seed %s: %v", f, err)
		}
		ids = append(ids, more...)
	}
	return ids
}

// WritePrefillReferenceForTest serializes docs/task-prefill-gap.md §3.1's CPU f32-activation
// reference for a later cross-process read (ReadPrefillReferenceForTest) — Phase A
// (decoder/prefill_ref_gen_test.go, its own process, CPU only) writes these; Phase B
// (metal/prefill_gate_ref_test.go) reads them back to score both Metal arms against a reference
// neither of them is. Layout, all little-endian: int32 vocab, int32 continuationN, seedLogits
// [vocab]float32, refTokens [continuationN]int32, then continuationN rows of [vocab]float32
// (refLogits). This is a private, single-machine, single-session scratch format — not versioned
// or exported for reuse beyond this pair, which is why it carries no header/magic beyond its two
// size fields.
func WritePrefillReferenceForTest(path string, seedLogits []float32, refTokens []int, refLogits [][]float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	vocab := int32(len(seedLogits))
	n := int32(len(refTokens))
	if err := binary.Write(w, binary.LittleEndian, vocab); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, n); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, seedLogits); err != nil {
		return err
	}
	toks32 := make([]int32, len(refTokens))
	for i, v := range refTokens {
		toks32[i] = int32(v)
	}
	if err := binary.Write(w, binary.LittleEndian, toks32); err != nil {
		return err
	}
	for _, row := range refLogits {
		if int32(len(row)) != vocab {
			return os.ErrInvalid
		}
		if err := binary.Write(w, binary.LittleEndian, row); err != nil {
			return err
		}
	}
	return w.Flush()
}

// ReadPrefillReferenceForTest is WritePrefillReferenceForTest's reader. See that function for the
// layout.
func ReadPrefillReferenceForTest(path string) (seedLogits []float32, refTokens []int, refLogits [][]float32, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	var vocab, n int32
	if err = binary.Read(r, binary.LittleEndian, &vocab); err != nil {
		return nil, nil, nil, err
	}
	if err = binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, nil, nil, err
	}
	seedLogits = make([]float32, vocab)
	if err = binary.Read(r, binary.LittleEndian, seedLogits); err != nil {
		return nil, nil, nil, err
	}
	toks32 := make([]int32, n)
	if err = binary.Read(r, binary.LittleEndian, toks32); err != nil {
		return nil, nil, nil, err
	}
	refTokens = make([]int, n)
	for i, v := range toks32 {
		refTokens[i] = int(v)
	}
	refLogits = make([][]float32, n)
	for i := range refLogits {
		row := make([]float32, vocab)
		if err = binary.Read(r, binary.LittleEndian, row); err != nil {
			return nil, nil, nil, err
		}
		refLogits[i] = row
	}
	return seedLogits, refTokens, refLogits, nil
}
