package constrain

import (
	"encoding/json"
	"math"
	"math/rand"
	"testing"
)

// bytesVocab builds a [][]byte from string tokens.
func bytesVocab(toks ...string) [][]byte {
	out := make([][]byte, len(toks))
	for i, s := range toks {
		out[i] = []byte(s)
	}
	return out
}

// TestMasker_emptySurfaceAndPaddedVocab gates M26: a non-EOS id with no surface
// bytes must be masked (TryBytes(nil) is vacuously true, so leaving it legal
// livelocks decoding), and a logits slice sized to the model's padded vocab (longer
// than the tokenizer table, as on Qwen/Gemma) must not panic and must mask every
// id past the table — across Process, MaskAt, and Commit.
func TestMasker_emptySurfaceAndPaddedVocab(t *testing.T) {
	// id 3 is a non-EOS control token with empty surface; id 5 is EOS (also empty).
	vocab := bytesVocab("{", "}", `"a"`, "", ":", "")
	const eos = 5
	m := NewMasker(JSON(), vocab, []int{eos})
	neg := func(l []float32, id int) bool { return math.IsInf(float64(l[id]), -1) }

	// Padded model vocab: logits longer than the tokenizer table.
	logits := make([]float32, len(vocab)+4)
	m.Process(nil, logits) // must not panic on out-of-table ids
	if !neg(logits, 3) {
		t.Error("empty-surface non-EOS token 3 must be masked (else livelock)")
	}
	for id := len(vocab); id < len(logits); id++ {
		if !neg(logits, id) {
			t.Errorf("padded id %d (past the tokenizer table) must be masked", id)
		}
	}

	// MaskAt over the same padded width: same guarantees, no panic.
	l2 := make([]float32, len(vocab)+4)
	m.MaskAt(m.GrammarClone(), l2)
	if !neg(l2, 3) {
		t.Error("MaskAt: empty-surface token 3 must be masked")
	}
	if !neg(l2, len(vocab)+2) {
		t.Error("MaskAt: padded id past the table must be masked")
	}

	// Commit / TokenBytes on an out-of-range id must not panic.
	m.Commit(len(vocab) + 2)
	if b := m.TokenBytes(len(vocab) + 2); b != nil {
		t.Errorf("TokenBytes past the table = %v, want nil", b)
	}
}

// TestMasker_masksToGrammar: at the start of a JSON document only value-starting
// tokens (and whitespace) survive the mask; structural tokens that can't begin a
// value, and the EOS token (document not complete), are −∞.
func TestMasker_masksToGrammar(t *testing.T) {
	//                0    1    2      3    4    5    6    7    8(eos)
	vocab := bytesVocab("{", "}", `"a"`, ":", "1", ",", " ", "x", "")
	const eos = 8
	m := NewMasker(JSON(), vocab, []int{eos})

	logits := make([]float32, len(vocab))
	m.Process(nil, logits) // no tokens generated yet → state jsValue

	allowed := map[int]bool{0: true, 2: true, 4: true, 6: true} // { "a" 1 space
	for id := range vocab {
		masked := math.IsInf(float64(logits[id]), -1)
		if allowed[id] && masked {
			t.Errorf("token %d (%q) masked, want allowed", id, vocab[id])
		}
		if !allowed[id] && !masked {
			t.Errorf("token %d (%q) allowed, want masked", id, vocab[id])
		}
	}
}

// TestMasker_eosGating: the EOS token is masked until the grammar CanEnd, then
// allowed. Drive "{}" and check EOS flips from masked to allowed at completion.
func TestMasker_eosGating(t *testing.T) {
	vocab := bytesVocab("{", "}", "") // 0 1 2(eos)
	const eos = 2
	m := NewMasker(JSON(), vocab, []int{eos})

	eosMasked := func(gen []int) bool {
		logits := make([]float32, len(vocab))
		m.Process(gen, logits)
		return math.IsInf(float64(logits[eos]), -1)
	}
	if !eosMasked(nil) {
		t.Error(`EOS allowed at start (empty doc is incomplete)`)
	}
	if !eosMasked([]int{0}) {
		t.Error(`EOS allowed after "{" (incomplete)`)
	}
	if eosMasked([]int{0, 1}) {
		t.Error(`EOS masked after "{}" (complete) — want allowed`)
	}
}

// TestMasker_stopWhenComplete: with StopWhenComplete, once the document is
// complete every non-EOS token is masked so the only legal move is to stop.
func TestMasker_stopWhenComplete(t *testing.T) {
	vocab := bytesVocab("{", "}", " ", "") // 0 1 2 3(eos)
	const eos = 3
	m := NewMasker(JSON(), vocab, []int{eos}).StopWhenComplete()

	logits := make([]float32, len(vocab))
	m.Process([]int{0, 1}, logits) // "{}" → complete
	// Only EOS survives; "{", "}", and even whitespace are masked.
	for id := range eos {
		if !math.IsInf(float64(logits[id]), -1) {
			t.Errorf("token %d (%q) allowed after complete doc, want masked", id, vocab[id])
		}
	}
	if math.IsInf(float64(logits[eos]), -1) {
		t.Error("EOS masked after complete doc, want allowed")
	}
}

// TestConstrainedDecode_alwaysValidJSON is the hard-invariant test: drive the
// masker with RANDOM logits over a synthetic JSON vocabulary and confirm that
// whatever it produces — whenever it stops at a CanEnd point — is valid JSON per
// encoding/json. No model needed; the guarantee is structural. If the grammar
// ever accepted an invalid sequence, json.Valid would catch it; if it ever
// dead-ended (no legal token while incomplete), the loop fails explicitly.
func TestConstrainedDecode_alwaysValidJSON(t *testing.T) {
	// A vocabulary rich enough to build nested JSON. (No \u escapes — surrogate
	// validation is the one spot a byte grammar and encoding/json can disagree.)
	vocab := bytesVocab(
		"{", "}", "[", "]", `"`, ":", ",", " ", "\n",
		"a", "b", "c", "ab", "key", "x", // string-body fragments
		"0", "1", "2", "9", "-", ".", "e", "12", "345",
		"true", "false", "null",
		`""`, `"a"`, `"ab"`, "123", "-4.5", // whole-token values
		"@", "}}", "qq", // traps the mask must exclude where illegal
	)
	eos := len(vocab)
	vocab = append(vocab, nil) // EOS id
	m := NewMasker(JSON(), vocab, []int{eos})

	const trials = 3000
	completed := 0
	for trial := range trials {
		m.Reset()
		rng := rand.New(rand.NewSource(int64(trial) + 1))
		var gen []int
		done := false
		for step := range 300 {
			logits := make([]float32, len(vocab))
			for i := range logits {
				logits[i] = float32(rng.NormFloat64())
			}
			m.Process(gen, logits)

			// Stop sometimes once the document is complete.
			if m.CanEnd() && rng.Float64() < 0.3 {
				done = true
				break
			}
			// Pick the highest-logit allowed non-EOS token (random exploration).
			best, bi := float32(math.Inf(-1)), -1
			for id, v := range logits {
				if id == eos {
					continue
				}
				if v > best {
					best, bi = v, id
				}
			}
			if bi < 0 { // only EOS is legal → must be a completion point
				if !m.CanEnd() {
					t.Fatalf("trial %d step %d: dead end (no legal token, not CanEnd); gen=%v", trial, step, gen)
				}
				done = true
				break
			}
			gen = append(gen, bi)
		}
		if !done {
			continue // hit the step cap mid-document; the prefix is valid, just unfinished
		}
		completed++
		var buf []byte
		for _, id := range gen {
			buf = append(buf, vocab[id]...)
		}
		if !json.Valid(buf) {
			t.Fatalf("trial %d: constrained output is NOT valid JSON: %q", trial, buf)
		}
	}
	if completed < trials/2 {
		t.Fatalf("only %d/%d trials reached a complete document — test not exercising completion", completed, trials)
	}
	t.Logf("%d/%d trials produced a complete, valid JSON document under random logits", completed, trials)
}

// M-27: StopWhenComplete truncated a top-level scalar at its first completion point.
//
// CanEnd is a MAY-end predicate — `1` is a complete integer document and `12` is a longer
// one — but StopWhenComplete read it as MUST-end and masked every non-EOS token there. So
// `response_format: {"type":"integer"}` could only ever return a SINGLE DIGIT, and
// `{"enum":[1,10,100]}` could only ever produce `1`. No test caught it because none drove a
// TOP-LEVEL scalar: every existing case is an object or array, whose completion point really
// does admit nothing but whitespace, so the bug is invisible there.
//
// This drives the real Masker and asks what it permits, rather than asserting a generated
// string — the defect is in the mask, and a sampler that happened to pick EOS would hide it.
func TestMasker_stopWhenComplete_scalarsMayStillExtend(t *testing.T) {
	// "1" "2" "0" " " eos — a digit continuation and a whitespace continuation, so the two
	// cases the rule must separate are both in the vocabulary.
	vocab := bytesVocab("1", "2", "0", " ", "")
	const eos = 4
	allowed := func(t *testing.T, g Grammar, prefix []int) map[string]bool {
		t.Helper()
		m := NewMasker(g, vocab, []int{eos}).StopWhenComplete()
		logits := make([]float32, len(vocab))
		m.Process(prefix, logits)
		out := map[string]bool{}
		for id, v := range logits {
			if !math.IsInf(float64(v), -1) {
				out[string(vocab[id])] = true
			}
		}
		return out
	}

	t.Run("integer", func(t *testing.T) {
		g, err := JSONSchema([]byte(`{"type":"integer"}`))
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		got := allowed(t, g, []int{0}) // committed "1" — a complete integer, and extensible
		if !got[""] {
			t.Error("EOS masked after \"1\" — a complete integer must be allowed to stop")
		}
		if !got["2"] {
			t.Error(`"2" masked after "1": {"type":"integer"} can only ever return one digit (M-27)`)
		}
		if got[" "] {
			t.Error(`whitespace allowed after a complete "1" — StopWhenComplete must still ` +
				`suppress trailing filler, or the fix has simply disabled it`)
		}
	})

	t.Run("enum with a shared prefix", func(t *testing.T) {
		// 1 is a member AND a prefix of 10 and 100: the completion point sits mid-literal.
		g, err := JSONSchema([]byte(`{"enum":[1,10,100]}`))
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		got := allowed(t, g, []int{0}) // committed "1"
		if !got[""] {
			t.Error(`EOS masked after "1" — 1 is itself a member of the enum`)
		}
		if !got["0"] {
			t.Error(`"0" masked after "1": the enum can only ever produce 1, never 10 or 100 (M-27)`)
		}
	})

	// THE OTHER DIRECTION, which matters as much: a fix that simply stopped masking would
	// pass everything above and silently delete StopWhenComplete. A completed OBJECT admits
	// nothing but whitespace, and that must still be masked down to EOS alone.
	t.Run("object still stops", func(t *testing.T) {
		ov := bytesVocab("{", "}", " ", "")
		const oeos = 3
		m := NewMasker(JSON(), ov, []int{oeos}).StopWhenComplete()
		logits := make([]float32, len(ov))
		m.Process([]int{0, 1}, logits) // "{}"
		for id := range oeos {
			if !math.IsInf(float64(logits[id]), -1) {
				t.Errorf("token %q allowed after a complete {} — StopWhenComplete is not stopping", ov[id])
			}
		}
		if math.IsInf(float64(logits[oeos]), -1) {
			t.Error("EOS masked after {}")
		}
	})
}
