package constrain

// The plain-string fast path.
//
// Masking is O(V) grammar walks per decode step — 151,936 of them on Qwen2.5 — and the
// measured cost is dominated by TryBytes's snapshot/restore of the frame stack rather than by
// the byte walk itself (audit P-20, measured in G37: 6.03 ms/step at `fsStr`, and the cost
// tracks stack DEPTH, not token length).
//
// But inside a JSON string, both grammars answer with the same three-line rule:
//
//	'"'      → closes the string   (legal, changes state)
//	'\\'     → escape              (legal, changes state)
//	b < 0x20 → illegal
//	otherwise → legal, AND THE STATE DOES NOT MOVE
//
// So for any token containing none of those three byte classes, legality inside a string is a
// property of the TOKEN ALONE, not of the grammar — precomputable once per vocabulary and
// answerable with one bit test. Measured on Qwen2.5's vocab: 96.88% of ids qualify (1,226 ids,
// 0.81%, contain '"' or '\\'; 3,518, 2.32%, contain a control byte).
//
// EXACT, not probabilistic. A Bloom filter was considered and is dominated here: its false
// positives would force the very walk being avoided, and at 19 KB for the whole vocabulary
// there is no space pressure to trade accuracy for. TestPlainString_exact proves the fast path
// agrees with the full walk for EVERY id, rather than sampling.
//
// Conservative by construction: only "definitely legal and state-invariant" is fast-pathed.
// Control-byte tokens are NOT classified illegal here even though most are, because a token
// like `"` + 0x0A closes the string before the control byte is read and is then judged in a
// different state. Everything not provably safe takes the ordinary walk.

// plainStringGrammar is implemented by grammars that can report being inside a JSON string
// with no pending escape. Optional: a grammar that does not implement it simply never takes
// the fast path, so third-party Grammar implementations keep working unchanged.
type plainStringGrammar interface {
	InPlainString() bool
}

// bitset is a fixed-size bit vector over token ids.
type bitset []uint64

func newBitset(n int) bitset { return make(bitset, (n+63)/64) }

func (b bitset) set(i int) { b[i>>6] |= 1 << (uint(i) & 63) }

func (b bitset) has(i int) bool { return i>>6 < len(b) && b[i>>6]&(1<<(uint(i)&63)) != 0 }

// buildPlainString marks every id whose surface bytes are non-empty and contain no '"', no
// '\\' and no byte < 0x20 — the ids that are unconditionally legal inside a JSON string and
// leave the string state untouched.
func buildPlainString(tokens [][]byte) bitset {
	bs := newBitset(len(tokens))
	for id, b := range tokens {
		if len(b) == 0 {
			continue // no surface bytes: masked by the empty-surface guard, not by us
		}
		ok := true
		for _, c := range b {
			if c < 0x20 || c == '"' || c == '\\' {
				ok = false
				break
			}
		}
		if ok {
			bs.set(id)
		}
	}
	return bs
}

// inPlainString reports whether g is in a plain-string state, so plainOK may be consulted.
func inPlainString(g Grammar) bool {
	ps, ok := g.(plainStringGrammar)
	return ok && ps.InPlainString()
}
