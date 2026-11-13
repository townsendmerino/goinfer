package kvmgr

import "slices"

// SequenceState is the live KV state of one sequence, whether held in a CPU session cache or a GPU resident slot.
// Prefix scoring, window-horizon checks, cross-sequence prefix sharing, turn rewinding, and post-turn reconciliation
// operate on this contract so CPU and resident KV share one set of rules.
type SequenceState interface {
	Len() int
	Tokens() []int
	Root() Root
	Images() []ImageSpan
	Classes() []ClassRun
	Recurrent() bool
	Checkpoint() int
	Floor() int
	// ReusablePrefix reports how many of the first prefix positions remain physically servable when a turn continues
	// at position prefix: prefix itself when every layer's window horizon and state cover it, or 0 when dropped rows
	// or non-positional state require a cold prefill.
	ReusablePrefix(prefix int) int
	// Truncate rewinds the backing store to target positions and reports whether the cut is exact.
	Truncate(target int) bool
}

// CommonPrefixLen returns the length of the longest common leading run of a and b.
func CommonPrefixLen(a, b []int) int {
	n := min(len(a), len(b))
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

// ClipClassRuns returns runs restricted to rows [0, n), with adjacent runs of the same class merged.
func ClipClassRuns(runs []ClassRun, n int) []ClassRun {
	return clipClasses(runs, n)
}

// RecordTurnClasses builds the class runs of a completed sequence of totalLen tokens whose turn reused matched leading
// rows, prefilled prompt[matched:min(promptLen,totalLen)] with pass, and decoded the remaining tokens as ClassExact.
func RecordTurnClasses(prior []ClassRun, matched, promptLen, totalLen int, pass Class) []ClassRun {
	out := clipClasses(prior, min(matched, totalLen))
	if end := min(promptLen, totalLen); end > matched {
		out = appendClass(out, ClassRun{Start: matched, Len: end - matched, Class: pass})
	}
	if totalLen > promptLen {
		out = appendClass(out, ClassRun{Start: promptLen, Len: totalLen - promptLen, Class: ClassExact})
	}
	return out
}

// WindowNeedStart returns the first key position a query at absolute position pos attends to under sliding window
// window. A non-positive window attends from 0.
func WindowNeedStart(pos, window int) int {
	if window <= 0 {
		return 0
	}
	return max(pos-window+1, 0)
}

// WindowHorizon returns the oldest key position still physically held after written positions in a windowed store of
// capacity window + max(slack, 0). A non-positive window or an unwrapped store holds from position 0.
func WindowHorizon(written, window, slack int) int {
	if window <= 0 {
		return 0
	}
	capN := window + max(slack, 0)
	if written <= capN {
		return 0
	}
	return written - capN
}

// WindowReusable reports whether a forward continuing at position prefix finds its sliding window held by a store whose
// oldest resident position is base.
func WindowReusable(base, prefix, window int) bool {
	return base == 0 || window <= 0 || WindowNeedStart(prefix, window) >= base
}

// WindowStepStatus classifies the outcome of advancing a windowed slot's base for a write.
type WindowStepStatus uint8

const (
	WindowStepOK WindowStepStatus = iota
	WindowStepBelowBase
	WindowStepOverCap
)

// WindowNextBase computes the base before and after writing n rows starting at p0 into a windowed slot of capacity
// capN with window size window and rollback budget rollback. A fresh sequence (p0 == 0) resets the base to 0.
func WindowNextBase(base, p0, n, window, capN, rollback int) (from, to, low int, status WindowStepStatus) {
	if p0 == 0 {
		base = 0
	}
	low = WindowNeedStart(p0, window)
	if low < base {
		return base, base, low, WindowStepBelowBase
	}
	if p0+n-base <= capN {
		return base, base, low, WindowStepOK
	}
	newBase := max(base, min(low, p0+n-(window-1)-rollback))
	if p0+n-newBase > capN {
		return base, newBase, low, WindowStepOverCap
	}
	return base, newBase, low, WindowStepOK
}

// RingTruncateExact reports whether rewinding a sliding-window ring of size window with written positions to target
// leaves every position of the post-truncation window physically resident.
func RingTruncateExact(written, window, target int) bool {
	if target >= written || window <= 0 || written <= window {
		return true
	}
	return WindowHorizon(written, window, 0) <= target-window+1
}

// SafeReuseLen clamps a candidate prefix match against promptLen (always leaving at least one token to compute), the
// sequence's floor, and the backing store's reusable-prefix horizon (nil when every prefix is servable).
func SafeReuseLen(matched, promptLen, floor int, reusable func(int) int) int {
	if matched <= 0 || promptLen <= 1 {
		return 0
	}
	n := min(matched, promptLen-1)
	if n < floor {
		return 0
	}
	if reusable != nil {
		n = reusable(n)
		if n <= 0 {
			return 0
		}
		n = min(n, promptLen-1)
		if n < floor {
			return 0
		}
	}
	return n
}

// ScoreSequence returns how many leading tokens of req seq already holds and can serve: token-id prefix match under
// equal roots, atomic image spans, recurrent strict-extension or checkpoint restart rules, floor, and window horizon.
func ScoreSequence(seq SequenceState, req Request) int {
	if seq == nil || seq.Len() == 0 || len(req.Tokens) == 0 || seq.Root() != req.Root {
		return 0
	}
	n := scoreSeqTokens(seq.Tokens(), seq.Images(), seq.Checkpoint(), req, seq.Recurrent())
	return SafeReuseLen(n, len(req.Tokens), seq.Floor(), seq.ReusablePrefix)
}

// scoreSeqTokens scores req's tokens and image claims against a stored sequence's tokens, image spans, and checkpoint,
// assuming the roots already match.
func scoreSeqTokens(tokens []int, images []ImageSpan, ckpt int, req Request, recurrent bool) int {
	if len(tokens) == 0 || len(req.Tokens) == 0 {
		return 0
	}
	if recurrent {
		return scoreRecurrentTokens(tokens, images, ckpt, req)
	}
	n := min(len(req.Tokens)-1, len(tokens))
	i, bi := 0, 0
	for i < n {
		for bi < len(images) && images[bi].end() <= i {
			bi++
		}
		if bi < len(images) && images[bi].Start == i {
			sp := images[bi]
			if c, ok := claimAt(req.Images, sp.Start); ok && c.Len == sp.Len && sp.Hash != 0 && c.Hash == sp.Hash && sp.end() <= n {
				i = sp.end()
				bi++
				continue
			}
			return i
		}
		if tokens[i] != req.Tokens[i] {
			return i
		}
		i++
	}
	return i
}

// scoreRecurrentTokens is scoreSeqTokens for a recurrent sequence: the full sequence length on a strict extension with
// identical image spans over the reused prefix, else the checkpoint position when Request.Checkpoint is set and the
// prompt extends past the checkpoint with no image spans on either side.
func scoreRecurrentTokens(tokens []int, images []ImageSpan, ckpt int, req Request) int {
	n := len(tokens)
	if len(req.Tokens) <= n || !slices.Equal(tokens, req.Tokens[:n]) {
		if !req.Checkpoint || ckpt <= 0 || len(req.Tokens) <= ckpt || len(images) > 0 || len(req.Images) > 0 ||
			!slices.Equal(tokens[:ckpt], req.Tokens[:ckpt]) {
			return 0
		}
		return ckpt
	}
	for _, sp := range images {
		c, ok := claimAt(req.Images, sp.Start)
		if !ok || sp.Hash == 0 || c.Hash != sp.Hash || c.Len != sp.Len {
			return 0
		}
	}
	for _, c := range req.Images {
		if c.Start >= n {
			continue
		}
		if c.end() > n {
			return 0
		}
		held := false
		for _, sp := range images {
			if sp.Start == c.Start && sp.Len == c.Len {
				held = true
				break
			}
		}
		if !held {
			return 0
		}
	}
	return n
}

// SharedPrefixHit returns how many leading tokens of prompt another sequence can copy from seq: whole blocks of a
// common prefix whose rows are ClassFast under base weights, capped so the requester's remaining suffix also runs on
// ClassFast.
func SharedPrefixHit(seq SequenceState, prompt []int, cm ClassMap, canBatch bool) int {
	n := len(prompt)
	if seq == nil || n == 0 || seq.Root().Adapter != 0 || !canBatch || cm.Pass(n, n) != ClassFast {
		return 0
	}
	hit := min(CommonPrefixLen(seq.Tokens(), prompt), n-1) / BlockRows * BlockRows
	classes := seq.Classes()
	for p := 0; p < hit; p++ {
		if classAt(classes, p) != ClassFast {
			hit = p / BlockRows * BlockRows
			break
		}
	}
	return cm.CapHit(n, hit)
}

// RewindSequence rewinds seq in place for an incoming turn req and returns how many leading tokens of req remain live.
// A clean empty cache is left untouched; a root mismatch or an inexact cut resets seq to 0 and returns 0.
func RewindSequence(seq SequenceState, req Request, clean bool) int {
	if clean || seq == nil {
		return 0
	}
	if seq.Len() > 0 && seq.Root() != req.Root {
		seq.Truncate(0)
		return 0
	}
	raw := scoreSeqTokens(seq.Tokens(), seq.Images(), seq.Checkpoint(), req, seq.Recurrent())
	matched := SafeReuseLen(raw, len(req.Tokens), seq.Floor(), seq.ReusablePrefix)
	if !seq.Truncate(matched) {
		seq.Truncate(0)
		return 0
	}
	return matched
}

// RewindPrefix applies the token-prefix rewind contract over plain slices and a truncate callback: clean caches are
// untouched, an adapter change resets to 0, at least one prompt token is left to compute, and an inexact cut resets to
// 0 and returns 0.
func RewindPrefix(tokens, prompt []int, clean, adapterChanged bool, truncate func(int) bool) int {
	if clean {
		return 0
	}
	if adapterChanged {
		truncate(0)
		return 0
	}
	matched := SafeReuseLen(CommonPrefixLen(tokens, prompt), len(prompt), 0, nil)
	if !truncate(matched) {
		truncate(0)
		return 0
	}
	return matched
}

// ReconcileSequence reconciles a sequence's committed token list seq against the backing KV position pos after a turn.
// A turn cut off mid-write (pos < len(seq)) clamps to pos unless the store carries recurrent state, which cannot be
// rolled back and resets to 0; an inexact truncation at the final length also resets to 0.
func ReconcileSequence(pos int, seq []int, recurrent func() bool, truncate func(int) bool) ([]int, bool) {
	rolledBack := pos < len(seq)
	if rolledBack {
		seq = seq[:pos]
	}
	if rolledBack && recurrent != nil && recurrent() {
		truncate(0)
		return nil, false
	}
	if !truncate(len(seq)) {
		truncate(0)
		return nil, false
	}
	return seq, true
}

// DecideRecurrentCheckpoint evaluates whether an acquired reuse length on a recurrent sequence requires restoring its
// saved checkpoint, leaves the live state as-is, or declines back to a cold prefill.
func DecideRecurrentCheckpoint(reuse, slotLen, promptLen, minTail int, recurrent, haveCheckpoint bool) (nextReuse int, restore bool, declined string) {
	if reuse <= 0 || reuse >= slotLen || !recurrent {
		return reuse, false, ""
	}
	if !haveCheckpoint {
		return 0, false, "no checkpoint kept"
	}
	if promptLen-reuse < minTail {
		return 0, false, "suffix under the final-pass minimum"
	}
	return reuse, true, ""
}

// slotSequenceState adapts one slot of Slots to SequenceState.
type slotSequenceState struct {
	slots     *Slots
	slot      int
	recurrent bool
	reusable  func(int) int
	truncate  func(int) bool
}

// State returns a SequenceState view of slot in a, with no recurrent state and full window retention.
func (a *Slots) State(slot int) SequenceState {
	return a.StateWith(slot, false, nil, nil)
}

// StateWith returns a SequenceState view of slot in a with backend hooks for recurrent state, window-horizon checks,
// and in-place truncation.
func (a *Slots) StateWith(slot int, recurrent bool, reusable func(int) int, truncate func(int) bool) SequenceState {
	return slotSequenceState{slots: a, slot: slot, recurrent: recurrent, reusable: reusable, truncate: truncate}
}

func (s slotSequenceState) seq() *Seq {
	if s.slots == nil || s.slot < 0 || s.slot >= s.slots.Count() {
		return nil
	}
	return s.slots.Seq(s.slot)
}

func (s slotSequenceState) Len() int {
	if q := s.seq(); q != nil {
		return q.Len()
	}
	return 0
}

func (s slotSequenceState) Tokens() []int {
	if q := s.seq(); q != nil {
		return q.Tokens()
	}
	return nil
}

func (s slotSequenceState) Root() Root {
	if q := s.seq(); q != nil {
		return q.Root()
	}
	return Root{}
}

func (s slotSequenceState) Images() []ImageSpan {
	if q := s.seq(); q != nil {
		return q.Images()
	}
	return nil
}

func (s slotSequenceState) Classes() []ClassRun {
	if q := s.seq(); q != nil {
		return q.Classes()
	}
	return nil
}

func (s slotSequenceState) Recurrent() bool { return s.recurrent }

func (s slotSequenceState) Checkpoint() int {
	if q := s.seq(); q != nil {
		c, _ := q.Checkpoint()
		return c
	}
	return 0
}

func (s slotSequenceState) Floor() int {
	if s.slots == nil || s.slot < 0 || s.slot >= len(s.slots.floors) {
		return 0
	}
	return s.slots.floors[s.slot]
}

func (s slotSequenceState) ReusablePrefix(prefix int) int {
	if prefix <= 0 {
		return 0
	}
	if prefix > s.Len() {
		return 0
	}
	if s.reusable != nil {
		return s.reusable(prefix)
	}
	return prefix
}

func (s slotSequenceState) Truncate(target int) bool {
	if s.truncate != nil {
		return s.truncate(target)
	}
	if s.slots == nil || s.slot < 0 || s.slot >= s.slots.Count() {
		return false
	}
	if target == 0 {
		s.slots.Forget(s.slot)
		return true
	}
	return target <= s.Len() && (!s.recurrent || target == s.Len())
}
