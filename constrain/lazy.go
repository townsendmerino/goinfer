package constrain

import "bytes"

// LazyMasker applies a Masker's grammar only once the model commits to it: it masks nothing
// until a trigger (a tool-call opener such as "<tool_call>") appears in the generated output,
// then constrains from the trigger on, and disarms again as soon as the grammar's document is
// complete. Task T2, docs/tasks/task-tool-grammar-union-2026-09.md: under tool_choice "auto" a
// prose answer must stay legal (ground rule 1), so the constraint cannot start at token 1 — it
// arms on the model's own decision to call, and from there the call cannot be malformed.
// Disarming after each complete call means a second call in the same turn is constrained
// independently (T5, decided: repeated wrapper, each constrained).
//
// The grammar must START with the trigger's bytes (build it with the trigger as its prefix):
// on arming, the bytes from the trigger's first byte through the end of the latest token are
// fed to a freshly Reset grammar. If the grammar rejects them — the trigger text appeared in
// prose, or the model continued the opener in a shape the grammar cannot express — the lazy
// masker FAILS OPEN for that occurrence and keeps scanning: bytes already emitted cannot be
// taken back, and a grammar that cannot back out is worse than none (the task's own words).
//
// Use Process as the LogitProcessor. It is NOT a grammar-fused speculative masker (the grammar
// is not live from token 1), so callers must not hand it to the fused-spec path.
type LazyMasker struct {
	m        *Masker
	triggers [][]byte
	maxTrig  int
	scanned  int    // generated tokens already folded in
	window   []byte // unarmed output not yet ruled out as the start of a trigger
	armed    bool
	arms     int // how many times it armed (a call started)
}

// NewLazyMasker wraps m (whose grammar begins with a trigger's bytes) to arm on any of triggers.
func NewLazyMasker(m *Masker, triggers ...string) *LazyMasker {
	l := &LazyMasker{m: m}
	for _, t := range triggers {
		if t == "" {
			continue
		}
		l.triggers = append(l.triggers, []byte(t))
		l.maxTrig = max(l.maxTrig, len(t))
	}
	return l
}

// Armed reports whether the grammar is currently in force; Arms how many calls it has armed on.
func (l *LazyMasker) Armed() bool { return l.armed }
func (l *LazyMasker) Arms() int   { return l.arms }

// Gate is a decoder.SamplingParams.LogitProcessorGate: it folds the generated ids in and reports
// whether the NEXT step must be masked. With it set, the decoder keeps its on-device fast paths
// for every step before the opener (and after a completed call) and pays for full logits only
// while a call is being written — so a prose turn costs what it cost with no constraint at all.
func (l *LazyMasker) Gate(generated []int) bool {
	for ; l.scanned < len(generated); l.scanned++ {
		l.fold(l.m.tokenBytes(generated[l.scanned]))
	}
	return l.armed
}

// Process is a decoder.SamplingParams.LogitProcessor.
func (l *LazyMasker) Process(generated []int, logits []float32) {
	for ; l.scanned < len(generated); l.scanned++ {
		l.fold(l.m.tokenBytes(generated[l.scanned]))
	}
	if l.armed {
		l.m.MaskAt(l.m.g, logits)
	}
}

func (l *LazyMasker) fold(b []byte) {
	if l.armed {
		// The mask allowed only grammar-legal tokens, so b is accepted; the check keeps a caller
		// that bypasses the mask (or a padded-vocab id) from corrupting the grammar state.
		if !l.m.g.TryBytes(b) {
			l.disarm()
			return
		}
		l.m.g.Commit(b)
		if l.m.g.CanEnd() {
			l.disarm() // the call is complete: the model is free again (EOS, prose, another call)
		}
		return
	}
	l.window = append(l.window, b...)
	for {
		i, ok := l.firstTrigger()
		if !ok {
			break
		}
		pending := l.window[i:]
		l.m.g.Reset()
		if l.m.g.TryBytes(pending) {
			l.m.g.Commit(pending)
			l.armed, l.window = true, l.window[:0]
			l.arms++
			if l.m.g.CanEnd() { // the whole call arrived inside one token
				l.disarm()
			}
			return
		}
		l.window = l.window[i+1:] // fail open for this occurrence; a later one may still arm
	}
	// Keep only a tail that could still begin a trigger split across tokens.
	if keep := l.maxTrig - 1; len(l.window) > keep {
		l.window = append(l.window[:0], l.window[len(l.window)-keep:]...)
	}
}

func (l *LazyMasker) firstTrigger() (int, bool) {
	best := -1
	for _, t := range l.triggers {
		if i := bytes.Index(l.window, t); i >= 0 && (best < 0 || i < best) {
			best = i
		}
	}
	return best, best >= 0
}

func (l *LazyMasker) disarm() {
	l.armed = false
	l.window = l.window[:0]
}
