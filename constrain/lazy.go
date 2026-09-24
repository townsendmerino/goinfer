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

	// Anchored mode (NewCallKeyLazyMasker): instead of a literal opener, arm when the OUTPUT ITSELF
	// begins as a named call object. head holds the output until that is decided; decided stops the
	// scan for good (armed once, or ruled out).
	anchored bool
	skip     []byte
	head     []byte
	decided  bool
}

// NewCallKeyLazyMasker is the lazy masker for a family whose call has NO opener (llama3: the call is a
// bare JSON object — task option c). It arms only when the output begins, after whitespace and an
// optional skip marker (llama3's "<|python_tag|>"), with `{` + `"name"` + `:` + `"` (JSON whitespace
// allowed between them): by then the model has written the name key of a call object, the same
// commitment an opener signals. The grammar is fed from the `{`, so build it with an EMPTY prefix.
// An output that begins any other way is never masked at all, so there is nothing to back out of.
// It arms at most once per generation.
func NewCallKeyLazyMasker(m *Masker, skip string) *LazyMasker {
	return &LazyMasker{m: m, anchored: true, skip: []byte(skip)}
}

type anchor int

const (
	anchorMore  anchor = iota // not enough output to decide
	anchorArm                 // arm, feeding the grammar from the returned index
	anchorNever               // this output is not a named call object; never arm
)

// matchCallKey decides whether out begins as `{"name": "` after whitespace and an optional skip marker.
func matchCallKey(out, skip []byte) (int, anchor) {
	isWS := func(c byte) bool { return c == ' ' || c == '\t' || c == '\n' || c == '\r' }
	i := 0
	for i < len(out) && isWS(out[i]) {
		i++
	}
	if len(skip) > 0 {
		rest := out[i:]
		switch {
		case bytes.HasPrefix(rest, skip):
			i += len(skip)
			for i < len(out) && isWS(out[i]) {
				i++
			}
		case len(rest) < len(skip) && bytes.HasPrefix(skip, rest):
			return 0, anchorMore
		}
	}
	start := i
	for _, lit := range []string{"{", `"name"`, ":", `"`} {
		for i < len(out) && isWS(out[i]) && lit != "{" {
			i++
		}
		for k := 0; k < len(lit); k++ {
			if i >= len(out) {
				return 0, anchorMore
			}
			if out[i] != lit[k] {
				return 0, anchorNever
			}
			i++
		}
	}
	return start, anchorArm
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
	if l.anchored {
		l.foldAnchored(b)
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

func (l *LazyMasker) foldAnchored(b []byte) {
	if l.decided {
		return
	}
	l.head = append(l.head, b...)
	start, st := matchCallKey(l.head, l.skip)
	switch st {
	case anchorMore:
		return
	case anchorNever:
		l.decided, l.head = true, nil
		return
	}
	l.decided = true
	pending := l.head[start:]
	l.head = nil
	l.m.g.Reset()
	if !l.m.g.TryBytes(pending) {
		return // fail open: the bytes after the key cannot be a call to a supplied tool's grammar prefix
	}
	l.m.g.Commit(pending)
	l.armed = true
	l.arms++
	if l.m.g.CanEnd() {
		l.disarm()
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
