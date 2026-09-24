package constrain

import (
	"math"
	"testing"
)

// A toy vocabulary: prose words, a whole-trigger token, a trigger split across two tokens, the
// pieces of a call, and EOS (id eosID).
var lazyVocab = [][]byte{
	[]byte("Hello"), []byte(" world"), []byte("<tool_call>"), []byte("<tool"), []byte("_call>"),
	[]byte("\n"), []byte(`{"name": "`), []byte("read"), []byte("git_commit"), []byte(`", "arguments": {"x": "1"}}`),
	[]byte("\n</tool_call>"), []byte(" tag"), []byte(" "), nil,
}

const (
	tHello, tWorld, tTrig, tTrigA, tTrigB, tNL, tOpen, tRead, tGitCommit, tArgs, tClose, tTag, tSpace, eosID = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
)

func newLazy(t *testing.T) *LazyMasker {
	t.Helper()
	g, err := ToolCallsGrammar("<tool_call>", "\n</tool_call>", "arguments", false, unionTools)
	if err != nil {
		t.Fatal(err)
	}
	return NewLazyMasker(NewMasker(g, lazyVocab, []int{eosID}), "<tool_call>")
}

// step runs Process over the generated ids and returns which ids stayed legal.
func step(l *LazyMasker, gen []int) map[int]bool {
	logits := make([]float32, len(lazyVocab))
	l.Process(gen, logits)
	legal := map[int]bool{}
	for id, v := range logits {
		if !math.IsInf(float64(v), -1) {
			legal[id] = true
		}
	}
	return legal
}

// The non-forcing gate: an output that never contains the trigger is never masked, at any step.
func TestLazyMasker_neverMasksWithoutTrigger(t *testing.T) {
	l := newLazy(t)
	gen := []int{}
	for _, id := range []int{tHello, tWorld, tSpace, tTag, tHello} {
		if got := step(l, gen); len(got) != len(lazyVocab) {
			t.Fatalf("after %v: %d of %d ids legal — masked without a trigger", gen, len(got), len(lazyVocab))
		}
		gen = append(gen, id)
	}
	if l.Arms() != 0 {
		t.Errorf("armed %d times on prose", l.Arms())
	}
}

func TestLazyMasker_armsOnTrigger_andConstrains(t *testing.T) {
	for name, gen := range map[string][]int{"whole token": {tHello, tTrig}, "split across tokens": {tHello, tTrigA, tTrigB}} {
		t.Run(name, func(t *testing.T) {
			l := newLazy(t)
			legal := step(l, gen)
			if !l.Armed() {
				t.Fatal("did not arm on the trigger")
			}
			if legal[tHello] || legal[eosID] || legal[tRead] {
				t.Errorf("after the trigger: prose=%v eos=%v name-without-opener=%v legal; want all false", legal[tHello], legal[eosID], legal[tRead])
			}
			if !legal[tNL] || !legal[tOpen] {
				t.Errorf("after the trigger: newline=%v opener=%v legal; want both", legal[tNL], legal[tOpen])
			}
			legal = step(l, append(gen, tNL, tOpen))
			if !legal[tRead] || legal[tGitCommit] {
				t.Errorf("at the name: read=%v git_commit=%v; want true, false (unknown tool)", legal[tRead], legal[tGitCommit])
			}
		})
	}
}

// A space instead of a newline after the opener is still a call the grammar accepts (the lazy
// grammar's prefix is the trigger alone; JSON admits leading whitespace).
func TestLazyMasker_toleratesSpaceAfterOpener(t *testing.T) {
	l := newLazy(t)
	if legal := step(l, []int{tTrig, tSpace}); !l.Armed() || !legal[tOpen] {
		t.Errorf("armed=%v opener-legal=%v after \"<tool_call> \"", l.Armed(), legal[tOpen])
	}
}

// The trigger inside prose, continued in a shape no call can take, fails OPEN: no masking.
func TestLazyMasker_failsOpenOnProseMention(t *testing.T) {
	l := newLazy(t)
	gen := []int{tHello, tTrig, tTag} // "Hello<tool_call> tag" — the second token arrives with the trigger
	// fold the trigger token alone first: it arms, and " tag" is then masked, as intended
	step(l, gen[:2])
	if !l.Armed() {
		t.Fatal("a bare trigger token must arm (the model committed to a call)")
	}
	// but a token that carries the trigger AND an impossible continuation cannot arm
	l2 := NewLazyMasker(NewMasker(mustUnion(t), [][]byte{[]byte("use <tool_call> tags"), []byte("x"), nil}, []int{2}), "<tool_call>")
	logits := make([]float32, 3)
	l2.Process([]int{0}, logits)
	if l2.Armed() || math.IsInf(float64(logits[1]), -1) {
		t.Errorf("armed=%v on \"use <tool_call> tags\"; the continuation is impossible, so it must fail open", l2.Armed())
	}
}

// Completing a call disarms (the model is free again), and a second call arms again (T5).
func TestLazyMasker_disarmsAfterCall_andRearms(t *testing.T) {
	l := newLazy(t)
	call := []int{tTrig, tNL, tOpen, tRead, tArgs}
	legal := step(l, call)
	if legal[eosID] {
		t.Error("EOS legal before the call's suffix — the call would be cut off")
	}
	gen := append(append([]int{}, call...), tClose)
	legal = step(l, gen)
	if l.Armed() || len(legal) != len(lazyVocab) {
		t.Fatalf("after a complete call: armed=%v, %d/%d legal; want disarmed and unmasked", l.Armed(), len(legal), len(lazyVocab))
	}
	gen = append(gen, tNL, tTrig)
	step(l, gen)
	if !l.Armed() || l.Arms() != 2 {
		t.Errorf("second call: armed=%v arms=%d; want true, 2", l.Armed(), l.Arms())
	}
}

func mustUnion(t *testing.T) Grammar {
	t.Helper()
	g, err := ToolCallsGrammar("<tool_call>", "\n</tool_call>", "arguments", false, unionTools)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// Gate and Process agree: Gate reports armed exactly when Process would mask, and calling Gate then
// Process on the same ids folds each token once (the decoder calls both).
func TestLazyMasker_gateMatchesProcess(t *testing.T) {
	l := newLazy(t)
	gen := []int{}
	for _, id := range []int{tHello, tTrig, tNL, tOpen, tRead, tArgs, tClose, tWorld} {
		gen = append(gen, id)
		armed := l.Gate(gen)
		legal := step(l, gen)
		masked := len(legal) != len(lazyVocab)
		if armed != masked {
			t.Fatalf("after %v: Gate=%v but Process masked=%v", gen, armed, masked)
		}
	}
	if l.Arms() != 1 || l.Armed() {
		t.Errorf("arms=%d armed=%v after one complete call and prose; want 1, false", l.Arms(), l.Armed())
	}
}

// llama3 (option c): no opener, so the anchored masker arms on the output beginning `{"name": "`.
var llamaVocab = [][]byte{
	[]byte("Sure"), []byte(" here"), []byte("{"), []byte(`"name"`), []byte(": "), []byte(`"`), []byte("read"),
	[]byte("git_commit"), []byte(`", "parameters": {"x": "1"}}`), []byte("<|python_tag|>"), []byte(`{"answer": 42}`),
	[]byte(`{"name": "init 5.1.2`), []byte(" "), []byte("\n"), []byte(`{"na`), []byte(`me": "`), nil,
}

const (
	lSure, lHere, lBrace, lNameKey, lColon, lQuote, lRead, lGitCommit, lArgs, lPyTag, lJSONProse, lGarbageInOne, lSpace, lNL, lSplitA, lSplitB, lEOS = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
)

func newLlamaLazy(t *testing.T) *LazyMasker {
	t.Helper()
	g, err := ToolCallsGrammar("", "", "parameters", false, unionTools)
	if err != nil {
		t.Fatal(err)
	}
	return NewCallKeyLazyMasker(NewMasker(g, llamaVocab, []int{lEOS}), "<|python_tag|>")
}

func llamaStep(l *LazyMasker, gen []int) (legal map[int]bool, masked bool) {
	logits := make([]float32, len(llamaVocab))
	l.Process(gen, logits)
	legal = map[int]bool{}
	for id, v := range logits {
		if !math.IsInf(float64(v), -1) {
			legal[id] = true
		}
	}
	return legal, len(legal) != len(llamaVocab)
}

// Non-forcing: prose, a JSON answer that is not a named call, and prose-then-JSON never mask.
func TestCallKeyLazyMasker_neverMasksNonCalls(t *testing.T) {
	for name, gen := range map[string][]int{
		"prose":              {lSure, lHere, lBrace, lNameKey, lColon, lQuote}, // the key appears, but not at the start
		"json, other key":    {lJSONProse, lSpace, lBrace},
		"leading ws + prose": {lSpace, lNL, lSure},
	} {
		t.Run(name, func(t *testing.T) {
			l := newLlamaLazy(t)
			for i := 1; i <= len(gen); i++ {
				if _, masked := llamaStep(l, gen[:i]); masked {
					t.Fatalf("masked after %v", gen[:i])
				}
			}
			if l.Arms() != 0 {
				t.Errorf("armed on %s", name)
			}
		})
	}
}

// A named call object arms at the name, and the name is then constrained to supplied tools.
func TestCallKeyLazyMasker_armsOnNameKey(t *testing.T) {
	for name, gen := range map[string][]int{
		"plain":        {lBrace, lNameKey, lColon, lQuote},
		"leading ws":   {lNL, lSpace, lBrace, lNameKey, lColon, lQuote},
		"python_tag":   {lPyTag, lBrace, lNameKey, lColon, lQuote},
		"split tokens": {lSplitA, lSplitB},
	} {
		t.Run(name, func(t *testing.T) {
			l := newLlamaLazy(t)
			for i := 1; i < len(gen); i++ {
				if _, masked := llamaStep(l, gen[:i]); masked {
					t.Fatalf("masked before the name key was complete, after %v", gen[:i])
				}
			}
			legal, masked := llamaStep(l, gen)
			if !l.Armed() || !masked {
				t.Fatalf("armed=%v masked=%v after the name key", l.Armed(), masked)
			}
			if !legal[lRead] || legal[lGitCommit] || legal[lEOS] {
				t.Errorf("at the name: read=%v git_commit=%v eos=%v; want true,false,false", legal[lRead], legal[lGitCommit], legal[lEOS])
			}
		})
	}
	// completing the call disarms, and it never re-arms (one call per turn for this form)
	l := newLlamaLazy(t)
	gen := []int{lBrace, lNameKey, lColon, lQuote, lRead, lArgs}
	if _, masked := llamaStep(l, gen); masked || l.Armed() {
		t.Errorf("after a complete call: masked=%v armed=%v", masked, l.Armed())
	}
	gen = append(gen, lNL, lBrace, lNameKey, lColon, lQuote)
	if _, masked := llamaStep(l, gen); masked || l.Arms() != 1 {
		t.Errorf("re-armed on a second object: masked=%v arms=%d", masked, l.Arms())
	}
}

// A token that carries the key AND a name no tool has fails open (it cannot be taken back).
func TestCallKeyLazyMasker_failsOpenOnImpossibleName(t *testing.T) {
	l := newLlamaLazy(t)
	if _, masked := llamaStep(l, []int{lGarbageInOne}); masked || l.Armed() {
		t.Errorf("masked=%v armed=%v on %q", masked, l.Armed(), llamaVocab[lGarbageInOne])
	}
}

func TestMatchCallKey(t *testing.T) {
	skip := []byte("<|python_tag|>")
	for _, c := range []struct {
		in    string
		start int
		st    anchor
	}{
		{`{"name": "`, 0, anchorArm},
		{`  {"name":"`, 2, anchorArm},
		{"<|python_tag|>{\"name\" : \"", 14, anchorArm},
		{"<|pyth", 0, anchorMore},
		{`{"na`, 0, anchorMore},
		{`{"name": `, 0, anchorMore},
		{`{"type":`, 0, anchorNever},
		{`Sure`, 0, anchorNever},
		{`[{"name": "`, 0, anchorNever},
	} {
		start, st := matchCallKey([]byte(c.in), skip)
		if st != c.st || (st == anchorArm && start != c.start) {
			t.Errorf("matchCallKey(%q) = %d,%v; want %d,%v", c.in, start, st, c.start, c.st)
		}
	}
}
