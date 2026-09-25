package modelload

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// S3 Build item 4: the load-time half of the swap tripwire, which serve, chat and fit all reach through
// this package. Tested through the guarded load (loadGuarded, which Load calls once the tokenizer is in;
// no committed .gguf fixture carries both weights and a tokenizer), not through swapguard alone: a guard that is armed but whose abort channel never reaches
// decoder.Load would pass every swapguard test and protect nothing. Here the guard is replaced by one
// that has already tripped (a closed abort channel, the state a real trip leaves), so a .gguf direct
// build must come back ErrLoadAborted with the guard's wrapped message — and a load that is not a .gguf
// direct build must not arm the guard at all.
func TestLoadGuarded_swapGuardAbortReachesTheLoad(t *testing.T) {
	gguf := filepath.Join("..", "..", "testdata", "glm-tiny.gguf") // committed; its generic loader observes LoadAbort in parallelLayers
	armed := 0
	var gotAdvice string
	orig := startLoadGuard
	t.Cleanup(func() { startLoadGuard = orig })
	startLoadGuard = func(path string, _ decoder.Options, advice string) (<-chan struct{}, func(error) error, func()) {
		armed++
		gotAdvice = advice
		ch := make(chan struct{})
		close(ch) // already tripped
		wrap := func(err error) error {
			if errors.Is(err, decoder.ErrLoadAborted) {
				return errors.Join(errors.New("TRIPPED-WRAP"), err)
			}
			return err
		}
		return ch, wrap, func() {}
	}

	_, err := loadGuarded(gguf, decoder.Options{Quant: "int4"}, "use a smaller -quant")
	if armed != 1 {
		t.Fatalf("a .gguf direct load armed the guard %d times, want 1 (load error: %v)", armed, err)
	}
	if err == nil || !errors.Is(err, decoder.ErrLoadAborted) {
		t.Fatalf("load with a tripped guard = %v, want decoder.ErrLoadAborted — the abort channel did not reach decoder.Load", err)
	}
	if !strings.Contains(err.Error(), "TRIPPED-WRAP") {
		t.Errorf("error %q was not passed through the guard's wrapErr, so the user would not see the priced message", err)
	}
	if gotAdvice == "" {
		t.Error("the caller's advice did not reach the abort message")
	}

	armed = 0
	if _, err := loadGuarded(gguf, decoder.Options{Quant: "int4", StreamWeights: true}, "x"); err != nil && errors.Is(err, decoder.ErrLoadAborted) {
		t.Errorf("a streamed load was aborted by the guard: %v", err)
	}
	if armed != 0 {
		t.Errorf("a -stream-weights load armed the guard %d times; only a .gguf direct build observes LoadAbort", armed)
	}
}
