package decoder

import (
	"errors"
	"testing"
)

// TestLoadGGUFBytes_abortPropagatesErrLoadAborted is the end-to-end plumbing
// check: Options.LoadAbort -> LoadGGUFBytes -> buildGGUFWeights ->
// buildWeightsFromGGUF -> parallelLayers, through the real public entry point a
// caller (internal/serveapp's future swap-guard consumer) will actually use. A
// pre-closed channel keeps this deterministic without needing to race a real
// swap trip against a real build.
func TestLoadGGUFBytes_abortPropagatesErrLoadAborted(t *testing.T) {
	raw, _, _, _ := tinyNormRopeGGUF("llama")
	abort := make(chan struct{})
	close(abort)
	_, err := LoadGGUFBytes(raw, Options{LoadAbort: abort})
	if !errors.Is(err, ErrLoadAborted) {
		t.Fatalf("got error %v, want errors.Is(err, ErrLoadAborted)", err)
	}
}
