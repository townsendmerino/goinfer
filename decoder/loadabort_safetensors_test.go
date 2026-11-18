package decoder

import (
	"errors"
	"testing"
)

// TestLoadWeights_safetensorsHonorsAbort is the safetensors twin of TestLoadGGUFBytes_abortPropagatesErrLoadAborted: a
// closed abort channel handed to loadWeights (what Options.LoadAbort becomes) stops a resident safetensors build with
// ErrLoadAborted, through the generic per-layer loader. A pre-closed channel keeps it deterministic.
func TestLoadWeights_safetensorsHonorsAbort(t *testing.T) {
	dir := tinyLlamaDir(t)
	abort := make(chan struct{})
	close(abort)
	if _, err := loadWeights(dir, quantNone, false, true, false, nil, abort); !errors.Is(err, ErrLoadAborted) {
		t.Fatalf("got error %v, want errors.Is(err, ErrLoadAborted)", err)
	}
	if _, err := loadWeights(dir, quantNone, false, true, false, nil, nil); err != nil {
		t.Fatalf("a nil abort must still load: %v", err)
	}
}
