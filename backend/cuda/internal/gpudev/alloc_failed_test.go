//go:build linux

package gpudev

import (
	"errors"
	"strings"
	"testing"
)

// Callers that can degrade recover this panic and tell it from a bug by these words (internal/runtime/mem.DeviceAllocFailed, whose own test
// holds the same words). Reword it and every out-of-memory decline becomes a crash, with nothing failing until a card runs out, so change the
// two together.
func TestAllocFailedPanic_saysWhatMemReads(t *testing.T) {
	if got := AllocFailedPanic("kv", 64, errors.New("x")); !strings.Contains(got, "device allocation failed") {
		t.Errorf("the allocation panic says %q", got)
	}
}
