//go:build darwin

package gpudev

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

// TestViTSource_isPinned holds vitMSL, joined from the vit_*.metal files, to its recorded sha256, so a file reordered,
// dropped or joined twice is red. A deliberate edit to a vit_*.metal file records the new hash here.
func TestViTSource_isPinned(t *testing.T) {
	const want = "f6ea1fca0cd0f3080cda3cb761e166405b2d8c606129cbce73ee0e438652d335"
	sum := sha256.Sum256([]byte(vitMSL))
	if got := hex.EncodeToString(sum[:]); got != want {
		t.Errorf("vitMSL (%d bytes): sha256 %s, want %s: a vit_*.metal file is reordered, missing or doubled in the join, "+
			"or a kernel was edited and its new hash is not recorded here", len(vitMSL), got, want)
	}
}
