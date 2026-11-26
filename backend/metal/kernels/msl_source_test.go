package kernels

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

// TestMSLSource_librariesArePinned holds each library joined from the .metal files to its recorded sha256, so a file
// reordered, dropped or joined twice is red. A deliberate edit to any of a library's sources records its new hash here.
func TestMSLSource_librariesArePinned(t *testing.T) {
	for _, c := range []struct{ name, src, want string }{
		{"AllKernels", AllKernels, "13be9bfc566b3db326f9bf3d08e22a64e11d28640112b94c57c4f513fe1195c6"},
		{"PrefillKernels", PrefillKernels, "335322cb51f5a86a7240bde48d1afc7cb045b5d65d336d3308e12e93ebe89c0c"},
	} {
		sum := sha256.Sum256([]byte(c.src))
		if got := hex.EncodeToString(sum[:]); got != c.want {
			t.Errorf("%s (%d bytes): sha256 %s, want %s: a source file is reordered, missing or doubled in the join, "+
				"or a kernel was edited and its new hash is not recorded here", c.name, len(c.src), got, c.want)
		}
	}
}
