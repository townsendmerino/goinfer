package decoder

import (
	"flag"
	"runtime"
	"testing"
)

// The suite samples no heap profile unless one is asked for (boundSuiteHeapProfile): each GC walks every heap-profile bucket
// the process ever made, so a bucket-heavy test made every later forced GC slower.
func TestSuite_heapProfileSamplingOffUnlessAskedFor(t *testing.T) {
	asked := false
	if f := flag.Lookup("test.memprofile"); f != nil && f.Value.String() != "" {
		asked = true
	}
	switch {
	case !asked && runtime.MemProfileRate != 0:
		t.Errorf("runtime.MemProfileRate = %d with no -test.memprofile, want 0", runtime.MemProfileRate)
	case asked && runtime.MemProfileRate == 0:
		t.Errorf("runtime.MemProfileRate = 0 with -test.memprofile set: the profile would be empty")
	}
}
