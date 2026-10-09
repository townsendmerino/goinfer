//go:build goinfer_testhooks

package prequant

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// G-DS1's planted defect: with layers 0 and 1 written in each other's place, the byte-identity comparison must fail.
func TestDirStream_plantedLayerSwapIsCaught(t *testing.T) {
	defer decoder.SetStreamDirSwapForTest(true)()
	res, str, skip, ns := dirStreamCompare(t, "../../testdata/llama-tiny", "int4", false, decoder.GIWTargetForBackend("cpu"))
	if skip != "" || ns {
		t.Fatalf("llama-tiny must stream for this control (skip %q, noStream %v)", skip, ns)
	}
	why := dirStreamDiff(t, res, str)
	if why == "" {
		t.Fatal("the gate is BLIND: a streamed bundle with layers 0 and 1 swapped compares identical to the resident one")
	}
	t.Logf("caught: %s", why)
}
