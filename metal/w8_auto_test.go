//go:build darwin

package metal

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestAutoBackend_int8int8 (slice 5, docs/tasks/task-metal-int8-2026-10.md): with -backend auto on Metal, a dense
// int8int8 model goes resident on the native int8 path, at the precision it loaded at; an int8int8 MoE, which Metal
// would re-quantize to int4, is dropped for the CPU and says why.
func TestAutoBackend_int8int8(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	if c := decoder.AutoBackend(); c.Backend != "metal" {
		t.Skipf("auto chose %q here", c.Backend)
	}
	dense, err := decoder.Load("../testdata/llama-attnfa-tiny", decoder.Options{Backend: "auto", Quant: "int8int8", ResidentContext: 512})
	if err != nil {
		t.Fatalf("load dense: %v", err)
	}
	defer dense.Close()
	if dp := dense.DecodePath(); !strings.Contains(dp, "metal-resident (int8int8)") {
		t.Fatalf("dense int8int8 under auto: DecodePath %q, want the native Metal path", dp)
	}
	dir := t.TempDir()
	writeMoEDistinct(t, dir, genTinyWeights(rand.New(rand.NewSource(5))), 8, 2, rand.New(rand.NewSource(6)))
	moe, err := decoder.Load(dir, decoder.Options{Backend: "auto", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load moe: %v", err)
	}
	defer moe.Close()
	if why := moe.ResidentDecline(); !strings.Contains(why, "metal would re-quantize it to int4") {
		t.Fatalf("int8int8 MoE under auto: decline %q (DecodePath %q), want it kept on the CPU for its precision", why, moe.DecodePath())
	}
}
