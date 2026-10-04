//go:build darwin

package metal

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestAutoBackend_int8int8 (slice 5, docs/tasks/task-metal-int8-2026-10.md): with -backend auto on Metal, a dense
// int8int8 model goes resident on the native int8 path, at the precision it loaded at, and so does an int8int8 MoE
// (slice 4); with the MoE path off, Metal would re-quantize it to int4, and auto drops it for the CPU and says why.
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
	// The generic MoE runs natively since slice 4 (nativeInt8MoE), so auto keeps it on Metal; with that path off Metal
	// would re-quantize it, and auto drops it for the CPU.
	moe, err := decoder.Load(dir, decoder.Options{Backend: "auto", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load moe: %v", err)
	}
	if dp := moe.DecodePath(); !strings.Contains(dp, "metal-resident (int8int8)") {
		t.Fatalf("int8int8 MoE under auto: DecodePath %q, want the native Metal path", dp)
	}
	moe.Close()
	prev := nativeInt8MoE
	nativeInt8MoE = false
	defer func() { nativeInt8MoE = prev }()
	req, err := decoder.Load(dir, decoder.Options{Backend: "auto", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load moe: %v", err)
	}
	defer req.Close()
	if why := req.ResidentDecline(); !strings.Contains(why, "metal would re-quantize it to int4") {
		t.Fatalf("int8int8 MoE re-quantized under auto: decline %q (DecodePath %q), want it kept on the CPU for its precision", why, req.DecodePath())
	}
}
