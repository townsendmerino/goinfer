//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestFlashDecode_defaultOnRealResident is the hardware half of the R6 default flip: on a real resident, with
// GOINFER_CUDA_FLASH_DECODE genuinely UNSET the lane loads at the registered S with the default floor, and =0 leaves it off. The unit
// tests pin how the variable resolves; only this shows the resident actually ends up wired that way, which is what "default ON" means.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestFlashDecode_defaultOnRealResident -v
func TestFlashDecode_defaultOnRealResident(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint on the GPU)")
	}
	path := modelPath("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	load := func(t *testing.T) *cudaResident {
		t.Helper()
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		t.Cleanup(func() { m.Close() })
		rf, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok || rf == nil {
			t.Skipf("not CUDA-resident (%s)", m.ResidentDecline())
		}
		return rf
	}

	t.Run("unset is ON", func(t *testing.T) {
		t.Setenv("GOINFER_CUDA_FLASH_DECODE", "x") // restores the original at cleanup
		unsetenv(t, "GOINFER_CUDA_FLASH_DECODE")
		t.Setenv("GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", "x")
		unsetenv(t, "GOINFER_CUDA_FLASH_DECODE_MIN_KEYS")
		rf := load(t)
		if rf.faSplit != flashDecodeDefaultSplit || rf.faCombine == (Pipeline{}) {
			t.Fatalf("default did not wire the lane: faSplit=%d (want %d), combine loaded=%v", rf.faSplit, flashDecodeDefaultSplit, rf.faCombine != (Pipeline{}))
		}
		if rf.faMinKeys != flashDecodeDefaultMinKeys {
			t.Fatalf("default floor = %d, want %d", rf.faMinKeys, flashDecodeDefaultMinKeys)
		}
		t.Logf("default: lane ON, S=%d, floor=%d keys, verify lane=%v", rf.faSplit, rf.faMinKeys, rf.faVerify)
	})
	t.Run("=0 is OFF", func(t *testing.T) {
		t.Setenv("GOINFER_CUDA_FLASH_DECODE", "0")
		rf := load(t)
		if rf.faSplit != 0 {
			t.Fatalf("GOINFER_CUDA_FLASH_DECODE=0 left the lane on: faSplit=%d", rf.faSplit)
		}
	})
}
