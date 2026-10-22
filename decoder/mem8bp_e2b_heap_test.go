package decoder

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync/atomic"
	"testing"
	"time"
)

// TestMEM8bp_e2bTranscodeHeap is SQE-G2 (docs/tasks/task-memory-campaign-2026-10.md, MEM8(b′) gates): StreamTranscodeGGUF of
// E2B from ~/models at int4 for the metal target, the Go heap in use sampled every 20 ms through it, and the output's sha256
// and size (hashed as written, not kept on disk). Not timed; a real checkpoint.
//
//	GOINFER_MEM8BP_E2B=1 go test -count=1 -timeout 20m -run '^TestMEM8bp_e2bTranscodeHeap$' -v ./decoder/
func TestMEM8bp_e2bTranscodeHeap(t *testing.T) {
	if os.Getenv("GOINFER_MEM8BP_E2B") == "" {
		t.Skip("set GOINFER_MEM8BP_E2B=1 (transcodes E2B from ~/models)")
	}
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "gemma-4-e2b-gguf", "gemma-4-E2B_q4_0-it.gguf")
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	base := ms.HeapInuse
	var peak atomic.Uint64
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		tick := time.NewTicker(20 * time.Millisecond)
		defer tick.Stop()
		for {
			var s runtime.MemStats
			runtime.ReadMemStats(&s)
			if s.HeapInuse > peak.Load() {
				peak.Store(s.HeapInuse)
			}
			select {
			case <-stop:
				return
			case <-tick.C:
			}
		}
	}()
	h := sha256.New()
	t0 := time.Now()
	n, err := StreamTranscodeGGUF(context.Background(), path, h, "int4", false, GIWTargetMetal, filepath.Base(path))
	close(stop)
	<-done
	if err != nil {
		t.Fatalf("StreamTranscodeGGUF: %v", err)
	}
	fmt.Fprintf(os.Stderr, "[SQE-G2] E2B int4 metal transcode: %d B in %.1fs, sha256 %x; Go heap in use: %.3f GB before, peak %.3f GB\n",
		n, time.Since(t0).Seconds(), h.Sum(nil), float64(base)/1e9, float64(peak.Load())/1e9)
}
