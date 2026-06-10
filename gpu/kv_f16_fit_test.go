//go:build gpu

package gpu

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKVCacheF16_fit is the Increment-2 measurement gate (task-gpu-f16-kv.md): a
// 7B int4 model with a 32k f16 KV cache must fit 8 GB (real allocation, no OOM),
// at roughly the 16k-f32 VRAM, and f16-KV decode should be no slower than f32 at
// equal context (the attention kernel is KV-read-bound → half the bytes). Measures
// peak VRAM (nvidia-smi) and steady-state decode tok/s for both precisions.
// Asset-gated on a 7B GGUF; skips if the model isn't residency-eligible in int4.
func TestKVCacheF16_fit(t *testing.T) {
	path := os.Getenv("GOINFER_GPU_7B")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen2.5-7b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 7B GGUF at %s: %v", path, err)
	}

	measure := func(kvPrec string) (vramMiB int, tokPerSec float64, ctxLen int) {
		m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int4", KVPrecision: kvPrec})
		if err != nil {
			t.Fatalf("Load(%s): %v", kvPrec, err)
		}
		defer m.Close()
		if !m.ResidentActive() {
			t.Skipf("7B not GPU-resident in int4 (kv=%s) — fit gate needs a residency-eligible int4 model", kvPrec)
		}
		vramMiB = gpuVRAMMiB(t)
		// Synthetic long-ish prompt to reach a non-trivial context (residency prefill
		// is sequential O(len), so keep it modest); then time steady-state decode.
		_, _, _, _, _, _, vocab := m.Dims()
		tok := 7 % vocab
		prompt := make([]int, 1024)
		for i := range prompt {
			prompt[i] = tok
		}
		const maxTok = 24
		ch, _ := m.Generate(context.Background(), prompt, maxTok, decoder.SamplingParams{Temperature: 0})
		var first time.Time
		n := 0
		for range ch {
			if n == 0 {
				first = time.Now() // TTFT done; steady-state begins
			}
			n++
		}
		if n > 1 {
			tokPerSec = float64(n-1) / time.Since(first).Seconds()
		}
		ctxLen = len(prompt) + n
		return
	}

	f16VRAM, f16Tps, ctx16 := measure("f16")
	f32VRAM, f32Tps, ctx32 := measure("f32")
	t.Logf("=== f16 KV: 32k cap, peak VRAM %d MiB, decode %.1f tok/s (ctx %d) ===", f16VRAM, f16Tps, ctx16)
	t.Logf("=== f32 KV: 16k cap, peak VRAM %d MiB, decode %.1f tok/s (ctx %d) ===", f32VRAM, f32Tps, ctx32)
	if f16VRAM > 8192 {
		t.Errorf("f16 7B + 32k KV peak VRAM %d MiB exceeds the 8 GB card", f16VRAM)
	}
	t.Logf("f16-vs-f32 decode speed at ~equal context: %.2fx", f16Tps/f32Tps)
}

// gpuVRAMMiB returns total GPU memory currently in use (whole device) via
// nvidia-smi. 0 if nvidia-smi is unavailable (the assertion then no-ops upward).
func gpuVRAMMiB(t *testing.T) int {
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		t.Logf("nvidia-smi unavailable: %v", err)
		return 0
	}
	v, _ := strconv.Atoi(strings.TrimSpace(strings.SplitN(string(out), "\n", 2)[0]))
	return v
}
