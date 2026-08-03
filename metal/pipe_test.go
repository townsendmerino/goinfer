//go:build darwin

package metal

import (
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestEncodeAhead — compares the synchronous forward vs the pipelined executor (encode-ahead),
// both best-of-N warm. Encode-ahead should shrink the wall toward the GPU-busy floor (~13.1ms)
// by hiding the ~0.9ms host encode bubble behind the previous token's GPU execution. Parity is
// unchanged (same kernels); this only reorders when the encode happens.
func TestEncodeAhead(t *testing.T) {
	requireHeavyModel(t)
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present: %v", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := BuildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	// ---- parity: pipelined path must produce the SAME argmax as the synchronous path ----
	// The KV cache is positional (each pos overwrites its slot), so running a teacher-forced
	// sequence through the pipe then through sync recomputes the same states. Encode-ahead only
	// reorders when the encode happens, not the math.
	ids := []int{785, 12095, 8948, 264, 6236, 1140, 13, 358, 3003, 264}
	pipeArg := make([]int, len(ids))
	for i, id := range ids {
		e := make([]float32, r.H)
		r.embed.Row(id, e)
		pipeArg[i] = argmaxF(r.ForwardEmbPipe(e, i))
	}
	mism := 0
	for i, id := range ids {
		e := make([]float32, r.H)
		r.embed.Row(id, e)
		if argmaxF(r.ForwardEmb(e, i)) != pipeArg[i] {
			mism++
		}
	}
	if mism != 0 {
		t.Fatalf("encode-ahead parity FAIL: %d/%d argmax differ from synchronous", mism, len(ids))
	}
	t.Logf("encode-ahead parity: %d/%d argmax identical to synchronous path", len(ids), len(ids))

	emb := make([]float32, r.H)
	r.embed.Row(12095, emb) // a fixed embedding is enough for a throughput measurement

	best := func(n int, f func(i int)) time.Duration {
		for i := range 8 {
			f(i)
		}
		b := time.Hour
		for i := range n {
			t0 := time.Now()
			f(i)
			if dt := time.Since(t0); dt < b {
				b = dt
			}
		}
		return b
	}

	sync := best(60, func(i int) { r.ForwardEmb(emb, 30+i) })
	pipe := best(60, func(i int) { r.ForwardEmbPipe(emb, 30+i) })
	gpuBusy, _ := r.LastGPUTimes()
	t.Logf("synchronous ForwardEmb : %.2f ms/token (%.1f tok/s)", sync.Seconds()*1000, 1/sync.Seconds())
	t.Logf("pipelined  ForwardEmbPipe: %.2f ms/token (%.1f tok/s)", pipe.Seconds()*1000, 1/pipe.Seconds())
	t.Logf("GPU-busy floor: %.2f ms/token (%.1f tok/s)  | encode-ahead recovered %.2f ms",
		gpuBusy*1000, 1/gpuBusy, sync.Seconds()*1000-pipe.Seconds()*1000)
	r.stopExec()
}
