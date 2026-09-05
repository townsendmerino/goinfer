//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillTTFT scopes the "batched Metal prefill as a pure TTFT lever" question
// (independent of P10/speculative decode): if the bit-identity gap were fixed and
// batched PrefillLast shipped for prompt ingestion, how much TTFT would it actually
// buy on a real prompt-length range, vs today's shipped sequential Forward-per-token
// prefill? See the "Metal batched prefill... a TTFT lever" note in
// docs/ollama-chase.md §A2-Metal and docs/task-gpu-batched-prefill.md (the WebGPU
// analog, GATED as a wash on this hardware class — this measures the NATIVE metal/
// path, a different kernel).
//
// Position is passed explicitly to every Forward/PrefillLast call (metalResident has
// no internal position state — TruncateTo/Reset are no-ops), so a fresh "from empty"
// measurement per P just means starting the position counter at 0; no cache reset
// needed between runs.
//
//	GOINFER_HEAVY_TESTS=1 go test ./metal -run TestPrefillTTFT -v -timeout 30m
func TestPrefillTTFT(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	path := os.Getenv("GOINFER_METAL_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s (set GOINFER_METAL_MODEL)", path)
	}
	t.Setenv("GOINFER_METAL_BATCHED_PREFILL", "1") // measurement-only; see prior findings

	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Skip("metal resident not built for this model")
	}
	_, _, _, _, _, _, vocab := m.Dims()
	emb := func(i int) []float32 { return m.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1)) }

	fmt.Printf("\nPREFILL TTFT — %s, int4, sequential (today's shipped default) vs batched PrefillLast\n", path)
	fmt.Printf("%-6s %-16s %-16s %-10s\n", "P", "sequential_ms", "batched_ms", "speedup")
	for _, p := range []int{256, 1024, 3900} {
		embs := make([][]float32, p)
		for i := range embs {
			embs[i] = emb(i)
		}

		// SEQUENTIAL — today's shipped default: one Forward call per prompt token,
		// from an empty cache (position 0), matching how a real prompt is ingested.
		t0 := time.Now()
		for i := 0; i < p; i++ {
			if _, e := rf.Forward(embs[i], i); e != nil {
				t.Fatalf("sequential P=%d pos=%d: %v", p, i, e)
			}
		}
		seq := time.Since(t0)

		// BATCHED — one PrefillLast call over the same P embeddings, same absolute
		// positions [0,P). Overwrites the KV slots the sequential run just wrote;
		// that's fine, only the timing is compared, not the resulting cache content.
		t1 := time.Now()
		if _, e := rf.PrefillLast(context.Background(), embs, 0); e != nil {
			t.Fatalf("batched P=%d: %v", p, e)
		}
		batched := time.Since(t1)

		seqMs := float64(seq.Microseconds()) / 1000
		batchedMs := float64(batched.Microseconds()) / 1000
		speedup := seqMs / batchedMs
		fmt.Printf("%-6d %-16.2f %-16.2f %-10.2fx\n", p, seqMs, batchedMs, speedup)
		t.Logf("P=%d sequential=%.2fms batched=%.2fms speedup=%.2fx", p, seqMs, batchedMs, speedup)
	}
}
