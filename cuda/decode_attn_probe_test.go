//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestDecodeAttn2048Probe drives the DECODE (M=1) attention kernel at ~2048 KV depth on the real
// 1.5B, so ncu can profile it: PrefillLast builds the 2048-token cache (using attn_batched), then a
// run of Forward calls decode at pos 2048+ — those launch the M=1 `attention` kernel (glue.ptx) at
// nKeys≈2048. Prefill used attn_batched, so `--kernel-name attention` targets ONLY decode attention.
// Investigating the comparative deficit (goinfer decode 221→97 tok/s from 128→2048 ctx vs current
// Ollama holding ~188) — a HYPOTHESIS to test at the hardware, not a diagnosis carried from prefill.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -c -o /tmp/decattn && \
//	  sudo env ... ncu --kernel-name attention --launch-skip N /tmp/decattn -test.run TestDecodeAttn2048Probe
func TestDecodeAttn2048Probe(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
	}
	const path = "/home/francis/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest().(*cudaResident)
	_, _, _, _, _, _, vocab := mc.Dims()

	const N = 2048
	embs := make([][]float32, N)
	var s uint32 = 12345
	for i := range embs {
		s = s*1664525 + 1013904223
		embs[i] = append([]float32(nil), mc.EmbedResidentForTest(int(s>>8)%(vocab-1))...)
	}
	lg, err := rf.PrefillLast(embs, 0) // builds the 2048-deep cache via attn_batched (not the decode kernel)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	cur := lg
	for i := 0; i < 24; i++ { // decode at pos 2048.. — each launches the M=1 `attention` at nKeys≈2048
		tok := argmaxF(cur)
		l, e := rf.Forward(mc.EmbedResidentForTest(tok), N+i)
		if e != nil {
			t.Fatalf("decode: %v", e)
		}
		cur = l
	}
}
