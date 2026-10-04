//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestW8Prefill_passAgainstSequential (int8 slice 2, docs/tasks/task-metal-int8-2026-10.md): a native int8 resident now
// takes the batched prefill pass (gemm_w8f16 tiles). Per prompt of the prefill gate's set A, the last position's logits
// from the pass against the sequential decode loop's, on the native int8 path and, as the yardstick, on the int4 path
// (whose pass has shipped since L1 with the pooled fidelity gate): logit cosine, argmax agreement, KL(sequential ‖
// pass). Wall time of each, reported. The native arm runs with precise math (the shipped choice) and with fast math kept. Exploratory, by day; the pass's fidelity gate is the pooled one, at night.
// GOINFER_W8_GATE_MODEL picks the checkpoint (default the 0.5B); GOINFER_W8PF_M the prompt length (default 256).
//
//	GOINFER_W8PF=1 go test -tags goinfer_testhooks -count=1 -run '^TestW8Prefill_passAgainstSequential$' -v ./metal/
func TestW8Prefill_passAgainstSequential(t *testing.T) {
	if os.Getenv("GOINFER_W8PF") != "1" {
		t.Skip("set GOINFER_W8PF=1")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if p := os.Getenv("GOINFER_W8_GATE_MODEL"); p != "" {
		path = p
	}
	M := 256
	if v := os.Getenv("GOINFER_W8PF_M"); v != "" {
		fmt.Sscan(v, &M)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	files := decoder.PrefillGatePromptSetFor("a")[:6]
	for _, arm := range []struct {
		quant        string
		native, fast bool
	}{{"int8int8", true, false}, {"int8int8", true, true}, {"int4", false, false}} {
		prev, prevF := nativeInt8, w8FastMath
		nativeInt8, w8FastMath = arm.native, arm.fast
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: arm.quant, ResidentContext: M + 64})
		nativeInt8, w8FastMath = prev, prevF
		if err != nil {
			t.Fatalf("load %s: %v", arm.quant, err)
		}
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok || a.r.w8 != arm.native || !a.r.prefillOK {
			t.Fatalf("%s: resident %v, native %v, prefillOK %v", arm.quant, ok, ok && a.r.w8, ok && a.r.prefillOK)
		}
		{ // warm: the first pass compiles the prefill library; keep it out of the timings
			w := decoder.PrefillGateProseIDsForTest(t, tk, files[0], M)[:M]
			we := make([][]float32, M)
			for i, id := range w {
				we[i] = m.EmbedResidentForTest(id)
			}
			if _, err := a.PrefillLast(context.Background(), we, 0); err != nil {
				t.Fatalf("warm pass: %v", err)
			}
		}
		var minCos, sumKL, tPass, tSeq float64
		minCos = 1
		agree := 0
		for _, f := range files {
			ids := decoder.PrefillGateProseIDsForTest(t, tk, f, M)[:M]
			embs := make([][]float32, M)
			for i, id := range ids {
				embs[i] = m.EmbedResidentForTest(id)
			}
			st := time.Now()
			pass, err := a.PrefillLast(context.Background(), embs, 0)
			if err != nil {
				t.Fatalf("pass: %v", err)
			}
			tPass += time.Since(st).Seconds() * 1e3
			pass = append([]float32(nil), pass...)
			st = time.Now()
			var seq []float32
			for i, e := range embs {
				seq = a.r.ForwardEmb(e, i)
			}
			tSeq += time.Since(st).Seconds() * 1e3
			c := cosF(pass, seq)
			minCos = math.Min(minCos, c)
			if argmaxF(pass) == argmaxF(seq) {
				agree++
			}
			sumKL += klLogits(seq, pass)
		}
		n := float64(len(files))
		fmt.Fprintf(os.Stderr, "[w8-prefill] %s fast-math=%v (%s) M=%d, %d prompts: min cosine %.6f, argmax %d/%d, mean KL(seq ‖ pass) %.5f; pass %.1f ms, sequential %.1f ms (mean), %.2fx\n",
			arm.quant, arm.fast || !arm.native, m.DecodePath(), M, len(files), minCos, agree, len(files), sumKL/n, tPass/n, tSeq/n, tSeq/tPass)
		m.Close()
	}
}
