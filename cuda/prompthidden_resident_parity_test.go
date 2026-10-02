//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPromptHiddenResidentCUDA: decoder.Model.PromptHidden on a CUDA-resident Qwen3.5 (the JEV decision models' family)
// with a PEFT adapter merged at load answers from the device, through the ResidentHiddenLast seam HiddenLast uses, and
// stays within hiddenLastResidentBarCUDA of the CPU PromptHidden on the same weights. Two checks: PromptHidden's answer is
// bit-identical to the runner's own HiddenLast on the same ids (so the dispatch took the resident, not a silent CPU
// fallback), and it tracks the CPU path's (so the device kernels, DeltaNet included, compute the right thing). Two
// prompt lengths, the second after the first, so a DeltaNet state left over from one call would show in the next.
//
// Both sides run int4, as the Metal twin does (metal/prompthidden_resident_parity_test.go records why).
func TestPromptHiddenResidentCUDA(t *testing.T) {
	requireCUDADevice(t) // CI's cuda job has no driver (libcuda.so.1): skip there, as every other device test does
	const ckpt, lora = "../decoder/testdata/qwen3_5-tiny", "../decoder/testdata/qwen3_5-tiny-lora"
	opts := decoder.Options{Backend: "cuda", Quant: "int4", LoRA: lora}
	mRes, err := decoder.Load(ckpt, opts)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("qwen3_5-tiny did not go resident — decode path %q; decline: %s", mRes.DecodePath(), mRes.ResidentDecline())
	}
	rh, ok := rf.(decoder.ResidentHiddenLast)
	if !ok {
		t.Fatal("the cuda resident runner does not implement decoder.ResidentHiddenLast")
	}
	opts.Backend = "cpu"
	mCPU, err := decoder.Load(ckpt, opts)
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()
	_, _, _, _, _, _, vocab := mCPU.Dims()
	for _, n := range []int{40, 9} {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*37 + 11) % vocab
		}
		got, err := mRes.PromptHidden(context.Background(), ids)
		if err != nil {
			t.Fatalf("%d tokens: resident PromptHidden: %v", n, err)
		}
		embs := make([][]float32, n)
		for i, id := range ids {
			embs[i] = mRes.EmbedResidentForTest(id)
		}
		direct, err := rh.HiddenLast(context.Background(), embs, 0)
		if err != nil {
			t.Fatalf("%d tokens: runner HiddenLast: %v", n, err)
		}
		for i := range got {
			if got[i] != direct[i] {
				t.Fatalf("%d tokens: PromptHidden differs from the runner's HiddenLast at %d (%v vs %v): it did not take the resident", n, i, got[i], direct[i])
			}
		}
		want, err := mCPU.PromptHidden(context.Background(), ids)
		if err != nil {
			t.Fatalf("%d tokens: cpu PromptHidden: %v", n, err)
		}
		cos, maxAbs := cosF32(want, got)
		t.Logf("%d tokens: resident vs CPU cosine %.8f, max |diff| %.3g", n, cos, maxAbs)
		if cos < hiddenLastResidentBarCUDA {
			t.Errorf("%d tokens: resident PromptHidden cosine %.8f < %.4f against the CPU", n, cos, hiddenLastResidentBarCUDA)
		}
	}
}
